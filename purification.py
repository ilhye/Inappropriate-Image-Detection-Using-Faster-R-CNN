import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Resize
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from PIL import Image
import numpy as np
import os
import sys

# --- Replace the old guided_diffusion import block with a focused import of the repo purifier ---
# Try to import the actual GuidedDiffusionPur purification implementation (diff_purify)
# from the local GuidedDiffusionPur/purification folder. This avoids forcing the
# guided_diffusion package import used for model construction.
_guided_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GuidedDiffusionPur"))
if _guided_repo not in sys.path:
    sys.path.insert(0, _guided_repo)

HAVE_GD_PURIFY = False
try:
    # the repo's purification routine (the code you inspected)
    from purification.diff_purify import diff_purify
    # helpers used by the purify routine
    from pytorch_diffusion.diffusion import Diffusion as GD_Diffusion  # optional, may be used elsewhere
    from utils import raw_to_diff, diff_to_raw
    HAVE_GD_PURIFY = True
    print("Using GuidedDiffusionPur.diff_purify from GuidedDiffusionPur/purification")
except Exception as _e:
    HAVE_GD_PURIFY = False
    diff_purify = None
    GD_Diffusion = None
    # keep original guided-diffusion import fallback if you still want it below
    # Note: we deliberately do not import guided_diffusion.script_util here to avoid the
    # previous behavior you wanted to avoid.

class DiffusionModel(nn.Module):
    """
    Wrapper around guided-diffusion model+diffusion. If guided-diffusion is available
    it will create and load a real model; otherwise it will use a lightweight placeholder.
    """
    def __init__(self, model_path=None, image_size=256, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        if HAVE_GUIDED:
            args = model_and_diffusion_defaults()
            args["image_size"] = image_size
            # create model + diffusion objects
            self.model, self.diffusion = create_model_and_diffusion(**args)
            self.model.to(self.device)
            # try loading checkpoint if provided
            if model_path and os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location="cpu")
                # handle common checkpoint dict formats
                if "state_dict" in ckpt:
                    state = ckpt["state_dict"]
                else:
                    state = ckpt
                # allow for strict=False to accept partial matches
                try:
                    self.model.load_state_dict(state, strict=False)
                    print(f"Loaded checkpoint from {model_path}")
                except Exception as ex:
                    print(f"Could not fully load checkpoint ({ex}); continuing with model defaults.")
        else:
            print("Using dummy linear model as diffusion model placeholder.")
            self.model = nn.Linear(784, 784)
            self.diffusion = None
            self.model.to(self.device)

    def forward(self, x, t):
        """
        Guided-diffusion model expects inputs in the range [-1, 1] and t as a tensor of timesteps.
        This wrapper calls the underlying model and returns its prediction.
        """
        if HAVE_GUIDED:
            return self.model(x, t)
        else:
            # placeholder behaviour: flatten and pass through linear
            b, c, h, w = x.shape
            x_flat = x.view(b, -1)
            out = self.model(x_flat).view(b, c, h, w)
            return out


def adversarial_anti_aliasing(image_tensor, sigma=1.0):
    """
    Apply Gaussian anti-aliasing (spatial smoothing) to reduce high-frequency/adversarial
    perturbations. Operates on a 4D tensor (B,C,H,W) with values in [0,1].

    Returns a tensor with same device/dtype as input.
    """
    from scipy.ndimage import gaussian_filter

    if not torch.is_tensor(image_tensor):
        raise TypeError("image_tensor must be a torch.Tensor")

    device = image_tensor.device
    dtype = image_tensor.dtype

    # Work on CPU numpy for scipy
    img = image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()  # H,W,C
    img = np.clip(img, 0.0, 1.0)

    smoothed = np.zeros_like(img)
    # apply gaussian per channel
    for c in range(img.shape[2]):
        smoothed[..., c] = gaussian_filter(img[..., c], sigma=sigma, mode='reflect')

    smoothed = np.clip(smoothed, 0.0, 1.0)
    pil = Image.fromarray((smoothed * 255).astype(np.uint8))
    tensor = ToTensor()(pil).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def _to_model_space(x):
    """
    Convert [0,1] image tensor to guided-diffusion model space [-1,1].
    """
    return x * 2.0 - 1.0


def _to_image_space(x):
    """
    Convert model outputs in [-1,1] back to image space [0,1].
    """
    return (x + 1.0) / 2.0


def load_image_as_tensor(path):
    pil = Image.open(path).convert("RGB")
    tensor = ToTensor()(pil).unsqueeze(0)  # [1,C,H,W] in [0,1]
    return tensor

def adversarial_purification(adversarial_image, diffusion_model, num_purifystep=None):
    """
    Purify adversarial image by running only the reverse diffusion (denoising) process.
    """
    device = diffusion_model.device
    model = diffusion_model.model
    diffusion = diffusion_model.diffusion

    x_orig = adversarial_image.to(device)
    if x_orig.dim() == 3:
        x_orig = x_orig.unsqueeze(0)

    # resize if needed
    _, c, h, w = x_orig.shape
    if diffusion_model.image_size and (h != diffusion_model.image_size or w != diffusion_model.image_size):
        pil = Image.fromarray((x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        pil = pil.resize((diffusion_model.image_size, diffusion_model.image_size), Image.BILINEAR)
        x_orig = ToTensor()(pil).unsqueeze(0).to(device)

    # Prefer the repo's diff_purify implementation if available
    if HAVE_GD_PURIFY and diff_purify is not None and diffusion is not None:
        try:
            # prepare max_iter from caller or default config
            max_iter = int(num_purifystep) if num_purifystep is not None else getattr(diffusion_model, "purify_steps", getattr(diffusion_model, "config", None) and getattr(diffusion_model.config.purification, "max_iter", 1) or 1)
            # diff_purify expects raw->diff transforms internally; pass model's diffusion and config
            cfg = getattr(diffusion_model, "config", None)
            # call diff_purify(x, diffusion, max_iter, mode, config) as in repo
            images = diff_purify(x_orig, diffusion, max_iter, mode="purification", config=cfg)
            # diff_purify returns a list of purified images (per iteration); take last if list
            if isinstance(images, (list, tuple)) and len(images) > 0:
                out = images[-1]
                return out.detach().cpu()
            # if it returned a single tensor
            if torch.is_tensor(images):
                return images.detach().cpu()
        except Exception as e:
            print("diff_purify failed, falling back to local guided-diffusion:", e)

    # Fallback: existing guided-diffusion or conceptual implementation (unchanged)
    # convert to [-1,1]
    x_model = _to_model_space(x_orig)

    if HAVE_GUIDED and diffusion is not None:
        try:
            steps = num_purifystep or 50
            purified_model_space = diffusion.p_sample_loop(
                model,
                x_model.shape,
                clip_denoised=True,
                init_image=x_model,
                num_purifysteps=steps
            )
            purified = _to_image_space(purified_model_space).clamp(0.0, 1.0)
            return purified.cpu()
        except TypeError:
            purified = x_model.clone()
            steps = num_purifystep or 50
            for t in reversed(range(steps)):
                t_tensor = torch.full((purified.shape[0],), t, dtype=torch.long, device=device)
                with torch.no_grad():
                    noise_pred = model(purified, t_tensor)
                purified = purified - 0.1 * noise_pred
                purified = purified.clamp(-1.0, 1.0)

            purified = _to_image_space(purified).clamp(0.0, 1.0)
            return purified.cpu()

    return x_orig.cpu()


def purification_check(original, purified, threshold_psnr=20.0):
    """ Compare original vs purified image with PSNR score.
    Args:
        original (torch.Tensor): Original image tensor in [0,1]
        purified (torch.Tensor): Purified image tensor in [0,1]
        threshold_psnr (float): PSNR threshold to consider purification successful

    Returns:
        (torch.Tensor, bool, float): Purified tensor, is_purified flag, PSNR score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    original, purified = original.to(device), purified.to(device)

    # resize purified if necessary
    if original.shape[-2:] != purified.shape[-2:]:
        purified = F.interpolate(purified, size=original.shape[-2:], mode="bilinear", align_corners=False)

    psnr_score = psnr(original, purified).item()
    is_purified = psnr_score >= threshold_psnr

    print(f"🔍 PSNR = {psnr_score:.2f}")
    print(f"🎯 {'GOOD QUALITY' if is_purified else 'DEGRADED'}")

    return purified.cpu(), is_purified, psnr_score

# def main(image_path):
#     # Testing dummy adversarial image, anti-alias, and purify using local guided-diffusion package.
#     # print("Creating a dummy adversarial image (28x28 RGB) ...")
#     # dummy_image = np.random.rand(28, 28, 3) * 0.1 + 0.5
#     # dummy_image = np.clip(dummy_image, 0, 1)

#     adversarial_image_tensor = load_image_as_tensor(image_path)

#     # locate checkpoint
#     default_ckpt = os.path.join(os.path.dirname(__file__), "..", "guided_diffusion", "models", "256x256_diffusion.pt")
#     if not os.path.exists(default_ckpt):
#         raise FileNotFoundError("Pretrained diffusion checkpoint not found at " + default_ckpt)

#     diffusion_model = DiffusionModel(model_path=default_ckpt, image_size=256)

#     # Step 1: Anti-aliasing
#     anti_aliased = adversarial_anti_aliasing(adversarial_image_tensor, sigma=1.0)

#     # Step 2: Purification
#     purified = adversarial_purification(anti_aliased, diffusion_model)

#     print("Purification finished; purified tensor shape:", purified.shape)

#     # Step 3: Quality Check

# if __name__ == "__main__":
#     main()