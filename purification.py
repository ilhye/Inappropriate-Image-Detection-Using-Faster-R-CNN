import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Resize
from PIL import Image
import numpy as np
import os
import sys

# Import the local guided-diffusion package from the project's guided_diffusion folder
try:
    guided_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "guided_diffusion"))
    sys.path.insert(0, guided_path)
    # guided_diffusion package inside guided_diffusion
    from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
    from  guided_diffusion.gaussian_diffusion import _extract_into_tensor
    HAVE_GUIDED = True
except Exception:
    HAVE_GUIDED = False
    print("guided-diffusion not available from guided_diffusion; falling back to conceptual implementation.")

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

    # convert to [-1,1]
    x_model = _to_model_space(x_orig)

    if HAVE_GUIDED and diffusion is not None:
        try:
            steps = num_purifystep or 50
            # Try direct loop using init image (pure reverse)
            purified_model_space = diffusion.p_sample_loop(
                model,
                x_model.shape,
                clip_denoised=True,
                init_image=x_model,       # <--- directly start from the adversarial image
                num_purifysteps=steps     # some forks accept this argument
            )
            purified = _to_image_space(purified_model_space).clamp(0.0, 1.0)
            print("Purifying")
            return purified.cpu()
        except TypeError:
            # fallback: manual denoising loop
            purified = x_model.clone()
            steps = num_purifystep or 50
            for t in reversed(range(steps)):
                t_tensor = torch.full((purified.shape[0],), t, dtype=torch.long, device=device)
                with torch.no_grad():
                    noise_pred = model(purified, t_tensor)
                # update rule: subtract predicted noise
                purified = purified - 0.1 * noise_pred
                purified = purified.clamp(-1.0, 1.0)

            purified = _to_image_space(purified).clamp(0.0, 1.0)
            return purified.cpu()

    # Fallback dummy
    return x_orig.cpu()

def load_image_as_tensor(path):
    pil = Image.open(path).convert("RGB")
    tensor = ToTensor()(pil).unsqueeze(0)  # [1,C,H,W] in [0,1]
    return tensor

def main(image_path):
    # Testing dummy adversarial image, anti-alias, and purify using local guided-diffusion package.
    # print("Creating a dummy adversarial image (28x28 RGB) ...")
    # dummy_image = np.random.rand(28, 28, 3) * 0.1 + 0.5
    # dummy_image = np.clip(dummy_image, 0, 1)

    adversarial_image_tensor = load_image_as_tensor(image_path)

    # locate checkpoint
    default_ckpt = os.path.join(os.path.dirname(__file__), "..", "guided_diffusion", "models", "256x256_diffusion.pt")
    if not os.path.exists(default_ckpt):
        raise FileNotFoundError("Pretrained diffusion checkpoint not found at " + default_ckpt)

    diffusion_model = DiffusionModel(model_path=default_ckpt, image_size=256)

    # Step 1: Anti-aliasing
    anti_aliased = adversarial_anti_aliasing(adversarial_image_tensor, sigma=1.0)

    # Step 2: Purification
    purified = adversarial_purification(anti_aliased, diffusion_model)

    print("Purification finished; purified tensor shape:", purified.shape)

if __name__ == "__main__":
    main()
