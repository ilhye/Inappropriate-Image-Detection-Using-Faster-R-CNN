import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Resize
from PIL import Image
import numpy as np
import os
import sys

# Import the local guided-diffusion package from the project's GuidedDiffusionPur folder
try:
    guided_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GuidedDiffusionPur"))
    sys.path.insert(0, guided_path)
    # guided_diffusion package inside GuidedDiffusionPur
    from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
    from guided_diffusion.gaussian_diffusion import _extract_into_tensor
    HAVE_GUIDED = True
except Exception:
    HAVE_GUIDED = False
    print("guided-diffusion not available from GuidedDiffusionPur; falling back to conceptual implementation.")

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


def adversarial_purification(adversarial_image, diffusion_model, num_reverse_steps=None, noise_timesteps=200):
    """
    Purify an adversarial image using a guided-diffusion reverse process.

    Returns:
        purified_image (torch.Tensor) in [0,1], same shape as input.
    """
    device = diffusion_model.device
    model = diffusion_model.model
    diffusion = diffusion_model.diffusion

    x_orig = adversarial_image.to(device)
    # ensure 4D
    if x_orig.dim() == 3:
        x_orig = x_orig.unsqueeze(0)

    # Resize/cast if model expects different size (best-effort)
    _, c, h, w = x_orig.shape
    if diffusion_model.image_size and (h != diffusion_model.image_size or w != diffusion_model.image_size):
        pil = Image.fromarray((x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        pil = pil.resize((diffusion_model.image_size, diffusion_model.image_size), Image.BILINEAR)
        x_orig = ToTensor()(pil).unsqueeze(0).to(device)

    # convert to model input range
    x_model = _to_model_space(x_orig)

    if HAVE_GUIDED and diffusion is not None:
        try:
            # Determine max timestep from diffusion if available
            num_timesteps = getattr(diffusion, "num_timesteps", noise_timesteps)
            t_T = num_timesteps - 1

            # make timestep tensor
            t = torch.full((x_model.shape[0],), t_T, dtype=torch.long, device=device)

            # add noise at timestep T using q_sample if available
            if hasattr(diffusion, "q_sample"):
                noise = torch.randn_like(x_model)
                x_T = diffusion.q_sample(x_model, t, noise=noise)
            else:
                # fallback: add Gaussian noise
                x_T = x_model + torch.randn_like(x_model) * 0.1

            # try to use p_sample_loop and supply init image if supported by local version
            if hasattr(diffusion, "p_sample_loop"):
                try:
                    # many guided-diffusion forks accept an 'init_image' or 'init_x' keyword
                    purified_model_space = diffusion.p_sample_loop(
                        model,
                        (x_T.shape[0], x_T.shape[1], x_T.shape[2], x_T.shape[3]),
                        init_image=x_T,
                        clip_denoised=True,
                    )
                except TypeError:
                    # try without init_image; some implementations always start from noise
                    purified_model_space = diffusion.p_sample_loop(
                        model,
                        (x_T.shape[0], x_T.shape[1], x_T.shape[2], x_T.shape[3]),
                        clip_denoised=True,
                    )
                    # if this started from pure noise instead of x_T, we fall back to iterative method below
            else:
                # iterative denoising using available model API: conservative approach
                purified_model_space = x_T.clone()
                # choose a reduced number of steps to be faster/safe
                steps = num_reverse_steps or min(200, num_timesteps)
                for timestep in reversed(range(num_timesteps - 1, num_timesteps - steps - 1, -1)):
                    t_tensor = torch.full((x_T.shape[0],), timestep, dtype=torch.long, device=device)
                    with torch.no_grad():
                        model_out = model(purified_model_space, t_tensor)
                    # use a small step size to avoid collapse; this is a heuristic fallback
                    purified_model_space = purified_model_space - model_out * (1.0 / max(1, steps / 10.0))

            # convert back to image space [0,1]
            purified = _to_image_space(purified_model_space).clamp(0.0, 1.0)
            return purified.cpu()
        except Exception as e:
            print(f"Guided purification failed ({e}), falling back to conceptual loop.")

    # Fallback denoising loop if guided-diffusion unavailable or failed
    purified = x_model.clone()
    steps = num_reverse_steps or 50
    for t in range(steps, 0, -1):
        t_tensor = torch.full((purified.shape[0],), t, dtype=torch.long, device=device)
        with torch.no_grad():
            noise_pred = diffusion_model(purified, t_tensor)
        # small-step update (heuristic)
        purified = purified - noise_pred * 0.01
        purified = torch.clamp(purified, -1.0, 1.0)

    purified = _to_image_space(purified).clamp(0.0, 1.0)
    return purified.cpu()


def main():
    # Testing dummy adversarial image, anti-alias, and purify using local guided-diffusion package.
    print("Creating a dummy adversarial image (28x28 RGB) ...")
    dummy_image = np.random.rand(28, 28, 3) * 0.1 + 0.5
    dummy_image = np.clip(dummy_image, 0, 1)
    adversarial_image_tensor = ToTensor()(Image.fromarray((dummy_image * 255).astype(np.uint8))).unsqueeze(0)

    # locate a reasonable checkpoint inside the GuidedDiffusionPur folder
    default_ckpt = os.path.join(os.path.dirname(__file__), "..", "GuidedDiffusionPur", "models", "256x256_diffusion.pt")
    default_ckpt = os.path.abspath(default_ckpt)
    if not os.path.exists(default_ckpt):
        default_ckpt = None

    diffusion_model = DiffusionModel(model_path=default_ckpt, image_size=256)

    # Step 1: Anti-aliasing
    anti_aliased = adversarial_anti_aliasing(adversarial_image_tensor, sigma=1.0)

    # Step 2: Purification
    purified = adversarial_purification(anti_aliased, diffusion_model)

    print("Purification finished; purified tensor shape:", purified.shape)

if __name__ == "__main__":
    main()
