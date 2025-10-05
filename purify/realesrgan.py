# realesrgan_wrapper.py
import torch
from PIL import Image
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from real_esrgan.model import RealESRGAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealESRGANWrapper:
    @staticmethod
    def enhance(image: Image.Image):
        print("Enter enhance")
        load = RealESRGANWrapper.load_model(device=DEVICE)
        print("Loaded model, will return")
        return RealESRGANWrapper.run_sr(load, image)

    @staticmethod
    def load_model(model_path="models/RealESRGAN_x2.pth", scale=2, device=DEVICE):
        """Load the pretrained model
        Args:
            model_path: Path to the model weights
            scale: Upscaling factor (default is 4)
            device: Device to load the model on (default is CUDA if available)
        Returns:
            The loaded RealESRGAN model
        """
        upsampler = RealESRGAN(device=device, scale=scale)
        upsampler.load_weights(model_path, download=False)
        return upsampler

    @staticmethod
    def run_sr(upsampler, image_pil: Image.Image) -> Image.Image:
        """"Run super-resolution
        Args:
            upsampler: The loaded RealESRGAN model
            image_pil: Input image in PIL format
        Returns:
            The super-resolved image in PIL format
        """
        print("Upsaling image")
        return upsampler.predict(image_pil)
