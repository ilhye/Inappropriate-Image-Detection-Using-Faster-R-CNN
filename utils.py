"""
===========================================================
Program: Utils
Programmer/s: Cristina C. Villasor
Date Written: Oct. 5, 2025
Last Revised: Oct. 5, 2025

Purpose: This contains functions reusable functions 

Program Fits in the General System Design:
- Called in routes, frcnn, and purification for image processing
===========================================================
"""
import numpy as np

from PIL import Image

def convert_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    return Image.fromarray(image)

def convert_to_np(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array."""
    return np.array(image)

def psnr(original:np.ndarray, purified: np.ndarray) -> float:
    """Compute PSNR between two images
        Args:
            original: Original image as a NumPy array
            purified: Purified image as a NumPy array
        Returns:
            PSNR value as a float
    """
    mse = np.mean((original.astype(np.float32) - purified.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

