import cv2
from cv2.ximgproc import guidedFilter, anisotropicDiffusion
from utils import *
import numpy as np

class Purifier:
    @staticmethod
    def process(input_img):
        """Purify image using anisotropic diffusion"""
        # Convert input to numpy array if it's a PIL Image
        orig_img = convert_to_np(input_img)

        # Anisotropic > Super-Resolution
        purified = Purifier.anisotropic(orig_img)
        psnr_value = psnr(original=orig_img, purified=purified)

        print(f"PSNR in: {psnr_value:.2f}dB")
        return purified

    # Remove noise while preserving edges
    @staticmethod
    def anisotropic(input_img, alpha=0.1, K=15, iterations=10, option=1):
        """Purifying image using anisotropic diffusion
        Args:
            input_img: Input image to be purified
            alpha: Conduction coefficient prevent over-smoothing
            K: Sensitivity to edges
            iterations: Number of iterations, chosen to provide sufficient denoising without computational overhead
            option: 1 for Perona-Malik (more aggressive), 2 for Tukey's biweight function (relies on region)
        Return:
            img: The purified image
        """
        img = input_img.astype(np.float32)
        for _ in range(iterations):
            # Computes differences between pixel and its neighbors (gradient)
            # axis = 0: vertical; axis = 1: horizontal
            north = np.roll(img, -1, axis=0) - img
            south = np.roll(img, 1, axis=0) - img
            east = np.roll(img, -1, axis=1) - img
            west = np.roll(img, 1, axis=1) - img

            # Conduction function
            # c_n, c_s, c_e, c_w: controls how much diffusion occurs in each direction
            # K: controls the sensitivity to edges
            # if small gradient == smoothen
            # if high gradient == edges, preserve
            if option == 1:
                c_n = np.exp(-(north/K)**2)
                c_s = np.exp(-(south/K)**2)
                c_e = np.exp(-(east/K)**2)
                c_w = np.exp(-(west/K)**2)
            else: 
                c_n = 1.0 / (1.0 + (north/K)**2)
                c_s = 1.0 / (1.0 + (south/K)**2)
                c_e = 1.0 / (1.0 + (east/K)**2)
                c_w = 1.0 / (1.0 + (west/K)**2)

            # Update image
            # Blur the pixels based on the conduction values
            img += alpha * (c_n*north + c_s*south + c_e*east + c_w*west)

        # Return as image
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img