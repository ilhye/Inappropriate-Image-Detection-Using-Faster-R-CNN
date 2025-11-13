"""
===========================================================
Program: Purification
Programmer/s: Catherine Joy R. Pailden and Cristina C. Villasor
Date Written: Oct. 5, 2025
Last Revised: Oct. 14, 2025

Purpose: This applies filtering technique that reduces noises while preserving important features like an edge

Program Fits in the General System Design:
- This is used right after taking an image input from user
- Called in the routes and frcnn.py for video processing
- Output is also used by super-resolution module

Algorithm: 
- Convert image into float
- For 10 iterations: 
    - Compute differences between pixel and its neighbors (gradient)
    - Compute conduction values based on gradient and K
    - Update pixel values based on conduction values and alpha 

Data Structures and Controls: 
- Uses 2D arrays for image representation
- Uses loops for iterative processing
- Uses if-else condition for conduction function choice
===========================================================
"""
import numpy as np

class Purifier:
    @staticmethod
    def process(input_img):
        """Purify image using anisotropic diffusion"""
        # Convert input to numpy array if it's a PIL Image
        orig_img = np.array(input_img)

        # Anisotropic > Super-Resolution
        purified_img = Purifier.anisotropic(orig_img)
        psnr_value = Purifier.psnr(original=orig_img, purified=purified_img)
        print(f"PSNR in: {psnr_value:.2f}dB")
        return purified_img

    @staticmethod
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