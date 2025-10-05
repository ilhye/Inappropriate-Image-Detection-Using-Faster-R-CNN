import cv2
from cv2.ximgproc import guidedFilter, anisotropicDiffusion
from utils import *
import numpy as np

class Purifier:
    @staticmethod
    def process(input_img):
        # Convert input to numpy array if it's a PIL Image
        orig_img = convert_to_np(input_img)
        
        anti_aliased = Purifier.anti_aliasing(orig_img)
        denoise = Purifier.bilateral(anti_aliased)
        purified = Purifier.guided(denoise, orig_img)
        psnr_value = psnr(original=orig_img, purified=purified)

        if psnr_value < 20:
            purified = Purifier.anisotropic(purified)
            psnr_value = psnr(original=orig_img, purified=purified)

        print(f"PSNR in: {psnr_value:.2f}dB")
        return purified

    # Anti-aliasing
    # Blend edges to to its surrounding pixels
    @staticmethod
    def anti_aliasing(input_img):
        h, w = input_img.shape[:2]
        image_small = cv2.resize(input_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        image_aa = cv2.resize(image_small, (w, h), interpolation=cv2.INTER_CUBIC)
        return image_aa
    
    # Denoising
    # Pixel same range == Blur
    # Pixel varies range == Preserve
    # input_img: Input
    # diameter of neighbor pixel
    # sigma color, tolerance color
    # sigma space, influence of pixels
    @staticmethod
    def bilateral(image_aa):
        denoise = convert_to_np(image_aa)  
        for _ in range(64):
            denoise = cv2.bilateralFilter(
                src=denoise, d=5, sigmaColor=8, sigmaSpace=8)
        return denoise

    # Edge preservation
    # orig_img: original image for guidance
    @staticmethod
    def guided(denoise, orig_img):
        for _ in range(4):
            filtered = guidedFilter(
                guide=orig_img, src=denoise, radius=4, eps=16)
        return filtered
    
    # More aggressive denoising than guided
    # img_float: convert img to float
    # alpha: conduction coefficient
    # K: conduction coefficient
    # iterations: number of iterations
    # options: 1 for Perona-Malik (more aggressive), 2 for Tukey's biweight function (relies on region)
    @staticmethod
    def anisotropic(input_img):
        img_float = input_img.astype(np.float32) / 255.0
        diffused_img = anisotropicDiffusion(img_float, alpha=0.1, K=15, iterations=10, options=1)
        diffused_img = cv2.normalize(diffused_img, None, 0, 255, cv2.NORM_MINMAX)
        return diffused_img