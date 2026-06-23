import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import skimage.filters
import skimage.morphology
import skimage.measure
import matplotlib.pyplot as plt
import torch
import torch.nn

class DiffusionMaskRefiner:
    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.noise_history = []

    def forward_diffusion(self, img: np.ndarray, t: int) -> np.ndarray:
        noise = np.random.normal(0, 1, img.shape).astype(np.float32)
        alpha_bar_t = self.alpha_bars[t]
        noisy = np.sqrt(alpha_bar_t) * img + np.sqrt(1 - alpha_bar_t) * noise
        noisy = np.clip(noisy, 0, 1)
        self.noise_history.append(noisy)
        return noisy

    def reverse_diffusion_step(self, noisy_img: np.ndarray, t: int) -> np.ndarray:
        estimated_noise = noisy_img - gaussian_filter(noisy_img, sigma=1.0)
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        denoised = (noisy_img - np.sqrt(1 - alpha_bar_t) * estimated_noise) / np.sqrt(self.alpha_bars[t])
        
        if t > 0:
            noise = np.random.normal(0, 0.1, noisy_img.shape).astype(np.float32)
            denoised = denoised + np.sqrt(beta_t) * noise
            
        denoised = np.clip(denoised, 0, 1)
        return denoised

    def full_reverse_process(self, noisy_img: np.ndarray, start_t: int) -> list:
        img_t = noisy_img.copy()
        results = []
        for t in range(start_t, -1, -1):
            img_t = self.reverse_diffusion_step(img_t, t)
            if t % 5 == 0:
                results.append(img_t)
        return results

    def generate_initial_noisy_mask(self, img: np.ndarray, t=25) -> np.ndarray:
        self.noise_history = []
        t_bounded = min(t, self.num_steps - 1)
        return self.forward_diffusion(img, t_bounded)

class MaskRefiner:
    def __init__(self):
        pass

    def otsu_threshold(self, img: np.ndarray) -> np.ndarray:
        img_uint8 = (img * 255).astype(np.uint8)
        try:
            threshold = skimage.filters.threshold_otsu(img_uint8)
            binary_mask = (img_uint8 > threshold).astype(np.uint8) * 255
        except ValueError:
            binary_mask = np.zeros_like(img_uint8)
        return binary_mask

    def morphological_cleanup(self, mask: np.ndarray, open_radius=3, close_radius=5) -> np.ndarray:
        bool_mask = mask > 127
        opened = skimage.morphology.binary_opening(bool_mask, skimage.morphology.disk(open_radius))
        closed = skimage.morphology.binary_closing(opened, skimage.morphology.disk(close_radius))
        return (closed.astype(np.uint8) * 255)

    def extract_lesion_regions(self, mask: np.ndarray) -> list:
        bool_mask = mask > 127
        labeled = skimage.measure.label(bool_mask)
        props = skimage.measure.regionprops(labeled)
        
        regions = []
        for r in props:
            if r.area > 100:
                regions.append({
                    "area": float(r.area),
                    "centroid": r.centroid,
                    "bbox": r.bbox,
                    "eccentricity": float(r.eccentricity),
                    "solidity": float(r.solidity)
                })
        return regions

    def refine_mask(self, img: np.ndarray) -> dict:
        raw_mask = self.otsu_threshold(img)
        refined_mask = self.morphological_cleanup(raw_mask)
        regions = self.extract_lesion_regions(refined_mask)
        
        return {
            "raw_mask": raw_mask,
            "refined_mask": refined_mask,
            "regions": regions,
            "lesion_count": len(regions)
        }
