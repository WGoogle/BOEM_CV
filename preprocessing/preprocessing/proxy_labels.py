"""
Proxy Label Generation Module
-----------------------------
Generates weak supervision labels using classical image processing:
1. Grayscale conversion
2. CLAHE enhancement
3. Gaussian blur for stability
4. Otsu thresholding
5. Morphological operations (opening & closing)
6. Contour filtering by area, eccentricity, and solidity
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ProxyLabelGenerator:
    """Generates proxy labels for weak supervision using classical CV."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with proxy label parameters from config.PROXY_LABEL
        """
        self.config = config
        
        # Create morphological kernels
        self.opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config['morph_opening_kernel'], config['morph_opening_kernel'])
        )
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config['morph_closing_kernel'], config['morph_closing_kernel'])
        )
    
    def convert_to_grayscale(self, image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def apply_gaussian_blur(self, gray: np.ndarray):
        """Mild blur to merge split nodule fragments before thresholding."""
        if not self.config.get('apply_gaussian_blur', True):
            return gray
        ksize = self.config['gaussian_kernel_size']
        sigma = self.config['gaussian_sigma']
        return cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    def adaptive_threshold(self, gray: np.ndarray):
        """
        Stats-based threshold: keeps pixels that are significantly darker
        than the image mean.

        threshold = mean - threshold_k * std
        clamped to [threshold_min, threshold_max] for safety.

        Why not Otsu: the preprocessed image has a bimodal histogram where the
        dominant mode is background (mean ~100).  Otsu picks ~86, which labels
        29 % of the image as nodule.  Our method anchors to the actual image
        statistics and yields ~5-9 % coverage — consistent with more of the 14ish percent we see on presentation.
        """
        m   = float(gray.mean())
        s   = float(gray.std())
        k   = self.config.get('threshold_k', 2.2)
        lo  = self.config.get('threshold_min', 20)
        hi  = self.config.get('threshold_max', 70)
        t   = float(np.clip(m - k * s, lo, hi))

        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        logger.debug(f"Adaptive threshold: mean={m:.1f} std={s:.1f} k={k} → t={t:.1f}")
        return binary

    def morphological_cleanup(self, binary: np.ndarray):
        """
        Opening removes isolated noise pixels.
        Closing merges split halves of the same nodule.
        """
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self.opening_kernel)
        closed = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, self.closing_kernel)
        logger.debug("Applied morphological opening and closing")
        return closed

    def compute_contour_features(self, contour):
        """Compute area, eccentricity, and solidity for a contour."""
        area = cv2.contourArea(contour)
        if area < 5:
            return {'area': area, 'eccentricity': 1.0, 'solidity': 0.0}

        # Eccentricity via fitted ellipse
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                _, (MA, ma), _ = cv2.fitEllipse(contour)
                long, short = max(MA, ma), min(MA, ma)
                if long > 0:
                    eccentricity = float(np.sqrt(1.0 - (short / long) ** 2))
            except Exception:
                pass

        # Solidity = area / convex hull area
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity   = area / hull_area if hull_area > 0 else 0.0

        return {'area': area, 'eccentricity': eccentricity, 'solidity': solidity}

    def filter_contours(self, binary: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Keep only contours whose area, eccentricity, and solidity fall within
        the configured nodule ranges.

        Eccentricity is applied as a SIZE-DEPENDENT filter:
        - Small blobs (< large_area_threshold px²): strict max_eccentricity
          — rejects elongated sediment trails and thin noise.
        - Large blobs (≥ large_area_threshold px²): relaxed large_eccentricity_limit
          — nodule clusters merge into slightly elongated shapes; we still want them.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], np.zeros_like(binary)

        large_area_thresh  = self.config.get('large_area_threshold',      200)
        large_ecc_limit    = self.config.get('large_eccentricity_limit',  0.95)

        filtered = []
        for c in contours:
            f = self.compute_contour_features(c)

            if not (self.config['min_contour_area'] <= f['area'] <= self.config['max_contour_area']):
                continue
            if f['solidity'] < self.config['min_solidity']:
                continue

            # Size-dependent eccentricity threshold
            ecc_limit = large_ecc_limit if f['area'] >= large_area_thresh \
                        else self.config['max_eccentricity']
            if not (self.config['min_eccentricity'] <= f['eccentricity'] <= ecc_limit):
                continue

            filtered.append(c)

        mask = np.zeros_like(binary)
        if filtered:
            cv2.drawContours(mask, filtered, -1, 255, thickness=cv2.FILLED)

        logger.debug(f"Filtered {len(filtered)}/{len(contours)} contours")
        return filtered, mask
    
    def generate_proxy_label(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Full proxy label generation pipeline.

        Args:
            image: Preprocessed mosaic/patch (H, W, C) BGR

        Returns:
            proxy_mask: Binary mask (H, W) — 0 = background, 255 = nodule
            stats:      Generation statistics dict
        """
        # 1. Grayscale
        gray = self.convert_to_grayscale(image)

        # 2. Mild blur (merge split fragments)
        blurred = self.apply_gaussian_blur(gray)

        # 3. Stats-based adaptive threshold (dark pixels = nodules)
        binary = self.adaptive_threshold(blurred)

        # 4. Morphological cleanup
        cleaned = self.morphological_cleanup(binary)

        # 5. Shape-based contour filtering
        contours, proxy_mask = self.filter_contours(cleaned)

        n_before = len(cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        stats = {
            'num_candidates_before_filter': n_before,
            'num_nodules_after_filter':     len(contours),
            'coverage_percent':             100.0 * np.sum(proxy_mask > 0) / proxy_mask.size,
        }

        logger.info(
            f"Proxy label: {stats['num_nodules_after_filter']} nodules, "
            f"{stats['coverage_percent']:.2f}% coverage"
        )
        return proxy_mask, stats


def visualize_proxy_label(
    original: np.ndarray,
    proxy_mask: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create visualization with proxy label overlaid on original image.
    
    Args:
        original: Original image (H, W, C) BGR
        proxy_mask: Binary mask (H, W) with 0=background, 255=nodule
        alpha: Transparency of overlay
        
    Returns:
        Visualization image (H, W, C) BGR
    """
    # Create colored mask (white nodules)
    colored_mask = np.zeros_like(original)
    colored_mask[proxy_mask > 0] = [255, 255, 255]
    
    # Blend
    overlay = cv2.addWeighted(original, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay
