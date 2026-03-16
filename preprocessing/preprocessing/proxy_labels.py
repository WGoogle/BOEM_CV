"""
Proxy Label Generation Module
-----------------------------
Generates weak supervision labels using classical image processing:
1. Grayscale conversion
2. Gaussian blur for speckle suppression
3. Multi-scale black top-hat transform (background-normalized)
4. Percentile threshold on top-hat response
5. Morphological cleanup (optional opening + closing)
6. Contour filtering by area, eccentricity, solidity, and circularity
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

        # Morphological kernels for cleanup
        opening_k = config['morph_opening_kernel']
        self.opening_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_k, opening_k))
            if opening_k > 1 else None
        )
        closing_k = config['morph_closing_kernel']
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_k, closing_k)
        )

        # Multi-scale top-hat structuring elements
        self.tophat_kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            for k in config.get('tophat_kernel_sizes', [11, 21, 41])
        ]

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def apply_gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """Mild blur to suppress sediment speckle before top-hat."""
        if not self.config.get('apply_gaussian_blur', True):
            return gray
        ksize = self.config['gaussian_kernel_size']
        sigma = self.config['gaussian_sigma']
        return cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    def multi_scale_tophat(self, gray: np.ndarray) -> np.ndarray:
        """Pixel-wise max of black top-hat responses across all kernel scales."""
        responses = []
        for kernel in self.tophat_kernels:
            response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            responses.append(response)

        combined = np.max(responses, axis=0).astype(np.uint8)
        logger.debug(
            f"Multi-scale top-hat: {len(self.tophat_kernels)} scales, "
            f"max response={combined.max()}, mean={combined.mean():.1f}"
        )
        return combined

    def smooth_tophat_response(self, response: np.ndarray) -> np.ndarray:
        """Gaussian-smooth the top-hat response to suppress texture speckle."""
        sigma = self.config.get('tophat_response_blur_sigma', 2.0)
        if sigma <= 0:
            return response
        return cv2.GaussianBlur(response, (0, 0), sigma)

    def percentile_threshold(self, tophat_response: np.ndarray) -> np.ndarray:
        """Percentile threshold on the top-hat response, floored to avoid noise."""
        pct = self.config.get('tophat_threshold_percentile', 90)
        t = float(np.percentile(tophat_response, pct))
        floor = self.config.get('tophat_threshold_floor', 15)
        t = max(t, floor)

        binary = np.where(tophat_response > t, np.uint8(255), np.uint8(0))
        logger.debug(f"Top-hat threshold: percentile({pct})={t:.1f}, floor={floor}, foreground pixels={np.sum(binary > 0)}")
        return binary

    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Opening removes noise pixels; closing merges split nodule halves."""
        if self.opening_kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.opening_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.closing_kernel)
        logger.debug("Applied morphological cleanup")
        return binary

    def compute_contour_features(self, contour) -> dict:
        """Compute area, eccentricity, solidity, and circularity for a contour."""
        area = cv2.contourArea(contour)
        if area < 5:
            return {'area': area, 'eccentricity': 1.0, 'solidity': 0.0, 'circularity': 0.0}

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

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        return {
            'area': area,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'circularity': circularity,
        }

    def filter_contours(self, binary: np.ndarray) -> Tuple[list, np.ndarray, dict]:
        """Filter contours by area, eccentricity, solidity, and circularity."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], np.zeros_like(binary), {
                'area': 0, 'solidity': 0, 'eccentricity': 0, 'circularity': 0
            }

        large_area_thresh = self.config.get('large_area_threshold', 200)
        large_ecc_limit = self.config.get('large_eccentricity_limit', 0.95)
        min_circularity = self.config.get('min_circularity', 0.3)

        filtered = []
        rejected = {'area': 0, 'solidity': 0, 'eccentricity': 0, 'circularity': 0}

        for c in contours:
            f = self.compute_contour_features(c)

            if not (self.config['min_contour_area'] <= f['area'] <= self.config['max_contour_area']):
                rejected['area'] += 1
                continue
            if f['solidity'] < self.config['min_solidity']:
                rejected['solidity'] += 1
                continue

            # Size-dependent eccentricity threshold
            ecc_limit = large_ecc_limit if f['area'] >= large_area_thresh \
                else self.config['max_eccentricity']
            if not (self.config['min_eccentricity'] <= f['eccentricity'] <= ecc_limit):
                rejected['eccentricity'] += 1
                continue

            if f['circularity'] < min_circularity:
                rejected['circularity'] += 1
                continue

            filtered.append(c)

        mask = np.zeros_like(binary)
        if filtered:
            cv2.drawContours(mask, filtered, -1, 255, thickness=cv2.FILLED)

        logger.debug(f"Filtered {len(filtered)}/{len(contours)} contours")
        return filtered, mask, rejected

    def generate_proxy_label(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Full proxy label generation pipeline.

        Args:
            image: Preprocessed mosaic/patch (H, W, C) BGR

        Returns:
            proxy_mask: Binary mask (H, W) -- 0 = background, 255 = nodule
            stats:      Generation statistics dict
        """
        total_pixels = image.shape[0] * image.shape[1]

        # 1. Grayscale
        gray = self.convert_to_grayscale(image)

        # 2. Mild blur (suppress sediment speckle)
        blurred = self.apply_gaussian_blur(gray)

        # 3. Multi-scale black top-hat (background-normalized)
        tophat_response = self.multi_scale_tophat(blurred)

        # 4. Smooth top-hat response
        smoothed = self.smooth_tophat_response(tophat_response)

        # 5. Threshold
        binary = self.percentile_threshold(smoothed)
        coverage_after_threshold = 100.0 * np.sum(binary > 0) / total_pixels

        # 6. Morphological cleanup
        cleaned = self.morphological_cleanup(binary)
        coverage_after_morph = 100.0 * np.sum(cleaned > 0) / total_pixels

        # 7. Shape-based contour filtering
        n_before = len(cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        contours, proxy_mask, rejected = self.filter_contours(cleaned)

        coverage_final = 100.0 * np.sum(proxy_mask > 0) / total_pixels
        stats = {
            'num_candidates_before_filter': n_before,
            'num_nodules_after_filter':     len(contours),
            'coverage_percent':             coverage_final,
            'coverage_after_threshold_pct': coverage_after_threshold,
            'coverage_after_morph_pct':     coverage_after_morph,
            'rejected_area':                rejected['area'],
            'rejected_solidity':            rejected['solidity'],
            'rejected_eccentricity':        rejected['eccentricity'],
            'rejected_circularity':         rejected['circularity'],
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
    """Overlay proxy label mask on original image."""
    colored_mask = np.zeros_like(original)
    colored_mask[proxy_mask > 0] = [255, 255, 255]

    overlay = cv2.addWeighted(original, 1 - alpha, colored_mask, alpha, 0)
    return overlay
