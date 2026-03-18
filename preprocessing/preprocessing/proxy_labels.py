"""
Proxy Label Generation Module
-----------------------------
Generates weak supervision labels using classical image processing:
1. Grayscale conversion
2. CLAHE enhancement (local contrast normalisation)
3. Gaussian blur for stability
4. Multi-scale black top-hat transform (dark-blob detection)
5. Percentile-based thresholding with hard floor
6. Morphological operations (opening & closing)
7. Contour filtering by area, eccentricity, solidity, and circularity
   with per-reason rejection statistics
"""

import cv2
import numpy as np
from typing import Optional, Tuple
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

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        if not self.config.get('apply_clahe', True):
            return gray
        # clahe_tile_size is the desired tile SIZE in pixels (not the grid count).
        # Compute number of tiles so each tile is approximately that size.
        # OpenCV tileGridSize=(cols, rows) — note column-first ordering.
        tile_px = self.config.get('clahe_tile_size', 32)
        h, w    = gray.shape[:2]
        n_cols  = max(1, round(w / tile_px))
        n_rows  = max(1, round(h / tile_px))
        clahe   = cv2.createCLAHE(
            clipLimit=self.config.get('clahe_clip_limit', 2.0),
            tileGridSize=(n_cols, n_rows),
        )
        return clahe.apply(gray)

    def multi_scale_tophat(self, gray: np.ndarray) -> np.ndarray:
        """
        Multi-scale black top-hat transform for dark-blob detection.

        Black top-hat = morphological_close(image, SE) − image.
        The closing fills dark pits smaller than the structuring element,
        so the difference highlights *only* dark features at that spatial
        scale.  Unlike Gaussian background subtraction, it does not respond
        to gradual illumination gradients.

        Multiple SE radii are evaluated and combined with a pixel-wise max
        so nodules across a range of diameters are captured in one pass.

        A texture gate suppresses the response in high-texture (sediment
        grain) regions: local std is computed at a fine scale and used to
        weight down the top-hat response where the surface is rough.

        Returns the raw (float32) top-hat response — thresholding is done
        separately.
        """
        radii = self.config.get('tophat_radii', [12, 20, 30])
        combined = np.zeros(gray.shape, dtype=np.float32)

        for r in radii:
            se_size = 2 * r + 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
            tophat = cv2.subtract(closed, gray).astype(np.float32)
            combined = np.maximum(combined, tophat)

        # Texture gate: suppress top-hat response in high-texture (sediment)
        # regions.  Fine-scale local std captures grain texture; nodule
        # surfaces are smooth and have low local std.
        tex_sigma = self.config.get('tophat_texture_sigma', 2.5)
        tex_thresh = self.config.get('tophat_texture_threshold', 10.0)
        blur_fine = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), tex_sigma)
        local_diff = np.abs(gray.astype(np.float32) - blur_fine)
        texture_score = cv2.GaussianBlur(local_diff, (0, 0), tex_sigma * 2.0)
        # Weight: 1.0 for smooth (nodule), ramps to 0 for rough (sediment)
        texture_weight = np.clip(1.0 - texture_score / tex_thresh, 0.0, 1.0)
        combined = combined * texture_weight

        logger.debug(
            f"Multi-scale top-hat: radii={radii}, "
            f"response max={combined.max():.1f}, mean={combined.mean():.2f}, "
            f"texture gate σ={tex_sigma}, thresh={tex_thresh}"
        )
        return combined

    def tophat_percentile_threshold(
        self, tophat_response: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Percentile-based threshold on the top-hat response.

        Only the top (100 − P)% of *positive* top-hat values are kept as
        nodule candidates.  A hard floor prevents the threshold from dropping
        into the noise range on images with very little real signal.

        Returns:
            binary: uint8 mask (0/255)
            threshold: the computed threshold value (for logging)
        """
        pct = self.config.get('tophat_percentile', 85)
        floor = self.config.get('tophat_threshold_floor', 8.0)

        pos = tophat_response[tophat_response > 0]
        if pos.size > 0:
            threshold = float(np.percentile(pos, pct))
        else:
            threshold = floor

        threshold = max(threshold, floor)

        binary = (tophat_response >= threshold).astype(np.uint8) * 255
        logger.debug(
            f"Top-hat threshold: P{pct} of positive values = {threshold:.1f} "
            f"(floor={floor})"
        )
        return binary, threshold
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

        # Solidity = area / convex hull area
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity  = area / hull_area if hull_area > 0 else 0.0

        # Circularity = 4π·area / perimeter²
        # 1.0 = perfect circle; lower = more elongated or irregular.
        # More direct shape discriminator than eccentricity alone: an elongated
        # sediment streak has low circularity even if its ellipse fit looks OK.
        perimeter   = cv2.arcLength(contour, True)
        circularity = float((4.0 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0.0

        return {
            'area': area,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'circularity': circularity,
        }

    def filter_contours(self, binary: np.ndarray) -> Tuple[list, np.ndarray, dict]:
        """
        Keep only contours whose shape metrics fall within configured nodule ranges.
        Returns per-reason rejection counts for debugging.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], np.zeros_like(binary), {}

        large_area_thresh = self.config.get('large_area_threshold',     200)
        large_ecc_limit   = self.config.get('large_eccentricity_limit', 0.95)
        min_circ          = self.config.get('min_circularity',          0.35)

        # Per-reason rejection counters
        reject = {
            'area_too_small': 0,
            'area_too_large': 0,
            'low_solidity': 0,
            'eccentricity': 0,
            'low_circularity': 0,
        }

        filtered = []
        for c in contours:
            f = self.compute_contour_features(c)

            if f['area'] < self.config['min_contour_area']:
                reject['area_too_small'] += 1
                continue
            if f['area'] > self.config['max_contour_area']:
                reject['area_too_large'] += 1
                continue
            if f['solidity'] < self.config['min_solidity']:
                reject['low_solidity'] += 1
                continue

            # Size-dependent eccentricity threshold
            ecc_limit = large_ecc_limit if f['area'] >= large_area_thresh \
                        else self.config['max_eccentricity']
            if not (self.config['min_eccentricity'] <= f['eccentricity'] <= ecc_limit):
                reject['eccentricity'] += 1
                continue

            # Circularity: rejects elongated sediment features
            if f['circularity'] < min_circ:
                reject['low_circularity'] += 1
                continue

            filtered.append(c)

        mask = np.zeros_like(binary)
        if filtered:
            cv2.drawContours(mask, filtered, -1, 255, thickness=cv2.FILLED)

        total_rejected = sum(reject.values())
        logger.info(
            f"Contour filter: {len(filtered)} kept / {len(contours)} candidates "
            f"({total_rejected} rejected)"
        )
        for reason, count in reject.items():
            if count > 0:
                logger.info(f"  rejected [{reason}]: {count}")

        return filtered, mask, reject
    
    def generate_proxy_label(
        self,
        image: np.ndarray,
        debug_dir: Optional[str] = None,
        debug_prefix: str = "debug",
    ) -> Tuple[np.ndarray, dict]:
        """
        Full proxy label generation pipeline.

        Args:
            image:        Preprocessed mosaic/patch (H, W, C) BGR
            debug_dir:    If set, saves intermediate images to this directory
                          so you can inspect where the pipeline breaks down.
            debug_prefix: Filename prefix for debug images.

        Returns:
            proxy_mask: Binary mask (H, W) — 0 = background, 255 = nodule
            stats:      Generation statistics dict
        """
        def _dbg(name: str, img: np.ndarray):
            """Save a debug image and log its pixel statistics."""
            logger.info(
                f"  [{name}] shape={img.shape} dtype={img.dtype} "
                f"min={int(img.min())} max={int(img.max())} "
                f"mean={img.mean():.1f} nonzero={int(np.count_nonzero(img))}"
            )
            if debug_dir is not None:
                import os
                os.makedirs(debug_dir, exist_ok=True)
                path = os.path.join(debug_dir, f"{debug_prefix}_{name}.png")
                if img.dtype != np.uint8:
                    img_save = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img_save = img
                cv2.imwrite(path, img_save)

        logger.info("=== Proxy label pipeline diagnostics ===")

        # 1. Grayscale
        gray = self.convert_to_grayscale(image)
        _dbg("1_gray", gray)

        # 2. CLAHE (optional — disabled by default)
        gray = self.apply_clahe(gray)
        _dbg("2_clahe", gray)

        # Keep pre-blur grayscale for absolute intensity gate.
        gray_raw = gray.copy()

        # 3. Gaussian blur — smooths sediment grain texture so the top-hat
        #    responds to nodule-sized blobs, not individual grains.
        blurred = self.apply_gaussian_blur(gray)
        _dbg("3_blurred", blurred)

        # 4. Multi-scale black top-hat transform
        #    closing(img) − img highlights dark pits at each SE scale.
        #    Inherently ignores gradual illumination gradients and fine
        #    sediment texture (both are larger or smaller than the SE).
        tophat_response = self.multi_scale_tophat(blurred)
        _dbg("4a_tophat_response", tophat_response)

        # 4b. Percentile-based thresholding with hard floor
        binary, tophat_thresh = self.tophat_percentile_threshold(tophat_response)
        _dbg("4b_binary_tophat", binary)

        # 4c. Absolute intensity gate — only the darkest pixels in the
        #     original grayscale can be nodule candidates.
        if self.config.get('adaptive_abs_intensity', False):
            pct = self.config.get('adaptive_abs_percentile', 12)
            abs_max = float(np.percentile(gray_raw, pct))
            logger.info(
                f"  Adaptive abs gate: P{pct} of grayscale = {abs_max:.0f}"
            )
        else:
            abs_max = self.config.get('absolute_intensity_max', None)

        if abs_max is not None:
            abs_gate = (gray_raw <= abs_max).astype(np.uint8) * 255
            binary = cv2.bitwise_and(binary, abs_gate)
            _dbg("4c_abs_gate", abs_gate)
            _dbg("4d_binary_gated", binary)
            logger.debug(f"Absolute intensity gate: raw gray ≤ {abs_max}")

        # 5. Morphological cleanup
        cleaned = self.morphological_cleanup(binary)
        _dbg("5_cleaned", cleaned)

        # 6. Shape-based contour filtering with per-reason rejection stats
        contours_raw, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours, proxy_mask, rejection_stats = self.filter_contours(cleaned)
        _dbg("6_proxy_mask", proxy_mask)

        stats = {
            'num_candidates_before_filter': len(contours_raw),
            'num_nodules_after_filter':     len(contours),
            'coverage_percent':             100.0 * np.sum(proxy_mask > 0) / proxy_mask.size,
            'tophat_threshold':             tophat_thresh,
            'rejection_stats':              rejection_stats,
        }

        logger.info(
            f"Proxy label: {stats['num_nodules_after_filter']} nodules "
            f"(from {stats['num_candidates_before_filter']} candidates), "
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
