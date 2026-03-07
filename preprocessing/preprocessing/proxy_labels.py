"""
Proxy Label Generation Module
-----------------------------
Generates weak supervision labels using classical image processing:
1. Grayscale conversion
2. CLAHE enhancement (local contrast normalisation)
3. Gaussian blur for stability
4. Local-darkness threshold (morphological background subtraction)
5. Morphological operations (opening & closing)
6. Contour filtering by area, eccentricity, solidity, and circularity
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

    def local_darkness_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Threshold based on local darkness relative to a Gaussian-blur background
        estimate.  This is the key design decision for generalisation:

        Rather than a global threshold or morphological closing, the local
        background is estimated with a large Gaussian blur (the local weighted
        mean).  Subtracting gives local_darkness = background_mean - gray:

          • Sediment pixels are near their local mean → local_darkness ≈ 0
          • Nodule pixels are darker than the local mean → local_darkness > 0

        Why Gaussian, not morphological closing?
        After CLAHE the inter-nodule sediment gaps reach brightness 200-255 (the
        local histogram maximum).  A morphological closing picks these bright
        peaks as the background reference, making even average sediment appear
        "dark" (89 % of pixels exceed a threshold of 15).  A Gaussian gives
        the local *mean* instead; average sediment sits near that mean and
        scores ≈ 0, while dark nodules sit well below it and score ≥ 30.

        bg_blur_sigma controls the Gaussian radius (should be larger than the
        biggest expected nodule so the background estimate is not pulled down
        by individual nodules, yet small enough to handle illumination gradients).

        local_darkness_threshold sets the minimum contrast (mean-subtracted)
        to be considered a nodule candidate.
        """
        bg_sigma   = self.config.get('bg_blur_sigma', 40)
        darkness_t = self.config.get('local_darkness_threshold', 15)

        # Large Gaussian blur → smooth local mean background estimate
        background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), bg_sigma)

        # local_darkness: how much darker than local mean (clamped to ≥ 0)
        local_darkness = np.clip(
            background - gray.astype(np.float32), 0, 255
        ).astype(np.uint8)

        _, binary = cv2.threshold(local_darkness, darkness_t, 255, cv2.THRESH_BINARY)
        logger.debug(
            f"Local-darkness threshold: bg_blur σ={bg_sigma}px, "
            f"darkness_t={darkness_t}"
        )
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

    def filter_contours(self, binary: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Keep only contours whose shape metrics fall within configured nodule ranges.

        Four independent shape tests:

        Area — hard size bounds, rejects sub-pixel noise and huge blobs that
          are clearly sediment patches, not nodules.

        Solidity (area / convex-hull area) — rejects ragged, branching blobs
          that sediment aggregates produce; nodules are convex or near-convex.

        Eccentricity — size-dependent: small blobs use a strict limit to reject
          thin sediment streaks; large blobs (merging nodule pairs) get a relaxed
          limit because the merged outline becomes slightly elongated.

        Circularity (4π·area / perimeter²) — the tightest discriminator.
          A sediment trail or elongated streak has low circularity even when its
          fitted ellipse looks reasonable.  Nodules are compact and score ≥ 0.3.
          This filter catches cases that pass the other three tests.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], np.zeros_like(binary)

        large_area_thresh = self.config.get('large_area_threshold',     200)
        large_ecc_limit   = self.config.get('large_eccentricity_limit', 0.95)
        min_circ          = self.config.get('min_circularity',          0.35)

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

            # Circularity: rejects elongated sediment features
            if f['circularity'] < min_circ:
                continue

            filtered.append(c)

        mask = np.zeros_like(binary)
        if filtered:
            cv2.drawContours(mask, filtered, -1, 255, thickness=cv2.FILLED)

        logger.debug(
            f"Contour filter: {len(filtered)} kept / {len(contours)} candidates "
            f"(area, solidity, eccentricity, circularity≥{min_circ})"
        )
        return filtered, mask
    
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
                # Scale float images to 0-255 for saving
                if img.dtype != np.uint8:
                    img_save = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img_save = img
                cv2.imwrite(path, img_save)

        logger.info("=== Proxy label pipeline diagnostics ===")

        # 1. Grayscale
        gray = self.convert_to_grayscale(image)
        _dbg("1_gray", gray)

        # 2. CLAHE
        gray = self.apply_clahe(gray)
        _dbg("2_clahe", gray)

        # 3. Mild Gaussian blur
        blurred = self.apply_gaussian_blur(gray)
        _dbg("3_blurred", blurred)

        # 4. Local-darkness threshold — compute the raw darkness map first so we
        #    can log it before thresholding, then threshold.
        bg_sigma   = self.config.get('bg_blur_sigma', 40)
        darkness_t = self.config.get('local_darkness_threshold', 15)
        background     = cv2.GaussianBlur(blurred.astype(np.float32), (0, 0), bg_sigma)
        local_darkness = np.clip(background - blurred.astype(np.float32), 0, 255).astype(np.uint8)
        _dbg("4a_background", background.astype(np.uint8))
        _dbg("4b_local_darkness", local_darkness)   # ← KEY: if max is low, threshold is too high

        _, binary = cv2.threshold(local_darkness, darkness_t, 255, cv2.THRESH_BINARY)
        _dbg("4c_binary", binary)

        # 5. Morphological cleanup
        cleaned = self.morphological_cleanup(binary)
        _dbg("5_cleaned", cleaned)

        # 6. Shape-based contour filtering
        contours_raw, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, proxy_mask = self.filter_contours(cleaned)
        _dbg("6_proxy_mask", proxy_mask)

        stats = {
            'num_candidates_before_filter': len(contours_raw),
            'num_nodules_after_filter':     len(contours),
            'coverage_percent':             100.0 * np.sum(proxy_mask > 0) / proxy_mask.size,
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
