"""
Preprocessing Module for Seafloor Mosaics
-----------------------------------------
Implements the preprocessing pipeline from the poster:
1. Orientation normalization
2. Color space conversion and white balance
3. CLAHE contrast enhancement
4. Bilateral noise reduction
5. Patch extraction with filtering
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MosaicPreprocessor:
    """Preprocesses raw seafloor mosaics according to the paper methodology."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with preprocessing parameters from config.PREPROCESSING
        """
        self.config = config
        self.clahe = None
        if config['apply_clahe']:
            self.clahe = cv2.createCLAHE(
                clipLimit=config['clahe_clip_limit'],
                tileGridSize=config['clahe_tile_grid_size']
            )
    
    def normalize_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure consistent orientation. 
        If image is taller than wide, rotate 90 degrees.
        """
        h, w = image.shape[:2]
        if h > w:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            logger.debug("Rotated image 90° clockwise for orientation normalization")
        return image
    
    def convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGBA -> RGB -> BGR for OpenCV compatibility.
        Handles various input formats.
        """
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            logger.debug("Converted RGBA to RGB")
        
        if image.shape[2] == 3:  # Ensure BGR for OpenCV
            # OpenCV uses BGR by default, but if loaded from PIL it's RGB
            # We'll work in BGR throughout
            pass
        
        return image
    
    def apply_gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gray-world white balance to reduce illumination bias from AUV/ROV lighting.
        """
        if not self.config['apply_gray_world']:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Compute channel means
        b_mean = np.mean(img_float[:, :, 0])
        g_mean = np.mean(img_float[:, :, 1])
        r_mean = np.mean(img_float[:, :, 2])
        
        # Gray world assumption: average gray = (b+g+r)/3
        gray = (b_mean + g_mean + r_mean) / 3.0
        
        # Avoid division by zero
        if b_mean == 0 or g_mean == 0 or r_mean == 0:
            logger.warning("Zero channel mean detected, skipping white balance")
            return image
        
        # Scale each channel
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (gray / b_mean), 0, 255)
        img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (gray / g_mean), 0, 255)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (gray / r_mean), 0, 255)
        
        logger.debug("Applied gray-world white balance")
        return img_float.astype(np.uint8)
    
    def apply_clahe_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE in LAB color space to boost local contrast.
        """
        if not self.config['apply_clahe'] or self.clahe is None:
            return image
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.debug("Applied CLAHE in LAB space")
        return enhanced
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to suppress backscatter and sensor noise 
        while preserving nodule edges.
        """
        if not self.config['apply_bilateral']:
            return image
        
        filtered = cv2.bilateralFilter(
            image,
            d=self.config['bilateral_d'],
            sigmaColor=self.config['bilateral_sigma_color'],
            sigmaSpace=self.config['bilateral_sigma_space']
        )
        
        logger.debug("Applied bilateral filter")
        return filtered

    def apply_nodule_boost(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance dark nodule blobs while suppressing sediment texture using two
        complementary signals:

        1. Morphological Bottom-Hat Transform
           A morphological closing with an elliptical SE sized to the expected
           maximum nodule diameter fills in dark nodule "holes", turning them
           bright.  Subtracting the original (bottom-hat = close(L) - L) gives
           a response that is large only where a dark compact blob smaller than
           the SE exists — i.e., exactly where nodules sit.  This avoids the
           Gaussian-blur halo artefact of the previous approach.

        2. Sediment Texture Gate
           Fine-grained sediment has high local standard deviation at small
           pixel scales (2-3 px) because individual grains differ in brightness.
           Nodule surfaces are smooth, so their local std is low at the same
           scale.  A per-pixel weight ramps from 1 → 0 as the local std rises
           above `sediment_texture_threshold`, preventing sediment from being
           darkened even if it happens to sit near the bottom-hat response.

        Both signals are computed on the PRE-CLAHE L channel to avoid the
        bright halos CLAHE creates around nodule edges.
        """
        if not self.config.get('apply_nodule_boost', False):
            return image

        amplify       = self.config.get('nodule_boost', 2.0)
        morph_radius  = self.config.get('nodule_morph_radius', 20)
        tex_sigma     = self.config.get('sediment_texture_sigma', 2.0)
        tex_threshold = self.config.get('sediment_texture_threshold', 12.0)
        max_darken    = self.config.get('nodule_max_darkening', 70)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Use pre-CLAHE L to avoid halo artefacts (set by preprocess_mosaic).
        l_ref    = getattr(self, '_l_pre_clahe', l)
        l_ref_u8 = l_ref.astype(np.uint8)

        # ── 1. Bottom-Hat Transform ──────────────────────────────────────────
        # SE diameter larger than the biggest expected nodule so closing fully
        # fills dark nodule regions.  Elliptical SE matches nodule morphology.
        se_size    = 2 * morph_radius + 1
        se         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        closed     = cv2.morphologyEx(l_ref_u8, cv2.MORPH_CLOSE, se)
        bottom_hat = cv2.subtract(closed, l_ref_u8).astype(np.float32)  # [0, 255]

        # ── 2. Sediment Texture Gate ─────────────────────────────────────────
        # Local mean at fine scale; deviation from it ≈ grain-level texture.
        blur_fine     = cv2.GaussianBlur(l_ref_u8.astype(np.float32), (0, 0), tex_sigma)
        abs_diff      = np.abs(l_ref_u8.astype(np.float32) - blur_fine)
        texture_score = cv2.GaussianBlur(abs_diff, (0, 0), tex_sigma * 1.5)

        # Weight: 1 for smooth surfaces (nodules), ramps to 0 for textured sediment.
        texture_weight = np.clip(1.0 - texture_score / tex_threshold, 0.0, 1.0)

        # ── 3. Combined Darkening ────────────────────────────────────────────
        # nodule_signal ∈ [0, 1]: high only where bottom-hat response is strong
        # AND the surface is smooth (low texture).
        nodule_signal = (bottom_hat / 255.0) * texture_weight
        darkening     = np.clip(amplify * nodule_signal * bottom_hat, 0.0, max_darken)

        l_boosted = np.clip(l.astype(np.float32) - darkening, 0, 255).astype(np.uint8)

        logger.debug(
            f"Nodule boost: bottom-hat SE r={morph_radius}px, "
            f"tex gate σ={tex_sigma}/{tex_threshold}, "
            f"max_darken={max_darken}, amplify={amplify}"
        )
        return cv2.cvtColor(cv2.merge([l_boosted, a, b]), cv2.COLOR_LAB2BGR)
    
    def apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen nodule boundaries by applying unsharp masking on the L channel.
        Controlled by config keys: apply_unsharp_mask, unsharp_sigma, unsharp_strength.
        """
        if not self.config.get('apply_unsharp_mask', False):
            return image

        sigma    = self.config.get('unsharp_sigma', 2.0)
        strength = self.config.get('unsharp_strength', 0.5)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        blur    = cv2.GaussianBlur(l, (0, 0), sigma)
        l_sharp = np.clip(l.astype(np.int16) + strength * (l.astype(np.int16) - blur.astype(np.int16)), 0, 255).astype(np.uint8)

        sharpened = cv2.merge([l_sharp, a, b])
        logger.debug("Applied unsharp mask")
        return cv2.cvtColor(sharpened, cv2.COLOR_LAB2BGR)
    
    
    def preprocess_mosaic(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a single mosaic.
        
        Args:
            image: Input mosaic (H, W, C) in BGR format
            
        Returns:
            Preprocessed mosaic (H, W, 3) uint8
        """
        
        # 1. Orientation normalization
        image = self.normalize_orientation(image)
        
        # 2. Color space conversion (handles RGBA if needed)
        image = self.convert_color_space(image)
        
        # 3. Gray-world white balance
        image = self.apply_gray_world_white_balance(image)
        
        # 4. Bilateral filtering
        image = self.apply_bilateral_filter(image)

        # Snapshot the L channel before CLAHE so apply_nodule_boost can use
        # true (halo-free) contrast for its darkness estimate
        self._l_pre_clahe = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))[0]

        # 5. CLAHE in LAB space
        image = self.apply_clahe_lab(image)
        
        # 6. applying nodule boost 
        image = self.apply_nodule_boost(image)

        # 7. Unsharp masking 
        image = self.apply_unsharp_mask(image)
        
        logger.info(f"Preprocessed mosaic: shape={image.shape}, dtype={image.dtype}")
        return image


def extract_patches(
    image: np.ndarray,
    patch_h: int,
    patch_w: int,
    stride_v: int,
    stride_h: int,
    min_std: float = 5.0,
    min_mean: float = 10.0
) -> Tuple[list, list]:
    """
    Extract overlapping patches from mosaic with quality filtering.
    
    Args:
        image: Preprocessed mosaic (H, W, C)
        patch_h: Patch height
        patch_w: Patch width
        stride_v: Vertical stride
        stride_h: Horizontal stride
        min_std: Minimum standard deviation to keep patch
        min_mean: Minimum mean intensity to keep patch
        
    Returns:
        patches: List of patches (patch_h, patch_w, C)
        coordinates: List of (y, x) top-left coordinates
    """
    H, W = image.shape[:2]
    patches = []
    coordinates = []
    
    # Generate patch positions
    y_positions = list(range(0, H - patch_h + 1, stride_v))
    x_positions = list(range(0, W - patch_w + 1, stride_h))
    
    # Ensure we cover the right and bottom edges
    if (H - patch_h) % stride_v != 0 and H >= patch_h:
        y_positions.append(H - patch_h)
    if (W - patch_w) % stride_h != 0 and W >= patch_w:
        x_positions.append(W - patch_w)
    
    # Remove duplicates
    y_positions = sorted(list(set(y_positions)))
    x_positions = sorted(list(set(x_positions)))
    
    # Extract patches with filtering
    for y in y_positions:
        for x in x_positions:
            patch = image[y:y+patch_h, x:x+patch_w]
            
            # Quality checks
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.shape[2] == 3 else patch
            patch_std = np.std(gray)
            patch_mean = np.mean(gray)
            
            if patch_std >= min_std and patch_mean >= min_mean:
                patches.append(patch.copy())
                coordinates.append((y, x))
    
    logger.info(f"Extracted {len(patches)} valid patches from {len(y_positions)*len(x_positions)} total positions")
    return patches, coordinates


def load_mosaic(filepath: Path) -> np.ndarray:
    """
    Load a mosaic from file (supports .tif, .tiff, .png).
    
    Returns:
        Image in BGR format for OpenCV
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() in ['.tif', '.tiff']:
        # Use OpenCV for TIFF
        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load TIFF: {filepath}")
    elif filepath.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Loaded mosaic: {filepath.name}, shape={image.shape}")
    return image


def save_mosaic(image: np.ndarray, filepath: Path):
    """Save mosaic to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(filepath), image)
    if not success:
        raise RuntimeError(f"Failed to save image: {filepath}")
    
    logger.info(f"Saved mosaic: {filepath}")
