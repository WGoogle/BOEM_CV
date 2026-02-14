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
        
        # 4. CLAHE in LAB space
        image = self.apply_clahe_lab(image)
        
        # 5. Bilateral filtering
        image = self.apply_bilateral_filter(image)
        
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
