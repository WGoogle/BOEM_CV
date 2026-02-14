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
        
        # Create CLAHE for grayscale enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Create morphological kernels
        self.opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config['morph_opening_kernel'], config['morph_opening_kernel'])
        )
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config['morph_closing_kernel'], config['morph_closing_kernel'])
        )
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance local contrast."""
        enhanced = self.clahe.apply(gray)
        logger.debug("Applied CLAHE enhancement")
        return enhanced
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to stabilize thresholding."""
        if not self.config['apply_gaussian_blur']:
            return image
        
        ksize = self.config['gaussian_kernel_size']
        sigma = self.config['gaussian_sigma']
        
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        logger.debug(f"Applied Gaussian blur (ksize={ksize}, sigma={sigma})")
        return blurred
    
    def otsu_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's method for automatic thresholding.
        Returns binary mask where 255 = nodule candidate, 0 = background.
        """
        # Otsu's method
        threshold_value, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Invert if needed (we want nodules to be bright, background dark)
        # In underwater imagery, nodules are typically brighter than sediment
        # But we'll invert to get nodules as white (255)
        binary_inv = cv2.bitwise_not(binary)
        
        logger.debug(f"Otsu threshold value: {threshold_value:.2f}")
        return binary_inv
    
    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening and closing to remove artifacts and fill gaps.
        - Opening: removes small bright spots (noise)
        - Closing: fills small holes in nodules
        """
        # Opening (erosion followed by dilation)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.opening_kernel)
        
        # Closing (dilation followed by erosion)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.closing_kernel)
        
        logger.debug("Applied morphological opening and closing")
        return closed
    
    def compute_contour_features(self, contour) -> dict:
        """
        Compute geometric features of a contour for filtering.
        
        Returns:
            Dict with area, eccentricity, solidity
        """
        area = cv2.contourArea(contour)
        
        if area < 5:  # Too small to compute features reliably
            return {'area': area, 'eccentricity': 1.0, 'solidity': 0.0}
        
        # Fit ellipse to compute eccentricity (if enough points)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (MA, ma), _ = ellipse
                
                # Avoid division by zero
                if MA > 0:
                    eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA >= ma else 0.0
                else:
                    eccentricity = 0.0
            except:
                eccentricity = 0.0
        else:
            eccentricity = 0.0
        
        # Solidity = area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        return {
            'area': area,
            'eccentricity': eccentricity,
            'solidity': solidity
        }
    
    def filter_contours(self, binary: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Find contours and filter by area, eccentricity, and solidity.
        
        Returns:
            filtered_contours: List of valid contours
            filtered_mask: Binary mask with only valid nodules
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            logger.debug("No contours found")
            return [], np.zeros_like(binary)
        
        # Filter contours
        filtered_contours = []
        for contour in contours:
            features = self.compute_contour_features(contour)
            
            # Apply filters
            if (self.config['min_contour_area'] <= features['area'] <= self.config['max_contour_area'] and
                self.config['min_eccentricity'] <= features['eccentricity'] <= self.config['max_eccentricity'] and
                features['solidity'] >= self.config['min_solidity']):
                filtered_contours.append(contour)
        
        # Create filtered mask
        filtered_mask = np.zeros_like(binary)
        if len(filtered_contours) > 0:
            cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
        
        logger.debug(f"Filtered {len(filtered_contours)}/{len(contours)} contours")
        return filtered_contours, filtered_mask
    
    def generate_proxy_label(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Full proxy label generation pipeline.
        
        Args:
            image: Preprocessed patch/mosaic (H, W, C) in BGR
            
        Returns:
            proxy_mask: Binary mask (H, W) with 0=background, 255=nodule
            stats: Dictionary with generation statistics
        """
        # 1. Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # 2. Enhance contrast
        enhanced = self.enhance_contrast(gray)
        
        # 3. Gaussian blur for stability
        blurred = self.apply_gaussian_blur(enhanced)
        
        # 4. Otsu thresholding
        binary = self.otsu_threshold(blurred)
        
        # 5. Morphological cleanup
        cleaned = self.morphological_cleanup(binary)
        
        # 6. Filter contours
        contours, proxy_mask = self.filter_contours(cleaned)
        
        # Compute statistics
        stats = {
            'num_candidates_before_filter': len(cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]),
            'num_nodules_after_filter': len(contours),
            'coverage_percent': (np.sum(proxy_mask > 0) / proxy_mask.size) * 100
        }
        
        logger.info(f"Generated proxy label: {stats['num_nodules_after_filter']} nodules, "
                   f"{stats['coverage_percent']:.2f}% coverage")
        
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
