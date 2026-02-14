"""
Metrics Module
-------------
Compute BOEM-relevant ecological metrics:
- Nodule count
- Nodules per square meter
- Percent coverage
- Size distribution
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional 
import logging
import json

logger = logging.getLogger(__name__)


class NoduleDensityAnalyzer:
    """Analyze nodule density and coverage from segmentation masks."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Metrics configuration from config.METRICS
        """
        self.config = config
        self.meters_per_pixel = config['meters_per_pixel']
        self.connectivity = config['connectivity']
        self.min_nodule_size = config['min_nodule_size']
        self.size_bins = config['size_bins']
    
    def analyze_mosaic(
        self,
        binary_mask: np.ndarray,
        mosaic_shape: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Compute all metrics for a single mosaic.
        
        Args:
            binary_mask: Binary segmentation (H, W) {0, 255}
            mosaic_shape: Optional shape for area calculation (H, W)
            
        Returns:
            Dictionary with all metrics
        """
        if mosaic_shape is None:
            mosaic_shape = binary_mask.shape
        
        H, W = mosaic_shape
        
        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=self.connectivity
        )
        
        # Filter by size (exclude background label 0)
        valid_nodules = []
        valid_areas = []
        valid_centroids = []
        
        for i in range(1, num_labels):  # Skip background (0)
            area_px = stats[i, cv2.CC_STAT_AREA]
            
            if area_px >= self.min_nodule_size:
                valid_nodules.append(i)
                valid_areas.append(area_px)
                valid_centroids.append(centroids[i])
        
        # Convert areas to physical units (m²)
        valid_areas_m2 = [area * (self.meters_per_pixel ** 2) for area in valid_areas]
        
        # Compute metrics
        total_nodules = len(valid_nodules)
        mosaic_area_m2 = H * W * (self.meters_per_pixel ** 2)
        nodules_per_m2 = total_nodules / mosaic_area_m2 if mosaic_area_m2 > 0 else 0.0
        
        # Coverage
        total_coverage_px = sum(valid_areas)
        total_pixels = H * W
        percent_coverage = (total_coverage_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        
        # Size distribution
        size_distribution = self._compute_size_distribution(valid_areas)
        
        # Statistics
        if len(valid_areas) > 0:
            mean_area = np.mean(valid_areas)
            median_area = np.median(valid_areas)
            std_area = np.std(valid_areas)
            min_area = np.min(valid_areas)
            max_area = np.max(valid_areas)
            
            mean_area_m2 = np.mean(valid_areas_m2)
            median_area_m2 = np.median(valid_areas_m2)
        else:
            mean_area = median_area = std_area = min_area = max_area = 0.0
            mean_area_m2 = median_area_m2 = 0.0
        
        metrics = {
            'total_nodules': total_nodules,
            'mosaic_area_m2': mosaic_area_m2,
            'nodules_per_m2': nodules_per_m2,
            'percent_coverage': percent_coverage,
            'size_distribution': size_distribution,
            'area_statistics_px': {
                'mean': float(mean_area),
                'median': float(median_area),
                'std': float(std_area),
                'min': float(min_area),
                'max': float(max_area)
            },
            'area_statistics_m2': {
                'mean': float(mean_area_m2),
                'median': float(median_area_m2),
            },
            'centroids': valid_centroids,
            'individual_areas_px': valid_areas,
            'individual_areas_m2': valid_areas_m2
        }
        
        logger.info(f"Analyzed mosaic: {total_nodules} nodules, "
                   f"{nodules_per_m2:.2f} nodules/m², {percent_coverage:.2f}% coverage")
        
        return metrics
    
    def _compute_size_distribution(self, areas: List[float]) -> Dict[str, int]:
        """
        Compute size distribution histogram.
        
        Args:
            areas: List of nodule areas in pixels
            
        Returns:
            Dictionary with bin labels and counts
        """
        if len(areas) == 0:
            return {f"{self.size_bins[i]}-{self.size_bins[i+1]}": 0 
                   for i in range(len(self.size_bins) - 1)}
        
        # Compute histogram
        hist, _ = np.histogram(areas, bins=self.size_bins)
        
        # Create labeled distribution
        distribution = {}
        for i in range(len(self.size_bins) - 1):
            bin_label = f"{self.size_bins[i]}-{self.size_bins[i+1]}"
            distribution[bin_label] = int(hist[i])
        
        return distribution
    
    def analyze_dataset(
        self,
        masks_dir: Path,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Analyze all mosaics in a directory and aggregate statistics.
        
        Args:
            masks_dir: Directory containing binary masks
            output_path: Optional path to save aggregated results
            
        Returns:
            Aggregated metrics dictionary
        """
        masks_dir = Path(masks_dir)
        mask_files = sorted(masks_dir.glob("*.png"))
        
        if len(mask_files) == 0:
            logger.warning(f"No mask files found in {masks_dir}")
            return {}
        
        logger.info(f"Analyzing {len(mask_files)} mosaics from {masks_dir}")
        
        all_metrics = []
        
        for mask_path in mask_files:
            # Load mask
            binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                continue
            
            # Analyze
            metrics = self.analyze_mosaic(binary_mask)
            metrics['filename'] = mask_path.name
            all_metrics.append(metrics)
        
        # Aggregate statistics
        total_nodules = sum(m['total_nodules'] for m in all_metrics)
        total_area_m2 = sum(m['mosaic_area_m2'] for m in all_metrics)
        avg_nodules_per_m2 = np.mean([m['nodules_per_m2'] for m in all_metrics])
        avg_coverage = np.mean([m['percent_coverage'] for m in all_metrics])
        
        aggregated = {
            'total_mosaics': len(all_metrics),
            'total_nodules': total_nodules,
            'total_area_m2': total_area_m2,
            'average_nodules_per_m2': float(avg_nodules_per_m2),
            'average_percent_coverage': float(avg_coverage),
            'per_mosaic_metrics': all_metrics
        }
        
        logger.info(f"Dataset analysis: {total_nodules} total nodules, "
                   f"{avg_nodules_per_m2:.2f} avg nodules/m², {avg_coverage:.2f}% avg coverage")
        
        # Save results
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                # Remove centroids for JSON serialization
                serializable = aggregated.copy()
                for m in serializable['per_mosaic_metrics']:
                    m.pop('centroids', None)
                    m.pop('individual_areas_px', None)
                    m.pop('individual_areas_m2', None)
                
                json.dump(serializable, f, indent=2)
            
            logger.info(f"Saved aggregated metrics: {output_path}")
        
        return aggregated


def draw_nodule_centroids(
    image: np.ndarray,
    centroids: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
    thickness: int = -1
) -> np.ndarray:
    """
    Draw nodule centroids on image.
    
    Args:
        image: Input image (H, W, 3) BGR
        centroids: List of (x, y) centroid coordinates
        color: Color for centroids (BGR)
        radius: Radius of centroid circles
        thickness: -1 for filled, positive for outline
        
    Returns:
        Image with centroids drawn
    """
    result = image.copy()
    
    for centroid in centroids:
        x, y = int(centroid[0]), int(centroid[1])
        cv2.circle(result, (x, y), radius, color, thickness)
    
    return result


def create_summary_visualization(
    original: np.ndarray,
    binary_mask: np.ndarray,
    metrics: Dict,
    overlay_alpha: float = 0.4
) -> np.ndarray:
    """
    Create comprehensive visualization with overlay, centroids, and text.
    
    Args:
        original: Original mosaic (H, W, 3) BGR
        binary_mask: Binary mask (H, W) {0, 255}
        metrics: Metrics dictionary from analyze_mosaic
        overlay_alpha: Transparency for overlay
        
    Returns:
        Visualization image
    """
    # Create overlay
    overlay = original.copy()
    colored_mask = np.zeros_like(original)
    colored_mask[binary_mask > 0] = [255, 255, 255]
    result = cv2.addWeighted(overlay, 1 - overlay_alpha, colored_mask, overlay_alpha, 0)
    
    # Draw centroids
    if 'centroids' in metrics:
        result = draw_nodule_centroids(result, metrics['centroids'])
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 255, 255)  # Yellow
    
    texts = [
        f"Nodules: {metrics['total_nodules']}",
        f"Density: {metrics['nodules_per_m2']:.2f} /m2",
        f"Coverage: {metrics['percent_coverage']:.2f}%"
    ]
    
    y_offset = 30
    for i, text in enumerate(texts):
        cv2.putText(result, text, (10, y_offset + i * 30), font, font_scale, text_color, font_thickness)
    
    return result
