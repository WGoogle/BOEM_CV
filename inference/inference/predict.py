"""
Inference Module
---------------
Sliding-window inference for full mosaics with:
- Overlapping patch prediction
- Probability map blending
- Full-mosaic reconstruction
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class MosaicInference:
    """Sliding window inference for full mosaics."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            model: Trained segmentation model
            device: torch device
            config: Inference configuration from config.INFERENCE
            transform: Augmentation transforms (validation transforms)
        """
        self.model = model
        self.device = device
        self.config = config
        self.transform = transform
        
        if self.transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        self.model.eval()
        logger.info("Inference engine initialized")
    
    @torch.no_grad()
    def predict_mosaic(self, mosaic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation for full mosaic using sliding window.
        
        Args:
            mosaic: Preprocessed mosaic (H, W, 3) BGR
            
        Returns:
            probability_map: (H, W) float32 in [0, 1]
            binary_mask: (H, W) uint8 {0, 255}
        """
        H, W = mosaic.shape[:2]
        patch_h, patch_w = self.config['patch_size']
        stride_v, stride_h = self.config['stride']
        
        # Initialize accumulation arrays
        prob_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Generate patch positions
        y_positions = list(range(0, max(1, H - patch_h + 1), stride_v))
        x_positions = list(range(0, max(1, W - patch_w + 1), stride_h))
        
        # Ensure edge coverage
        if (H - patch_h) % stride_v != 0 and H >= patch_h:
            y_positions.append(H - patch_h)
        if (W - patch_w) % stride_h != 0 and W >= patch_w:
            x_positions.append(W - patch_w)
        
        # Remove duplicates
        y_positions = sorted(list(set(y_positions)))
        x_positions = sorted(list(set(x_positions)))
        
        total_patches = len(y_positions) * len(x_positions)
        logger.info(f"Predicting {total_patches} patches for mosaic of size {H}x{W}")
        
        # Process patches in batches
        batch_size = self.config.get('batch_size', 32)
        patch_batch = []
        coord_batch = []
        
        pbar = tqdm(total=total_patches, desc="Inference")
        
        for y in y_positions:
            for x in x_positions:
                # Extract patch
                patch = mosaic[y:y+patch_h, x:x+patch_w].copy()
                
                # Convert BGR to RGB
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                
                # Store patch and coordinates
                patch_batch.append(patch_rgb)
                coord_batch.append((y, x))
                
                # Process batch when full
                if len(patch_batch) >= batch_size:
                    self._process_batch(patch_batch, coord_batch, prob_map, count_map, patch_h, patch_w)
                    patch_batch = []
                    coord_batch = []
                    pbar.update(batch_size)
        
        # Process remaining patches
        if len(patch_batch) > 0:
            self._process_batch(patch_batch, coord_batch, prob_map, count_map, patch_h, patch_w)
            pbar.update(len(patch_batch))
        
        pbar.close()
        
        # Average overlapping predictions
        prob_map = np.divide(prob_map, count_map, where=count_map > 0)
        
        # Threshold to create binary mask
        threshold = self.config.get('probability_threshold', 0.5)
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        
        logger.info(f"Inference complete. Coverage: {(binary_mask > 0).sum() / binary_mask.size * 100:.2f}%")
        
        return prob_map, binary_mask
    
    def _process_batch(
        self,
        patches: list,
        coordinates: list,
        prob_map: np.ndarray,
        count_map: np.ndarray,
        patch_h: int,
        patch_w: int
    ):
        """Process a batch of patches and accumulate predictions."""
        # Transform patches
        transformed_patches = []
        for patch in patches:
            transformed = self.transform(image=patch)
            transformed_patches.append(transformed['image'])
        
        # Stack into batch tensor
        batch_tensor = torch.stack(transformed_patches).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.sigmoid(outputs)
            probs = probs.cpu().numpy()
        
        # Accumulate predictions
        for i, (y, x) in enumerate(coordinates):
            prob_patch = probs[i, 0]  # (H, W)
            prob_map[y:y+patch_h, x:x+patch_w] += prob_patch
            count_map[y:y+patch_h, x:x+patch_w] += 1.0
    
    def predict_from_file(
        self,
        mosaic_path: Path,
        save_prob_map: bool = True,
        save_binary_mask: bool = True,
        output_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mosaic from file, predict, and optionally save results.
        
        Args:
            mosaic_path: Path to preprocessed mosaic
            save_prob_map: Whether to save probability map
            save_binary_mask: Whether to save binary mask
            output_dir: Output directory for results
            
        Returns:
            probability_map, binary_mask
        """
        mosaic_path = Path(mosaic_path)
        
        # Load mosaic
        mosaic = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
        if mosaic is None:
            raise ValueError(f"Failed to load mosaic: {mosaic_path}")
        
        logger.info(f"Loaded mosaic: {mosaic_path.name}, shape={mosaic.shape}")
        
        # Predict
        prob_map, binary_mask = self.predict_mosaic(mosaic)
        
        # Save results
        if output_dir is not None and (save_prob_map or save_binary_mask):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = mosaic_path.stem
            
            if save_prob_map:
                prob_map_path = output_dir / f"{base_name}_prob.png"
                prob_map_uint8 = (prob_map * 255).astype(np.uint8)
                cv2.imwrite(str(prob_map_path), prob_map_uint8)
                logger.info(f"Saved probability map: {prob_map_path}")
            
            if save_binary_mask:
                mask_path = output_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), binary_mask)
                logger.info(f"Saved binary mask: {mask_path}")
        
        return prob_map, binary_mask


def visualize_segmentation(
    original: np.ndarray,
    binary_mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    overlay_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create visualization with segmentation overlay.
    
    Args:
        original: Original mosaic (H, W, 3) BGR
        binary_mask: Binary segmentation (H, W) {0, 255}
        prob_map: Optional probability map (H, W) [0, 1]
        alpha: Overlay transparency
        overlay_color: Color for overlay (BGR)
        
    Returns:
        Visualization (H, W, 3) BGR
    """
    # Create colored overlay
    overlay = original.copy()
    colored_mask = np.zeros_like(original)
    colored_mask[binary_mask > 0] = overlay_color
    
    # Blend
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def visualize_probability_map(
    prob_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Visualize probability map with colormap.
    
    Args:
        prob_map: Probability map (H, W) [0, 1]
        colormap: OpenCV colormap
        
    Returns:
        Colored visualization (H, W, 3) BGR
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(prob_uint8, colormap)
    return colored
