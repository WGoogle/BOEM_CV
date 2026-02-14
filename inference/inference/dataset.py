"""
Dataset Module for Nodule Segmentation
--------------------------------------
PyTorch Dataset for loading patches and proxy labels with augmentation.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class NoduleSegmentationDataset(Dataset):
    """Dataset for patch-based nodule segmentation with proxy labels."""
    
    def __init__(
        self,
        patch_paths: list,
        mask_paths: list,
        transform: Optional[A.Compose] = None,
        mode: str = 'train'
    ):
        """
        Args:
            patch_paths: List of paths to preprocessed image patches
            mask_paths: List of paths to corresponding proxy label masks
            transform: Albumentations transforms
            mode: 'train', 'val', or 'test'
        """
        assert len(patch_paths) == len(mask_paths), "Mismatch between patches and masks"
        
        self.patch_paths = [Path(p) for p in patch_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.transform = transform
        self.mode = mode
        
        logger.info(f"Created {mode} dataset with {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.patch_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and transform a single patch-mask pair.
        
        Returns:
            image: Tensor (C, H, W) float32, normalized to [0, 1]
            mask: Tensor (1, H, W) float32, binary {0, 1}
        """
        # Load image and mask
        image = cv2.imread(str(self.patch_paths[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Failed to load patch {idx}: {self.patch_paths[idx]}")
        
        # Convert BGR to RGB for standard processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize mask to binary {0, 1}
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask has channel dimension [1, H, W]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Default: just convert to tensor and normalize
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask


def get_training_augmentation(config: dict) -> A.Compose:
    """
    Build training augmentation pipeline based on poster methodology.
    Includes: brightness/contrast, color jitter, Gaussian blur, geometric transforms.
    """
    aug_config = config.get('train', {})
    
    transforms = [
        # Brightness and contrast adjustments
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.get('brightness_limit', 0.2),
            contrast_limit=aug_config.get('contrast_limit', 0.2),
            p=0.5
        ),
        
        # Color jitter (simulate underwater color variation)
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        
        # Gaussian blur (simulate focus issues)
        A.GaussianBlur(
            blur_limit=aug_config.get('blur_limit', 7),
            p=0.3
        ),
        
        # Gaussian noise
        A.GaussNoise(p=0.2),
        
        # Geometric transforms
        A.HorizontalFlip(p=aug_config.get('horizontal_flip_p', 0.5)),
        A.VerticalFlip(p=aug_config.get('vertical_flip_p', 0.5)),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=aug_config.get('rotate_limit', 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
        
        # Elastic transform (underwater distortion)
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
        
        # Normalization and conversion to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_validation_augmentation() -> A.Compose:
    """Validation augmentation (only normalization)."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def create_dataloaders(
    train_patches: list,
    train_masks: list,
    val_patches: list,
    val_masks: list,
    augmentation_config: dict,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = NoduleSegmentationDataset(
        patch_paths=train_patches,
        mask_paths=train_masks,
        transform=get_training_augmentation(augmentation_config),
        mode='train'
    )
    
    val_dataset = NoduleSegmentationDataset(
        patch_paths=val_patches,
        mask_paths=val_masks,
        transform=get_validation_augmentation(),
        mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches")
    
    return train_loader, val_loader
