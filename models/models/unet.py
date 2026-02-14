"""
Model Architecture Module
-------------------------
U-Net with ResNet34 encoder for nodule segmentation.
Uses segmentation_models_pytorch library.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import logging

logger = logging.getLogger(__name__)


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create U-Net model with ResNet34 encoder.
    
    Args:
        config: Model configuration from config.MODEL
        device: torch device
        
    Returns:
        model: U-Net model
    """
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=config['in_channels'],
        classes=config['classes'],
        activation=config['activation']
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created U-Net with {config['encoder_name']} encoder")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) after sigmoid
            target: Ground truth (B, 1, H, W) binary {0, 1}
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss as specified in poster."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits (B, 1, H, W)
            target: Ground truth (B, 1, H, W) binary {0, 1}
        """
        # BCE on logits
        bce_loss = self.bce(pred, target)
        
        # Dice on probabilities
        pred_prob = torch.sigmoid(pred)
        dice_loss = self.dice(pred_prob, target)
        
        # Combined
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU/Jaccard index).
    
    Args:
        pred: Predicted probabilities (B, 1, H, W) after sigmoid
        target: Ground truth (B, 1, H, W) binary {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score (float)
    """
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predicted probabilities (B, 1, H, W) after sigmoid
        target: Ground truth (B, 1, H, W) binary {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score (float)
    """
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
    
    return dice.item()
