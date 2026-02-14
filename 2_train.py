#!/usr/bin/env python3
"""
Step 2: Training Pipeline
-------------------------
Train U-Net model with proxy labels.

Usage:
    python 2_train.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config
from models.unet import create_model, CombinedLoss
from data.dataset import create_dataloaders
from training.trainer import Trainer
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_patch_manifest():
    """Load patch manifest from preprocessing step."""
    manifest_path = config.PATCHES_DIR / "patch_manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patch manifest not found: {manifest_path}\n"
            "Please run 1_preprocess_and_label.py first"
        )
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    logger.info(f"Loaded manifest with {len(manifest)} patches")
    return manifest


def split_dataset(manifest, train_split=0.8, val_split=0.1, test_split=0.1, random_seed=42):
    """Split patches into train/val/test sets."""
    logger.info(f"Splitting dataset: train={train_split}, val={val_split}, test={test_split}")
    
    # Extract paths
    image_paths = [item['image_path'] for item in manifest]
    mask_paths = [item['mask_path'] for item in manifest]
    
    # First split: train+val vs test
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_paths, mask_paths, 
        test_size=test_split, 
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_size_adjusted = val_split / (train_split + val_split)
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks,
        test_size=val_size_adjusted,
        random_state=random_seed
    )
    
    logger.info(f"Split sizes: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    # Save split info
    split_info = {
        'train': {'images': train_images, 'masks': train_masks},
        'val': {'images': val_images, 'masks': val_masks},
        'test': {'images': test_images, 'masks': test_masks}
    }
    
    split_path = config.PATCHES_DIR / "split_info.json"
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved split info: {split_path}")
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def main():
    """Run training pipeline."""
    # Setup logging
    setup_logging(
        log_dir=config.LOGS_DIR,
        log_level='INFO',
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info("="*80)
    logger.info("TRAINING PIPELINE")
    logger.info("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE['use_gpu'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Load manifest and split dataset
    manifest = load_patch_manifest()
    train_images, train_masks, val_images, val_masks, test_images, test_masks = split_dataset(
        manifest,
        train_split=config.TRAINING['train_split'],
        val_split=config.TRAINING['val_split'],
        test_split=config.TRAINING['test_split'],
        random_seed=config.TRAINING['random_seed']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_patches=train_images,
        train_masks=train_masks,
        val_patches=val_images,
        val_masks=val_masks,
        augmentation_config=config.AUGMENTATION,
        batch_size=config.TRAINING['batch_size'],
        num_workers=config.TRAINING['num_workers'],
        pin_memory=config.TRAINING['pin_memory']
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config.MODEL, device)
    
    # Create loss function
    criterion = CombinedLoss(
        bce_weight=config.TRAINING['bce_weight'],
        dice_weight=config.TRAINING['dice_weight']
    )
    logger.info(f"Using Combined Loss (BCE weight: {config.TRAINING['bce_weight']}, "
               f"Dice weight: {config.TRAINING['dice_weight']})")
    
    # Create optimizer
    if config.TRAINING['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAINING['learning_rate'],
            weight_decay=config.TRAINING['weight_decay']
        )
    elif config.TRAINING['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.TRAINING['learning_rate'],
            weight_decay=config.TRAINING['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.TRAINING['optimizer']}")
    
    logger.info(f"Using optimizer: {config.TRAINING['optimizer']}")
    
    # Create learning rate scheduler
    scheduler = None
    if config.TRAINING['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize IoU
            patience=config.TRAINING['scheduler_patience'],
            factor=config.TRAINING['scheduler_factor'],
            min_lr=config.TRAINING.get('scheduler_min_lr', 1e-7)
        )
        logger.info("Using ReduceLROnPlateau scheduler")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config.TRAINING,
        checkpoint_dir=config.CHECKPOINTS_DIR,
        scheduler=scheduler
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(num_epochs=config.TRAINING['num_epochs'])
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation IoU: {trainer.best_val_metric:.4f}")
    logger.info(f"Checkpoints saved to: {config.CHECKPOINTS_DIR}")
    logger.info(f"\nNext step: Run 3_inference.py to predict on full mosaics")


if __name__ == "__main__":
    main()
