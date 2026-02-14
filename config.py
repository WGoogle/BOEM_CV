"""
Configuration file for Polymetallic Nodule Segmentation Pipeline
-----------------------------------------------------------------
All hyperparameters, paths, and settings are defined here.
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_MOSAICS_DIR = DATA_DIR / "raw_mosaics"  # PLACEHOLDER: Put your .tif files here
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
PROXY_LABELS_DIR = OUTPUT_DIR / "proxy_labels"
PATCHES_DIR = OUTPUT_DIR / "patches"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_MOSAICS_DIR, OUTPUT_DIR, PREPROCESSED_DIR, 
                 PROXY_LABELS_DIR, PATCHES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# PREPROCESSING PARAMETERS (Based on poster methodology)
# ==============================================================================
PREPROCESSING = {
    # Color normalization
    'apply_gray_world': True,
    'gray_world_percentile': 50,
    
    # CLAHE parameters (LAB space)
    'apply_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    
    # Bilateral filtering for noise reduction
    'apply_bilateral': True,
    'bilateral_d': 9,
    'bilateral_sigma_color': 75,
    'bilateral_sigma_space': 75,
    
    # Patch extraction
    'patch_height': 192,
    'patch_width': 256,
    'patch_stride_vertical': 128,
    'patch_stride_horizontal': 128,
    'min_patch_std': 5.0,
    'min_patch_mean': 10.0,
}

# ==============================================================================
# PROXY LABEL GENERATION (Classical CV for weak supervision)
# ==============================================================================
PROXY_LABEL = {
    # Otsu thresholding parameters
    'apply_gaussian_blur': True,
    'gaussian_kernel_size': 5,
    'gaussian_sigma': 1.0,
    
    # Morphological operations
    'morph_opening_kernel': 3,
    'morph_closing_kernel': 5,
    
    # Contour filtering (nodule characteristics)
    'min_contour_area': 20,
    'max_contour_area': 2000,
    'min_eccentricity': 0.0,
    'max_eccentricity': 0.95,
    'min_solidity': 0.5,
}

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
MODEL = {
    'architecture': 'Unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'activation': None,
}

# ==============================================================================
# DEVICE
# ==============================================================================
DEVICE = {
    'use_gpu': True,  # Set False to force CPU
}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
TRAINING = {
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'random_seed': 42,
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'bce_weight': 0.5,
    'dice_weight': 0.5,
    'use_scheduler': True,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'early_stopping': True,
    'early_stopping_patience': 15,
    'use_amp': True,
    'num_workers': 4,
    'pin_memory': True,
}

# ==============================================================================
# AUGMENTATION
# ==============================================================================
AUGMENTATION = {
    'train': {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'blur_limit': 7,
        'rotate_limit': 15,
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
    },
    'val': {},
    'test': {},
}

# ==============================================================================
# INFERENCE
# ==============================================================================
INFERENCE = {
    'patch_size': (192, 256),
    'stride': (128, 128),
    'batch_size': 32,
    'probability_threshold': 0.5,
    'blend_mode': 'average',
}

# ==============================================================================
# METRICS
# ==============================================================================
METRICS = {
    'meters_per_pixel': 0.005,  # ADJUST THIS
    'compute_nodules_per_sqm': True,
    'compute_percent_coverage': True,
    'size_bins': [0, 50, 100, 200, 500, 1000, 2000],
    'connectivity': 8,
    'min_nodule_size': 20,
}

VISUALIZATION = {
    'save_preprocessed': True,
    'save_proxy_labels': True,
    'save_probability_maps': True,
    'save_segmentation_overlays': True,
    'overlay_alpha': 0.4,
}
