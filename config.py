"""
Configuration file for Polymetallic Nodule Segmentation Pipeline
-----------------------------------------------------------------
All hyperparameters, paths, and settings are defined here.

NOTE: Keys here must exactly match what preprocess.py and proxy_labels.py read.
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_MOSAICS_DIR = DATA_DIR / "raw_mosaics"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
PROXY_LABELS_DIR = OUTPUT_DIR / "proxy_labels"
PATCHES_DIR = OUTPUT_DIR / "patches"
MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

for dir_path in [DATA_DIR, RAW_MOSAICS_DIR, OUTPUT_DIR, PREPROCESSED_DIR,
                 PROXY_LABELS_DIR, PATCHES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# PREPROCESSING  (read by preprocessing/preprocessing/preprocess.py)
# ==============================================================================
PREPROCESSING = {
    # --- Color normalization ---
    'apply_gray_world': True,

    # --- CLAHE in LAB space (MosaicPreprocessor.apply_clahe_lab) ---
    'apply_clahe': True,
    'clahe_clip_limit': 1.5,           # lowered from 3.0 — less sediment contrast boost
    'clahe_tile_grid_size': (16, 16),  # fewer, larger tiles → smoother local enhancement

    # --- Bilateral filtering (MosaicPreprocessor.apply_bilateral_filter) ---
    'apply_bilateral': True,
    'bilateral_d': 5,
    'bilateral_sigma_color': 50,
    'bilateral_sigma_space': 50,

    # --- Nodule boost (MosaicPreprocessor.apply_nodule_boost) ---
    'apply_nodule_boost': True,
    'nodule_boost': 2.0,           # amplification multiplier
    'nodule_morph_radius': 20,     # bottom-hat SE radius in pixels
    'sediment_texture_sigma': 2.0, # fine-scale Gaussian sigma for texture gate
    'sediment_texture_threshold': 12.0,  # texture score at which weight → 0
    'nodule_max_darkening': 70,    # maximum L-channel darkening applied

    # --- Sediment fade (MosaicPreprocessor.apply_sediment_fade) ---
    # Blends bright (sediment) regions toward a smooth large-scale version
    # of the L channel, washing out gray blobs without touching dark nodules.
    'apply_sediment_fade': True,
    'sediment_fade_blur_sigma': 15.0,  # large blur sigma to erase blurb structure
    'sediment_fade_strength': 0.6,     # 0 = no fade, 1 = fully replace with smooth
    'sediment_l_threshold': 80,        # L values above this are treated as sediment

    # --- Unsharp mask (MosaicPreprocessor.apply_unsharp_mask) ---
    'apply_unsharp_mask': True,
    'unsharp_sigma': 1.5,
    'unsharp_strength': 0.2,

    # --- Patch extraction (extract_patches) ---
    'patch_height': 192,
    'patch_width': 256,
    'patch_stride_vertical': 128,
    'patch_stride_horizontal': 128,
    'min_patch_std': 5.0,
    'min_patch_mean': 10.0,
}

# ==============================================================================
# PROXY LABEL GENERATION  (read by preprocessing/preprocessing/proxy_labels.py)
# ==============================================================================
PROXY_LABEL = {
    # --- Grayscale Gaussian blur before top-hat ---
    # Smooths sediment grain texture so the top-hat responds to
    # nodule-sized dark blobs, not individual grains.
    'apply_gaussian_blur': True,
    'gaussian_kernel_size': 15,
    'gaussian_sigma': 5.0,

    # --- CLAHE on the grayscale image ---
    # DISABLED: CLAHE amplifies local contrast at grain scale (exactly the
    # texture we want to reject). On this dataset it inflates >127 pixels
    # from 8.7% → 43.2%, making 30% of the image fire as nodule candidates.
    'apply_clahe': False,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': 32,

    # --- Multi-scale black top-hat transform ---
    # Replaces Gaussian background subtraction.  Morphological closing fills
    # dark pits smaller than the SE; subtracting the original gives a response
    # that is large *only* for dark blobs at the specified spatial scales.
    # Radii should be ≥ 12 to avoid responding to grain-level texture.
    'tophat_radii': [12, 20, 30],        # SE radii in pixels (small/medium/large)

    # Texture gate: suppress top-hat response in high local-std (grainy
    # sediment) areas.  Nodule surfaces are smooth → low local std.
    'tophat_texture_sigma': 2.5,         # fine-scale blur for texture measurement
    'tophat_texture_threshold': 10.0,    # texture score where weight → 0

    # Percentile-based threshold on the (texture-gated) top-hat response.
    # Only the brightest (100 − P)% of positive values are kept.
    # The floor prevents the threshold from dropping into noise.
    'tophat_percentile': 96,             # keep top 4% of top-hat response
    'tophat_threshold_floor': 15.0,      # never threshold below this value

    # Absolute intensity gate — only the darkest pixels in the grayscale
    # can be nodule candidates.
    'adaptive_abs_intensity': True,
    'adaptive_abs_percentile': 8,        # abs_max = Pth percentile of the grayscale
    'absolute_intensity_max': 85,        # fixed fallback when adaptive is False

    # --- Morphological cleanup ---
    # Opening kernel = 5 removes sub-grain noise blobs that survive the blur.
    # Closing kernel = 9 merges split nodule halves without bridging neighbours.
    'morph_opening_kernel': 5,
    'morph_closing_kernel': 9,

    # --- Contour shape filtering (ProxyLabelGenerator.filter_contours) ---
    'min_contour_area': 50,        # reject tiny noise blobs (<~8px diameter)
    'max_contour_area': 3000,      # allows large/merged nodule clusters
    'min_eccentricity': 0.0,
    'max_eccentricity': 0.80,     # tighter: reject elongated sediment streaks
    'min_solidity': 0.60,
    'min_circularity': 0.45,      # tighter: nodules are compact & round
    'large_area_threshold': 200,
    'large_eccentricity_limit': 0.95,
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
    'use_gpu': True,
}

# ==============================================================================
# TRAINING
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
    'meters_per_pixel': 0.005,
    'compute_nodules_per_sqm': True,
    'compute_percent_coverage': True,
    'size_bins': [0, 50, 100, 200, 500, 1000, 2000],
    'connectivity': 8,
    'min_nodule_size': 20,
}

# ==============================================================================
# VISUALIZATION
# ==============================================================================
VISUALIZATION = {
    'save_preprocessed': True,
    'save_proxy_labels': True,
    'save_probability_maps': True,
    'save_segmentation_overlays': True,
    'overlay_alpha': 0.4,
}
