#!/usr/bin/env python3
"""
Step 1: Preprocessing Pipeline
------------------------------
Preprocesses raw mosaics and generates proxy labels for training.

Usage:
    python 1_preprocess_and_label.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config
from preprocessing.preprocessing.preprocess import MosaicPreprocessor, extract_patches, load_mosaic, save_mosaic
from preprocessing.preprocessing.proxy_labels import ProxyLabelGenerator, visualize_proxy_label
from utils.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def preprocess_mosaics():
    """Preprocess all raw mosaics."""
    logger.info("="*80)
    logger.info("STEP 1: PREPROCESSING RAW MOSAICS")
    logger.info("="*80)
    
    # Get list of mosaics
    raw_dir = config.RAW_MOSAICS_DIR
    mosaic_files = list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")) + \
                   list(raw_dir.glob("*.png"))
    
    if len(mosaic_files) == 0:
        logger.error(f"No mosaic files found in {raw_dir}")
        logger.error("Please place your .tif/.tiff/.png files in the data/raw_mosaics directory")
        return []
    
    logger.info(f"Found {len(mosaic_files)} mosaics to preprocess")
    
    # Initialize preprocessor
    preprocessor = MosaicPreprocessor(config.PREPROCESSING)
    
    # Process each mosaic
    preprocessed_files = []
    
    for mosaic_path in tqdm(mosaic_files, desc="Preprocessing mosaics"):
        try:
            # Load
            mosaic = load_mosaic(mosaic_path)
            
            # Preprocess
            preprocessed = preprocessor.preprocess_mosaic(mosaic)
            
            # Save
            output_path = config.PREPROCESSED_DIR / f"{mosaic_path.stem}_preprocessed.png"
            save_mosaic(preprocessed, output_path)
            
            preprocessed_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to preprocess {mosaic_path.name}: {e}")
            continue
    
    logger.info(f"Preprocessed {len(preprocessed_files)}/{len(mosaic_files)} mosaics")
    return preprocessed_files


def generate_proxy_labels(preprocessed_files):
    """Generate proxy labels for all preprocessed mosaics."""
    logger.info("="*80)
    logger.info("STEP 2: GENERATING PROXY LABELS")
    logger.info("="*80)
    
    # Initialize label generator
    label_generator = ProxyLabelGenerator(config.PROXY_LABEL)
    
    # Process each mosaic
    proxy_label_files = []
    
    for mosaic_path in tqdm(preprocessed_files, desc="Generating proxy labels"):
        try:
            # Load preprocessed mosaic
            mosaic = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
            
            # Generate proxy label
            proxy_mask, stats = label_generator.generate_proxy_label(mosaic)
            
            # Save mask
            mask_path = config.PROXY_LABELS_DIR / f"{mosaic_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), proxy_mask)
            
            # Save visualization
            if config.VISUALIZATION['save_proxy_labels']:
                vis = visualize_proxy_label(mosaic, proxy_mask, alpha=config.VISUALIZATION['overlay_alpha'])
                vis_path = config.PROXY_LABELS_DIR / f"{mosaic_path.stem}_vis.png"
                cv2.imwrite(str(vis_path), vis)
            
            proxy_label_files.append(mask_path)
            
        except Exception as e:
            logger.error(f"Failed to generate proxy label for {mosaic_path.name}: {e}")
            continue
    
    logger.info(f"Generated {len(proxy_label_files)}/{len(preprocessed_files)} proxy labels")
    return proxy_label_files


def extract_and_save_patches(preprocessed_files, proxy_label_files):
    """Extract patches from preprocessed mosaics and masks."""
    logger.info("="*80)
    logger.info("STEP 3: EXTRACTING PATCHES")
    logger.info("="*80)
    
    patch_config = config.PREPROCESSING
    
    # Create patch directories
    patch_images_dir = config.PATCHES_DIR / "images"
    patch_masks_dir = config.PATCHES_DIR / "masks"
    patch_images_dir.mkdir(parents=True, exist_ok=True)
    patch_masks_dir.mkdir(parents=True, exist_ok=True)
    
    total_patches = 0
    patch_manifest = []
    
    for mosaic_path, mask_path in tqdm(zip(preprocessed_files, proxy_label_files), 
                                       total=len(preprocessed_files),
                                       desc="Extracting patches"):
        try:
            # Load mosaic and mask
            mosaic = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Extract patches
            image_patches, coords = extract_patches(
                mosaic,
                patch_h=patch_config['patch_height'],
                patch_w=patch_config['patch_width'],
                stride_v=patch_config['patch_stride_vertical'],
                stride_h=patch_config['patch_stride_horizontal'],
                min_std=patch_config['min_patch_std'],
                min_mean=patch_config['min_patch_mean']
            )
            
            # Extract corresponding mask patches
            for i, (y, x) in enumerate(coords):
                patch_h = patch_config['patch_height']
                patch_w = patch_config['patch_width']
                mask_patch = mask[y:y+patch_h, x:x+patch_w]
                
                # Save patches
                patch_id = f"{mosaic_path.stem}_patch_{i:04d}"
                image_patch_path = patch_images_dir / f"{patch_id}.png"
                mask_patch_path = patch_masks_dir / f"{patch_id}.png"
                
                cv2.imwrite(str(image_patch_path), image_patches[i])
                cv2.imwrite(str(mask_patch_path), mask_patch)
                
                patch_manifest.append({
                    'image_path': str(image_patch_path),
                    'mask_path': str(mask_patch_path),
                    'source_mosaic': mosaic_path.name,
                    'coordinates': (y, x)
                })
                
                total_patches += 1
        
        except Exception as e:
            logger.error(f"Failed to extract patches from {mosaic_path.name}: {e}")
            continue
    
    logger.info(f"Extracted {total_patches} total patches")
    
    # Save manifest
    import json
    manifest_path = config.PATCHES_DIR / "patch_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(patch_manifest, f, indent=2)
    logger.info(f"Saved patch manifest: {manifest_path}")
    
    return patch_manifest


def main():
    """Run full preprocessing pipeline."""
    # Setup logging
    setup_logging(
        log_dir=config.LOGS_DIR,
        log_level='INFO',
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info("Starting preprocessing pipeline")
    logger.info(f"Raw mosaics directory: {config.RAW_MOSAICS_DIR}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    
    # Step 1: Preprocess mosaics
    preprocessed_files = preprocess_mosaics()
    
    if len(preprocessed_files) == 0:
        logger.error("No mosaics were preprocessed. Exiting.")
        return
    
    # Step 2: Generate proxy labels
    proxy_label_files = generate_proxy_labels(preprocessed_files)
    
    if len(proxy_label_files) == 0:
        logger.error("No proxy labels were generated. Exiting.")
        return
    
    # Step 3: Extract patches
    patch_manifest = extract_and_save_patches(preprocessed_files, proxy_label_files)
    
    logger.info("="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Preprocessed mosaics: {len(preprocessed_files)}")
    logger.info(f"Proxy labels: {len(proxy_label_files)}")
    logger.info(f"Patches extracted: {len(patch_manifest)}")
    logger.info(f"\nNext step: Run 2_train.py to train the model")


if __name__ == "__main__":
    main()
