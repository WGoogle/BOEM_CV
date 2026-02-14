#!/usr/bin/env python3
"""
Demo Script: Test Preprocessing on Example Data
----------------------------------------------
Quick test of preprocessing and proxy label generation.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent))

import config
from preprocessing.preprocess import MosaicPreprocessor, load_mosaic, save_mosaic
from preprocessing.proxy_labels import ProxyLabelGenerator, visualize_proxy_label
from utils.logger import setup_logging

# Setup logging
setup_logging(config.LOGS_DIR, log_level='INFO', log_to_file=False, log_to_console=True)
logger = logging.getLogger(__name__)

def demo_preprocessing():
    """Test preprocessing on example mosaics."""
    logger.info("="*80)
    logger.info("DEMO: Testing Preprocessing Pipeline")
    logger.info("="*80)
    
    # Get example mosaics
    mosaic_files = list(config.RAW_MOSAICS_DIR.glob("*.png"))
    
    if len(mosaic_files) == 0:
        logger.error(f"No mosaics found in {config.RAW_MOSAICS_DIR}")
        return
    
    logger.info(f"Found {len(mosaic_files)} example mosaics")
    
    # Create demo output directory
    demo_dir = config.OUTPUT_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    preprocessor = MosaicPreprocessor(config.PREPROCESSING)
    label_generator = ProxyLabelGenerator(config.PROXY_LABEL)
    
    # Process each mosaic
    for i, mosaic_path in enumerate(mosaic_files, 1):
        logger.info(f"\n[{i}/{len(mosaic_files)}] Processing: {mosaic_path.name}")
        
        try:
            # Load
            mosaic = load_mosaic(mosaic_path)
            logger.info(f"  Loaded: shape={mosaic.shape}, dtype={mosaic.dtype}")
            
            # Preprocess
            preprocessed = preprocessor.preprocess_mosaic(mosaic)
            logger.info(f"  Preprocessed: shape={preprocessed.shape}")
            
            # Generate proxy label
            proxy_mask, stats = label_generator.generate_proxy_label(preprocessed)
            logger.info(f"  Proxy label stats: {stats}")
            
            # Create visualization
            vis = visualize_proxy_label(preprocessed, proxy_mask, alpha=0.4)
            
            # Save results
            base_name = mosaic_path.stem
            
            # Save original
            save_mosaic(mosaic, demo_dir / f"{base_name}_1_original.png")
            
            # Save preprocessed
            save_mosaic(preprocessed, demo_dir / f"{base_name}_2_preprocessed.png")
            
            # Save proxy mask
            cv2.imwrite(str(demo_dir / f"{base_name}_3_proxy_mask.png"), proxy_mask)
            
            # Save visualization
            cv2.imwrite(str(demo_dir / f"{base_name}_4_visualization.png"), vis)
            
            logger.info(f"  ✓ Saved results to {demo_dir}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {demo_dir}")
    logger.info("\nYou can now:")
    logger.info("1. Check the demo results")
    logger.info("2. Adjust parameters in config.py if needed")
    logger.info("3. Add your full dataset to data/raw_mosaics/")
    logger.info("4. Run: python 1_preprocess_and_label.py")

if __name__ == "__main__":
    demo_preprocessing()
