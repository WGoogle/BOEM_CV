#!/usr/bin/env python3
"""
Step 3: Inference and Metrics Pipeline
--------------------------------------
Run inference on full mosaics and compute density metrics.

Usage:
    python 3_inference.py [--checkpoint best]
"""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config
from models.unet import create_model
from inference.predict import MosaicInference, visualize_segmentation, visualize_probability_map
from utils.metrics import NoduleDensityAnalyzer, create_summary_visualization
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, model, device):
    """Load model checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        logger.info(f"Validation metrics: {checkpoint['val_metrics']}")
    
    return model


def run_inference_on_mosaics(model, device, mosaic_paths):
    """Run inference on all mosaics."""
    logger.info("="*80)
    logger.info("RUNNING INFERENCE ON MOSAICS")
    logger.info("="*80)
    
    # Create inference engine
    inference_engine = MosaicInference(
        model=model,
        device=device,
        config=config.INFERENCE
    )
    
    # Create output directories
    prob_maps_dir = config.RESULTS_DIR / "probability_maps"
    binary_masks_dir = config.RESULTS_DIR / "binary_masks"
    overlays_dir = config.RESULTS_DIR / "overlays"
    
    for dir_path in [prob_maps_dir, binary_masks_dir, overlays_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each mosaic
    results = []
    
    for mosaic_path in tqdm(mosaic_paths, desc="Inference"):
        try:
            # Load mosaic
            mosaic = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
            if mosaic is None:
                logger.error(f"Failed to load mosaic: {mosaic_path}")
                continue
            
            # Predict
            prob_map, binary_mask = inference_engine.predict_mosaic(mosaic)
            
            # Save results
            base_name = mosaic_path.stem
            
            # Probability map
            if config.VISUALIZATION['save_probability_maps']:
                prob_map_vis = visualize_probability_map(prob_map)
                prob_path = prob_maps_dir / f"{base_name}_prob.png"
                cv2.imwrite(str(prob_path), prob_map_vis)
            
            # Binary mask
            mask_path = binary_masks_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_path), binary_mask)
            
            # Overlay visualization
            if config.VISUALIZATION['save_segmentation_overlays']:
                overlay = visualize_segmentation(
                    mosaic, binary_mask, 
                    alpha=config.VISUALIZATION['overlay_alpha']
                )
                overlay_path = overlays_dir / f"{base_name}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
            
            results.append({
                'mosaic_path': mosaic_path,
                'mask_path': mask_path,
                'mosaic_shape': mosaic.shape[:2]
            })
            
        except Exception as e:
            logger.error(f"Failed to process {mosaic_path.name}: {e}")
            continue
    
    logger.info(f"Inference complete: {len(results)}/{len(mosaic_paths)} mosaics processed")
    return results


def compute_metrics(inference_results):
    """Compute nodule density metrics for all mosaics."""
    logger.info("="*80)
    logger.info("COMPUTING METRICS")
    logger.info("="*80)
    
    # Create metrics analyzer
    analyzer = NoduleDensityAnalyzer(config.METRICS)
    
    # Create output directory
    metrics_dir = config.RESULTS_DIR / "metrics"
    visualizations_dir = config.RESULTS_DIR / "visualizations"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each mosaic
    all_metrics = []
    
    for result in tqdm(inference_results, desc="Computing metrics"):
        try:
            # Load binary mask
            binary_mask = cv2.imread(str(result['mask_path']), cv2.IMREAD_GRAYSCALE)
            
            # Analyze
            metrics = analyzer.analyze_mosaic(binary_mask, result['mosaic_shape'])
            metrics['mosaic_name'] = result['mosaic_path'].stem
            
            # Save individual metrics
            import json
            metrics_copy = metrics.copy()
            metrics_copy.pop('centroids', None)  # Remove for JSON serialization
            
            metrics_file = metrics_dir / f"{result['mosaic_path'].stem}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_copy, f, indent=2)
            
            # Create summary visualization
            mosaic = cv2.imread(str(result['mosaic_path']), cv2.IMREAD_COLOR)
            summary_vis = create_summary_visualization(
                mosaic, binary_mask, metrics,
                overlay_alpha=config.VISUALIZATION['overlay_alpha']
            )
            vis_path = visualizations_dir / f"{result['mosaic_path'].stem}_summary.png"
            cv2.imwrite(str(vis_path), summary_vis)
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to compute metrics for {result['mosaic_path'].name}: {e}")
            continue
    
    # Aggregate metrics
    if len(all_metrics) > 0:
        total_nodules = sum(m['total_nodules'] for m in all_metrics)
        total_area_m2 = sum(m['mosaic_area_m2'] for m in all_metrics)
        avg_density = np.mean([m['nodules_per_m2'] for m in all_metrics])
        avg_coverage = np.mean([m['percent_coverage'] for m in all_metrics])
        
        aggregated = {
            'total_mosaics': len(all_metrics),
            'total_nodules': int(total_nodules),
            'total_area_m2': float(total_area_m2),
            'average_nodules_per_m2': float(avg_density),
            'average_percent_coverage': float(avg_coverage),
            'per_mosaic_summary': [
                {
                    'mosaic_name': m['mosaic_name'],
                    'total_nodules': m['total_nodules'],
                    'nodules_per_m2': m['nodules_per_m2'],
                    'percent_coverage': m['percent_coverage']
                }
                for m in all_metrics
            ]
        }
        
        # Save aggregated metrics
        import json
        agg_path = config.RESULTS_DIR / "aggregated_metrics.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Saved aggregated metrics: {agg_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("METRICS SUMMARY")
        logger.info("="*80)
        logger.info(f"Total mosaics analyzed: {aggregated['total_mosaics']}")
        logger.info(f"Total nodules detected: {aggregated['total_nodules']}")
        logger.info(f"Total area surveyed: {aggregated['total_area_m2']:.2f} m²")
        logger.info(f"Average density: {aggregated['average_nodules_per_m2']:.2f} nodules/m²")
        logger.info(f"Average coverage: {aggregated['average_percent_coverage']:.2f}%")
        
        return aggregated, all_metrics
    else:
        logger.warning("No metrics computed")
        return None, []


def create_summary_plots(metrics_list):
    """Create summary plots for the dive."""
    logger.info("Creating summary plots...")
    
    if len(metrics_list) == 0:
        logger.warning("No metrics available for plotting")
        return
    
    # Extract data
    mosaic_names = [m['mosaic_name'] for m in metrics_list]
    densities = [m['nodules_per_m2'] for m in metrics_list]
    coverages = [m['percent_coverage'] for m in metrics_list]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Density plot
    ax1.bar(range(len(mosaic_names)), densities, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Mosaic Index', fontsize=12)
    ax1.set_ylabel('Nodules per m²', fontsize=12)
    ax1.set_title('Nodule Density Across Mosaics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Coverage plot
    ax2.bar(range(len(mosaic_names)), coverages, color='coral', edgecolor='black')
    ax2.set_xlabel('Mosaic Index', fontsize=12)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('Seafloor Coverage Across Mosaics', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.RESULTS_DIR / "summary_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved summary plots: {plot_path}")


def main():
    """Run inference and metrics pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run inference on full mosaics')
    parser.add_argument('--checkpoint', type=str, default='best', 
                       choices=['best', 'last'],
                       help='Which checkpoint to use for inference')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_dir=config.LOGS_DIR,
        log_level='INFO',
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info("="*80)
    logger.info("INFERENCE AND METRICS PIPELINE")
    logger.info("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE['use_gpu'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Creating model...")
    model = create_model(config.MODEL, device)
    
    # Load checkpoint
    checkpoint_path = config.CHECKPOINTS_DIR / f"{args.checkpoint}.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first by running 2_train.py")
        return
    
    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    # Get preprocessed mosaics
    mosaic_paths = list(config.PREPROCESSED_DIR.glob("*_preprocessed.png"))
    
    if len(mosaic_paths) == 0:
        logger.error(f"No preprocessed mosaics found in {config.PREPROCESSED_DIR}")
        logger.error("Please run 1_preprocess_and_label.py first")
        return
    
    logger.info(f"Found {len(mosaic_paths)} preprocessed mosaics")
    
    # Run inference
    inference_results = run_inference_on_mosaics(model, device, mosaic_paths)
    
    # Compute metrics
    aggregated_metrics, all_metrics = compute_metrics(inference_results)
    
    # Create summary plots
    if all_metrics:
        create_summary_plots(all_metrics)
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info(f"- Probability maps: {config.RESULTS_DIR / 'probability_maps'}")
    logger.info(f"- Binary masks: {config.RESULTS_DIR / 'binary_masks'}")
    logger.info(f"- Overlays: {config.RESULTS_DIR / 'overlays'}")
    logger.info(f"- Metrics: {config.RESULTS_DIR / 'metrics'}")
    logger.info(f"- Visualizations: {config.RESULTS_DIR / 'visualizations'}")


if __name__ == "__main__":
    main()
