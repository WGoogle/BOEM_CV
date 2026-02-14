# Polymetallic Nodule Segmentation Pipeline

Automated deep-learning segmentation and density analysis of polymetallic nodules in deep-sea seafloor imagery using weakly supervised U-Net with ResNet34 encoder.

## 📋 Overview

This pipeline implements the methodology from the BOEM Symposium poster:

1. **Preprocessing**: Color normalization, CLAHE enhancement, bilateral filtering
2. **Proxy Label Generation**: Weakly supervised labels via Otsu thresholding + morphological operations
3. **U-Net Training**: ResNet34 encoder with hybrid BCE + Dice loss
4. **Full-Mosaic Inference**: Sliding window prediction with probability map blending
5. **Metrics Computation**: Nodule count, density (nodules/m²), percent coverage, size distribution

## 🚀 Quick Start

### Installation

```bash
# Clone or download this directory
cd nodule_segmentation

# Install dependencies
pip install -r requirements.txt
```

### Directory Setup

```
nodule_segmentation/
├── data/
│   └── raw_mosaics/          # PUT YOUR .TIF/.PNG MOSAICS HERE
├── outputs/                   # All outputs go here
│   ├── preprocessed/
│   ├── proxy_labels/
│   ├── patches/
│   ├── checkpoints/
│   ├── results/
│   └── logs/
├── config.py                  # Configuration file (EDIT THIS)
├── 1_preprocess_and_label.py # Step 1: Preprocessing
├── 2_train.py                 # Step 2: Training
└── 3_inference.py             # Step 3: Inference & Metrics
```

### Usage

**Step 1: Preprocessing and Proxy Label Generation**
```bash
python 1_preprocess_and_label.py
```
- Preprocesses raw mosaics (CLAHE, bilateral filtering, white balance)
- Generates proxy labels using classical CV (Otsu + morphology)
- Extracts patches for training

**Step 2: Training**
```bash
python 2_train.py
```
- Trains U-Net with ResNet34 encoder
- Uses hybrid BCE + Dice loss
- Implements early stopping and learning rate scheduling
- Saves best and last checkpoints

**Step 3: Inference and Metrics**
```bash
python 3_inference.py --checkpoint best
```
- Runs sliding window inference on full mosaics
- Computes probability maps and binary masks
- Calculates nodule density metrics
- Generates summary visualizations and plots

## ⚙️ Configuration

Edit `config.py` to customize:

### Critical Parameters to Adjust

1. **Pixel-to-meter conversion** (line ~95):
```python
'meters_per_pixel': 0.005,  # ADJUST THIS based on your AUV/ROV specs!
```

2. **Data path**:
```python
RAW_MOSAICS_DIR = DATA_DIR / "raw_mosaics"  # Your .tif files go here
```

3. **Training hyperparameters**:
```python
TRAINING = {
    'batch_size': 16,          # Reduce if out of memory
    'num_epochs': 100,
    'learning_rate': 1e-4,
    ...
}
```

4. **Patch size and stride**:
```python
PREPROCESSING = {
    'patch_height': 192,
    'patch_width': 256,
    'patch_stride_vertical': 128,
    'patch_stride_horizontal': 128,
    ...
}
```

## 📊 Outputs

### After Step 1 (Preprocessing)
- `outputs/preprocessed/` - Enhanced mosaics
- `outputs/proxy_labels/` - Binary masks and visualizations
- `outputs/patches/` - Extracted training patches

### After Step 2 (Training)
- `outputs/checkpoints/best.pth` - Best model checkpoint
- `outputs/checkpoints/training_history.json` - Training metrics
- `outputs/logs/` - Training logs

### After Step 3 (Inference)
- `outputs/results/probability_maps/` - Probability heatmaps
- `outputs/results/binary_masks/` - Final segmentations
- `outputs/results/overlays/` - Visualizations with overlays
- `outputs/results/metrics/` - Per-mosaic JSON metrics
- `outputs/results/aggregated_metrics.json` - Dataset-level summary
- `outputs/results/visualizations/` - Summary images with annotations
- `outputs/results/summary_plots.png` - Density and coverage bar charts

## 📈 Metrics Computed

For each mosaic:
- **Total nodules**: Connected components count
- **Nodules per m²**: Density metric
- **Percent coverage**: % of seafloor covered by nodules
- **Size distribution**: Histogram of nodule sizes
- **Area statistics**: Mean, median, std, min, max

Aggregated across dataset:
- Average density
- Average coverage
- Total nodules detected
- Total area surveyed

## 🔧 Troubleshooting

### Out of Memory Error
Reduce batch size in `config.py`:
```python
TRAINING = {
    'batch_size': 8,  # or 4
    ...
}
```

### No mosaics found
Ensure your `.tif` or `.png` files are in:
```
data/raw_mosaics/
```

### CUDA not available
The pipeline will automatically fall back to CPU. To force CPU:
```python
DEVICE = {
    'use_gpu': False,
    ...
}
```

### Proxy labels look poor
Adjust thresholding parameters in `config.py`:
```python
PROXY_LABEL = {
    'min_contour_area': 20,    # Smaller = more nodules
    'max_contour_area': 2000,  # Larger = bigger nodules
    'max_eccentricity': 0.95,  # Lower = more circular
    ...
}
```

## 📚 Model Architecture

**U-Net** with **ResNet34 encoder** (pretrained on ImageNet):
- Encoder: ResNet34 backbone (extracts features)
- Decoder: U-Net upsampling path (reconstructs segmentation)
- Loss: Hybrid BCE + Dice (handles class imbalance)
- Augmentation: Brightness/contrast, blur, rotation, flips, elastic transforms

## 🎯 Key Assumptions

1. **Pixel-to-meter ratio**: Default 0.005 m/px (5mm/pixel). **MUST BE ADJUSTED** for your specific imagery.

2. **Nodule characteristics**:
   - Size: 20-2000 pixels
   - Shape: Relatively circular (eccentricity < 0.95)
   - Appearance: Brighter than background sediment

3. **Image format**: Expects BGR color images from underwater cameras

## 📄 Citation

If you use this pipeline, please cite:

```
Manasvi Lodha, Inez Alvarez, Hitha Varganti, Vihaan Khandelwal
"Automated Deep-Learning Segmentation and Density Analysis of 
Polymetallic Nodules in Deep-Sea Seafloor Imagery"
BOEM Symposium, 2024
```

## 🤝 Acknowledgments

Bureau of Ocean Energy Management (BOEM):
- Kimberly E. Baldwin
- Erick Huchzermeyer  
- Kevin Smith
- Obediah Racicot

## 📧 Contact

For questions or issues, contact:
- Manasvi Lodha: manasvi.lodha@berkeley.edu

---

**Note**: This is a research pipeline. Validate results on your specific dataset before operational use.
