#!/usr/bin/env python3
"""
Step 1: Preprocessing + Proxy Labelling Pipeline
-------------------------------------------------
CoralNet-inspired design:
  - Idempotent: already-processed files are skipped on re-runs
  - State-tracked: a JSON manifest records every mosaic's status
  - Step-isolated: each stage (preprocess / label / patch) can be re-run
    independently without restarting the whole pipeline
  - Graceful degradation: per-file errors are logged and skipped; a summary
    shows what succeeded/failed at the end

Usage:
    python 1_preprocess_and_label.py                 # full run
    python 1_preprocess_and_label.py --force         # re-run even if done
    python 1_preprocess_and_label.py --step preprocess
    python 1_preprocess_and_label.py --step label
    python 1_preprocess_and_label.py --step patch
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

import config
from preprocessing.preprocessing.preprocess import (
    MosaicPreprocessor, extract_patches, load_mosaic, save_mosaic
)
from preprocessing.preprocessing.proxy_labels import ProxyLabelGenerator, visualize_proxy_label
from utils.utils.logger import setup_logging

logger = logging.getLogger(__name__)

MANIFEST_PATH = config.OUTPUT_DIR / "pipeline_manifest.json"


# ---------------------------------------------------------------------------
# Manifest helpers  (CoralNet pattern: audit trail for every file)
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Load the pipeline state manifest, creating it if absent."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"version": "1.0", "mosaics": {}}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def mosaic_key(mosaic_path: Path) -> str:
    return mosaic_path.stem


# ---------------------------------------------------------------------------
# Step 1: Preprocess raw mosaics
# ---------------------------------------------------------------------------

def run_preprocessing(mosaic_files: list, manifest: dict, force: bool) -> list:
    logger.info("=" * 70)
    logger.info("STEP 1: PREPROCESSING RAW MOSAICS")
    logger.info("=" * 70)

    preprocessor = MosaicPreprocessor(config.PREPROCESSING)
    done, skipped, failed = [], [], []

    for mosaic_path in tqdm(mosaic_files, desc="Preprocessing"):
        key = mosaic_key(mosaic_path)
        entry = manifest["mosaics"].setdefault(key, {})
        out_path = config.PREPROCESSED_DIR / f"{key}_preprocessed.png"

        # Skip if already done and output exists
        if not force and entry.get("preprocessed") and out_path.exists():
            skipped.append(mosaic_path)
            continue

        try:
            mosaic = load_mosaic(mosaic_path)
            # Base preprocessing (steps 1-5) — used by proxy labeler to
            # preserve natural nodule-vs-sediment contrast.
            base = preprocessor.preprocess_base(mosaic)
            base_path = config.PREPROCESSED_DIR / f"{key}_preprocessed_base.png"
            save_mosaic(base, base_path)

            # Full preprocessing (base + nodule boost/sediment fade/unsharp)
            # — used for training patches. Pass base to skip steps 1-5 again.
            processed = preprocessor.preprocess_mosaic(mosaic, base=base)
            save_mosaic(processed, out_path)

            entry["preprocessed"] = str(out_path)
            entry["preprocessed_base"] = str(base_path)
            entry["preprocessed_at"] = datetime.now(timezone.utc).isoformat()
            entry["source"] = str(mosaic_path)
            entry["shape"] = list(processed.shape)
            save_manifest(manifest)
            done.append(out_path)

        except Exception as exc:
            logger.error(f"  FAILED {mosaic_path.name}: {exc}")
            entry["preprocess_error"] = str(exc)
            save_manifest(manifest)
            failed.append(mosaic_path)

    logger.info(
        f"Preprocessing: {len(done)} done, {len(skipped)} skipped, "
        f"{len(failed)} failed"
    )
    return done + [
        Path(manifest["mosaics"][mosaic_key(p)]["preprocessed"])
        for p in skipped
        if manifest["mosaics"].get(mosaic_key(p), {}).get("preprocessed")
    ]


# ---------------------------------------------------------------------------
# Step 2: Generate proxy labels
# ---------------------------------------------------------------------------

def run_proxy_labelling(preprocessed_paths: list, manifest: dict, force: bool) -> list:
    logger.info("=" * 70)
    logger.info("STEP 2: GENERATING PROXY LABELS")
    logger.info("=" * 70)

    # Validate label config before starting (CoralNet: fail fast)
    required_keys = [
        "tophat_radii", "tophat_percentile", "tophat_threshold_floor",
        "min_contour_area", "max_contour_area",
        "min_solidity", "min_circularity",
    ]
    missing = [k for k in required_keys if k not in config.PROXY_LABEL]
    if missing:
        logger.error(f"PROXY_LABEL config is missing keys: {missing}")
        logger.error("Fix config.py before continuing.")
        return []

    label_gen = ProxyLabelGenerator(config.PROXY_LABEL)
    debug_dir = str(config.OUTPUT_DIR / "debug_proxy")
    done, skipped, failed = [], [], []

    for pre_path in tqdm(preprocessed_paths, desc="Proxy labelling"):
        key = pre_path.stem.replace("_preprocessed", "")
        entry = manifest["mosaics"].setdefault(key, {})
        mask_path = config.PROXY_LABELS_DIR / f"{key}_mask.png"

        if not force and entry.get("proxy_label") and mask_path.exists():
            skipped.append(mask_path)
            continue

        try:
            # Use the base-preprocessed (lightly processed) image for proxy
            # labeling so that natural nodule-vs-sediment contrast is intact.
            # Fall back to the fully preprocessed image if base is missing
            # (e.g. for files processed before this change).
            base_path = Path(str(pre_path).replace("_preprocessed.png", "_preprocessed_base.png"))
            load_path = base_path if base_path.exists() else pre_path
            mosaic = cv2.imread(str(load_path), cv2.IMREAD_COLOR)
            if mosaic is None:
                raise ValueError(f"Could not read {load_path}")

            proxy_mask, stats = label_gen.generate_proxy_label(
                mosaic,
                debug_dir=debug_dir,
                debug_prefix=key,
            )

            # Reject degenerate masks (CoralNet: graceful degradation)
            coverage = stats["coverage_percent"]
            if coverage < 0.1:
                logger.warning(
                    f"  {key}: coverage={coverage:.2f}% — suspiciously sparse, "
                    f"check debug images in {debug_dir}"
                )
            elif coverage > 60.0:
                logger.warning(
                    f"  {key}: coverage={coverage:.2f}% — suspiciously dense, "
                    f"check debug images in {debug_dir}"
                )

            cv2.imwrite(str(mask_path), proxy_mask)

            if config.VISUALIZATION["save_proxy_labels"]:
                vis = visualize_proxy_label(
                    mosaic, proxy_mask, alpha=config.VISUALIZATION["overlay_alpha"]
                )
                vis_path = config.PROXY_LABELS_DIR / f"{key}_vis.png"
                cv2.imwrite(str(vis_path), vis)

            entry["proxy_label"] = str(mask_path)
            entry["proxy_stats"] = stats
            entry["proxy_label_at"] = datetime.now(timezone.utc).isoformat()
            save_manifest(manifest)
            done.append(mask_path)

        except Exception as exc:
            logger.error(f"  FAILED {pre_path.name}: {exc}")
            entry["proxy_label_error"] = str(exc)
            save_manifest(manifest)
            failed.append(pre_path)

    logger.info(
        f"Proxy labelling: {len(done)} done, {len(skipped)} skipped, "
        f"{len(failed)} failed"
    )
    return done + skipped


# ---------------------------------------------------------------------------
# Step 3: Extract patches
# ---------------------------------------------------------------------------

def run_patch_extraction(
    preprocessed_paths: list,
    proxy_mask_paths: list,
    manifest: dict,
    force: bool,
) -> list:
    logger.info("=" * 70)
    logger.info("STEP 3: EXTRACTING PATCHES")
    logger.info("=" * 70)

    patch_cfg = config.PREPROCESSING
    manual_dir = config.MANUAL_LABELS_DIR
    images_dir = config.PATCHES_DIR / "images"
    masks_dir = config.PATCHES_DIR / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    patch_manifest = []
    total_patches = 0
    skipped_mosaics = 0

    path_pairs = list(zip(preprocessed_paths, proxy_mask_paths))

    for pre_path, mask_path in tqdm(path_pairs, desc="Extracting patches"):
        key = pre_path.stem.replace("_preprocessed", "")
        entry = manifest["mosaics"].setdefault(key, {})

        # Skip if this mosaic's patches are already recorded and files exist
        if not force and entry.get("patches"):
            first = images_dir / f"{entry['patches'][0]}.png"
            if first.exists():
                patch_manifest.extend(entry.get("patch_records", []))
                skipped_mosaics += 1
                continue

        try:
            mosaic = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mosaic is None or mask is None:
                raise ValueError(f"Could not read {pre_path} or {mask_path}")

            img_patches, coords = extract_patches(
                mosaic,
                patch_h=patch_cfg["patch_height"],
                patch_w=patch_cfg["patch_width"],
                stride_v=patch_cfg["patch_stride_vertical"],
                stride_h=patch_cfg["patch_stride_horizontal"],
                min_std=patch_cfg["min_patch_std"],
                min_mean=patch_cfg["min_patch_mean"],
            )

            mosaic_patch_ids = []
            mosaic_records = []

            for i, (y, x) in enumerate(coords):
                ph = patch_cfg["patch_height"]
                pw = patch_cfg["patch_width"]
                patch_id = f"{key}_patch_{i:04d}"

                # Manual label overrides proxy (CoralNet pattern: explicit > inferred)
                manual_path = manual_dir / f"{patch_id}.png"
                if manual_path.exists():
                    mask_patch = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
                    label_source = "manual"
                else:
                    mask_patch = mask[y:y + ph, x:x + pw]
                    label_source = "proxy"

                img_patch_path = images_dir / f"{patch_id}.png"
                msk_patch_path = masks_dir / f"{patch_id}.png"
                cv2.imwrite(str(img_patch_path), img_patches[i])
                cv2.imwrite(str(msk_patch_path), mask_patch)

                record = {
                    "patch_id": patch_id,
                    "image_path": str(img_patch_path),
                    "mask_path": str(msk_patch_path),
                    "source_mosaic": pre_path.name,
                    "coordinates": [y, x],
                    "label_source": label_source,
                }
                mosaic_patch_ids.append(patch_id)
                mosaic_records.append(record)
                total_patches += 1

            entry["patches"] = mosaic_patch_ids
            entry["patch_records"] = mosaic_records
            entry["patches_at"] = datetime.now(timezone.utc).isoformat()
            patch_manifest.extend(mosaic_records)
            save_manifest(manifest)

        except Exception as exc:
            logger.error(f"  FAILED {pre_path.name}: {exc}")
            entry["patch_error"] = str(exc)
            save_manifest(manifest)

    # Write flat patch manifest for use by 2_train.py
    flat_manifest_path = config.PATCHES_DIR / "patch_manifest.json"
    with open(flat_manifest_path, "w") as f:
        json.dump(patch_manifest, f, indent=2)

    logger.info(
        f"Patch extraction: {total_patches} patches from "
        f"{len(path_pairs) - skipped_mosaics} mosaics "
        f"({skipped_mosaics} mosaics skipped, already done)"
    )
    logger.info(f"Manifest saved: {flat_manifest_path}")
    return patch_manifest


# ---------------------------------------------------------------------------
# Input discovery and validation
# ---------------------------------------------------------------------------

def find_raw_mosaics() -> list:
    exts = ["*.tif", "*.tiff", "*.png"]
    files = []
    for ext in exts:
        files.extend(config.RAW_MOSAICS_DIR.glob(ext))
    return sorted(files)


def validate_inputs(mosaic_files: list) -> bool:
    if not mosaic_files:
        logger.error(f"No mosaics found in {config.RAW_MOSAICS_DIR}")
        logger.error("Place .tif / .tiff / .png files there and re-run.")
        return False
    logger.info(f"Found {len(mosaic_files)} mosaic(s) to process")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocessing + proxy labelling pipeline")
    parser.add_argument(
        "--step",
        choices=["preprocess", "label", "patch"],
        default=None,
        help="Run only a single stage (default: all three)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if they were already done",
    )
    args = parser.parse_args()

    setup_logging(
        log_dir=config.LOGS_DIR,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True,
    )

    logger.info("=" * 70)
    logger.info("BOEM CV  —  Preprocessing + Proxy Labelling Pipeline")
    logger.info("=" * 70)
    logger.info(f"Raw mosaics : {config.RAW_MOSAICS_DIR}")
    logger.info(f"Output dir  : {config.OUTPUT_DIR}")
    logger.info(f"Force re-run: {args.force}")
    if args.step:
        logger.info(f"Running single step: {args.step}")

    manifest = load_manifest()
    mosaic_files = find_raw_mosaics()

    if not validate_inputs(mosaic_files):
        return

    run_all = args.step is None

    # ── Stage 1: Preprocess ──────────────────────────────────────────────────
    if run_all or args.step == "preprocess":
        preprocessed = run_preprocessing(mosaic_files, manifest, args.force)
    else:
        # Collect paths recorded in manifest for downstream stages
        preprocessed = [
            Path(v["preprocessed"])
            for v in manifest["mosaics"].values()
            if v.get("preprocessed") and Path(v["preprocessed"]).exists()
        ]
        logger.info(f"Skipping preprocess step — using {len(preprocessed)} cached files")

    if not preprocessed:
        logger.error("No preprocessed mosaics available. Exiting.")
        return

    # ── Stage 2: Proxy labels ────────────────────────────────────────────────
    if run_all or args.step == "label":
        proxy_masks = run_proxy_labelling(preprocessed, manifest, args.force)
    else:
        proxy_masks = [
            Path(v["proxy_label"])
            for v in manifest["mosaics"].values()
            if v.get("proxy_label") and Path(v["proxy_label"]).exists()
        ]
        logger.info(f"Skipping label step — using {len(proxy_masks)} cached masks")

    if not proxy_masks:
        logger.error("No proxy labels available. Exiting.")
        return

    # ── Stage 3: Patch extraction ────────────────────────────────────────────
    if run_all or args.step == "patch":
        # Pair preprocessed images with their proxy masks by stem
        mask_by_key = {
            p.stem.replace("_mask", ""): p for p in proxy_masks
        }
        paired_pre, paired_mask = [], []
        for pre in preprocessed:
            key = pre.stem.replace("_preprocessed", "")
            if key in mask_by_key:
                paired_pre.append(pre)
                paired_mask.append(mask_by_key[key])
            else:
                logger.warning(f"No proxy mask found for {pre.name}, skipping patches")

        patch_manifest = run_patch_extraction(
            paired_pre, paired_mask, manifest, args.force
        )
    else:
        patch_manifest = []
        for v in manifest["mosaics"].values():
            patch_manifest.extend(v.get("patch_records", []))
        logger.info(f"Skipping patch step — {len(patch_manifest)} patches in manifest")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Preprocessed mosaics : {len(preprocessed)}")
    logger.info(f"  Proxy label masks    : {len(proxy_masks)}")
    logger.info(f"  Patches in manifest  : {len(patch_manifest)}")

    manual_count = sum(1 for r in patch_manifest if r.get("label_source") == "manual")
    if manual_count:
        logger.info(
            f"  Manual label overrides: {manual_count} / {len(patch_manifest)} patches"
        )

    logger.info(f"\nPipeline manifest: {MANIFEST_PATH}")
    logger.info("Next step: python 2_train.py")


if __name__ == "__main__":
    main()
