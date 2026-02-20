"""
Classical Baseline for Whiteboard Segmentation

Non-learning baselines to establish a lower-bound reference:
  1. Adaptive thresholding (Gaussian, blockSize=51, C=15)
  2. Otsu thresholding (global automatic threshold)

Both include morphological opening cleanup (2×2 kernel).

Usage:
    python scripts/classical_baseline.py --images-dir dataset/images --masks-dir dataset/masks
    python scripts/classical_baseline.py --test-split-config ../SegmentationResearchPaper/configs/test_splits.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"

# Import calculate_metrics from compare_models
sys.path.insert(0, str(REPO_ROOT / "compare_models"))
from compare_models import calculate_metrics


def adaptive_threshold(image_gray, block_size=51, C=15):
    """Apply adaptive Gaussian thresholding.

    Args:
        image_gray: uint8 grayscale image.
        block_size: Size of the neighborhood area (must be odd).
        C: Constant subtracted from the weighted mean.

    Returns:
        Binary mask (uint8, 0 or 255), strokes = 255.
    """
    # Adaptive threshold — dark strokes on light background
    # THRESH_BINARY_INV: pixels darker than local mean - C become 255 (stroke)
    mask = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )
    return mask


def otsu_threshold(image_gray):
    """Apply Otsu's global thresholding.

    Returns:
        Binary mask (uint8, 0 or 255), strokes = 255.
    """
    _, mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def morphological_cleanup(mask, kernel_size=2):
    """Apply morphological opening to remove small noise.

    Args:
        mask: Binary mask (uint8).
        kernel_size: Size of erosion/dilation kernel.

    Returns:
        Cleaned binary mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cleaned


def run_baseline(images_dir, masks_dir, method, img_height=768, img_width=1024,
                 test_split_config=None, test_split=None):
    """Run a classical baseline on a set of images.

    Args:
        images_dir: Path to images directory.
        masks_dir: Path to ground-truth masks directory.
        method: 'adaptive' or 'otsu'.
        img_height, img_width: Resolution for evaluation.
        test_split_config: Optional path to test_splits.json.
        test_split: 'core', 'thin', or 'both' — filter images by group.

    Returns:
        dict with method name, per-image results, and aggregated metrics.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Collect image files
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    # Filter by test-split
    if test_split_config and test_split:
        with open(test_split_config) as f:
            splits = json.load(f)
        allowed_ids = set()
        if test_split in ('core', 'both'):
            allowed_ids.update(splits.get('test_core', []))
        if test_split in ('thin', 'both'):
            allowed_ids.update(splits.get('test_thin', []))
        filtered = []
        for p in image_files:
            m = re.match(r'^(image_\d+)', p.stem)
            base_id = m.group(1) if m else p.stem
            # Only use original images (exact match), not augmented variants
            if base_id in allowed_ids and p.stem == base_id:
                filtered.append(p)
        image_files = filtered

    per_image = []
    for img_path in image_files:
        # Find matching mask
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            # Try without augment suffix
            m = re.match(r'^(image_\d+)', img_path.stem)
            if m:
                mask_path = masks_dir / f"{m.group(1)}.png"
        if not mask_path.exists():
            continue

        # Load and resize
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (img_width, img_height))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # Apply thresholding
        if method == 'adaptive':
            pred_mask = adaptive_threshold(gray)
        elif method == 'otsu':
            pred_mask = otsu_threshold(gray)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cleanup
        pred_mask = morphological_cleanup(pred_mask)

        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_resized)
        per_image.append({
            'image': img_path.name,
            'metrics': metrics,
        })

    # Aggregate
    if per_image:
        metric_keys = ['iou', 'f1', 'precision', 'recall', 'pixel_acc', 'dice',
                        'edge_iou', 'boundary_f1']
        aggregate = {}
        for k in metric_keys:
            values = [r['metrics'][k] for r in per_image if k in r['metrics']]
            if values:
                aggregate[k] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
    else:
        aggregate = {}

    return {
        'method': method,
        'timestamp': datetime.now().isoformat(),
        'resolution': f'{img_width}x{img_height}',
        'num_images': len(per_image),
        'results': aggregate,
        'per_image': per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Classical baseline for whiteboard segmentation")
    parser.add_argument("--images-dir", type=str, default=str(REPO_ROOT / "dataset" / "images"),
                       help="Path to images directory")
    parser.add_argument("--masks-dir", type=str, default=str(REPO_ROOT / "dataset" / "masks"),
                       help="Path to ground-truth masks directory")
    parser.add_argument("--output-dir", type=str,
                       default=str(PRIVATE_REPO / "results"),
                       help="Output directory for results JSON")
    parser.add_argument("--img-height", type=int, default=768,
                       help="Image height for evaluation")
    parser.add_argument("--img-width", type=int, default=1024,
                       help="Image width for evaluation")
    parser.add_argument("--test-split-config", type=str, default=None,
                       help="Path to test_splits.json for grouped evaluation")
    parser.add_argument("--test-split", type=str, default=None,
                       choices=["core", "thin", "both"],
                       help="Filter images by split group")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for method in ('adaptive', 'otsu'):
        print(f"\nRunning {method} thresholding baseline...")
        result = run_baseline(
            args.images_dir, args.masks_dir, method,
            img_height=args.img_height, img_width=args.img_width,
            test_split_config=args.test_split_config,
            test_split=args.test_split,
        )
        all_results[method] = result

        if result['results']:
            r = result['results']
            print(f"  {method}: IoU={r['iou']['mean']:.4f} (+-{r['iou']['std']:.4f}), "
                  f"F1={r['f1']['mean']:.4f} (+-{r['f1']['std']:.4f}), "
                  f"BF1={r['boundary_f1']['mean']:.4f}")
        else:
            print(f"  {method}: No images processed")

    # Save results
    output_path = output_dir / "classical_baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
