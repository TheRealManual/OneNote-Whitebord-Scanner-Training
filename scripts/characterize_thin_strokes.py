"""
Characterize thin-stroke images with measurable metrics.

Computes per-image statistics to objectively define which images
qualify as "thin-stroke" vs "core":
  - stroke_pixel_fraction: fraction of mask pixels that are foreground
  - mean_stroke_width: average skeleton branch width
  - median_stroke_width: median skeleton branch width
  - skeleton_length: total skeleton length in pixels
  - aspect_ratio: skeleton_length / mean_stroke_width (higher = thinner)

Usage:
    python scripts/characterize_thin_strokes.py
    python scripts/characterize_thin_strokes.py --output-json ../SegmentationResearchPaper/results/thin_stroke_characterization.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"


def compute_stroke_stats(mask_path):
    """Compute stroke-width and coverage statistics from a binary mask.

    Returns dict with stroke_pixel_fraction, mean_stroke_width,
    median_stroke_width, skeleton_length, aspect_ratio.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    binary = (mask > 127).astype(np.uint8)
    total_pixels = binary.size
    stroke_pixels = int(binary.sum())
    stroke_pixel_fraction = stroke_pixels / total_pixels if total_pixels > 0 else 0

    if stroke_pixels == 0:
        return {
            "stroke_pixel_fraction": 0.0,
            "stroke_pixels": 0,
            "total_pixels": total_pixels,
            "mean_stroke_width": 0.0,
            "median_stroke_width": 0.0,
            "skeleton_length": 0,
            "aspect_ratio": 0.0,
        }

    # Compute distance transform to find stroke width at each point
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Skeletonize using morphological thinning
    skeleton = cv2.ximgproc.thinning(binary * 255) if hasattr(cv2, 'ximgproc') else _simple_skeleton(binary)
    skel_binary = (skeleton > 127).astype(np.uint8) if skeleton.max() > 1 else skeleton

    skeleton_length = int(skel_binary.sum())

    if skeleton_length == 0:
        return {
            "stroke_pixel_fraction": float(stroke_pixel_fraction),
            "stroke_pixels": stroke_pixels,
            "total_pixels": total_pixels,
            "mean_stroke_width": 0.0,
            "median_stroke_width": 0.0,
            "skeleton_length": 0,
            "aspect_ratio": 0.0,
        }

    # Width at each skeleton pixel = 2 * distance_transform value
    widths = dist_transform[skel_binary > 0] * 2.0
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    # Aspect ratio: skeleton length / mean width (higher = thinner strokes)
    aspect_ratio = skeleton_length / mean_width if mean_width > 0 else 0.0

    return {
        "stroke_pixel_fraction": float(stroke_pixel_fraction),
        "stroke_pixels": stroke_pixels,
        "total_pixels": total_pixels,
        "mean_stroke_width": round(mean_width, 2),
        "median_stroke_width": round(median_width, 2),
        "skeleton_length": skeleton_length,
        "aspect_ratio": round(aspect_ratio, 1),
    }


def _simple_skeleton(binary):
    """Fallback skeletonization if cv2.ximgproc is not available."""
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy() * 255

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        diff = img - opened
        skel = cv2.bitwise_or(skel, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def main():
    parser = argparse.ArgumentParser(description="Characterize thin-stroke images")
    parser.add_argument("--masks-dir", type=str,
                       default=str(REPO_ROOT / "dataset" / "masks"),
                       help="Path to ground-truth masks")
    parser.add_argument("--output-json", type=str,
                       default=str(PRIVATE_REPO / "results" / "thin_stroke_characterization.json"),
                       help="Where to save characterization JSON")
    parser.add_argument("--test-split-config", type=str,
                       default=str(PRIVATE_REPO / "configs" / "test_splits.json"),
                       help="Test split config for labeling core/thin")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)

    # Load test-split config for labeling
    with open(args.test_split_config) as f:
        splits = json.load(f)
    core_ids = set(splits.get("test_core", []))
    thin_ids = set(splits.get("test_thin", []))
    all_test_ids = core_ids | thin_ids

    # Find all original mask files (no augmented)
    mask_files = sorted(masks_dir.glob("image_*.png"))
    # Filter to originals only (no _aug suffix)
    import re
    originals = [p for p in mask_files if re.match(r'^image_\d+$', p.stem)]

    results = []
    for mask_path in originals:
        img_id = mask_path.stem
        stats = compute_stroke_stats(mask_path)
        if stats is None:
            continue

        # Label group
        if img_id in thin_ids:
            group = "thin"
        elif img_id in core_ids:
            group = "core"
        elif img_id in all_test_ids:
            group = "test"
        else:
            group = "train"

        entry = {"image_id": img_id, "group": group}
        entry.update(stats)
        results.append(entry)

    # Sort by stroke_pixel_fraction
    results.sort(key=lambda x: x["stroke_pixel_fraction"])

    # Summary stats
    thin_stats = [r for r in results if r["group"] == "thin"]
    core_stats = [r for r in results if r["group"] == "core"]
    train_stats = [r for r in results if r["group"] == "train"]
    all_stats = results

    def summarize(subset):
        if not subset:
            return {}
        fracs = [r["stroke_pixel_fraction"] for r in subset]
        widths = [r["mean_stroke_width"] for r in subset if r["mean_stroke_width"] > 0]
        return {
            "count": len(subset),
            "stroke_fraction_mean": round(float(np.mean(fracs)), 6),
            "stroke_fraction_std": round(float(np.std(fracs)), 6),
            "stroke_fraction_min": round(float(np.min(fracs)), 6),
            "stroke_fraction_max": round(float(np.max(fracs)), 6),
            "mean_stroke_width_mean": round(float(np.mean(widths)), 2) if widths else 0,
            "mean_stroke_width_std": round(float(np.std(widths)), 2) if widths else 0,
        }

    output = {
        "description": "Per-image stroke characterization for thin-stroke definition",
        "methodology": "Stroke pixel fraction = foreground/total. "
                       "Stroke width = 2 * distance_transform at skeleton pixels. "
                       "Skeleton via morphological thinning.",
        "summary": {
            "thin": summarize(thin_stats),
            "core": summarize(core_stats),
            "train": summarize(train_stats),
            "all": summarize(all_stats),
        },
        "per_image": results,
    }

    # Print table
    print(f"\n{'Image ID':<15} {'Group':<7} {'Stroke%':>8} {'Width':>7} {'SkelLen':>8} {'Aspect':>8}")
    print("-" * 60)
    for r in results:
        frac_pct = r["stroke_pixel_fraction"] * 100
        print(f"{r['image_id']:<15} {r['group']:<7} {frac_pct:>7.3f}% {r['mean_stroke_width']:>6.1f}px "
              f"{r['skeleton_length']:>7d} {r['aspect_ratio']:>8.1f}")

    print(f"\nSummary:")
    for grp_name in ["thin", "core", "train", "all"]:
        s = output["summary"][grp_name]
        if s:
            print(f"  {grp_name}: stroke fraction = {s['stroke_fraction_mean']*100:.3f}% ± {s['stroke_fraction_std']*100:.3f}%, "
                  f"mean width = {s['mean_stroke_width_mean']:.1f}px ± {s['mean_stroke_width_std']:.1f}px (n={s['count']})")

    # Save
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
