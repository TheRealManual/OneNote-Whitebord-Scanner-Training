"""
Generate dataset manifest with SHA256 hashes and split assignments.

Produces a JSON manifest listing every image+mask pair with:
  - filename, SHA256 hash (image and mask)
  - split assignment (train, val, test_core, test_thin)
  - image dimensions

This supports reproducibility even if the dataset cannot be shared publicly.

Usage:
    python scripts/generate_dataset_manifest.py
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest")
    parser.add_argument("--images-dir", type=str,
                       default=str(REPO_ROOT / "dataset" / "images"))
    parser.add_argument("--masks-dir", type=str,
                       default=str(REPO_ROOT / "dataset" / "masks"))
    parser.add_argument("--test-split-config", type=str,
                       default=str(PRIVATE_REPO / "configs" / "test_splits.json"))
    parser.add_argument("--output-json", type=str,
                       default=str(PRIVATE_REPO / "configs" / "dataset_manifest.json"))
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    # Load test-split config
    with open(args.test_split_config) as f:
        splits = json.load(f)
    core_ids = set(splits.get("test_core", []))
    thin_ids = set(splits.get("test_thin", []))
    exclude_ids = set(splits.get("train_exclude", []))

    # Collect all image files
    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png")) +
        list(images_dir.glob("*.jpeg"))
    )

    entries = []
    originals = 0
    augmented = 0

    for img_path in image_files:
        stem = img_path.stem
        # Extract base ID
        m = re.match(r'^(image_\d+)', stem)
        base_id = m.group(1) if m else stem
        is_augmented = stem != base_id

        if is_augmented:
            augmented += 1
        else:
            originals += 1

        # Find mask
        mask_path = masks_dir / f"{base_id}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{stem}.png"

        # Determine split
        if base_id in core_ids:
            split = "test_core"
        elif base_id in thin_ids:
            split = "test_thin"
        elif base_id in exclude_ids:
            split = "excluded"
        else:
            split = "train"  # Will be subdivided into train/val by the 80/20 split

        # Image dimensions
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] if img is not None else (0, 0)

        entry = {
            "image_file": img_path.name,
            "base_id": base_id,
            "is_augmented": is_augmented,
            "split": split,
            "image_sha256": sha256_file(img_path),
            "image_height": h,
            "image_width": w,
        }

        if mask_path.exists():
            entry["mask_file"] = mask_path.name
            entry["mask_sha256"] = sha256_file(mask_path)
        else:
            entry["mask_file"] = None
            entry["mask_sha256"] = None

        entries.append(entry)

    # Summary
    split_counts = {}
    for e in entries:
        s = e["split"]
        split_counts[s] = split_counts.get(s, 0) + 1

    manifest = {
        "description": "Dataset manifest for reproducibility. "
                       "SHA256 hashes allow verification of exact image/mask files used.",
        "generated_by": "scripts/generate_dataset_manifest.py",
        "total_files": len(entries),
        "original_images": originals,
        "augmented_images": augmented,
        "split_counts": split_counts,
        "test_splits": {
            "test_core": sorted(core_ids),
            "test_thin": sorted(thin_ids),
        },
        "train_val_split": "Alphabetical sort → first 80% train, last 20% val. "
                           "Applied after excluding test IDs and their augmented variants.",
        "entries": entries,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset manifest: {len(entries)} files ({originals} original, {augmented} augmented)")
    print(f"Splits: {split_counts}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
