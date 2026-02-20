"""
Evaluate All Trained Models on Test Set

Runs inference on held-out test images (test_core + test_thin) for every
trained model in the experiment directories. Produces per-model JSON results
with grouped metrics (overall, core, thin).

This script is independent of compare_models.py's pairwise comparison —
it evaluates each model in isolation.

Usage:
    python scripts/evaluate_all_models.py
    python scripts/evaluate_all_models.py --experiments-dir ../SegmentationResearchPaper/experiments
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"

# Reuse boundary_f1 and calculate_metrics from compare_models
sys.path.insert(0, str(REPO_ROOT / "compare_models"))
from compare_models import calculate_metrics, boundary_f1


def load_model(model_dir):
    """Load a trained model and its config from a directory."""
    model_path = model_dir / "whiteboard_seg_best.pt"
    config_path = model_dir / "training_history.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return model, config


def run_inference(model, image_path, img_size=(768, 1024), device="cpu"):
    """Run inference on a single image, return predicted mask at original resolution."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)

    img_resized = img.resize((img_size[1], img_size[0]))  # (W, H)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output["out"]
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

    pred_mask = (pred * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return pred_mask


def discover_models(experiments_dir):
    """Find all model directories that have whiteboard_seg_best.pt + training_history.json."""
    models = []
    experiments_dir = Path(experiments_dir)
    for pt_file in sorted(experiments_dir.rglob("whiteboard_seg_best.pt")):
        model_dir = pt_file.parent
        hist_file = model_dir / "training_history.json"
        if hist_file.exists():
            rel = model_dir.relative_to(experiments_dir)
            models.append({
                "dir": model_dir,
                "name": str(rel).replace("\\", "/"),
            })
    return models


def get_test_images(images_dir, masks_dir, split_config_path):
    """Get test image paths filtered by test split config (both = core + thin)."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    with open(split_config_path) as f:
        splits = json.load(f)

    allowed_ids = set(splits.get("test_core", [])) | set(splits.get("test_thin", []))
    core_ids = set(splits.get("test_core", []))
    thin_ids = set(splits.get("test_thin", []))

    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    test_images = []
    for p in all_images:
        m = re.match(r"^(image_\d+)", p.stem)
        base_id = m.group(1) if m else p.stem
        if base_id in allowed_ids:
            # Only use original images, not augmented variants
            if p.stem == base_id:
                mask_path = masks_dir / f"{base_id}.png"
                if mask_path.exists():
                    group = []
                    if base_id in core_ids:
                        group.append("core")
                    if base_id in thin_ids:
                        group.append("thin")
                    test_images.append({
                        "image_path": p,
                        "mask_path": mask_path,
                        "base_id": base_id,
                        "groups": group,
                    })

    return test_images, splits


def evaluate_model_on_test_set(model, config, test_images, device="cpu"):
    """Evaluate a single model on the test set, return per-image and grouped metrics."""
    cfg = config.get("config", {})
    img_height = cfg.get("img_height", 768)
    img_width = cfg.get("img_width", 1024)
    img_size = (img_height, img_width)

    model = model.to(device)
    per_image = []

    for item in test_images:
        pred_mask = run_inference(model, item["image_path"], img_size=img_size, device=device)
        gt_mask = np.array(Image.open(item["mask_path"]).convert("L"))
        metrics = calculate_metrics(pred_mask, gt_mask)

        per_image.append({
            "image": item["base_id"],
            "groups": item["groups"],
            "metrics": metrics,
        })

    # Group aggregation
    def aggregate(results):
        if not results:
            return {}
        keys = ["iou", "f1", "precision", "recall", "pixel_acc", "dice",
                "edge_iou", "boundary_f1"]
        agg = {"count": len(results)}
        for k in keys:
            vals = [r["metrics"][k] for r in results if k in r["metrics"]]
            if vals:
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        return agg

    overall = aggregate(per_image)
    core = aggregate([r for r in per_image if "core" in r["groups"]])
    thin = aggregate([r for r in per_image if "thin" in r["groups"]])

    return {
        "per_image": per_image,
        "grouped": {
            "overall": overall,
            "core": core,
            "thin": thin,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models on held-out test set")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(PRIVATE_REPO / "experiments"),
                        help="Root experiments directory")
    parser.add_argument("--images-dir", type=str,
                        default=str(REPO_ROOT / "dataset" / "images"),
                        help="Path to dataset images")
    parser.add_argument("--masks-dir", type=str,
                        default=str(REPO_ROOT / "dataset" / "masks"),
                        help="Path to dataset masks")
    parser.add_argument("--test-split-config", type=str,
                        default=str(PRIVATE_REPO / "configs" / "test_splits.json"),
                        help="Path to test_splits.json")
    parser.add_argument("--output-dir", type=str,
                        default=str(PRIVATE_REPO / "results"),
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda or cpu). Auto-detects if not set.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover all trained models
    models_info = discover_models(args.experiments_dir)
    print(f"Found {len(models_info)} trained models\n")

    # Get test images
    test_images, splits = get_test_images(args.images_dir, args.masks_dir, args.test_split_config)
    print(f"Test set: {len(test_images)} images")
    for t in test_images:
        print(f"  {t['base_id']} — groups: {t['groups']}")
    print()

    # Evaluate each model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total = len(models_info)

    for i, minfo in enumerate(models_info):
        name = minfo["name"]
        print("=" * 70)
        print(f"[{i+1}/{total}] Evaluating: {name}")
        print("=" * 70)

        try:
            model, config = load_model(minfo["dir"])
            cfg = config.get("config", {})
            print(f"  Loss: {cfg.get('loss_type', '?')}, Seed: {cfg.get('seed', '?')}, "
                  f"Resolution: {cfg.get('img_resolution', '?')}")

            t0 = time.time()
            result = evaluate_model_on_test_set(model, config, test_images, device=device)
            elapsed = time.time() - t0

            # Print summary
            g = result["grouped"]
            print(f"  Overall: F1={g['overall']['f1']['mean']:.4f}, "
                  f"IoU={g['overall']['iou']['mean']:.4f}, "
                  f"BF1={g['overall']['boundary_f1']['mean']:.4f}")
            if g.get("core"):
                print(f"  Core:    F1={g['core']['f1']['mean']:.4f}, "
                      f"IoU={g['core']['iou']['mean']:.4f}, "
                      f"BF1={g['core']['boundary_f1']['mean']:.4f}")
            if g.get("thin"):
                print(f"  Thin:    F1={g['thin']['f1']['mean']:.4f}, "
                      f"IoU={g['thin']['iou']['mean']:.4f}, "
                      f"BF1={g['thin']['boundary_f1']['mean']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[name] = {
                "model_dir": str(minfo["dir"]),
                "config": {
                    "loss_type": cfg.get("loss_type"),
                    "seed": cfg.get("seed"),
                    "img_resolution": cfg.get("img_resolution"),
                    "loss_function": cfg.get("loss_function"),
                    "img_height": cfg.get("img_height"),
                    "img_width": cfg.get("img_width"),
                },
                "eval_time_seconds": round(elapsed, 2),
                "results": result,
            }
            print()

            # Free GPU memory
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}
            print()

    # Save master results file
    master_output = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "test_images": [t["base_id"] for t in test_images],
        "test_split_config": str(args.test_split_config),
        "num_models": total,
        "num_test_images": len(test_images),
        "models": all_results,
    }

    output_path = output_dir / "test_set_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(master_output, f, indent=2)
    print(f"\nMaster results saved to: {output_path}")

    # Print final summary table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"{'Model':<35} {'F1':>6} {'IoU':>6} {'BF1':>6} | {'Core F1':>8} {'Thin F1':>8}")
    print("-" * 90)

    for name in sorted(all_results.keys()):
        r = all_results[name]
        if "error" in r:
            print(f"{name:<35} ERROR: {r['error']}")
            continue
        g = r["results"]["grouped"]
        overall_f1 = g["overall"]["f1"]["mean"]
        overall_iou = g["overall"]["iou"]["mean"]
        overall_bf1 = g["overall"]["boundary_f1"]["mean"]
        core_f1 = g["core"]["f1"]["mean"] if g.get("core") else float("nan")
        thin_f1 = g["thin"]["f1"]["mean"] if g.get("thin") else float("nan")
        print(f"{name:<35} {overall_f1:>6.4f} {overall_iou:>6.4f} {overall_bf1:>6.4f} | "
              f"{core_f1:>8.4f} {thin_f1:>8.4f}")

    print("=" * 90)


if __name__ == "__main__":
    main()
