"""
Generate qualitative comparison grids for the paper.

Two modes:
  1. Multi-model comparison grid:
     Rows = test images, Columns = Original | GT | Loss1 | Loss2 | ...
     Uses best seed per loss type from test_set_evaluation.json.

  2. Single-model error grid (legacy):
     Original | GT | Prediction | Error Overlay

Usage:
    # Multi-model grid (default — paper figure)
    python scripts/generate_qualitative.py

    # Thin-stroke only
    python scripts/generate_qualitative.py --subset thin

    # Single model error overlay
    python scripts/generate_qualitative.py --single-model \
        --model-dir ../SegmentationResearchPaper/experiments/loss_study/tversky_seed42
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"

sys.path.insert(0, str(REPO_ROOT / "compare_models"))
from compare_models import calculate_metrics

LOSS_ORDER = ["CE", "Focal", "Dice", "Dice+Focal", "Tversky"]
LOSS_KEY_MAP = {
    "ce": "CE", "focal": "Focal", "dice": "Dice",
    "dice_focal": "Dice+Focal", "tversky": "Tversky",
}

THIN_IDS = ["image_22", "image_24", "image_27", "image_33", "image_36"]
CORE_IDS = ["image_3", "image_13", "image_14", "image_15", "image_16", "image_17", "image_28"]


# ============================================================================
# Inference helpers
# ============================================================================

def load_model(model_path, device="cpu"):
    """Load a DeepLabV3-MobileNetV3 model."""
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_mask(model, img_path, img_size=(768, 1024), device="cpu"):
    """Run inference and return predicted mask at original resolution."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (W, H)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output['out']
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

    pred_mask = (pred * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return pred_mask


def find_best_seeds(test_eval_path):
    """From test_set_evaluation.json, find best seed per loss type.

    Returns dict: {loss_label: model_dir_name} e.g. {"Tversky": "loss_study/tversky_seed42"}
    """
    with open(test_eval_path) as f:
        data = json.load(f)

    best = {}
    for model_name, model_data in data["models"].items():
        if not model_name.startswith("loss_study/"):
            continue
        if "error" in model_data:
            continue
        cfg = model_data["config"]
        loss_label = LOSS_KEY_MAP.get(cfg["loss_type"], cfg["loss_type"])
        overall_f1 = model_data["results"]["grouped"]["overall"]["f1"]["mean"]

        if loss_label not in best or overall_f1 > best[loss_label]["f1"]:
            best[loss_label] = {"f1": overall_f1, "dir": model_name}

    return {k: v["dir"] for k, v in best.items()}


# ============================================================================
# Error overlay
# ============================================================================

def create_error_overlay(original, pred_mask, gt_mask):
    """Color-coded error overlay: Green=TP, Red=FP, Blue=FN."""
    h, w = gt_mask.shape[:2]
    if original.shape[:2] != (h, w):
        original = cv2.resize(original, (w, h))

    overlay = original.copy()
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    overlay[tp > 0] = [0, 200, 0]
    overlay[fp > 0] = [200, 0, 0]
    overlay[fn > 0] = [0, 0, 200]

    alpha = 0.5
    blended = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)
    return blended


# ============================================================================
# Multi-model comparison grid
# ============================================================================

def generate_multi_model_grid(image_ids, images_dir, masks_dir, experiments_dir,
                               best_seeds, output_path, img_size=(768, 1024)):
    """Generate grid: rows=images, cols=Original|GT|CE|Focal|Dice|D+F|Tversky.

    Each prediction cell shows F1 score below.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses_present = [l for l in LOSS_ORDER if l in best_seeds]
    n_cols = 2 + len(losses_present)  # Original, GT, + each loss
    n_rows = len(image_ids)

    if n_rows == 0:
        print("No images to process.")
        return

    # Pre-load all models
    print(f"Loading {len(losses_present)} models on {device}...")
    models = {}
    for loss_label in losses_present:
        model_dir = Path(experiments_dir) / best_seeds[loss_label]
        model_path = model_dir / "whiteboard_seg_best.pt"
        if model_path.exists():
            models[loss_label] = load_model(str(model_path), device)
            print(f"  {loss_label}: {best_seeds[loss_label]}")
        else:
            print(f"  {loss_label}: MODEL NOT FOUND at {model_path}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    col_titles = ["Original", "Ground Truth"] + losses_present
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight='bold')

    for i, img_id in enumerate(image_ids):
        # Find image file
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = Path(images_dir) / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"Image not found: {img_id}")
            continue

        mask_path = Path(masks_dir) / f"{img_id}.png"
        if not mask_path.exists():
            print(f"Mask not found: {mask_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(img_id.replace("image_", "#"), fontsize=10, fontweight='bold')

        # GT
        axes[i, 1].imshow(gt, cmap='gray')

        # Each loss function prediction
        for j, loss_label in enumerate(losses_present):
            col = j + 2
            if loss_label in models:
                pred = predict_mask(models[loss_label], img_path, img_size, device)
                h, w = gt.shape[:2]
                if pred.shape[:2] != (h, w):
                    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

                metrics = calculate_metrics(pred, gt)
                axes[i, col].imshow(pred, cmap='gray')
                axes[i, col].set_xlabel(f"F1={metrics['f1']:.3f}", fontsize=9, color='blue')
            else:
                axes[i, col].text(0.5, 0.5, "N/A", ha='center', va='center',
                                  transform=axes[i, col].transAxes, fontsize=14)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


# ============================================================================
# Single-model error grid (legacy)
# ============================================================================

def generate_error_grid(image_ids, images_dir, masks_dir, model_dir,
                        output_path, img_size=(768, 1024)):
    """Original | GT | Prediction | Error overlay for one model."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(model_dir) / "whiteboard_seg_best.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = load_model(str(model_path), device)
    n = len(image_ids)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0, 3].set_title('Error Overlay', fontsize=12, fontweight='bold')

    for i, img_id in enumerate(image_ids):
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = Path(images_dir) / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        mask_path = Path(masks_dir) / f"{img_id}.png"
        if not mask_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        pred = predict_mask(model, img_path, img_size, device)
        h, w = gt.shape[:2]
        img_rgb = cv2.resize(img_rgb, (w, h))
        if pred.shape[:2] != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        error = create_error_overlay(img_rgb, pred, gt)
        metrics = calculate_metrics(pred, gt)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(img_id.replace("image_", "#"), fontsize=10, fontweight='bold')
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_xlabel(f"F1={metrics['f1']:.3f}", fontsize=9, color='blue')
        axes[i, 3].imshow(error)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison grids")
    parser.add_argument("--single-model", action="store_true",
                       help="Single-model error overlay mode")
    parser.add_argument("--model-dir", type=str,
                       help="Model directory (for single-model mode)")
    parser.add_argument("--subset", choices=["thin", "core", "all"], default="all",
                       help="Which test images to include")
    parser.add_argument("--images-dir", type=str,
                       default=str(REPO_ROOT / "dataset" / "images"),
                       help="Path to test images")
    parser.add_argument("--masks-dir", type=str,
                       default=str(REPO_ROOT / "dataset" / "masks"),
                       help="Path to ground-truth masks")
    parser.add_argument("--experiments-dir", type=str,
                       default=str(PRIVATE_REPO / "experiments"),
                       help="Root experiments directory")
    parser.add_argument("--results-dir", type=str,
                       default=str(PRIVATE_REPO / "results"),
                       help="Results dir with test_set_evaluation.json")
    parser.add_argument("--output-dir", type=str,
                       default=str(PRIVATE_REPO / "results" / "qualitative"),
                       help="Where to save grid images")
    parser.add_argument("--img-height", type=int, default=768)
    parser.add_argument("--img-width", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_height, args.img_width)

    # Select images
    if args.subset == "thin":
        image_ids = THIN_IDS
    elif args.subset == "core":
        image_ids = CORE_IDS
    else:
        image_ids = CORE_IDS + THIN_IDS

    if args.single_model:
        if not args.model_dir:
            print("--model-dir required for single-model mode")
            return
        suffix = args.subset
        out_path = output_dir / f"error_grid_{suffix}.png"
        generate_error_grid(image_ids, args.images_dir, args.masks_dir,
                           args.model_dir, out_path, img_size)
    else:
        # Multi-model comparison
        test_eval_path = Path(args.results_dir) / "test_set_evaluation.json"
        if not test_eval_path.exists():
            print(f"test_set_evaluation.json not found: {test_eval_path}")
            return

        best_seeds = find_best_seeds(test_eval_path)
        print(f"Best seeds: {json.dumps(best_seeds, indent=2)}")

        suffix = args.subset
        out_path = output_dir / f"multi_model_comparison_{suffix}.png"
        generate_multi_model_grid(
            image_ids, args.images_dir, args.masks_dir,
            args.experiments_dir, best_seeds, out_path, img_size
        )


if __name__ == "__main__":
    main()
