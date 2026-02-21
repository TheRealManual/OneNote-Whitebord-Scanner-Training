"""
Generate baseline-vs-deep failure comparison figure and scatter plot.

Produces two figures:
  1. baseline_vs_deep_failure.pdf — Grid showing images where adaptive
     baseline fails most: Original | GT | Adaptive | Best Deep | Error (Adaptive) | Error (Deep)
  2. baseline_vs_deep_scatter.pdf — Scatter: x=Adaptive F1, y=Tversky F1, diagonal=parity

Usage:
    python scripts/generate_baseline_comparison.py
"""

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGES_DIR = REPO_ROOT / "dataset" / "images"
MASKS_DIR = REPO_ROOT / "dataset" / "masks"
EXPERIMENTS_DIR = PRIVATE_REPO / "experiments"
RESULTS_DIR = PRIVATE_REPO / "results"
FIGURES_DIR = PRIVATE_REPO / "results" / "figures"

# Images where adaptive baseline is worst (sorted by adaptive F1 ascending)
FAILURE_IDS = ["image_3", "image_14", "image_13"]

THIN_IDS = {"image_22", "image_24", "image_27", "image_33", "image_36"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adaptive_threshold(image_gray, block_size=51, C=15):
    """Replicate the classical baseline's adaptive thresholding."""
    mask = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )
    return mask


def morphological_cleanup(mask, kernel_size=2):
    """Morphological opening to remove small noise (must match classical_baseline.py)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def load_model(model_path, device="cpu"):
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_mask(model, img_path, img_size=(768, 1024), device="cpu"):
    """Run inference matching evaluate_all_models.run_inference exactly."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (W, H)

    # Resize via PIL first, then transform — matches evaluate_all_models.py
    img_resized = img.resize((img_size[1], img_size[0]))  # PIL: (W, H)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output['out']
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

    pred_mask = (pred * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return pred_mask


def create_error_overlay(original, pred_mask, gt_mask):
    """Color-coded error: Green=TP, Red=FN, Blue=FP."""
    h, w = gt_mask.shape[:2]
    if original.shape[:2] != (h, w):
        original = cv2.resize(original, (w, h))

    overlay = original.copy()
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    overlay[tp > 0] = [0, 200, 0]    # green = correct
    overlay[fp > 0] = [0, 0, 200]    # blue = false positive
    overlay[fn > 0] = [200, 0, 0]    # red = missed stroke

    alpha = 0.5
    blended = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)
    return blended


def find_best_deep_model():
    """Find the best-seed model for Tversky (highest overall F1).
    Returns (model_dir_name, img_height, img_width)."""
    test_eval_path = RESULTS_DIR / "test_set_evaluation.json"
    with open(test_eval_path) as f:
        data = json.load(f)

    best_f1 = -1
    best_dir = None
    best_cfg = None
    for name, mdata in data["models"].items():
        if "error" in mdata:
            continue
        if not name.startswith("loss_study/"):
            continue
        cfg = mdata["config"]
        if cfg["loss_type"] != "tversky":
            continue
        f1 = mdata["results"]["grouped"]["overall"]["f1"]["mean"]
        if f1 > best_f1:
            best_f1 = f1
            best_dir = name
            best_cfg = cfg
    img_h = best_cfg.get("img_height", 768) if best_cfg else 768
    img_w = best_cfg.get("img_width", 1024) if best_cfg else 1024
    return best_dir, img_h, img_w


# ---------------------------------------------------------------------------
# Figure 1: Baseline failure grid
# ---------------------------------------------------------------------------

def generate_failure_grid():
    """Grid: rows = failure images, cols = Original | GT | Adaptive | Tversky | Error(Adaptive) | Error(Tversky)"""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_dir, img_h, img_w = find_best_deep_model()
    print(f"Best Tversky model: {best_dir}  (inference resolution {img_h}x{img_w})")
    model_path = EXPERIMENTS_DIR / best_dir / "whiteboard_seg_best.pt"
    model = load_model(str(model_path), device)

    n_rows = len(FAILURE_IDS)
    n_cols = 6  # Original | GT | Adaptive | Tversky | Error(Adapt) | Error(Tversky)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original", "Ground Truth", "Adaptive", "Tversky (best seed)",
                   "Error (Adaptive)", "Error (Tversky)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight='bold')

    for i, img_id in enumerate(FAILURE_IDS):
        # Find image file
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = IMAGES_DIR / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"Image not found: {img_id}")
            continue

        mask_path = MASKS_DIR / f"{img_id}.png"
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive prediction (at original resolution, with cleanup to match baseline)
        adaptive_pred = adaptive_threshold(gray)
        adaptive_pred = morphological_cleanup(adaptive_pred)

        # Deep model prediction (use model's training resolution)
        deep_pred = predict_mask(model, img_path, (img_h, img_w), device)

        h, w = gt.shape[:2]
        img_display = cv2.resize(img_rgb, (w, h))

        if adaptive_pred.shape[:2] != (h, w):
            adaptive_pred = cv2.resize(adaptive_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        if deep_pred.shape[:2] != (h, w):
            deep_pred = cv2.resize(deep_pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # Metrics
        adapt_metrics = calculate_metrics(adaptive_pred, gt)
        deep_metrics = calculate_metrics(deep_pred, gt)
        print(f"  {img_id}: adaptive F1={adapt_metrics['f1']:.4f}, "
              f"deep F1={deep_metrics['f1']:.4f}")

        # Error overlays
        adapt_error = create_error_overlay(img_display, adaptive_pred, gt)
        deep_error = create_error_overlay(img_display, deep_pred, gt)

        # Plot
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_ylabel(img_id.replace("image_", "#"), fontsize=11, fontweight='bold')

        axes[i, 1].imshow(gt, cmap='gray')

        axes[i, 2].imshow(adaptive_pred, cmap='gray')
        axes[i, 2].text(
            0.5, -0.06, f"F1 = {adapt_metrics['f1']:.3f}",
            transform=axes[i, 2].transAxes, fontsize=9, color='red',
            ha='center', va='top', fontweight='bold',
        )

        axes[i, 3].imshow(deep_pred, cmap='gray')
        axes[i, 3].text(
            0.5, -0.06, f"F1 = {deep_metrics['f1']:.3f}",
            transform=axes[i, 3].transAxes, fontsize=9, color='blue',
            ha='center', va='top', fontweight='bold',
        )

        axes[i, 4].imshow(adapt_error)
        axes[i, 5].imshow(deep_error)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = FIGURES_DIR / "baseline_vs_deep_failure"
    fig.savefig(str(out_path) + ".png", dpi=200, bbox_inches='tight')
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 2: Scatter plot (Adaptive F1 vs Tversky F1, per image)
# ---------------------------------------------------------------------------

def generate_scatter_plot():
    """Scatter: x=Adaptive F1, y=Tversky F1, diagonal=parity line."""
    test_eval_path = RESULTS_DIR / "test_set_evaluation.json"
    baseline_path = RESULTS_DIR / "classical_baseline_original_resolution.json"

    with open(test_eval_path) as f:
        data = json.load(f)
    with open(baseline_path) as f:
        bl = json.load(f)

    # Baseline per-image F1
    bl_per = {}
    for entry in bl["adaptive"]["per_image"]:
        name = entry["image"].replace(".png", "")
        bl_per[name] = entry["metrics"]["f1"]

    # Tversky per-image F1 (seed-averaged)
    tversky_per = {}
    for model_name, mdata in data["models"].items():
        if "error" in mdata or not model_name.startswith("loss_study/"):
            continue
        if mdata["config"]["loss_type"] != "tversky":
            continue
        for entry in mdata["results"]["per_image"]:
            name = entry["image"].replace(".png", "")
            tversky_per.setdefault(name, []).append(entry["metrics"]["f1"])

    # Build arrays
    images = sorted(bl_per.keys())
    x_adaptive = []
    y_tversky = []
    labels = []
    is_thin = []
    for img in images:
        x_adaptive.append(bl_per[img])
        y_tversky.append(np.mean(tversky_per[img]))
        labels.append(img.replace("image_", "#"))
        is_thin.append(img in THIN_IDS)

    x = np.array(x_adaptive)
    y = np.array(y_tversky)
    thin = np.array(is_thin)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Parity line
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    ax.plot(lims, lims, '--', color='gray', linewidth=1, label='Parity ($y = x$)')

    # Core vs thin markers
    ax.scatter(x[~thin], y[~thin], c='steelblue', s=70, zorder=3, edgecolors='black',
               linewidths=0.5, label='Core images')
    ax.scatter(x[thin], y[thin], c='darkorange', s=70, zorder=3, edgecolors='black',
               linewidths=0.5, marker='D', label='Thin-stroke images')

    # Label each point
    for i, lab in enumerate(labels):
        offset = (6, 4) if y[i] > x[i] else (6, -10)
        ax.annotate(lab, (x[i], y[i]), fontsize=7, textcoords='offset points',
                    xytext=offset, alpha=0.8)

    # Shade regions
    ax.fill_between(lims, lims, lims[1], alpha=0.05, color='blue', label='Deep wins')
    ax.fill_between(lims, lims[0], lims, alpha=0.05, color='red', label='Adaptive wins')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Adaptive Thresholding F1', fontsize=11)
    ax.set_ylabel('Tversky F1 (seed-averaged)', fontsize=11)
    ax.set_title('Per-Image: Adaptive vs Tversky', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "baseline_vs_deep_scatter"
    fig.savefig(str(out_path) + ".png", dpi=200, bbox_inches='tight')
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating baseline failure grid ===")
    generate_failure_grid()
    print("\n=== Generating scatter plot ===")
    generate_scatter_plot()
    print("\nDone.")
