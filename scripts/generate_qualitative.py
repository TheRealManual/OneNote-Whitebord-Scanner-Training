"""
Generate qualitative comparison grids for the paper.

Produces side-by-side grids:
  Original → Ground Truth → Prediction → Error Overlay

Usage:
    python scripts/generate_qualitative.py \
        --model-dir ../SegmentationResearchPaper/experiments/loss_study/dice_focal_seed42 \
        --images-dir dataset/images --masks-dir dataset/masks
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


def create_error_overlay(original, pred_mask, gt_mask):
    """Create a color-coded error overlay.

    Green = true positive, Red = false positive, Blue = false negative.
    """
    h, w = gt_mask.shape[:2]
    if original.shape[:2] != (h, w):
        original = cv2.resize(original, (w, h))

    overlay = original.copy()
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    overlay[tp > 0] = [0, 200, 0]    # Green: TP
    overlay[fp > 0] = [200, 0, 0]    # Red: FP
    overlay[fn > 0] = [0, 0, 200]    # Blue: FN

    # Blend with original
    alpha = 0.5
    blended = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)
    return blended


def generate_grid(image_paths, mask_paths, pred_masks, output_path, max_images=6):
    """Generate a grid of Original | GT | Prediction | Error.

    Args:
        image_paths: List of image file paths.
        mask_paths: List of ground-truth mask file paths.
        pred_masks: List of predicted masks (numpy arrays).
        output_path: Where to save the grid image.
        max_images: Maximum rows.
    """
    n = min(len(image_paths), max_images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0, 3].set_title('Error Overlay', fontsize=12, fontweight='bold')

    for i in range(n):
        img = cv2.imread(str(image_paths[i]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        pred = pred_masks[i]

        # Resize to match
        h, w = gt.shape[:2]
        img_rgb = cv2.resize(img_rgb, (w, h))
        if pred.shape[:2] != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        error = create_error_overlay(img_rgb, pred, gt)
        metrics = calculate_metrics(pred, gt)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(image_paths[i].stem, fontsize=9)
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_xlabel(f"F1={metrics['f1']:.3f}", fontsize=9)
        axes[i, 3].imshow(error)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def load_model_predictions(model_dir, images_dir, img_size=(768, 1024)):
    """Load a trained model and run inference on images.

    Returns list of (image_path, mask_path, pred_mask) tuples.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    model_path = Path(model_dir) / "whiteboard_seg_best.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return []

    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted(list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg")))
    results = []

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, dict):
                output = output['out']
            pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

        pred_mask = (pred * 255).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
        results.append((img_path, pred_mask))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison grids")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to model directory with whiteboard_seg_best.pt")
    parser.add_argument("--images-dir", type=str, default=str(REPO_ROOT / "dataset" / "images"),
                       help="Path to test images")
    parser.add_argument("--masks-dir", type=str, default=str(REPO_ROOT / "dataset" / "masks"),
                       help="Path to ground-truth masks")
    parser.add_argument("--output-dir", type=str,
                       default=str(PRIVATE_REPO / "results" / "qualitative"),
                       help="Where to save grid images")
    parser.add_argument("--max-images", type=int, default=6,
                       help="Maximum rows in grid")
    parser.add_argument("--img-height", type=int, default=768)
    parser.add_argument("--img-width", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir)
    results = load_model_predictions(
        args.model_dir, args.images_dir,
        img_size=(args.img_height, args.img_width)
    )

    if not results:
        print("No predictions generated.")
        return

    image_paths = []
    mask_paths = []
    pred_masks = []

    for img_path, pred in results:
        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            image_paths.append(img_path)
            mask_paths.append(mask_path)
            pred_masks.append(pred)

    if image_paths:
        grid_path = output_dir / "qualitative_grid.png"
        generate_grid(image_paths, mask_paths, pred_masks, grid_path,
                      max_images=args.max_images)
    else:
        print("No matching image/mask pairs found.")


if __name__ == "__main__":
    main()
