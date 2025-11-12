"""
PRODUCTION Test Script - Matches Scanner Settings Exactly
Tests model with same smooth Gaussian-blended tiling used in production scanner

PRODUCTION SETTINGS (must match scanner):
- Tile size: 768×1024 (H×W)
- Overlap: 50% (smooth Gaussian blending)
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Classes: 2 (background=0, stroke=1)
- Method: Full-image smooth tiled inference (NOT uncertain-tile)

This script uses the EXACT same preprocessing and inference as the scanner's
tile_segmentation.py infer_full_image_smooth() method.

Usage:
    python test_model_production.py
    python test_model_production.py --model models/whiteboard_seg_best.pt
"""

import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from torchvision import transforms


# PRODUCTION SETTINGS (must match scanner)
TILE_SIZE = (768, 1024)  # (height, width) - matches training resolution
OVERLAP = 0.5  # 50% overlap for smooth blending
NUM_CLASSES = 2  # Binary: background (0), stroke (1)


def load_model(model_path, num_classes=NUM_CLASSES):
    """Load trained model"""
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def create_gaussian_weight_map(height, width, sigma=None):
    """Create 2D Gaussian weight map for smooth tile blending"""
    if sigma is None:
        sigma = min(height, width) / 6
    
    y = np.linspace(-height/2, height/2, height)
    x = np.linspace(-width/2, width/2, width)
    
    gauss_y = np.exp(-(y**2) / (2 * sigma**2))
    gauss_x = np.exp(-(x**2) / (2 * sigma**2))
    
    weight_map = np.outer(gauss_y, gauss_x)
    weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min())
    
    return weight_map


def predict_smooth_tiled(model, image_path, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    PRODUCTION inference with smooth Gaussian-blended tiling
    Exactly matches scanner's infer_full_image_smooth() method
    """
    # Load full-resolution image
    img_full = Image.open(image_path).convert("RGB")
    full_width, full_height = img_full.size
    
    print(f"  Image: {full_width}×{full_height}")
    print(f"  Tiles: {tile_size[1]}×{tile_size[0]} with {overlap*100:.0f}% overlap")
    
    # Calculate stride
    tile_h, tile_w = tile_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))
    
    # Prepare transform (matches training)
    transform = transforms.Compose([
        transforms.Resize(tile_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize accumulation arrays
    prediction_sum = np.zeros((full_height, full_width), dtype=np.float32)
    weight_sum = np.zeros((full_height, full_width), dtype=np.float32)
    
    # Create Gaussian weight map
    weight_map = create_gaussian_weight_map(tile_h, tile_w)
    
    # Calculate grid
    tiles_h = (full_height - tile_h) // stride_h + 1
    tiles_w = (full_width - tile_w) // stride_w + 1
    total_tiles = tiles_h * tiles_w
    
    # Add edge tiles
    if full_width > tiles_w * stride_w:
        total_tiles += tiles_h
    if full_height > tiles_h * stride_h:
        total_tiles += tiles_w
    if full_width > tiles_w * stride_w and full_height > tiles_h * stride_h:
        total_tiles += 1
    
    print(f"  Processing {total_tiles} tiles...")
    
    tile_count = 0
    
    # Process main grid
    for y in range(0, full_height - tile_h + 1, stride_h):
        for x in range(0, full_width - tile_w + 1, stride_w):
            tile = img_full.crop((x, y, x + tile_w, y + tile_h))
            tile_tensor = transform(tile).unsqueeze(0)
            
            with torch.no_grad():
                output = model(tile_tensor)
                if isinstance(output, dict):
                    output = output["out"]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            
            prediction_sum[y:y+tile_h, x:x+tile_w] += pred * weight_map
            weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
            tile_count += 1
    
    # Process right edge
    if full_width > tiles_w * stride_w:
        x = full_width - tile_w
        for y in range(0, full_height - tile_h + 1, stride_h):
            tile = img_full.crop((x, y, full_width, y + tile_h))
            tile_tensor = transform(tile).unsqueeze(0)
            with torch.no_grad():
                output = model(tile_tensor)
                if isinstance(output, dict):
                    output = output["out"]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            prediction_sum[y:y+tile_h, x:x+tile_w] += pred * weight_map
            weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
            tile_count += 1
    
    # Process bottom edge
    if full_height > tiles_h * stride_h:
        y = full_height - tile_h
        for x in range(0, full_width - tile_w + 1, stride_w):
            tile = img_full.crop((x, y, x + tile_w, full_height))
            tile_tensor = transform(tile).unsqueeze(0)
            with torch.no_grad():
                output = model(tile_tensor)
                if isinstance(output, dict):
                    output = output["out"]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            prediction_sum[y:y+tile_h, x:x+tile_w] += pred * weight_map
            weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
            tile_count += 1
    
    # Process bottom-right corner
    if full_width > tiles_w * stride_w and full_height > tiles_h * stride_h:
        x = full_width - tile_w
        y = full_height - tile_h
        tile = img_full.crop((x, y, full_width, full_height))
        tile_tensor = transform(tile).unsqueeze(0)
        with torch.no_grad():
            output = model(tile_tensor)
            if isinstance(output, dict):
                output = output["out"]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        prediction_sum[y:y+tile_h, x:x+tile_w] += pred * weight_map
        weight_sum[y:y+tile_h, x:x+tile_w] += weight_map
        tile_count += 1
    
    # Normalize by weights (weighted average blending)
    weight_sum = np.maximum(weight_sum, 1e-8)
    final_prediction = (prediction_sum / weight_sum).round().astype(np.uint8)
    
    print(f"  ✓ Processed {tile_count} tiles")
    print(f"  ✓ Result: {full_width}×{full_height} ({np.count_nonzero(final_prediction)} stroke pixels)")
    
    return final_prediction


def calculate_metrics(prediction, mask_path):
    """Calculate accuracy metrics"""
    if not mask_path or not Path(mask_path).exists():
        return None
    
    mask = np.array(Image.open(mask_path).convert("L"))
    
    # Resize mask if needed
    if mask.shape != prediction.shape:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize((prediction.shape[1], prediction.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    
    mask = (mask > 127).astype(np.uint8)
    
    # Metrics
    correct = (prediction == mask).sum()
    pixel_acc = correct / mask.size
    
    intersection = ((prediction == 1) & (mask == 1)).sum()
    union = ((prediction == 1) | (mask == 1)).sum()
    iou_stroke = intersection / union if union > 0 else 0
    
    true_positive = intersection
    false_positive = ((prediction == 1) & (mask == 0)).sum()
    false_negative = ((prediction == 0) & (mask == 1)).sum()
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'pixel_accuracy': pixel_acc,
        'iou_stroke': iou_stroke,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def visualize(image_path, mask_path, prediction, save_path):
    """Visualize results"""
    img = Image.open(image_path).convert("RGB")
    
    # Resize for display
    display_scale = min(1.0, 2000 / max(img.size))
    display_size = (int(img.size[0] * display_scale), int(img.size[1] * display_scale))
    
    img_display = img.resize(display_size, Image.LANCZOS)
    pred_display = Image.fromarray(prediction).resize(display_size, Image.NEAREST)
    
    has_mask = mask_path and Path(mask_path).exists()
    
    fig, axes = plt.subplots(1, 3 if not has_mask else 4, figsize=(15 if not has_mask else 20, 5))
    
    axes[0].imshow(img_display)
    axes[0].set_title(f"Original\n{img.size[0]}×{img.size[1]}")
    axes[0].axis("off")
    
    axes[1].imshow(pred_display, cmap='gray')
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    
    overlay = np.array(img_display).copy()
    pred_arr = np.array(pred_display)
    overlay[pred_arr == 1] = [255, 0, 0]
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    if has_mask:
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_display = Image.fromarray((mask > 127).astype(np.uint8) * 255).resize(display_size, Image.NEAREST)
        axes[3].imshow(mask_display, cmap='gray')
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test model with PRODUCTION settings")
    parser.add_argument("--model", type=str, default="models/whiteboard_seg_best.pt")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--output", type=str, default="test_results_production")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRODUCTION TEST - Scanner-Matched Settings")
    print("="*70)
    print(f"Tile size:  {TILE_SIZE[1]}×{TILE_SIZE[0]} (W×H)")
    print(f"Overlap:    {OVERLAP*100:.0f}%")
    print(f"Classes:    {NUM_CLASSES} (background, stroke)")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    print("✓ Model loaded\n")
    
    # Setup paths
    dataset_dir = Path(args.dataset)
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Get images
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    print(f"Testing {len(images)} images...\n")
    
    all_metrics = []
    
    for img_path in images:
        print(f"📄 {img_path.name}")
        mask_path = mask_dir / img_path.name.replace(".jpg", ".png")
        
        # Predict
        prediction = predict_smooth_tiled(model, img_path, TILE_SIZE, OVERLAP)
        
        # Metrics
        metrics = calculate_metrics(prediction, mask_path)
        if metrics:
            all_metrics.append(metrics)
            print(f"  Accuracy: {metrics['pixel_accuracy']:.4f} | IoU: {metrics['iou_stroke']:.4f} | F1: {metrics['f1_score']:.4f}")
        
        # Visualize
        save_path = output_dir / f"{img_path.stem}_production.png"
        visualize(img_path, mask_path, prediction, save_path)
        print()
    
    # Summary
    if all_metrics:
        print("="*70)
        print("AVERAGE METRICS (PRODUCTION SETTINGS)")
        print("="*70)
        for key in all_metrics[0].keys():
            avg = sum(m[key] for m in all_metrics) / len(all_metrics)
            print(f"{key:20s}: {avg:.4f}")
        print("="*70)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
