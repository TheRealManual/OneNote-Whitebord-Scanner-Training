"""
Realistic Scanner Performance Test
Compares scanner F1/IoU to training F1/IoU using ground truth masks
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add scanner to path
scanner_dir = Path(r"C:\Users\thelo.NICKS_MAIN_PC\OneDrive\Desktop\Repos\OneNote-Whiteboard-Scanner\local-ai-backend")
sys.path.insert(0, str(scanner_dir))

from ai.tile_segmentation import TileSegmentation

def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and F1 score"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    iou = intersection / union if union > 0 else 0.0
    
    # F1 = 2 * precision * recall / (precision + recall)
    # F1 = 2 * TP / (2*TP + FP + FN) = 2 * intersection / (pred + gt)
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    f1 = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    
    return iou, f1

# Initialize scanner
print("="*70)
print("SCANNER PERFORMANCE TEST")
print("="*70)
print()
print("Loading scanner model...")
scanner = TileSegmentation()
print(f"✓ Scanner loaded")
print(f"  Model type: {scanner.model_type}")
print(f"  Input size: {scanner.input_size}")
print()

# Load test images and masks
data_dir = Path("dataset/test-images")
test_images = sorted((data_dir / "images").glob("*.png"))
test_masks = [data_dir / "masks" / img.name for img in test_images]

print(f"Found {len(test_images)} test images")
print()

# Run scanner on test set
total_iou = 0.0
total_f1 = 0.0
results = []

for img_path, mask_path in zip(test_images, test_masks):
    print(f"Testing: {img_path.name}")
    
    # Load image and ground truth
    img_pil = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gt_mask = np.array(Image.open(mask_path).convert("L"))
    
    # Run scanner
    pred_mask = scanner.infer_full_image_smooth(img_cv)
    
    if pred_mask is None:
        print(f"  ❌ Scanner failed")
        continue
    
    # Resize prediction to match ground truth size
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Calculate metrics
    iou, f1 = calculate_metrics(pred_mask, gt_mask)
    
    total_iou += iou
    total_f1 += f1
    results.append({'image': img_path.name, 'iou': iou, 'f1': f1})
    
    print(f"  IoU: {iou:.4f}, F1: {f1:.4f}")

# Summary
print()
print("="*70)
print("RESULTS")
print("="*70)
avg_iou = total_iou / len(results)
avg_f1 = total_f1 / len(results)

print(f"Average IoU: {avg_iou:.4f}")
print(f"Average F1: {avg_f1:.4f}")
print()
print("Model 4 Training Performance:")
print("  Test IoU: 0.7096")
print("  Test F1: 0.8204")
print()
print("Comparison:")
print(f"  IoU difference: {avg_iou - 0.7096:+.4f} ({((avg_iou - 0.7096) / 0.7096 * 100):+.2f}%)")
print(f"  F1 difference: {avg_f1 - 0.8204:+.4f} ({((avg_f1 - 0.8204) / 0.8204 * 100):+.2f}%)")
print()

if abs(avg_f1 - 0.8204) < 0.05:
    print("✓ Scanner performance matches training within 5% tolerance")
elif avg_f1 > 0.75:
    print("✓ Scanner performance is acceptable (F1 > 0.75)")
else:
    print("⚠️ Scanner performance below expected level")
