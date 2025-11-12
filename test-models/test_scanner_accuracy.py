"""
Test Scanner Accuracy on Dataset
Processes training images through the full scanner pipeline (as if they were photos)
and compares output to ground truth masks to calculate accuracy metrics.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List

# Add scanner backend to path
scanner_backend = Path(__file__).parent.parent / "OneNote-Whiteboard-Scanner" / "local-ai-backend"
sys.path.insert(0, str(scanner_backend))

from ai.hybrid_extractor import HybridStrokeExtractor
import json

def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate segmentation metrics"""
    # Ensure binary
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # Resize if needed
    if pred_binary.shape != gt_binary.shape:
        pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Calculate metrics
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    pred_positive = pred_binary.sum()
    gt_positive = gt_binary.sum()
    
    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0
    
    # Dice coefficient (F1 score)
    dice = 2 * intersection / (pred_positive + gt_positive) if (pred_positive + gt_positive) > 0 else 0.0
    
    # Precision and Recall
    true_positive = intersection
    false_positive = pred_positive - true_positive
    false_negative = gt_positive - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    
    # Pixel accuracy
    correct = (pred_binary == gt_binary).sum()
    total = gt_binary.size
    pixel_acc = correct / total
    
    return {
        'iou': iou,
        'dice': dice,
        'f1': dice,  # Same as dice
        'precision': precision,
        'recall': recall,
        'pixel_accuracy': pixel_acc,
        'stroke_pixels_pred': int(pred_positive),
        'stroke_pixels_gt': int(gt_positive)
    }


def visualize_comparison(image_path: Path, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                        metrics: Dict, output_path: Path):
    """Create visualization comparing prediction vs ground truth"""
    
    # Load original image
    img = Image.open(image_path).convert("RGB")
    
    # Resize for display
    display_scale = min(1.0, 2000 / max(img.size))
    display_size = (int(img.size[0] * display_scale), int(img.size[1] * display_scale))
    
    img_display = img.resize(display_size, Image.LANCZOS)
    
    # Resize masks for display
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # Ensure same size for comparison
    if pred_binary.shape != gt_binary.shape:
        pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    pred_display = cv2.resize(pred_binary, display_size, interpolation=cv2.INTER_NEAREST)
    gt_display = cv2.resize(gt_binary, display_size, interpolation=cv2.INTER_NEAREST)
    
    # Create difference map
    diff_map = np.zeros((*display_size[::-1], 3), dtype=np.uint8)
    scanner_only = (pred_display > gt_display)
    gt_only = (gt_display > pred_display)
    both = (pred_display > 0) & (gt_display > 0)
    
    diff_map[scanner_only] = [255, 0, 0]      # Red: Scanner extra (false positives)
    diff_map[gt_only] = [0, 255, 0]           # Green: Missed strokes (false negatives)
    diff_map[both] = [255, 255, 255]          # White: Correct detections
    
    # Create overlays
    overlay_scanner = np.array(img_display).copy()
    overlay_scanner[pred_display > 0] = [255, 0, 0]  # Red
    
    overlay_gt = np.array(img_display).copy()
    overlay_gt[gt_display > 0] = [0, 255, 0]  # Green
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Masks
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f"Original Image\n{img.size[0]}×{img.size[1]}", fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_display, cmap='gray')
    axes[0, 1].set_title(f"Scanner Output\n{metrics['stroke_pixels_pred']:,} pixels", fontsize=12, weight='bold', color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_display, cmap='gray')
    axes[0, 2].set_title(f"Ground Truth\n{metrics['stroke_pixels_gt']:,} pixels", fontsize=12, weight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays and difference
    axes[1, 0].imshow(overlay_scanner)
    axes[1, 0].set_title("Scanner Overlay (Red)", fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_gt)
    axes[1, 1].set_title("Ground Truth Overlay (Green)", fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_map)
    metrics_text = f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f}\n"
    metrics_text += f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}\n"
    metrics_text += f"Pixel Acc: {metrics['pixel_accuracy']:.4f}"
    axes[1, 2].set_title(f"Difference Map\n{metrics_text}", fontsize=10, weight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("SCANNER ACCURACY TEST ON DATASET")
    print("="*70)
    print("Processing training images through full scanner pipeline")
    print("Comparing output masks to ground truth annotations")
    print("="*70)
    print()
    
    # Initialize scanner
    print("Initializing scanner...")
    scanner = HybridStrokeExtractor()
    print("✓ Scanner ready")
    print()
    
    # Find test images and masks
    dataset_dir = Path("dataset")
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    
    image_files = sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    print()
    
    # Create output directory
    output_dir = Path("scanner_accuracy_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print("="*70)
        print(f"Processing {idx}/{len(image_files)}: {img_path.name}")
        print("="*70)
        
        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"⚠ No mask found, skipping")
            print()
            continue
        
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ Could not load image")
                continue
            
            print(f"  Image size: {img.shape[1]}×{img.shape[0]}")
            
            # Process through scanner (full pipeline as if from camera)
            print(f"  Processing through scanner pipeline...")
            result = scanner.process_image(img)
            
            if result is None or 'mask' not in result:
                print(f"  ❌ Scanner failed to process")
                continue
            
            scanner_mask = result['mask']
            print(f"  ✓ Scanner output: {scanner_mask.shape}")
            print(f"  ✓ Detected {np.count_nonzero(scanner_mask):,} stroke pixels")
            
            # Load ground truth mask
            gt_mask = np.array(Image.open(mask_path).convert("L"))
            print(f"  ✓ Ground truth: {gt_mask.shape}")
            print(f"  ✓ GT has {np.count_nonzero(gt_mask > 127):,} stroke pixels")
            
            # Calculate metrics
            metrics = calculate_metrics(scanner_mask, gt_mask)
            
            print(f"  📊 Metrics:")
            print(f"     IoU:         {metrics['iou']:.4f}")
            print(f"     F1/Dice:     {metrics['f1']:.4f}")
            print(f"     Precision:   {metrics['precision']:.4f}")
            print(f"     Recall:      {metrics['recall']:.4f}")
            print(f"     Pixel Acc:   {metrics['pixel_accuracy']:.4f}")
            
            # Save results
            results.append({
                'image': img_path.name,
                **metrics
            })
            
            # Create visualization
            viz_path = output_dir / f"{img_path.stem}_comparison.png"
            visualize_comparison(img_path, scanner_mask, gt_mask, metrics, viz_path)
            print(f"  ✓ Saved visualization: {viz_path.name}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if results:
        avg_iou = np.mean([r['iou'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_pixel_acc = np.mean([r['pixel_accuracy'] for r in results])
        
        print(f"Tested on {len(results)} images:")
        print(f"  Average IoU:        {avg_iou:.4f}")
        print(f"  Average F1:         {avg_f1:.4f}")
        print(f"  Average Precision:  {avg_precision:.4f}")
        print(f"  Average Recall:     {avg_recall:.4f}")
        print(f"  Average Pixel Acc:  {avg_pixel_acc:.4f}")
        print()
        
        # Per-image results table
        print("Per-Image Results:")
        print("-" * 70)
        print(f"{'Image':<20} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['image']:<20} {r['iou']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}")
        print("-" * 70)
        
        # Save results to file
        results_file = output_dir / "accuracy_results.txt"
        with open(results_file, 'w') as f:
            f.write("SCANNER ACCURACY TEST RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Tested on {len(results)} images\n")
            f.write(f"Average IoU:        {avg_iou:.4f}\n")
            f.write(f"Average F1:         {avg_f1:.4f}\n")
            f.write(f"Average Precision:  {avg_precision:.4f}\n")
            f.write(f"Average Recall:     {avg_recall:.4f}\n")
            f.write(f"Average Pixel Acc:  {avg_pixel_acc:.4f}\n\n")
            f.write("Per-Image Results:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Image':<20} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}\n")
            f.write("-" * 70 + "\n")
            for r in results:
                f.write(f"{r['image']:<20} {r['iou']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}\n")
        
        print(f"\n✅ Results saved to: {output_dir}")
        print(f"✅ Summary saved to: {results_file}")
    else:
        print("❌ No images were successfully processed")


if __name__ == "__main__":
    main()
