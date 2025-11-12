"""
Comprehensive Comparison: Trainer Model vs Scanner Pipeline vs Ground Truth

This script:
1. Runs test image through TRAINER model (pure ML inference)
2. Runs test image through SCANNER pipeline (full production stack)
3. Compares both results against ground truth mask
4. Identifies where quality is being lost

Shows metrics for:
- Trainer model output (ML quality baseline)
- Scanner pipeline output (production quality)
- Comparison against ground truth
- Visual overlays showing differences
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add paths
TRAINING_DIR = Path(__file__).parent
SCANNER_DIR = TRAINING_DIR.parent / "OneNote-Whiteboard-Scanner" / "local-ai-backend"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(SCANNER_DIR))

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import cv2

# Import scanner components
from ai.hybrid_extractor import HybridStrokeExtractor
from ai.tile_segmentation import TileSegmentation

print("=" * 80)
print("TRAINER vs SCANNER COMPARISON")
print("=" * 80)


def calculate_metrics(pred_mask, gt_mask):
    """Calculate segmentation metrics"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # Intersection and Union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # True/False Positives/Negatives
    tp = intersection
    fp = (pred_binary & ~gt_binary).sum()
    fn = (~pred_binary & gt_binary).sum()
    tn = (~pred_binary & ~gt_binary).sum()
    
    # Metrics
    iou = intersection / union if union > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pixel_acc': pixel_acc,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def load_trainer_model():
    """Load the trainer's best model"""
    model_path = TRAINING_DIR / "models" / "whiteboard_seg_best.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"Loading trainer model: {model_path}")
    
    # Create model
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✓ Trainer model loaded")
    return model


def run_trainer_inference(model, image_path):
    """Run inference using trainer model (pure ML)"""
    print(f"\n{'='*80}")
    print("STEP 1: TRAINER MODEL (Pure ML Inference)")
    print("="*80)
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f"Image size: {original_size[0]}×{original_size[1]}")
    
    # Resize to training resolution
    img_resized = img.resize((1024, 768))  # W×H
    
    # Transform (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    
    # Inference
    print("Running inference at 768×1024...")
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output['out']
        
        # Get probabilities
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    
    # Convert to binary mask (0=background, 1=stroke)
    trainer_mask = (pred * 255).astype(np.uint8)
    
    # Resize back to original size
    trainer_mask = cv2.resize(trainer_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    stroke_pixels = np.count_nonzero(trainer_mask)
    coverage = (stroke_pixels / trainer_mask.size) * 100
    print(f"✓ Trainer output: {stroke_pixels:,} stroke pixels ({coverage:.2f}% coverage)")
    
    return trainer_mask, np.array(img)


def run_scanner_pipeline(image_path):
    """Run full scanner pipeline (preprocessing + ML + postprocessing)"""
    print(f"\n{'='*80}")
    print("STEP 2: SCANNER PIPELINE (Full Production Stack)")
    print("="*80)
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    print(f"Image loaded: {img.shape[1]}×{img.shape[0]}")
    
    # Initialize scanner components
    print("Initializing scanner pipeline...")
    extractor = HybridStrokeExtractor()
    
    # Run full pipeline
    print("Running full scanner pipeline (preprocessing → ML → postprocessing)...")
    try:
        # This runs the EXACT same pipeline as production
        result = extractor.process_image(img)
        
        if result and 'mask' in result:
            scanner_mask = result['mask']
            stroke_pixels = np.count_nonzero(scanner_mask)
            coverage = (stroke_pixels / scanner_mask.size) * 100
            print(f"✓ Scanner output: {stroke_pixels:,} stroke pixels ({coverage:.2f}% coverage)")
            print(f"  Pipeline processed {len(result.get('strokes', []))} strokes")
            print(f"  Processing time: {result.get('metadata', {}).get('processing_time', 0):.2f}s")
            return scanner_mask
        else:
            print(f"❌ Scanner pipeline failed - result keys: {result.keys() if result else 'None'}")
            return None
            
    except Exception as e:
        print(f"❌ Scanner pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_visualization(original_img, gt_mask, trainer_mask, scanner_mask, 
                                   trainer_metrics, scanner_metrics, output_path):
    """Create comprehensive comparison visualization"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Trainer Model vs Scanner Pipeline vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Row 1: Original, Ground Truth, Overlays
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Overlay: Red=GT, Green=Trainer, Yellow=Both
    overlay1 = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlay1[gt_mask > 127] = [255, 0, 0]  # Red = GT only
    overlay1[trainer_mask > 127] = [0, 255, 0]  # Green = Trainer only
    overlay1[(gt_mask > 127) & (trainer_mask > 127)] = [255, 255, 0]  # Yellow = Both
    axes[0, 2].imshow(overlay1)
    axes[0, 2].set_title('GT (red) vs Trainer (green)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Trainer output and analysis
    axes[1, 0].imshow(trainer_mask, cmap='gray')
    axes[1, 0].set_title(f'Trainer Model Output\nIoU: {trainer_metrics["iou"]:.3f} | F1: {trainer_metrics["f1"]:.3f}', 
                        fontweight='bold')
    axes[1, 0].axis('off')
    
    # Trainer difference map
    diff_trainer = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    diff_trainer[(gt_mask > 127) & (trainer_mask > 127)] = [255, 255, 255]  # White = correct
    diff_trainer[(gt_mask <= 127) & (trainer_mask > 127)] = [255, 0, 0]  # Red = false positive
    diff_trainer[(gt_mask > 127) & (trainer_mask <= 127)] = [0, 255, 0]  # Green = false negative
    axes[1, 1].imshow(diff_trainer)
    axes[1, 1].set_title(f'Trainer Errors\nFP: {trainer_metrics["fp"]:,} | FN: {trainer_metrics["fn"]:,}', 
                        fontweight='bold')
    axes[1, 1].axis('off')
    
    # Trainer metrics text
    trainer_text = f"""TRAINER MODEL METRICS:
    
IoU (Overlap):        {trainer_metrics['iou']:.4f}
F1 Score:            {trainer_metrics['f1']:.4f}
Precision:           {trainer_metrics['precision']:.4f}
Recall:              {trainer_metrics['recall']:.4f}
Pixel Accuracy:      {trainer_metrics['pixel_acc']:.4f}

True Positives:      {trainer_metrics['tp']:,}
False Positives:     {trainer_metrics['fp']:,}
False Negatives:     {trainer_metrics['fn']:,}
"""
    axes[1, 2].text(0.1, 0.5, trainer_text, fontsize=10, family='monospace', 
                   verticalalignment='center')
    axes[1, 2].axis('off')
    
    # Row 3: Scanner output and analysis
    if scanner_mask is not None:
        axes[2, 0].imshow(scanner_mask, cmap='gray')
        axes[2, 0].set_title(f'Scanner Pipeline Output\nIoU: {scanner_metrics["iou"]:.3f} | F1: {scanner_metrics["f1"]:.3f}', 
                            fontweight='bold')
        axes[2, 0].axis('off')
        
        # Scanner difference map
        diff_scanner = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        diff_scanner[(gt_mask > 127) & (scanner_mask > 127)] = [255, 255, 255]  # White = correct
        diff_scanner[(gt_mask <= 127) & (scanner_mask > 127)] = [255, 0, 0]  # Red = false positive
        diff_scanner[(gt_mask > 127) & (scanner_mask <= 127)] = [0, 255, 0]  # Green = false negative
        axes[2, 1].imshow(diff_scanner)
        axes[2, 1].set_title(f'Scanner Errors\nFP: {scanner_metrics["fp"]:,} | FN: {scanner_metrics["fn"]:,}', 
                            fontweight='bold')
        axes[2, 1].axis('off')
        
        # Scanner metrics text
        scanner_text = f"""SCANNER PIPELINE METRICS:

IoU (Overlap):        {scanner_metrics['iou']:.4f}
F1 Score:            {scanner_metrics['f1']:.4f}
Precision:           {scanner_metrics['precision']:.4f}
Recall:              {scanner_metrics['recall']:.4f}
Pixel Accuracy:      {scanner_metrics['pixel_acc']:.4f}

True Positives:      {scanner_metrics['tp']:,}
False Positives:     {scanner_metrics['fp']:,}
False Negatives:     {scanner_metrics['fn']:,}

QUALITY LOSS:
IoU Loss:            {(trainer_metrics['iou'] - scanner_metrics['iou']):.4f}
F1 Loss:             {(trainer_metrics['f1'] - scanner_metrics['f1']):.4f}
"""
        axes[2, 2].text(0.1, 0.5, scanner_text, fontsize=10, family='monospace', 
                       verticalalignment='center')
    else:
        axes[2, 0].text(0.5, 0.5, 'Scanner pipeline failed', ha='center', va='center')
        axes[2, 1].text(0.5, 0.5, 'No scanner output', ha='center', va='center')
        axes[2, 2].text(0.5, 0.5, 'No scanner metrics', ha='center', va='center')
    
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    """Run comprehensive comparison"""
    
    # Paths
    test_images_dir = TRAINING_DIR / "dataset" / "test-images"
    images_dir = test_images_dir / "images"
    masks_dir = test_images_dir / "masks"
    output_dir = TRAINING_DIR / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Find test image
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"❌ No test images found in {images_dir}")
        print("Please add a test image to dataset/test-images/images/")
        return
    
    print(f"Found {len(image_files)} test image(s)")
    
    for img_path in image_files:
        print(f"\n{'='*80}")
        print(f"Processing: {img_path.name}")
        print("="*80)
        
        # Find corresponding mask
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"⚠️  No mask found: {mask_path}")
            continue
        
        # Load ground truth
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_binary = (gt_mask > 127).astype(np.uint8)
        gt_pixels = np.count_nonzero(gt_binary)
        print(f"Ground truth: {gt_pixels:,} stroke pixels ({(gt_pixels/gt_mask.size)*100:.2f}% coverage)")
        
        # Step 1: Trainer model inference
        trainer_model = load_trainer_model()
        if trainer_model is None:
            continue
        
        trainer_mask, original_img = run_trainer_inference(trainer_model, img_path)
        trainer_metrics = calculate_metrics(trainer_mask, gt_mask)
        
        print(f"\nTrainer Model Results:")
        print(f"  IoU:       {trainer_metrics['iou']:.4f}")
        print(f"  F1 Score:  {trainer_metrics['f1']:.4f}")
        print(f"  Precision: {trainer_metrics['precision']:.4f}")
        print(f"  Recall:    {trainer_metrics['recall']:.4f}")
        
        # Step 2: Scanner pipeline
        scanner_mask = run_scanner_pipeline(img_path)
        
        if scanner_mask is not None:
            # Resize scanner output to match ground truth size if needed
            if scanner_mask.shape != gt_mask.shape:
                scanner_mask = cv2.resize(scanner_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
            scanner_metrics = calculate_metrics(scanner_mask, gt_mask)
            
            print(f"\nScanner Pipeline Results:")
            print(f"  IoU:       {scanner_metrics['iou']:.4f}")
            print(f"  F1 Score:  {scanner_metrics['f1']:.4f}")
            print(f"  Precision: {scanner_metrics['precision']:.4f}")
            print(f"  Recall:    {scanner_metrics['recall']:.4f}")
            
            print(f"\nQuality Loss (Trainer → Scanner):")
            print(f"  IoU Loss:  {(trainer_metrics['iou'] - scanner_metrics['iou']):.4f}")
            print(f"  F1 Loss:   {(trainer_metrics['f1'] - scanner_metrics['f1']):.4f}")
        else:
            scanner_metrics = None
        
        # Create visualization
        output_path = output_dir / f"{img_path.stem}_comparison.png"
        create_comparison_visualization(
            original_img, gt_mask, trainer_mask, scanner_mask,
            trainer_metrics, scanner_metrics, output_path
        )
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
