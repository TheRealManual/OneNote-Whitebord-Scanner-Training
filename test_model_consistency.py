"""
Unified Model Validation Test
Tests that the exported model in the scanner produces the same results as the training model.
Ensures consistency between training evaluation and production deployment.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# Add scanner backend to path
scanner_dir = Path(r"C:\Users\thelo.NICKS_MAIN_PC\OneDrive\Desktop\Repos\OneNote-Whiteboard-Scanner\local-ai-backend")
sys.path.insert(0, str(scanner_dir))

from ai.tile_segmentation import TileSegmentation


def load_training_model():
    """Load Model 4 from training (best model)"""
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    
    model_path = Path("backup/4/whiteboard_seg_best.pt")
    print(f"Loading training model: {model_path}")
    
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    print("✓ Training model loaded")
    return model


def test_model_consistency():
    """Test that scanner model produces same results as training model"""
    
    print("="*70)
    print("UNIFIED MODEL VALIDATION TEST")
    print("="*70)
    print("Testing consistency between training and production models")
    print("="*70)
    print()
    
    # 1. Load training model (PyTorch .pt format)
    training_model = load_training_model()
    
    # 2. Initialize scanner's tile segmentation (uses exported .pts format)
    print()
    print(f"Loading scanner model from: {scanner_dir / 'models' / 'whiteboard_seg.pts'}")
    scanner = TileSegmentation()
    print(f"✓ Scanner model loaded")
    print(f"  Model type: {scanner.model_type}")
    print(f"  Input size: {scanner.input_size}")
    print()
    
    # 3. Load test images
    test_images_dir = Path("dataset/test-images/images")
    if not test_images_dir.exists():
        # Fallback to main dataset images
        test_images_dir = Path("dataset/images")
    
    test_images = sorted(test_images_dir.glob("*.png"))[:6]  # Use first 6 images
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    print()
    
    # ImageNet normalization (used by both models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Test each image
    results = []
    
    for idx, img_path in enumerate(test_images, 1):
        print("="*70)
        print(f"Testing {idx}/{len(test_images)}: {img_path.name}")
        print("="*70)
        
        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        print(f"  Image size: {img_pil.size[0]}×{img_pil.size[1]}")
        
        # === TEST 1: Training Model (Direct Inference) ===
        print()
        print("  [1] Testing Training Model...")
        
        # Resize to TRAINING resolution (1536x1536 - Model 4's native size)
        img_resized = img_pil.resize((1536, 1536), Image.LANCZOS)
        img_tensor = transforms.ToTensor()(img_resized)
        img_normalized = normalize(img_tensor).unsqueeze(0)
        
        with torch.no_grad():
            output = training_model(img_normalized)
            if isinstance(output, dict):
                output = output['out']
            pred = torch.softmax(output, dim=1)[0, 1]  # Stroke class probability
            training_mask_native = (pred > 0.5).cpu().numpy().astype(np.uint8) * 255
        
        # Resize training output back to original image size for fair comparison
        h_orig, w_orig = img_pil.size[1], img_pil.size[0]
        training_mask = cv2.resize(training_mask_native, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        training_mask = (training_mask > 127).astype(np.uint8) * 255
        
        training_pixels = np.count_nonzero(training_mask)
        print(f"      Model resolution: 1536×1536")
        print(f"      Output upscaled to: {w_orig}×{h_orig}")
        print(f"      Stroke pixels: {training_pixels:,}")
        
        # === TEST 2: Scanner Model (Production Pipeline) ===
        print()
        print("  [2] Testing Scanner Model (Full Pipeline)...")
        
        scanner_mask = scanner.infer_full_image_smooth(img_cv)
        
        if scanner_mask is None:
            print(f"      ❌ Scanner failed to process")
            continue
        
        scanner_pixels = np.count_nonzero(scanner_mask)
        print(f"      Output: {scanner_mask.shape[1]}×{scanner_mask.shape[0]}")
        print(f"      Stroke pixels: {scanner_pixels:,}")
        
        # === COMPARISON ===
        print()
        print("  [3] Comparing Results...")
        
        # Both masks now at original image resolution
        training_binary = (training_mask > 127).astype(np.uint8)
        scanner_binary = (scanner_mask > 127).astype(np.uint8)
        
        intersection = np.logical_and(training_binary, scanner_binary).sum()
        union = np.logical_or(training_binary, scanner_binary).sum()
        
        agreement = intersection / union if union > 0 else 0.0
        pixel_diff = abs(training_pixels - scanner_pixels)
        pixel_diff_pct = (pixel_diff / max(training_pixels, scanner_pixels) * 100) if max(training_pixels, scanner_pixels) > 0 else 0
        
        print(f"      Pixel Agreement (IoU): {agreement:.4f} ({agreement*100:.2f}%)")
        print(f"      Pixel Count Difference: {pixel_diff:,} ({pixel_diff_pct:.2f}%)")
        
        # Determine if models are consistent
        is_consistent = agreement > 0.95 and pixel_diff_pct < 10
        
        if is_consistent:
            print(f"      ✅ CONSISTENT - Models produce nearly identical results")
        else:
            print(f"      ⚠️  DIFFERENCE DETECTED")
            if agreement < 0.95:
                print(f"         - Low pixel agreement: {agreement:.4f} (target: >0.95)")
            if pixel_diff_pct >= 10:
                print(f"         - Large pixel count difference: {pixel_diff_pct:.2f}% (target: <10%)")
        
        results.append({
            'image': img_path.name,
            'training_pixels': training_pixels,
            'scanner_pixels': scanner_pixels,
            'agreement': agreement,
            'consistent': is_consistent
        })
        
        print()
    
    # === SUMMARY ===
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()
    
    if results:
        consistent_count = sum(1 for r in results if r['consistent'])
        avg_agreement = np.mean([r['agreement'] for r in results])
        
        print(f"Images tested: {len(results)}")
        print(f"Consistent: {consistent_count}/{len(results)} ({consistent_count/len(results)*100:.1f}%)")
        print(f"Average pixel agreement: {avg_agreement:.4f} ({avg_agreement*100:.2f}%)")
        print()
        
        if consistent_count == len(results):
            print("✅ ALL MODELS CONSISTENT!")
            print("   Training model and scanner model produce identical results.")
            print("   Safe to use in production.")
        else:
            print("⚠️  INCONSISTENCY DETECTED!")
            print("   Some images show differences between training and scanner models.")
            print("   Review individual test results above.")
        
        print()
        print("Detailed Results:")
        print("-"*70)
        print(f"{'Image':<20} {'Training':<12} {'Scanner':<12} {'Agreement':<12} {'Status'}")
        print("-"*70)
        for r in results:
            status = "✅ OK" if r['consistent'] else "⚠️  DIFF"
            print(f"{r['image']:<20} {r['training_pixels']:<12,} {r['scanner_pixels']:<12,} {r['agreement']:<12.4f} {status}")
        print("-"*70)
    else:
        print("❌ No images were successfully processed")
    
    print()
    print("="*70)


if __name__ == "__main__":
    test_model_consistency()
