"""
Test Scanner with Production Model
Verifies the scanner processes images correctly with smooth tiled inference
Outputs visualization like test_model_production.py for comparison
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add scanner backend to path
scanner_backend = Path(__file__).parent.parent / "OneNote-Whiteboard-Scanner" / "local-ai-backend"
sys.path.insert(0, str(scanner_backend))

from ai.hybrid_extractor import HybridStrokeExtractor
import json

# Load config
config_path = scanner_backend / "config_hybrid.json"
with open(config_path) as f:
    CONFIG = json.load(f)


def test_scanner_on_training_images():
    """Test scanner on the same images used for training/testing"""
    
    print("\n" + "="*70)
    print("SCANNER PRODUCTION TEST")
    print("="*70)
    print("Testing scanner's hybrid extractor with smooth tiled inference")
    print("="*70 + "\n")
    
    # Initialize extractor
    print("Initializing HybridStrokeExtractor...")
    extractor = HybridStrokeExtractor()
    print(f"✓ Extractor initialized")
    print(f"  Tile segmentation enabled: {extractor.tile_seg and extractor.tile_seg.enabled}")
    if extractor.tile_seg and extractor.tile_seg.enabled:
        print(f"  Tile size: {extractor.tile_seg.input_size}")
        print(f"  Model type: {extractor.tile_seg.model_type}")
    print()
    
    # Get test images from training dataset
    training_dataset = Path(__file__).parent / "dataset" / "images"
    if not training_dataset.exists():
        print(f"❌ Training dataset not found at {training_dataset}")
        return
    
    images = sorted(list(training_dataset.glob("*.png")) + list(training_dataset.glob("*.jpg")))
    print(f"Found {len(images)} test images\n")
    
    # Create output directory
    output_dir = Path(__file__).parent / "scanner_test_results"
    output_dir.mkdir(exist_ok=True)
    
    # Test configuration
    # Config is already loaded from config_hybrid.json with tile_overlap: 0.5
    print(f"Configuration:")
    print(f"  use_tile_segmentation: {CONFIG['stroke_extract']['use_tile_segmentation']}")
    print(f"  tile_overlap: {CONFIG['stroke_extract']['tile_overlap']}")
    print(f"  use_classical: {CONFIG['stroke_extract']['use_classical']}")
    print()
    
    # Process each image
    for idx, img_path in enumerate(images):
        print(f"{'='*70}")
        print(f"Processing {idx+1}/{len(images)}: {img_path.name}")
        print(f"{'='*70}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to load image")
            continue
        
        print(f"  Image size: {img.shape[1]}×{img.shape[0]}")
        
        # Process with scanner (uses config from config_hybrid.json)
        try:
            result = extractor.process_image(img)
            
            print(f"  ✓ Processing complete")
            print(f"  Strokes detected: {result.get('stroke_count', 'N/A')}")
            print(f"  Processing time: {result.get('processing_time_ms', 'N/A')} ms")
            
            # Extract mask from result
            if 'debug_mask' in result:
                mask = result['debug_mask']
            elif 'mask' in result:
                mask = result['mask']
            else:
                print(f"  ⚠ No mask in result, keys: {list(result.keys())}")
                continue
            
            # Visualize
            visualize_scanner_result(img_path, mask, output_dir, result)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"✅ Results saved to: {output_dir}")


def visualize_scanner_result(img_path, mask, output_dir, result):
    """Create visualization similar to test_model_production.py"""
    
    # Load original image
    img = Image.open(img_path).convert("RGB")
    
    # Resize for display
    display_scale = min(1.0, 2000 / max(img.size))
    display_size = (int(img.size[0] * display_scale), int(img.size[1] * display_scale))
    
    img_display = img.resize(display_size, Image.LANCZOS)
    
    # Resize mask for display
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray(mask)
    else:
        mask_pil = mask
    
    mask_display = mask_pil.resize(display_size, Image.NEAREST)
    mask_arr = np.array(mask_display)
    
    # Normalize mask to 0-1
    if mask_arr.max() > 1:
        mask_binary = (mask_arr > 127).astype(np.uint8)
    else:
        mask_binary = mask_arr.astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title(f"Original Image\n{img.size[0]}×{img.size[1]}")
    axes[0].axis("off")
    
    # Prediction mask
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title(f"Scanner Output Mask\n({np.count_nonzero(mask_binary)} stroke pixels)")
    axes[1].axis("off")
    
    # Overlay
    overlay = np.array(img_display).copy()
    overlay[mask_binary > 0] = [255, 0, 0]  # Red for strokes
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red=strokes)")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / f"{img_path.stem}_scanner.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {save_path.name}")
    plt.close()


if __name__ == "__main__":
    test_scanner_on_training_images()
