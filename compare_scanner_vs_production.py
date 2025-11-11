"""
Compare scanner output vs production test output
Shows side-by-side comparison and calculates differences
"""
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def compare_masks(scanner_mask_path, production_mask_path, output_path):
    """Compare two masks and visualize differences"""
    
    # Load masks
    scanner_img = Image.open(scanner_mask_path)
    production_img = Image.open(production_mask_path)
    
    # The visualizations are 3-panel, we need to extract just the mask panels
    # Let's extract the middle panel (the prediction mask)
    scanner_arr = np.array(scanner_img)
    production_arr = np.array(production_img)
    
    # These are multi-panel images, let's just load the raw masks instead
    # We need to get the actual mask data from the scanner result
    return None

def compare_raw_outputs():
    """Compare the actual mask outputs directly"""
    
    print("="*70)
    print("SCANNER vs PRODUCTION COMPARISON")
    print("="*70)
    print()
    
    # We need to extract masks from the test results
    # Let me create a test that generates comparable outputs
    
    import sys
    from pathlib import Path
    
    # Add scanner backend to path
    scanner_backend = Path(__file__).parent.parent / "OneNote-Whiteboard-Scanner" / "local-ai-backend"
    sys.path.insert(0, str(scanner_backend))
    
    from ai.hybrid_extractor import HybridStrokeExtractor
    import json
    
    # Load scanner config
    config_path = scanner_backend / "config_hybrid.json"
    with open(config_path) as f:
        CONFIG = json.load(f)
    
    # Initialize scanner
    print("Initializing scanner...")
    scanner = HybridStrokeExtractor()
    print()
    
    # Load production test model
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    from torchvision import transforms
    
    print("Loading production model...")
    device = torch.device('cpu')
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
    state_dict = torch.load("models/whiteboard_seg_best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print()
    
    # Setup production test preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test on first image
    test_image_path = Path("dataset/images/image_1.png")
    print(f"Testing on: {test_image_path.name}")
    print()
    
    # Load image
    img_pil = Image.open(test_image_path).convert("RGB")
    img_cv = cv2.imread(str(test_image_path))
    
    print(f"Image size: {img_pil.size[0]}×{img_pil.size[1]}")
    print()
    
    # ==========================================
    # SCANNER OUTPUT
    # ==========================================
    print("1. SCANNER OUTPUT")
    print("-" * 70)
    result = scanner.process_image(img_cv)
    scanner_mask = result['mask']
    
    print(f"   Shape: {scanner_mask.shape}")
    print(f"   Stroke pixels: {np.count_nonzero(scanner_mask):,}")
    print(f"   Min/Max: {scanner_mask.min()}/{scanner_mask.max()}")
    print(f"   Dtype: {scanner_mask.dtype}")
    print()
    
    # ==========================================
    # PRODUCTION TEST OUTPUT (Tiled inference - matches test_model_production.py)
    # ==========================================
    print("2. PRODUCTION TEST OUTPUT (Tiled inference with 50% overlap)")
    print("-" * 70)
    
    # Use the same tiled inference as test_model_production.py
    from test_model_production import predict_smooth_tiled
    
    production_mask = predict_smooth_tiled(
        model=model,
        image_path=test_image_path,
        tile_size=(768, 1024),  # H×W
        overlap=0.5
    )
    
    # Convert to 0-255 range
    production_mask = (production_mask * 255).astype(np.uint8)
    
    print(f"   Shape: {production_mask.shape}")
    print(f"   Stroke pixels: {np.count_nonzero(production_mask):,}")
    print(f"   Min/Max: {production_mask.min()}/{production_mask.max()}")
    print(f"   Dtype: {production_mask.dtype}")
    print()
    
    # ==========================================
    # COMPARISON
    # ==========================================
    print("="*70)
    print("COMPARISON")
    print("="*70)
    
    # Ensure both are binary
    scanner_binary = (scanner_mask > 127).astype(np.uint8)
    production_binary = (production_mask > 127).astype(np.uint8)
    
    # Calculate differences
    total_pixels = scanner_binary.size
    
    # Pixels that match
    matching = (scanner_binary == production_binary).sum()
    matching_pct = matching / total_pixels * 100
    
    # Pixels in scanner but not production (false positives)
    scanner_only = ((scanner_binary == 1) & (production_binary == 0)).sum()
    scanner_only_pct = scanner_only / total_pixels * 100
    
    # Pixels in production but not scanner (false negatives)
    production_only = ((scanner_binary == 0) & (production_binary == 1)).sum()
    production_only_pct = production_only / total_pixels * 100
    
    print(f"Matching pixels:     {matching:,} ({matching_pct:.2f}%)")
    print(f"Scanner only (FP):   {scanner_only:,} ({scanner_only_pct:.2f}%)")
    print(f"Production only (FN): {production_only:,} ({production_only_pct:.2f}%)")
    print()
    
    # Visual quality metrics
    # Check for jaggedness using edge detection
    scanner_edges = cv2.Canny(scanner_binary * 255, 100, 200)
    production_edges = cv2.Canny(production_binary * 255, 100, 200)
    
    scanner_edge_pixels = np.count_nonzero(scanner_edges)
    production_edge_pixels = np.count_nonzero(production_edges)
    
    print(f"Edge pixels (jaggedness indicator):")
    print(f"  Scanner:    {scanner_edge_pixels:,}")
    print(f"  Production: {production_edge_pixels:,}")
    print(f"  Ratio:      {scanner_edge_pixels / production_edge_pixels:.2f}x")
    print()
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Resize for display
    display_scale = min(1.0, 2000 / max(img_pil.size))
    display_size = (int(img_pil.size[0] * display_scale), int(img_pil.size[1] * display_scale))
    
    img_display = img_pil.resize(display_size, Image.LANCZOS)
    scanner_display = cv2.resize(scanner_binary, display_size, interpolation=cv2.INTER_NEAREST)
    production_display = cv2.resize(production_binary, display_size, interpolation=cv2.INTER_NEAREST)
    
    # Row 1: Original + Masks
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f"Original Image\n{img_pil.size[0]}×{img_pil.size[1]}", fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(scanner_display, cmap='gray')
    axes[0, 1].set_title(f"Scanner Output\n{np.count_nonzero(scanner_binary):,} pixels", fontsize=12, weight='bold', color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(production_display, cmap='gray')
    axes[0, 2].set_title(f"Production Test Output\n{np.count_nonzero(production_binary):,} pixels", fontsize=12, weight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays + Difference
    overlay_scanner = np.array(img_display).copy()
    overlay_scanner[scanner_display > 0] = [255, 0, 0]  # Red
    
    overlay_production = np.array(img_display).copy()
    overlay_production[production_display > 0] = [0, 255, 0]  # Green
    
    # Difference map (red=scanner only, green=production only, white=both)
    diff_map = np.zeros((*display_size[::-1], 3), dtype=np.uint8)
    scanner_only_map = (scanner_display > production_display)
    production_only_map = (production_display > scanner_display)
    both_map = (scanner_display > 0) & (production_display > 0)
    
    diff_map[scanner_only_map] = [255, 0, 0]      # Red: Scanner only (noise/artifacts)
    diff_map[production_only_map] = [0, 255, 0]   # Green: Production only (missed strokes)
    diff_map[both_map] = [255, 255, 255]          # White: Agreement
    
    axes[1, 0].imshow(overlay_scanner)
    axes[1, 0].set_title("Scanner Overlay (Red)", fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_production)
    axes[1, 1].set_title("Production Overlay (Green)", fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_map)
    axes[1, 2].set_title(f"Difference Map\nRed=Scanner only | Green=Prod only | White=Both\nAgreement: {matching_pct:.1f}%", 
                        fontsize=11, weight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path("comparison_scanner_vs_production.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison to: {output_path}")
    plt.close()
    
    # Also save individual masks for closer inspection
    cv2.imwrite("scanner_mask_raw.png", scanner_mask)
    cv2.imwrite("production_mask_raw.png", production_mask)
    print(f"✅ Saved raw masks: scanner_mask_raw.png, production_mask_raw.png")

if __name__ == "__main__":
    compare_raw_outputs()
