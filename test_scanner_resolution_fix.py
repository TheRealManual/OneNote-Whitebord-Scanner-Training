"""
Test: Verify that changing scanner input_size to 1536x1536 improves consistency
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

# Load test image
test_img_path = Path("dataset/images/image_14.png")
img_pil = Image.open(test_img_path).convert("RGB")
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

print(f"Testing on: {test_img_path.name} ({img_pil.size[0]}x{img_pil.size[1]})")
print()

# === Test with ORIGINAL scanner (1024x768) ===
print("="*70)
print("TEST 1: Original Scanner (1024x768)")
print("="*70)
scanner_old = TileSegmentation()
print(f"Input size: {scanner_old.input_size}")

mask_old = scanner_old.infer_full_image_smooth(img_cv)
pixels_old = np.count_nonzero(mask_old)
print(f"Stroke pixels: {pixels_old:,}")
print()

# === Test with UPDATED scanner (1536x1536) ===
print("="*70)
print("TEST 2: Updated Scanner (1536x1536)")
print("="*70)

# Manually override input_size for testing
scanner_new = TileSegmentation()
scanner_new.input_size = (1536, 1536)  # Override to match Model 4 training
print(f"Input size: {scanner_new.input_size}")

mask_new = scanner_new.infer_full_image_smooth(img_cv)
pixels_new = np.count_nonzero(mask_new)
print(f"Stroke pixels: {pixels_new:,}")
print()

# === Compare ===
print("="*70)
print("COMPARISON")
print("="*70)
print(f"Pixel difference: {pixels_new - pixels_old:,} ({((pixels_new - pixels_old) / pixels_old * 100):.2f}%)")
print()
print(f"Expected: ~30% MORE pixels with 1536x1536 (matches training model)")
print(f"Actual change: {((pixels_new - pixels_old) / pixels_old * 100):.2f}%")
