"""
Fix masks: Convert to binary (0 or 255 only), remove alpha, grayscale
"""

from PIL import Image
import numpy as np
from pathlib import Path

def fix_mask(mask_path):
    """Convert mask to pure binary grayscale"""
    # Load mask
    img = Image.open(mask_path)
    
    # Convert to grayscale (removes alpha if present)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy
    arr = np.array(img)
    
    # Threshold to binary: >127 = 255, <=127 = 0
    binary = np.where(arr > 127, 255, 0).astype(np.uint8)
    
    # Back to image
    fixed = Image.fromarray(binary, mode='L')
    
    # Save (overwrite original)
    fixed.save(mask_path)
    
    print(f"Fixed: {mask_path.name}")
    print(f"  Unique values: {np.unique(binary)}")
    print(f"  Stroke pixels: {(binary == 255).sum()} ({(binary==255).sum()/binary.size*100:.1f}%)")

def main():
    masks_dir = Path("dataset/masks")
    
    print("Fixing all masks...\n")
    
    for mask_path in sorted(masks_dir.glob("*.png")):
        fix_mask(mask_path)
        print()
    
    print("All masks fixed! They are now binary (0 and 255 only).")
    print("You can retrain the model now.")

if __name__ == "__main__":
    main()
