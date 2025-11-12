"""
Fix all corrupted masks:
1. Convert LA (with alpha) to L (grayscale)
2. Convert grayscale to binary (0 and 255 only)
3. Check if inverted (fix if needed)
"""
from PIL import Image
import numpy as np
import os

masks_dir = './masks'
backup_dir = './masks_backup'

# Create backup directory
os.makedirs(backup_dir, exist_ok=True)

# List of corrupted masks
bad_masks = [
    'image_12.png',
    'image_13.png', 
    'image_14.png',
    'image_15.png',
    'image_16.png',
    'image_17.png',
    'image_18.png',
    'image_19.png',
    'image_20.png',
    'image_21.png'
]

print('=' * 70)
print('FIXING CORRUPTED MASKS')
print('=' * 70)

for mask_file in bad_masks:
    mask_path = os.path.join(masks_dir, mask_file)
    backup_path = os.path.join(backup_dir, mask_file)
    
    print(f'\nProcessing {mask_file}...')
    
    # Backup original
    img = Image.open(mask_path)
    img.save(backup_path)
    print(f'  ✓ Backed up to {backup_dir}/')
    
    # Convert to numpy
    arr = np.array(img)
    
    # If LA mode, extract just the L channel
    if img.mode == 'LA':
        arr = arr[:, :, 0]  # Take first channel (luminance)
        print(f'  ✓ Converted LA → L (removed alpha channel)')
    
    # Check current state
    unique_before = len(np.unique(arr))
    coverage_before = (arr > 127).sum() / arr.size * 100
    
    print(f'  Before: {unique_before} unique values, {coverage_before:.1f}% coverage')
    
    # Binarize: >127 = stroke (255), <=127 = background (0)
    binary = (arr > 127).astype(np.uint8) * 255
    
    # Check if inverted (too much coverage = background is white)
    coverage_after = (binary > 127).sum() / binary.size * 100
    
    if coverage_after > 50:
        # Inverted! Flip it
        binary = 255 - binary
        coverage_after = (binary > 127).sum() / binary.size * 100
        print(f'  ✓ Inverted mask (was {100-coverage_after:.1f}% background)')
    
    print(f'  After: 2 unique values (0, 255), {coverage_after:.1f}% stroke coverage')
    
    # Save fixed mask
    fixed_img = Image.fromarray(binary, mode='L')
    fixed_img.save(mask_path)
    print(f'  ✅ Saved fixed mask')

print('\n' + '=' * 70)
print('DONE!')
print('=' * 70)
print(f'Fixed {len(bad_masks)} masks')
print(f'Originals backed up to: {backup_dir}/')
print('\nRun check_all_masks.py again to verify!')
