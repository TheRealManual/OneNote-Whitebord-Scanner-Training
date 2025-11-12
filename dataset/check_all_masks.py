"""
Check all masks in the dataset for proper formatting
"""
from PIL import Image
import numpy as np
import os

masks_dir = './masks'
images_dir = './images'

# Get all mask files
mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

print('=' * 70)
print('MASK VALIDATION REPORT')
print('=' * 70)

# Check for paired files
image_stems = {os.path.splitext(f)[0] for f in image_files}
mask_stems = {os.path.splitext(f)[0] for f in mask_files}

print(f'\nDataset Overview:')
print(f'  Total images: {len(image_files)}')
print(f'  Total masks: {len(mask_files)}')
print(f'  Paired: {len(image_stems & mask_stems)}')

unpaired_images = image_stems - mask_stems
unpaired_masks = mask_stems - image_stems

if unpaired_images:
    print(f'\n❌ Images without masks: {sorted(unpaired_images)}')
if unpaired_masks:
    print(f'\n❌ Masks without images: {sorted(unpaired_masks)}')

# Check each mask
print('\n' + '=' * 70)
print('Individual Mask Analysis:')
print('=' * 70)

all_good = True
issues = []

for mask_file in mask_files:
    mask_path = os.path.join(masks_dir, mask_file)
    m = Image.open(mask_path)
    a = np.array(m)
    
    # Get corresponding image
    stem = os.path.splitext(mask_file)[0]
    img_files = [f for f in image_files if os.path.splitext(f)[0] == stem]
    
    if img_files:
        img = Image.open(os.path.join(images_dir, img_files[0]))
        img_size = img.size
    else:
        img_size = None
    
    unique_vals = np.unique(a)
    stroke_percent = (a > 127).sum() / a.size * 100
    
    # Check for issues
    mask_issues = []
    
    if m.mode != 'L':
        mask_issues.append(f"Mode={m.mode} (should be L)")
        all_good = False
    
    if not (len(unique_vals) == 2 and set(unique_vals) == {0, 255}):
        mask_issues.append(f"{len(unique_vals)} unique values: {unique_vals[:10]}")
        all_good = False
    
    if img_size and (m.width, m.height) != img_size:
        mask_issues.append(f"Size mismatch: mask {m.size} vs image {img_size}")
        all_good = False
    
    if stroke_percent < 0.1:
        mask_issues.append(f"Very few strokes: {stroke_percent:.2f}%")
    
    if stroke_percent > 50:
        mask_issues.append(f"Too many strokes: {stroke_percent:.1f}% (might be inverted?)")
        all_good = False
    
    # Print result
    status = '✓' if not mask_issues else '❌'
    print(f'\n{status} {mask_file}:')
    print(f'    Size: {m.width}×{m.height}')
    print(f'    Mode: {m.mode}')
    print(f'    Unique values: {unique_vals}')
    print(f'    Stroke coverage: {stroke_percent:.2f}%')
    
    if img_size:
        print(f'    Image size: {img_size[0]}×{img_size[1]} {"✓" if (m.width, m.height) == img_size else "❌"}')
    
    if mask_issues:
        for issue in mask_issues:
            print(f'    ⚠️  {issue}')
        issues.append((mask_file, mask_issues))

# Summary
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)

if all_good and not unpaired_images and not unpaired_masks:
    print('✅ ALL MASKS ARE PERFECT!')
    print('   - All masks are binary (0 and 255 only)')
    print('   - All masks are grayscale (mode L)')
    print('   - All masks match image sizes')
    print('   - All images have corresponding masks')
else:
    print(f'❌ Found {len(issues)} masks with issues:')
    for mask_file, mask_issues in issues:
        print(f'\n  {mask_file}:')
        for issue in mask_issues:
            print(f'    - {issue}')
    
    if unpaired_images:
        print(f'\n  Missing masks for: {sorted(unpaired_images)}')
    if unpaired_masks:
        print(f'\n  Missing images for: {sorted(unpaired_masks)}')

print('\n' + '=' * 70)
