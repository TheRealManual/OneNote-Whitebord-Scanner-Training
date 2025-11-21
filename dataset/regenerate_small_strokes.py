"""
Regenerate augmented images for small stroke photos (images 22-37)
Uses gentler augmentation settings optimized for fine details
"""

from generate_augmented_dataset import AugmentationGenerator
from PIL import Image
from pathlib import Path
import shutil

def regenerate_small_stroke_images():
    """Regenerate augmented images for image_22 through image_37"""
    
    root = Path(__file__).parent
    images_dir = root / "images"
    masks_dir = root / "masks"
    
    # Images 22-37 (small stroke photos)
    image_numbers = range(22, 38)
    
    augmenter = AugmentationGenerator()
    variations_per_image = 10
    
    print(f"Regenerating augmented images for images 22-37")
    print(f"Gentler settings optimized for small strokes")
    print(f"Generating {variations_per_image} variations per image\n")
    
    total_generated = 0
    
    for img_num in image_numbers:
        img_name = f"image_{img_num}.png"
        img_path = images_dir / img_name
        
        if not img_path.exists():
            print(f"⚠️  {img_name} not found, skipping")
            continue
        
        print(f"Processing {img_name}...", end=' ')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Generate variations with gentler settings
        for var_idx in range(variations_per_image):
            # 30% strong, 70% weak (as before)
            use_strong = var_idx < 3
            
            augmented_img = augmenter.generate_variation(
                img,
                use_strong=use_strong,
                variation_id=var_idx
            )
            
            # Save augmented image
            output_name = f"image_{img_num}_aug{var_idx:02d}.png"
            augmented_img.save(images_dir / output_name)
            
            total_generated += 1
        
        print(f"✓ Generated {variations_per_image} variations")
    
    print(f"\n{'='*60}")
    print(f"REGENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Images regenerated: 22-37 (16 images)")
    print(f"Total augmented images: {total_generated}")
    print(f"Optimized for small stroke preservation:")
    print(f"  - Reduced blur: 0.2-0.8 (was 0.3-1.5)")
    print(f"  - Reduced noise: 2-6 sigma (was 3-12)")
    print(f"  - Reduced glare: 0.15-0.4 strength (was 0.2-0.6)")
    print(f"  - Reduced shadow: 0.1-0.25 strength (was 0.15-0.4)")
    print(f"{'='*60}")


if __name__ == '__main__':
    regenerate_small_stroke_images()
