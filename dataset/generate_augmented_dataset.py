"""
Generate Augmented Dataset with Realistic Variations

Creates 10 randomized variations per image with diverse augmentations:
- 70% weak: brightness, contrast, gamma, color jitter, noise, blur
- 30% strong: glare overlays, shadow bands, lighting variations

Focuses on DIVERSITY over volume - different angles, strengths, color temps.
Masks remain unchanged (just duplicated).

Usage:
    python generate_augmented_dataset.py
    
Output: dataset/randomized/images/ and dataset/randomized/masks/
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import random
import shutil


class AugmentationGenerator:
    """Generate diverse, realistic photo augmentations"""
    
    def __init__(self):
        # Predefined glare angles for diversity
        self.glare_angles = [15, 35, 55, 75, 95, 115, 135, 155]
        # Predefined shadow angles
        self.shadow_angles = [0, 30, 60, 90, 120, 150]
        # Color temperature presets (warm to cool)
        self.color_temps = [
            (1.15, 1.0, 0.85),   # Warm (tungsten)
            (1.1, 1.0, 0.9),     # Slightly warm
            (1.0, 1.0, 1.0),     # Neutral
            (0.95, 1.0, 1.1),    # Slightly cool
            (0.85, 0.95, 1.15),  # Cool (daylight)
        ]
    
    def apply_brightness_contrast(self, img, brightness_factor=None, contrast_factor=None):
        """Random brightness and contrast adjustment"""
        if brightness_factor is None:
            brightness_factor = random.uniform(0.7, 1.3)
        if contrast_factor is None:
            contrast_factor = random.uniform(0.8, 1.2)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        return img
    
    def apply_gamma_correction(self, img, gamma=None):
        """Gamma correction for exposure variation"""
        if gamma is None:
            gamma = random.uniform(0.7, 1.4)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def apply_color_jitter(self, img, saturation_factor=None, hue_shift=None):
        """Color jitter with saturation and hue variations"""
        if saturation_factor is None:
            saturation_factor = random.uniform(0.8, 1.3)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        
        # Hue shift
        if hue_shift is None:
            hue_shift = random.randint(-15, 15)
        
        if hue_shift != 0:
            img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
            img = Image.fromarray(cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        
        return img
    
    def apply_gaussian_blur(self, img, radius=None):
        """Gaussian blur for focus variation"""
        if radius is None:
            radius = random.uniform(0.3, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def apply_gaussian_noise(self, img, sigma=None):
        """Add realistic camera noise"""
        if sigma is None:
            sigma = random.uniform(3, 12)
        
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, sigma, img_array.shape)
        img_array = img_array + noise
        img_array = img_array.clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def apply_glare_overlay(self, img, angle=None, strength=None, position=None):
        """Realistic glare/reflection overlay"""
        if angle is None:
            angle = random.choice(self.glare_angles)
        if strength is None:
            strength = random.uniform(0.2, 0.6)
        
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create gradient glare
        if position is None:
            position = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
        
        # Position mapping
        positions = {
            'top-left': (w * 0.25, h * 0.25),
            'top-right': (w * 0.75, h * 0.25),
            'bottom-left': (w * 0.25, h * 0.75),
            'bottom-right': (w * 0.75, h * 0.75),
            'center': (w * 0.5, h * 0.5),
        }
        
        center_x, center_y = positions[position]
        
        # Create radial gradient
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        
        # Directional gradient based on angle
        angle_rad = np.radians(angle)
        directional = (x - center_x) * np.cos(angle_rad) + (y - center_y) * np.sin(angle_rad)
        directional = (directional - directional.min()) / (directional.max() - directional.min() + 1e-6)
        
        # Combine radial and directional
        glare_mask = (1 - dist / max_dist) * 0.7 + directional * 0.3
        glare_mask = np.clip(glare_mask, 0, 1)
        glare_mask = np.power(glare_mask, 2.5)  # Sharper falloff
        
        # Apply glare
        glare_mask = glare_mask[:, :, np.newaxis] * strength * 255
        img_array = img_array + glare_mask
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def apply_shadow_overlay(self, img, angle=None, strength=None, num_bands=None):
        """Realistic shadow bands (from objects, hands, etc.)"""
        if angle is None:
            angle = random.choice(self.shadow_angles)
        if strength is None:
            strength = random.uniform(0.15, 0.4)
        if num_bands is None:
            num_bands = random.randint(1, 3)
        
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        angle_rad = np.radians(angle)
        
        for _ in range(num_bands):
            # Random band position
            offset = random.uniform(-0.3, 0.3) * max(h, w)
            band_width = random.uniform(0.15, 0.35) * max(h, w)
            
            y, x = np.ogrid[:h, :w]
            # Rotated coordinate
            rotated = x * np.cos(angle_rad) + y * np.sin(angle_rad) + offset
            
            # Soft-edged band
            band_edge = 50  # Softness
            band_mask = np.clip((rotated - (max(h, w) * 0.3)) / band_edge, 0, 1)
            band_mask *= np.clip(((max(h, w) * 0.3 + band_width) - rotated) / band_edge, 0, 1)
            
            shadow_mask *= (1 - band_mask * strength)
        
        # Apply shadow
        shadow_mask = shadow_mask[:, :, np.newaxis]
        img_array = img_array * shadow_mask
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def apply_color_temperature(self, img, temp_preset=None):
        """Apply color temperature shift"""
        if temp_preset is None:
            temp_preset = random.choice(self.color_temps)
        
        r_factor, g_factor, b_factor = temp_preset
        
        img_array = np.array(img).astype(np.float32)
        img_array[:, :, 0] *= r_factor
        img_array[:, :, 1] *= g_factor
        img_array[:, :, 2] *= b_factor
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def generate_variation(self, img, use_strong=False, variation_id=0):
        """
        Generate a single variation with diverse augmentations
        
        Args:
            img: PIL Image
            use_strong: If True, apply strong augmentations (glare/shadow)
            variation_id: ID for deterministic diversity
        """
        img = img.copy()
        
        # Seed for reproducible diversity
        random.seed(hash((id(img), variation_id)))
        
        if use_strong:
            # STRONG augmentation (30%)
            aug_type = random.choice(['glare', 'shadow', 'glare+shadow'])
            
            if aug_type == 'glare':
                # Pure glare
                img = self.apply_glare_overlay(img)
                # Light color temp shift
                if random.random() > 0.5:
                    img = self.apply_color_temperature(img)
            
            elif aug_type == 'shadow':
                # Shadow bands
                img = self.apply_shadow_overlay(img)
                # Compensate with brightness
                img = self.apply_brightness_contrast(img, brightness_factor=random.uniform(1.1, 1.3))
            
            else:
                # Combined glare + shadow (challenging)
                img = self.apply_glare_overlay(img, strength=random.uniform(0.2, 0.4))
                img = self.apply_shadow_overlay(img, strength=random.uniform(0.1, 0.25))
            
            # Add some weak augmentation too
            if random.random() > 0.5:
                img = self.apply_gaussian_blur(img, radius=random.uniform(0.3, 0.8))
        
        else:
            # WEAK augmentation (70%)
            # Always apply some basic adjustments
            img = self.apply_brightness_contrast(img)
            
            # Random selection of 2-4 weak augmentations
            num_augs = random.randint(2, 4)
            weak_augs = random.sample([
                'gamma', 'color_jitter', 'blur', 'noise', 'color_temp'
            ], num_augs)
            
            for aug in weak_augs:
                if aug == 'gamma':
                    img = self.apply_gamma_correction(img)
                elif aug == 'color_jitter':
                    img = self.apply_color_jitter(img)
                elif aug == 'blur':
                    img = self.apply_gaussian_blur(img)
                elif aug == 'noise':
                    img = self.apply_gaussian_noise(img)
                elif aug == 'color_temp':
                    img = self.apply_color_temperature(img)
        
        # Reset random seed
        random.seed()
        
        return img


def generate_augmented_dataset(
    source_images_dir='images',
    source_masks_dir='masks',
    output_dir='randomized',
    variations_per_image=10,
    strong_ratio=0.3
):
    """
    Generate augmented dataset with diverse variations
    
    Args:
        source_images_dir: Path to original images
        source_masks_dir: Path to original masks
        output_dir: Output directory for augmented data
        variations_per_image: Number of variations per image (default: 10)
        strong_ratio: Ratio of strong augmentations (default: 0.3 = 30%)
    """
    # Setup paths
    root = Path(__file__).parent
    images_dir = root / source_images_dir
    masks_dir = root / source_masks_dir
    
    output_images_dir = root / output_dir / 'images'
    output_masks_dir = root / output_dir / 'masks'
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Generating {variations_per_image} variations per image ({int(strong_ratio*100)}% strong augmentations)")
    print(f"Output: {output_dir}/\n")
    
    augmenter = AugmentationGenerator()
    
    # Determine strong/weak split
    num_strong = int(variations_per_image * strong_ratio)
    num_weak = variations_per_image - num_strong
    
    total_generated = 0
    
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"[{img_idx}/{len(image_files)}] Processing {img_path.name}...", end=' ')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Find corresponding mask
        mask_path = masks_dir / img_path.name.replace('.jpg', '.png')
        if not mask_path.exists():
            mask_path = masks_dir / img_path.name
        
        if not mask_path.exists():
            print(f"⚠️  Mask not found for {img_path.name}, skipping")
            continue
        
        # Generate variations
        for var_idx in range(variations_per_image):
            # Determine if this should be strong or weak
            use_strong = var_idx < num_strong
            
            # Generate variation
            augmented_img = augmenter.generate_variation(
                img, 
                use_strong=use_strong,
                variation_id=var_idx
            )
            
            # Save augmented image
            output_name = f"{img_path.stem}_aug{var_idx:02d}{img_path.suffix}"
            augmented_img.save(output_images_dir / output_name)
            
            # Copy mask (unchanged)
            mask_output_name = f"{img_path.stem}_aug{var_idx:02d}.png"
            shutil.copy2(mask_path, output_masks_dir / mask_output_name)
            
            total_generated += 1
        
        print(f"✓ Generated {variations_per_image} variations")
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original images: {len(image_files)}")
    print(f"Generated variations: {total_generated} ({variations_per_image} per image)")
    print(f"Weak augmentations: {num_weak} per image ({100-int(strong_ratio*100)}%)")
    print(f"Strong augmentations: {num_strong} per image ({int(strong_ratio*100)}%)")
    print(f"\nOutput:")
    print(f"  Images: {output_images_dir}")
    print(f"  Masks:  {output_masks_dir}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Train with: python train_segmentation.py --data-dir {output_dir}")
    print(f"2. Compare val IoU/F1 with different seeds")
    print(f"3. Stop when metrics plateau across runs")


if __name__ == '__main__':
    generate_augmented_dataset(
        source_images_dir='images',
        source_masks_dir='masks',
        output_dir='randomized',
        variations_per_image=10,
        strong_ratio=0.3
    )
