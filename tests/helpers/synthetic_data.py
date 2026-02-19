"""
Synthetic data generators for tests.

All tests use these helpers instead of depending on the real private dataset.
Generates images and masks matching the project naming conventions:
  - image_22.png, image_22_aug00.png ... image_22_aug09.png
  - 1px thin-stroke masks, thick-stroke masks, empty masks
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_rgb_image(width: int = 64, height: int = 64, color=(255, 255, 255)) -> Image.Image:
    """Create a solid-color RGB image."""
    return Image.new("RGB", (width, height), color)


def create_binary_mask(width: int = 64, height: int = 64, fill: int = 0) -> Image.Image:
    """Create a single-channel binary mask (0/255)."""
    return Image.new("L", (width, height), fill)


def create_thin_stroke_mask(width: int = 64, height: int = 64) -> Image.Image:
    """Create a mask with thin 1px lines and small dots."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # horizontal thin line
    draw.line([(5, height // 2), (width - 5, height // 2)], fill=255, width=1)
    # vertical thin line
    draw.line([(width // 2, 5), (width // 2, height - 5)], fill=255, width=1)
    # a few dots
    for x in range(10, width - 10, 8):
        draw.point((x, height // 4), fill=255)
    return mask


def create_thick_stroke_mask(width: int = 64, height: int = 64) -> Image.Image:
    """Create a mask with thick strokes (3-5px wide)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.line([(5, height // 3), (width - 5, height // 3)], fill=255, width=5)
    draw.line([(5, 2 * height // 3), (width - 5, 2 * height // 3)], fill=255, width=4)
    draw.rectangle([(width // 4, height // 4), (3 * width // 4, 3 * height // 4)], outline=255, width=3)
    return mask


def create_random_stroke_mask(width: int = 64, height: int = 64, seed: int = 42) -> Image.Image:
    """Create a mask with random strokes (~2-5% foreground)."""
    rng = np.random.RandomState(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    n_lines = rng.randint(3, 8)
    for _ in range(n_lines):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        w = rng.randint(1, 3)
        draw.line([(x1, y1), (x2, y2)], fill=255, width=w)
    return mask


def populate_synthetic_dataset(root_dir, n_originals: int = 5, n_augments: int = 3,
                                img_size: tuple = (64, 64), start_id: int = 1):
    """Create a synthetic dataset directory matching the project structure.

    Creates:
        root_dir/images/  — image_N.png, image_N_augXX.png
        root_dir/masks/   — image_N.png (one mask per original)

    Args:
        root_dir: Path to dataset root (will create images/ and masks/ subdirs)
        n_originals: Number of original images
        n_augments: Number of augmented variants per original
        img_size: (width, height) for images
        start_id: Starting image ID number

    Returns:
        dict with keys 'originals', 'augmented', 'all_images', 'masks'
    """
    root = Path(root_dir)
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    originals = []
    augmented = []
    masks = []

    for i in range(start_id, start_id + n_originals):
        base_name = f"image_{i}"

        # Create original image + mask
        img = create_rgb_image(*img_size)
        mask = create_random_stroke_mask(*img_size, seed=i)

        img_path = img_dir / f"{base_name}.png"
        mask_path = mask_dir / f"{base_name}.png"
        img.save(img_path)
        mask.save(mask_path)
        originals.append(f"{base_name}.png")
        masks.append(f"{base_name}.png")

        # Create augmented variants
        for j in range(n_augments):
            aug_name = f"{base_name}_aug{j:02d}.png"
            aug_img = create_rgb_image(*img_size, color=(250 - j * 5, 250, 250))
            aug_img.save(img_dir / aug_name)
            augmented.append(aug_name)

    return {
        'originals': originals,
        'augmented': augmented,
        'all_images': originals + augmented,
        'masks': masks,
    }


def populate_thin_and_thick_dataset(root_dir, img_size: tuple = (64, 64)):
    """Create a dataset with explicit thin-stroke and thick-stroke images.

    Creates image_22, image_24, image_27 as thin-stroke examples,
    and image_3, image_13 as thick-stroke examples.
    """
    root = Path(root_dir)
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    thin_ids = [22, 24, 27]
    thick_ids = [3, 13]

    for img_id in thin_ids:
        name = f"image_{img_id}"
        create_rgb_image(*img_size).save(img_dir / f"{name}.png")
        create_thin_stroke_mask(*img_size).save(mask_dir / f"{name}.png")

    for img_id in thick_ids:
        name = f"image_{img_id}"
        create_rgb_image(*img_size).save(img_dir / f"{name}.png")
        create_thick_stroke_mask(*img_size).save(mask_dir / f"{name}.png")

    return {
        'thin_ids': [f"image_{i}" for i in thin_ids],
        'thick_ids': [f"image_{i}" for i in thick_ids],
    }
