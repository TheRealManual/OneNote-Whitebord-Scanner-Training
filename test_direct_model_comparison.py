"""
Direct comparison: Training .pt model vs Scanner .pts model (NO tile segmentation)
This isolates whether the models themselves are equivalent
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Load TRAINING model (.pt checkpoint format)
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

training_model = deeplabv3_mobilenet_v3_large(num_classes=2)
checkpoint = torch.load("backup/4/whiteboard_seg_best.pt", map_location='cpu', weights_only=False)
training_model.load_state_dict(checkpoint)  # Direct state dict, not wrapped
training_model.eval()
print("✓ Training model loaded (.pt checkpoint)")

# Load SCANNER model (.pts format)
scanner_model = torch.jit.load("C:/Users/thelo.NICKS_MAIN_PC/OneDrive/Desktop/Repos/OneNote-Whiteboard-Scanner/local-ai-backend/models/whiteboard_seg.pts", map_location='cpu')
scanner_model.eval()
print("✓ Scanner model loaded (.pts)")
print()

# Load test image
test_img = Image.open("dataset/images/image_14.png").convert("RGB")
print(f"Test image: {test_img.size[0]}×{test_img.size[1]}")
print()

# Preprocessing (ImageNet normalization)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Test at 1536×1536 (Model 4's native resolution)
print("="*70)
print("Testing at 1536×1536 (Model 4 native resolution)")
print("="*70)

img_resized = test_img.resize((1536, 1536), Image.LANCZOS)
img_tensor = transforms.ToTensor()(img_resized)
img_normalized = normalize(img_tensor).unsqueeze(0)

# Training model inference
with torch.no_grad():
    out_train = training_model(img_normalized)
    if isinstance(out_train, dict):
        out_train = out_train['out']
    prob_train = torch.softmax(out_train, dim=1)[0, 1]
    mask_train = (prob_train > 0.5).cpu().numpy().astype(np.uint8)

pixels_train = np.count_nonzero(mask_train)
print(f"Training model output: {mask_train.shape[1]}×{mask_train.shape[0]}, pixels: {pixels_train:,}")

# Scanner model inference
with torch.no_grad():
    out_scanner = scanner_model(img_normalized)
    if isinstance(out_scanner, dict):
        out_scanner = out_scanner['out']
    prob_scanner = torch.softmax(out_scanner, dim=1)[0, 1]
    mask_scanner = (prob_scanner > 0.5).cpu().numpy().astype(np.uint8)

pixels_scanner = np.count_nonzero(mask_scanner)
print(f"Scanner model output: {mask_scanner.shape[1]}×{mask_scanner.shape[0]}, pixels: {pixels_scanner:,}")

# Compare
pixel_diff = abs(pixels_train - pixels_scanner)
agreement = np.sum(mask_train == mask_scanner) / mask_train.size

print()
print(f"Pixel agreement: {agreement:.6f} ({agreement*100:.4f}%)")
print(f"Pixel count difference: {pixel_diff:,}")

if agreement > 0.9999:
    print("✓ Models are IDENTICAL")
else:
    print(f"⚠️ Models differ by {(1-agreement)*100:.4f}%")
