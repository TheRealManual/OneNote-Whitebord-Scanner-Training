"""
Training Script for Whiteboard Segmentation Model
DeepLabV3-MobileNetV3 Large for binary stroke segmentation

PRODUCTION SETTINGS (aligned with scanner):
- Resolution: 768×1024 (H×W) - matches scanner tile size
- Classes: 2 (background=0, stroke=1)
- Loss: Combined Dice (60%) + Focal (40%)
- Augmentation: Rotation, brightness, contrast, blur, sharpening
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Scanner uses these same settings for inference with 50% overlapping tiles.

Usage:
    python train_segmentation.py --epochs 100 --batch-size 1 --lr 0.0001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
from pathlib import Path
import argparse
import json


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation - better than CrossEntropy for imbalanced data"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Get probability for stroke class (class 1)
        pred_stroke = predictions[:, 1, :, :]
        
        # Flatten
        pred_flat = pred_stroke.contiguous().view(-1)
        target_flat = targets.contiguous().view(-1).float()
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples, better for imbalanced classes"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combination of Dice Loss and Focal Loss for best results"""
    def __init__(self, dice_weight=0.6, focal_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, predictions, targets):
        return self.dice_weight * self.dice(predictions, targets) + \
               self.focal_weight * self.focal(predictions, targets)


class WhiteboardDataset(Dataset):
    """Dataset for whiteboard images with segmentation masks"""
    
    def __init__(self, root_dir, train=True, augment=True, img_size=(768, 1024)):
        """
        Args:
            root_dir: Path to dataset with images/ and masks/ folders
            train: If True, use training split
            augment: If True, apply data augmentation
            img_size: (height, width) for training - higher = better quality
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.img_size = img_size
        
        # Get all image files
        self.files = sorted([f.name for f in self.img_dir.glob("*.jpg")] + 
                           [f.name for f in self.img_dir.glob("*.png")])
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # Train/val split (80/20)
        if len(self.files) == 1:
            self.files = self.files[:1]
        else:
            split_idx = max(1, int(len(self.files) * 0.8))
            if train:
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]
        
        print(f"{'Train' if train else 'Val'} dataset: {len(self.files)} images at {img_size[1]}x{img_size[0]}")
        
        # Base transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Enhanced augmentation
        self.augment = augment and train
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_dir / self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load mask - handle augmented images
        img_filename = self.files[idx]
        
        # Check if this is an augmented image (e.g., "image_1_aug00.png")
        if "_aug" in img_filename:
            # Extract original image name (e.g., "image_1_aug00.png" -> "image_1.png")
            base_name = img_filename.split("_aug")[0]
            # Try common extensions
            mask_path = self.mask_dir / f"{base_name}.png"
            if not mask_path.exists():
                mask_path = self.mask_dir / f"{base_name}.jpg"
        else:
            # Original image - use standard mask naming
            mask_path = self.mask_dir / img_filename.replace(".jpg", ".png")
        
        mask = Image.open(mask_path).convert("L")
        
        # Resize
        img = img.resize((self.img_size[1], self.img_size[0]))
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Enhanced augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random rotation (-10 to +10 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                img = img.rotate(angle, fillcolor=(255, 255, 255))
                mask = mask.rotate(angle, fillcolor=0)
            
            # Color jitter
            img = self.color_jitter(img)
            
            # Random brightness
            if np.random.rand() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(np.random.uniform(0.7, 1.3))
            
            # Random contrast
            if np.random.rand() > 0.5:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(np.random.uniform(0.8, 1.2))
            
            # Random blur
            if np.random.rand() > 0.7:
                img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 2.0)))
            
            # Random sharpening
            if np.random.rand() > 0.7:
                img = img.filter(ImageFilter.SHARPEN)
        
        # Convert to tensors
        img_tensor = self.img_transform(img)
        mask_array = np.array(mask)
        
        # Binary: 0=background, 1=stroke
        mask_array = (mask_array > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return img_tensor, mask_tensor


def train_model(args):
    """Train DeepLabV3-MobileNetV3 on whiteboard dataset"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Detailed CUDA verification
    print("\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Using device: {device} ✓")
    else:
        print(f"Using device: {device}")
        print("⚠️  WARNING: CUDA not available - training will be VERY slow on CPU!")
        print("⚠️  If you have a GPU, check CUDA installation:")
        print("   - Ensure NVIDIA drivers are installed")
        print("   - Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("="*60 + "\n")
    
    # Enable Automatic Mixed Precision (AMP) for faster GPU training
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        print("✓ AMP enabled - using mixed precision training (2x faster)\n")
        scaler = torch.cuda.amp.GradScaler()
    else:
        if args.use_amp and not torch.cuda.is_available():
            print("⚠️  AMP requested but CUDA not available - using FP32\n")
        scaler = None
    
    # Create datasets with higher resolution
    img_size = (args.img_height, args.img_width)
    train_dataset = WhiteboardDataset(args.data_dir, train=True, augment=True, img_size=img_size)
    val_dataset = WhiteboardDataset(args.data_dir, train=False, augment=False, img_size=img_size)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Loading DeepLabV3-MobileNetV3 Large...")
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    
    # Replace classifier for binary segmentation
    model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)
    model.aux_classifier = None  # Disable aux classifier
    
    model = model.to(device)
    
    # Handle small datasets
    small_data = (len(train_dataset) < 2) or (args.batch_size < 2)
    if small_data:
        print("WARNING: tiny dataset detected; switching to eval()-mode during training")
        model.eval()
    
    # Use Combined Loss (Dice + Focal) for best accuracy
    criterion = CombinedLoss(dice_weight=0.6, focal_weight=0.4)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}
    
    for epoch in range(args.epochs):
        # Train
        if not small_data:
            model.train()
        else:
            model.eval()
        
        train_loss = 0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward pass with AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    model_output = model(imgs)
                    if isinstance(model_output, dict):
                        outputs = model_output["out"]
                    else:
                        outputs = model_output
                    loss = criterion(outputs, masks)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 training
                model_output = model(imgs)
                if isinstance(model_output, dict):
                    outputs = model_output["out"]
                else:
                    outputs = model_output
                loss = criterion(outputs, masks)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        intersection_stroke = 0
        union_stroke = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                model_output = model(imgs)
                if isinstance(model_output, dict):
                    outputs = model_output["out"]
                else:
                    outputs = model_output
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                preds = torch.argmax(outputs, dim=1)
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()
                
                # IoU for strokes
                pred_stroke = (preds == 1)
                mask_stroke = (masks == 1)
                intersection_stroke += (pred_stroke & mask_stroke).sum().item()
                union_stroke += (pred_stroke | mask_stroke).sum().item()
                
                # Precision/Recall for strokes
                true_positive += (pred_stroke & mask_stroke).sum().item()
                false_positive += (pred_stroke & ~mask_stroke).sum().item()
                false_negative += (~pred_stroke & mask_stroke).sum().item()
        
        val_loss /= len(val_loader)
        pixel_acc = correct_pixels / total_pixels
        iou = intersection_stroke / union_stroke if union_stroke > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Log metrics
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"  Pixel Acc: {pixel_acc:.4f} | IoU: {iou:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(iou)
        history['val_f1'].append(f1)
        
        # Save best model (based on F1 score for balanced metric)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_best.pt")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}, F1={f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break
        
        scheduler.step()
    
    # Save final model and history
    torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_final.pt")
    with open(args.output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model


def export_onnx(model, args):
    """Export trained model to ONNX format"""
    print("\nExporting model...")
    
    model.eval()
    dummy_input = torch.randn(1, 3, args.img_height, args.img_width)
    
    onnx_path = args.output_dir / "whiteboard_seg.onnx"
    torchscript_path = args.output_dir / "whiteboard_seg.pts"
    
    # Try ONNX export
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=14,
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True
        )
        print(f"✓ Exported to ONNX: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"ONNX export failed: {e}")
        
        # Fallback: TorchScript
        try:
            scripted_model = torch.jit.trace(model, dummy_input)
            scripted_model.save(str(torchscript_path))
            print(f"✓ Exported to TorchScript: {torchscript_path}")
            return torchscript_path
        except Exception as e2:
            print(f"TorchScript export also failed: {e2}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Train whiteboard segmentation model")
    parser.add_argument("--data-dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size (use 1 for tiny datasets)")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=2,
                       help="Number of classes (2: background, stroke)")
    parser.add_argument("--img-height", type=int, default=768,
                       help="Image height for training")
    parser.add_argument("--img-width", type=int, default=1024,
                       help="Image width for training")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                       help="Number of warmup epochs")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use Automatic Mixed Precision for faster GPU training (2x speedup)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only export existing model")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)
    
    # Train or load model
    if not args.skip_training:
        model = train_model(args)
    else:
        print("Loading existing model...")
        model = deeplabv3_mobilenet_v3_large(weights=None)
        model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)
        model.aux_classifier = None
        model.load_state_dict(torch.load(args.output_dir / "whiteboard_seg_best.pt"))
    
    # Export to ONNX
    onnx_path = export_onnx(model, args)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"PyTorch model: {args.output_dir / 'whiteboard_seg_best.pt'}")
    if onnx_path:
        print(f"Exported model: {onnx_path}")
    print("="*60)


if __name__ == "__main__":
    main()
