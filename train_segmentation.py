"""
Training Script for Whiteboard Segmentation Model
DeepLabV3-MobileNetV3 Large for stroke/smudge/shadow classification
Target: 3-5 MB INT8 quantized ONNX model
"""

import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import os
from pathlib import Path
import argparse


class WhiteboardDataset(Dataset):
    """Dataset for whiteboard images with segmentation masks"""
    
    def __init__(self, root_dir, train=True, augment=True):
        """
        Args:
            root_dir: Path to dataset with images/ and masks/ folders
            train: If True, use training split
            augment: If True, apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        
        # Get all image files
        self.files = sorted([f.name for f in self.img_dir.glob("*.jpg")] + 
                           [f.name for f in self.img_dir.glob("*.png")])
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # Train/val split (80/20), but ensure at least 1 in each if only 1 image total
        if len(self.files) == 1:
            # Single image - use for both train and val (just for testing)
            if train:
                self.files = self.files[:1]
            else:
                self.files = self.files[:1]
        else:
            split_idx = max(1, int(len(self.files) * 0.8))
            if train:
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]
        
        print(f"{'Train' if train else 'Val'} dataset: {len(self.files)} images")
        
        # Base transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        self.augment = augment and train
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1
            )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_dir / self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load mask (ensure single channel/grayscale)
        mask_path = self.mask_dir / self.files[idx].replace(".jpg", ".png")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale/single channel
        
        # Resize
        img = img.resize((640, 480))
        mask = mask.resize((640, 480), Image.NEAREST)
        
        # Apply augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random color jitter
            img = self.color_jitter(img)
            
            # Random blur (simulate camera blur)
            if np.random.rand() > 0.7:
                img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 2.0)))
        
        # Convert to tensors
        img_tensor = self.img_transform(img)
        mask_array = np.array(mask)
        
        # Normalize mask values to class indices (0-3)
        # Assuming mask uses pixel values: 0=bg, 85=stroke, 170=smudge, 255=shadow
        # Map to classes: 0=bg, 1=stroke, 2=smudge, 3=shadow
        mask_array = mask_array // 85  # 0->0, 85->1, 170->2, 255->3
        mask_array = np.clip(mask_array, 0, 3)  # Ensure within bounds
        
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return img_tensor, mask_tensor


def train_model(args):
    """Train DeepLabV3-MobileNetV3 on whiteboard dataset"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = WhiteboardDataset(args.data_dir, train=True, augment=True)
    val_dataset = WhiteboardDataset(args.data_dir, train=False, augment=False)
    
    # DataLoader with num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Loading DeepLabV3-MobileNetV3 Large...")
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    
    # Replace classifier head for 4 classes
    # Classes: 0=background, 1=stroke, 2=smudge, 3=shadow
    model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)
    # aux_classifier uses 10 input channels from MobileNetV3 low-level features
    model.aux_classifier[4] = torch.nn.Conv2d(10, args.num_classes, kernel_size=1)
    
    # Disable auxiliary classifier to avoid dimension issues during training
    # (DeepLabV3 returns aux outputs in training mode which complicates loss calculation)
    model.aux_classifier = None
    
    model = model.to(device)
    # Detect very small datasets / tiny batch sizes that will trigger BatchNorm errors
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    small_data = (train_len < 2) or (args.batch_size < 2)
    if small_data:
        # Running the network in eval mode prevents BatchNorm from using per-batch
        # statistics (which require batch size > 1). We keep gradients enabled so
        # optimizer steps still update weight/bias parameters (affine) of BatchNorm
        # and other layers. This is a safe fallback for tiny datasets used for
        # quick tests; for real training provide >=2 training images and batch_size>=2.
        print("WARNING: tiny dataset or batch size detected; switching model to eval()-mode during training to avoid BatchNorm errors")
        model.eval()
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train (unless we're in the small-data fallback where model was set to eval())
        if not small_data:
            model.train()
        else:
            # keep eval() active so BatchNorm uses running stats instead of per-batch stats
            model.eval()
        train_loss = 0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward - handle both dict and tensor outputs
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
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
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
                
                # Calculate pixel accuracy
                preds = torch.argmax(outputs, dim=1)
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()
        
        val_loss /= len(val_loader)
        pixel_acc = correct_pixels / total_pixels
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Pixel Acc: {pixel_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_best.pt")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_final.pt")
    print("\nTraining complete!")
    
    return model


def export_onnx(model, args):
    """Export trained model to ONNX format (or TorchScript if ONNX fails)"""
    print("\nExporting model...")
    
    model.eval()
    dummy_input = torch.randn(1, 3, 480, 640)
    
    onnx_path = args.output_dir / "whiteboard_seg.onnx"
    torchscript_path = args.output_dir / "whiteboard_seg_scripted.pt"
    
    # Try ONNX export first
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
        
        # Try to verify if onnx is available (optional)
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verified successfully")
        except ImportError:
            print("  (ONNX verification skipped - onnx package not installed)")
        
        return onnx_path
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Falling back to TorchScript export...")
        
        # Fallback: Export as TorchScript
        try:
            scripted_model = torch.jit.trace(model, dummy_input)
            scripted_model.save(str(torchscript_path))
            print(f"✓ Exported to TorchScript: {torchscript_path}")
            print("  (TorchScript can be converted to ONNX later with proper tools)")
            return torchscript_path
        except Exception as e2:
            print(f"TorchScript export also failed: {e2}")
            return None


def quantize_model(onnx_path, args):
    """Quantize ONNX model to INT8"""
    print("\nQuantizing to INT8...")
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime not installed - skipping quantization")
        return None
    
    output_path = args.output_dir / "whiteboard_seg_int8.onnx"
    
    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    
    # Check file sizes
    original_size = onnx_path.stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train whiteboard segmentation model")
    parser.add_argument("--data-dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=25,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=4,
                       help="Number of classes (4: bg, stroke, smudge, shadow)")
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
    
    # Quantize (only if ONNX export succeeded)
    if onnx_path:
        quantized_path = quantize_model(onnx_path, args)
    else:
        quantized_path = None
        print("\nSkipping quantization (ONNX export was skipped)")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    if quantized_path:
        print(f"Quantized INT8 model: {quantized_path}")
        print(f"Target size: 3-5 MB")
        print(f"Ready for deployment with DirectML/ONNX Runtime")
    else:
        print(f"PyTorch model saved: {args.output_dir / 'whiteboard_seg_best.pt'}")
        print("ONNX export skipped - install 'onnx' package to export")
    print("="*60)


if __name__ == "__main__":
    main()
