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
import cv2
import os
import random
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Path constants for two-repo workspace setup
# Public repo (this file's repo): code + scripts
# Private repo (sibling): paper + results + experiment outputs
REPO_ROOT = Path(__file__).resolve().parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"
DEFAULT_OUTPUT_DIR = str(PRIVATE_REPO / "experiments" / "default")


def seed_everything(seed):
    """Set all random seeds for reproducibility.
    
    NOTE: For full PYTHONHASHSEED reproducibility, set it at the OS level
    BEFORE launching Python (e.g., `set PYTHONHASHSEED=42` in .bat).
    Setting os.environ here is best-effort only.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # best-effort; set in .bat for full effect
    print(f"\n\U0001f331 Random seed set to {seed} (deterministic mode)")
    print(f"   cudnn.deterministic=True, cudnn.benchmark=False")
    print(f"   PYTHONHASHSEED={seed} (best-effort; set in .bat for full effect)")


class _WorkerInitFn:
    """Picklable worker_init_fn for DataLoader multiprocessing on Windows.
    
    Using a callable class instead of a closure so it can be pickled
    by Windows' 'spawn' multiprocessing start method.
    """
    def __init__(self, seed):
        self.seed = seed
    
    def __call__(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def make_worker_init_fn(seed):
    """Create a worker_init_fn that seeds each DataLoader worker deterministically."""
    return _WorkerInitFn(seed)


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


class TverskyLoss(nn.Module):
    """Tversky Loss — generalizes Dice with separate FP/FN weighting.
    
    α controls false-positive penalty, β controls false-negative penalty.
    α < β penalizes missed strokes more, helpful for thin-structure recall.
    Default α=0.3, β=0.7 encourages higher recall on thin strokes.
    When α=β=0.5, this reduces to Dice loss.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        pred_stroke = predictions[:, 1, :, :]
        
        pred_flat = pred_stroke.contiguous().view(-1)
        target_flat = targets.contiguous().view(-1).float()
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CrossEntropyLossWrapper(nn.Module):
    """Wrapper around PyTorch CrossEntropyLoss for consistent interface."""
    def __init__(self):
        super(CrossEntropyLossWrapper, self).__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        return self.ce(predictions, targets)


class CombinedLoss(nn.Module):
    """Combination of Dice Loss and Focal Loss for best results"""
    def __init__(self, dice_weight=0.6, focal_weight=0.4, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, predictions, targets):
        return self.dice_weight * self.dice(predictions, targets) + \
               self.focal_weight * self.focal(predictions, targets)


LOSS_REGISTRY = {
    'ce': 'CrossEntropyLoss',
    'dice': 'DiceLoss',
    'focal': 'FocalLoss',
    'dice_focal': 'CombinedLoss (Dice+Focal)',
    'tversky': 'TverskyLoss',
}


def build_loss(args):
    """Factory function to build loss criterion from CLI args.
    
    Returns:
        criterion: nn.Module loss function
        loss_name: str human-readable name for logging
    """
    loss_type = args.loss
    
    if loss_type == 'ce':
        criterion = CrossEntropyLossWrapper()
        loss_name = 'CrossEntropyLoss'
    elif loss_type == 'dice':
        criterion = DiceLoss()
        loss_name = 'DiceLoss'
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        loss_name = f'FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma})'
    elif loss_type == 'dice_focal':
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma
        )
        loss_name = f'CombinedLoss(dice={args.dice_weight}, focal={args.focal_weight})'
    elif loss_type == 'tversky':
        criterion = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
        loss_name = f'TverskyLoss(alpha={args.tversky_alpha}, beta={args.tversky_beta})'
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: {list(LOSS_REGISTRY.keys())}")
    
    print(f"\n📊 Loss function: {loss_name}")
    return criterion, loss_name


class ToySegModel(nn.Module):
    """Tiny segmentation model for fast CPU-only testing.
    
    A simple 3-layer conv net that outputs [B, num_classes, H, W]
    wrapped in a dict with 'out' key to match DeepLabV3 interface.
    NOT suitable for real training — only for determinism/integration tests.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        out = self.conv3(h)
        return {'out': out}


class WhiteboardDataset(Dataset):
    """Dataset for whiteboard images with segmentation masks"""
    
    def __init__(self, root_dir, train=True, augment=True, img_size=(768, 1024),
                 exclude_list=None):
        """
        Args:
            root_dir: Path to dataset with images/ and masks/ folders
            train: If True, use training split
            augment: If True, apply data augmentation
            img_size: (height, width) for training - higher = better quality
            exclude_list: Optional list of base image IDs to exclude (e.g. ['image_3', 'image_22']).
                         Originals AND augmented variants (image_X_augNN) are excluded
                         via regex base-ID matching.
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.img_size = img_size
        
        # Get all image files
        self.files = sorted([f.name for f in self.img_dir.glob("*.jpg")] + 
                           [f.name for f in self.img_dir.glob("*.png")])
        
        # Exclude test-split images and their augmented variants
        if exclude_list:
            import re
            exclude_set = set(exclude_list)
            filtered = []
            for fname in self.files:
                stem = Path(fname).stem
                # Extract base ID: "image_3_aug05" → "image_3"
                m = re.match(r'^(image_\d+)', stem)
                base_id = m.group(1) if m else stem
                if base_id not in exclude_set:
                    filtered.append(fname)
            n_excluded = len(self.files) - len(filtered)
            self.files = filtered
            if n_excluded > 0:
                print(f"  Excluded {n_excluded} images matching test-split IDs")
        
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
            
            # CRITICAL: Slight mask erosion to prevent small text merging
            # This teaches the model to keep letters separated even when close together
            if np.random.rand() > 0.6:  # 40% of the time
                mask_array = np.array(mask)
                kernel = np.ones((2, 2), np.uint8)  # Small kernel for subtle effect
                mask_array = cv2.erode(mask_array, kernel, iterations=1)
                mask = Image.fromarray(mask_array)
        
        # Convert to tensors
        img_tensor = self.img_transform(img)
        mask_array = np.array(mask)
        
        # Binary: 0=background, 1=stroke
        mask_array = (mask_array > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return img_tensor, mask_tensor


def train_model(args):
    """Train DeepLabV3-MobileNetV3 on whiteboard dataset"""
    
    # Set seeds for reproducibility if requested
    if args.seed is not None:
        seed_everything(args.seed)
    
    # Optionally enable fully deterministic algorithms
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("\u26a0\ufe0f  torch.use_deterministic_algorithms(True, warn_only=True) enabled")
        print("   Non-deterministic ops will warn instead of error.")
        print(f"   CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not set')}")
    
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
        scaler = torch.amp.GradScaler('cuda')
    else:
        if args.use_amp and not torch.cuda.is_available():
            print("⚠️  AMP requested but CUDA not available - using FP32\n")
        scaler = None
    
    # Create datasets with higher resolution
    img_size = (args.img_height, args.img_width)
    
    # Load test-split exclusion list if provided
    exclude_list = None
    if getattr(args, 'test_split_config', None):
        import json as _json
        with open(args.test_split_config) as f:
            split_cfg = _json.load(f)
        exclude_list = split_cfg.get('train_exclude', [])
        print(f"\n📋 Test-split config: {args.test_split_config}")
        print(f"   Excluding {len(exclude_list)} base IDs from training")
    
    train_dataset = WhiteboardDataset(args.data_dir, train=True, augment=True, img_size=img_size,
                                       exclude_list=exclude_list)
    val_dataset = WhiteboardDataset(args.data_dir, train=False, augment=False, img_size=img_size,
                                     exclude_list=exclude_list)
    
    # DataLoader with parallel workers and prefetching for maximum GPU utilization
    # When seed is set, add worker_init_fn and generator for full reproducibility
    loader_kwargs = dict(
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
    )
    if args.seed is not None:
        loader_kwargs['worker_init_fn'] = make_worker_init_fn(args.seed)
        loader_kwargs['generator'] = torch.Generator().manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True,  # Avoid single-sample batches that crash BatchNorm
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        **loader_kwargs
    )
    
    # Create model
    if getattr(args, 'model', 'deeplabv3') == 'toy':
        print("Loading ToySegModel (test-only)...")
        model = ToySegModel(num_classes=args.num_classes)
    else:
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
    
    # Build loss function from CLI args (supports: ce, dice, focal, dice_focal, tversky)
    criterion, loss_name = build_loss(args)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    
    import datetime
    import platform
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_iou': [], 
        'val_f1': [],
        'config': {
            # Model Architecture
            'model': 'ToySegModel' if getattr(args, 'model', 'deeplabv3') == 'toy' else 'DeepLabV3-MobileNetV3-Large',
            'num_classes': args.num_classes,
            'pretrained': getattr(args, 'model', 'deeplabv3') != 'toy',
            
            # Training Hyperparameters
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': 'AdamW',
            'weight_decay': args.weight_decay,
            
            # Loss Function
            'loss_type': args.loss,
            'loss_function': loss_name,
            'dice_weight': args.dice_weight if args.loss in ('dice_focal',) else None,
            'focal_weight': args.focal_weight if args.loss in ('dice_focal',) else None,
            'focal_alpha': args.focal_alpha if args.loss in ('focal', 'dice_focal') else None,
            'focal_gamma': args.focal_gamma if args.loss in ('focal', 'dice_focal') else None,
            'tversky_alpha': args.tversky_alpha if args.loss == 'tversky' else None,
            'tversky_beta': args.tversky_beta if args.loss == 'tversky' else None,
            
            # Learning Rate Schedule
            'scheduler': 'CosineAnnealingLR with warmup',
            'warmup_epochs': args.warmup_epochs,
            
            # Regularization & Stopping
            'patience': args.patience,
            'early_stopping': True,
            
            # Data Configuration
            'img_height': args.img_height,
            'img_width': args.img_width,
            'img_resolution': f'{args.img_width}x{args.img_height}',
            'num_train_images': len(train_dataset),
            'num_val_images': len(val_dataset),
            'train_val_split': '80/20',
            'augmentation': True,
            
            # Performance Optimization
            'use_amp': use_amp,
            'num_workers': 12,
            'pin_memory': True,
            'prefetch_factor': 2,
            
            # Hardware & Environment
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'python_version': platform.python_version(),
            'platform': platform.system(),
            
            # Reproducibility
            'seed': args.seed,
            'deterministic': args.deterministic,
            'pythonhashseed_note': 'Set via .bat for full effect; os.environ is best-effort' if args.seed is not None else None,
            
            # Test-Split Configuration
            'test_split_config': getattr(args, 'test_split_config', None),
            'train_exclude_count': len(exclude_list) if exclude_list else 0,
            
            # Metadata for Analysis
            'training_start_time': datetime.datetime.now().isoformat(),
            'data_dir': str(args.data_dir),
        },
        'results': {
            # Will be filled at end of training
            'best_val_loss': None,
            'best_val_f1': None,
            'best_val_iou': None,
            'final_epoch': None,
            'total_training_time_seconds': None,
            'early_stopped': False,
        },
        'epoch_times': []  # Track time per epoch for performance analysis
    }
    
    for epoch in range(args.epochs):
        import time
        epoch_start_time = time.time()
        
        # Train
        if not small_data:
            model.train()
        else:
            model.eval()
        
        train_loss = 0
        train_loader_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch_idx, (imgs, masks) in enumerate(train_loader_progress):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward pass with AMP
            if use_amp:
                with torch.amp.autocast('cuda'):
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
        
        # Save best model (based on val_loss, track best F1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = f1
            best_val_iou = iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_best.pt")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}, F1={f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Track overall best F1 and IoU even if not best loss
        if f1 > best_val_f1:
            best_val_f1 = f1
        
        # Track best IoU (handle None initialization)
        current_best_iou = history['results'].get('best_val_iou')
        if current_best_iou is None or iou > current_best_iou:
            history['results']['best_val_iou'] = iou
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(round(epoch_time, 2))
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            history['results']['early_stopped'] = True
            history['results']['final_epoch'] = epoch + 1
            break
        
        scheduler.step()
    
    # Calculate training time and finalize results
    import time
    training_end_time = datetime.datetime.now()
    training_start_time = datetime.datetime.fromisoformat(history['config']['training_start_time'])
    total_training_seconds = (training_end_time - training_start_time).total_seconds()
    
    # Update final results
    if 'final_epoch' not in history['results'] or history['results']['final_epoch'] is None:
        history['results']['final_epoch'] = args.epochs
        history['results']['early_stopped'] = False
    
    history['results']['best_val_loss'] = float(best_val_loss)
    history['results']['best_val_f1'] = float(best_val_f1)
    history['results']['best_val_iou'] = float(best_val_iou) if 'best_val_iou' in locals() else float(max(history['val_iou']))
    history['results']['best_epoch'] = int(best_epoch) if 'best_epoch' in locals() else int(np.argmin(history['val_loss']) + 1)
    history['results']['total_training_time_seconds'] = round(total_training_seconds, 2)
    history['results']['avg_epoch_time_seconds'] = round(np.mean(history['epoch_times']), 2) if history['epoch_times'] else None
    history['results']['total_training_time_formatted'] = f"{int(total_training_seconds // 60)}m {int(total_training_seconds % 60)}s"
    history['config']['training_end_time'] = training_end_time.isoformat()
    
    # Save final model and history
    torch.save(model.state_dict(), args.output_dir / "whiteboard_seg_final.pt")
    with open(args.output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1 score: {best_val_f1:.4f}")
    print(f"Best IoU: {history['results']['best_val_iou']:.4f}")
    return model


def export_onnx(model, args):
    """Export trained model to ONNX format"""
    print("\nExporting model...")
    
    # Move model to CPU for export (avoids device mismatch issues)
    model = model.cpu()
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
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for trained models (default: ../SegmentationResearchPaper/experiments/default)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size (use 1 for tiny datasets)")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay for optimizer regularization")
    parser.add_argument("--num-classes", type=int, default=2,
                       help="Number of classes (2: background, stroke)")
    parser.add_argument("--img-height", type=int, default=768,
                       help="Image height for training")
    parser.add_argument("--img-width", type=int, default=1024,
                       help="Image width for training")
    parser.add_argument("--model", type=str, default="deeplabv3",
                       choices=["deeplabv3", "toy"],
                       help="Model architecture: deeplabv3 (production) or toy (test-only, fast CPU)")
    parser.add_argument("--loss", type=str, default="dice_focal",
                       choices=["ce", "dice", "focal", "dice_focal", "tversky"],
                       help="Loss function for training (default: dice_focal)")
    parser.add_argument("--dice-weight", type=float, default=0.6,
                       help="Weight for Dice loss in combined loss (default: 0.6)")
    parser.add_argument("--focal-weight", type=float, default=0.4,
                       help="Weight for Focal loss in combined loss (default: 0.4)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                       help="Focal loss alpha parameter - balance between classes (default: 0.25)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter - focus on hard examples (default: 2.0)")
    parser.add_argument("--tversky-alpha", type=float, default=0.3,
                       help="Tversky loss alpha (FP weight). Default: 0.3")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                       help="Tversky loss beta (FN weight). Default: 0.7")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                       help="Number of warmup epochs")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use Automatic Mixed Precision for faster GPU training (2x speedup)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (sets random, numpy, torch, cudnn)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable torch.use_deterministic_algorithms(True, warn_only=True) for best-effort reproducibility")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only export existing model")
    parser.add_argument("--test-split-config", type=str, default=None,
                       help="Path to test_splits.json — excludes listed image IDs from training")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train or load model
    if not args.skip_training:
        model = train_model(args)
    else:
        print("Loading existing model...")
        model = deeplabv3_mobilenet_v3_large(weights=None)
        model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)
        model.aux_classifier = None
        model.load_state_dict(torch.load(args.output_dir / "whiteboard_seg_best.pt"))
    
    # Export to ONNX (optional — failure here should not affect exit code)
    try:
        onnx_path = export_onnx(model, args)
    except Exception as e:
        print(f"Export failed (non-fatal): {e}")
        onnx_path = None
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"PyTorch model: {args.output_dir / 'whiteboard_seg_best.pt'}")
    if onnx_path:
        print(f"Exported model: {onnx_path}")
    print("="*60)


if __name__ == "__main__":
    main()
