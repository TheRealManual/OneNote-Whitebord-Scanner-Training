"""
COMPREHENSIVE MODEL COMPARISON TOOL

Compares two trained segmentation models to determine which performs better and WHY.
Helps answer critical training optimization questions:
  - Which batch size works better?
  - Does higher resolution improve quality?
  - Is more training time worth it?
  - Which loss function/optimizer is superior?

SETUP:
  compare_models/
    ├── model_1/
    │   ├── whiteboard_seg_best.pt       # Model weights
    │   └── training_history.json        # Training config & metrics
    ├── model_2/
    │   ├── whiteboard_seg_best.pt
    │   └── training_history.json
    ├── images/                          # Test images (1-N images)
    ├── masks/                           # Ground truth masks
    └── output/                          # Generated comparison results

OUTPUTS:
  1. Per-image comparison visualizations
  2. Aggregate metrics comparison
  3. Configuration difference analysis
  4. Optimization recommendations (what to change next)
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# Setup paths
SCRIPT_DIR = Path(__file__).parent
MODEL_1_DIR = SCRIPT_DIR / "model_1"
MODEL_2_DIR = SCRIPT_DIR / "model_2"
IMAGES_DIR = SCRIPT_DIR / "images"
MASKS_DIR = SCRIPT_DIR / "masks"
OUTPUT_DIR = SCRIPT_DIR / "output"

print("=" * 80)
print("MODEL COMPARISON TOOL")
print("=" * 80)
print(f"Model 1: {MODEL_1_DIR}")
print(f"Model 2: {MODEL_2_DIR}")
print(f"Output:  {OUTPUT_DIR}")
print()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_metrics(pred_mask, gt_mask):
    """Calculate comprehensive segmentation metrics"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # Pixel-level metrics
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    tp = intersection
    fp = (pred_binary & ~gt_binary).sum()
    fn = (~pred_binary & gt_binary).sum()
    tn = (~pred_binary & ~gt_binary).sum()
    
    # Core metrics
    iou = intersection / union if union > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    
    # Additional quality metrics
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0
    
    # Edge quality (boundary accuracy)
    pred_edges = cv2.Canny(pred_binary * 255, 100, 200)
    gt_edges = cv2.Canny(gt_binary * 255, 100, 200)
    edge_iou = np.logical_and(pred_edges, gt_edges).sum() / np.logical_or(pred_edges, gt_edges).sum() if np.logical_or(pred_edges, gt_edges).sum() > 0 else 0
    
    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'pixel_acc': float(pixel_acc),
        'dice': float(dice),
        'edge_iou': float(edge_iou),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'total_pixels': int(pred_mask.size),
        'stroke_pixels_pred': int(pred_binary.sum()),
        'stroke_pixels_gt': int(gt_binary.sum())
    }


def load_model_and_config(model_dir):
    """Load model weights and training configuration"""
    model_path = model_dir / "whiteboard_seg_best.pt"
    config_path = model_dir / "training_history.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # Load config
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Config: {config_path.name}")
        if 'config' in config:
            cfg = config['config']
            # Display core training parameters
            print(f"    Batch: {cfg.get('batch_size')}, LR: {cfg.get('learning_rate')}, "
                  f"Resolution: {cfg.get('img_resolution')}")
            # Display optimizer and loss parameters (with backward compatibility)
            wd = cfg.get('weight_decay', 1e-4)
            dice = cfg.get('dice_weight', 0.6)
            focal = cfg.get('focal_weight', 0.4)
            f_alpha = cfg.get('focal_alpha', 0.25)
            f_gamma = cfg.get('focal_gamma', 2.0)
            print(f"    Weight Decay: {wd}, Dice/Focal: {dice}/{focal}")
            print(f"    Focal Loss: alpha={f_alpha}, gamma={f_gamma}")
    else:
        print(f"  ⚠️  No config found: {config_path}")
    
    return model, config


def run_inference(model, image_path, img_size=(768, 1024)):
    """Run inference using trainer model (same as training)"""
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    # Resize to training resolution
    img_resized = img.resize((img_size[1], img_size[0]))  # W×H
    
    # Transform (ImageNet normalization - same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output['out']
        
        # Get predictions
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    
    # Convert to binary mask
    pred_mask = (pred * 255).astype(np.uint8)
    
    # Resize back to original size
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_mask, np.array(img)


def compare_configs(config1, config2):
    """Compare two training configurations and identify ALL differences"""
    if config1 is None or config2 is None:
        return {"error": "Missing configuration"}
    
    cfg1 = config1.get('config', {})
    cfg2 = config2.get('config', {})
    
    differences = {}
    
    # Get ALL keys from both configs
    all_keys = set(cfg1.keys()) | set(cfg2.keys())
    
    # Categorize config parameters
    categories = {
        'model_architecture': ['model_name', 'num_parameters', 'pretrained'],
        'training_hyperparameters': [
            'batch_size', 'learning_rate', 'epochs', 'optimizer', 
            'weight_decay', 'scheduler', 'warmup_epochs', 'patience',
            'use_amp', 'gradient_clip_max_norm'
        ],
        'data_augmentation': [
            'img_height', 'img_width', 'img_resolution',
            'augmentation_enabled', 'horizontal_flip', 'vertical_flip',
            'rotation_range', 'brightness_range', 'contrast_range'
        ],
        'loss_function': [
            'loss_function', 'dice_weight', 'focal_weight', 
            'loss_weights', 'class_weights'
        ],
        'dataset': [
            'num_train_images', 'num_val_images', 'train_val_split',
            'dataset_path', 'num_classes'
        ],
        'hardware_environment': [
            'device', 'cuda_available', 'gpu_name', 'num_workers',
            'pin_memory', 'use_compile'
        ],
        'other': []
    }
    
    # Categorize differences
    categorized_diffs = {cat: {} for cat in categories.keys()}
    
    for key in all_keys:
        val1 = cfg1.get(key)
        val2 = cfg2.get(key)
        
        if val1 != val2:
            diff_entry = {'model_1': val1, 'model_2': val2}
            
            # Find category
            found_category = False
            for cat, keys in categories.items():
                if key in keys:
                    categorized_diffs[cat][key] = diff_entry
                    found_category = True
                    break
            
            if not found_category:
                categorized_diffs['other'][key] = diff_entry
            
            differences[key] = diff_entry
    
    return {
        'all_differences': differences,
        'categorized_differences': categorized_diffs,
        'total_differences': len(differences)
    }


def generate_recommendations(config1, config2, results1, results2):
    """Generate comprehensive optimization recommendations analyzing ALL config parameters"""
    recommendations = []
    
    if not results1 or not results2:
        return ["Insufficient data for recommendations"]
    
    # Determine winner
    avg_f1_1 = np.mean([r['metrics']['f1'] for r in results1])
    avg_f1_2 = np.mean([r['metrics']['f1'] for r in results2])
    avg_iou_1 = np.mean([r['metrics']['iou'] for r in results1])
    avg_iou_2 = np.mean([r['metrics']['iou'] for r in results2])
    avg_edge_iou_1 = np.mean([r['metrics']['edge_iou'] for r in results1])
    avg_edge_iou_2 = np.mean([r['metrics']['edge_iou'] for r in results2])
    
    f1_improvement = ((avg_f1_2 - avg_f1_1) / avg_f1_1 * 100) if avg_f1_1 > 0 else 0
    iou_improvement = ((avg_iou_2 - avg_iou_1) / avg_iou_1 * 100) if avg_iou_1 > 0 else 0
    edge_improvement = ((avg_edge_iou_2 - avg_edge_iou_1) / avg_edge_iou_1 * 100) if avg_edge_iou_1 > 0 else 0
    
    winner = "Model 1" if avg_f1_1 > avg_f1_2 else "Model 2"
    winner_is_1 = avg_f1_1 > avg_f1_2
    better_config = config1 if winner_is_1 else config2
    worse_config = config2 if winner_is_1 else config1
    
    better_cfg = better_config.get('config', {})
    worse_cfg = worse_config.get('config', {})
    
    # Header
    recommendations.append("=" * 80)
    recommendations.append(f"**WINNER: {winner}**")
    recommendations.append(f"  F1: {max(avg_f1_1, avg_f1_2):.4f} vs {min(avg_f1_1, avg_f1_2):.4f} ({abs(f1_improvement):.1f}% {'better' if f1_improvement > 0 else 'worse'})")
    recommendations.append(f"  IoU: {max(avg_iou_1, avg_iou_2):.4f} vs {min(avg_iou_1, avg_iou_2):.4f} ({abs(iou_improvement):.1f}% {'better' if iou_improvement > 0 else 'worse'})")
    recommendations.append(f"  Edge IoU: {max(avg_edge_iou_1, avg_edge_iou_2):.4f} vs {min(avg_edge_iou_1, avg_edge_iou_2):.4f} ({abs(edge_improvement):.1f}% {'better' if edge_improvement > 0 else 'worse'})")
    recommendations.append("=" * 80)
    recommendations.append("")
    
    # Get comprehensive diff
    diff_data = compare_configs(better_config, worse_config)
    diff = diff_data.get('all_differences', {})
    categorized = diff_data.get('categorized_differences', {})
    
    if not diff:
        recommendations.append("✓ Both models have IDENTICAL configurations")
        recommendations.append("  Performance difference is due to training randomness/initialization")
        recommendations.append("")
        return recommendations
    
    recommendations.append(f"CONFIGURATION ANALYSIS ({diff_data.get('total_differences', 0)} differences found)")
    recommendations.append("")
    
    # ========================================================================
    # TRAINING HYPERPARAMETERS
    # ========================================================================
    hyper_diffs = categorized.get('training_hyperparameters', {})
    if hyper_diffs:
        recommendations.append("📊 TRAINING HYPERPARAMETERS:")
        recommendations.append("-" * 80)
        
        # Batch Size
        if 'batch_size' in hyper_diffs:
            bs_better = hyper_diffs['batch_size']['model_1'] if winner_is_1 else hyper_diffs['batch_size']['model_2']
            bs_worse = hyper_diffs['batch_size']['model_2'] if winner_is_1 else hyper_diffs['batch_size']['model_1']
            recommendations.append(f"✓ Batch Size: {bs_better} works better than {bs_worse}")
            
            if bs_better > bs_worse:
                recommendations.append(f"  → RECOMMENDATION: INCREASE batch size to {bs_better} or higher")
                recommendations.append(f"     Benefits: More stable gradients, better generalization, faster training")
                recommendations.append(f"     Caution: Requires more GPU memory (~{bs_better/bs_worse:.1f}x current usage)")
            else:
                recommendations.append(f"  → RECOMMENDATION: DECREASE batch size to {bs_better}")
                recommendations.append(f"     Benefits: More gradient updates per epoch, escapes local minima better")
                recommendations.append(f"     Trade-off: Noisier gradients, may need more epochs")
        
        # Learning Rate
        if 'learning_rate' in hyper_diffs:
            lr_better = hyper_diffs['learning_rate']['model_1'] if winner_is_1 else hyper_diffs['learning_rate']['model_2']
            lr_worse = hyper_diffs['learning_rate']['model_2'] if winner_is_1 else hyper_diffs['learning_rate']['model_1']
            recommendations.append(f"✓ Learning Rate: {lr_better} works better than {lr_worse}")
            
            if lr_better < lr_worse:
                recommendations.append(f"  → RECOMMENDATION: DECREASE learning rate to {lr_better}")
                recommendations.append(f"     Benefits: More precise weight updates, better fine-tuning, avoids overshooting")
                recommendations.append(f"     Trade-off: Slower convergence, may need more epochs")
            else:
                recommendations.append(f"  → RECOMMENDATION: INCREASE learning rate to {lr_better}")
                recommendations.append(f"     Benefits: Faster convergence, explores loss landscape better")
                recommendations.append(f"     Caution: May overshoot optimal weights, reduce if loss oscillates")
        
        # Optimizer
        if 'optimizer' in hyper_diffs:
            opt_better = hyper_diffs['optimizer']['model_1'] if winner_is_1 else hyper_diffs['optimizer']['model_2']
            opt_worse = hyper_diffs['optimizer']['model_2'] if winner_is_1 else hyper_diffs['optimizer']['model_1']
            recommendations.append(f"✓ Optimizer: {opt_better} works better than {opt_worse}")
            recommendations.append(f"  → RECOMMENDATION: USE {opt_better} optimizer")
        
        # Weight Decay
        if 'weight_decay' in hyper_diffs:
            wd_better = hyper_diffs['weight_decay']['model_1'] if winner_is_1 else hyper_diffs['weight_decay']['model_2']
            wd_worse = hyper_diffs['weight_decay']['model_2'] if winner_is_1 else hyper_diffs['weight_decay']['model_1']
            
            # Handle None values (old models without this parameter)
            wd_better = wd_better if wd_better is not None else 1e-4
            wd_worse = wd_worse if wd_worse is not None else 1e-4
            
            recommendations.append(f"✓ Weight Decay: {wd_better} works better than {wd_worse}")
            
            if wd_better > wd_worse:
                recommendations.append(f"  → RECOMMENDATION: INCREASE weight decay to {wd_better}")
                recommendations.append(f"     Benefits: Stronger regularization, reduces overfitting")
                recommendations.append(f"     Use: --weight-decay {wd_better}")
            else:
                recommendations.append(f"  → RECOMMENDATION: DECREASE weight decay to {wd_better}")
                recommendations.append(f"     Benefits: Less regularization, allows model to fit training data better")
                recommendations.append(f"     Use: --weight-decay {wd_better}")
        
        # Scheduler
        if 'scheduler' in hyper_diffs:
            sched_better = hyper_diffs['scheduler']['model_1'] if winner_is_1 else hyper_diffs['scheduler']['model_2']
            sched_worse = hyper_diffs['scheduler']['model_2'] if winner_is_1 else hyper_diffs['scheduler']['model_1']
            recommendations.append(f"✓ LR Scheduler: {sched_better} works better than {sched_worse}")
            recommendations.append(f"  → RECOMMENDATION: USE {sched_better} scheduler")
        
        # Warmup
        if 'warmup_epochs' in hyper_diffs:
            warm_better = hyper_diffs['warmup_epochs']['model_1'] if winner_is_1 else hyper_diffs['warmup_epochs']['model_2']
            warm_worse = hyper_diffs['warmup_epochs']['model_2'] if winner_is_1 else hyper_diffs['warmup_epochs']['model_1']
            recommendations.append(f"✓ Warmup Epochs: {warm_better} works better than {warm_worse}")
            recommendations.append(f"  → RECOMMENDATION: SET warmup to {warm_better} epochs")
        
        # Patience (Early Stopping)
        if 'patience' in hyper_diffs:
            pat_better = hyper_diffs['patience']['model_1'] if winner_is_1 else hyper_diffs['patience']['model_2']
            pat_worse = hyper_diffs['patience']['model_2'] if winner_is_1 else hyper_diffs['patience']['model_1']
            recommendations.append(f"✓ Early Stopping Patience: {pat_better} works better than {pat_worse}")
            
            if pat_better > pat_worse:
                recommendations.append(f"  → RECOMMENDATION: INCREASE patience to {pat_better}")
                recommendations.append(f"     Benefits: Allows more time for convergence, avoids premature stopping")
            else:
                recommendations.append(f"  → RECOMMENDATION: DECREASE patience to {pat_better}")
                recommendations.append(f"     Benefits: Stops training sooner, saves time, prevents overfitting")
        
        # AMP (Mixed Precision)
        if 'use_amp' in hyper_diffs:
            amp_better = hyper_diffs['use_amp']['model_1'] if winner_is_1 else hyper_diffs['use_amp']['model_2']
            amp_worse = hyper_diffs['use_amp']['model_2'] if winner_is_1 else hyper_diffs['use_amp']['model_1']
            recommendations.append(f"✓ AMP (Mixed Precision): {'ENABLED' if amp_better else 'DISABLED'} works better")
            
            if amp_better:
                recommendations.append(f"  → RECOMMENDATION: ENABLE AMP (--use-amp flag)")
                recommendations.append(f"     Benefits: ~2x faster training, ~40% less GPU memory, minimal quality loss")
            else:
                recommendations.append(f"  → RECOMMENDATION: DISABLE AMP")
                recommendations.append(f"     Benefits: Full FP32 precision, may improve accuracy slightly")
                recommendations.append(f"     Trade-off: Slower training, more GPU memory required")
        
        # Gradient Clipping
        if 'gradient_clip_max_norm' in hyper_diffs:
            clip_better = hyper_diffs['gradient_clip_max_norm']['model_1'] if winner_is_1 else hyper_diffs['gradient_clip_max_norm']['model_2']
            recommendations.append(f"✓ Gradient Clipping: {clip_better} works better")
            recommendations.append(f"  → RECOMMENDATION: SET gradient clip to {clip_better}")
        
        # Epochs
        if 'epochs' in hyper_diffs:
            ep_better = hyper_diffs['epochs']['model_1'] if winner_is_1 else hyper_diffs['epochs']['model_2']
            ep_worse = hyper_diffs['epochs']['model_2'] if winner_is_1 else hyper_diffs['epochs']['model_1']
            recommendations.append(f"✓ Max Epochs: {ep_better} works better than {ep_worse}")
            
            if ep_better > ep_worse:
                recommendations.append(f"  → RECOMMENDATION: INCREASE max epochs to {ep_better}+")
                recommendations.append(f"     Benefits: More training time, better convergence")
            else:
                recommendations.append(f"  → RECOMMENDATION: {ep_better} epochs is sufficient")
        
        recommendations.append("")
    
    # ========================================================================
    # DATA AUGMENTATION & RESOLUTION
    # ========================================================================
    aug_diffs = categorized.get('data_augmentation', {})
    if aug_diffs:
        recommendations.append("🖼️  DATA AUGMENTATION & RESOLUTION:")
        recommendations.append("-" * 80)
        
        # Resolution
        if 'img_height' in aug_diffs or 'img_width' in aug_diffs or 'img_resolution' in aug_diffs:
            res_better = f"{better_cfg.get('img_width', '?')}×{better_cfg.get('img_height', '?')}"
            res_worse = f"{worse_cfg.get('img_width', '?')}×{worse_cfg.get('img_height', '?')}"
            recommendations.append(f"✓ Resolution: {res_better} works better than {res_worse}")
            
            better_pixels = better_cfg.get('img_width', 1) * better_cfg.get('img_height', 1)
            worse_pixels = worse_cfg.get('img_width', 1) * worse_cfg.get('img_height', 1)
            
            if better_pixels > worse_pixels:
                recommendations.append(f"  → RECOMMENDATION: INCREASE resolution to {res_better} or higher")
                recommendations.append(f"     Benefits: Better detail preservation, improved thin line detection")
                recommendations.append(f"     Trade-off: ~{better_pixels/worse_pixels:.1f}x slower training, more GPU memory")
                recommendations.append(f"     Edge IoU improved by {abs(edge_improvement):.1f}% (critical for thin lines!)")
            else:
                recommendations.append(f"  → RECOMMENDATION: {res_better} resolution is sufficient")
                recommendations.append(f"     Benefits: Faster training, less GPU memory")
        
        # Augmentation settings
        aug_params = ['horizontal_flip', 'vertical_flip', 'rotation_range', 'brightness_range', 'contrast_range']
        for param in aug_params:
            if param in aug_diffs:
                val_better = aug_diffs[param]['model_1'] if winner_is_1 else aug_diffs[param]['model_2']
                val_worse = aug_diffs[param]['model_2'] if winner_is_1 else aug_diffs[param]['model_1']
                recommendations.append(f"✓ {param.replace('_', ' ').title()}: {val_better} works better than {val_worse}")
                recommendations.append(f"  → RECOMMENDATION: SET {param} to {val_better}")
        
        recommendations.append("")
    
    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    loss_diffs = categorized.get('loss_function', {})
    if loss_diffs:
        recommendations.append("⚖️  LOSS FUNCTION:")
        recommendations.append("-" * 80)
        
        if 'loss_function' in loss_diffs:
            loss_better = loss_diffs['loss_function']['model_1'] if winner_is_1 else loss_diffs['loss_function']['model_2']
            loss_worse = loss_diffs['loss_function']['model_2'] if winner_is_1 else loss_diffs['loss_function']['model_1']
            recommendations.append(f"✓ Loss Function: {loss_better} works better than {loss_worse}")
            recommendations.append(f"  → RECOMMENDATION: USE {loss_better}")
        
        if 'dice_weight' in loss_diffs or 'focal_weight' in loss_diffs:
            dice_better = better_cfg.get('dice_weight', 0.6)  # Default to old value if missing
            focal_better = better_cfg.get('focal_weight', 0.4)  # Default to old value if missing
            dice_worse = worse_cfg.get('dice_weight', 0.6)
            focal_worse = worse_cfg.get('focal_weight', 0.4)
            
            recommendations.append(f"✓ Loss Weights: Dice={dice_better}, Focal={focal_better} works better than Dice={dice_worse}, Focal={focal_worse}")
            recommendations.append(f"  → RECOMMENDATION: SET --dice-weight {dice_better} --focal-weight {focal_better}")
            
            # Analyze what the change means
            if focal_better > focal_worse:
                recommendations.append(f"     Analysis: Higher focal weight ({focal_better} vs {focal_worse}) reduces false positives")
                recommendations.append(f"     This helps with precision issues (model over-predicting strokes)")
            if dice_better > dice_worse:
                recommendations.append(f"     Analysis: Higher dice weight ({dice_better} vs {dice_worse}) improves overall overlap")
                recommendations.append(f"     This helps with recall issues (model missing strokes)")
        
        if 'focal_alpha' in loss_diffs or 'focal_gamma' in loss_diffs:
            alpha_better = better_cfg.get('focal_alpha', 0.25)
            gamma_better = better_cfg.get('focal_gamma', 2.0)
            alpha_worse = worse_cfg.get('focal_alpha', 0.25)
            gamma_worse = worse_cfg.get('focal_gamma', 2.0)
            
            if alpha_better != alpha_worse:
                recommendations.append(f"✓ Focal Alpha: {alpha_better} works better than {alpha_worse}")
                recommendations.append(f"  → RECOMMENDATION: SET --focal-alpha {alpha_better}")
                if alpha_better > alpha_worse:
                    recommendations.append(f"     Analysis: Higher alpha focuses more on hard-to-classify regions (small text, thin lines)")
                else:
                    recommendations.append(f"     Analysis: Lower alpha gives more balanced training")
            
            if gamma_better != gamma_worse:
                recommendations.append(f"✓ Focal Gamma: {gamma_better} works better than {gamma_worse}")
                recommendations.append(f"  → RECOMMENDATION: SET --focal-gamma {gamma_better}")
                if gamma_better > gamma_worse:
                    recommendations.append(f"     Analysis: Higher gamma puts more emphasis on misclassified pixels")
                else:
                    recommendations.append(f"     Analysis: Lower gamma provides smoother loss gradient")
        
        recommendations.append("")
    
    # ========================================================================
    # HARDWARE & ENVIRONMENT
    # ========================================================================
    hw_diffs = categorized.get('hardware_environment', {})
    if hw_diffs:
        recommendations.append("⚙️  HARDWARE & ENVIRONMENT:")
        recommendations.append("-" * 80)
        
        if 'num_workers' in hw_diffs:
            workers_better = hw_diffs['num_workers']['model_1'] if winner_is_1 else hw_diffs['num_workers']['model_2']
            recommendations.append(f"✓ DataLoader Workers: {workers_better} works better")
            recommendations.append(f"  → RECOMMENDATION: SET num_workers to {workers_better}")
        
        if 'pin_memory' in hw_diffs:
            pin_better = hw_diffs['pin_memory']['model_1'] if winner_is_1 else hw_diffs['pin_memory']['model_2']
            recommendations.append(f"✓ Pin Memory: {pin_better} works better")
            recommendations.append(f"  → RECOMMENDATION: {'ENABLE' if pin_better else 'DISABLE'} pin_memory")
        
        if 'use_compile' in hw_diffs:
            compile_better = hw_diffs['use_compile']['model_1'] if winner_is_1 else hw_diffs['use_compile']['model_2']
            recommendations.append(f"✓ torch.compile: {compile_better} works better")
            if compile_better:
                recommendations.append(f"  → RECOMMENDATION: ENABLE torch.compile for faster training")
            else:
                recommendations.append(f"  → RECOMMENDATION: DISABLE torch.compile (stability issues)")
        
        recommendations.append("")
    
    # ========================================================================
    # PERFORMANCE ANALYSIS
    # ========================================================================
    recommendations.append("📈 PERFORMANCE ANALYSIS:")
    recommendations.append("-" * 80)
    
    results_better = results1 if winner_is_1 else results2
    avg_precision = np.mean([r['metrics']['precision'] for r in results_better])
    avg_recall = np.mean([r['metrics']['recall'] for r in results_better])
    
    if avg_precision < avg_recall - 0.1:
        recommendations.append("⚠️  High Recall but Low Precision (many false positives)")
        recommendations.append("  → Model is over-predicting strokes")
        recommendations.append("  → SUGGESTIONS:")
        recommendations.append("     • Lower learning rate for finer weight adjustments")
        recommendations.append("     • Increase weight decay for stronger regularization")
        recommendations.append("     • Add dropout layers to reduce overfitting")
        recommendations.append("     • Increase focal loss weight (vs dice)")
    elif avg_recall < avg_precision - 0.1:
        recommendations.append("⚠️  High Precision but Low Recall (missing strokes)")
        recommendations.append("  → Model is under-predicting strokes")
        recommendations.append("  → SUGGESTIONS:")
        recommendations.append("     • Increase batch size for more examples per gradient update")
        recommendations.append("     • Higher resolution to capture thin details")
        recommendations.append("     • Increase dice loss weight (vs focal)")
        recommendations.append("     • Add more aggressive augmentation")
    else:
        recommendations.append("✓ Good balance between Precision and Recall")
    
    if max(avg_f1_1, avg_f1_2) < 0.5:
        recommendations.append("")
        recommendations.append("⚠️  Both models underperforming (F1 < 0.50)")
        recommendations.append("  → CRITICAL IMPROVEMENTS NEEDED:")
        recommendations.append("     • Collect more training data (currently {})".format(better_cfg.get('num_train_images', 'unknown')))
        recommendations.append("     • Try different architecture (ResNet, EfficientNet backbone)")
        recommendations.append("     • Verify data quality (masks match images correctly)")
        recommendations.append("     • Increase resolution significantly")
    elif max(avg_f1_1, avg_f1_2) > 0.7:
        recommendations.append("")
        recommendations.append("✓ Excellent performance (F1 > 0.70)!")
        recommendations.append("  → Model is production-ready")
    
    recommendations.append("")
    
    # ========================================================================
    # TRAINING EFFICIENCY
    # ========================================================================
    if 'results' in better_config and 'results' in worse_config:
        recommendations.append("⏱️  TRAINING EFFICIENCY:")
        recommendations.append("-" * 80)
        
        time_better = better_config.get('results', {}).get('avg_epoch_time_seconds', 0)
        time_worse = worse_config.get('results', {}).get('avg_epoch_time_seconds', 0)
        total_better = better_config.get('results', {}).get('total_training_time_seconds', 0)
        total_worse = worse_config.get('results', {}).get('total_training_time_seconds', 0)
        
        if time_better and time_worse:
            speedup = time_worse / time_better if time_better > 0 else 1
            if speedup > 1.2:
                recommendations.append(f"✓ {winner} is {speedup:.1f}x FASTER per epoch ({time_better:.1f}s vs {time_worse:.1f}s)")
                recommendations.append(f"  → Better quality AND faster training!")
            elif speedup < 0.8:
                recommendations.append(f"⚠️  {winner} is {1/speedup:.1f}x SLOWER per epoch ({time_better:.1f}s vs {time_worse:.1f}s)")
                recommendations.append(f"  → Trade-off: Better quality at cost of {abs((1-speedup)*100):.0f}% more time")
            else:
                recommendations.append(f"✓ Similar training speed ({time_better:.1f}s vs {time_worse:.1f}s per epoch)")
        
        if total_better and total_worse:
            recommendations.append(f"  Total training time: {total_better/60:.1f} min vs {total_worse/60:.1f} min")
        
        recommendations.append("")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    recommendations.append("=" * 80)
    recommendations.append("🎯 FINAL RECOMMENDATION:")
    recommendations.append("=" * 80)
    recommendations.append(f"Use {winner}'s configuration as baseline:")
    
    key_settings = []
    if 'batch_size' in diff:
        key_settings.append(f"batch_size={better_cfg.get('batch_size')}")
    if 'learning_rate' in diff:
        key_settings.append(f"lr={better_cfg.get('learning_rate')}")
    if 'img_height' in diff or 'img_width' in diff:
        key_settings.append(f"resolution={better_cfg.get('img_width')}×{better_cfg.get('img_height')}")
    if 'use_amp' in diff:
        key_settings.append(f"amp={better_cfg.get('use_amp')}")
    
    if key_settings:
        recommendations.append(f"  python train_segmentation.py --{' --'.join(key_settings)}")
    
    recommendations.append("")
    recommendations.append(f"Expected performance: F1={max(avg_f1_1, avg_f1_2):.4f}, IoU={max(avg_iou_1, avg_iou_2):.4f}")
    recommendations.append("=" * 80)
    
    return recommendations


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_per_image_comparison(img_name, original_img, gt_mask, 
                                mask1, mask2, metrics1, metrics2, 
                                config1, config2, output_path):
    """Create comprehensive per-image comparison visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Model Comparison: {img_name}', fontsize=16, fontweight='bold')
    
    # Row 1: Original, GT, Overlays
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title(f'Ground Truth\n{np.count_nonzero(gt_mask):,} pixels', fontweight='bold')
    ax2.axis('off')
    
    # Model 1 overlay (red=GT, blue=Pred, white=Both)
    overlay1 = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlay1[gt_mask > 127] = [255, 0, 0]
    overlay1[mask1 > 127] = [0, 0, 255]
    overlay1[(gt_mask > 127) & (mask1 > 127)] = [255, 255, 255]
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay1)
    ax3.set_title('Model 1: GT (red) vs Pred (blue)', fontweight='bold')
    ax3.axis('off')
    
    # Model 2 overlay
    overlay2 = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlay2[gt_mask > 127] = [255, 0, 0]
    overlay2[mask2 > 127] = [0, 255, 0]
    overlay2[(gt_mask > 127) & (mask2 > 127)] = [255, 255, 255]
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay2)
    ax4.set_title('Model 2: GT (red) vs Pred (green)', fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Model outputs and error maps
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(mask1, cmap='gray')
    ax5.set_title(f'Model 1 Output\nF1: {metrics1["f1"]:.3f} | IoU: {metrics1["iou"]:.3f}', 
                 fontweight='bold', color='blue')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(mask2, cmap='gray')
    ax6.set_title(f'Model 2 Output\nF1: {metrics2["f1"]:.3f} | IoU: {metrics2["iou"]:.3f}', 
                 fontweight='bold', color='green')
    ax6.axis('off')
    
    # Error maps (white=correct, red=FP, green=FN)
    error1 = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error1[(gt_mask > 127) & (mask1 > 127)] = [255, 255, 255]
    error1[(gt_mask <= 127) & (mask1 > 127)] = [255, 0, 0]
    error1[(gt_mask > 127) & (mask1 <= 127)] = [0, 255, 0]
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(error1)
    ax7.set_title(f'Model 1 Errors\nFP: {metrics1["fp"]:,} | FN: {metrics1["fn"]:,}', 
                 fontweight='bold')
    ax7.axis('off')
    
    error2 = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error2[(gt_mask > 127) & (mask2 > 127)] = [255, 255, 255]
    error2[(gt_mask <= 127) & (mask2 > 127)] = [255, 0, 0]
    error2[(gt_mask > 127) & (mask2 <= 127)] = [0, 255, 0]
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(error2)
    ax8.set_title(f'Model 2 Errors\nFP: {metrics2["fp"]:,} | FN: {metrics2["fn"]:,}', 
                 fontweight='bold')
    ax8.axis('off')
    
    # Row 3: Metrics comparison
    ax9 = fig.add_subplot(gs[2, :2])
    metrics_names = ['F1', 'IoU', 'Precision', 'Recall', 'Dice', 'Edge IoU']
    metrics_keys = ['f1', 'iou', 'precision', 'recall', 'dice', 'edge_iou']
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    vals1 = [metrics1[k] for k in metrics_keys]
    vals2 = [metrics2[k] for k in metrics_keys]
    
    ax9.bar(x - width/2, vals1, width, label='Model 1', color='blue', alpha=0.7)
    ax9.bar(x + width/2, vals2, width, label='Model 2', color='green', alpha=0.7)
    
    ax9.set_ylabel('Score', fontweight='bold')
    ax9.set_title('Metrics Comparison', fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    ax9.set_ylim([0, 1.0])
    
    # Configuration comparison
    ax10 = fig.add_subplot(gs[2, 2:])
    
    cfg1 = config1.get('config', {}) if config1 else {}
    cfg2 = config2.get('config', {}) if config2 else {}
    
    # Get comprehensive diff
    diff_data = compare_configs(config1, config2)
    all_diffs = diff_data.get('all_differences', {})
    
    config_text = "CONFIGURATION COMPARISON:\n\n"
    config_text += f"{'Parameter':<25} {'Model 1':<20} {'Model 2':<20}\n"
    config_text += "=" * 65 + "\n"
    
    # Show ALL differences, not just a subset
    if all_diffs:
        for key, vals in sorted(all_diffs.items()):
            val1 = vals['model_1']
            val2 = vals['model_2']
            
            # Format values nicely
            val1_str = str(val1) if val1 is not None else 'N/A'
            val2_str = str(val2) if val2 is not None else 'N/A'
            
            # Truncate long values
            if len(val1_str) > 18:
                val1_str = val1_str[:15] + '...'
            if len(val2_str) > 18:
                val2_str = val2_str[:15] + '...'
            
            config_text += f"{key:<25} {val1_str:<20} {val2_str:<20} ←\n"
    else:
        config_text += "No differences found (identical configurations)\n"
    
    # Add results
    res1 = config1.get('results', {}) if config1 else {}
    res2 = config2.get('results', {}) if config2 else {}
    
    config_text += "\n" + "=" * 65 + "\n"
    config_text += "TRAINING RESULTS:\n\n"
    config_text += f"{'Metric':<25} {'Model 1':<20} {'Model 2':<20}\n"
    config_text += "=" * 65 + "\n"
    config_text += f"{'Best Val F1':<25} {res1.get('best_val_f1', 0):<20.4f} {res2.get('best_val_f1', 0):<20.4f}\n"
    config_text += f"{'Best Val IoU':<25} {res1.get('best_val_iou', 0):<20.4f} {res2.get('best_val_iou', 0):<20.4f}\n"
    config_text += f"{'Epochs Trained':<25} {res1.get('final_epoch', 0):<20} {res2.get('final_epoch', 0):<20}\n"
    config_text += f"{'Avg Epoch Time (s)':<25} {res1.get('avg_epoch_time_seconds', 0):<20.1f} {res2.get('avg_epoch_time_seconds', 0):<20.1f}\n"
    config_text += f"{'Total Time (min)':<25} {res1.get('total_training_time_seconds', 0)/60:<20.1f} {res2.get('total_training_time_seconds', 0)/60:<20.1f}\n"
    
    ax10.text(0.05, 0.95, config_text, fontsize=8, family='monospace',
             verticalalignment='top', transform=ax10.transAxes)
    ax10.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_aggregate_comparison(all_results_1, all_results_2, config1, config2, output_path):
    """Create aggregate comparison across all test images"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Aggregate Model Comparison - All Test Images', fontsize=16, fontweight='bold')
    
    # Calculate aggregate metrics
    metrics_keys = ['f1', 'iou', 'precision', 'recall', 'dice', 'edge_iou']
    
    agg1 = {k: np.mean([r['metrics'][k] for r in all_results_1]) for k in metrics_keys}
    agg2 = {k: np.mean([r['metrics'][k] for r in all_results_2]) for k in metrics_keys}
    
    std1 = {k: np.std([r['metrics'][k] for r in all_results_1]) for k in metrics_keys}
    std2 = {k: np.std([r['metrics'][k] for r in all_results_2]) for k in metrics_keys}
    
    # Bar chart comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    metrics_names = ['F1', 'IoU', 'Precision', 'Recall', 'Dice', 'Edge IoU']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    vals1 = [agg1[k] for k in metrics_keys]
    vals2 = [agg2[k] for k in metrics_keys]
    err1 = [std1[k] for k in metrics_keys]
    err2 = [std2[k] for k in metrics_keys]
    
    ax1.bar(x - width/2, vals1, width, yerr=err1, label='Model 1', 
           color='blue', alpha=0.7, capsize=5)
    ax1.bar(x + width/2, vals2, width, yerr=err2, label='Model 2', 
           color='green', alpha=0.7, capsize=5)
    
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title('Average Metrics Across All Test Images (with std dev)', 
                 fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Per-image breakdown
    ax2 = fig.add_subplot(gs[1, 0])
    
    image_names = [r['image_name'] for r in all_results_1]
    f1_scores_1 = [r['metrics']['f1'] for r in all_results_1]
    f1_scores_2 = [r['metrics']['f1'] for r in all_results_2]
    
    x_imgs = np.arange(len(image_names))
    ax2.plot(x_imgs, f1_scores_1, 'o-', label='Model 1', color='blue', linewidth=2)
    ax2.plot(x_imgs, f1_scores_2, 's-', label='Model 2', color='green', linewidth=2)
    ax2.set_xlabel('Test Image', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score per Image', fontweight='bold')
    ax2.set_xticks(x_imgs)
    ax2.set_xticklabels([name[:15] for name in image_names], rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # IoU comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    iou_scores_1 = [r['metrics']['iou'] for r in all_results_1]
    iou_scores_2 = [r['metrics']['iou'] for r in all_results_2]
    
    ax3.plot(x_imgs, iou_scores_1, 'o-', label='Model 1', color='blue', linewidth=2)
    ax3.plot(x_imgs, iou_scores_2, 's-', label='Model 2', color='green', linewidth=2)
    ax3.set_xlabel('Test Image', fontweight='bold')
    ax3.set_ylabel('IoU', fontweight='bold')
    ax3.set_title('IoU per Image', fontweight='bold')
    ax3.set_xticks(x_imgs)
    ax3.set_xticklabels([name[:15] for name in image_names], rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision-Recall scatter
    ax4 = fig.add_subplot(gs[1, 2])
    
    prec1 = [r['metrics']['precision'] for r in all_results_1]
    rec1 = [r['metrics']['recall'] for r in all_results_1]
    prec2 = [r['metrics']['precision'] for r in all_results_2]
    rec2 = [r['metrics']['recall'] for r in all_results_2]
    
    ax4.scatter(rec1, prec1, s=100, alpha=0.6, label='Model 1', color='blue', edgecolors='black')
    ax4.scatter(rec2, prec2, s=100, alpha=0.6, label='Model 2', color='green', marker='s', edgecolors='black')
    
    # Add F1 isolines
    recall_range = np.linspace(0.01, 0.99, 100)
    for f1_val in [0.3, 0.5, 0.7, 0.9]:
        precision_for_f1 = f1_val * recall_range / (2 * recall_range - f1_val)
        ax4.plot(recall_range, precision_for_f1, '--', color='gray', alpha=0.3, linewidth=1)
        ax4.text(0.9, precision_for_f1[-10], f'F1={f1_val}', fontsize=8, color='gray')
    
    ax4.set_xlabel('Recall', fontweight='bold')
    ax4.set_ylabel('Precision', fontweight='bold')
    ax4.set_title('Precision-Recall Trade-off', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    # Configuration comparison text
    ax5 = fig.add_subplot(gs[2, :])
    
    diff_data = compare_configs(config1, config2)
    recommendations = generate_recommendations(config1, config2, all_results_1, all_results_2)
    
    summary_text = "COMPARISON SUMMARY\n"
    summary_text += "=" * 100 + "\n\n"
    
    summary_text += f"Model 1 Average: F1={agg1['f1']:.4f} (±{std1['f1']:.4f}), IoU={agg1['iou']:.4f} (±{std1['iou']:.4f})\n"
    summary_text += f"Model 2 Average: F1={agg2['f1']:.4f} (±{std2['f1']:.4f}), IoU={agg2['iou']:.4f} (±{std2['iou']:.4f})\n\n"
    
    # Show categorized differences
    categorized = diff_data.get('categorized_differences', {})
    total_diffs = diff_data.get('total_differences', 0)
    
    if total_diffs > 0:
        summary_text += f"CONFIGURATION DIFFERENCES ({total_diffs} total):\n\n"
        
        categories_to_show = {
            'training_hyperparameters': 'Training Hyperparameters',
            'data_augmentation': 'Data & Augmentation',
            'loss_function': 'Loss Function',
            'hardware_environment': 'Hardware Settings'
        }
        
        for cat_key, cat_name in categories_to_show.items():
            cat_diffs = categorized.get(cat_key, {})
            if cat_diffs:
                summary_text += f"  {cat_name}:\n"
                for key, vals in cat_diffs.items():
                    summary_text += f"    {key}: {vals['model_1']} vs {vals['model_2']}\n"
                summary_text += "\n"
    else:
        summary_text += "No configuration differences found\n\n"
    
    summary_text += "TOP RECOMMENDATIONS:\n"
    # Show first 12 recommendation lines (truncated for display)
    for i, rec in enumerate(recommendations[:12]):
        if len(rec) > 95:
            summary_text += f"{rec[:92]}...\n"
        else:
            summary_text += f"{rec}\n"
    
    if len(recommendations) > 12:
        summary_text += f"\n... ({len(recommendations) - 12} more recommendations in JSON output)\n"
    
    ax5.text(0.02, 0.98, summary_text, fontsize=8, family='monospace',
            verticalalignment='top', transform=ax5.transAxes)
    ax5.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive model comparison"""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load models and configs
    print("Loading models...")
    try:
        model1, config1 = load_model_and_config(MODEL_1_DIR)
        print("✓ Model 1 loaded\n")
    except Exception as e:
        print(f"❌ Failed to load Model 1: {e}")
        return
    
    try:
        model2, config2 = load_model_and_config(MODEL_2_DIR)
        print("✓ Model 2 loaded\n")
    except Exception as e:
        print(f"❌ Failed to load Model 2: {e}")
        return
    
    # Get test images
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"❌ No test images found in {IMAGES_DIR}")
        print("Please add test images to compare_models/images/")
        return
    
    print(f"Found {len(image_files)} test image(s)\n")
    
    # Process each image
    all_results_1 = []
    all_results_2 = []
    
    for img_path in image_files:
        print("=" * 80)
        print(f"Processing: {img_path.name}")
        print("=" * 80)
        
        # Load ground truth
        mask_path = MASKS_DIR / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"⚠️  No mask found: {mask_path.name}, skipping...")
            continue
        
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        print(f"Ground truth: {np.count_nonzero(gt_mask > 127):,} stroke pixels\n")
        
        # Get inference resolution from configs
        res1 = (config1.get('config', {}).get('img_height', 768), 
                config1.get('config', {}).get('img_width', 1024))
        res2 = (config2.get('config', {}).get('img_height', 768),
                config2.get('config', {}).get('img_width', 1024))
        
        # Model 1 inference
        print("Running Model 1 inference...")
        mask1, original_img = run_inference(model1, img_path, img_size=res1)
        metrics1 = calculate_metrics(mask1, gt_mask)
        print(f"  F1: {metrics1['f1']:.4f}, IoU: {metrics1['iou']:.4f}")
        
        all_results_1.append({
            'image_name': img_path.name,
            'metrics': metrics1,
            'mask': mask1
        })
        
        # Model 2 inference
        print("Running Model 2 inference...")
        mask2, _ = run_inference(model2, img_path, img_size=res2)
        metrics2 = calculate_metrics(mask2, gt_mask)
        print(f"  F1: {metrics2['f1']:.4f}, IoU: {metrics2['iou']:.4f}\n")
        
        all_results_2.append({
            'image_name': img_path.name,
            'metrics': metrics2,
            'mask': mask2
        })
        
        # Create per-image comparison
        output_path = OUTPUT_DIR / f"{img_path.stem}_comparison.png"
        print(f"Generating comparison visualization...")
        create_per_image_comparison(
            img_path.name, original_img, gt_mask,
            mask1, mask2, metrics1, metrics2,
            config1, config2, output_path
        )
        print(f"✓ Saved: {output_path.name}\n")
    
    # Create aggregate comparison
    if len(all_results_1) > 0:
        print("=" * 80)
        print("Generating aggregate comparison...")
        print("=" * 80)
        
        agg_output_path = OUTPUT_DIR / "aggregate_comparison.png"
        create_aggregate_comparison(all_results_1, all_results_2, config1, config2, agg_output_path)
        print(f"✓ Saved: {agg_output_path.name}\n")
        
        # Save JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_1': {
                'config': config1.get('config', {}) if config1 else {},
                'results': config1.get('results', {}) if config1 else {},
                'test_metrics': {
                    'images': [r['image_name'] for r in all_results_1],
                    'f1_scores': [r['metrics']['f1'] for r in all_results_1],
                    'iou_scores': [r['metrics']['iou'] for r in all_results_1],
                    'average_f1': float(np.mean([r['metrics']['f1'] for r in all_results_1])),
                    'average_iou': float(np.mean([r['metrics']['iou'] for r in all_results_1])),
                }
            },
            'model_2': {
                'config': config2.get('config', {}) if config2 else {},
                'results': config2.get('results', {}) if config2 else {},
                'test_metrics': {
                    'images': [r['image_name'] for r in all_results_2],
                    'f1_scores': [r['metrics']['f1'] for r in all_results_2],
                    'iou_scores': [r['metrics']['iou'] for r in all_results_2],
                    'average_f1': float(np.mean([r['metrics']['f1'] for r in all_results_2])),
                    'average_iou': float(np.mean([r['metrics']['iou'] for r in all_results_2])),
                }
            },
            'comparison': {
                'config_differences': compare_configs(config1, config2),
                'recommendations': generate_recommendations(config1, config2, all_results_1, all_results_2)
            }
        }
        
        json_output_path = OUTPUT_DIR / "comparison_summary.json"
        with open(json_output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved JSON summary: {json_output_path.name}\n")
    
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - Per-image comparisons: <image>_comparison.png")
    print("  - Aggregate comparison: aggregate_comparison.png")
    print("  - JSON summary: comparison_summary.json")


if __name__ == "__main__":
    main()
