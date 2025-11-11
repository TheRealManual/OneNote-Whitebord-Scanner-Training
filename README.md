# OneNote Whiteboard Scanner - Training Repository

This repository contains all training data, models, and tools for training the whiteboard stroke segmentation model.

**Related Repository:** [OneNote-Whiteboard-Scanner](../OneNote-Whiteboard-Scanner) - Main application repo

## Repository Structure

```
OneNote-Whitebord-Scanner-Training/
├── dataset/
│   ├── images/          # Training images (whiteboard photos)
│   └── masks/           # Segmentation masks (labeled)
├── models/              # Trained models and checkpoints
│   ├── whiteboard_seg_best.pt       # Best PyTorch model
│   ├── whiteboard_seg_final.pt      # Final PyTorch model
│   ├── whiteboard_seg.onnx          # ONNX export (if available)
│   ├── whiteboard_seg_int8.onnx     # Quantized INT8 model
│   └── training_history.json        # Training runs history
├── training_ui/
│   └── templates/       # HTML templates for training UI
├── train_segmentation.py    # Training script
├── training_ui.py           # Web-based training UI server
├── start-training-ui.bat    # Quick start script for training UI
├── ML_TRAINING_GUIDE.md     # Detailed ML training guide
└── TRAINING_UI_GUIDE.md     # Training UI user guide
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision pillow numpy flask flask-cors onnxruntime
```

**Optional (for ONNX export):**
```bash
pip install onnx  # May fail on Windows due to path limits
```

### 2. Prepare Dataset

Add your labeled data:
- Place whiteboard images in `dataset/images/`
- Place corresponding segmentation masks in `dataset/masks/`

**Mask format:**
- Grayscale PNG images
- Pixel values: 0=background, 85=stroke, 170=smudge, 255=shadow

**Recommended:**
- Minimum: 20-50 image/mask pairs
- Ideal: 80-150 pairs for good model quality

### 3. Option A: Train via UI (Recommended)

1. Run the training UI:
   ```bash
   python training_ui.py
   ```
   Or double-click `start-training-ui.bat`

2. Open browser: http://localhost:5001

3. Upload images and masks, configure training, and start!

### 4. Option B: Train via Command Line

```bash
python train_segmentation.py --data-dir dataset --output-dir models --epochs 25 --batch-size 4 --lr 0.001
```

**Arguments:**
- `--data-dir` - Dataset directory (default: `dataset`)
- `--output-dir` - Output directory for models (default: `models`)
- `--epochs` - Number of training epochs (default: 25)
- `--batch-size` - Batch size (default: 4)
- `--lr` - Learning rate (default: 0.001)

## Model Outputs

After training, you'll find:

1. **PyTorch Models:**
   - `models/whiteboard_seg_best.pt` - Best model (use this!)
   - `models/whiteboard_seg_final.pt` - Final epoch model

2. **ONNX Models (if onnx installed):**
   - `models/whiteboard_seg.onnx` - Float32 ONNX model
   - `models/whiteboard_seg_int8.onnx` - Quantized INT8 model (3-5 MB)

## Deploying Trained Models

Copy the trained INT8 ONNX model to the main OneNote-Whiteboard-Scanner repo:

```bash
copy models\whiteboard_seg_int8.onnx ..\OneNote-Whiteboard-Scanner\local-ai-backend\models\
```

Update `config_hybrid.json` to use your trained model:
```json
{
  "tile_segmentation": {
    "use_tile_segmentation": true,
    "model_path": "models/whiteboard_seg_int8.onnx"
  }
}
```

## Training Tips

### Getting Good Results

1. **Diverse Data:** Capture whiteboards with various:
   - Lighting conditions (bright, dim, uneven)
   - Writing styles (markers, dry erase, different colors)
   - Backgrounds (clean whiteboards, smudges, shadows)

2. **Quality Labels:** Ensure masks accurately label:
   - Class 1 (stroke): All pen/marker strokes
   - Class 2 (smudge): Eraser marks, fingerprints
   - Class 3 (shadow): Shadows from objects or people

3. **Training Settings:**
   - Start with default settings (25 epochs, batch size 4)
   - If overfitting (val_loss increases): reduce epochs or add data
   - If underfitting (both losses high): increase epochs or learning rate

### Labeling Tools

Use CVAT (https://cvat.ai) or similar for creating masks:
1. Upload images to CVAT
2. Use "Polygon" annotation mode
3. Label all strokes, smudges, and shadows
4. Export as PNG masks with proper class values

## Troubleshooting

**"Target X is out of bounds"**
- Masks must use exact pixel values: 0, 85, 170, 255
- Ensure masks are grayscale PNG, not RGB

**"BatchNorm error"**
- Need at least 2 images for training
- Or use batch_size=1 (model auto-switches to eval mode)

**"ONNX install fails"**
- Windows path length issue
- Skip ONNX export and use PyTorch model
- Or install in shorter path (C:\Python313)

## Files to Keep vs. Delete

### Keep in Git:
- `train_segmentation.py`
- `training_ui.py`
- `ML_TRAINING_GUIDE.md`
- `TRAINING_UI_GUIDE.md`
- `start-training-ui.bat`
- `README.md` (this file)

### Don't commit to Git:
- `dataset/` - Large image files
- `models/*.pt` - Large PyTorch checkpoints
- `models/*.onnx` - Model files
- `models/training_history.json` - Can be large

Add to `.gitignore`:
```
dataset/
models/*.pt
models/*.onnx
models/training_history.json
```

## Support

For questions or issues:
1. Check `ML_TRAINING_GUIDE.md` for detailed training info
2. Check `TRAINING_UI_GUIDE.md` for UI usage
3. Review main repo documentation

---


