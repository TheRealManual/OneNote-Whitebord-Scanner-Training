# OneNote Whiteboard Scanner - Training Repository

This repository contains training data, models, and tools for training the whiteboard stroke segmentation model using DeepLabV3 + MobileNetV3-Large.

**Related Repository:** [OneNote-Whiteboard-Scanner](../OneNote-Whiteboard-Scanner) - Main application repo

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production training code |
| `paper/infra` | Research paper branch — loss ablation, boundary metrics, experiment pipeline |

> The `paper/infra` branch adds research features (seed determinism, 5 loss functions, boundary F1 metrics, test-split configuration, classical baselines, and a full experiment runner). It is never merged into `main`.

## Repository Structure

```
OneNote-Whitebord-Scanner-Training/
├── dataset/
│   ├── images/              # Training images (whiteboard photos)
│   ├── masks/               # Segmentation masks (binary: 0=bg, 255=stroke)
│   ├── generate_augmented_dataset.py
│   ├── check_all_masks.py
│   └── fix_all_masks.py
├── models_1/                # Production model checkpoint
│   ├── whiteboard_seg_best.pt
│   ├── whiteboard_seg_final.pt
│   └── whiteboard_seg.pts
├── compare_models/          # Model comparison tool
│   └── compare_models.py
├── scripts/                 # Experiment & analysis scripts (paper/infra)
│   ├── run_experiments.bat  # Full experiment matrix runner
│   ├── aggregate_results.py # CSV + LaTeX table generation
│   ├── generate_plots.py    # Bar charts, training curves
│   ├── generate_qualitative.py  # Visual comparison grids
│   └── classical_baseline.py    # Adaptive threshold + Otsu baseline
├── tests/                   # 80 tests (paper/infra)
│   ├── conftest.py
│   └── ...
├── train_segmentation.py    # Main training script
├── export_model.py          # ONNX/TorchScript export
├── training_ui.py           # Web-based training UI
├── check_cuda.py            # GPU detection utility
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `pillow`, `numpy`, `opencv-python`, `tqdm`, `matplotlib`, `scipy`

Optional: `onnx` (for ONNX export), `flask` + `flask-cors` (for training UI)

### 2. Prepare Dataset

- Place whiteboard images in `dataset/images/`
- Place corresponding binary masks in `dataset/masks/`
- Mask format: grayscale PNG, 0 = background, 255 = stroke
- 34 original images + 340 augmented variants included

### 3. Train via Command Line

```bash
python train_segmentation.py --data-dir dataset --output-dir models --epochs 100 --batch-size 2 --lr 0.0002
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 2e-4 | Learning rate |
| `--img-height` | 768 | Input image height |
| `--img-width` | 1024 | Input image width |
| `--loss` | dice_focal | Loss function: `ce`, `dice`, `focal`, `dice_focal`, `tversky` |
| `--seed` | None | Random seed for reproducibility |
| `--deterministic` | off | Enable deterministic algorithms (warn-only mode) |
| `--use-amp` | off | Mixed precision training |
| `--test-split-config` | None | Path to test_splits.json for holdout exclusion |

### 4. Train via UI

```bash
python training_ui.py
```
Open http://localhost:5001

## Model Architecture

- **Backbone:** DeepLabV3 with MobileNetV3-Large (ImageNet pretrained)
- **Task:** Binary segmentation (background vs stroke)
- **Parameters:** ~11M
- **Exports:** PyTorch (.pt), ONNX (.onnx), TorchScript (.pts)

## Research Paper (paper/infra branch)

The `paper/infra` branch contains all infrastructure for a research paper on thin-structure segmentation under extreme class imbalance.

### Experiment Matrix

| Experiment | Runs | Description |
|-----------|------|-------------|
| Loss ablation | 15 | 5 losses × 3 seeds at 1152×1536 |
| Resolution study | 6 | 2 resolutions × 3 seeds with dice_focal |
| Classical baseline | 1 | Adaptive threshold + Otsu |

### Loss Functions

| Key | Loss | Notes |
|-----|------|-------|
| `ce` | Cross-Entropy | Standard baseline |
| `dice` | Dice | Region overlap |
| `focal` | Focal (α=0.25, γ=2) | Hard example focus |
| `dice_focal` | Dice + Focal (0.6/0.4) | Combined (default) |
| `tversky` | Tversky (α=0.3, β=0.7) | Recall-weighted Dice generalization |

### Metrics

- Pixel Accuracy, IoU, F1, Precision, Recall
- Boundary F1 (BF1) with resolution-scaled tolerance

### Running Experiments

```bash
scripts\run_experiments.bat
```

Outputs go to the private companion repo (`../SegmentationResearchPaper/`).

## Test Suite (paper/infra)

```bash
pytest tests/ -v
```

80 tests covering: seed determinism, loss functions, boundary metrics, test-split filtering, classical baselines, output isolation, results pipeline.

## Deploying Trained Models

Copy the ONNX model to the main scanner application:

```bash
copy models_1\whiteboard_seg.onnx ..\OneNote-Whiteboard-Scanner\local-ai-backend\models\
```
