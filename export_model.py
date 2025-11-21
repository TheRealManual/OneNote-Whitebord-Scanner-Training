"""
Export trained PyTorch model to TorchScript format for deployment

PRODUCTION SETTINGS:
- Input shape: 768×1024 (H×W) - matches training and scanner
- Format: TorchScript (.pts) for fast inference
- Output: models/whiteboard_seg.pts (42.6 MB)

This exported model will be used by the scanner with:
- Tile size: 768×1024
- Overlap: 50%
- Method: Smooth Gaussian-blended tiling

Usage:
    python export_model.py
"""

import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from pathlib import Path


class ModelWrapper(torch.nn.Module):
    """Wrapper to extract 'out' from DeepLabV3 output dict"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output['out']
        return output


def export_to_torchscript(model_path, output_path, num_classes=2):
    """
    Export trained model to TorchScript (.pts) format
    
    Args:
        model_path: Path to trained .pt model
        output_path: Path to save .pts TorchScript model
        num_classes: Number of output classes (2 for binary segmentation)
    """
    print(f"Loading model from: {model_path}")
    
    # Load model architecture
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None  # Remove auxiliary classifier
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Wrap model to return tensor instead of dict
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model has {num_classes} output classes")
    
    # Create example input for tracing - USE TRAINING RESOLUTION (2560×2560)
    # This matches the resolution used for the best model
    example_input = torch.randn(1, 3, 2560, 2560)
    
    # Trace model
    print(f"Tracing model with input shape: {example_input.shape}")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, example_input)
    
    # Save TorchScript model
    print(f"Saving TorchScript model to: {output_path}")
    traced_model.save(str(output_path))
    
    # Verify saved model
    print(f"Verifying saved model...")
    loaded = torch.jit.load(str(output_path))
    test_output = loaded(example_input)
    print(f"✓ Model loads and runs correctly")
    print(f"✓ Output shape: {test_output.shape}")
    
    # Check file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {size_mb:.2f} MB")
    
    print(f"\n✅ Export successful!")
    return output_path


if __name__ == "__main__":
    # Export model 2 (the 2560x2560 trained model)
    model_dir = Path(__file__).parent / "models_1"
    
    # Try best model first, then final model
    if (model_dir / "whiteboard_seg_best.pt").exists():
        model_path = model_dir / "whiteboard_seg_best.pt"
        print("Using BEST model (highest validation IoU)")
    elif (model_dir / "whiteboard_seg_final.pt").exists():
        model_path = model_dir / "whiteboard_seg_final.pt"
        print("Using FINAL model (last epoch)")
    else:
        raise FileNotFoundError("No trained model found in models_1! Train a model first.")
    
    output_path = model_dir / "whiteboard_seg.pts"
    
    export_to_torchscript(model_path, output_path, num_classes=2)
    
    print(f"\n📦 Ready for deployment!")
    print(f"Copy this file to the scanner's models folder:")
    print(f"  {output_path}")
