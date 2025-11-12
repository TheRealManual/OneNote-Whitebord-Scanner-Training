"""
Quick CUDA Check Script
Run this before training to verify GPU is available and working
"""

import torch
import sys

print("\n" + "="*70)
print("CUDA VERIFICATION")
print("="*70)

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\n" + "-"*70)
    print("Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation test PASSED")
        print(f"✓ GPU is ready for training!")
    except Exception as e:
        print(f"✗ GPU computation test FAILED: {e}")
        cuda_available = False
else:
    print("\n⚠️  CUDA NOT AVAILABLE")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU in system")
    print("  2. NVIDIA drivers not installed")
    print("  3. PyTorch installed without CUDA support")
    print("\nTo install CUDA-enabled PyTorch:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("="*70)

# Exit with error code if CUDA not available
if not cuda_available:
    print("\n⚠️  Training on CPU will be 20-50x slower!")
    sys.exit(1)
else:
    print("\n✓ Ready for GPU training!")
    sys.exit(0)
