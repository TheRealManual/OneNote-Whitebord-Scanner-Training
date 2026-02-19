"""
Branch 1 test: verify seed_everything() seeds all RNG sources correctly.

After calling seed_everything(N), random, numpy, and torch must all produce
the same sequence every time for the same seed.
"""

import random

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_segmentation import seed_everything


def test_seed_sets_python_random():
    """random.random() returns the same value after seeding twice with same seed."""
    seed_everything(42)
    val1 = random.random()

    seed_everything(42)
    val2 = random.random()

    assert val1 == val2, f"Python random not reproducible: {val1} != {val2}"


def test_seed_sets_numpy_random():
    """np.random.rand() returns the same value after seeding twice with same seed."""
    seed_everything(42)
    val1 = np.random.rand()

    seed_everything(42)
    val2 = np.random.rand()

    assert val1 == val2, f"NumPy random not reproducible: {val1} != {val2}"


def test_seed_sets_torch_random():
    """torch.rand(1) returns the same value after seeding twice with same seed."""
    seed_everything(42)
    val1 = torch.rand(1).item()

    seed_everything(42)
    val2 = torch.rand(1).item()

    assert val1 == val2, f"Torch random not reproducible: {val1} != {val2}"


def test_different_seeds_give_different_values():
    """Sanity check: different seeds must produce different outputs."""
    seed_everything(42)
    v42 = (random.random(), np.random.rand(), torch.rand(1).item())

    seed_everything(123)
    v123 = (random.random(), np.random.rand(), torch.rand(1).item())

    assert v42 != v123, "Different seeds should give different random values"


def test_seed_sets_cudnn_flags():
    """After seeding, cudnn flags should be set for determinism."""
    seed_everything(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
