"""
Shared pytest fixtures for the whiteboard segmentation test suite.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.synthetic_data import populate_synthetic_dataset


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory (pytest built-in tmp_path)."""
    return tmp_path


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a small synthetic dataset in a temp dir and return its path."""
    ds_root = tmp_path / "dataset"
    info = populate_synthetic_dataset(ds_root, n_originals=5, n_augments=3, img_size=(64, 64))
    return ds_root, info


@pytest.fixture
def synthetic_dataset_10(tmp_path):
    """Create 10 synthetic image/mask pairs (no augments) for DataLoader tests."""
    ds_root = tmp_path / "dataset"
    info = populate_synthetic_dataset(ds_root, n_originals=10, n_augments=0, img_size=(64, 64))
    return ds_root, info


@pytest.fixture
def output_dir(tmp_path):
    """Provide a clean temp output directory (never the private repo)."""
    out = tmp_path / "output"
    out.mkdir()
    return out
