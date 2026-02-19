"""
Branch 5 — classical-baseline: verify the baseline runs and outputs correct JSON schema.

Tests:
  1. adaptive_threshold returns uint8 binary mask of correct shape
  2. otsu_threshold returns uint8 binary mask of correct shape
  3. morphological_cleanup preserves shape and type
  4. run_baseline produces correct schema on synthetic data
  5. Output JSON has required top-level keys
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))

from classical_baseline import (
    adaptive_threshold,
    otsu_threshold,
    morphological_cleanup,
    run_baseline,
)
from tests.helpers.synthetic_data import populate_synthetic_dataset


class TestAdaptiveThreshold:
    def test_returns_correct_shape_and_dtype(self):
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = adaptive_threshold(gray)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_output_is_binary(self):
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        mask = adaptive_threshold(gray)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})


class TestOtsuThreshold:
    def test_returns_correct_shape_and_dtype(self):
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = otsu_threshold(gray)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_output_is_binary(self):
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        mask = otsu_threshold(gray)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})


class TestMorphologicalCleanup:
    def test_preserves_shape_and_type(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        cleaned = morphological_cleanup(mask)
        assert cleaned.shape == mask.shape
        assert cleaned.dtype == np.uint8

    def test_removes_small_noise(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Single pixel noise
        mask[30, 30] = 255
        cleaned = morphological_cleanup(mask, kernel_size=2)
        assert cleaned[30, 30] == 0


class TestRunBaseline:
    def test_adaptive_baseline_produces_schema(self, tmp_path):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=3, n_augments=0,
                                   img_size=(64, 64))
        result = run_baseline(
            ds_root / "images", ds_root / "masks", 'adaptive',
            img_height=64, img_width=64,
        )
        assert result['method'] == 'adaptive'
        assert 'results' in result
        assert 'per_image' in result
        assert result['num_images'] == 3

    def test_otsu_baseline_produces_schema(self, tmp_path):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=3, n_augments=0,
                                   img_size=(64, 64))
        result = run_baseline(
            ds_root / "images", ds_root / "masks", 'otsu',
            img_height=64, img_width=64,
        )
        assert result['method'] == 'otsu'
        assert result['num_images'] == 3

    def test_aggregate_metrics_have_required_keys(self, tmp_path):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=3, n_augments=0,
                                   img_size=(64, 64))
        result = run_baseline(
            ds_root / "images", ds_root / "masks", 'adaptive',
            img_height=64, img_width=64,
        )
        agg = result['results']
        for key in ('iou', 'f1', 'precision', 'recall', 'dice', 'boundary_f1'):
            assert key in agg, f"Missing aggregate key: {key}"
            assert 'mean' in agg[key]
            assert 'std' in agg[key]
