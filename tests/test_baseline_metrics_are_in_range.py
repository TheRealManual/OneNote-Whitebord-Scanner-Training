"""
Branch 5 — classical-baseline: verify all baseline metrics are numeric and in [0, 1].

Tests:
  1. Per-image metrics are floats in [0, 1]
  2. Aggregate mean/std are floats in [0, 1]
  3. Both adaptive and otsu methods produce valid ranges
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))

from classical_baseline import run_baseline
from tests.helpers.synthetic_data import populate_synthetic_dataset

METRIC_KEYS = ('iou', 'f1', 'precision', 'recall', 'pixel_acc', 'dice',
               'edge_iou', 'boundary_f1')


class TestBaselineMetricsInRange:
    def _run(self, tmp_path, method):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=3, n_augments=0,
                                   img_size=(64, 64))
        return run_baseline(
            ds_root / "images", ds_root / "masks", method,
            img_height=64, img_width=64,
        )

    def test_adaptive_per_image_metrics_in_range(self, tmp_path):
        result = self._run(tmp_path, 'adaptive')
        for item in result['per_image']:
            for key in METRIC_KEYS:
                val = item['metrics'][key]
                assert isinstance(val, (int, float)), f"{key} is not numeric: {type(val)}"
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_otsu_per_image_metrics_in_range(self, tmp_path):
        result = self._run(tmp_path, 'otsu')
        for item in result['per_image']:
            for key in METRIC_KEYS:
                val = item['metrics'][key]
                assert isinstance(val, (int, float)), f"{key} is not numeric: {type(val)}"
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_aggregate_mean_in_range(self, tmp_path):
        result = self._run(tmp_path, 'adaptive')
        for key in METRIC_KEYS:
            if key in result['results']:
                mean_val = result['results'][key]['mean']
                assert 0.0 <= mean_val <= 1.0, f"Mean {key}={mean_val} out of [0,1]"

    def test_aggregate_std_non_negative(self, tmp_path):
        result = self._run(tmp_path, 'otsu')
        for key in METRIC_KEYS:
            if key in result['results']:
                std_val = result['results'][key]['std']
                assert std_val >= 0.0, f"Std {key}={std_val} is negative"
