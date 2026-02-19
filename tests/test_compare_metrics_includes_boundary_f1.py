"""
Branch 3 — boundary-metrics: verify calculate_metrics() includes boundary_f1.

Tests:
  1. boundary_f1 key is present in returned dict
  2. boundary_f1 value is a float in [0, 1]
  3. Perfect overlap gives boundary_f1 == 1.0
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))
from compare_models import calculate_metrics


def _rect_mask(h=64, w=64, top=10, left=10, bottom=54, right=54):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


class TestCalculateMetricsIncludesBF1:
    def test_boundary_f1_key_present(self):
        gt = _rect_mask()
        pred = _rect_mask(top=12, left=12, bottom=56, right=56)
        metrics = calculate_metrics(pred, gt)
        assert 'boundary_f1' in metrics

    def test_boundary_f1_is_float_in_range(self):
        gt = _rect_mask()
        pred = _rect_mask(top=12, left=12, bottom=56, right=56)
        metrics = calculate_metrics(pred, gt)
        bf1 = metrics['boundary_f1']
        assert isinstance(bf1, float)
        assert 0.0 <= bf1 <= 1.0

    def test_perfect_overlap_boundary_f1_is_one(self):
        mask = _rect_mask()
        metrics = calculate_metrics(mask, mask)
        assert metrics['boundary_f1'] == 1.0

    def test_both_empty_boundary_f1_is_one(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        metrics = calculate_metrics(empty, empty)
        assert metrics['boundary_f1'] == 1.0

    def test_all_original_metrics_still_present(self):
        """Ensure adding BF1 didn't break existing metrics."""
        gt = _rect_mask()
        pred = _rect_mask()
        metrics = calculate_metrics(pred, gt)
        expected_keys = {'iou', 'f1', 'precision', 'recall', 'pixel_acc',
                         'dice', 'edge_iou', 'boundary_f1', 'tp', 'fp', 'fn', 'tn',
                         'total_pixels', 'stroke_pixels_pred', 'stroke_pixels_gt'}
        assert expected_keys.issubset(set(metrics.keys()))
