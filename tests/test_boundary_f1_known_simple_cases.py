"""
Branch 3 — boundary-metrics: test boundary_f1() on known synthetic cases.

Tests:
  1. Perfect match → BF1 == 1.0
  2. Shifted mask → BF1 < 1.0
  3. Both empty → BF1 == 1.0
  4. One empty → BF1 == 0.0
  5. Small tolerance vs large tolerance
  6. Result always in [0, 1]
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))
from compare_models import boundary_f1


def _rect_mask(h=64, w=64, top=10, left=10, bottom=54, right=54):
    """Create a mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


class TestBoundaryF1PerfectMatch:
    def test_identical_masks_give_bf1_one(self):
        mask = _rect_mask()
        assert boundary_f1(mask, mask) == 1.0

    def test_identical_circle_masks(self):
        import cv2
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 40, 255, -1)
        assert boundary_f1(mask, mask) == 1.0


class TestBoundaryF1Shifted:
    def test_shifted_mask_bf1_below_one(self):
        gt = _rect_mask()
        pred = _rect_mask(top=14, left=14, bottom=58, right=58)
        score = boundary_f1(gt, pred)
        assert 0.0 < score < 1.0

    def test_larger_shift_gives_lower_bf1(self):
        gt = _rect_mask()
        pred_small = _rect_mask(top=12, left=12, bottom=56, right=56)
        pred_large = _rect_mask(top=20, left=20, bottom=60, right=60)
        assert boundary_f1(gt, pred_small) > boundary_f1(gt, pred_large)


class TestBoundaryF1EdgeCases:
    def test_both_empty_returns_one(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert boundary_f1(empty, empty) == 1.0

    def test_pred_empty_gt_not_returns_zero(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        gt = _rect_mask()
        assert boundary_f1(empty, gt) == 0.0

    def test_gt_empty_pred_not_returns_zero(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        pred = _rect_mask()
        assert boundary_f1(pred, empty) == 0.0


class TestBoundaryF1Tolerance:
    def test_larger_tolerance_gives_higher_or_equal_bf1(self):
        gt = _rect_mask()
        pred = _rect_mask(top=14, left=14, bottom=58, right=58)
        bf1_t1 = boundary_f1(gt, pred, tolerance=1)
        bf1_t5 = boundary_f1(gt, pred, tolerance=5)
        assert bf1_t5 >= bf1_t1

    def test_auto_tolerance_is_at_least_1(self):
        # For a 64×64 image: max(64,64)/1536 * 2 ≈ 0.08 → rounds to 0, but min is 1
        gt = _rect_mask()
        pred = _rect_mask(top=11, left=11, bottom=55, right=55)
        score = boundary_f1(gt, pred)  # auto tolerance
        assert 0.0 <= score <= 1.0


class TestBoundaryF1InRange:
    def test_random_masks_bf1_in_zero_one(self):
        rng = np.random.RandomState(42)
        for _ in range(10):
            pred = (rng.rand(64, 64) > 0.7).astype(np.uint8) * 255
            gt = (rng.rand(64, 64) > 0.7).astype(np.uint8) * 255
            score = boundary_f1(pred, gt)
            assert 0.0 <= score <= 1.0, f"BF1 out of range: {score}"
