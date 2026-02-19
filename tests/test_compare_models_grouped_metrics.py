"""
Branch 4 — test-split-config: verify grouped metrics output from compare_models.

Tests:
  1. calculate_metrics() returns all expected keys (including boundary_f1)
  2. Grouped metrics helper can separate core and thin IDs
  3. Edge case: empty group returns empty dict
"""

import sys
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))
from compare_models import calculate_metrics


def _rect_mask(h=64, w=64, top=10, left=10, bottom=54, right=54):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def _fake_results(image_names, gt, preds):
    """Build a list of result dicts matching compare_models format."""
    results = []
    for name, pred in zip(image_names, preds):
        metrics = calculate_metrics(pred, gt)
        results.append({'image_name': name, 'metrics': metrics})
    return results


def _group_avg(results, ids):
    """Same grouping logic as compare_models main()."""
    group = []
    for r in results:
        m = re.match(r'^(image_\d+)', Path(r['image_name']).stem)
        base_id = m.group(1) if m else ''
        if base_id in ids:
            group.append(r)
    if not group:
        return {}
    return {
        'count': len(group),
        'average_f1': float(np.mean([r['metrics']['f1'] for r in group])),
        'average_iou': float(np.mean([r['metrics']['iou'] for r in group])),
        'average_boundary_f1': float(np.mean([r['metrics']['boundary_f1'] for r in group])),
    }


class TestGroupedMetrics:
    def test_core_group_filters_correctly(self):
        gt = _rect_mask()
        names = ["image_3.png", "image_13.png", "image_22.png"]
        preds = [_rect_mask(top=12, left=12, bottom=56, right=56)] * 3
        results = _fake_results(names, gt, preds)

        core_ids = {"image_3", "image_13"}
        grp = _group_avg(results, core_ids)
        assert grp['count'] == 2
        assert 0.0 <= grp['average_f1'] <= 1.0

    def test_thin_group_filters_correctly(self):
        gt = _rect_mask()
        names = ["image_3.png", "image_22.png", "image_24.png"]
        preds = [_rect_mask()] * 3  # perfect match
        results = _fake_results(names, gt, preds)

        thin_ids = {"image_22", "image_24"}
        grp = _group_avg(results, thin_ids)
        assert grp['count'] == 2
        assert grp['average_f1'] == 1.0

    def test_empty_group_returns_empty_dict(self):
        gt = _rect_mask()
        results = _fake_results(["image_3.png"], gt, [_rect_mask()])
        grp = _group_avg(results, {"image_99"})
        assert grp == {}

    def test_overall_includes_all(self):
        gt = _rect_mask()
        names = ["image_3.png", "image_22.png"]
        preds = [_rect_mask()] * 2
        results = _fake_results(names, gt, preds)

        all_ids = {"image_3", "image_22"}
        grp = _group_avg(results, all_ids)
        assert grp['count'] == 2
