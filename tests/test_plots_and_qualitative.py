"""
Branch 6 — results-pipeline: test plot generation and qualitative grid helpers.

Tests:
  1. generate_plots produces .png files from mock CSV data
  2. create_error_overlay returns correct shape and dtype
  3. generate_grid produces a .png file
"""

import sys
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_models"))

from generate_plots import plot_loss_study, plot_resolution_study, safe_float
from generate_qualitative import create_error_overlay, generate_grid


def _write_mock_csv(csv_path, records):
    """Write a mock all_runs.csv."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _mock_records():
    """Generate mock experiment records."""
    records = []
    for loss in ('dice', 'focal', 'dice_focal'):
        for seed in (42, 123, 7):
            records.append({
                'run_name': f'{loss}_seed{seed}',
                'group': 'loss_study',
                'loss_type': loss,
                'seed': str(seed),
                'img_height': '1152',
                'img_width': '1536',
                'resolution': '1536x1152',
                'best_val_f1': str(np.random.uniform(0.7, 0.95)),
                'best_val_iou': str(np.random.uniform(0.6, 0.85)),
            })
    for res, h, w in [('1024x768', '768', '1024'), ('1536x1152', '1152', '1536')]:
        for seed in (42, 123, 7):
            records.append({
                'run_name': f'{res}_seed{seed}',
                'group': 'resolution_study',
                'loss_type': 'dice_focal',
                'seed': str(seed),
                'img_height': h,
                'img_width': w,
                'resolution': res,
                'best_val_f1': str(np.random.uniform(0.7, 0.95)),
                'best_val_iou': str(np.random.uniform(0.6, 0.85)),
            })
    return records


class TestPlotLossStudy:
    def test_produces_png(self, tmp_path):
        records = _mock_records()
        plot_loss_study(records, tmp_path)
        png = tmp_path / "loss_study_bar.png"
        assert png.exists()
        assert png.stat().st_size > 0


class TestPlotResolutionStudy:
    def test_produces_png(self, tmp_path):
        records = _mock_records()
        plot_resolution_study(records, tmp_path)
        png = tmp_path / "resolution_study_bar.png"
        assert png.exists()
        assert png.stat().st_size > 0


class TestSafeFloat:
    def test_valid_float(self):
        assert safe_float('0.85') == 0.85

    def test_empty_string(self):
        assert safe_float('') == 0.0

    def test_none(self):
        assert safe_float(None) == 0.0


class TestErrorOverlay:
    def test_returns_correct_shape(self):
        original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[10:30, 10:30] = 255
        gt = np.zeros((64, 64), dtype=np.uint8)
        gt[15:35, 15:35] = 255

        overlay = create_error_overlay(original, pred, gt)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8


class TestGenerateGrid:
    def test_produces_png(self, tmp_path):
        # Create synthetic image and mask files
        import cv2
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        for i in range(3):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[10:50, 10:50] = 255
            cv2.imwrite(str(img_dir / f"image_{i}.png"), img)
            cv2.imwrite(str(mask_dir / f"image_{i}.png"), mask)

        pred_masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(3)]
        for p in pred_masks:
            p[12:48, 12:48] = 255

        image_paths = sorted(img_dir.glob("*.png"))
        mask_paths = [mask_dir / p.name for p in image_paths]

        grid_path = tmp_path / "grid.png"
        generate_grid(image_paths, mask_paths, pred_masks, grid_path, max_images=3)
        assert grid_path.exists()
        assert grid_path.stat().st_size > 0
