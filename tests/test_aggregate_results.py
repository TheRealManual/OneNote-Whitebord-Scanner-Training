"""
Branch 6 — results-pipeline: test aggregate_results on mock experiment data.

Tests:
  1. find_training_histories finds JSON files recursively
  2. parse_run extracts correct fields from training_history.json
  3. write_csv produces valid CSV
  4. group_stats computes correct mean/std
  5. latex_table produces valid LaTeX string
"""

import sys
import json
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from aggregate_results import (
    find_training_histories,
    parse_run,
    write_csv,
    group_stats,
    latex_table,
)


def _make_mock_history(output_dir, run_name="dice_focal_seed42", group="loss_study",
                       loss_type="dice_focal", seed=42, f1=0.85, iou=0.78):
    """Create a minimal training_history.json in a mock experiment structure."""
    run_dir = output_dir / group / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = {
        'train_loss': [0.5, 0.3, 0.2],
        'val_loss': [0.4, 0.25, 0.18],
        'val_iou': [0.6, 0.7, iou],
        'val_f1': [0.7, 0.8, f1],
        'config': {
            'model': 'DeepLabV3-MobileNetV3-Large',
            'num_classes': 2,
            'epochs': 3,
            'batch_size': 2,
            'learning_rate': 0.0002,
            'optimizer': 'AdamW',
            'weight_decay': 0.0001,
            'loss_type': loss_type,
            'loss_function': f'{loss_type}',
            'scheduler': 'CosineAnnealingLR with warmup',
            'warmup_epochs': 5,
            'patience': 15,
            'img_height': 1152,
            'img_width': 1536,
            'img_resolution': '1536x1152',
            'num_train_images': 200,
            'num_val_images': 50,
            'use_amp': True,
            'device': 'cuda',
            'seed': seed,
            'deterministic': True,
            'training_start_time': '2026-02-19T10:00:00',
            'data_dir': 'dataset',
        },
        'results': {
            'best_val_loss': 0.18,
            'best_val_f1': f1,
            'best_val_iou': iou,
            'final_epoch': 3,
            'total_training_time_seconds': 120.0,
            'early_stopped': False,
        },
        'epoch_times': [40.0, 40.0, 40.0],
    }
    path = run_dir / "training_history.json"
    path.write_text(json.dumps(history, indent=2))
    return path


class TestFindTrainingHistories:
    def test_finds_files_recursively(self, tmp_path):
        _make_mock_history(tmp_path, "run1", "loss_study")
        _make_mock_history(tmp_path, "run2", "resolution_study")
        found = find_training_histories(tmp_path)
        assert len(found) == 2

    def test_returns_empty_for_empty_dir(self, tmp_path):
        found = find_training_histories(tmp_path)
        assert len(found) == 0


class TestParseRun:
    def test_extracts_correct_fields(self, tmp_path):
        path = _make_mock_history(tmp_path, "dice_seed42", "loss_study",
                                  loss_type="dice", seed=42, f1=0.9, iou=0.85)
        record = parse_run(path)
        assert record['loss_type'] == 'dice'
        assert record['seed'] == 42
        assert record['best_val_f1'] == 0.9
        assert record['best_val_iou'] == 0.85
        assert record['group'] == 'loss_study'
        assert record['run_name'] == 'dice_seed42'


class TestWriteCsv:
    def test_produces_valid_csv(self, tmp_path):
        records = [
            {'name': 'run1', 'f1': 0.85},
            {'name': 'run2', 'f1': 0.90},
        ]
        csv_path = tmp_path / "test.csv"
        write_csv(records, csv_path)
        assert csv_path.exists()
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]['name'] == 'run1'


class TestGroupStats:
    def test_computes_mean_std(self):
        records = [
            {'loss_type': 'dice', 'best_val_f1': 0.8},
            {'loss_type': 'dice', 'best_val_f1': 0.9},
            {'loss_type': 'focal', 'best_val_f1': 0.7},
        ]
        stats = group_stats(records, 'loss_type', 'best_val_f1')
        assert 'dice' in stats
        assert 'focal' in stats
        assert abs(stats['dice']['mean'] - 0.85) < 1e-6
        assert stats['dice']['n'] == 2
        assert stats['focal']['n'] == 1


class TestLatexTable:
    def test_produces_valid_latex(self):
        stats = {
            'dice': {'mean': 0.85, 'std': 0.05, 'n': 3},
            'focal': {'mean': 0.78, 'std': 0.03, 'n': 3},
        }
        tex = latex_table(stats, 'Test Caption', 'tab:test', 'F1')
        assert r'\begin{table}' in tex
        assert r'\end{table}' in tex
        assert 'dice' in tex
        assert '0.8500' in tex
