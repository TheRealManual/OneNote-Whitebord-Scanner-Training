"""
Generate publication-quality plots from aggregated experiment results.

Produces:
  1. Bar charts: IoU/F1/EdgeIoU/BF1 by loss function (with error bars)
  2. Thin vs core performance comparison
  3. Resolution effect curves

Usage:
    python scripts/generate_plots.py --results-dir ../SegmentationResearchPaper/results
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


PRIVATE_REPO = Path(__file__).resolve().parent.parent.parent / "SegmentationResearchPaper"


def load_csv(csv_path):
    """Load all_runs.csv into a list of dicts."""
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def safe_float(val, default=0.0):
    """Convert to float, returning default if empty or invalid."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def plot_loss_study(records, output_dir):
    """Bar chart of F1/IoU by loss function with error bars."""
    loss_records = [r for r in records if r.get('group') == 'loss_study']
    if not loss_records:
        print("No loss study data — skipping loss plot")
        return

    # Group by loss type
    groups = {}
    for r in loss_records:
        lt = r.get('loss_type', 'unknown')
        groups.setdefault(lt, {'f1': [], 'iou': []})
        groups[lt]['f1'].append(safe_float(r.get('best_val_f1')))
        groups[lt]['iou'].append(safe_float(r.get('best_val_iou')))

    labels = sorted(groups.keys())
    f1_means = [np.mean(groups[l]['f1']) for l in labels]
    f1_stds = [np.std(groups[l]['f1']) for l in labels]
    iou_means = [np.mean(groups[l]['iou']) for l in labels]
    iou_stds = [np.std(groups[l]['iou']) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, label='F1', capsize=5)
    bars2 = ax.bar(x + width/2, iou_means, width, yerr=iou_stds, label='IoU', capsize=5)

    ax.set_xlabel('Loss Function')
    ax.set_ylabel('Score')
    ax.set_title('Loss Function Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "loss_study_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_resolution_study(records, output_dir):
    """Bar chart of F1 by resolution."""
    res_records = [r for r in records if r.get('group') == 'resolution_study']
    if not res_records:
        print("No resolution study data — skipping resolution plot")
        return

    groups = {}
    for r in res_records:
        res = r.get('resolution', 'unknown')
        groups.setdefault(res, []).append(safe_float(r.get('best_val_f1')))

    labels = sorted(groups.keys())
    means = [np.mean(groups[l]) for l in labels]
    stds = [np.std(groups[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=stds, capsize=5, color=['#4C72B0', '#55A868'])
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Val F1')
    ax.set_title('Resolution Effect on F1')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "resolution_study_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_training_curves(experiments_dir, output_dir):
    """Plot training/val loss curves from training_history.json files."""
    histories = sorted(experiments_dir.rglob("training_history.json"))
    if not histories:
        print("No training history files — skipping curves")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for h_path in histories[:6]:  # Limit to 6 for readability
        with open(h_path) as f:
            data = json.load(f)
        label = h_path.parent.name
        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
        val_f1 = data.get('val_f1', [])

        if train_loss:
            ax1.plot(train_loss, label=f'{label} (train)', alpha=0.7)
        if val_loss:
            ax1.plot(val_loss, label=f'{label} (val)', linestyle='--', alpha=0.7)
        if val_f1:
            ax2.plot(val_f1, label=label, alpha=0.7)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--results-dir", type=str,
                       default=str(PRIVATE_REPO / "results"),
                       help="Directory containing all_runs.csv")
    parser.add_argument("--experiments-dir", type=str,
                       default=str(PRIVATE_REPO / "experiments"),
                       help="Root of experiment directories (for training curves)")
    parser.add_argument("--output-dir", type=str,
                       default=str(PRIVATE_REPO / "results" / "plots"),
                       help="Where to save plot images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.results_dir) / "all_runs.csv"
    if csv_path.exists():
        records = load_csv(csv_path)
        print(f"Loaded {len(records)} runs from {csv_path}")
        plot_loss_study(records, output_dir)
        plot_resolution_study(records, output_dir)
    else:
        print(f"CSV not found: {csv_path}")
        print("Run scripts/aggregate_results.py first.")

    experiments_dir = Path(args.experiments_dir)
    if experiments_dir.exists():
        plot_training_curves(experiments_dir, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
