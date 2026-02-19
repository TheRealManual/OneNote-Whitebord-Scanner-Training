"""
Aggregate experiment results into CSV and LaTeX tables.

Reads training_history.json files from experiment directories and produces:
  1. CSV table with all runs and metrics
  2. LaTeX tables for the paper (loss study, resolution study)
  3. Summary statistics (mean ± std per group)

Usage:
    python scripts/aggregate_results.py --experiments-dir ../SegmentationResearchPaper/experiments
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


PRIVATE_REPO = Path(__file__).resolve().parent.parent.parent / "SegmentationResearchPaper"


def find_training_histories(experiments_dir):
    """Recursively find all training_history.json files."""
    return sorted(experiments_dir.rglob("training_history.json"))


def parse_run(history_path):
    """Parse a single training_history.json into a flat record."""
    with open(history_path) as f:
        data = json.load(f)

    cfg = data.get('config', {})
    res = data.get('results', {})

    # Infer experiment group from path
    rel = history_path.parent.name  # e.g. "dice_focal_seed42"
    parent_group = history_path.parent.parent.name  # e.g. "loss_study"

    return {
        'run_name': rel,
        'group': parent_group,
        'loss_type': cfg.get('loss_type', ''),
        'seed': cfg.get('seed', ''),
        'img_height': cfg.get('img_height', ''),
        'img_width': cfg.get('img_width', ''),
        'resolution': cfg.get('img_resolution', ''),
        'epochs_trained': res.get('final_epoch', ''),
        'best_val_loss': res.get('best_val_loss', ''),
        'best_val_f1': res.get('best_val_f1', ''),
        'best_val_iou': res.get('best_val_iou', ''),
        'early_stopped': res.get('early_stopped', ''),
        'training_time_s': res.get('total_training_time_seconds', ''),
        'model': cfg.get('model', ''),
        'batch_size': cfg.get('batch_size', ''),
        'learning_rate': cfg.get('learning_rate', ''),
        'path': str(history_path),
    }


def write_csv(records, output_path):
    """Write records to CSV."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def group_stats(records, group_key, metric_key):
    """Compute mean ± std for a metric grouped by a key."""
    groups = {}
    for r in records:
        gk = r.get(group_key, 'unknown')
        val = r.get(metric_key)
        if val is not None and val != '':
            groups.setdefault(gk, []).append(float(val))

    stats = {}
    for gk, vals in groups.items():
        stats[gk] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals),
        }
    return stats


def latex_table(stats, caption, label, metric_name='F1'):
    """Generate a LaTeX table from group stats."""
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\begin{tabular}{lcc}')
    lines.append(r'\toprule')
    lines.append(f'Configuration & {metric_name} (mean $\\pm$ std) & N \\\\')
    lines.append(r'\midrule')
    for gk, s in sorted(stats.items()):
        lines.append(f'{gk} & ${s["mean"]:.4f} \\pm {s["std"]:.4f}$ & {s["n"]} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--experiments-dir", type=str,
                       default=str(PRIVATE_REPO / "experiments"),
                       help="Root of experiment directories")
    parser.add_argument("--output-dir", type=str,
                       default=str(PRIVATE_REPO / "results"),
                       help="Where to write aggregated outputs")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all runs
    histories = find_training_histories(experiments_dir)
    print(f"Found {len(histories)} training_history.json files")

    if not histories:
        print("No experiment results found. Run scripts/run_experiments.bat first.")
        return

    # Parse all runs
    records = [parse_run(h) for h in histories]

    # Write CSV
    csv_path = output_dir / "all_runs.csv"
    write_csv(records, csv_path)
    print(f"CSV: {csv_path}")

    # Loss study stats
    loss_records = [r for r in records if r['group'] == 'loss_study']
    if loss_records:
        f1_stats = group_stats(loss_records, 'loss_type', 'best_val_f1')
        iou_stats = group_stats(loss_records, 'loss_type', 'best_val_iou')

        latex = latex_table(f1_stats, 'Loss Function Ablation — Val F1', 'tab:loss-f1', 'F1')
        latex_path = output_dir / "loss_study_f1.tex"
        latex_path.write_text(latex)
        print(f"LaTeX: {latex_path}")

        print("\nLoss Study F1 (mean ± std):")
        for k, v in sorted(f1_stats.items()):
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")

    # Resolution study stats
    res_records = [r for r in records if r['group'] == 'resolution_study']
    if res_records:
        f1_stats = group_stats(res_records, 'resolution', 'best_val_f1')

        latex = latex_table(f1_stats, 'Resolution Study — Val F1', 'tab:res-f1', 'F1')
        latex_path = output_dir / "resolution_study_f1.tex"
        latex_path.write_text(latex)
        print(f"LaTeX: {latex_path}")

        print("\nResolution Study F1 (mean ± std):")
        for k, v in sorted(f1_stats.items()):
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")

    # Summary JSON
    summary = {
        'total_runs': len(records),
        'loss_study_runs': len(loss_records) if loss_records else 0,
        'resolution_study_runs': len(res_records) if res_records else 0,
    }
    summary_path = output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
