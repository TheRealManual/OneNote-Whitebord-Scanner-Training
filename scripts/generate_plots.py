"""
Generate publication-quality plots from experiment results.

Reads test_set_evaluation.json, training_history.json files, classical baseline
results, and statistical tests to produce all paper figures.

Produces:
  1. Loss study bar chart — F1/IoU/BF1 by loss function with error bars
  2. Core vs Thin grouped bar chart — thin-stroke subset analysis
  3. Core-Thin gap chart — which loss narrows the gap most
  4. Resolution comparison — F1/IoU/BF1 at two resolutions
  5. Training curves — loss and val F1 over epochs (one line per loss type)
  6. Multi-metric heatmap — losses × metrics
  7. Per-image F1 scatter — all losses across all 12 test images

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --output-dir ../SegmentationResearchPaper/results/figures
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"

# Consistent styling
LOSS_ORDER = ["CE", "Focal", "Dice", "Dice+Focal", "Tversky"]
LOSS_COLORS = {
    "CE": "#d62728",        # red
    "Focal": "#ff7f0e",     # orange
    "Dice": "#2ca02c",      # green
    "Dice+Focal": "#1f77b4", # blue
    "Tversky": "#9467bd",   # purple
}
LOSS_KEY_MAP = {
    "ce": "CE",
    "focal": "Focal",
    "dice": "Dice",
    "dice_focal": "Dice+Focal",
    "tversky": "Tversky",
}

# Paper-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'sans-serif',
})


def load_test_eval(results_dir):
    """Load test_set_evaluation.json."""
    path = Path(results_dir) / "test_set_evaluation.json"
    with open(path) as f:
        return json.load(f)


def load_baseline(results_dir):
    """Load classical_baseline_test_split.json."""
    path = Path(results_dir) / "classical_baseline_test_split.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def extract_loss_study_metrics(data):
    """Extract per-loss mean±std metrics from test evaluation data.

    Returns dict: {loss_label: {metric_key: {mean, std, seeds: [vals]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: {"seeds": []}))

    for model_name, model_data in data["models"].items():
        if not model_name.startswith("loss_study/"):
            continue
        if "error" in model_data:
            continue

        config = model_data["config"]
        loss_raw = config["loss_type"]
        loss_label = LOSS_KEY_MAP.get(loss_raw, loss_raw)

        grouped = model_data["results"]["grouped"]
        for group_name in ["overall", "core", "thin"]:
            g = grouped.get(group_name, {})
            for metric in ["f1", "iou", "boundary_f1", "boundary_iou"]:
                if metric in g:
                    key = f"{group_name}_{metric}"
                    results[loss_label][key]["seeds"].append(g[metric]["mean"])

    # Compute mean ± std across seeds
    final = {}
    for loss_label, metrics in results.items():
        final[loss_label] = {}
        for key, val in metrics.items():
            seeds = val["seeds"]
            final[loss_label][key] = {
                "mean": float(np.mean(seeds)),
                "std": float(np.std(seeds)),
                "seeds": seeds,
            }
    return final


def extract_resolution_metrics(data):
    """Extract resolution study metrics."""
    results = defaultdict(lambda: defaultdict(lambda: {"seeds": []}))

    for model_name, model_data in data["models"].items():
        if not model_name.startswith("resolution_study/"):
            continue
        if "error" in model_data:
            continue

        config = model_data["config"]
        res = config.get("img_resolution", "unknown")
        # Normalize resolution label
        if "768" in res or "1024" in res:
            res_label = "768×1024"
        else:
            res_label = "1536×1152"

        grouped = model_data["results"]["grouped"]
        for group_name in ["overall", "core", "thin"]:
            g = grouped.get(group_name, {})
            for metric in ["f1", "iou", "boundary_f1", "boundary_iou"]:
                if metric in g:
                    key = f"{group_name}_{metric}"
                    results[res_label][key]["seeds"].append(g[metric]["mean"])

    final = {}
    for res_label, metrics in results.items():
        final[res_label] = {}
        for key, val in metrics.items():
            seeds = val["seeds"]
            final[res_label][key] = {
                "mean": float(np.mean(seeds)),
                "std": float(np.std(seeds)),
            }
    return final


# ============================================================================
# PLOT 1: Loss Study Bar Chart (F1, IoU, BF1)
# ============================================================================
def plot_loss_study_bars(loss_metrics, output_dir):
    """Grouped bar chart: F1, IoU, BF1 by loss function."""
    metrics = ["overall_f1", "overall_iou", "overall_boundary_f1"]
    metric_labels = ["F1", "IoU", "Boundary F1"]
    metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Add Boundary IoU if available
    sample_loss = next(iter(loss_metrics.values()), {})
    if "overall_boundary_iou" in sample_loss:
        metrics.append("overall_boundary_iou")
        metric_labels.append("Boundary IoU")
        metric_colors.append("#9467bd")

    losses = [l for l in LOSS_ORDER if l in loss_metrics]

    x = np.arange(len(losses))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    offsets = [(i - (n_metrics - 1) / 2) * width for i in range(n_metrics)]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        means = [loss_metrics[l].get(metric, {}).get("mean", 0) for l in losses]
        stds = [loss_metrics[l].get(metric, {}).get("std", 0) for l in losses]

        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=label, capsize=4, alpha=0.85,
               color=color, edgecolor=matplotlib.colors.to_rgba(color, 1.0),
               linewidth=1)

    ax.set_xlabel("Loss Function")
    ax.set_ylabel("Score")
    ax.set_title("Test-Set Performance by Loss Function")
    ax.set_xticks(x)
    ax.set_xticklabels(losses)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_loss_study_bars.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 2: Core vs Thin Subset Comparison
# ============================================================================
def plot_core_vs_thin(loss_metrics, output_dir):
    """Grouped bar chart showing Core F1 vs Thin F1 for each loss."""
    losses = [l for l in LOSS_ORDER if l in loss_metrics]
    x = np.arange(len(losses))
    width = 0.35

    core_means = [loss_metrics[l]["core_f1"]["mean"] for l in losses]
    core_stds = [loss_metrics[l]["core_f1"]["std"] for l in losses]
    thin_means = [loss_metrics[l]["thin_f1"]["mean"] for l in losses]
    thin_stds = [loss_metrics[l]["thin_f1"]["std"] for l in losses]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width/2, core_means, width, yerr=core_stds,
           label="Core (7 images)", capsize=4, color="#4C72B0", alpha=0.85,
           edgecolor="#2c5282", linewidth=1)
    ax.bar(x + width/2, thin_means, width, yerr=thin_stds,
           label="Thin-stroke (5 images)", capsize=4, color="#DD8452", alpha=0.85,
           edgecolor="#a0522d", linewidth=1)

    # Add gap annotations
    for i, l in enumerate(losses):
        gap = core_means[i] - thin_means[i]
        mid_y = (core_means[i] + thin_means[i]) / 2
        ax.annotate(f"Δ={gap:.3f}", xy=(i + width/2 + 0.05, mid_y),
                    fontsize=8, color="#555", ha="left", va="center")

    ax.set_xlabel("Loss Function")
    ax.set_ylabel("F1 Score")
    ax.set_title("Core vs Thin-Stroke F1 by Loss Function")
    ax.set_xticks(x)
    ax.set_xticklabels(losses)
    ax.legend()
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_core_vs_thin.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 3: Core-Thin Gap Chart
# ============================================================================
def plot_gap_chart(loss_metrics, output_dir):
    """Horizontal bar chart of Core–Thin F1 gap per loss."""
    losses = [l for l in LOSS_ORDER if l in loss_metrics]
    gaps = [loss_metrics[l]["core_f1"]["mean"] - loss_metrics[l]["thin_f1"]["mean"]
            for l in losses]

    # Sort by gap
    sorted_pairs = sorted(zip(losses, gaps), key=lambda x: x[1])
    sorted_losses, sorted_gaps = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [LOSS_COLORS[l] for l in sorted_losses]
    bars = ax.barh(sorted_losses, sorted_gaps, color=colors, alpha=0.85,
                   edgecolor=[matplotlib.colors.to_rgba(c, 1.0) for c in colors],
                   linewidth=1.2)

    for bar, gap in zip(bars, sorted_gaps):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{gap:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Core–Thin F1 Gap (lower = more equitable)")
    ax.set_title("Thin-Stroke Performance Gap by Loss Function")
    ax.set_xlim(0, max(sorted_gaps) * 1.3)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_core_thin_gap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 4: Resolution Comparison
# ============================================================================
def plot_resolution_comparison(res_metrics, output_dir):
    """Bar chart comparing two resolutions on F1, IoU, BF1 (overall + core + thin)."""
    resolutions = sorted(res_metrics.keys())
    metrics = [("overall_f1", "Overall F1"), ("core_f1", "Core F1"),
               ("thin_f1", "Thin F1"), ("overall_boundary_f1", "BF1")]

    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, res in enumerate(resolutions):
        means = [res_metrics[res][m[0]]["mean"] for m in metrics]
        stds = [res_metrics[res][m[0]]["std"] for m in metrics]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
               label=res, capsize=4, color=colors[i], alpha=0.85,
               edgecolor=matplotlib.colors.to_rgba(colors[i], 1.0), linewidth=1)

    ax.set_ylabel("Score")
    ax.set_title("Resolution Effect on Segmentation Quality (Dice+Focal Loss)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend()
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_resolution_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 5: Training Curves (one line per loss type, best seed)
# ============================================================================
def plot_training_curves(experiments_dir, output_dir):
    """Plot val F1 and training loss curves, one line per loss type (best seed by val F1)."""
    loss_study_dir = Path(experiments_dir) / "loss_study"
    if not loss_study_dir.exists():
        print("No loss_study directory — skipping training curves")
        return

    # Find best seed per loss type
    best_runs = {}
    for hist_path in sorted(loss_study_dir.rglob("training_history.json")):
        with open(hist_path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        loss_raw = cfg.get("loss_type", "")
        loss_label = LOSS_KEY_MAP.get(loss_raw, loss_raw)
        best_f1 = data.get("results", {}).get("best_val_f1", 0)

        if loss_label not in best_runs or best_f1 > best_runs[loss_label]["best_f1"]:
            best_runs[loss_label] = {
                "data": data,
                "best_f1": best_f1,
                "seed": cfg.get("seed"),
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for loss_label in LOSS_ORDER:
        if loss_label not in best_runs:
            continue
        data = best_runs[loss_label]["data"]
        seed = best_runs[loss_label]["seed"]
        color = LOSS_COLORS[loss_label]

        train_loss = data.get("train_loss", [])
        val_loss = data.get("val_loss", [])
        val_f1 = data.get("val_f1", [])
        epochs = list(range(1, len(train_loss) + 1))

        if train_loss:
            ax1.plot(epochs, train_loss, color=color, alpha=0.4, linewidth=0.8)
        if val_loss:
            ax1.plot(epochs, val_loss, color=color, linewidth=1.5,
                     label=f"{loss_label} (seed {seed})")
        if val_f1:
            ax2.plot(list(range(1, len(val_f1) + 1)), val_f1,
                     color=color, linewidth=1.5,
                     label=f"{loss_label} (seed {seed})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Over Training")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 0.85)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_training_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 6: Multi-metric Heatmap
# ============================================================================
def plot_metric_heatmap(loss_metrics, output_dir):
    """Heatmap: losses × metrics, color = score."""
    losses = [l for l in LOSS_ORDER if l in loss_metrics]
    metrics = [
        ("overall_f1", "F1"),
        ("overall_iou", "IoU"),
        ("overall_boundary_f1", "BF1"),
        ("overall_boundary_iou", "B-IoU"),
        ("core_f1", "Core F1"),
        ("thin_f1", "Thin F1"),
    ]
    # Only include metrics that exist in the data
    metrics = [(k, l) for k, l in metrics
               if any(k in loss_metrics.get(loss, {}) for loss in losses)]
    metric_keys = [m[0] for m in metrics]
    metric_labels = [m[1] for m in metrics]

    data_matrix = np.zeros((len(losses), len(metrics)))
    for i, l in enumerate(losses):
        for j, mk in enumerate(metric_keys):
            data_matrix[i, j] = loss_metrics[l].get(mk, {}).get("mean", 0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=0.25, vmax=0.75)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(losses)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticklabels(losses)

    # Annotate cells
    for i in range(len(losses)):
        for j in range(len(metric_labels)):
            val = data_matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    ax.set_title("Test-Set Metrics by Loss Function")
    fig.colorbar(im, ax=ax, label="Score", shrink=0.8)

    plt.tight_layout()
    out = output_dir / "fig_metric_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================================
# PLOT 7: Per-Image F1 Scatter (Core vs Thin highlighted)
# ============================================================================
def plot_per_image_f1(data, output_dir):
    """Per-image F1 for each loss (averaged across seeds), highlighting core vs thin."""
    loss_image_scores = defaultdict(lambda: defaultdict(list))

    for model_name, model_data in data["models"].items():
        if not model_name.startswith("loss_study/"):
            continue
        if "error" in model_data:
            continue
        config = model_data["config"]
        loss_label = LOSS_KEY_MAP.get(config["loss_type"], config["loss_type"])

        for img_result in model_data["results"]["per_image"]:
            img_id = img_result["image"]
            loss_image_scores[loss_label][img_id].append(img_result["metrics"]["f1"])

    # Average across seeds
    losses = [l for l in LOSS_ORDER if l in loss_image_scores]
    image_ids = sorted(data["test_images"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(image_ids))
    width = 0.15
    n_losses = len(losses)

    for i, loss_label in enumerate(losses):
        means = [np.mean(loss_image_scores[loss_label][img]) for img in image_ids]
        offset = (i - n_losses / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=loss_label,
               color=LOSS_COLORS[loss_label], alpha=0.85, edgecolor="white", linewidth=0.5)

    # Annotate thin images
    thin_ids = {"image_22", "image_24", "image_27", "image_33", "image_36"}
    for j, img_id in enumerate(image_ids):
        if img_id in thin_ids:
            ax.axvspan(j - 0.45, j + 0.45, alpha=0.08, color="orange")
            ax.text(j, -0.03, "thin", ha="center", fontsize=7, color="orange",
                    fontstyle="italic")

    ax.set_xlabel("Test Image")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Image F1 by Loss Function (orange bands = thin-stroke images)")
    ax.set_xticks(x)
    ax.set_xticklabels([img.replace("image_", "") for img in image_ids])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "fig_per_image_f1.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots")
    parser.add_argument("--results-dir", type=str,
                        default=str(PRIVATE_REPO / "results"),
                        help="Directory with test_set_evaluation.json etc.")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(PRIVATE_REPO / "experiments"),
                        help="Root experiments directory (for training curves)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PRIVATE_REPO / "results" / "figures"),
                        help="Where to save figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_test_eval(args.results_dir)
    loss_metrics = extract_loss_study_metrics(data)
    res_metrics = extract_resolution_metrics(data)

    print(f"Loaded {data['num_models']} models, {data['num_test_images']} test images")
    print(f"Loss types: {list(loss_metrics.keys())}")
    print(f"Resolutions: {list(res_metrics.keys())}")
    print()

    # Generate all plots
    plot_loss_study_bars(loss_metrics, output_dir)
    plot_core_vs_thin(loss_metrics, output_dir)
    plot_gap_chart(loss_metrics, output_dir)
    plot_resolution_comparison(res_metrics, output_dir)
    plot_training_curves(args.experiments_dir, output_dir)
    plot_metric_heatmap(loss_metrics, output_dir)
    plot_per_image_f1(data, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
