"""
Statistical Significance Tests for Test-Set Evaluation

Performs paired t-tests and Wilcoxon signed-rank tests on per-image metrics
between all loss function pairs. For each loss, per-image scores are averaged
across 3 seeds first, then paired tests compare the 12 per-image means.

Usage:
    python scripts/statistical_tests.py
    python scripts/statistical_tests.py --results-file ../SegmentationResearchPaper/results/test_set_evaluation.json
"""

import argparse
import json
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRIVATE_REPO = REPO_ROOT.parent / "SegmentationResearchPaper"

LOSS_STUDY_LOSSES = ["ce", "dice", "focal", "dice_focal", "tversky"]
METRICS_TO_TEST = ["f1", "iou", "boundary_f1"]


def load_results(results_path):
    """Load test_set_evaluation.json."""
    with open(results_path) as f:
        return json.load(f)


def extract_per_image_scores(data, metric="f1"):
    """Extract per-image metric scores grouped by loss type.

    For each loss type, averages across 3 seeds to get one score per image.

    Returns:
        dict[str, dict[str, float]]: {loss_type: {image_id: mean_score}}
        list[str]: sorted image ids
    """
    # Collect: loss_type -> seed -> image_id -> score
    raw = defaultdict(lambda: defaultdict(lambda: {}))

    for model_name, model_data in data["models"].items():
        if not model_name.startswith("loss_study/"):
            continue
        if "error" in model_data:
            continue

        config = model_data["config"]
        loss_type = config["loss_type"]
        seed = config["seed"]

        for img_result in model_data["results"]["per_image"]:
            image_id = img_result["image"]
            score = img_result["metrics"][metric]
            raw[loss_type][seed][image_id] = score

    # Average across seeds for each (loss, image)
    averaged = {}
    image_ids = sorted(data["test_images"])

    for loss_type in LOSS_STUDY_LOSSES:
        seeds = raw[loss_type]
        averaged[loss_type] = {}
        for img_id in image_ids:
            scores = [seeds[s][img_id] for s in seeds if img_id in seeds[s]]
            averaged[loss_type][img_id] = float(np.mean(scores))

    return averaged, image_ids


def run_paired_tests(scores_a, scores_b, image_ids):
    """Run paired t-test and Wilcoxon signed-rank test on two score arrays."""
    a = np.array([scores_a[img] for img in image_ids])
    b = np.array([scores_b[img] for img in image_ids])
    diff = a - b

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test (non-parametric)
    # Handle case where all differences are zero
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) == 0:
        w_stat, w_pval = 0.0, 1.0
    else:
        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = float("nan"), float("nan")

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "n": len(image_ids),
        "per_image_diff": {img: float(d) for img, d in zip(image_ids, diff)},
    }


def significance_symbol(p):
    """Return significance symbol for p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--results-file", type=str,
                        default=str(PRIVATE_REPO / "results" / "test_set_evaluation.json"))
    parser.add_argument("--output-file", type=str,
                        default=str(PRIVATE_REPO / "results" / "statistical_tests.json"))
    args = parser.parse_args()

    data = load_results(args.results_file)
    all_results = {}

    for metric in METRICS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"METRIC: {metric.upper()}")
        print(f"{'='*80}")

        averaged, image_ids = extract_per_image_scores(data, metric=metric)

        # Print per-image scores
        print(f"\nPer-image {metric} (averaged across 3 seeds):")
        print(f"{'Image':<12}", end="")
        for loss in LOSS_STUDY_LOSSES:
            print(f" {loss:>12}", end="")
        print()
        print("-" * (12 + 13 * len(LOSS_STUDY_LOSSES)))

        for img_id in image_ids:
            print(f"{img_id:<12}", end="")
            for loss in LOSS_STUDY_LOSSES:
                print(f" {averaged[loss][img_id]:>12.4f}", end="")
            print()

        # Print means
        print(f"{'MEAN':<12}", end="")
        for loss in LOSS_STUDY_LOSSES:
            vals = [averaged[loss][img] for img in image_ids]
            print(f" {np.mean(vals):>12.4f}", end="")
        print()

        # Pairwise tests
        print(f"\nPairwise comparisons ({metric}):")
        print(f"{'Comparison':<25} {'Δ Mean':>8} {'t-stat':>8} {'t p-val':>10} {'W p-val':>10} {'Sig':>5}")
        print("-" * 75)

        metric_results = {}
        pairs = list(itertools.combinations(LOSS_STUDY_LOSSES, 2))

        for loss_a, loss_b in pairs:
            result = run_paired_tests(averaged[loss_a], averaged[loss_b], image_ids)
            key = f"{loss_a}_vs_{loss_b}"
            metric_results[key] = result

            sig = significance_symbol(result["wilcoxon_pvalue"])
            print(f"{loss_a + ' vs ' + loss_b:<25} {result['mean_diff']:>+8.4f} "
                  f"{result['t_statistic']:>8.3f} {result['t_pvalue']:>10.6f} "
                  f"{result['wilcoxon_pvalue']:>10.6f} {sig:>5}")

        all_results[metric] = metric_results

    # Highlight key comparisons
    print(f"\n{'='*80}")
    print("KEY COMPARISONS (Wilcoxon p-values)")
    print(f"{'='*80}")

    key_pairs = [
        ("tversky", "dice_focal", "Tversky vs Dice+Focal (ranks #1 vs #2)"),
        ("dice_focal", "dice", "Dice+Focal vs Dice (ranks #2 vs #3)"),
        ("tversky", "dice", "Tversky vs Dice (ranks #1 vs #3)"),
        ("dice", "ce", "Dice vs CE (tier boundary)"),
        ("dice", "focal", "Dice vs Focal (tier boundary)"),
        ("tversky", "ce", "Tversky vs CE (top vs bottom)"),
    ]

    for metric in METRICS_TO_TEST:
        print(f"\n{metric.upper()}:")
        for loss_a, loss_b, desc in key_pairs:
            key = f"{loss_a}_vs_{loss_b}"
            if key not in all_results[metric]:
                # Try reverse order
                key = f"{loss_b}_vs_{loss_a}"
            r = all_results[metric][key]
            sig = significance_symbol(r["wilcoxon_pvalue"])
            mean_a = r["mean_a"]
            mean_b = r["mean_b"]
            # Determine which direction the key was stored
            if key.startswith(loss_a):
                diff = r["mean_diff"]
            else:
                diff = -r["mean_diff"]
                mean_a, mean_b = mean_b, mean_a
            print(f"  {desc:<45} Δ={diff:>+.4f}  W p={r['wilcoxon_pvalue']:.6f}  {sig}")

    # Bonferroni correction note
    n_comparisons = len(list(itertools.combinations(LOSS_STUDY_LOSSES, 2)))
    print(f"\nNote: {n_comparisons} pairwise comparisons per metric.")
    print(f"Bonferroni-corrected α = 0.05/{n_comparisons} = {0.05/n_comparisons:.4f}")
    print(f"Apply correction when interpreting marginal results.")

    # Save full results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
