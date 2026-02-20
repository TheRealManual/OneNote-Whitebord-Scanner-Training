"""Extract all numerical data needed for the paper."""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

PRIV = Path(__file__).resolve().parent.parent.parent / "SegmentationResearchPaper"

# Test set evaluation
d = json.load(open(PRIV / "results" / "test_set_evaluation.json"))
models = d["models"]

loss_data = defaultdict(lambda: defaultdict(list))
res_data = defaultdict(lambda: defaultdict(list))

metrics_list = ["f1", "iou", "boundary_f1", "boundary_iou"]

for name, r in models.items():
    if "error" in r:
        continue
    cfg = r["config"]
    g = r["results"]["grouped"]

    if name.startswith("loss_study/"):
        loss = cfg["loss_type"]
        for m in metrics_list:
            loss_data[loss][m].append(g["overall"][m]["mean"])
            loss_data[loss]["core_" + m].append(g["core"][m]["mean"])
            loss_data[loss]["thin_" + m].append(g["thin"][m]["mean"])

    if name.startswith("resolution_study/"):
        res = cfg["img_resolution"]
        for m in metrics_list:
            res_data[res][m].append(g["overall"][m]["mean"])
            res_data[res]["core_" + m].append(g["core"][m]["mean"])
            res_data[res]["thin_" + m].append(g["thin"][m]["mean"])

print("=== LOSS STUDY (mean +/- std over 3 seeds) ===")
for loss in ["ce", "focal", "dice", "dice_focal", "tversky"]:
    dd = loss_data[loss]
    print(f"\n{loss}:")
    for m in metrics_list:
        vals = dd[m]
        print(f"  {m:>15}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    for m in metrics_list:
        vals = dd["core_" + m]
        print(f"  core_{m:>9}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    for m in metrics_list:
        vals = dd["thin_" + m]
        print(f"  thin_{m:>9}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

print("\n=== RESOLUTION STUDY ===")
for res in sorted(res_data.keys()):
    dd = res_data[res]
    print(f"\n{res}:")
    for m in metrics_list:
        vals = dd[m]
        print(f"  {m:>15}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# Classical baseline
bl = json.load(open(PRIV / "results" / "classical_baseline_original_resolution.json"))
print("\n=== CLASSICAL BASELINE (original resolution) ===")
for method in ["adaptive", "otsu"]:
    r = bl[method]["results"]
    print(f"\n{method}:")
    for k in ["f1", "iou", "boundary_f1", "boundary_iou"]:
        if k in r:
            print(f"  {k}: {r[k]['mean']:.4f} +/- {r[k]['std']:.4f}")

# Thin stroke characterization
ts = json.load(open(PRIV / "results" / "thin_stroke_characterization.json"))
print("\n=== THIN STROKE CHARACTERIZATION ===")
for group in ["thin", "core", "train", "all"]:
    s = ts["summary"][group]
    print(f"{group}: fraction={s['stroke_fraction_mean']:.3f}% +/- {s['stroke_fraction_std']:.3f}%, width={s['mean_stroke_width_mean']:.1f}px +/- {s['mean_stroke_width_std']:.1f}px, n={s['count']}")

# Statistical tests
st = json.load(open(PRIV / "results" / "statistical_tests.json"))
print("\n=== KEY STATISTICAL TESTS (Wilcoxon p) ===")
for metric in st:
    print(f"\n{metric}:")
    for comp in st[metric]["comparisons"]:
        print(f"  {comp['pair'][0]} vs {comp['pair'][1]}: delta={comp['delta_mean']:.4f}, W_p={comp['wilcoxon_p']:.6f}, sig={comp['significance']}")
