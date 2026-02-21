"""Generate LaTeX snippet tables from experiment results.

Reads test_set_evaluation.json and classical_baseline_original_resolution.json
to produce camera-ready LaTeX tables that are \input{} into the paper.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

PRIV = Path(__file__).resolve().parent.parent.parent / "SegmentationResearchPaper"
SNIPPETS = PRIV / "paper" / "snippets"
SNIPPETS.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────
eval_data = json.load(open(PRIV / "results" / "test_set_evaluation.json"))
models = eval_data["models"]
baseline = json.load(open(PRIV / "results" / "classical_baseline_original_resolution.json"))
stats = json.load(open(PRIV / "results" / "statistical_tests.json"))

metrics_list = ["f1", "iou", "boundary_f1", "boundary_iou"]
groups = ["overall", "core", "thin"]

# Aggregate by loss type and resolution
loss_agg = defaultdict(lambda: defaultdict(list))
res_agg = defaultdict(lambda: defaultdict(list))

for name, r in models.items():
    if "error" in r:
        continue
    cfg = r["config"]
    g = r["results"]["grouped"]

    if name.startswith("loss_study/"):
        loss = cfg["loss_type"]
        for grp in groups:
            for m in metrics_list:
                key = f"{grp}_{m}"
                loss_agg[loss][key].append(g[grp][m]["mean"])

    if name.startswith("resolution_study/"):
        res = cfg["img_resolution"]
        for grp in groups:
            for m in metrics_list:
                key = f"{grp}_{m}"
                res_agg[res][key].append(g[grp][m]["mean"])

LOSS_DISPLAY = {
    "ce": "CE",
    "focal": "Focal",
    "dice": "Dice",
    "dice_focal": "Dice+Focal",
    "tversky": "Tversky",
}
LOSS_ORDER = ["ce", "focal", "dice", "dice_focal", "tversky"]


def fmt(vals):
    """Format mean ± std."""
    m = np.mean(vals)
    s = np.std(vals)
    return f"${m:.3f} \\pm {s:.3f}$"


# ── 1. Loss study table (test-set, multi-metric) ──────────
with open(SNIPPETS / "loss_study_f1.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            "}\n")
    f.write("\\toprule\n")
    f.write("Loss & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{IoU}"
            " & \\multicolumn{2}{c}{BF1} & \\multicolumn{2}{c}{B-IoU} \\\\\n")
    f.write("\\midrule\n")
    for loss in LOSS_ORDER:
        dd = loss_agg[loss]
        row = [LOSS_DISPLAY[loss]]
        for m in metrics_list:
            vals = dd[f"overall_{m}"]
            row.append(f"{np.mean(vals):.3f}")
            row.append(f"{np.std(vals):.3f}")
        f.write(" & ".join(row) + " \\\\\n")
    # Baseline row
    ada = baseline["adaptive"]["results"]
    row = ["Adaptive (baseline)"]
    for m in metrics_list:
        if m in ada:
            row.append(f"{ada[m]['mean']:.3f}")
            row.append(f"{ada[m]['std']:.3f}")
        else:
            row.extend(["--", "--"])
    f.write("\\midrule\n")
    f.write(" & ".join(row) + " \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote loss_study_f1.tex")


# ── 2. Loss study: Core vs Thin breakdown ──────────────────
with open(SNIPPETS / "loss_study_core_thin.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3]"
            "}\n")
    f.write("\\toprule\n")
    f.write("Loss & \\multicolumn{2}{c}{Core F1} & \\multicolumn{2}{c}{Thin F1}"
            " & {Gap} \\\\\n")
    f.write("\\midrule\n")
    for loss in LOSS_ORDER:
        dd = loss_agg[loss]
        core_vals = dd["core_f1"]
        thin_vals = dd["thin_f1"]
        core_m = np.mean(core_vals)
        thin_m = np.mean(thin_vals)
        gap = core_m - thin_m
        f.write(f"{LOSS_DISPLAY[loss]} & {core_m:.3f} & {np.std(core_vals):.3f}"
                f" & {thin_m:.3f} & {np.std(thin_vals):.3f}"
                f" & {gap:.3f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote loss_study_core_thin.tex")


# ── 3. Resolution study table ──────────────────────────────
with open(SNIPPETS / "resolution_study_f1.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            "}\n")
    f.write("\\toprule\n")
    f.write("Resolution & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{IoU}"
            " & \\multicolumn{2}{c}{BF1} & \\multicolumn{2}{c}{B-IoU} \\\\\n")
    f.write("\\midrule\n")
    for res in ["1024x768", "1536x1152"]:
        dd = res_agg[res]
        label = res.replace("x", "$\\times$")
        row = [label]
        for m in metrics_list:  # f1, iou, boundary_f1, boundary_iou
            vals = dd[f"overall_{m}"]
            row.append(f"{np.mean(vals):.3f}")
            row.append(f"{np.std(vals):.3f}")
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote resolution_study_f1.tex")


# ── 4. Classical baseline table ─────────────────────────────
with open(SNIPPETS / "classical_baseline.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            " S[table-format=1.3] @{${\\pm}$} S[table-format=1.3]"
            "}\n")
    f.write("\\toprule\n")
    f.write("Method & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{IoU}"
            " & \\multicolumn{2}{c}{BF1} & \\multicolumn{2}{c}{B-IoU} \\\\\n")
    f.write("\\midrule\n")
    for method in ["adaptive", "otsu"]:
        r = baseline[method]["results"]
        label = method.capitalize()
        row = [label]
        for m in metrics_list:  # f1, iou, boundary_f1, boundary_iou
            if m in r:
                row.append(f"{r[m]['mean']:.3f}")
                row.append(f"{r[m]['std']:.3f}")
            else:
                row.extend(["--", "--"])
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote classical_baseline.tex")


# ── 5. Statistical significance table ──────────────────────
with open(SNIPPETS / "statistical_significance.tex", "w") as f:
    f.write("\\begin{tabular}{l r r r l}\n")
    f.write("\\toprule\n")
    f.write("Comparison & $\\Delta$ F1 & $p$ (Wilcoxon) & $p$ (paired $t$) & Sig. \\\\\n")
    f.write("\\midrule\n")

    key_pairs = [
        ("ce", "dice", "CE vs.~Dice"),
        ("ce", "tversky", "CE vs.~Tversky"),
        ("focal", "dice_focal", "Focal vs.~Dice+Focal"),
        ("dice", "dice_focal", "Dice vs.~Dice+Focal"),
        ("dice", "tversky", "Dice vs.~Tversky"),
        ("dice_focal", "tversky", "Dice+Focal vs.~Tversky"),
    ]
    f1_tests = stats["f1"]
    for a, b, label in key_pairs:
        key = f"{a}_vs_{b}"
        if key not in f1_tests:
            key = f"{b}_vs_{a}"
        t = f1_tests[key]
        delta = t["mean_diff"]
        wp = t["wilcoxon_pvalue"]
        tp = t["t_pvalue"]
        # Significance markers
        if wp < 0.001:
            sig = "***"
        elif wp < 0.005:
            sig = "**"
        elif wp < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        f.write(f"{label} & {delta:+.4f} & {wp:.4f} & {tp:.6f} & {sig} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote statistical_significance.tex")


# ── 6. Robustness table (median, IQR, min, paired wins) ───
# Aggregate per-image, seed-averaged F1 for each loss
loss_per_image_f1 = defaultdict(lambda: defaultdict(list))
for name, r in models.items():
    if "error" in r or not name.startswith("loss_study/"):
        continue
    loss = r["config"]["loss_type"]
    for item in r["results"]["per_image"]:
        img = item["image"]
        loss_per_image_f1[loss][img].append(item["metrics"]["f1"])

# Baseline per-image (normalise keys: strip .png to match deep model keys)
baseline_pi = baseline["adaptive"]["per_image"]
bl_f1 = {item["image"].replace(".png", ""): item["metrics"]["f1"] for item in baseline_pi}

with open(SNIPPETS / "robustness.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=1.3]"
            " S[table-format=1.3]"
            " S[table-format=1.3]"
            " S[table-format=1.3]"
            " S[table-format=1.3]"
            " r"
            "}\n")
    f.write("\\toprule\n")
    f.write("Method & {Mean} & {Median} & {IQR} & {Min} & {Max} & {Wins/12} \\\\\n")
    f.write("\\midrule\n")
    # Deep models
    for loss in LOSS_ORDER:
        dd = loss_per_image_f1[loss]
        img_f1s = np.array([np.mean(dd[img]) for img in sorted(dd.keys())])
        q25, q75 = np.percentile(img_f1s, 25), np.percentile(img_f1s, 75)
        # Paired wins vs CE (for Dice-family) or vs Dice (for CE/Focal)
        # Use wins vs adaptive baseline
        wins = sum(1 for img in sorted(dd.keys())
                   if np.mean(dd[img]) > bl_f1.get(img, 999))
        f.write(f"{LOSS_DISPLAY[loss]} & {img_f1s.mean():.3f}"
                f" & {np.median(img_f1s):.3f}"
                f" & {q75-q25:.3f}"
                f" & {img_f1s.min():.3f}"
                f" & {img_f1s.max():.3f}"
                f" & {wins}/12 \\\\\n")
    # Baseline — count images where adaptive beats the best deep model (Tversky)
    bl_arr = np.array(list(bl_f1.values()))
    q25, q75 = np.percentile(bl_arr, 25), np.percentile(bl_arr, 75)
    # Find best deep model per image (across all losses)
    best_deep = {}
    for loss in LOSS_ORDER:
        dd = loss_per_image_f1[loss]
        for img in dd:
            avg = np.mean(dd[img])
            best_deep[img] = max(best_deep.get(img, -1), avg)
    bl_wins = sum(1 for img in bl_f1 if bl_f1[img] > best_deep.get(img, 999))
    f.write("\\midrule\n")
    f.write(f"Adaptive & {bl_arr.mean():.3f}"
            f" & {np.median(bl_arr):.3f}"
            f" & {q75-q25:.3f}"
            f" & {bl_arr.min():.3f}"
            f" & {bl_arr.max():.3f}"
            f" & {bl_wins}/12 \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote robustness.tex")


# ── 7. Enhanced stats table with effect sizes ──────────────
with open(SNIPPETS / "statistical_significance.tex", "w") as f:
    f.write("\\begin{tabular}{l"
            " S[table-format=+1.4]"
            " S[table-format=+1.4]"
            " S[table-format=1.4]"
            " l"
            "}\n")
    f.write("\\toprule\n")
    f.write("Comparison & {$\\Delta$F1 (mean)} & {$\\Delta$F1 (median)}"
            " & {$p$ (Wilcoxon)} & {Sig.} \\\\\n")
    f.write("\\midrule\n")

    key_pairs = [
        ("ce", "focal", "CE vs.~Focal"),
        ("ce", "dice", "CE vs.~Dice"),
        ("ce", "dice_focal", "CE vs.~Dice+Focal"),
        ("ce", "tversky", "CE vs.~Tversky"),
        ("focal", "dice", "Focal vs.~Dice"),
        ("focal", "dice_focal", "Focal vs.~Dice+Focal"),
        ("focal", "tversky", "Focal vs.~Tversky"),
        ("dice", "dice_focal", "Dice vs.~Dice+Focal"),
        ("dice", "tversky", "Dice vs.~Tversky"),
        ("dice_focal", "tversky", "Dice+Focal vs.~Tversky"),
    ]
    f1_tests = stats["f1"]
    for a, b, label in key_pairs:
        key = f"{a}_vs_{b}"
        sign = 1.0
        if key not in f1_tests:
            key = f"{b}_vs_{a}"
            sign = -1.0  # flip sign when key order is reversed
        t = f1_tests[key]
        mean_delta = sign * t["mean_diff"]
        wp = t["wilcoxon_pvalue"]
        # Compute median paired diff from per-image data (same sign as mean_diff: a - b)
        diffs = []
        for img in sorted(loss_per_image_f1[a].keys()):
            va = np.mean(loss_per_image_f1[a][img])
            vb = np.mean(loss_per_image_f1[b][img])
            diffs.append(va - vb)  # same convention as mean_diff
        median_delta = np.median(diffs)
        if wp < 0.001:
            sig = "***"
        elif wp < 0.005:
            sig = "**"
        elif wp < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        f.write(f"{label} & {mean_delta:+.4f} & {median_delta:+.4f}"
                f" & {wp:.4f} & {sig} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("Wrote statistical_significance.tex (enhanced with effect sizes)")
print("\nAll snippets generated.")
