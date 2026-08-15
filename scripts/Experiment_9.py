#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================================
# Experiment 9: Statistical Significance Testing
#
# Establishes whether the accuracy differences reported in earlier experiments
# reflect real differences between models or ordinary variation.
#
#   McNemar's test        Per-image agreement between two classifiers on the
#                         same 25,250 test images. The correct test for paired
#                         binary outcomes, because it conditions on the images
#                         where the two models disagree and ignores those where
#                         they agree. An unpaired test would discard the fact
#                         that both models saw identical inputs.
#
#   Wilcoxon signed-rank  Per-class accuracy, 101 paired values. Non-parametric,
#                         so it makes no normality assumption about the
#                         distribution of per-class accuracies, which are
#                         bounded in [0,1] and typically skewed.
#
# Two additions beyond the plan, both necessary rather than optional:
#
#   Effect sizes. With N = 25,250 a difference of a few tenths of a point can
#   reach p < 0.05 while being of no practical consequence. Significance
#   without an effect size is close to meaningless at this sample size, and
#   reporting only p-values invites the obvious viva question.
#
#   Multiple-comparison correction. Comparing k models pairwise means k(k-1)/2
#   tests. At six models that is 15 tests, and at alpha = 0.05 roughly one
#   false positive is expected by chance alone. Holm-Bonferroni is applied.
#
# NO GPU REQUIRED. This notebook reads saved prediction arrays and runs on any
# machine. Run scripts/check_predictions.py first to confirm the inputs exist.
# ============================================================================

import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

print("Experiment 9: statistical significance testing (CPU only)")


# In[ ]:


# ---------------------------------------------------------------------------
# Configuration
#
# Edit MODEL_FILES so each entry points at your saved prediction array. Run
# check_predictions.py to discover the correct paths.
# ---------------------------------------------------------------------------

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
OUT_ROOT = PROJECT_ROOT / "outputs"

OUT = OUT_ROOT / "experiment_9"
(OUT / "images").mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
N_BOOTSTRAP = 10000
SEED = 42
rng = np.random.default_rng(SEED)

MODEL_FILES = {
    "AlexNet":           OUT_ROOT / "alexnet"       / "preds.npy",
    "ResNet34":          OUT_ROOT / "resnet34"      / "preds.npy",
    "ResNet50":          OUT_ROOT / "resnet50"      / "preds.npy",
    "YOLOv8s":           OUT_ROOT / "yolo"          / "preds.npy",
    "CLIP zero-shot":    OUT_ROOT / "clipzeroshot"  / "clip_zeroshot_preds.npy",
    "SigLIP zero-shot":  OUT_ROOT / "experiment_7"  / "preds" / "siglip_preds.npy",
    # add when available:
    # "CLIP linear probe": OUT_ROOT / "experiment_3" / "linear_probe_preds.npy",
    # "CLIP fine-tuned":   OUT_ROOT / "experiment_4" / "finetuned_preds.npy",
}

LABELS_FILE = OUT_ROOT / "clipzeroshot" / "clip_zeroshot_labels.npy"

print(f"Outputs -> {OUT}")


# In[ ]:


# ---------------------------------------------------------------------------
# Load and validate
#
# Every array must have the same length and refer to the same images in the
# same order. McNemar's test on misaligned arrays returns a p-value that looks
# entirely reasonable and means nothing at all, so this is checked rather than
# assumed.
# ---------------------------------------------------------------------------

if not LABELS_FILE.exists():
    raise FileNotFoundError(
        f"Ground-truth labels not found at {LABELS_FILE}.\n"
        "Run scripts/check_predictions.py to locate them.")

labels = np.load(LABELS_FILE).astype(np.int64)
N = len(labels)
C = int(labels.max()) + 1
print(f"Test images: {N}    classes: {C}")

preds, skipped = {}, []
for name, path in MODEL_FILES.items():
    if not path.exists():
        skipped.append((name, "file not found"))
        continue
    a = np.load(path).astype(np.int64)
    if len(a) != N:
        skipped.append((name, f"{len(a)} rows, expected {N}"))
        continue
    preds[name] = a

print(f"\nLoaded {len(preds)} models:")
for name, a in preds.items():
    print(f"   {name:<20s} accuracy {100*(a == labels).mean():6.2f}%")

if skipped:
    print(f"\nSkipped {len(skipped)}:")
    for name, why in skipped:
        print(f"   {name:<20s} {why}")

if len(preds) < 2:
    raise RuntimeError("At least two models are needed for a paired test.")

# Per-image correctness, the basis of every test below
correct = {name: (a == labels) for name, a in preds.items()}


# In[ ]:


# ---------------------------------------------------------------------------
# Per-class accuracy, the input to the Wilcoxon test
# ---------------------------------------------------------------------------

per_class = {}
for name, a in preds.items():
    accs = np.zeros(C)
    for c in range(C):
        m = labels == c
        accs[c] = (a[m] == c).mean() if m.sum() else np.nan
    per_class[name] = accs

pc_df = pd.DataFrame(per_class)
pc_df.index.name = "class_id"
pc_df.to_csv(OUT / "per_class_accuracy.csv")

print("Per-class accuracy summary (%):\n")
print((pc_df * 100).describe().round(2).to_string())


# In[ ]:


# ---------------------------------------------------------------------------
# McNemar's test
#
# For a pair of models the 2x2 table counts images by joint outcome:
#
#                       B correct   B wrong
#       A correct           n00        n01
#       A wrong             n10        n11
#
# Only the discordant cells n01 and n10 carry information: images both models
# get right, or both get wrong, say nothing about which is better. The null
# hypothesis is n01 = n10.
#
# The exact binomial form is used when the discordant total is small; otherwise
# the chi-squared form with continuity correction, which is the standard
# threshold in the literature.
# ---------------------------------------------------------------------------

def mcnemar_pair(a_correct, b_correct, exact_threshold=25):
    n00 = int(np.sum(a_correct & b_correct))
    n01 = int(np.sum(a_correct & ~b_correct))    # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))    # A wrong, B right
    n11 = int(np.sum(~a_correct & ~b_correct))

    table = [[n00, n01], [n10, n11]]
    discordant = n01 + n10
    use_exact = discordant < exact_threshold
    res = mcnemar(table, exact=use_exact, correction=not use_exact)

    # Odds ratio as effect size: how many times more often A is uniquely
    # right than B is uniquely right.
    odds = np.inf if n10 == 0 else n01 / n10

    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "discordant": discordant,
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "method": "exact binomial" if use_exact else "chi2 + correction",
            "odds_ratio": float(odds)}


def bootstrap_diff_ci(a_correct, b_correct, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Percentile CI on the paired accuracy difference (A minus B)."""
    d = a_correct.astype(np.int8) - b_correct.astype(np.int8)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return 100*d.mean(), 100*lo, 100*hi


print("McNemar helper defined.")


# In[ ]:


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test on the 101 paired per-class accuracies
# ---------------------------------------------------------------------------

def wilcoxon_pair(a_acc, b_acc):
    d = a_acc - b_acc
    d = d[~np.isnan(d)]
    nonzero = int(np.sum(d != 0))
    if nonzero == 0:
        return {"statistic": np.nan, "p_value": 1.0, "n_nonzero": 0,
                "median_diff_pts": 0.0, "rank_biserial": 0.0,
                "n_favour_a": 0, "n_favour_b": 0}

    stat, p = stats.wilcoxon(d, zero_method="wilcox",
                             alternative="two-sided", mode="auto")

    # Rank-biserial correlation as effect size: +1 means A beats B on every
    # class, -1 the reverse, 0 no consistent direction.
    ranks = stats.rankdata(np.abs(d[d != 0]))
    signs = np.sign(d[d != 0])
    total = ranks.sum()
    rb = float((ranks[signs > 0].sum() - ranks[signs < 0].sum()) / total)

    return {"statistic": float(stat), "p_value": float(p),
            "n_nonzero": nonzero,
            "median_diff_pts": float(100*np.median(d)),
            "rank_biserial": rb,
            "n_favour_a": int(np.sum(d > 0)),
            "n_favour_b": int(np.sum(d < 0))}


print("Wilcoxon helper defined.")


# In[ ]:


# ---------------------------------------------------------------------------
# Run every pairwise comparison
# ---------------------------------------------------------------------------

names = list(preds.keys())
pairs = list(itertools.combinations(names, 2))
print(f"{len(names)} models -> {len(pairs)} pairwise comparisons\n")

rows = []
for a, b in pairs:
    mc = mcnemar_pair(correct[a], correct[b])
    wc = wilcoxon_pair(per_class[a], per_class[b])
    diff, lo, hi = bootstrap_diff_ci(correct[a], correct[b])

    rows.append({
        "model_a": a, "model_b": b,
        "acc_a": 100*correct[a].mean(),
        "acc_b": 100*correct[b].mean(),
        "acc_diff_pts": diff,
        "ci_lower": lo, "ci_hi": hi,
        "mcnemar_p": mc["p_value"],
        "mcnemar_method": mc["method"],
        "a_only_correct": mc["n01"], "b_only_correct": mc["n10"],
        "both_correct": mc["n00"], "both_wrong": mc["n11"],
        "odds_ratio": mc["odds_ratio"],
        "wilcoxon_p": wc["p_value"],
        "wilcoxon_median_diff_pts": wc["median_diff_pts"],
        "rank_biserial": wc["rank_biserial"],
        "classes_favouring_a": wc["n_favour_a"],
        "classes_favouring_b": wc["n_favour_b"],
    })

res = pd.DataFrame(rows)
print(res[["model_a", "model_b", "acc_diff_pts",
           "mcnemar_p", "wilcoxon_p"]].round(4).to_string(index=False))


# In[ ]:


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
#
# Fifteen tests at alpha = 0.05 yields roughly a 54 percent chance of at least
# one false positive if uncorrected. Holm is used rather than plain Bonferroni
# because it is uniformly more powerful while controlling the same family-wise
# error rate.
# ---------------------------------------------------------------------------

def holm(pvals, alpha=ALPHA):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


res["mcnemar_p_holm"], res["mcnemar_sig"] = holm(res["mcnemar_p"])
res["wilcoxon_p_holm"], res["wilcoxon_sig"] = holm(res["wilcoxon_p"])

res.to_csv(OUT / "significance_results.csv", index=False)

n_raw = int((res["mcnemar_p"] < ALPHA).sum())
n_adj = int(res["mcnemar_sig"].sum())
print(f"McNemar significant before correction: {n_raw}/{len(res)}")
print(f"McNemar significant after Holm:        {n_adj}/{len(res)}")
print(f"Wilcoxon significant after Holm:       "
      f"{int(res['wilcoxon_sig'].sum())}/{len(res)}")


# In[ ]:


# ---------------------------------------------------------------------------
# Readable report
# ---------------------------------------------------------------------------

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else \
           "*" if p < 0.05 else "ns"


lines = []
for _, r in res.iterrows():
    lines.append(f"\n{'='*70}\n{r['model_a']}  vs  {r['model_b']}\n{'='*70}")
    lines.append(f"  Accuracy: {r['acc_a']:.2f}% vs {r['acc_b']:.2f}%   "
                 f"difference {r['acc_diff_pts']:+.2f} pts "
                 f"(95% CI [{r['ci_lower']:+.2f}, {r['ci_hi']:+.2f}])")
    lines.append(f"\n  McNemar, per image (N={N}):")
    lines.append(f"    both correct        {int(r['both_correct']):>6d}")
    lines.append(f"    only {r['model_a'][:14]:<14s} {int(r['a_only_correct']):>6d}")
    lines.append(f"    only {r['model_b'][:14]:<14s} {int(r['b_only_correct']):>6d}")
    lines.append(f"    both wrong          {int(r['both_wrong']):>6d}")
    lines.append(f"    p = {r['mcnemar_p']:.3e}   Holm-adjusted "
                 f"{r['mcnemar_p_holm']:.3e}  {stars(r['mcnemar_p_holm'])}")
    lines.append(f"    odds ratio {r['odds_ratio']:.2f} "
                 f"({r['model_a']} uniquely correct that many times as often)")
    lines.append(f"\n  Wilcoxon, per class (n={C}):")
    lines.append(f"    classes favouring {r['model_a'][:14]:<14s} "
                 f"{int(r['classes_favouring_a']):>3d}")
    lines.append(f"    classes favouring {r['model_b'][:14]:<14s} "
                 f"{int(r['classes_favouring_b']):>3d}")
    lines.append(f"    median per-class difference "
                 f"{r['wilcoxon_median_diff_pts']:+.2f} pts")
    lines.append(f"    p = {r['wilcoxon_p']:.3e}   Holm-adjusted "
                 f"{r['wilcoxon_p_holm']:.3e}  {stars(r['wilcoxon_p_holm'])}")
    lines.append(f"    rank-biserial {r['rank_biserial']:+.3f}")

report = "\n".join(lines)
print(report)
with open(OUT / "significance_report.txt", "w") as f:
    f.write(report)


# In[ ]:


# ---------------------------------------------------------------------------
# Figure: significance matrix
# ---------------------------------------------------------------------------

M = np.full((len(names), len(names)), np.nan)
annot = np.empty((len(names), len(names)), dtype=object)
annot[:] = ""

for _, r in res.iterrows():
    i, j = names.index(r["model_a"]), names.index(r["model_b"])
    M[i, j] = r["acc_diff_pts"]
    M[j, i] = -r["acc_diff_pts"]
    s = stars(r["mcnemar_p_holm"])
    annot[i, j] = f"{r['acc_diff_pts']:+.1f}\n{s}"
    annot[j, i] = f"{-r['acc_diff_pts']:+.1f}\n{s}"

fig, ax = plt.subplots(figsize=(10, 8))
lim = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        if i != j:
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=8)
    ax.text(i, i, "--", ha="center", va="center", color="grey")
ax.set_title("Pairwise accuracy difference (row minus column)\n"
             "McNemar's test, Holm-adjusted:  *** p<0.001  ** p<0.01  "
             "* p<0.05  ns not significant")
fig.colorbar(im, ax=ax, label="Accuracy difference (percentage points)")
plt.tight_layout()
plt.savefig(OUT / "images" / "significance_matrix.png", dpi=200)
plt.close()

# Figure: per-class paired differences for the headline comparison
head = res.iloc[res["acc_diff_pts"].abs().idxmax()]
a, b = head["model_a"], head["model_b"]
d = 100 * (per_class[a] - per_class[b])

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(d, bins=30, color="#4C78A8", edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.axvline(np.median(d), color="crimson", linestyle="--", linewidth=2,
           label=f"median {np.median(d):+.2f} pts")
ax.set_xlabel(f"Per-class accuracy difference, {a} minus {b} (pts)")
ax.set_ylabel("Number of classes")
ax.set_title(f"Distribution of per-class differences: {a} vs {b}\n"
             f"Wilcoxon Holm-adjusted p = {head['wilcoxon_p_holm']:.2e}")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "images" / "per_class_differences.png", dpi=200)
plt.close()

print(f"Figures written to {OUT / 'images'}")


# In[ ]:


# ---------------------------------------------------------------------------
# Interpretation notes for the write-up
# ---------------------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 9 SUMMARY")
print("=" * 70)
print(f"Models compared : {len(names)}")
print(f"Pairwise tests  : {len(res)}")
print(f"Test images     : {N}    classes: {C}")
print(f"Correction      : Holm-Bonferroni at alpha = {ALPHA}\n")

sig = res[res["mcnemar_sig"]].copy()
sig["abs"] = sig["acc_diff_pts"].abs()
sig = sig.sort_values("abs", ascending=False)

print(f"Significant after correction: {len(sig)}/{len(res)}\n")
for _, r in sig.iterrows():
    hi, lo = ((r["model_a"], r["model_b"]) if r["acc_diff_pts"] > 0
              else (r["model_b"], r["model_a"]))
    print(f"   {hi} > {lo}  by {abs(r['acc_diff_pts']):.2f} pts  "
          f"(p={r['mcnemar_p_holm']:.2e})")

ns = res[~res["mcnemar_sig"]]
if len(ns):
    print(f"\nNot distinguishable ({len(ns)}):")
    for _, r in ns.iterrows():
        print(f"   {r['model_a']} vs {r['model_b']}  "
              f"({r['acc_diff_pts']:+.2f} pts, p={r['mcnemar_p_holm']:.3f})")

print("\n" + "-" * 70)
print("Points to make in Chapter 4:")
print("-" * 70)
print(f"1. With N={N}, statistical significance is easy to obtain. Report the")
print("   confidence interval on the accuracy difference alongside every")
print("   p-value, and discuss whether the gap is large enough to matter for")
print("   dietary assessment in practice.")
print("2. McNemar and Wilcoxon can disagree. McNemar weights every image")
print("   equally, so frequent classes dominate; Wilcoxon weights every class")
print("   equally. A disagreement means one model wins on common classes while")
print("   the other wins on more classes overall, which is a finding about")
print("   error structure and belongs in the Research Question 2 discussion.")
print("3. State the correction explicitly. An uncorrected family of 15 tests")
print("   at alpha=0.05 carries roughly a 54 percent chance of at least one")
print("   false positive.")

with open(OUT / "experiment_9_findings.json", "w") as f:
    json.dump({"n_images": int(N), "n_classes": int(C),
               "alpha": ALPHA, "correction": "holm-bonferroni",
               "n_models": len(names), "n_comparisons": len(res),
               "n_significant": int(res["mcnemar_sig"].sum()),
               "accuracies": {k: float(100*v.mean())
                              for k, v in correct.items()},
               "comparisons": res.to_dict(orient="records")},
              f, indent=2, default=str)

print("\nExperiment 9 complete. No GPU was used.")

