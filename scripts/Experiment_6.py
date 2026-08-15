#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================================
# Experiment 6: Few-Shot CLIP at Low K
#
# Objective: test CLIP's pre-training advantage at genuinely low data volumes,
# complementing Experiment 5. Experiment 5's smallest subset (10 percent) is
# still 68 images per class; here K = 1, 2, 4, 8, 16.
#
# Design notes:
#   * 5 trials per K with different sampling seeds. At K=1 the classifier sees
#     101 images total, and which image is drawn per class swings accuracy by
#     several points, so a single run is noise rather than a measurement.
#     Results are reported as mean +/- standard deviation.
#   * The probe's regularisation strength C dominates behaviour at low K, so C
#     is selected on the held-out validation split, never on the test set.
#   * Few-shot images are drawn from the same 68,175-image training pool used
#     in Experiment 5, so nothing leaks from validation or test.
#   * Zero-shot CLIP (Experiment 1) is the K=0 anchor.
# ============================================================================

import torch
import socket

print("Hostname:", socket.gethostname())
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Run inside a Slurm GPU allocation.")
print("GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda")


# In[ ]:


import os
import json
import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless: must precede the pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


# In[ ]:


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_VALUES      = [1, 2, 4, 8, 16]
N_TRIALS      = 5                    # sampling repetitions per K
C_GRID        = [0.01, 0.1, 1.0, 10.0, 100.0]   # selected on validation
SPLIT_SEED    = 1234                 # must match Experiment 5
VAL_FRACTION  = 0.10                 # must match Experiment 5
BATCH_SIZE    = 64
NUM_WORKERS   = 4
MODEL_NAME    = "openai/clip-vit-base-patch32"

# Experiment 1 zero-shot baseline (the K=0 anchor)
ZERO_SHOT_TOP1 = 84.19

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"K values: {K_VALUES}   trials per K: {N_TRIALS}")
print(f"Zero-shot anchor: {ZERO_SHOT_TOP1:.2f}%")


# In[ ]:


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:                       # running inside a notebook
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT    = PROJECT_ROOT / "data"

OUT = PROJECT_ROOT / "outputs" / "experiment_6"
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "preds").mkdir(parents=True, exist_ok=True)

# Feature caches from Experiment 5 are reused if present, to avoid a second
# pass of the CLIP encoder over 100k images.
EXP5_CACHE = PROJECT_ROOT / "outputs" / "experiment_5" / "clip_linear_probe"

results_csv   = OUT / "fewshot_results.csv"
summary_csv   = OUT / "fewshot_summary.csv"
curve_png     = OUT / "images" / "fewshot_curve.png"
combined_png  = OUT / "images" / "fewshot_vs_experiment5.png"

print("Project root:", PROJECT_ROOT)
print("Outputs:", OUT)


# In[ ]:


# ---------------------------------------------------------------------------
# Splits: identical three-way partition to Experiment 5
#
#   train pool  68,175 (675 per class)  <- few-shot images drawn from here
#   validation   7,575 ( 75 per class)  <- C selection happens here
#   test        25,250 (250 per class)  <- evaluated once per fitted probe
# ---------------------------------------------------------------------------

train_raw = Food101(root=str(DATA_ROOT), split="train", download=True,
                    transform=None)
test_raw  = Food101(root=str(DATA_ROOT), split="test",  download=True,
                    transform=None)

CLASS_NAMES  = train_raw.classes
NUM_CLASSES  = len(CLASS_NAMES)
TRAIN_LABELS = np.array(train_raw._labels)

print(f"Train: {len(train_raw)}   Test: {len(test_raw)}   Classes: {NUM_CLASSES}")


# In[ ]:


def stratified_train_val_split(labels, val_fraction, seed):
    """Same function as Experiment 5, with the same SPLIT_SEED, so the
    partition is byte-for-byte identical across experiments."""
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[lab].append(idx)

    rng = np.random.default_rng(seed)
    train_pool, val_idx = [], []
    for lab, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val].tolist())
        train_pool.extend(idxs[n_val:].tolist())
    return sorted(train_pool), sorted(val_idx)


TRAIN_POOL_IDX, VAL_IDX = stratified_train_val_split(
    TRAIN_LABELS, VAL_FRACTION, SPLIT_SEED)

assert len(set(TRAIN_POOL_IDX) & set(VAL_IDX)) == 0, "train/val overlap"
print(f"Train pool: {len(TRAIN_POOL_IDX)}   Validation: {len(VAL_IDX)}")
print(f"Per class: {len(TRAIN_POOL_IDX)//NUM_CLASSES} train, "
      f"{len(VAL_IDX)//NUM_CLASSES} val")


# In[ ]:


# ---------------------------------------------------------------------------
# CLIP feature extraction (frozen encoder, inference only)
# ---------------------------------------------------------------------------

clip_model = CLIPModel.from_pretrained(MODEL_NAME,
                                       use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in clip_model.parameters())
print(f"CLIP loaded: {MODEL_NAME}  ({n_params:,} parameters, all frozen)")


# In[ ]:


# ---------------------------------------------------------------------------
# Feature extraction
#
# Version note, verified empirically rather than assumed:
#
#   transformers 5.x  CLIPModel.get_image_features returns a
#                     BaseModelOutputWithPooling whose pooler_output has
#                     ALREADY had visual_projection applied. It is the 512-d
#                     embedding in CLIP's shared image-text space.
#                     (last_hidden_state remains the 768-d unprojected
#                     sequence, so it must not be used here.)
#   transformers 4.x  the same call returns a plain 512-d tensor.
#
# Either way the correct action is to take the pooled tensor and apply no
# projection. Applying visual_projection on top of pooler_output raises a
# shape error, because that layer maps 768 -> 512 and its input is already
# 512-d.
#
# The helper below is dimension-aware rather than version-aware: it projects
# only if the tensor it receives is genuinely unprojected and a matching
# projection layer exists. That stays correct under either library behaviour
# instead of hard-coding one of them.
# ---------------------------------------------------------------------------

EXPECTED_DIM = 512


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def _pooled(out):
    # Normalise the return type of get_image_features.
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(
        f"Unexpected return type from get_image_features: "
        f"{type(out).__name__} with no pooler_output.")


def _to_shared_space(t, projection, expected_dim):
    # Already in the shared space: return unchanged. This is the path taken
    # for CLIP under both transformers 4.x and 5.x.
    if t.shape[-1] == expected_dim:
        return t
    # Genuinely unprojected: project, but only if the shapes agree.
    if projection is not None and t.shape[-1] == projection.in_features:
        return projection(t)
    raise ValueError(
        f"Got a {t.shape[-1]}-d tensor; expected {expected_dim}-d, and no "
        f"compatible projection is available.")


@torch.no_grad()
def encode_images(images):
    # Returns L2-normalised 512-d CLIP image embeddings.
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    out = _pooled(clip_model.get_image_features(**inputs))
    out = _to_shared_space(out, getattr(clip_model, "visual_projection", None),
                           EXPECTED_DIM)
    return out / out.norm(dim=-1, keepdim=True)


@torch.no_grad()
def extract_features(dataset, indices, desc):
    ds = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)
    feats, labs = [], []
    for images, labels in tqdm(loader, desc=desc):
        feats.append(encode_images(images).cpu())
        labs.extend(labels)

    feats = torch.cat(feats).numpy()
    if feats.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected {EXPECTED_DIM}-d CLIP features, got {feats.shape[1]}.")
    return feats, np.array(labs)


# In[ ]:


# Smoke test on four images before committing to the full dataset. Encoding
# 100k images takes tens of minutes, so a shape error should surface in
# seconds rather than after a long run.

_probe_imgs = [train_raw[i][0] for i in TRAIN_POOL_IDX[:4]]
_probe = encode_images(_probe_imgs)

print(f"Smoke test output: {tuple(_probe.shape)}")
assert _probe.shape == (4, EXPECTED_DIM), _probe.shape
assert torch.allclose(_probe.norm(dim=-1),
                      torch.ones(4, device=_probe.device), atol=1e-4), \
    "features are not unit norm"
print("Encoder verified: correct dimensionality, unit-norm output.")

del _probe, _probe_imgs


# In[ ]:


def _validate(feats, labs, expected_n, source):
    # Cached arrays are checked exactly as freshly extracted ones are.
    # Without this, a stale cache written by earlier code would load without
    # complaint and corrupt every result downstream.
    if feats.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"{source}: expected {EXPECTED_DIM}-d CLIP features, got "
            f"{feats.shape[1]}. Delete this cache and re-extract.")
    if expected_n is not None and len(feats) != expected_n:
        raise ValueError(
            f"{source}: expected {expected_n} rows, got {len(feats)}. "
            f"The split may have changed; delete the cache and re-extract.")
    if len(feats) != len(labs):
        raise ValueError(
            f"{source}: {len(feats)} features but {len(labs)} labels.")
    return feats, labs


def load_or_extract(cache_stem, dataset, indices, desc):
    # Reuse the Experiment 5 cache where available, validating every path.
    local_f = OUT / f"{cache_stem}_features.npy"
    local_l = OUT / f"{cache_stem}_labels.npy"
    exp5_f  = EXP5_CACHE / f"{cache_stem}_features.npy"
    exp5_l  = EXP5_CACHE / f"{cache_stem}_labels.npy"

    n = len(indices) if indices is not None else len(dataset)

    # Both files must exist: a job killed between the two np.save calls would
    # otherwise leave a features file with no matching labels file.
    if local_f.exists() and local_l.exists():
        print(f"Loaded cached {cache_stem} features (experiment 6)")
        return _validate(np.load(local_f), np.load(local_l), n,
                         f"experiment_6/{cache_stem}")

    if exp5_f.exists() and exp5_l.exists():
        print(f"Reusing {cache_stem} features from Experiment 5")
        return _validate(np.load(exp5_f), np.load(exp5_l), n,
                         f"experiment_5/{cache_stem}")

    feats, labs = extract_features(dataset, indices, desc)
    np.save(local_f, feats)
    np.save(local_l, labs)
    return feats, labs


# In[ ]:


pool_feats, pool_labels = load_or_extract(
    "pool", train_raw, TRAIN_POOL_IDX, "Encoding train pool")
val_feats, val_labels = load_or_extract(
    "val", train_raw, VAL_IDX, "Encoding validation split")
test_feats, test_labels = load_or_extract(
    "test", test_raw, None, "Encoding test set")

print("Pool :", pool_feats.shape)
print("Val  :", val_feats.shape)
print("Test :", test_feats.shape)

# Map dataset index -> row within pool_feats
POS_IN_POOL = {idx: i for i, idx in enumerate(TRAIN_POOL_IDX)}
POOL_BY_CLASS = defaultdict(list)
for idx in TRAIN_POOL_IDX:
    POOL_BY_CLASS[TRAIN_LABELS[idx]].append(idx)


# In[ ]:


# ---------------------------------------------------------------------------
# Few-shot sampling
#
# Nested across K within a trial: the K=1 image for a class is also present in
# that trial's K=2 set, and so on. Nesting means the curve reflects how much
# data was added rather than which images happened to be drawn at each K.
# ---------------------------------------------------------------------------

def sample_k_shot(k, trial_seed):
    """Return dataset indices: k images per class, drawn from the train pool."""
    rng = np.random.default_rng(trial_seed)
    selected = []
    for cls in sorted(POOL_BY_CLASS):
        idxs = np.array(POOL_BY_CLASS[cls])
        rng.shuffle(idxs)
        selected.extend(idxs[:k].tolist())
    return selected


# Sanity check: nesting within a trial, and correct counts
_t = 0
for k_small, k_big in zip(K_VALUES, K_VALUES[1:]):
    s = set(sample_k_shot(k_small, 1000 + _t))
    b = set(sample_k_shot(k_big,   1000 + _t))
    assert s.issubset(b), f"K={k_small} not nested inside K={k_big}"
for k in K_VALUES:
    assert len(sample_k_shot(k, 1000)) == k * NUM_CLASSES
print("Sampling verified: nested across K, correct counts.")


# In[ ]:


# ---------------------------------------------------------------------------
# Few-shot probe
#
# C is chosen on the validation split. At K=1 the optimal C differs from the
# K=16 optimum by orders of magnitude, so leaving it fixed at 1.0 would
# understate few-shot performance and misrepresent the curve.
# ---------------------------------------------------------------------------

def fit_fewshot_probe(k, trial_seed, verbose=False):
    indices = sample_k_shot(k, trial_seed)
    rows = [POS_IN_POOL[i] for i in indices]
    X, y = pool_feats[rows], pool_labels[rows]

    best_C, best_val_acc, best_clf = None, -1.0, None
    for C in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=C,
                                 random_state=trial_seed)
        clf.fit(X, y)
        val_acc = float((clf.predict(val_feats) == val_labels).mean())
        if verbose:
            print(f"      C={C:<7g} val={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_C, best_val_acc, best_clf = C, val_acc, clf

    test_preds = best_clf.predict(test_feats)
    test_acc = float((test_preds == test_labels).mean())
    return test_acc, best_val_acc, best_C, test_preds


# In[ ]:


records = []

for k in K_VALUES:
    print(f"\n{'='*66}\nK = {k}  ({k * NUM_CLASSES} training images total)\n{'='*66}")
    for trial in range(N_TRIALS):
        trial_seed = 1000 + trial
        test_acc, val_acc, C, preds = fit_fewshot_probe(k, trial_seed)

        records.append({"K": k, "trial": trial, "seed": trial_seed,
                        "train_images": k * NUM_CLASSES,
                        "best_C": C,
                        "val_acc": val_acc * 100,
                        "test_acc": test_acc * 100})

        print(f"  trial {trial}: C={C:<7g} val={val_acc*100:5.2f}%  "
              f"TEST={test_acc*100:5.2f}%")

        if trial == 0:      # keep one prediction vector per K for Exp 10/11
            np.save(OUT / "preds" / f"clip_fewshot_K{k}_preds.npy",
                    preds.astype(np.int16))

np.save(OUT / "preds" / "clip_fewshot_labels.npy", test_labels.astype(np.int16))

records_df = pd.DataFrame(records)
records_df.to_csv(results_csv, index=False)
print(f"\nSaved per-trial results -> {results_csv}")


# In[ ]:


# ---------------------------------------------------------------------------
# Aggregate: mean and standard deviation across trials
# ---------------------------------------------------------------------------

summary = (records_df.groupby("K")
           .agg(train_images=("train_images", "first"),
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std"),
                min_test_acc=("test_acc", "min"),
                max_test_acc=("test_acc", "max"),
                modal_C=("best_C", lambda s: s.mode().iloc[0]))
           .reset_index())

summary["vs_zeroshot_pts"] = summary["mean_test_acc"] - ZERO_SHOT_TOP1
summary.to_csv(summary_csv, index=False)

print(summary.round(2).to_string(index=False))
print(f"\nSaved summary -> {summary_csv}")


# In[ ]:


# ---------------------------------------------------------------------------
# Figure: few-shot curve with error bars and the zero-shot anchor
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(summary["K"], summary["mean_test_acc"],
            yerr=summary["std_test_acc"].fillna(0),
            marker="o", markersize=9, linewidth=2, capsize=5,
            color="#B279A2", label="CLIP linear probe (few-shot)")

ax.axhline(ZERO_SHOT_TOP1, linestyle="--", linewidth=2, color="#4C78A8",
           label=f"CLIP zero-shot, K=0 ({ZERO_SHOT_TOP1:.2f}%)")

for _, r in summary.iterrows():
    ax.annotate(f"{r['mean_test_acc']:.1f}%", (r["K"], r["mean_test_acc"]),
                textcoords="offset points", xytext=(0, 13), ha="center",
                fontsize=9)

ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.set_xlabel("Labelled images per class (K)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Experiment 6: Few-Shot CLIP on Food-101\n"
             f"mean +/- s.d. over {N_TRIALS} sampling trials, "
             "evaluated on the full 25,250-image test set")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(curve_png, dpi=200)
plt.close()
print(f"Figure -> {curve_png}")


# In[ ]:


# ---------------------------------------------------------------------------
# Figure: few-shot placed on the Experiment 5 axis
#
# Converts K to a percentage of the 675-image-per-class training pool, so the
# low-data regime can be read against the Experiment 5 curves. This is the
# figure that answers "how little data does CLIP need to match a CNN trained
# on everything".
# ---------------------------------------------------------------------------

exp5_csv = PROJECT_ROOT / "outputs" / "experiment_5" / \
           "experiment5_data_efficiency_results.csv"

fig, ax = plt.subplots(figsize=(10, 6))

per_class_pool = len(TRAIN_POOL_IDX) / NUM_CLASSES        # 675
summary["pct_of_pool"] = summary["train_images"] / len(TRAIN_POOL_IDX) * 100

ax.errorbar(summary["pct_of_pool"], summary["mean_test_acc"],
            yerr=summary["std_test_acc"].fillna(0),
            marker="*", markersize=14, linewidth=2, capsize=4,
            color="#B279A2", label="CLIP few-shot (Experiment 6)")

if exp5_csv.exists():
    exp5 = pd.read_csv(exp5_csv)
    colors = {"AlexNet": "#E45756", "ResNet34": "#F58518",
              "ResNet50": "#4C78A8", "YOLOv8s-cls": "#54A24B",
              "CLIP Linear Probe": "#B279A2"}
    markers = {"AlexNet": "o", "ResNet34": "s", "ResNet50": "^",
               "YOLOv8s-cls": "D", "CLIP Linear Probe": "P"}
    for model, grp in exp5.groupby("Model"):
        grp = grp.sort_values("Training Data (%)")
        ax.plot(grp["Training Data (%)"], grp["Accuracy (%)"],
                marker=markers.get(model, "o"), color=colors.get(model),
                linewidth=2, markersize=7, alpha=0.85,
                label=f"{model} (Experiment 5)")
else:
    print(f"Experiment 5 results not found at {exp5_csv}; "
          "plotting few-shot curve alone.")

ax.axhline(ZERO_SHOT_TOP1, linestyle="--", linewidth=1.8, color="#4C78A8",
           alpha=0.7, label=f"CLIP zero-shot ({ZERO_SHOT_TOP1:.2f}%)")

ax.set_xscale("log")
ax.set_xlabel("Training data used (% of the 68,175-image training pool, log scale)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Experiments 5 and 6 combined: data efficiency across the full range")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(combined_png, dpi=200)
plt.close()
print(f"Figure -> {combined_png}")


# In[ ]:


# ---------------------------------------------------------------------------
# Headline findings for the Chapter 4 write-up
# ---------------------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 6 SUMMARY")
print("=" * 70)
print(f"Zero-shot (K=0):  {ZERO_SHOT_TOP1:.2f}%\n")

for _, r in summary.iterrows():
    delta = r["mean_test_acc"] - ZERO_SHOT_TOP1
    sd = 0.0 if pd.isna(r["std_test_acc"]) else r["std_test_acc"]
    print(f"K={int(r['K']):>2d}  ({int(r['train_images']):>4d} images, "
          f"{r['pct_of_pool']:5.2f}% of pool):  "
          f"{r['mean_test_acc']:5.2f}% +/- {sd:4.2f}   "
          f"({delta:+5.2f} pts vs zero-shot)")

crossover = summary[summary["mean_test_acc"] > ZERO_SHOT_TOP1]
print()
if not crossover.empty:
    k_star = int(crossover.iloc[0]["K"])
    print(f"Few-shot overtakes zero-shot at K={k_star} "
          f"({k_star * NUM_CLASSES} labelled images, "
          f"{crossover.iloc[0]['pct_of_pool']:.2f}% of the training pool).")
else:
    print(f"Few-shot does not overtake zero-shot within K <= {max(K_VALUES)}. "
          "This is itself a finding: CLIP's pre-trained alignment is stronger "
          "than what a probe can learn from this little supervision.")

max_sd = summary["std_test_acc"].max()
print(f"\nLargest standard deviation across trials: {max_sd:.2f} pts "
      f"(at K={int(summary.loc[summary['std_test_acc'].idxmax(), 'K'])}), "
      "which is why single-run few-shot numbers are unreliable.")

with open(OUT / "experiment_6_findings.json", "w") as f:
    json.dump({"zero_shot_top1": ZERO_SHOT_TOP1,
               "k_values": K_VALUES,
               "n_trials": N_TRIALS,
               "summary": summary.to_dict(orient="records")}, f, indent=2)

print("\nExperiment 6 complete.")

