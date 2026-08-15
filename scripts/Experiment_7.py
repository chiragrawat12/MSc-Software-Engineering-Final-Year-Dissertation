#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================================
# Experiment 7: Second VLM Family - SigLIP (Zero-Shot)
#
# Objective: establish that findings about "vision-language models" are not
# findings about CLIP specifically, by running an identical zero-shot pipeline
# on a model trained with a different contrastive objective.
#
# CLIP applies a softmax over the similarities within a batch, so the loss for
# any one pair depends on every other pair and the objective is coupled to
# batch size. SigLIP replaces this with a pairwise sigmoid loss, treating each
# image-text pair as an independent binary decision.
#
# Checkpoint: google/siglip-base-patch16-224, chosen as the closest match to
# CLIP ViT-B/32 in scale and input resolution. A larger SigLIP variant would
# confound the loss-function comparison with a model-capacity difference.
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


# Dependency check. SiglipTokenizer needs both SentencePiece and protobuf.
# Failing here with a clear message beats a stack trace from inside
# transformers halfway through the notebook.
#
#     pip install sentencepiece protobuf

_missing = []
for _pkg in ("sentencepiece", "google.protobuf"):
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg.split(".")[0])

if _missing:
    raise ImportError(
        "SigLIP's tokenizer requires: " + ", ".join(_missing) + "\n"
        "Install with:  pip install " + " ".join(_missing))

print("Tokenizer dependencies present: sentencepiece, protobuf")


# In[ ]:


import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: must precede the pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from transformers import SiglipModel, SiglipProcessor
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# In[ ]:


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME  = "google/siglip-base-patch16-224"
BATCH_SIZE  = 64
NUM_WORKERS = 8
SEED        = 42

# Experiment 1 CLIP baseline, for the direct comparison
CLIP_ZERO_SHOT_TOP1 = 84.19

torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:                       # running inside a notebook
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT    = PROJECT_ROOT / "data"

OUT = PROJECT_ROOT / "outputs" / "experiment_7"
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "preds").mkdir(parents=True, exist_ok=True)

print("Model:", MODEL_NAME)
print("Outputs:", OUT)


# In[ ]:


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

siglip_model = SiglipModel.from_pretrained(MODEL_NAME).to(device)
siglip_processor = SiglipProcessor.from_pretrained(MODEL_NAME)
siglip_model.eval()
for p in siglip_model.parameters():
    p.requires_grad = False

total_params  = sum(p.numel() for p in siglip_model.parameters())
vision_params = sum(p.numel() for p in siglip_model.vision_model.parameters())
text_params   = sum(p.numel() for p in siglip_model.text_model.parameters())

print(f"Total parameters : {total_params:,}")
print(f"  vision encoder : {vision_params:,}")
print(f"  text encoder   : {text_params:,}")

cfg = siglip_model.config
print(f"\nImage size : {cfg.vision_config.image_size}")
print(f"Patch size : {cfg.vision_config.patch_size}")
n_patches = (cfg.vision_config.image_size // cfg.vision_config.patch_size) ** 2
print(f"Patch tokens: {n_patches}   (CLIP ViT-B/32 has 49)")
print(f"Embedding dim: {cfg.text_config.hidden_size}   (CLIP has 512)")


# In[ ]:


# ---------------------------------------------------------------------------
# Test set
#
# shuffle=False throughout, so index i of the saved predictions refers to the
# same image as in every other experiment. Experiment 11's paired tests depend
# on this.
# ---------------------------------------------------------------------------

test_raw = Food101(root=str(DATA_ROOT), split="test", download=True,
                   transform=None)

CLASS_NAMES = test_raw.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Test size: {len(test_raw)}   Classes: {NUM_CLASSES}")
print("First five classes:", CLASS_NAMES[:5])


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


test_loader = DataLoader(test_raw, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, collate_fn=collate_fn)


# In[ ]:


# ---------------------------------------------------------------------------
# Prompt templates: identical to Experiment 2
# ---------------------------------------------------------------------------

def format_class_name(name):
    return name.replace("_", " ")


PROMPT_TEMPLATES = {
    "a photo of {class}":                    lambda c: f"a photo of {c}",
    "a photo of {class}, a type of food":    lambda c: f"a photo of {c}, a type of food",
    "a dish of {class}":                     lambda c: f"a dish of {c}",
    "{class}, a food dish":                  lambda c: f"{c}, a food dish",
    "an image of {class} food":              lambda c: f"an image of {c} food",
}

BASELINE_TEMPLATE = "a photo of {class}, a type of food"

for name, fn in PROMPT_TEMPLATES.items():
    print(f"{name:40s} -> {fn(format_class_name(CLASS_NAMES[0]))}")


# In[ ]:


# ---------------------------------------------------------------------------
# Text encoding
#
# Three things must be right here.
#
# 1. Dependencies. SiglipTokenizer is SentencePiece-based and additionally
#    requires protobuf. Both are checked at the top of this notebook so the
#    failure is a clear message rather than a stack trace from inside
#    transformers.
#
# 2. padding="max_length". SigLIP was trained with every caption padded to a
#    fixed 64-token length. Passing padding=True (the dynamic padding used for
#    CLIP) presents a sequence-length distribution the model never saw and
#    measurably degrades zero-shot accuracy.
#
# 3. Return type and projection. Verified empirically: SiglipModel has no
#    visual_projection or text_projection module at all. get_text_features
#    and get_image_features return a BaseModelOutputWithPooling whose
#    pooler_output IS the 768-d embedding. No projection is applied, unlike
#    CLIP where the projection is applied internally before the value is
#    returned.
# ---------------------------------------------------------------------------

EXPECTED_DIM = siglip_model.config.text_config.hidden_size    # 768 for base/16
print(f"Expected embedding dimension: {EXPECTED_DIM}")
print(f"Model has visual_projection: {hasattr(siglip_model, 'visual_projection')}")
print(f"Model has text_projection  : {hasattr(siglip_model, 'text_projection')}")


def _pooled(out):
    # Normalise the return type of get_text_features / get_image_features.
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(
        f"Unexpected return type: {type(out).__name__} with no pooler_output.")


@torch.no_grad()
def get_text_features(prompts):
    inputs = siglip_processor(text=prompts, padding="max_length",
                              return_tensors="pt").to(device)
    feats = _pooled(siglip_model.get_text_features(**inputs))
    if feats.shape[-1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected {EXPECTED_DIM}-d text features, got {feats.shape[-1]}.")
    return (feats / feats.norm(dim=-1, keepdim=True)).cpu()


template_text_features = {}
for name, fn in PROMPT_TEMPLATES.items():
    prompts = [fn(format_class_name(c)) for c in CLASS_NAMES]
    template_text_features[name] = get_text_features(prompts)
    print(f"{name:40s} encoded -> {tuple(template_text_features[name].shape)}")

# Embedding-level ensemble: average across templates, then renormalise
ensemble_features = torch.stack(list(template_text_features.values())).mean(0)
ensemble_features = ensemble_features / ensemble_features.norm(dim=-1,
                                                               keepdim=True)
print(f"\nEnsemble features: {tuple(ensemble_features.shape)}")


# In[ ]:


# ---------------------------------------------------------------------------
# Image encoding: one pass over the test set, reused for every template
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images(images):
    inputs = siglip_processor(images=images, return_tensors="pt").to(device)
    feats = _pooled(siglip_model.get_image_features(**inputs))
    if feats.shape[-1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected {EXPECTED_DIM}-d image features, got {feats.shape[-1]}.")
    return feats / feats.norm(dim=-1, keepdim=True)


# Smoke test on four images before committing to the full test set.
_probe = encode_images([test_raw[i][0] for i in range(4)])
print(f"Smoke test output: {tuple(_probe.shape)}")
assert _probe.shape == (4, EXPECTED_DIM), _probe.shape
assert torch.allclose(_probe.norm(dim=-1),
                      torch.ones(4, device=_probe.device), atol=1e-4)
print("Encoder verified: correct dimensionality, unit-norm output.")
del _probe


# In[ ]:


image_features_path = OUT / "siglip_image_features.npy"
labels_path         = OUT / "siglip_test_labels.npy"

if image_features_path.exists() and labels_path.exists():
    all_image_features = torch.from_numpy(np.load(image_features_path))
    all_labels = np.load(labels_path)
    if all_image_features.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Cached features are {all_image_features.shape[1]}-d, expected "
            f"{EXPECTED_DIM}. Delete {image_features_path} and re-run.")
    if len(all_image_features) != len(test_raw):
        raise ValueError(
            f"Cache has {len(all_image_features)} rows but the test set has "
            f"{len(test_raw)}. Delete the cache and re-run.")
    print("Loaded cached image features:", tuple(all_image_features.shape))
else:
    feats, labs = [], []
    for images, labels in tqdm(test_loader, desc="Encoding test images"):
        feats.append(encode_images(images).cpu())
        labs.extend(labels)

    all_image_features = torch.cat(feats)
    all_labels = np.array(labs)

    if len(all_image_features) != len(test_raw):
        raise ValueError(
            f"Encoded {len(all_image_features)} images but the test set has "
            f"{len(test_raw)}.")

    np.save(image_features_path, all_image_features.numpy())
    np.save(labels_path, all_labels)
    print("Image features:", tuple(all_image_features.shape))


# In[ ]:


# ---------------------------------------------------------------------------
# Scoring
#
# SigLIP's head is logit_scale * (image . text) + logit_bias, followed by a
# sigmoid. Both the scale and the bias are applied uniformly across the 101
# classes, so neither changes the argmax or the top-5 ordering; they are
# included so the reported scores are the model's calibrated outputs.
#
# Note that a sigmoid produces independent per-pair match probabilities that do
# not sum to one across classes, unlike CLIP's softmax. A softmax is applied
# alongside it purely so the two models' confidence values are on a comparable
# footing in the write-up.
# ---------------------------------------------------------------------------

logit_scale = siglip_model.logit_scale.exp().cpu()
logit_bias  = siglip_model.logit_bias.cpu()
print(f"logit_scale = {logit_scale.item():.4f}")
print(f"logit_bias  = {logit_bias.item():.4f}")


def evaluate(text_features, image_features, labels):
    logits = image_features @ text_features.T * logit_scale + logit_bias
    top5 = torch.topk(logits, k=5, dim=-1).indices.numpy()
    top1 = top5[:, 0]
    top1_acc = float((top1 == labels).mean() * 100)
    top5_acc = float(np.mean([l in row for l, row in zip(labels, top5)]) * 100)
    return top1_acc, top5_acc, top1, top5, logits


# In[ ]:


# ---------------------------------------------------------------------------
# Per-template results, plus the ensemble
# ---------------------------------------------------------------------------

rows = []
for name, tf in template_text_features.items():
    t1, t5, _, _, _ = evaluate(tf, all_image_features, all_labels)
    rows.append({"prompt_style": name, "top1_acc": t1, "top5_acc": t5})
    print(f"{name:40s} Top-1: {t1:5.2f}%   Top-5: {t5:5.2f}%")

ens_t1, ens_t5, ens_top1, ens_top5, ens_logits = evaluate(
    ensemble_features, all_image_features, all_labels)
rows.append({"prompt_style": "ENSEMBLE (avg of all 5)",
             "top1_acc": ens_t1, "top5_acc": ens_t5})
print(f"\n{'ENSEMBLE (avg of all 5)':40s} Top-1: {ens_t1:5.2f}%   "
      f"Top-5: {ens_t5:5.2f}%")

prompt_df = (pd.DataFrame(rows)
             .sort_values("top1_acc", ascending=False)
             .reset_index(drop=True))
prompt_df.to_csv(OUT / "siglip_prompt_results.csv", index=False)
print("\n" + prompt_df.round(2).to_string(index=False))


# In[ ]:


# ---------------------------------------------------------------------------
# Headline result: the baseline template, matching Experiment 1's protocol
# ---------------------------------------------------------------------------

baseline_tf = template_text_features[BASELINE_TEMPLATE]
top1_acc, top5_acc, preds, top5_preds, logits = evaluate(
    baseline_tf, all_image_features, all_labels)

print(f"SigLIP zero-shot (baseline prompt)")
print(f"  Top-1: {top1_acc:.2f}%")
print(f"  Top-5: {top5_acc:.2f}%")
print(f"\nCLIP zero-shot (Experiment 1)")
print(f"  Top-1: {CLIP_ZERO_SHOT_TOP1:.2f}%")
print(f"\nDifference: {top1_acc - CLIP_ZERO_SHOT_TOP1:+.2f} points")

report = classification_report(all_labels, preds,
                               target_names=CLASS_NAMES, digits=4)
with open(OUT / "siglip_classification_report.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Prompt: {BASELINE_TEMPLATE}\n")
    f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n\n")
    f.write(report)
print(f"\nReport -> {OUT / 'siglip_classification_report.txt'}")


# In[ ]:


# ---------------------------------------------------------------------------
# Persist artefacts, index-aligned with every other experiment
# ---------------------------------------------------------------------------

cm = confusion_matrix(all_labels, preds, labels=range(NUM_CLASSES))
per_class_acc = cm.diagonal() / cm.sum(axis=1)

np.save(OUT / "preds" / "siglip_preds.npy",  preds.astype(np.int16))
np.save(OUT / "preds" / "siglip_labels.npy", all_labels.astype(np.int16))
np.save(OUT / "preds" / "siglip_top5.npy",   top5_preds.astype(np.int16))
np.save(OUT / "preds" / "siglip_logits.npy", logits.numpy().astype(np.float16))
np.save(OUT / "preds" / "siglip_per_class_acc.npy",
        per_class_acc.astype(np.float32))
np.save(OUT / "preds" / "siglip_confusion_matrix.npy", cm.astype(np.int32))

# Guard against a silent ordering mismatch, which would invalidate the paired
# tests in Experiment 11 while leaving the aggregate accuracy plausible.
clip_labels_path = PROJECT_ROOT / "outputs" / "clipzeroshot" / \
                   "clip_zeroshot_labels.npy"
if clip_labels_path.exists():
    clip_labels = np.load(clip_labels_path)
    if np.array_equal(all_labels.astype(np.int16), clip_labels.astype(np.int16)):
        print("Label ordering matches the CLIP run: predictions are pairable.")
    else:
        raise AssertionError(
            "Label ordering differs from the CLIP run. Predictions are NOT "
            "pairable and Experiment 11 would be invalid.")
else:
    print(f"CLIP labels not found at {clip_labels_path}; "
          "skipping the alignment check.")


# In[ ]:


# ---------------------------------------------------------------------------
# Text-embedding similarity matrix, for the Research Question 2 analysis
#
# This is the operationalisation of semantic proximity between class names.
# Chapter 4 correlates it against off-diagonal confusion rates, and compares
# that correlation with the one obtained from CNN feature-space centroids.
# ---------------------------------------------------------------------------

text_sim = (baseline_tf @ baseline_tf.T).numpy()
np.save(OUT / "siglip_text_similarity.npy", text_sim.astype(np.float32))

tri = np.triu_indices(NUM_CLASSES, k=1)
pair_sims = text_sim[tri]
order = np.argsort(-pair_sims)[:15]

print("15 most semantically similar class-name pairs (SigLIP text space):\n")
for rank, o in enumerate(order, 1):
    i, j = tri[0][o], tri[1][o]
    print(f"{rank:>2d}. {CLASS_NAMES[i]:<28s} <-> {CLASS_NAMES[j]:<28s} "
          f"{pair_sims[o]:.4f}")


# In[ ]:


# ---------------------------------------------------------------------------
# Most-confused class pairs
# ---------------------------------------------------------------------------

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_offdiag = cm_norm.copy()
np.fill_diagonal(cm_offdiag, 0)

confused = []
for i in range(NUM_CLASSES):
    j = int(np.argmax(cm_offdiag[i]))
    confused.append((CLASS_NAMES[i], CLASS_NAMES[j], float(cm_offdiag[i, j]),
                     float(text_sim[i, j])))
confused.sort(key=lambda x: -x[2])

print("Top 10 most-confused pairs, with their text-space similarity:\n")
print(f"{'true':<28s} {'predicted':<28s} {'rate':>7s} {'text sim':>9s}")
for t, p, rate, sim in confused[:10]:
    print(f"{t:<28s} {p:<28s} {rate*100:6.1f}% {sim:9.4f}")

pd.DataFrame(confused, columns=["true_class", "predicted_class",
                                "confusion_rate", "text_similarity"]).to_csv(
    OUT / "siglip_confused_pairs.csv", index=False)


# In[ ]:


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# 1. Prompt sensitivity
plt.figure(figsize=(10, 6))
colors = ["#54A24B" if "ENSEMBLE" in s else "#4C78A8"
          for s in prompt_df["prompt_style"]]
plt.barh(prompt_df["prompt_style"], prompt_df["top1_acc"], color=colors)
for i, v in enumerate(prompt_df["top1_acc"]):
    plt.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
plt.xlabel("Top-1 Accuracy (%)")
plt.title("SigLIP Zero-Shot Accuracy by Prompt Style (Food-101)")
plt.gca().invert_yaxis()
plt.xlim(0, max(prompt_df["top1_acc"]) * 1.12)
plt.tight_layout()
plt.savefig(OUT / "images" / "siglip_prompt_comparison.png", dpi=200)
plt.close()

# 2. Confusion matrix
plt.figure(figsize=(20, 18))
sns.heatmap(cm_offdiag, cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("SigLIP Zero-Shot Confusion Matrix (Food-101, diagonal removed)",
          fontsize=16)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(OUT / "images" / "siglip_confusion_matrix.png", dpi=200)
plt.close()

# 3. Best and worst classes
per_class = sorted(zip(CLASS_NAMES, per_class_acc * 100), key=lambda x: x[1])
worst15, best15 = per_class[:15], per_class[-15:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh([x[0] for x in worst15], [x[1] for x in worst15], color="crimson")
axes[0].set_title("15 Worst-Performing Classes")
axes[0].set_xlabel("Accuracy (%)")
axes[1].barh([x[0] for x in best15], [x[1] for x in best15], color="seagreen")
axes[1].set_title("15 Best-Performing Classes")
axes[1].set_xlabel("Accuracy (%)")
plt.suptitle("SigLIP Zero-Shot Per-Class Accuracy (Food-101)")
plt.tight_layout()
plt.savefig(OUT / "images" / "siglip_per_class_accuracy.png", dpi=200)
plt.close()

print("Figures written to", OUT / "images")


# In[ ]:


# ---------------------------------------------------------------------------
# Direct comparison against CLIP
# ---------------------------------------------------------------------------

comparison = pd.DataFrame([
    {"model": "CLIP ViT-B/32",
     "objective": "softmax contrastive",
     "patch_tokens": 49,
     "embed_dim": 512,
     "top1_acc": CLIP_ZERO_SHOT_TOP1},
    {"model": "SigLIP base/16",
     "objective": "pairwise sigmoid",
     "patch_tokens": n_patches,
     "embed_dim": cfg.text_config.hidden_size,
     "top1_acc": top1_acc},
])
comparison["delta_vs_clip"] = comparison["top1_acc"] - CLIP_ZERO_SHOT_TOP1
comparison.to_csv(OUT / "siglip_vs_clip.csv", index=False)
print(comparison.round(2).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(comparison["model"], comparison["top1_acc"],
        color=["#4C78A8", "#54A24B"])
for i, v in enumerate(comparison["top1_acc"]):
    plt.text(i, v + 0.6, f"{v:.2f}%", ha="center", fontsize=11)
plt.ylabel("Top-1 Accuracy (%)")
plt.title("Zero-Shot Food-101: CLIP vs SigLIP")
plt.ylim(0, max(comparison["top1_acc"]) * 1.15)
plt.tight_layout()
plt.savefig(OUT / "images" / "siglip_vs_clip.png", dpi=200)
plt.close()


# In[ ]:


# ---------------------------------------------------------------------------
# Findings summary
# ---------------------------------------------------------------------------

best_single = prompt_df[prompt_df["prompt_style"] != "ENSEMBLE (avg of all 5)"]
spread = best_single["top1_acc"].max() - best_single["top1_acc"].min()

print("=" * 70)
print("EXPERIMENT 7 SUMMARY")
print("=" * 70)
print(f"SigLIP base/16 zero-shot Top-1 : {top1_acc:.2f}%")
print(f"SigLIP base/16 zero-shot Top-5 : {top5_acc:.2f}%")
print(f"CLIP ViT-B/32 zero-shot Top-1  : {CLIP_ZERO_SHOT_TOP1:.2f}%")
print(f"Difference                     : {top1_acc - CLIP_ZERO_SHOT_TOP1:+.2f} pts")
print()
print(f"Best single prompt : {best_single.iloc[0]['prompt_style']} "
      f"({best_single.iloc[0]['top1_acc']:.2f}%)")
print(f"Ensemble           : {ens_t1:.2f}% "
      f"({ens_t1 - best_single.iloc[0]['top1_acc']:+.2f} pts vs best single)")
print(f"Prompt spread      : {spread:.2f} pts across the five templates")
print()
print("Compare this spread against the equivalent figure from Experiment 2 to")
print("state whether SigLIP is more or less prompt-sensitive than CLIP.")

with open(OUT / "experiment_7_findings.json", "w") as f:
    json.dump({"model": MODEL_NAME,
               "top1_acc": top1_acc,
               "top5_acc": top5_acc,
               "clip_top1_acc": CLIP_ZERO_SHOT_TOP1,
               "delta_vs_clip": top1_acc - CLIP_ZERO_SHOT_TOP1,
               "ensemble_top1": ens_t1,
               "prompt_spread_pts": float(spread),
               "patch_tokens": int(n_patches),
               "embed_dim": int(cfg.text_config.hidden_size),
               "total_params": int(total_params),
               "prompt_results": prompt_df.to_dict(orient="records")},
              f, indent=2)

print("\nExperiment 7 complete.")

