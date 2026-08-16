#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
Experiment 5: Data-Efficiency Curve (CNN vs CLIP)

Trains AlexNet, ResNet34, ResNet50, YOLOv8s-cls and a CLIP linear probe on
stratified subsets of Food-101 (10, 25, 50, 75, 100 percent) and plots test
accuracy against training set size.
"""

# In[2]:

import argparse
import copy
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# In[3]:

import matplotlib
matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt

# In[4]:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# In[5]:

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                    help="Run seed: initialisation and shuffling.")
    p.add_argument("--split-seed", type=int, default=1234,
                    help="Split seed: held constant across all runs and models.")
    p.add_argument("--epochs", type=int, default=60,
                    help="Max epochs per run (matches Chapter 3 prototypes).")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--models", nargs="+", default=["alexnet", "resnet34",
                    "resnet50", "yolo", "clip"])
    p.add_argument("--fractions", nargs="+", type=float,
                    default=[0.10, 0.25, 0.50, 0.75, 1.00])
    # parse_known_args rather than parse_args, so the script also runs inside
    # a Jupyter kernel, which injects its own -f connection-file argument.
    parsed, _ = p.parse_known_args()
    return parsed

# In[6]:

args = parse_args()

# In[7]:

SUBSET_FRACTIONS = args.fractions
NUM_CLASSES = 101

# In[8]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Run inside a Slurm GPU allocation.")
print("GPU:", torch.cuda.get_device_name(0))

# In[ ]:

# Enable TF32 and cuDNN autotuning for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# In[ ]:

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# In[ ]:

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# In[ ]:

set_seed(args.seed)
g = torch.Generator()
g.manual_seed(args.seed)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

# In[ ]:

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    SCRIPT_DIR = Path.cwd()

# In[ ]:

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
FOOD101_IMG_DIR = DATA_ROOT / "food-101" / "images"

# In[ ]:

EXP = "experiment_5"
OUT_ROOT = PROJECT_ROOT / "outputs" / EXP
MODELS_ROOT = PROJECT_ROOT / "models" / EXP

# In[ ]:

MODEL_KEYS = ["alexnet", "resnet34", "resnet50", "yolo", "clip_linear_probe"]

# In[ ]:

# Pre-create the output/checkpoint folder structure for every model up front
for key in MODEL_KEYS:
    (OUT_ROOT / key / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / key / "preds").mkdir(parents=True, exist_ok=True)
    (MODELS_ROOT / key / "best_models").mkdir(parents=True, exist_ok=True)
    (MODELS_ROOT / key / "saved_models").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)

# In[ ]:

# --- Ultralytics: keep every artefact inside the project tree ---------------
# Without this, val() writes to ./runs/classify/val and the AMP check
# downloads its reference checkpoint into the repository root.
YOLO_RUNS_DIR = OUT_ROOT / "yolo" / "runs"
YOLO_WEIGHTS_DIR = MODELS_ROOT / "yolo" / "pretrained"
YOLO_SUBSETS_DIR = DATA_ROOT / EXP / "yolo_subsets"
for d in (YOLO_RUNS_DIR, YOLO_WEIGHTS_DIR, YOLO_SUBSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# In[ ]:

# Only import/configure ultralytics when YOLO is actually part of this run
if "yolo" in args.models:
    from ultralytics import YOLO, settings as ultra_settings
    ultra_settings.update({
        "runs_dir": str(YOLO_RUNS_DIR.resolve()),
        "weights_dir": str(YOLO_WEIGHTS_DIR.resolve()),
        "datasets_dir": str(YOLO_SUBSETS_DIR.resolve()),
    })

# In[ ]:

print("Project root:", PROJECT_ROOT)
print("Outputs:", OUT_ROOT)

# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

# In[ ]:

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

# In[ ]:

def train_transform(size):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        NORM,
    ])

# In[ ]:

def eval_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORM,
    ])

# In[ ]:

# Load the train split once just to get per-image labels and class names,
# without paying the cost of loading actual images (transform=None)
_index_probe = Food101(root=str(DATA_ROOT), split="train", download=True,
                        transform=None)
TRAIN_LABELS = np.array(_index_probe._labels)
CLASS_NAMES = _index_probe.classes
del _index_probe

# In[ ]:

def stratified_train_val_split(labels, val_fraction, seed):
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

# In[ ]:

# Split held constant across all models/fractions, so every model sees the
# same validation set and the same pool to draw training subsets from
TRAIN_POOL_IDX, VAL_IDX = stratified_train_val_split(
    TRAIN_LABELS, args.val_fraction, args.split_seed)

# In[ ]:

print(f"Train pool: {len(TRAIN_POOL_IDX)} Validation: {len(VAL_IDX)}")
assert len(set(TRAIN_POOL_IDX) & set(VAL_IDX)) == 0

# In[ ]:

def nested_subset_indices(pool_idx, labels, fraction, seed):
    """Stratified, nested subset of the training pool."""
    by_class = defaultdict(list)
    for idx in pool_idx:
        by_class[labels[idx]].append(idx)

    rng = np.random.default_rng(seed)  # same seed for every fraction
    selected = []
    for lab in sorted(by_class):
        idxs = np.array(by_class[lab])
        rng.shuffle(idxs)  # identical ordering each call
        n = max(1, int(round(len(idxs) * fraction)))
        selected.extend(idxs[:n].tolist())
    return sorted(selected)

# In[ ]:

# Because the shuffle order is identical for every fraction, each smaller
# subset is a strict subset of the next larger one (nested), so the curve
# reflects adding data rather than resampling different data each time
SUBSETS = {f: nested_subset_indices(TRAIN_POOL_IDX, TRAIN_LABELS, f,
                                     args.split_seed)
           for f in SUBSET_FRACTIONS}

# In[ ]:

for f, idxs in SUBSETS.items():
    print(f"  {int(f*100):>3d}% subset -> {len(idxs):>6d} images "
          f"({len(idxs)//NUM_CLASSES} per class)")

# In[ ]:

# Verify nesting
_fracs = sorted(SUBSETS)
for a, b in zip(_fracs, _fracs[1:]):
    assert set(SUBSETS[a]).issubset(set(SUBSETS[b])), "subsets are not nested"
print("Nesting verified.")

# ----------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------

# In[ ]:

class AlexNet(nn.Module):
    """Same implementation as the Chapter 3 prototype (227x227 input, LRN)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# In[ ]:

def make_alexnet():
    return AlexNet(NUM_CLASSES)

# In[ ]:

def make_resnet34():
    m = models.resnet34(weights=None)
    m.fc = nn.Linear(512, NUM_CLASSES)
    return m

# In[ ]:

def make_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, NUM_CLASSES)
    return m

# In[ ]:

# name, constructor, and required input size for each CNN architecture
CNN_SPECS = {
    "alexnet": ("AlexNet", make_alexnet, 227),
    "resnet34": ("ResNet34", make_resnet34, 224),
    "resnet50": ("ResNet50", make_resnet50, 224),
}

# ----------------------------------------------------------------------------
# CNN training
# ----------------------------------------------------------------------------

# In[ ]:

def build_loaders(key, size, fraction):
    tr_ds = Food101(root=str(DATA_ROOT), split="train", download=False,
                     transform=train_transform(size))
    ev_ds = Food101(root=str(DATA_ROOT), split="train", download=False,
                     transform=eval_transform(size))
    te_ds = Food101(root=str(DATA_ROOT), split="test", download=False,
                     transform=eval_transform(size))

    train_loader = DataLoader(
        Subset(tr_ds, SUBSETS[fraction]), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, drop_last=True,
        worker_init_fn=seed_worker, generator=g)

    val_loader = DataLoader(
        Subset(ev_ds, VAL_IDX), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    test_loader = DataLoader(
        te_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# In[ ]:

@torch.no_grad()
def evaluate(model, loader, collect=False):
    model.eval()
    correct = total = 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if collect:
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    acc = correct / total
    if collect:
        return acc, np.array(preds_all, np.int16), np.array(labels_all, np.int16)
    return acc

# In[ ]:

def train_cnn(key, fraction):
    name, ctor, size = CNN_SPECS[key]
    pct = int(fraction * 100)
    tag = f"{key}_{pct}pct"
    print(f"\n{'='*70}\n{name} @ {pct}% ({len(SUBSETS[fraction])} images)\n{'='*70}")

    train_loader, val_loader, test_loader = build_loaders(key, size, fraction)

    model = ctor().to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                           weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_path = MODELS_ROOT / key / "best_models" / f"best_{tag}.pth"
    final_path = MODELS_ROOT / key / "saved_models" / f"{tag}.pth"

    best_val, no_improve, best_state = 0.0, 0, None
    history = {"epoch": [], "train_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"{tag} ep{epoch}",
                                    leave=False):
            images = images.to(device, non_blocking=True) \
                .to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / len(train_loader)

        # ---- selection on the VALIDATION split, never on test --------------
        val_acc = evaluate(model, val_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        print(f"  ep{epoch:>3d} loss={train_loss:.4f} val={val_acc*100:.2f}%")

        if val_acc > best_val:
            best_val, no_improve = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final_path)

    # ---- single final evaluation on the test set --------------------------
    test_acc, preds, labels = evaluate(model, test_loader, collect=True)
    np.save(OUT_ROOT / key / "preds" / f"{tag}_preds.npy", preds)
    np.save(OUT_ROOT / key / "preds" / f"{tag}_labels.npy", labels)
    with open(OUT_ROOT / key / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  best val={best_val*100:.2f}% TEST={test_acc*100:.2f}%")

    del model
    torch.cuda.empty_cache()
    return test_acc, best_val

# ----------------------------------------------------------------------------
# YOLOv8s-cls
# ----------------------------------------------------------------------------

# In[ ]:

def build_yolo_dirs(fraction):
    """train/ val/ test/ hierarchy of symlinks for one subset size."""
    pct = int(fraction * 100)
    root = YOLO_SUBSETS_DIR / f"frac_{pct}"
    if root.exists():
        shutil.rmtree(root)

    samples = Food101(root=str(DATA_ROOT), split="train", download=False,
                       transform=None)._image_files
    test_samples = Food101(root=str(DATA_ROOT), split="test", download=False,
                            transform=None)._image_files

    def link(split, paths):
        for p in paths:
            p = Path(p)
            dest_dir = root / split / p.parent.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / p.name
            if not dest.exists():
                os.symlink(p.resolve(), dest)

    link("train", [samples[i] for i in SUBSETS[fraction]])
    link("val", [samples[i] for i in VAL_IDX])
    link("test", test_samples)
    return root

# In[ ]:

def train_yolo(fraction):
    pct = int(fraction * 100)
    tag = f"yolov8s_{pct}pct"
    print(f"\n{'='*70}\nYOLOv8s-cls @ {pct}%\n{'='*70}")

    subset_dir = build_yolo_dirs(fraction)
    model = YOLO("yolov8s-cls.yaml")  # .yaml = from scratch, no weights

    model.train(
        data=str(subset_dir.resolve()),
        epochs=args.epochs,
        imgsz=224,
        batch=args.batch_size,
        optimizer="SGD", lr0=0.01, momentum=0.9, weight_decay=5e-4,
        device=0,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=False,  # skips the AMP check that downloads a model
        project=str(YOLO_RUNS_DIR),
        name=tag,
        exist_ok=True,
        verbose=False,
        plots=True,
    )

    # split="test" and an explicit project/name, so nothing lands in the root
    metrics = model.val(
        data=str(subset_dir.resolve()),
        split="test",
        imgsz=224,
        batch=args.batch_size,
        device=0,
        project=str(YOLO_RUNS_DIR),
        name=f"{tag}_test",
        exist_ok=True,
    )

    test_acc = float(metrics.top1)

    best_weights = YOLO_RUNS_DIR / tag / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights,
                      MODELS_ROOT / "yolo" / "saved_models" / f"{tag}.pt")

    print(f"  TEST={test_acc*100:.2f}%")
    return test_acc, None

# ----------------------------------------------------------------------------
# CLIP linear probe
#
# Uses LogisticRegression with the same settings as Experiment 3, so the
# 100 percent point of this curve reproduces the Experiment 3 result rather
# than contradicting it.
# ----------------------------------------------------------------------------

# In[ ]:

def clip_feature_extractor():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor

# In[ ]:

# CLIP's shared image-text space is 512-d. Depending on the transformers
# release, get_image_features returns either a plain tensor (already projected)
# or a BaseModelOutputWithPooling whose pooler_output is the vision tower's
# 768-d PRE-projection representation. Those are different spaces: a probe
# fitted on the 768-d output is not aligned with CLIP's text embeddings, which
# produces no error, only wrong numbers. The projection is therefore applied
# explicitly when needed, and the dimensionality is checked on every path.
CLIP_EMBED_DIM = 512

# In[9]:

@torch.no_grad()
def encode_images(model, processor, images):
    """Projected, L2-normalised 512-d CLIP image embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    out = model.get_image_features(**inputs)

    if not torch.is_tensor(out):
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
        out = pooled
    else:
        out = out

    # Only project if genuinely unprojected (768-d), never if already 512-d
    if out.shape[-1] != CLIP_EMBED_DIM:
        if out.shape[-1] == model.visual_projection.in_features:
            out = model.visual_projection(out)
        else:
            raise ValueError(f"Got a {out.shape[-1]}-d tensor; no compatible projection available.")

    return out / out.norm(dim=-1, keepdim=True)

# In[ ]:

@torch.no_grad()
def extract_features(model, processor, dataset, indices, desc):
    def collate(batch):
        return [b[0] for b in batch], [b[1] for b in batch]

    ds = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=collate)

    feats, labs = [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        feats.append(encode_images(model, processor, images).cpu())
        labs.extend(labels)

    feats = torch.cat(feats).numpy()
    if feats.shape[1] != CLIP_EMBED_DIM:
        raise ValueError(
            f"Expected {CLIP_EMBED_DIM}-d projected CLIP features, got "
            f"{feats.shape[1]}. visual_projection was not applied.")
    return feats, np.array(labs)

# In[ ]:

def validate_cache(feats, labs, expected_n, source):
    """Cached arrays are checked exactly as freshly extracted ones are."""
    if feats.shape[1] != CLIP_EMBED_DIM:
        raise ValueError(
            f"{source}: expected {CLIP_EMBED_DIM}-d projected CLIP features, "
            f"got {feats.shape[1]}. This cache was written by code that did "
            f"not apply visual_projection. Delete it and re-extract.")
    if expected_n is not None and len(feats) != expected_n:
        raise ValueError(
            f"{source}: expected {expected_n} rows, got {len(feats)}. "
            f"The split may have changed; delete the cache and re-extract.")
    if len(feats) != len(labs):
        raise ValueError(
            f"{source}: {len(feats)} features but {len(labs)} labels.")
    return feats, labs

# In[ ]:

def run_clip_probe():
    model, processor = clip_feature_extractor()

    train_raw = Food101(root=str(DATA_ROOT), split="train", download=False,
                         transform=None)
    test_raw = Food101(root=str(DATA_ROOT), split="test", download=False,
                        transform=None)

    cache = OUT_ROOT / "clip_linear_probe"

    # Extract the full training pool once, then slice per fraction.
    # Both files must exist: a job killed between the two np.save calls would
    # otherwise leave a features file with no matching labels file.
    pool_f = cache / "pool_features.npy"
    pool_l = cache / "pool_labels.npy"
    if pool_f.exists() and pool_l.exists():
        pool_feats, pool_labels = validate_cache(
            np.load(pool_f), np.load(pool_l), len(TRAIN_POOL_IDX),
            "cached pool features")
        print(f"Loaded cached pool features: {pool_feats.shape}")
    else:
        pool_feats, pool_labels = extract_features(
            model, processor, train_raw, TRAIN_POOL_IDX, "CLIP train pool")
        np.save(pool_f, pool_feats)
        np.save(pool_l, pool_labels)

    test_f = cache / "test_features.npy"
    test_l = cache / "test_labels.npy"
    if test_f.exists() and test_l.exists():
        test_feats, test_labels = validate_cache(
            np.load(test_f), np.load(test_l), len(test_raw),
            "cached test features")
        print(f"Loaded cached test features: {test_feats.shape}")
    else:
        test_feats, test_labels = extract_features(
            model, processor, test_raw, None, "CLIP test")
        np.save(test_f, test_feats)
        np.save(test_l, test_labels)

    # Map each pool index to its row position in the cached pool array,
    # since SUBSETS holds dataset indices, not row positions
    pos_in_pool = {idx: i for i, idx in enumerate(TRAIN_POOL_IDX)}

    results = {}
    for fraction in SUBSET_FRACTIONS:
        pct = int(fraction * 100)
        rows = [pos_in_pool[i] for i in SUBSETS[fraction]]
        X, y = pool_feats[rows], pool_labels[rows]

        print(f"\nCLIP linear probe @ {pct}% ({len(rows)} images)")
        clf = LogisticRegression(max_iter=1000, C=1.0,
                                  random_state=args.seed)
        clf.fit(X, y)

        preds = clf.predict(test_feats)
        acc = float((preds == test_labels).mean())
        results[fraction] = acc
        print(f"  TEST={acc*100:.2f}%")

        np.save(OUT_ROOT / "clip_linear_probe" / "preds" /
                f"clip_probe_{pct}pct_preds.npy", preds.astype(np.int16))

    np.save(OUT_ROOT / "clip_linear_probe" / "preds" / "clip_probe_labels.npy",
            test_labels.astype(np.int16))

    del model
    torch.cuda.empty_cache()
    return results

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

# In[ ]:

def plot_single_model(model_key, display_name, results):
    """Per-model curve. This is what the original notebook never produced."""
    fracs = sorted(results)
    x = [int(f * 100) for f in fracs]
    y = [results[f] * 100 for f in fracs]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2, markersize=8, color="#4C78A8")
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=9)
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Data Efficiency: {display_name} (Food-101)")
    plt.xticks(x)
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUT_ROOT / model_key / "images" / f"{model_key}_data_efficiency_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  figure -> {out}")

    pd.DataFrame({"Training Data (%)": x, "Accuracy (%)": y}).to_csv(
        OUT_ROOT / model_key / f"{model_key}_data_efficiency_results.csv",
        index=False)

# In[ ]:

def plot_combined(all_results):
    markers = {"AlexNet": "o", "ResNet34": "s", "ResNet50": "^",
               "YOLOv8s-cls": "D", "CLIP Linear Probe": "*"}
    colors = {"AlexNet": "#E45756", "ResNet34": "#F58518",
              "ResNet50": "#4C78A8", "YOLOv8s-cls": "#54A24B",
              "CLIP Linear Probe": "#B279A2"}

    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        fracs = sorted(res)
        plt.plot([int(f * 100) for f in fracs], [res[f] * 100 for f in fracs],
                  marker=markers.get(name, "o"), label=name,
                  color=colors.get(name), linewidth=2, markersize=8)
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Experiment 5: Data-Efficiency Curve (Food-101)")
    plt.xticks([int(f * 100) for f in SUBSET_FRACTIONS])
    plt.ylim(0, 100)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_ROOT / "images" / "experiment5_data_efficiency_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"\nCombined figure -> {out}")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

# In[ ]:

DISPLAY = {"alexnet": "AlexNet", "resnet34": "ResNet34",
           "resnet50": "ResNet50", "yolo": "YOLOv8s-cls",
           "clip": "CLIP Linear Probe"}
OUTKEY = {"alexnet": "alexnet", "resnet34": "resnet34", "resnet50": "resnet50",
          "yolo": "yolo", "clip": "clip_linear_probe"}

# In[ ]:

all_results, val_records = {}, []

# In[ ]:

# Run every requested model across all data fractions, collecting results
# for the combined comparison plot at the end
for key in args.models:
    results = {}
    if key in CNN_SPECS:
        for fraction in SUBSET_FRACTIONS:
            test_acc, best_val = train_cnn(key, fraction)
            results[fraction] = test_acc
            val_records.append({"Model": DISPLAY[key],
                                 "Training Data (%)": int(fraction * 100),
                                 "Val Accuracy (%)": best_val * 100,
                                 "Test Accuracy (%)": test_acc * 100})
    elif key == "yolo":
        for fraction in SUBSET_FRACTIONS:
            test_acc, _ = train_yolo(fraction)
            results[fraction] = test_acc
    elif key == "clip":
        results = run_clip_probe()
    else:
        raise ValueError(f"Unknown model key: {key}")

    all_results[DISPLAY[key]] = results
    plot_single_model(OUTKEY[key], DISPLAY[key], results)

# ---- combined outputs ------------------------------------------------------

# In[ ]:

rows = [{"Model": m, "Training Data (%)": int(f * 100),
         "Accuracy (%)": a * 100}
        for m, res in all_results.items() for f, a in res.items()]
df = pd.DataFrame(rows)
df.to_csv(OUT_ROOT / "experiment5_data_efficiency_results.csv", index=False)
print("\n" + df.pivot(index="Training Data (%)", columns="Model",
                       values="Accuracy (%)").round(2).to_string())

# In[ ]:

if val_records:
    pd.DataFrame(val_records).to_csv(
        OUT_ROOT / "experiment5_val_vs_test.csv", index=False)

# In[ ]:

plot_combined(all_results)

# ---- plateau analysis ------------------------------------------------------

# In[ ]:

# For each model, find the smallest data fraction whose accuracy comes within
# TOLERANCE_PP of the final (100%) accuracy — i.e. where extra data stops helping
TOLERANCE_PP = 1.5
plateau = []
for name, res in all_results.items():
    fracs = sorted(res)
    accs = [res[f] * 100 for f in fracs]
    final = accs[-1]
    hit = next((int(f * 100) for f, a in zip(fracs, accs)
                if final - a <= TOLERANCE_PP), None)
    plateau.append({
        "Model": name,
        "Final Accuracy (100% data)": round(final, 2),
        "Plateau Reached At (%)": hit,
        "Accuracy At Plateau (%)": round(res[hit / 100], 4) * 100 if hit else None,
    })

# In[ ]:

plateau_df = pd.DataFrame(plateau).sort_values("Plateau Reached At (%)")
print("\n" + plateau_df.to_string(index=False))
plateau_df.to_csv(OUT_ROOT / "experiment5_plateau_summary.csv", index=False)

# In[ ]:

print("\nExperiment 5 complete.")
