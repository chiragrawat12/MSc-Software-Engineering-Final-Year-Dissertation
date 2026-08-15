#!/usr/bin/env python
# coding: utf-8

# # Experiment 8: Combined CNN Hyperparameter Grid Search
# 
# **Objective:** Find the optimal training configuration for the best-performing CNN
# architecture identified in Experiment 5.
# 
# **Grid:**
# - Batch size: {16, 32, 64}
# - Optimizer: {SGD, Adam, AdamW}
# - LR scheduler: {fixed, ReduceLROnPlateau, cosine}
# - Total configurations: 3 x 3 x 3 = 27
# 
# **Design notes / assumptions made explicit for the methodology section**
# 
# - **Model choice -- ResNet50, not YOLOv8:** YOLOv8 uses Ultralytics' own trainer, which does
#   not natively support `ReduceLROnPlateau` and exposes a different optimizer API. ResNet50 uses
#   a standard PyTorch training loop, so the same 27 configurations can be applied identically and
#   fairly across the whole grid. This keeps the search methodologically clean.
# - **Reduced epoch budget:** 18 epochs with patience 3 (vs. the 60/5 budget used for the main
#   CNN training runs elsewhere in the dissertation). This is standard practice for a
#   hyperparameter sweep -- 27 full-length training runs would be prohibitively expensive, and a
#   relative comparison between configurations does not require each one to reach full
#   convergence.
# - **Full training pool:** all 68,175 training images are used for every run, with the same
#   90/10 stratified train/validation split as Experiments 5 and 6 (`split_seed=1234`), so results
#   remain directly comparable across the dissertation.
# - **Optimizer hyperparameters (fixed per optimizer, not swept):**
#   - SGD: `lr=0.01, momentum=0.9, weight_decay=5e-4` -- matches the baseline CNN training
#     configuration used in Experiment 5.
#   - Adam: `lr=1e-3, weight_decay=5e-4` -- standard default learning rate for Adam.
#   - AdamW: `lr=1e-3, weight_decay=1e-2` -- PyTorch's default weight decay for AdamW, which is
#     designed for larger, decoupled weight decay values.
# - **`ReduceLROnPlateau` monitor metric:** validation accuracy, `mode="max"`, `factor=0.5`,
#   `patience=1`. This was not specified in the brief; validation accuracy was chosen because it
#   is the primary metric tracked throughout every other experiment in this dissertation.
# - **Cosine schedule:** `CosineAnnealingLR` with `T_max=18` (one full cycle per run), matching
#   the scheduler used for CLIP fine-tuning in Experiment 4.
# - **Reporting:** per the brief, individual configurations are not narrated one-by-one. All 27
#   runs are logged to a single results table, aggregated into one summary table (mean +/- std
#   grouped by each hyperparameter), and the single best configuration is highlighted and
#   evaluated in full (classification report + confusion matrix).

# In[ ]:


import os
import json
import copy
import time
import random
import socket
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless -- must precede the pyplot import
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


print("Hostname:", socket.gethostname())
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Run inside a Slurm GPU allocation.")
print("GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# In[ ]:


SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


# ### Paths

# In[ ]:


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

EXP = "experiment_8"
OUT = PROJECT_ROOT / "outputs" / EXP
MODELS_ROOT = PROJECT_ROOT / "models" / EXP

for d in [OUT / "images", OUT / "preds", OUT / "histories", MODELS_ROOT / "checkpoints"]:
    d.mkdir(parents=True, exist_ok=True)

results_csv = OUT / "grid_results.csv"
summary_csv = OUT / "grid_summary.csv"
best_config_json = OUT / "best_config.json"
findings_json = OUT / "experiment_8_findings.json"

print("Project root:", PROJECT_ROOT)
print("Outputs:", OUT)


# ### Data -- same stratified split as Experiments 5 and 6

# In[ ]:


train_raw = Food101(root=str(DATA_ROOT), split="train", download=True, transform=None)
test_raw = Food101(root=str(DATA_ROOT), split="test", download=True, transform=None)

CLASS_NAMES = train_raw.classes
NUM_CLASSES = len(CLASS_NAMES)
TRAIN_LABELS = np.array(train_raw._labels)

print(f"Train: {len(train_raw)}  Test: {len(test_raw)}  Classes: {NUM_CLASSES}")

VAL_FRACTION = 0.10
SPLIT_SEED = 1234  # must match Experiments 5 and 6

def stratified_train_val_split(labels, val_fraction, seed):
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[lab].append(idx)

    rng = np.random.default_rng(seed)
    train_pool, val_idx = [], []
    for lab, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        val_idx.extend(idxs[:n_val].tolist())
        train_pool.extend(idxs[n_val:].tolist())
    return sorted(train_pool), sorted(val_idx)

TRAIN_POOL_IDX, VAL_IDX = stratified_train_val_split(TRAIN_LABELS, VAL_FRACTION, SPLIT_SEED)
assert len(set(TRAIN_POOL_IDX) & set(VAL_IDX)) == 0, "train/val overlap"
print(f"Train pool: {len(TRAIN_POOL_IDX)}  Validation: {len(VAL_IDX)}")


# ### Transforms

# In[ ]:


IMG_SIZE = 224  # standard ResNet50 input size

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def train_transform(size):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        NORM,
    ])

def eval_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORM,
    ])


# ### Model

# In[ ]:


def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, NUM_CLASSES)
    return m


# ### Grid configuration and optimizer/scheduler builders

# In[ ]:


BATCH_SIZES = [16, 32, 64]
OPTIMIZER_NAMES = ["sgd", "adam", "adamw"]
SCHEDULER_NAMES = ["fixed", "plateau", "cosine"]
EPOCHS = 18
PATIENCE = 3
NUM_WORKERS = 8

GRID = list(itertools.product(BATCH_SIZES, OPTIMIZER_NAMES, SCHEDULER_NAMES))
print(f"Total configurations: {len(GRID)}")


def build_optimizer(name, params):
    if name == "sgd":
        return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif name == "adam":
        return optim.Adam(params, lr=1e-3, weight_decay=5e-4)
    elif name == "adamw":
        return optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    if name == "fixed":
        return None
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler: {name}")


# In[ ]:


def build_loaders(batch_size):
    train_ds = Food101(root=str(DATA_ROOT), split="train", download=False, transform=train_transform(IMG_SIZE))
    eval_ds = Food101(root=str(DATA_ROOT), split="train", download=False, transform=eval_transform(IMG_SIZE))
    test_ds = Food101(root=str(DATA_ROOT), split="test", download=False, transform=eval_transform(IMG_SIZE))

    train_loader = DataLoader(
        Subset(train_ds, TRAIN_POOL_IDX),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        Subset(eval_ds, VAL_IDX),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# In[ ]:


@torch.no_grad()
def evaluate(model, loader, collect=False):
    model.eval()
    correct = total = 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
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
        return acc, np.array(preds_all, dtype=np.int16), np.array(labels_all, dtype=np.int16)
    return acc


# ### Single-configuration training run

# In[ ]:


def run_single_config(batch_size, opt_name, sched_name, run_idx, total_runs):
    tag = f"bs{batch_size}_{opt_name}_{sched_name}"
    print(f"[{run_idx}/{total_runs}] Starting {tag}", flush=True)
    start_time = time.time()

    train_loader, val_loader, test_loader = build_loaders(batch_size)

    model = build_resnet50().to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(opt_name, model.parameters())
    scheduler = build_scheduler(sched_name, optimizer, EPOCHS)

    best_val_acc, no_improve, best_state = 0.0, 0, None
    history = []
    epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        val_acc = evaluate(model, val_loader)

        if sched_name == "plateau":
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "lr": current_lr})

        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test_acc, test_preds, test_labels = evaluate(model, test_loader, collect=True)
    elapsed_min = (time.time() - start_time) / 60

    torch.save(best_state, MODELS_ROOT / "checkpoints" / f"{tag}.pth")
    np.save(OUT / "preds" / f"{tag}_preds.npy", test_preds)
    np.save(OUT / "preds" / f"{tag}_labels.npy", test_labels)
    pd.DataFrame(history).to_csv(OUT / "histories" / f"{tag}_history.csv", index=False)

    del model
    torch.cuda.empty_cache()

    result = {
        "batch_size": batch_size,
        "optimizer": opt_name,
        "scheduler": sched_name,
        "best_val_acc": best_val_acc * 100,
        "test_acc": test_acc * 100,
        "epochs_run": epoch,
        "train_time_min": elapsed_min,
        "tag": tag,
    }
    print(f"[{run_idx}/{total_runs}] Done {tag}: val={best_val_acc*100:.2f}%  "
          f"test={test_acc*100:.2f}%  ({elapsed_min:.1f} min)", flush=True)
    return result


# ### Run the full 3 x 3 x 3 grid
# 
# Results are saved to `grid_results.csv` after every single run (not just at the end), so an
# interrupted job never loses completed configurations.

# In[ ]:


results = []

for i, (bs, opt_name, sched_name) in enumerate(GRID, start=1):
    result = run_single_config(bs, opt_name, sched_name, i, len(GRID))
    results.append(result)
    pd.DataFrame(results).to_csv(results_csv, index=False)  # incremental save

results_df = pd.DataFrame(results)
print(results_df.round(2).to_string(index=False))


# ### Summary table -- mean and standard deviation across runs, grouped by each hyperparameter

# In[ ]:


def summarize_factor(df, factor):
    g = df.groupby(factor)["test_acc"].agg(["mean", "std", "count"]).reset_index()
    g.insert(0, "Factor", factor)
    g = g.rename(columns={
        factor: "Level",
        "mean": "Mean Test Acc (%)",
        "std": "Std Dev (%)",
        "count": "N Runs",
    })
    return g[["Factor", "Level", "Mean Test Acc (%)", "Std Dev (%)", "N Runs"]]

summary_df = pd.concat([
    summarize_factor(results_df, "batch_size"),
    summarize_factor(results_df, "optimizer"),
    summarize_factor(results_df, "scheduler"),
], ignore_index=True)

summary_df.to_csv(summary_csv, index=False)
print(summary_df.round(2).to_string(index=False))
print(f"\nSummary table saved -> {summary_csv}")


# ### Best-performing configuration

# In[ ]:


best_row = results_df.loc[results_df["test_acc"].idxmax()]

print("=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)
print(f"Batch size : {best_row['batch_size']}")
print(f"Optimizer  : {best_row['optimizer']}")
print(f"Scheduler  : {best_row['scheduler']}")
print(f"Val Acc    : {best_row['best_val_acc']:.2f}%")
print(f"Test Acc   : {best_row['test_acc']:.2f}%")
print(f"Epochs run : {int(best_row['epochs_run'])}")

with open(best_config_json, "w") as f:
    json.dump(best_row.to_dict(), f, indent=2)
print(f"\nBest config saved -> {best_config_json}")


# ### Figures

# In[ ]:


sorted_df = results_df.sort_values("test_acc", ascending=False).reset_index(drop=True)
colors = ["darkorange" if tag == best_row["tag"] else "steelblue" for tag in sorted_df["tag"]]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(sorted_df["tag"], sorted_df["test_acc"], color=colors)
ax.set_xticklabels(sorted_df["tag"], rotation=90, fontsize=7)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Experiment 8: ResNet50 Hyperparameter Grid Search (27 Configurations)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "images" / "grid_search_results.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, factor in zip(axes, ["batch_size", "optimizer", "scheduler"]):
    sub = summary_df[summary_df["Factor"] == factor]
    ax.bar(sub["Level"].astype(str), sub["Mean Test Acc (%)"],
           yerr=sub["Std Dev (%)"], capsize=5, color="seagreen")
    ax.set_title(f"By {factor}")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "images" / "grid_summary_by_factor.png", dpi=200)
plt.close()

print("Figures saved.")


# ### Full evaluation of the best configuration

# In[ ]:


best_tag = best_row["tag"]
best_preds = np.load(OUT / "preds" / f"{best_tag}_preds.npy")
best_labels = np.load(OUT / "preds" / f"{best_tag}_labels.npy")

report = classification_report(best_labels, best_preds, target_names=CLASS_NAMES)
with open(OUT / "best_config_classification_report.txt", "w") as f:
    f.write(f"Best configuration: {best_tag}\n")
    f.write(f"Test Accuracy: {best_row['test_acc']:.2f}%\n\n")
    f.write(report)
print(report)

cm = confusion_matrix(best_labels, best_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(20, 18))
sns.heatmap(cm_normalized, cmap="Purples", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix -- Best Config ({best_tag})", fontsize=16)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(OUT / "images" / "best_config_confusion_matrix.png", dpi=200)
plt.close()

print("Confusion matrix saved.")


# In[ ]:


findings = {
    "grid_size": len(GRID),
    "epochs_per_run": EPOCHS,
    "patience": PATIENCE,
    "batch_sizes": BATCH_SIZES,
    "optimizers": OPTIMIZER_NAMES,
    "schedulers": SCHEDULER_NAMES,
    "best_config": best_row.to_dict(),
    "summary": summary_df.to_dict(orient="records"),
}
with open(findings_json, "w") as f:
    json.dump(findings, f, indent=2)

print("=" * 60)
print("EXPERIMENT 8 COMPLETE")
print("=" * 60)
print(f"Best configuration: {best_tag} -> Test Accuracy {best_row['test_acc']:.2f}%")
print(f"Findings saved -> {findings_json}")

