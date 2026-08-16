#!/usr/bin/env python
"""
Regenerate per-image test-set predictions from already-trained CNN weights.

Your original training scripts printed a classification_report and discarded
the predictions. Experiment 9 (McNemar, Wilcoxon) and the per-experiment error
analysis both need them saved and index-aligned.

INFERENCE ONLY. No optimiser, no backward pass, no weights are modified. Each
model is loaded, run over the Food-101 test set with shuffle=False, and its
predictions written to disk.

Usage:
  python regenerate_predictions.py                     # all models
  python regenerate_predictions.py --models resnet50    # just one
  python regenerate_predictions.py --list                # show discovered paths
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+",
               default=["alexnet", "resnet34", "resnet50", "yolo"])
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--list", action="store_true",
               help="Report which weight files were found, then exit.")
args = p.parse_args()

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Locate weights. Several naming conventions are tried, because the training
# scripts used more than one.
# ---------------------------------------------------------------------------

WEIGHT_CANDIDATES = {
    "alexnet": ["models/alexnet/saved_models/alexnet_food101.pth",
                "models/alexnet/best_models/best_alexnet_model.pth"],
    "resnet34": ["models/resnet34/saved_models/resnet34_food101.pth",
                 "models/resnet34/best_models/best_resnet34_model.pth"],
    "resnet50": ["models/resnet50/saved_models/resnet50_food101.pth",
                 "models/resnet50/best_models/best_resnet50_model.pth"],
    "yolo": ["models/yolo/saved_models/yolov8s_food101.pt",
             "outputs/yolo/runs/classify/yolov8s_food101/weights/best.pt"],
}

def find_weights(key):
    for rel in WEIGHT_CANDIDATES[key]:
        f = PROJECT_ROOT / rel
        if f.exists():
            return f
    # last resort: search the tree
    pattern = "*.pt" if key == "yolo" else "*.pth"
    hits = [f for f in PROJECT_ROOT.rglob(pattern) if key in f.name.lower()]
    return hits[0] if hits else None

print("\nWeight files:")
located = {}
for k in ["alexnet", "resnet34", "resnet50", "yolo"]:
    w = find_weights(k)
    located[k] = w
    size = f"{w.stat().st_size/1e6:.0f} MB" if w else ""
    print(f"  {k:<10s} {'FOUND ' if w else 'MISSING '} "
          f"{w.relative_to(PROJECT_ROOT) if w else ''} {size}")

if args.list:
    raise SystemExit(0)

missing = [k for k in args.models if located.get(k) is None]
if missing:
    print(f"\nNo weights for: {', '.join(missing)}")
    print("These models would need retraining, not just re-evaluation.")

# ---------------------------------------------------------------------------
# Architectures. AlexNet is the custom implementation, matching Chapter 3.
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def eval_transform(size):
    return transforms.Compose([transforms.Resize((size, size)),
                                transforms.ToTensor(), NORM])

# Each entry maps a model key to its constructor and the input size it expects
SPECS = {
    "alexnet": (lambda: AlexNet(101), 227),
    "resnet34": (lambda: _resnet(models.resnet34, 512), 224),
    "resnet50": (lambda: _resnet(models.resnet50, 2048), 224),
}

def _resnet(fn, feat):
    m = fn(weights=None)
    m.fc = nn.Linear(feat, 101)
    return m

# ---------------------------------------------------------------------------
# Evaluation. shuffle=False is essential: index i must refer to the same image
# for every model, or the paired tests in Experiment 9 are invalid.
# ---------------------------------------------------------------------------

def save_outputs(key, preds, labels, top5, logits):
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "preds.npy", preds.astype(np.int16))
    np.save(out / "labels.npy", labels.astype(np.int16))
    np.save(out / "top5.npy", top5.astype(np.int16))
    np.save(out / "logits.npy", logits.astype(np.float16))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=range(101))
    np.save(out / "confusion_matrix.npy", cm.astype(np.int32))
    np.save(out / "per_class_acc.npy",
            (cm.diagonal() / np.maximum(1, cm.sum(axis=1))).astype(np.float32))

    top1 = float((preds == labels).mean() * 100)
    top5a = float(np.mean([l in r for l, r in zip(labels, top5)]) * 100)
    with open(out / "metrics.json", "w") as f:
        json.dump({"model": key, "top1": top1, "top5": top5a,
                   "n": int(len(labels))}, f, indent=2)
    print(f"  top-1 {top1:.2f}% top-5 {top5a:.2f}% -> {out}")
    return top1

@torch.no_grad()
def run_torch_model(key):
    ctor, size = SPECS[key]
    w = located[key]
    print(f"\n{key}: loading {w.relative_to(PROJECT_ROOT)}")

    model = ctor()
    state = torch.load(w, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    ds = Food101(root=str(DATA_ROOT), split="test", download=False,
                 transform=eval_transform(size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    P, L, T5, LG = [], [], [], []
    for x, y in tqdm(loader, desc=f"{key}: test set"):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=(device.type == "cuda"),
                             dtype=torch.bfloat16):
            out = model(x).float()
        t5 = torch.topk(out, k=5, dim=-1).indices.cpu().numpy()
        P.extend(t5[:, 0].tolist())
        T5.extend(t5.tolist())
        L.extend(y.tolist())
        LG.append(out.cpu().numpy())

    return save_outputs(key, np.array(P), np.array(L),
                         np.array(T5), np.concatenate(LG))

@torch.no_grad()
def run_yolo():
    from ultralytics import YOLO
    w = located["yolo"]
    print(f"\nyolo: loading {w.relative_to(PROJECT_ROOT)}")
    model = YOLO(str(w))

    # Iterate the same Food101 test set so ordering matches the other models.
    ds = Food101(root=str(DATA_ROOT), split="test", download=False,
                 transform=None)
    P, L, T5, LG = [], [], [], []
    for i in tqdm(range(len(ds)), desc="yolo: test set"):
        img, y = ds[i]
        r = model.predict(img, imgsz=224, verbose=False, device=0)[0]
        probs = r.probs.data.cpu().numpy()
        t5 = np.argsort(-probs)[:5]
        P.append(int(t5[0])); T5.append(t5.tolist())
        L.append(int(y)); LG.append(probs)

    return save_outputs("yolo", np.array(P), np.array(L),
                         np.array(T5), np.stack(LG))

# ---------------------------------------------------------------------------

results = {}
for key in args.models:
    if located.get(key) is None:
        continue
    try:
        results[key] = run_yolo() if key == "yolo" else run_torch_model(key)
    except Exception as e:
        print(f"  {key} FAILED: {type(e).__name__}: {e}")

# Verify every saved label array agrees, which is what makes the predictions
# pairable in Experiment 9.
print("\n" + "=" * 60)
ref = None
for key in results:
    lab = np.load(OUT_ROOT / key / "labels.npy")
    if ref is None:
        ref, ref_key = lab, key
        print(f"Reference ordering: {key} ({len(lab)} images)")
    elif np.array_equal(lab, ref):
        print(f"  {key}: ordering matches {ref_key}")
    else:
        print(f"  {key}: ORDERING DIFFERS from {ref_key}. Not pairable.")

print("\nAccuracies:")
for k, v in results.items():
    print(f"  {k:<10s} {v:6.2f}%")
print("\nNow re-run check_predictions.py; it should report everything present.")
