#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
print("CUDA available:", torch.cuda.is_available())   # Must print True
print("GPU:", torch.cuda.get_device_name(0))          # Should print RTX 4050


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import Food101
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import random
from dotenv import load_dotenv
from huggingface_hub import login


# In[5]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[7]:


load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=hf_token)


# In[14]:


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
data_root = os.path.join(script_dir, "..", "data")

exp4_dir = os.path.join(script_dir, "..", "outputs", "experiment_4")
exp4_img_dir = os.path.join(exp4_dir, "images")
exp4_ckpt_dir = os.path.join(exp4_dir, "checkpoints")
os.makedirs(exp4_img_dir, exist_ok=True)
os.makedirs(exp4_ckpt_dir, exist_ok=True)

best_model_path = os.path.join(exp4_ckpt_dir, "clip_finetuned_best.pt")
last_model_path = os.path.join(exp4_ckpt_dir, "clip_finetuned_last.pt")

finetune_report_path = os.path.join(exp4_dir, "finetune_classification_report.txt")
finetune_history_csv = os.path.join(exp4_dir, "finetune_training_history.csv")
finetune_comparison_csv = os.path.join(exp4_dir, "finetune_vs_zeroshot_vs_linearprobe.csv")

finetune_loss_curve_png = os.path.join(exp4_img_dir, "finetune_loss_curve.png")
finetune_acc_curve_png = os.path.join(exp4_img_dir, "finetune_accuracy_curve.png")
finetune_comparison_png = os.path.join(exp4_img_dir, "finetune_vs_zeroshot_vs_linearprobe.png")
finetune_confmat_png = os.path.join(exp4_img_dir, "finetune_confusion_matrix.png")


# In[15]:


CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",
    "batch_size": 32,
    "num_epochs": 12,          # within the 10-15 range specified
    "learning_rate": 1e-5,     # small LR to avoid catastrophic forgetting
    "weight_decay": 0.01,
    "unfreeze_text_encoder": True,   # set False to fine-tune image encoder only
    "warmup_frac": 0.05,
    "max_grad_norm": 1.0,
    "val_split_frac": 0.10,    # held-out slice of train for validation/early stopping
    "num_workers": 4,
    "use_amp": True,           # mixed precision for speed/memory
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Config:", CONFIG)


# In[16]:


clip_model = CLIPModel.from_pretrained(CONFIG["model_name"], use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(CONFIG["model_name"])


# In[17]:


# Always unfreeze the vision tower (image encoder) — this is the core requirement.
for p in clip_model.vision_model.parameters():
    p.requires_grad = True
for p in clip_model.visual_projection.parameters():
    p.requires_grad = True

if CONFIG["unfreeze_text_encoder"]:
    for p in clip_model.text_model.parameters():
        p.requires_grad = True
    for p in clip_model.text_projection.parameters():
        p.requires_grad = True
else:
    for p in clip_model.text_model.parameters():
        p.requires_grad = False
    for p in clip_model.text_projection.parameters():
        p.requires_grad = False

clip_model.logit_scale.requires_grad = True

trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in clip_model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.1f}%)")


# In[18]:


full_train_dataset = Food101(root=data_root, split="train", download=True, transform=None)
test_dataset = Food101(root=data_root, split="test", download=True, transform=None)

class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"Train size (full): {len(full_train_dataset)}  Test size: {len(test_dataset)}  Classes: {num_classes}")


# In[19]:


val_size = int(len(full_train_dataset) * CONFIG["val_split_frac"])
train_size = len(full_train_dataset) - val_size

train_subset, val_subset = torch.utils.data.random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
print(f"Train (fine-tune) size: {len(train_subset)}  Val size: {len(val_subset)}")


# In[20]:


def format_class_name(name):
    return name.replace("_", " ")

def build_prompts(names):
    return [f"a photo of {format_class_name(c)}, a type of food" for c in names]

class_prompts = build_prompts(class_names)


# In[21]:


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(
    train_subset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=CONFIG["num_workers"], collate_fn=collate_fn, drop_last=True
)
val_loader = DataLoader(
    val_subset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"], collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"], collate_fn=collate_fn
)

# --- Cell 12: Pre-tokenize class prompts (text side, re-encoded every step since
# the text encoder may be trainable and its output changes as weights update) ---
text_inputs = clip_processor(text=class_prompts, return_tensors="pt", padding=True).to(device)

def get_text_features_live():
    """Encode the fixed class prompts through the (possibly trainable) text encoder."""
    text_output = clip_model.get_text_features(**text_inputs)
    if hasattr(text_output, "pooler_output"):
        text_features = text_output.pooler_output
    elif hasattr(text_output, "text_embeds"):
        text_features = text_output.text_embeds
    else:
        text_features = text_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_image_features_live(images):
    """Encode a batch of PIL images through the (trainable) vision encoder."""
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    image_output = clip_model.get_image_features(**inputs)
    if hasattr(image_output, "pooler_output"):
        image_features = image_output.pooler_output
    elif hasattr(image_output, "image_embeds"):
        image_features = image_output.image_embeds
    else:
        image_features = image_output
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


# In[24]:


optimizer = AdamW(
    filter(lambda p: p.requires_grad, clip_model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

num_training_steps = CONFIG["num_epochs"] * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_amp"])


# In[25]:


@torch.no_grad()
def evaluate_model(loader, desc="Evaluating"):
    clip_model.eval()
    all_preds = []
    all_labels = []
    all_top5 = []

    text_features = get_text_features_live()  # fixed within this eval pass

    for images, labels in tqdm(loader, desc=desc, leave=False):
        image_features = get_image_features_live(images)
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

        top5 = torch.topk(probs, k=5, dim=-1).indices.cpu().numpy()
        top1 = top5[:, 0]

        all_preds.extend(top1.tolist())
        all_top5.extend(top5.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_top5 = np.array(all_top5)

    top1_acc = (all_preds == all_labels).mean() * 100
    top5_acc = np.mean([label in row for label, row in zip(all_labels, all_top5)]) * 100

    return top1_acc, top5_acc, all_preds, all_labels, all_top5


# In[26]:


history = []
best_val_acc = -1.0

for epoch in range(1, CONFIG["num_epochs"] + 1):
    clip_model.train()
    epoch_loss = 0.0
    num_batches = 0
    start_time = time.time()

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
    for images, labels in progress:
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CONFIG["use_amp"]):
            image_features = get_image_features_live(images)
            text_features = get_text_features_live()

            logit_scale = clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T

            # Image-to-text classification loss (cross-entropy over 101 classes)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, clip_model.parameters()),
            CONFIG["max_grad_norm"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / num_batches
    epoch_time = time.time() - start_time

    val_top1, val_top5, _, _, _ = evaluate_model(val_loader, desc=f"Val (epoch {epoch})")

    history.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_top1_acc": val_top1,
        "val_top5_acc": val_top5,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_sec": epoch_time,
    })

    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
          f"val_top1={val_top1:.2f}%  val_top5={val_top5:.2f}%  "
          f"time={epoch_time:.1f}s")

    # Save last checkpoint every epoch
    torch.save(clip_model.state_dict(), last_model_path)

    # Save best checkpoint on val top-1 accuracy
    if val_top1 > best_val_acc:
        best_val_acc = val_top1
        torch.save(clip_model.state_dict(), best_model_path)
        print(f"  -> New best val accuracy: {best_val_acc:.2f}% (checkpoint saved)")


# In[ ]:


history_df = pd.DataFrame(history)
history_df.to_csv(finetune_history_csv, index=False)
print(history_df)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", color="tomato")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Fine-Tuning Loss Curve")

axes[1].plot(history_df["epoch"], history_df["val_top1_acc"], marker="o", label="Val Top-1", color="steelblue")
axes[1].plot(history_df["epoch"], history_df["val_top5_acc"], marker="o", label="Val Top-5", color="seagreen")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(finetune_loss_curve_png, dpi=200)
plt.show()


# In[ ]:


clip_model.load_state_dict(torch.load(best_model_path, map_location=device))
clip_model.to(device)

test_top1, test_top5, test_preds, test_labels, test_top5_preds = evaluate_model(
    test_loader, desc="Final Test Evaluation"
)
print(f"\nFine-Tuned CLIP — Test Top-1: {test_top1:.2f}%  Test Top-5: {test_top5:.2f}%")


# In[ ]:


report = classification_report(test_labels, test_preds, target_names=class_names)
print(report)

with open(finetune_report_path, "w") as f:
    f.write(f"Fine-Tuned CLIP Test Top-1 Accuracy: {test_top1:.2f}%\n")
    f.write(f"Fine-Tuned CLIP Test Top-5 Accuracy: {test_top5:.2f}%\n\n")
    f.write(report)


# In[ ]:


import seaborn as sns

cm = confusion_matrix(test_labels, test_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(20, 18))
sns.heatmap(cm_normalized, cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.title("Fully Fine-Tuned CLIP — Confusion Matrix (Food-101)", fontsize=16)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(finetune_confmat_png, dpi=200)
plt.show()


# In[ ]:


# Update these two values from your actual saved results if they differ.
zero_shot_top1_acc = 84.19     # from Experiment 1
linear_probe_top1_acc = None   # e.g. load from outputs/experiment_3/linear_probe_vs_zeroshot.csv

exp3_csv_path = os.path.join(script_dir, "..", "outputs", "experiment_3", "linear_probe_vs_zeroshot.csv")
if os.path.exists(exp3_csv_path):
    exp3_df = pd.read_csv(exp3_csv_path)
    lp_row = exp3_df[exp3_df["method"].str.contains("Linear Probe", case=False)]
    if not lp_row.empty:
        linear_probe_top1_acc = float(lp_row.iloc[0]["top1_acc"])

if linear_probe_top1_acc is None:
    linear_probe_top1_acc = 0.0  # placeholder if Experiment 3 CSV not found
    print("Warning: could not load Experiment 3 linear probe accuracy; using placeholder 0.0")

comparison_df = pd.DataFrame([
    {"method": "Zero-Shot CLIP (Experiment 1)", "top1_acc": zero_shot_top1_acc},
    {"method": "Linear Probe (Experiment 3)", "top1_acc": linear_probe_top1_acc},
    {"method": "Full Fine-Tune (Experiment 4)", "top1_acc": test_top1},
])
comparison_df["improvement_vs_zeroshot_pts"] = comparison_df["top1_acc"] - zero_shot_top1_acc
comparison_df.to_csv(finetune_comparison_csv, index=False)
print(comparison_df)


# In[ ]:


plt.figure(figsize=(7, 5))
colors = ["steelblue", "seagreen", "darkorange"]
plt.bar(comparison_df["method"], comparison_df["top1_acc"], color=colors)
plt.ylabel("Top-1 Accuracy (%)")
plt.title("CLIP on Food-101: Zero-Shot vs Linear Probe vs Full Fine-Tune")
plt.xticks(rotation=15, ha="right")
for i, v in enumerate(comparison_df["top1_acc"]):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(finetune_comparison_png, dpi=200)
plt.show()

print("\nExperiment 4 complete.")
print(f"Zero-shot: {zero_shot_top1_acc:.2f}%  |  Linear probe: {linear_probe_top1_acc:.2f}%  |  "
      f"Full fine-tune: {test_top1:.2f}%")

