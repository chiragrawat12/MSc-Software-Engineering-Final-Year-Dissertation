#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))


# In[ ]:


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
from dotenv import load_dotenv
from huggingface_hub import login


# In[ ]:


load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=hf_token)


# In[ ]:


script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, "..", "data")

exp3_dir = os.path.join(script_dir, "..", "outputs", "experiment_3")
exp3_img_dir = os.path.join(exp3_dir, "images")
os.makedirs(exp3_img_dir, exist_ok=True)

clip_train_features_path = os.path.join(exp3_dir, "clip_train_features.npy")
clip_train_labels_path   = os.path.join(exp3_dir, "clip_train_labels.npy")
clip_test_features_path  = os.path.join(exp3_dir, "clip_test_features.npy")
clip_test_labels_path    = os.path.join(exp3_dir, "clip_test_labels.npy")

linear_probe_report_path = os.path.join(exp3_dir, "linear_probe_classification_report.txt")
linear_probe_results_csv = os.path.join(exp3_dir, "linear_probe_vs_zeroshot.csv")
linear_probe_comparison_png = os.path.join(exp3_img_dir, "linear_probe_vs_zeroshot.png")
linear_probe_confmat_png = os.path.join(exp3_img_dir, "linear_probe_confusion_matrix.png")


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False   # freeze CLIP image encoder entirely


# In[ ]:


train_dataset_raw = Food101(root=data_root, split="train", download=True, transform=None)
test_dataset_raw  = Food101(root=data_root, split="test",  download=True, transform=None)

class_names = train_dataset_raw.classes
print(f"Train size: {len(train_dataset_raw)}  Test size: {len(test_dataset_raw)}  Classes: {len(class_names)}")


# In[ ]:


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

train_loader = DataLoader(
    train_dataset_raw, batch_size=64, shuffle=False,
    num_workers=4, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset_raw, batch_size=64, shuffle=False,
    num_workers=4, collate_fn=collate_fn
)


# In[ ]:


def extract_features(loader, desc):
    """Run frozen CLIP image encoder over a dataloader, return L2-normalized features + labels."""
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            image_output = clip_model.get_image_features(**inputs)

            if hasattr(image_output, "pooler_output"):
                feats = image_output.pooler_output
            elif hasattr(image_output, "image_embeds"):
                feats = image_output.image_embeds
            else:
                feats = image_output

            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.extend(labels)

    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = np.array(all_labels)
    return all_feats, all_labels


# In[ ]:


if os.path.exists(clip_train_features_path) and os.path.exists(clip_train_labels_path):
    train_features = np.load(clip_train_features_path)
    train_labels = np.load(clip_train_labels_path)
    print("Loaded cached train features:", train_features.shape)
else:
    train_features, train_labels = extract_features(train_loader, "Encoding train images")
    np.save(clip_train_features_path, train_features)
    np.save(clip_train_labels_path, train_labels)
    print("Train features shape:", train_features.shape)


# In[ ]:


exp2_test_features_path = os.path.join(script_dir, "..", "outputs", "experiment_2", "clip_image_features.npy")
exp2_test_labels_path   = os.path.join(script_dir, "..", "outputs", "experiment_2", "clip_test_labels.npy")

if os.path.exists(clip_test_features_path) and os.path.exists(clip_test_labels_path):
    test_features = np.load(clip_test_features_path)
    test_labels = np.load(clip_test_labels_path)
    print("Loaded cached test features:", test_features.shape)
elif os.path.exists(exp2_test_features_path) and os.path.exists(exp2_test_labels_path):
    # Reuse features already extracted in Experiment 2 to avoid recomputation
    test_features = np.load(exp2_test_features_path)
    test_labels = np.load(exp2_test_labels_path)
    np.save(clip_test_features_path, test_features)
    np.save(clip_test_labels_path, test_labels)
    print("Reused Experiment 2 test features:", test_features.shape)
else:
    test_features, test_labels = extract_features(test_loader, "Encoding test images")
    np.save(clip_test_features_path, test_features)
    np.save(clip_test_labels_path, test_labels)
    print("Test features shape:", test_features.shape)


# In[ ]:


print("Training linear probe (logistic regression) on frozen CLIP features...")
start = time.time()

classifier = LogisticRegression(
    max_iter=1000,
    C=1.0,
    multi_class="multinomial",
    n_jobs=-1,
    random_state=42
)
classifier.fit(train_features, train_labels)

train_time = time.time() - start
print(f"Training completed in {train_time:.1f}s")


# In[ ]:


test_preds = classifier.predict(test_features)
linear_probe_acc = accuracy_score(test_labels, test_preds) * 100
print(f"Linear Probe Top-1 Accuracy: {linear_probe_acc:.2f}%")

report = classification_report(test_labels, test_preds, target_names=class_names)
print(report)

with open(linear_probe_report_path, "w") as f:
    f.write(f"Linear Probe Top-1 Accuracy: {linear_probe_acc:.2f}%\n\n")
    f.write(report)


# In[ ]:


# Update this value if your Experiment 1 baseline differs
zero_shot_top1_acc = 84.19  # from Experiment 1 zero-shot CLIP result

comparison_df = pd.DataFrame([
    {"method": "Zero-Shot CLIP (Experiment 1)", "top1_acc": zero_shot_top1_acc},
    {"method": "Linear Probe on Frozen CLIP (Experiment 3)", "top1_acc": linear_probe_acc},
])
comparison_df["improvement_pts"] = comparison_df["top1_acc"] - zero_shot_top1_acc
comparison_df.to_csv(linear_probe_results_csv, index=False)
print(comparison_df)


# In[ ]:


plt.figure(figsize=(6, 5))
plt.bar(comparison_df["method"], comparison_df["top1_acc"], color=["steelblue", "seagreen"])
plt.ylabel("Top-1 Accuracy (%)")
plt.title("Zero-Shot vs Linear Probe (CLIP ViT-B/32, Food-101)")
plt.xticks(rotation=15, ha="right")
for i, v in enumerate(comparison_df["top1_acc"]):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(linear_probe_comparison_png, dpi=200)
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_labels, test_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(20, 18))
sns.heatmap(cm_normalized, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Linear Probe Confusion Matrix (Food-101)", fontsize=16)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(linear_probe_confmat_png, dpi=200)
plt.show()

print("\nExperiment 3 complete.")
print(f"Zero-shot: {zero_shot_top1_acc:.2f}%  ->  Linear probe: {linear_probe_acc:.2f}%  ({linear_probe_acc - zero_shot_top1_acc:+.2f} pts)")

