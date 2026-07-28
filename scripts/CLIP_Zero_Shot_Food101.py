#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))


# In[4]:


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from huggingface_hub import login


# In[5]:


load_dotenv()  # reads .env from the current working directory

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=hf_token)


# In[ ]:


script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, '..', 'data')
clip_zeroshot_classification_report = os.path.join(script_dir, '..', 'outputs/clipzeroshot/clip_zeroshot_classification_report.txt')
clip_zeroshot_preds = os.path.join(script_dir, '..', 'outputs/clipzeroshot/clip_zeroshot_preds.npy')
clip_zeroshot_labels = os.path.join(script_dir, '..', 'outputs/clipzeroshot/clip_zeroshot_labels.npy')
clip_zeroshot_top5preds = os.path.join(script_dir, '..', 'outputs/clipzeroshot/clip_zeroshot_top5preds.npy')
clip_confusion_matrix_heatmap = os.path.join(script_dir, '..', 'outputs/clipzeroshot/images/clip_confusion_matrix_heatmap.png')
clip_top10_confused_pairs = os.path.join(script_dir, '..', 'outputs/clipzeroshot/images/clip_top10_confused_pairs.png')
clip_per_class_accuracy = os.path.join(script_dir, '..', 'outputs/clipzeroshot/images/clip_per_class_accuracy.png')
clip_sample_misclassifications = os.path.join(script_dir, '..', 'outputs/clipzeroshot/images/clip_sample_misclassifications.png')


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

clip_model.eval()  # inference only, no training


# In[6]:


test_dataset_raw = Food101(root=data_root, split='test', download=True, transform=None)

print(f"Test size: {len(test_dataset_raw)}")
print(f"Classes: {len(test_dataset_raw.classes)}")

class_names = test_dataset_raw.classes
print(class_names[:5])


# In[7]:


def format_class_name(name):
    return name.replace("_", " ")

text_prompts = [f"a photo of {format_class_name(c)}, a type of food" for c in class_names]

text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_output = clip_model.get_text_features(**text_inputs)

    if hasattr(text_output, "pooler_output"):
        text_features = text_output.pooler_output
    elif hasattr(text_output, "text_embeds"):
        text_features = text_output.text_embeds
    else:
        text_features = text_output

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("Text features shape:", text_features.shape)  # should print [101, 512]


# In[8]:


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

test_loader = DataLoader(
    test_dataset_raw,
    batch_size=64,
    shuffle=False,
    num_workers=4,    
    collate_fn=collate_fn
)


# In[9]:


all_preds = []
all_top5_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Zero-shot CLIP inference"):
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)

        image_output = clip_model.get_image_features(**image_inputs)

        if hasattr(image_output, "pooler_output"):
            image_features = image_output.pooler_output
        elif hasattr(image_output, "image_embeds"):
            image_features = image_output.image_embeds
        else:
            image_features = image_output

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T) * clip_model.logit_scale.exp()
        probs = similarity.softmax(dim=-1)

        top5 = torch.topk(probs, k=5, dim=-1).indices.cpu().numpy()
        top1 = top5[:, 0]

        all_preds.extend(top1.tolist())
        all_top5_preds.extend(top5.tolist())
        all_labels.extend(labels)


# In[10]:


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_top5_preds = np.array(all_top5_preds)

top1_acc = (all_preds == all_labels).mean() * 100
top5_acc = np.mean([label in top5row for label, top5row in zip(all_labels, all_top5_preds)]) * 100

print(f"\nZero-Shot CLIP Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Zero-Shot CLIP Top-5 Accuracy: {top5_acc:.2f}%")


# In[11]:


report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

with open(clip_zeroshot_classification_report, "w") as f:
    f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n\n")
    f.write(report)


# In[12]:


cm = confusion_matrix(all_labels, all_preds)

cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
np.fill_diagonal(cm_normalized, 0)

confused_pairs = []
for i in range(len(class_names)):
    j = np.argmax(cm_normalized[i])
    confused_pairs.append((class_names[i], class_names[j], cm_normalized[i, j]))

confused_pairs.sort(key=lambda x: -x[2])
print("Top 10 most confused class pairs:")
for true_c, pred_c, rate in confused_pairs[:10]:
    print(f"{true_c} -> {pred_c}: {rate*100:.1f}%")


# In[13]:


np.save(clip_zeroshot_preds, all_preds)
np.save(clip_zeroshot_labels, all_labels)
np.save(clip_zeroshot_top5preds, all_top5_preds)

results_summary = {
    "model": "CLIP ViT-B/32 (zero-shot)",
    "top1_accuracy": top1_acc,
    "top5_accuracy": top5_acc,
}
print(results_summary)


# In[14]:


import seaborn as sns

plt.figure(figsize=(20, 18))
sns.heatmap(cm_normalized, cmap="Reds", xticklabels=class_names, yticklabels=class_names)
plt.title("CLIP Zero-Shot Confusion Matrix (Food-101)", fontsize=16)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(clip_confusion_matrix_heatmap, dpi=200)
plt.show()


# In[15]:


pairs_labels = [f"{a} → {b}" for a, b, _ in confused_pairs[:10]]
pairs_values = [rate * 100 for _, _, rate in confused_pairs[:10]]

plt.figure(figsize=(10, 6))
plt.barh(pairs_labels[::-1], pairs_values[::-1], color="tomato")
plt.xlabel("Confusion Rate (%)")
plt.title("Top 10 Most Confused Class Pairs — CLIP Zero-Shot")
plt.tight_layout()
plt.savefig(clip_top10_confused_pairs, dpi=200)
plt.show()


# In[16]:


from sklearn.metrics import accuracy_score

per_class_acc = []
for i, cname in enumerate(class_names):
    mask = all_labels == i
    acc = (all_preds[mask] == all_labels[mask]).mean() * 100
    per_class_acc.append((cname, acc))

per_class_acc.sort(key=lambda x: x[1])

worst_15 = per_class_acc[:15]
best_15 = per_class_acc[-15:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh([x[0] for x in worst_15], [x[1] for x in worst_15], color="crimson")
axes[0].set_title("15 Worst-Performing Classes")
axes[0].set_xlabel("Accuracy (%)")

axes[1].barh([x[0] for x in best_15], [x[1] for x in best_15], color="seagreen")
axes[1].set_title("15 Best-Performing Classes")
axes[1].set_xlabel("Accuracy (%)")

plt.tight_layout()
plt.savefig(clip_per_class_accuracy, dpi=200)
plt.show()


# In[18]:


import random

test_dataset_display = Food101(root=data_root, split='test', download=False, transform=None)

wrong_indices = np.where(all_preds != all_labels)[0]
sample_wrong = random.sample(list(wrong_indices), 8)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, idx in zip(axes.flatten(), sample_wrong):
    img, true_label = test_dataset_display[idx]
    pred_label = all_preds[idx]
    ax.imshow(img)
    ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(clip_sample_misclassifications, dpi=200)
plt.show()

