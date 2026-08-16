#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
from huggingface_hub import login

# In[5]:

load_dotenv()  # reads .env from the current working directory

# HF_TOKEN is needed to download the CLIP model/weights from Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")
login(token=hf_token)

# In[ ]:

# Paths for cached image features/labels and the prompt engineering results (Experiment 2)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, '..', 'data')
clip_image_features = os.path.join(script_dir, '..', 'outputs/experiment_2/clip_image_features.npy')
clip_test_labels = os.path.join(script_dir, '..', 'outputs/experiment_2/clip_test_labels.npy')
prompt_engineering_results = os.path.join(script_dir, '..', 'outputs/experiment_2/prompt_engineering_results.csv')
prompt_engineering_comparison = os.path.join(script_dir, '..', 'outputs/experiment_2/images/prompt_engineering_comparison.png')

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

def get_text_features(prompts):
    """Encode a list of text prompts and return normalized embeddings on CPU."""
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_output = clip_model.get_text_features(**text_inputs)

    # Different transformers versions return text embeddings under different attribute names
    if hasattr(text_output, "pooler_output"):
        text_features = text_output.pooler_output
    elif hasattr(text_output, "text_embeds"):
        text_features = text_output.text_embeds
    else:
        text_features = text_output

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu()

def evaluate(text_features, image_features, labels):
    """Given text and image embeddings, compute Top-1 and Top-5 accuracy."""
    similarity = image_features @ text_features.T * clip_model.logit_scale.exp().cpu()
    probs = similarity.softmax(dim=-1)
    top5 = torch.topk(probs, k=5, dim=-1).indices.numpy()
    top1 = top5[:, 0]

    top1_acc = (top1 == labels).mean() * 100
    top5_acc = np.mean([label in row for label, row in zip(labels, top5)]) * 100
    return top1_acc, top5_acc, top1, top5

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

# Encode all test images once and cache the embeddings, since only the text side
# changes across the different prompt styles tested below
all_image_features = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Encoding test images"):
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        image_output = clip_model.get_image_features(**image_inputs)

        if hasattr(image_output, "pooler_output"):
            image_features = image_output.pooler_output
        elif hasattr(image_output, "image_embeds"):
            image_features = image_output.image_embeds
        else:
            image_features = image_output

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features.cpu())
        all_labels.extend(labels)

all_image_features = torch.cat(all_image_features, dim=0)  # shape: (25250, 512)
all_labels = np.array(all_labels)

np.save(clip_image_features, all_image_features.numpy())
np.save(clip_test_labels, all_labels)

print("Image features shape:", all_image_features.shape)

# In[ ]:

# Reproduce the Experiment 1 baseline prompt style for comparison against the new prompt templates
baseline_prompts = [f"a photo of {format_class_name(c)}, a type of food" for c in class_names]
baseline_text_features = get_text_features(baseline_prompts)

baseline_top1, baseline_top5, _, _ = evaluate(baseline_text_features, all_image_features, all_labels)
print(f"Baseline (Experiment 1) — Top-1: {baseline_top1:.2f}% Top-5: {baseline_top5:.2f}%")

# In[ ]:

# Candidate prompt templates to compare against the baseline
prompt_templates = {
    "a photo of {class}": lambda c: f"a photo of {c}",
    "a photo of {class}, a type of food": lambda c: f"a photo of {c}, a type of food",
    "a dish of {class}": lambda c: f"a dish of {c}",
    "{class}, a food dish": lambda c: f"{c}, a food dish",
    "an image of {class} food": lambda c: f"an image of {c} food",
}

# In[ ]:

results = []
template_text_features = {}

for name, fn in prompt_templates.items():
    prompts = [fn(format_class_name(c)) for c in class_names]
    tf = get_text_features(prompts)
    template_text_features[name] = tf

    top1, top5, _, _ = evaluate(tf, all_image_features, all_labels)
    results.append({"prompt_style": name, "top1_acc": top1, "top5_acc": top5})
    print(f"{name:45s} Top-1: {top1:.2f}% Top-5: {top5:.2f}%")

# Prompt ensembling: average the TEXT EMBEDDINGS across all 5 templates, then re-normalize.
# This is embedding-level ensembling, not "pick the best prompt".
ensemble_features = torch.stack(list(template_text_features.values())).mean(dim=0)
ensemble_features = ensemble_features / ensemble_features.norm(dim=-1, keepdim=True)

ens_top1, ens_top5, _, _ = evaluate(ensemble_features, all_image_features, all_labels)
results.append({"prompt_style": "ENSEMBLE (avg of all 5)", "top1_acc": ens_top1, "top5_acc": ens_top5})

print(f"\n{'ENSEMBLE (avg of all 5)':45s} Top-1: {ens_top1:.2f}% Top-5: {ens_top5:.2f}%")

# In[ ]:

results_df = pd.DataFrame(results).sort_values("top1_acc", ascending=False).reset_index(drop=True)
print(results_df)
results_df.to_csv(prompt_engineering_results, index=False)

# Compare the ensemble against the best-performing individual prompt template
best_single = results_df[results_df["prompt_style"] != "ENSEMBLE (avg of all 5)"].iloc[0]
ensemble_row = results_df[results_df["prompt_style"] == "ENSEMBLE (avg of all 5)"].iloc[0]

print(f"\nBest single prompt: {best_single['prompt_style']} ({best_single['top1_acc']:.2f}%)")
print(f"Ensemble accuracy: {ensemble_row['top1_acc']:.2f}%")
print(f"Ensemble vs best single: {ensemble_row['top1_acc'] - best_single['top1_acc']:+.2f} pts")

# In[ ]:

plt.figure(figsize=(10, 6))
plt.barh(results_df["prompt_style"], results_df["top1_acc"], color="steelblue")
plt.xlabel("Top-1 Accuracy (%)")
plt.title("CLIP Zero-Shot Accuracy by Prompt Style (Food-101)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(prompt_engineering_comparison, dpi=200)
plt.show()
