#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print("CUDA available:", torch.cuda.is_available())   # Must print True
print("GPU:", torch.cuda.get_device_name(0))          # Should print RTX 4050


# In[6]:


import os
import shutil
from pathlib import Path
from torchvision.datasets import Food101
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from ultralytics import YOLO
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib


# In[ ]:


script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, '..', 'data')
FOOD101_IMG_DIR = Path(script_dir) / '..' / 'data' / 'food-101' / 'images'
YOLO_DIR        = Path(script_dir) / '..' / 'data' / 'yolo_food101'
META_DIR        = Path(script_dir) / '..' / 'data' / 'food-101' / 'meta'
PROJECT         = os.path.join(script_dir,'..', 'outputs/yolo/runs/classify')
results_csv_path         = os.path.join(script_dir,'..', 'outputs/yolo/runs/classify/yolov8s_food101/results.csv')
best_model_path         = os.path.join(script_dir,'..', 'outputs/yolo/runs/classify/yolov8s_food101/weights/best.pt')
saved_model_path         = os.path.join(script_dir,'..', 'models/yolo/saved_models/yolov8s_food101.pt')
output_image_path         = os.path.join(script_dir,'..', 'outputs/yolo/images/yolov8_training_curves.png')


# In[7]:


dummy_transform = transforms.ToTensor()
train_dataset = Food101(root=data_root, split='train', download=True, transform=dummy_transform)
test_dataset  = Food101(root=data_root, split='test',  download=True, transform=dummy_transform)

print(f"Train size: {len(train_dataset)}")
print(f"Test size:  {len(test_dataset)}")
print(f"Classes:    {len(train_dataset.classes)}")


# In[8]:


# YOLOv8 needs this exact folder layout:
#
# yolo_food101/
#   train/
#     apple_pie/  <- images here
#     baby_back_ribs/
#     ...
#   val/
#     apple_pie/
#     ...
#
# We use symlinks so we don't duplicate 4GB of data

def read_split(split_file):
    with open(split_file) as f:
        return [l.strip() for l in f.readlines()]

train_items = read_split(META_DIR / 'train.txt')
test_items  = read_split(META_DIR / 'test.txt')

print(f"Train items: {len(train_items)}")
print(f"Test  items: {len(test_items)}")

for split, items in [('train', train_items), ('val', test_items)]:
    for item in items:
        class_name, img_name = item.split('/')
        dest_dir = YOLO_DIR / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        src  = (FOOD101_IMG_DIR / class_name / (img_name + '.jpg')).resolve()
        dest = (dest_dir / (img_name + '.jpg')).resolve()
        if not dest.exists():
            try:
                os.symlink(src, dest)
            except Exception:
                shutil.copy2(src, dest)  # fallback if symlink fails on Windows

print("\nYOLO dataset folder structure ready!")
print(f"Location: {YOLO_DIR.resolve()}")


# In[9]:


# yolov8s-cls = small YOLOv8 classification model (recommended for RTX 4050)
# weights=None equivalent: using .yaml builds from scratch (no pretrained weights)
# This keeps the comparison fair with AlexNet and ResNet

model = YOLO('yolov8s-cls.yaml')  # from scratch

results = model.train(
    data=str(YOLO_DIR.resolve()),
    epochs=60,
    imgsz=224,           # same image size as ResNet34/50
    batch=64,            # same batch size as AlexNet/ResNet
    device=0,            # RTX 4050
    patience=5,          # early stopping — same as AlexNet/ResNet
    workers=4,
    project=PROJECT,
    name='yolov8s_food101',
    exist_ok=True,
    verbose=True
)

print("\nTraining complete!")
print("Best model saved at: runs/classify/yolov8s_food101/weights/best.pt")


# In[45]:


matplotlib.use('Agg')

results_csv = results_csv_path
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

print("Available columns:", df.columns.tolist())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(df['epoch'], df['train/loss'], 'b-o', markersize=4)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(df['epoch'], df['metrics/accuracy_top1'] * 100, 'g-o', markersize=4)
ax2.set_title('Validation Accuracy (Top-1)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(output_image_path, dpi=150)
plt.show()

best_acc = df['metrics/accuracy_top1'].max() * 100
print(f"\nBest Top-1 Accuracy: {best_acc:.2f}%")


# In[46]:


best_model = YOLO(best_model_path)

metrics = best_model.val(
    data=str(YOLO_DIR.resolve()),
    split='val',
    imgsz=224,
    batch=64,
    device=0
)

print(f"\nTop-1 Accuracy: {metrics.top1 * 100:.2f}%")
print(f"Top-5 Accuracy: {metrics.top5 * 100:.2f}%")


# In[47]:


shutil.copy(best_model_path, saved_model_path)
print("Model saved to saved_models/yolov8s_food101.pt")


# In[48]:


model_reload = YOLO(saved_model_path)
print("Model loaded successfully!")

