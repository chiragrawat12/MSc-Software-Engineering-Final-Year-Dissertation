#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
print("CUDA available:", torch.cuda.is_available())   # Must print True
print("GPU:", torch.cuda.get_device_name(0))          # Should print RTX 4050


# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import Food101
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import os


# In[ ]:


# Paths are built relative to this script so the project can be moved or run on
# another machine without editing hard-coded locations.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, '..', 'data')
best_model = os.path.join(script_dir, '..', 'models/resnet50/best_models/best_resnet50_model.pth')
image_output = os.path.join(script_dir, '..', 'outputs/resnet50/images/resnet50_training_curves.png')
saved_model = os.path.join(script_dir, '..', 'models/resnet50/saved_models/resnet50_food101.pth')


# In[10]:


# Identical augmentation pipeline to the ResNet-34 run so that any difference in
# results comes from network depth rather than from the preprocessing.
# Resize to 256 first so the random crop can sample a different region each epoch.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # ImageNet channel statistics, the standard normalisation for these models
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# No augmentation at test time. Resized straight to 224 so evaluation is
# deterministic and comparable across runs and across architectures.
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[11]:


# download=True only fetches the archive on the first run. The same data_root is
# shared with the other model scripts so the dataset is downloaded once.
train_dataset = Food101(root=data_root, split='train', download=True, transform=train_transform)
test_dataset  = Food101(root=data_root, split='test',  download=True, transform=test_transform)

# Sanity check against the published Food-101 split sizes
print(f"Train size: {len(train_dataset)}")   # 75,750
print(f"Test size:  {len(test_dataset)}")    # 25,250
print(f"Classes:    {len(train_dataset.classes)}")  # 101


# In[12]:


# Test loader is left unshuffled so predictions stay aligned with the label
# order used later in the classification report.
# Batch size and worker count match the other runs to keep the comparison fair.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


# In[13]:


device = torch.device("cuda")
# weights=None means the network is trained from scratch rather than fine-tuned
# from ImageNet, matching the other models in the comparison.
model = models.resnet50(weights=None)
# Replace the 1000-class ImageNet head with one sized for Food-101.
# 2048 rather than 512 here because ResNet-50 uses bottleneck blocks, which
# expand the channel count by a factor of 4 at the final stage.
model.fc = nn.Linear(2048, 101)
model = model.to(device)

print(model)


# In[14]:


# CrossEntropyLoss applies softmax internally, so the model returns raw logits.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# Learning rate drops by a factor of 10 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[15]:


# Training stops early if validation accuracy fails to improve for
# `patience` consecutive epochs.
best_acc   = 0.0
patience   = 5
no_improve = 0

NUM_EPOCHS = 60
train_losses, val_accuracies = [], []

for epoch in range(1, NUM_EPOCHS + 1):

    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        # Gradients accumulate by default, so they are cleared each batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Mean loss per batch, so the value is comparable across epochs
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Validation ---
    # eval() switches batch norm to running statistics and disables dropout;
    # no_grad() skips gradient tracking to save memory.
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    scheduler.step()

    # --- Early Stopping & Best Model Saving ---
    # Checkpoint is overwritten only on improvement, so the file always holds
    # the best weights seen so far rather than the most recent ones.
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model)
        no_improve = 0
        print(f"New best model saved ({best_acc * 100:.2f}%)")
    else:
        no_improve += 1
        print(f"No improvement ({no_improve}/{patience})")

    if no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Reload the best checkpoint, since the weights left in memory are from the
# final epoch and may be worse than the best one.
model.load_state_dict(torch.load(best_model, weights_only=True, map_location=device))
model.eval()
print(f"\nTraining complete. Best Val Accuracy: {best_acc * 100:.2f}%")


# In[9]:


# Loss and accuracy curves side by side, used to check for overfitting and to
# see the effect of the learning rate drops.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-o')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Axis length comes from the recorded history rather than NUM_EPOCHS, since
# early stopping may have ended training sooner.
ax2.plot(range(1, len(val_accuracies) + 1), [a * 100 for a in val_accuracies], 'g-o')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(image_output, dpi=150)
plt.show()


# In[10]:


# Final evaluation pass. Predictions and labels are collected on the CPU so they
# can be passed to scikit-learn.
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        preds = model(images).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Top-1 accuracy: fraction of images where the highest scoring class is correct
top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nTop-1 Accuracy: {top1 * 100:.2f}%")

# Per-class precision, recall and F1, which shows which food categories the
# model confuses rather than just the overall score.
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


# In[11]:


# Note: this creates 'saved_models' in the current working directory, which is
# not the directory `saved_model` points to. The target folder must already
# exist or the save on the next line will fail.
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), saved_model)
print("Model saved!")


# In[12]:


# Reload check into a separate variable so the trained model above is left
# intact. The architecture must be rebuilt with the same 101-class head before
# the state dict will load.
# weights_only=True loads tensors only, which avoids unpickling arbitrary code.
model_reload = models.resnet50(weights=None)
model_reload.fc = nn.Linear(2048, 101)
model_reload.load_state_dict(
    torch.load(saved_model, weights_only=True, map_location=device)
)
model_reload = model_reload.to(device)
model_reload.eval()

