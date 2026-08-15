#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
print("CUDA available:", torch.cuda.is_available())   # Must print True
print("GPU:", torch.cuda.get_device_name(0))          # Should print RTX 4050


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import Food101
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os


# In[ ]:


# All paths are built relative to this script so the project can be moved or
# run on a different machine without editing hard-coded locations.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, '..', 'data')
best_model = os.path.join(script_dir, '..', 'models/alexnet/best_models/best_alexnet_model.pth')
image_output = os.path.join(script_dir, '..', 'outputs/alexnet/images/alexnet_training_curves.png')
saved_model = os.path.join(script_dir, '..', 'models/alexnet/saved_models/alexnet_food101.pth')


# In[11]:


# Resize to 256 then randomly crop to 227 so each epoch sees a slightly
# different region of the image. 227 is the input size AlexNet expects.
# Flip and colour jitter add further variation to limit overfitting.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # ImageNet channel statistics, standard for pretrained-style preprocessing
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# No augmentation at test time. Resized straight to 227 so evaluation is
# deterministic and every run is comparable.
test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[12]:


# download=True only fetches the archive on the first run, afterwards it is
# skipped if the extracted data is already present in data_root.
train_dataset = Food101(root=data_root, split='train', download=True, transform=train_transform)
test_dataset  = Food101(root=data_root, split='test',  download=True, transform=test_transform)

# Sanity check against the published Food-101 split sizes
print(f"Train size: {len(train_dataset)}")   # 75,750
print(f"Test size:  {len(test_dataset)}")    # 25,250
print(f"Classes:    {len(train_dataset.classes)}")  # 101


# In[13]:


# Test loader is left unshuffled so predictions stay aligned with the label
# order used later in the classification report.
# pin_memory speeds up the host to GPU copy; num_workers loads batches in parallel.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


# In[14]:


# AlexNet built to the original 2012 architecture, with the final layer
# resized from 1000 ImageNet classes to the 101 Food-101 classes.
class AlexNet(nn.Module):
    def __init__(self, num_classes=101):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),    # Conv1
            nn.ReLU(inplace=True),
            # Local Response Normalisation as used in the original paper.
            # Superseded by batch norm in later architectures, kept here to
            # stay faithful to the baseline.
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1), # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            # Dropout on both hidden FC layers, where most of the parameters
            # sit and where overfitting is most likely.
            nn.Dropout(p=0.5),
            # 256 x 6 x 6 is the feature map size produced by a 227x227 input.
            # Changing the input resolution breaks this line.
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 101 food classes
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten from dim 1 onwards to keep the batch dimension intact
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda")
model = AlexNet(num_classes=101).to(device)
print(model)


# In[15]:


# CrossEntropyLoss applies softmax internally, so the model returns raw logits.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# Learning rate drops by a factor of 10 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[16]:


# Training stops early if validation accuracy fails to improve for
# `patience` consecutive epochs.
best_acc = 0.0
patience = 5
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
    # eval() disables dropout; no_grad() skips gradient tracking to save memory
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

    # --- Early Stopping ---
    # Checkpoint is overwritten only when accuracy improves, so the file always
    # holds the best weights seen so far rather than the most recent ones.
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

# Reload the best checkpoint, since the weights in memory are from the last
# epoch and may be worse than the best one.
model.load_state_dict(torch.load(best_model, weights_only=True, map_location=device))
model.eval()
print(f"\nTraining complete. Best Val Accuracy: {best_acc * 100:.2f}%")


# In[ ]:


# Loss and accuracy curves side by side, used to check for overfitting and
# to see the effect of the learning rate drops.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-o')
ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')

# Length of train_losses is used for the x axis because early stopping may
# have ended training before NUM_EPOCHS.
ax2.plot(range(1, len(train_losses) + 1), [a*100 for a in val_accuracies], 'g-o')
ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(image_output, dpi=150)
plt.show()


# In[ ]:


# Final evaluation pass. Predictions and labels are collected on the CPU so
# they can be passed to scikit-learn.
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        preds = model(images).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Top-1 Accuracy
top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nTop-1 Accuracy: {top1 * 100:.2f}%")

# Per-class precision, recall and F1, which shows which food categories the
# model confuses rather than just the overall score.
print(classification_report(all_labels, all_preds,
                             target_names=train_dataset.classes))


# In[ ]:


# Note: this creates 'saved_models' in the current working directory, which is
# not the directory `saved_model` points to. The target folder must already
# exist or the save will fail.
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), saved_model)
print("Model saved!")


# In[ ]:


# Reload check: rebuild the architecture and load the saved weights back in to
# confirm the checkpoint is usable for inference later.
# weights_only=True loads tensors only, which avoids unpickling arbitrary code.
model = AlexNet(num_classes=101).to(device)
model.load_state_dict(
    torch.load(saved_model,
               weights_only=True,
               map_location=device)
)
model.eval()

