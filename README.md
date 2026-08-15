# Convolutional Neural Networks and Vision-Language Models for Food Image Classification

MSc Software Engineering dissertation, University of Limerick, 2026.
Supervised by J.J. Collins.

This repository contains the code, job scripts and results for an empirical
comparison of four convolutional architectures trained from scratch on Food-101
against two pre-trained vision-language models used zero-shot.

---

## Summary of results

All figures are top-1 accuracy on the full 25,250 image Food-101 test set.

| Model | Family | Top-1 | Top-5 | Food-101 labels used |
|---|---|---|---|---|
| AlexNet | CNN | 53.35% | 79.94% | 68,175 |
| ResNet50 | CNN | 71.52% | 91.48% | 68,175 |
| ResNet34 | CNN | 71.82% | 91.50% | 68,175 |
| YOLOv8s-cls | CNN | 84.26% | 96.44% | 68,175 |
| CLIP ViT-B/32 | VLM (zero-shot) | 84.19% | 97.36% | 0 |
| SigLIP base/16 | VLM (zero-shot) | **91.62%** | **98.99%** | 0 |

Two findings worth highlighting:

- YOLOv8s and zero-shot CLIP could not be separated statistically
  (McNemar, Holm-corrected *p* = 0.774, 95% CI on the difference
  [−0.40, +0.56]).
- A linear probe on frozen CLIP features reached 83.06% using 10% of the
  training data, which exceeds every convolutional network trained on 100%.

---

## Requirements

- Python 3.12
- CUDA 12.1 and an NVIDIA GPU
- Roughly 10 GB of disk space for the dataset

All results in the dissertation were produced on an NVIDIA H100 NVL under Slurm.
The scripts run on any CUDA GPU, but training times will differ.

---

## Setup

```bash
git clone https://github.com/chiragrawat12/MSc-Software-Engineering-Final-Year-Dissertation
cd MSc-Software-Engineering-Final-Year-Dissertation

python -m venv ~/envs/myenv
source ~/envs/myenv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.4.70 transformers scikit-learn statsmodels scipy
pip install numpy pandas pillow matplotlib seaborn tqdm sentencepiece protobuf
```

`sentencepiece` and `protobuf` are required by SigLIP's tokenizer and will cause
Experiment 7 to fail if missing.

### Dataset

Food-101 downloads automatically on first run. Fetch it once from an interactive
session before submitting any batch job, because two jobs extracting into the
same directory at once will corrupt it.

```bash
python -c "from torchvision.datasets import Food101; Food101(root='./data', split='train', download=True)"
```

The archive is about 5 GB and extracts to roughly 101,000 files.

### Verifying the GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Every training script aborts rather than falling back to CPU if CUDA is
unavailable, since a CPU run would be impractically slow.

---

## Folder structure

```
.
├── scripts/     14 Python scripts, one per prototype and per experiment
├── jobs/        14 Slurm submission scripts
├── outputs/     13 result directories, one per prototype and per experiment
├── logs/        Slurm stdout and stderr for every job submitted
└── data/        Food-101 (downloaded, not tracked in git)
```

Model weights are not in this repository. They are held in the OneDrive archive
linked in the dissertation, along with a copy of everything here.

### scripts/

| Script | Purpose |
|---|---|
| `AlexNet_Food101.py` | AlexNet baseline, trained from scratch |
| `ResNet34_Food101.py` | ResNet34 baseline |
| `ResNet50_Food101.py` | ResNet50 baseline |
| `YOLOv8_Food101.py` | YOLOv8s-cls baseline |
| `CLIP_Zero_Shot_Food101.py` | Experiment 1: zero-shot CLIP |
| `Experiment_2.py` | Prompt engineering and ensembling |
| `Experiment_3.py` | Linear probe on frozen CLIP features |
| `Experiment_4.py` | Full fine-tuning of CLIP |
| `Experiment_5.py` | Data-efficiency curve, five subset sizes |
| `Experiment_6.py` | Few-shot CLIP, K = 1 to 16 |
| `Experiment_7.py` | SigLIP zero-shot |
| `Experiment_8.py` | ResNet50 hyperparameter grid search, 27 runs |
| `Experiment_9.py` | McNemar and Wilcoxon significance testing |
| `regenerate_predictions.py` | Recovers per-image predictions from saved weights |

### outputs/

Each directory holds the results for one prototype or experiment. Files common
to most of them:

| File | Contents |
|---|---|
| `preds.npy` | Top-1 prediction per test image, 25,250 values |
| `labels.npy` | Ground truth, same ordering |
| `per_class_acc.npy` | Per-class accuracy, 101 values |
| `confusion_matrix.npy` | 101 × 101 confusion matrix |
| `metrics.json` | Top-1 and top-5 accuracy |
| `images/` | Figures |

The test set is loaded with `shuffle=False` everywhere, so index *i* of any
`preds.npy` refers to the same image across all models. This is what makes the
paired statistical tests in Experiment 9 valid.

---

## Running the experiments

Order matters. Some experiments reuse cached CLIP features written by earlier
ones, so running out of sequence causes unnecessary recomputation.

```bash
# 1. Convolutional baselines, any order
sbatch jobs/alexnet.slurm
sbatch jobs/resnet34.slurm
sbatch jobs/resnet50.slurm
sbatch jobs/yolo.slurm

# 2. Zero-shot CLIP, caches the image features used by Experiments 2 and 3
sbatch jobs/clipzeroshot.slurm

# 3. Prompt engineering and linear probe
sbatch jobs/experiment_2.slurm
sbatch jobs/experiment_3.slurm

# 4. Fine-tuning
sbatch jobs/experiment_4.slurm

# 5. Data efficiency, then few-shot which reuses its cached features
sbatch jobs/experiment_5.slurm
sbatch jobs/experiment_6.slurm

# 6. SigLIP and the grid search, both independent
sbatch jobs/experiment_7.slurm
sbatch jobs/experiment_8.slurm

# 7. Significance testing, requires every earlier run to have finished
sbatch jobs/experiment_9.slurm
```

Monitor with `squeue -u $USER` and follow output with `tail -f logs/<name>/*.out`.

Without Slurm, run the scripts directly:

```bash
python scripts/ResNet50_Food101.py
```

### Approximate runtimes on an H100

| Job | Time |
|---|---|
| AlexNet, ResNet34, ResNet50 | 1 to 4 hours each |
| YOLOv8s | up to 10 hours |
| Experiments 1, 2, 3, 7 | under 1 hour each |
| Experiment 4 | several hours |
| Experiment 5 | up to 20 hours, 25 training runs |
| Experiment 8 | up to 30 hours, 27 training runs |
| Experiments 6 and 9 | minutes, no training |

Experiment 9 performs no GPU work at all. It reads saved prediction arrays and
runs two statistical tests, so it can be run on a laptop.

---

## Reproducibility

Every script accepts a `--seed` argument and seeds Python, NumPy, PyTorch and
each DataLoader worker at startup.

The validation split is generated from a separate fixed seed held constant across
all runs and all architectures, so the same 7,575 images are used for model
selection everywhere. Food-101 ships with only a train and a test split, so a
stratified 10% validation set is carved from the training data, giving 68,175
training, 7,575 validation and 25,250 test images. Early stopping and checkpoint
selection use the validation split only, and the test set is evaluated once per
run after the best checkpoint has been restored.

Results in the dissertation come from single runs at seed 42. Experiment 6 is the
exception, repeating each value of *K* across five sampling trials, because the
variance at one image per class is too large for a single run to be meaningful.

---

## Data and model sources

- **Food-101**, Bossard et al. (2014), released by the Computer Vision Laboratory
  at ETH Zurich for non-commercial research use.
- **CLIP ViT-B/32**, `openai/clip-vit-base-patch32`, MIT licence.
- **SigLIP base/16**, `google/siglip-base-patch16-224`, Apache 2.0.

The four convolutional models were trained from random initialisation and use no
pre-trained weights.

---

## Citation

If you refer to this work:

```
Singh, C. (2026). A Comparative Analysis of Convolutional Neural Networks and
Vision-Language Models for Food Image Classification. MSc dissertation,
University of Limerick.
```
