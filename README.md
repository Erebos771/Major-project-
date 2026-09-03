# HybridViT-CAB → Multimodal Weed Classification

Reproduction of "HybridViT-CAB: a vision transformer and convolutional attention network for
precision weed detection in agricultural systems" (Scientific Reports, 2026), followed by a
from-scratch multi-stream/multimodal upgrade aimed at beating the paper's 88% accuracy and
fixing its weakest class (grass, 68% recall).

Status: scaffolding + data pipeline in progress. See `PLAN.md` for the build order and
`results/` for metrics/tables/figures as they land.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Kaggle: [Weed Detection in Soybean Crops](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops)

Manual download: grab the dataset zip from the Kaggle page above and unzip it into
`dataset/dataset/` so that each class (`broadleaf`, `grass`, `soil`, `soybean`) has its own
subfolder of `.tif` images (already done for this checkout). Then run:

```bash
python -m src.data.prepare_dataset
```

This builds the class-balanced (max 1200/class, broadleaf capped at 1195), 224x224 JPEG,
stratified 80/20 (seed 42) split described in `PLAN.md`.

## Project layout

```
src/data/            dataset prep, balanced subset builder, PyTorch Dataset/DataLoader, augmentations
src/models/streams/  Stream 1 (ViT+CAB), Stream 2 (veg/color), Stream 3 (texture), Stream 4 (CLIP)
src/models/          fusion heads (concat + attention-weighted), full multimodal model
src/train/           training scripts, loss functions (focal / class-weighted), cross-validation
src/eval/            metrics, confusion matrices, Grad-CAM
configs/             run configs (yaml)
results/             tables, figures, gradcam outputs
checkpoints/         saved model weights
```
