# HybridViT-CAB → Multimodal Weed Classification

Reproduction of "HybridViT-CAB: a vision transformer and convolutional attention network for
precision weed detection in agricultural systems" (Scientific Reports, 2026), followed by a
from-scratch multi-stream/multimodal upgrade aimed at beating the paper's 88% accuracy and
fixing its weakest class (grass, 68% recall).

**Result: 99.48% accuracy, 0.9998 AUC, grass F1 up from the paper's 0.79 to 0.99.**

## Results

Final metrics vs. the paper's Table 2, and the incremental ablation showing what each added
component contributed. Full breakdown at [results/tables/ablation_table.md](results/tables/ablation_table.md).

| Stage | Accuracy | AUC | Grass F1 |
|---|---|---|---|
| Paper (HybridViT-CAB, reported) | 88.0% | 0.970 | 0.79 |
| Stream 1 — our ViT+CAB reproduction | 92.2% | 0.987 | 0.85 |
| + Stream 2 (vegetation/color indices) | 93.0% | 0.990 | 0.88 |
| + Stream 3 (GLCM + Gabor texture) | 97.6% | 0.999 | 0.96 |
| + Stream 4 (frozen CLIP embedding), concat fusion | 99.1% | 0.9998 | 0.99 |
| **+ Stream 4, attention-weighted fusion (final model)** | **99.5%** | **0.9998** | **0.99** |

Texture (Stream 3) was the single biggest lever — grass, broadleaf, and soybean are all
low-color-variance plant matter and differ mainly in leaf/blade shape, which the paper's
RGB-only architecture never explicitly modeled. CLIP's pretrained semantic prior closed most
of the remaining gap. Confusion matrices for every stage are in
[results/figures/](results/figures/); the final model makes only 5 mistakes across all 959
test images (all grass→broadleaf).

Grad-CAM comparison (paper-style Stream 1 vs. the final model) is in
[results/gradcam/](results/gradcam/) — the final model's activation concentrates more tightly
on individual leaf/blade edges rather than diffuse foliage regions.

## What was built

- **Dataset pipeline** ([src/data/prepare_dataset.py](src/data/prepare_dataset.py)): rebuilds
  the paper's class-balanced subset (max 1200/class, 1195 cap for broadleaf) from the raw
  Kaggle TIFFs, converts to 224×224 JPEG, stratified 80/20 split (seed 42). Verified against
  the paper's Table 1 (broadleaf capped at 1191, the archive's actual count — 4 short of the
  paper's claimed 1195 — everything else matches exactly, including the 959-image test set).
- **Stream 1** ([src/models/streams/vit_cab.py](src/models/streams/vit_cab.py)): the paper's
  own architecture, rebuilt from scratch in PyTorch — Conv2D patch embedding → 2 transformer
  encoder blocks → GAP (ViT branch), plus a Conv-BN-ReLU ×2 → squeeze-excitation → GAP branch
  (CAB), fused via concat → Dense → Dropout → Dense. 425K params vs. the paper's reported
  412,676 (~3% off; some architectural details aren't fully specified in the paper).
- **Stream 2** ([src/data/vegetation_features.py](src/data/vegetation_features.py),
  [streams/veg_color.py](src/models/streams/veg_color.py)): per-pixel ExG, ExG−ExR, VARI, HSV,
  Lab stacked into a 9-channel input, fed through a small CNN.
- **Stream 3** ([src/data/texture_features.py](src/data/texture_features.py),
  [streams/texture.py](src/models/streams/texture.py)): sliding-window GLCM statistics
  (contrast/homogeneity/energy/correlation) + an 8-orientation/frequency Gabor filter bank,
  12 channels total, through a small CNN. Cached at 56×56/float16 (not the full 224×224/float32
  it's computed at) — the branch downsamples to that resolution internally anyway, and storing
  it small avoided a random-I/O bottleneck that turned each training epoch from ~3 min into ~10.
- **Stream 4** ([streams/clip_semantic.py](src/models/streams/clip_semantic.py)): frozen
  `openai/clip-vit-base-patch32` image embeddings (512-d), precomputed once and cached, with a
  small learned projection head on top.
- **Fusion** ([src/models/fusion.py](src/models/fusion.py)): both a naive-concat head and a
  learned per-sample attention-weighted head (softmax gate over the 4 streams) were trained and
  compared — attention fusion won (99.48% vs. concat's 99.06%) and is the final model.
- **Training** ([src/train/](src/train/)): Adam, cosine-annealed LR (1e-3→1e-5), batch 32, up
  to 25 epochs, early stopping (patience 5, gated to not fire before epoch 10 — an earlier run
  without that gate stopped prematurely on validation-loss noise while LR was still high,
  undertraining the model and giving a misleadingly bad ablation number; see git history).
  Also implemented but not exercised on the final model (near-ceiling performance made them
  low-value to chase further): heavy augmentation + CutMix/MixUp
  ([src/data/augmentation.py](src/data/augmentation.py)), focal + class-weighted loss
  ([src/train/losses.py](src/train/losses.py)), 5-fold CV
  ([src/train/cross_validate.py](src/train/cross_validate.py)), and a ConvNeXt-Tiny fine-tune
  comparison ([src/train/train_pretrained_backbone.py](src/train/train_pretrained_backbone.py)).
- **Eval** ([src/eval/](src/eval/)): accuracy/AUC/per-class P-R-F1/confusion matrix
  ([metrics.py](src/eval/metrics.py)), confusion matrix plots
  ([plot_confusion.py](src/eval/plot_confusion.py)), Grad-CAM for both the Stream 1 and full
  multi-stream models ([gradcam.py](src/eval/gradcam.py)), and the ablation table builder
  ([build_ablation_table.py](src/eval/build_ablation_table.py)).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Kaggle: [Weed Detection in Soybean Crops](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops)

Manual download: unzip into `dataset/dataset/` so each class (`broadleaf`, `grass`, `soil`,
`soybean`) has its own subfolder of `.tif` images (already done for this checkout). Then:

```bash
python -m src.data.prepare_dataset                                    # builds data/processed/
python -m src.data.precompute_features --streams veg,texture,clip     # caches Stream 2-4 features
```

## Reproducing the results

```bash
# Stream 1 alone (paper reproduction baseline)
python -m src.train.train_stream1

# Incremental ablation
python -m src.train.train_multistream --streams rgb,veg --fusion concat --tag rgb_veg
python -m src.train.train_multistream --streams rgb,veg,texture --fusion concat --tag rgb_veg_texture
python -m src.train.train_multistream --streams rgb,veg,texture,clip --fusion concat --tag full_concat
python -m src.train.train_multistream --streams rgb,veg,texture,clip --fusion attention --tag full_attention

# Eval / figures
python -m src.eval.plot_confusion --metrics results/tables/full_attention_test_metrics.json \
    --out results/figures/full_attention_confusion.png --title "Final model"
python -m src.eval.gradcam --model full --ckpt checkpoints/full_attention_best.pt --tag final_model
python -m src.eval.build_ablation_table --tags stream1,rgb_veg,rgb_veg_texture,full_concat,full_attention \
    --labels "Stream1,+veg/color,+texture,+CLIP concat,+CLIP attention"
```

## Project layout

```
src/data/            dataset prep, balanced subset builder, feature precompute/caching,
                      PyTorch Datasets, augmentation
src/models/streams/  Stream 1 (ViT+CAB), Stream 2 (veg/color), Stream 3 (texture), Stream 4 (CLIP)
src/models/          fusion heads (concat + attention), full multi-stream model
src/train/           training scripts, losses (focal/class-weighted), cross-validation
src/eval/            metrics, confusion matrices, Grad-CAM, ablation table
configs/             (reserved for run configs)
results/             tables, figures, gradcam outputs — see subfolders for every stage's numbers
checkpoints/         saved model weights, one per ablation stage (best-val-loss checkpoint)
```

See [PLAN.md](PLAN.md) for build-order tracking and methodology notes.
