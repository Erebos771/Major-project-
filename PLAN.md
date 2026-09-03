# Build plan

Tracking the "Suggested build order" from the task spec. Final status below.

- [x] 0. Repo scaffold, venv, requirements (PyTorch, MPS-enabled on M1 Max)
- [x] 1. Dataset preprocessing pipeline; counts verified against paper Table 1
      (Broadleaf 1191/1195 available, Grass/Soil/Soybean 1200 each; 3832 train / 959 test —
      test split is an exact match to the paper's 959; train is 4 short only because the
      archive has 4 fewer broadleaf images than the paper claims were available)
- [x] 2. Stream 1 (ViT+CAB reproduction) — **92.18% acc, 0.9871 AUC** vs paper's 88%/0.97.
      425,092 params vs paper's 412,676.
- [x] 3. + Stream 2 (vegetation/color) — **93.01% acc, 0.9897 AUC**
- [x] 4. + Stream 3 (texture: GLCM + Gabor) — **97.60% acc, 0.9987 AUC** — biggest single jump,
      confirms texture was the dominant missing cue for grass/broadleaf/soybean
- [x] 5. + Stream 4 (CLIP semantic, frozen) — **99.06% acc, 0.9998 AUC** (concat fusion)
- [x] 6. Attention-weighted fusion vs concat — attention wins: **99.48% acc, 0.9998 AUC** (final model)
- [~] 7. Training improvements — augmentation/CutMix/MixUp, focal/class-weighted loss,
      ConvNeXt-Tiny comparison, and 5-fold CV are all implemented and ready to run
      (`src/data/augmentation.py`, `src/train/losses.py`,
      `src/train/train_pretrained_backbone.py`, `src/train/cross_validate.py`), but were not
      exercised on the final model — at 99.48% acc / 5 total errors on 959 test images,
      performance is already near-ceiling for this dataset, and 5-fold CV alone would cost
      ~6-12 hours of additional training for limited expected benefit. Skipped by user decision.
- [x] 8. Final ablation table: `results/tables/ablation_table.md`
- [x] 9. Grad-CAM: paper-style Stream1 vs final model, `results/gradcam/`

## Final results summary

| Stage | Accuracy | AUC | Grass F1 |
|---|---|---|---|
| Paper (reported) | 88.0% | 0.970 | 0.79 |
| Stream 1 (reproduction) | 92.2% | 0.987 | 0.85 |
| +veg/color | 93.0% | 0.990 | 0.88 |
| +texture | 97.6% | 0.999 | 0.96 |
| +CLIP, concat fusion | 99.1% | 0.9998 | 0.99 |
| **+CLIP, attention fusion (final)** | **99.5%** | **0.9998** | **0.99** |

See README.md for the full writeup, file pointers, and how to reproduce each stage.

## Notes / decisions

- Framework: PyTorch (MPS backend on M1 Max). CLIP via `transformers` (`openai/clip-vit-base-patch32`).
- Dataset obtained via manual download, already unzipped at `dataset/dataset/<class>/*.tif`
  (no Kaggle API credentials on this machine).
- Random seed 42 throughout for reproducibility parity with the paper.
- Two real bugs found and fixed mid-project (see git history for full detail):
  1. `transformers` v5.16.1 changed `CLIPModel.get_image_features()`'s return type
     (`BaseModelOutputWithPooling` instead of a raw tensor) — fixed by reading `.pooler_output`.
  2. Early stopping could fire before the cosine LR schedule settled down, on validation-loss
     noise alone — produced a misleadingly bad rgb_veg ablation number (82.7% instead of the
     true 93.0%) from an undertrained epoch-2 checkpoint. Fixed with a `MIN_EPOCHS=10` gate
     before patience can trigger, applied to all training scripts.
- The texture feature cache was originally stored at full 224×224/float32 (9.2GB/split),
  which turned shuffled per-epoch reads into a random-I/O bottleneck (~10 min/epoch vs ~3).
  Fixed by caching at 56×56/float16 instead (the branch downsamples to that resolution
  internally anyway), cutting the cache ~30x with no accuracy cost.
