# Build plan

Tracking the "Suggested build order" from the task spec. Check items off as they land.

- [x] 0. Repo scaffold, venv, requirements (PyTorch, MPS-enabled on M1 Max)
- [x] 1. Dataset preprocessing pipeline; counts verified against paper Table 1
      (Broadleaf 1191/1195 available, Grass/Soil/Soybean 1200 each; 3832 train / 959 test —
      test split is an exact match, train is 4 short only because the archive has 4 fewer
      broadleaf images than the paper claims were available)
- [~] 2. Stream 1 (ViT+CAB reproduction) — training in progress (`src/train/train_stream1.py`),
      params 425,092 vs paper's 412,676
- [x] 3-6 code written: Stream 2 (`src/data/vegetation_features.py`, `streams/veg_color.py`),
      Stream 3 (`src/data/texture_features.py`, `streams/texture.py`), Stream 4
      (`streams/clip_semantic.py`), fusion heads (`src/models/fusion.py`, concat + attention) —
      not yet trained/evaluated
- [x] 7. code written: heavy augmentation + CutMix/MixUp (`src/data/augmentation.py`),
      focal/class-weighted loss (`src/train/losses.py`), 5-fold CV (`src/train/cross_validate.py`),
      ConvNeXt-Tiny fine-tune comparison (`src/train/train_pretrained_backbone.py`) — not yet run
- [ ] 8. Final ablation table (`src/eval/build_ablation_table.py` ready; needs each stage's run)
- [ ] 9. Grad-CAM (`src/eval/gradcam.py` ready) — paper's model vs final model

## Notes / decisions

- Framework: PyTorch (MPS backend on M1 Max). CLIP via `transformers` (`openai/clip-vit-base-patch32`).
- Dataset obtained via manual download, already unzipped at `dataset/dataset/<class>/*.tif`
  (no Kaggle API credentials on this machine).
- Random seed 42 throughout for reproducibility parity with the paper.
