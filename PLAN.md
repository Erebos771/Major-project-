# Build plan

Tracking the "Suggested build order" from the task spec. Check items off as they land.

- [x] 0. Repo scaffold, venv, requirements (PyTorch, MPS-enabled on M1 Max)
- [ ] 1. Dataset download (manual, user-provided) + preprocessing pipeline; verify counts
      (Broadleaf 1195 / Grass 1200 / Soil 1200 / Soybean 1200, 3836 train / 959 test)
- [ ] 2. Stream 1 (ViT+CAB reproduction) trained standalone; confirm ~88% acc / 0.97 AUC
- [ ] 3. + Stream 2 (vegetation/color: ExG, ExG-ExR, VARI, HSV, Lab)
- [ ] 4. + Stream 3 (texture: GLCM, Gabor) — targeted at grass recall
- [ ] 5. + Stream 4 (CLIP semantic embedding, frozen)
- [ ] 6. Attention-weighted fusion vs concat fusion ablation
- [ ] 7. Augmentation, focal/class-weighted loss, 5-fold CV, stronger backbone comparison
- [ ] 8. Final ablation table (accuracy/AUC/per-class P-R-F1 at each stage)
- [ ] 9. Grad-CAM: paper's model vs final model

## Notes / decisions

- Framework: PyTorch (MPS backend on M1 Max). CLIP via `transformers` (`openai/clip-vit-base-patch32`).
- Dataset obtained via manual download (no Kaggle API credentials on this machine).
- Random seed 42 throughout for reproducibility parity with the paper.
