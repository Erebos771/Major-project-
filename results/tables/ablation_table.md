# Ablation table

| Stage | Accuracy | AUC (macro) | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|
| **Paper (HybridViT-CAB, reported)** | 88.0% | 0.970 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.81 | 0.83 | 0.82 | |
| &nbsp;&nbsp;grass | | | 0.94 | 0.68 | 0.79 | |
| &nbsp;&nbsp;soil | | | 1.00 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 0.79 | 1.00 | 0.88 | |
| **Stream1 (ViT+CAB repro)** | 92.2% | 0.987 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.87 | 0.90 | 0.89 | |
| &nbsp;&nbsp;grass | | | 0.86 | 0.85 | 0.85 | |
| &nbsp;&nbsp;soil | | | 0.99 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 0.96 | 0.94 | 0.95 | |
| **+veg/color** | 93.0% | 0.990 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.87 | 0.93 | 0.90 | |
| &nbsp;&nbsp;grass | | | 0.89 | 0.87 | 0.88 | |
| &nbsp;&nbsp;soil | | | 1.00 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 0.97 | 0.93 | 0.95 | |
| **+texture** | 97.6% | 0.999 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.96 | 0.98 | 0.97 | |
| &nbsp;&nbsp;grass | | | 0.96 | 0.96 | 0.96 | |
| &nbsp;&nbsp;soil | | | 1.00 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 0.99 | 0.96 | 0.97 | |
| **+CLIP concat** | 99.1% | 1.000 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.98 | 1.00 | 0.99 | |
| &nbsp;&nbsp;grass | | | 1.00 | 0.97 | 0.99 | |
| &nbsp;&nbsp;soil | | | 1.00 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 1.00 | 0.99 | 0.99 | |
| **+CLIP attention (BEST)** | 99.5% | 1.000 | | | | |
| &nbsp;&nbsp;broadleaf | | | 0.98 | 1.00 | 0.99 | |
| &nbsp;&nbsp;grass | | | 1.00 | 0.98 | 0.99 | |
| &nbsp;&nbsp;soil | | | 1.00 | 1.00 | 1.00 | |
| &nbsp;&nbsp;soybean | | | 1.00 | 1.00 | 1.00 | |
