"""Plot a confusion matrix (with per-class counts) from a saved *_test_metrics.json file.

Usage:
    python -m src.eval.plot_confusion --metrics results/tables/stream1_test_metrics.json \
        --out results/figures/stream1_confusion.png --title "Stream 1 (paper reproduction)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.dataset import CLASSES


def plot(metrics_path: str, out_path: str, title: str) -> None:
    with open(metrics_path) as f:
        metrics = json.load(f)
    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nacc={metrics['accuracy']*100:.1f}%  AUC={metrics['auc_macro']:.3f}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    plot(args.metrics, args.out, args.title)
