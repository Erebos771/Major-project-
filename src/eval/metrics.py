"""Evaluation metrics: accuracy, macro/per-class AUC, precision/recall/F1, confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.dataset import CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """y_true, y_pred: (N,) int labels. y_proba: (N, num_classes) softmax probs."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(CLASSES)
    }

    return {
        "accuracy": float(acc),
        "auc_macro": float(auc_macro),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"Accuracy: {metrics['accuracy']*100:.2f}%   AUC (macro OvR): {metrics['auc_macro']:.4f}")
    lines.append(f"{'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}")
    for cls in CLASSES:
        m = metrics["per_class"][cls]
        lines.append(f"{cls:<10} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f} {m['support']:>8}")
    lines.append("Confusion matrix (rows=true, cols=pred), order = " + ", ".join(CLASSES))
    for row in metrics["confusion_matrix"]:
        lines.append("  " + " ".join(f"{v:>5}" for v in row))
    return "\n".join(lines)
