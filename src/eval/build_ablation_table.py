"""Aggregate results/tables/<tag>_test_metrics.json files from each ablation stage into one
markdown table: accuracy, AUC, per-class P/R/F1, mirroring the paper's Table 2 layout,
plus a comparison row for the paper's own reported numbers.

Usage:
    python -m src.eval.build_ablation_table --tags stream1,rgb_veg,rgb_veg_texture,full_concat,full_attention \
        --labels "RGB only (Stream1)","+veg/color","+texture","+CLIP (concat)","+CLIP (attention)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import CLASSES

PAPER_ROW = {
    "label": "Paper (HybridViT-CAB, reported)",
    "accuracy": 0.88,
    "auc_macro": 0.97,
    "per_class": {
        "broadleaf": {"precision": 0.81, "recall": 0.83, "f1": 0.82},
        "grass": {"precision": 0.94, "recall": 0.68, "f1": 0.79},
        "soil": {"precision": 1.00, "recall": 1.00, "f1": 1.00},
        "soybean": {"precision": 0.79, "recall": 1.00, "f1": 0.88},
    },
}


def load(tag: str) -> dict:
    path = Path("results/tables") / f"{tag}_test_metrics.json"
    with open(path) as f:
        m = json.load(f)
    return m


def row_md(label: str, m: dict) -> list[str]:
    lines = [f"| **{label}** | {m['accuracy']*100:.1f}% | {m['auc_macro']:.3f} | | | | |"]
    for cls in CLASSES:
        pc = m["per_class"][cls]
        lines.append(f"| &nbsp;&nbsp;{cls} | | | {pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1']:.2f} | |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated result tags")
    ap.add_argument("--labels", required=True, help="comma-separated display labels, same order/count as tags")
    ap.add_argument("--out", default="results/tables/ablation_table.md")
    args = ap.parse_args()

    tags = args.tags.split(",")
    labels = [s.strip().strip('"') for s in args.labels.split(",")]
    assert len(tags) == len(labels), f"{len(tags)} tags vs {len(labels)} labels"

    out_lines = [
        "# Ablation table",
        "",
        "| Stage | Accuracy | AUC (macro) | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|---|---|",
    ]
    out_lines += row_md(PAPER_ROW["label"], PAPER_ROW)
    for tag, label in zip(tags, labels):
        m = load(tag)
        out_lines += row_md(label, m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"Saved {args.out}")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
