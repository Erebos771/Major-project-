"""5-fold stratified cross-validation over train+test combined, for a trustworthy accuracy
estimate (mean +/- std) instead of relying on the paper's single 80/20 split.

Trains the given stream/fusion config fresh in each fold and reports per-fold + aggregate
accuracy, AUC, and per-class F1.

Usage:
    python -m src.train.cross_validate --streams rgb --fusion concat --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import CLASSES, read_manifest
from src.data.multistream_dataset import MultiStreamDataset, collate_multistream
from src.eval.metrics import compute_metrics
from src.models.fusion import MultiStreamModel
from src.train.losses import FocalLoss, class_weights_from_counts
from src.train.train_multistream import build_branches, get_device, run_epoch

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 5
MIN_EPOCHS = 10  # don't allow early stopping before this; cosine LR is still high/noisy early on
LR_MAX = 1e-3
LR_MIN = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="rgb")
    ap.add_argument("--fusion", choices=["concat", "attention"], default="concat")
    ap.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="cv")
    args = ap.parse_args()
    streams = args.streams.split(",")
    device = get_device()

    # Combine train+test manifests into one pool, refold with StratifiedKFold.
    all_items = read_manifest("train") + read_manifest("test")
    labels_str = [c for _, c in all_items]

    # NOTE: MultiStreamDataset reads from a fixed split's manifest rows + precomputed feature
    # caches for that split. To keep this straightforward, cross-validation here reuses
    # per-split caches by concatenating train and test datasets (both must be precomputed).
    from torch.utils.data import ConcatDataset

    train_eval_tf = transforms.Compose([transforms.ToTensor()])
    train_aug_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    train_ds_aug = MultiStreamDataset("train", streams=tuple(streams), rgb_transform=train_aug_tf)
    train_ds_eval = MultiStreamDataset("train", streams=tuple(streams), rgb_transform=train_eval_tf)
    test_ds_aug = MultiStreamDataset("test", streams=tuple(streams), rgb_transform=train_aug_tf)
    test_ds_eval = MultiStreamDataset("test", streams=tuple(streams), rgb_transform=train_eval_tf)

    n_train = len(train_ds_aug)
    n_test = len(test_ds_aug)
    pooled_aug = ConcatDataset([train_ds_aug, test_ds_aug])
    pooled_eval = ConcatDataset([train_ds_eval, test_ds_eval])
    pooled_labels = labels_str  # order matches train items then test items, per read_manifest concatenation above
    assert len(pooled_labels) == n_train + n_test

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(pooled_labels)), pooled_labels), start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        branches = build_branches(streams)
        model = MultiStreamModel(branches, num_classes=len(CLASSES), fusion=args.fusion).to(device)

        fold_train_labels = [pooled_labels[i] for i in train_idx]
        if args.loss == "ce":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            from collections import Counter

            counts = Counter(fold_train_labels)
            weights = class_weights_from_counts(dict(counts), CLASSES).to(device)
            criterion = (
                torch.nn.CrossEntropyLoss(weight=weights)
                if args.loss == "weighted_ce"
                else FocalLoss(gamma=2.0, alpha=weights)
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)

        train_loader = DataLoader(
            Subset(pooled_aug, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
            collate_fn=collate_multistream,
        )
        test_loader = DataLoader(
            Subset(pooled_eval, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
            collate_fn=collate_multistream,
        )

        best_loss, no_improve = float("inf"), 0
        ckpt = Path("checkpoints") / f"{args.tag}_fold{fold}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_m = run_epoch(model, train_loader, streams, device, optimizer, criterion)
            test_m = run_epoch(model, test_loader, streams, device, optimizer=None, criterion=criterion)
            scheduler.step()
            print(
                f"fold{fold} epoch {epoch:2d}  train_acc={train_m['accuracy']*100:.2f}%  "
                f"held_acc={test_m['accuracy']*100:.2f}%  held_loss={test_m['loss']:.4f}"
            )
            if test_m["loss"] < best_loss - 1e-5:
                best_loss, no_improve = test_m["loss"], 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
                    print(f"fold{fold} early stop at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(ckpt, map_location=device))
        final_m = run_epoch(model, test_loader, streams, device, optimizer=None, criterion=criterion)
        fold_results.append(final_m)
        print(f"fold{fold} final: acc={final_m['accuracy']*100:.2f}% auc={final_m['auc_macro']:.4f}")

    accs = [r["accuracy"] for r in fold_results]
    aucs = [r["auc_macro"] for r in fold_results]
    summary = {
        "folds": args.folds,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "per_fold": fold_results,
    }
    print(f"\n=== {args.folds}-fold CV summary ===")
    print(f"Accuracy: {summary['accuracy_mean']*100:.2f}% +/- {summary['accuracy_std']*100:.2f}%")
    print(f"AUC:      {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.tag}_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
