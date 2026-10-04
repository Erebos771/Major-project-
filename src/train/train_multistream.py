"""Train the multi-stream fusion model with a configurable stream subset and fusion type.
Powers the incremental ablation: RGB-only -> +veg -> +texture -> +CLIP, x {concat, attention} fusion.

Requires `python -m src.data.precompute_features` to have been run first for any of
veg/texture/clip that are included.

Usage:
    python -m src.train.train_multistream --streams rgb,veg,texture,clip --fusion concat \
        --loss focal --tag full_concat
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import CLASSES, read_manifest
from src.data.multistream_dataset import MultiStreamDataset, collate_multistream
from src.eval.metrics import compute_metrics, format_report
from src.models.fusion import MultiStreamModel
from src.models.streams.clip_semantic import ClipBranch
from src.models.streams.texture import TextureBranch
from src.models.streams.veg_color import VegColorBranch
from src.models.streams.vit_cab import CABBranch, ViTBranch
from src.train.losses import FocalLoss, class_weights_from_counts

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 5
MIN_EPOCHS = 10  # don't allow early stopping before this; cosine LR is still high/noisy early on
LR_MAX = 1e-3
LR_MIN = 1e-5
VAL_FRAC = 0.10

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_branches(streams: list[str]) -> dict:
    branches = {}
    if "rgb" in streams:
        # RGB contributes two 64-d sub-streams: ViT branch + CAB branch (paper's Stream 1 halves).
        branches["rgb_vit"] = ViTBranch()
        branches["rgb_cab"] = CABBranch()
    if "veg" in streams:
        branches["veg"] = VegColorBranch()
    if "texture" in streams:
        branches["texture"] = TextureBranch()
    if "clip" in streams:
        branches["clip"] = ClipBranch()
    return branches


def inputs_for_branches(batch_inputs: dict, streams: list[str], device) -> dict:
    """Maps the raw batch dict (keyed by 'rgb'/'veg'/'texture'/'clip') to the per-branch dict
    the MultiStreamModel expects (keyed by branch name, e.g. 'rgb_vit'/'rgb_cab')."""
    out = {}
    if "rgb" in streams:
        out["rgb_vit"] = batch_inputs["rgb"].to(device)
        out["rgb_cab"] = batch_inputs["rgb"].to(device)
    if "veg" in streams:
        out["veg"] = batch_inputs["veg"].to(device)
    if "texture" in streams:
        out["texture"] = batch_inputs["texture"].to(device)
    if "clip" in streams:
        out["clip"] = batch_inputs["clip"].to(device)
    return out


def run_epoch(model, loader, streams, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for batch_inputs, labels in loader:
        labels = labels.to(device)
        model_inputs = inputs_for_branches(batch_inputs, streams, device)
        with torch.set_grad_enabled(is_train):
            logits = model(model_inputs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_proba.append(proba)
        all_pred.append(proba.argmax(axis=1))
        all_true.append(labels.cpu().numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_proba = np.concatenate(all_proba)
    metrics = compute_metrics(all_true, all_pred, all_proba)
    metrics["loss"] = total_loss / n
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="rgb,veg,texture,clip")
    ap.add_argument("--fusion", choices=["concat", "attention"], default="concat")
    ap.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    ap.add_argument("--tag", default="multistream")
    args = ap.parse_args()
    streams = args.streams.split(",")

    device = get_device()
    print(f"Using device: {device}  streams={streams}  fusion={args.fusion}  loss={args.loss}")

    train_items = read_manifest("train")
    labels_str = [c for _, c in train_items]
    idx = list(range(len(train_items)))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=labels_str)

    rgb_train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    rgb_eval_tf = transforms.Compose([transforms.ToTensor()])

    full_train_aug = MultiStreamDataset("train", streams=tuple(streams), rgb_transform=rgb_train_tf)
    full_train_eval = MultiStreamDataset("train", streams=tuple(streams), rgb_transform=rgb_eval_tf)
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_eval, val_idx)
    test_ds = MultiStreamDataset("test", streams=tuple(streams), rgb_transform=rgb_eval_tf)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_multistream
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_multistream
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_multistream
    )

    branches = build_branches(streams)
    model = MultiStreamModel(branches, num_classes=len(CLASSES), fusion=args.fusion).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    if args.loss == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "weighted_ce":
        counts = Counter(labels_str)
        weights = class_weights_from_counts(dict(counts), CLASSES).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:  # focal
        counts = Counter(labels_str)
        weights = class_weights_from_counts(dict(counts), CLASSES).to(device)
        criterion = FocalLoss(gamma=2.0, alpha=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    ckpt_path = Path("checkpoints") / f"{args.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, streams, device, optimizer, criterion)
        val_metrics = run_epoch(model, val_loader, streams, device, optimizer=None, criterion=criterion)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{args.tag}] epoch {epoch:2d}/{MAX_EPOCHS}  lr={lr_now:.2e}  "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']*100:.2f}%  "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']*100:.2f}%"
        )
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
        )
        if val_metrics["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
                print(f"[{args.tag}] Early stopping at epoch {epoch}.")
                break

    elapsed = time.time() - t0
    print(f"[{args.tag}] Training done in {elapsed/60:.1f} min. Best val_loss={best_val_loss:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, streams, device, optimizer=None, criterion=criterion)
    report = format_report(test_metrics, title=f"{args.tag} - TEST SET")
    print("\n" + report)

    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.tag}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(results_dir / f"{args.tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(results_dir / f"{args.tag}_report.txt", "w") as f:
        f.write(report + "\n")
    print(f"Saved: checkpoints/{args.tag}_best.pt, results/tables/{args.tag}_*")


if __name__ == "__main__":
    main()
