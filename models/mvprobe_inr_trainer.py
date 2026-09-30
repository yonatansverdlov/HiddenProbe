#!/usr/bin/env python3
"""INR classification with the released MVProbe four-view encoder.

The trainer consumes one selected hidden Linear weight matrix from each INR.
The output layer is deliberately excluded from the layer sweep. Hyperparameter
selection can pass --skip_test_eval so the held-out test split is not loaded or
evaluated until the final selected configuration is rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import _remap_siren_keys
from models.logging_utils import print_eval, print_run_config, print_seed_result
from models.mvprobe import ProbeXClassification


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_split_json(data_dir: str, split_json: str) -> Path:
    p = Path(split_json).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (Path(data_dir) / p).resolve()


def _sorted_weight_keys(state_dict):
    state_dict = _remap_siren_keys(state_dict)
    keys = [k for k in state_dict if k.startswith("seq.") and k.endswith(".weight")]
    return sorted(keys, key=lambda k: int(k.split(".")[1]))


class INRWeightLayerDataset(Dataset):
    """One selected hidden SIREN weight matrix per INR."""

    def __init__(self, data_dir: str, split_json: str, split: str, layer_index: int):
        self.data_dir = Path(data_dir).expanduser().resolve()
        split_path = _resolve_split_json(data_dir, split_json)
        with split_path.open() as f:
            record = json.load(f)[split]

        self.paths = [self.data_dir / Path(p) for p in record["path"]]
        self.labels = np.asarray(record["label"], dtype=np.int64)
        if not self.paths:
            raise RuntimeError(f"Empty {split} split in {split_path}")

        first = torch.load(self.paths[0], map_location="cpu", weights_only=False)
        if "label" in first:
            first = dict(first)
            first.pop("label")
        first = _remap_siren_keys(first)
        all_weight_keys = _sorted_weight_keys(first)
        if len(all_weight_keys) < 2:
            raise RuntimeError(f"Expected at least one hidden layer and one output layer; got {all_weight_keys}")

        # Exclude the final RGB/output Linear weight from every sweep.
        self.hidden_weight_keys = all_weight_keys[:-1]
        if not 0 <= layer_index < len(self.hidden_weight_keys):
            raise ValueError(
                f"layer_index={layer_index} invalid; hidden layers are "
                f"0..{len(self.hidden_weight_keys)-1}: {self.hidden_weight_keys}"
            )
        self.layer_index = int(layer_index)
        self.layer_name = self.hidden_weight_keys[layer_index]

        sample = first[self.layer_name]
        if sample.ndim != 2:
            raise RuntimeError(
                f"MVProbe expects a 2-D Linear weight; {self.layer_name} has {tuple(sample.shape)}"
            )
        self.matrix_shape = tuple(sample.shape)

        # Each sweep launches many short processes. Re-reading tens of thousands
        # of tiny INR checkpoints for every configuration would dominate runtime,
        # so materialize the selected matrix once per layer and split.
        cache_root = self.data_dir / ".mvprobe_cache" / split_path.stem
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / f"layer{self.layer_index}_{split}.pt"
        if cache_file.exists():
            cached = torch.load(cache_file, map_location="cpu", weights_only=False)
            if (
                cached.get("layer_name") == self.layer_name
                and tuple(cached.get("matrix_shape", ())) == self.matrix_shape
                and len(cached["labels"]) == len(self.labels)
            ):
                self.matrices = cached["matrices"].float()
                self.labels = cached["labels"].numpy().astype(np.int64, copy=False)
                return

        matrices = []
        for i, path in enumerate(self.paths):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            if "label" in sd:
                sd = dict(sd)
                sd.pop("label")
            sd = _remap_siren_keys(sd)
            x = sd[self.layer_name].detach().float()
            if tuple(x.shape) != self.matrix_shape:
                raise RuntimeError(
                    f"Inconsistent shape for {path}:{self.layer_name}; "
                    f"expected {self.matrix_shape}, got {tuple(x.shape)}"
                )
            matrices.append(x)
            if (i + 1) % 5000 == 0:
                print(
                    f"[mvprobe-cache] {split} layer {self.layer_index}: "
                    f"{i + 1:,}/{len(self.paths):,}",
                    flush=True,
                )

        self.matrices = torch.stack(matrices, dim=0)
        label_tensor = torch.from_numpy(self.labels.copy()).long()
        tmp = cache_file.with_suffix(".tmp")
        torch.save(
            {
                "layer_name": self.layer_name,
                "matrix_shape": self.matrix_shape,
                "matrices": self.matrices,
                "labels": label_tensor,
            },
            tmp,
        )
        os.replace(tmp, cache_file)
        print(f"[mvprobe-cache] wrote {cache_file}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        count += y.numel()
    return {
        "loss": loss_sum / max(count, 1),
        "acc": correct / max(count, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--n_classes", type=int, required=True)
    ap.add_argument("--layer_index", type=int, required=True)
    ap.add_argument("--n_probes", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--rep_dim", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=5)
    ap.add_argument("--plateau_min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_test_eval", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    fix_seed(args.seed)
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = INRWeightLayerDataset(args.data_dir, args.split_json, "train", args.layer_index)
    val_ds = INRWeightLayerDataset(args.data_dir, args.split_json, "val", args.layer_index)
    test_ds = None if args.skip_test_eval else INRWeightLayerDataset(
        args.data_dir, args.split_json, "test", args.layer_index
    )

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers)
        if test_ds is not None else None
    )

    model = ProbeXClassification(
        input_shape=train_ds.matrix_shape,
        n_probes=args.n_probes,
        proj_dim=args.proj_dim,
        rep_dim=args.rep_dim,
        n_classes=args.n_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print_run_config(
        method="MVProbe",
        task="classification",
        dataset=f"{args.dataset_name} / layer {args.layer_index}",
        seed=args.seed,
        experiment=Path(args.out_dir).name,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=(0 if test_ds is None else len(test_ds)),
        probes=args.n_probes,
        parameters=total_params,
        trainable=total_params,
        device=str(device),
    )
    print(
        f"Layer: {args.layer_index} {train_ds.layer_name} -> X{train_ds.matrix_shape}; "
        f"proj_dim={args.proj_dim} rep_dim={args.rep_dim} lr={args.lr:g} "
        f"bs={args.batch_size} wd={args.weight_decay:g}",
        flush=True,
    )

    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.numel()
            train_count += y.numel()

        val = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val["acc"])
        is_best = val["acc"] > best_val_acc
        if is_best:
            best_val_acc = val["acc"]
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start
        per_epoch = elapsed / epoch
        print_eval(
            task="classification",
            step=epoch,
            epoch=epoch,
            train_loss=train_loss_sum / max(train_count, 1),
            val_value=val["acc"],
            test_value=None,
            elapsed=elapsed,
            remaining=per_epoch * max(0, args.epochs - epoch),
            new_best=is_best,
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    summary = {
        "method": "MVProbe",
        "task": "classification",
        "dataset": args.dataset_name,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "layer_name": train_ds.layer_name,
        "matrix_shape": list(train_ds.matrix_shape),
        "n_probes": args.n_probes,
        "proj_dim": args.proj_dim,
        "rep_dim": args.rep_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_min_lr": args.plateau_min_lr,
        "epochs": args.epochs,
        "params": total_params,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "skip_test_eval": bool(args.skip_test_eval),
    }

    if test_loader is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test = evaluate(model, test_loader, device)
        summary["final_test_acc"] = test["acc"]
        summary["final_test_loss"] = test["loss"]
        print_seed_result(
            task="classification",
            seed=args.seed,
            best_epoch=best_epoch,
            best_step=best_epoch,
            val_value=best_val_acc,
            test_value=test["acc"],
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
