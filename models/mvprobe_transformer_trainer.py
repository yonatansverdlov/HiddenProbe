#!/usr/bin/env python3
"""MNIST-Transformers model-accuracy regression with the official MVProbe four-view encoder.

Data protocol is shared with HiddenProbe/ProbeGen:
  data/mnist_transformer/tpcache_mnist_{train,val,test}_cut00_s0.pt
  Filter at threshold 0, split 70/15/15 using seed 0, epoch-75 models.

During hyperparameter selection, pass --skip_test_eval. Test is then never loaded
or evaluated. Final runs select the best epoch by full validation Kendall tau and
touch test exactly once after restoring that best validation checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path

# Allow direct execution as: python models/mvprobe_svhn_trainer.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import make_split
from models.logging_utils import print_eval, print_run_config, print_seed_result
from models.progress import show_progress
from models.metrics_cnn import kendall_tau_b
from models.mvprobe import ProbeXRegression as MVProbeRegression
from models.probex import ProbeXRegression as OriginalProbeXRegression


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Selected weight matrices from both transformer blocks, for single-layer
# ProbeX and MVProbe baselines. No final classifier weights are included.
TRANSFORMER_WEIGHT_KEYS = (
    "encoder.0.attention.queries.weight",
    "encoder.0.attention.values.weight",
    "encoder.0.fc1.weight",
    "encoder.1.attention.queries.weight",
    "encoder.1.attention.values.weight",
    "encoder.1.fc1.weight",
)


class TransformerWeightLayerDataset(Dataset):
    """Cache one transformer weight matrix per network for the fixed threshold-0 split.

    The source is the exact cache built by:
        python main.py transformer cache --dataset mnist --seed 0 --cut_off 0 --data_root data

    During sweeps, load ONLY the selected layer cache, not each full transformer zoo.
    """

    def __init__(self, data_dir: str, split_csv: str, split: str, layer_index: int):
        from models.transformer.intake import load_zoo_cached

        if not 0 <= layer_index < len(TRANSFORMER_WEIGHT_KEYS):
            raise ValueError(f"layer_index must be 0..{len(TRANSFORMER_WEIGHT_KEYS)-1}")

        self.layer_index = int(layer_index)
        self.layer_name = TRANSFORMER_WEIGHT_KEYS[self.layer_index]
        cache_dir = Path(data_dir) / "weight_probe_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"mnist_transformer_cut00_L{self.layer_index}_{split}.pt"

        if cache_file.exists():
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            if payload["layer_name"] != self.layer_name:
                raise RuntimeError(f"Unexpected cache key in {cache_file}")
            self.matrices = payload["matrices"]
            self.targets = payload["labels"]
        else:
            zoo = load_zoo_cached(data_dir, "mnist", split, seed=0, cut_off=0.0)
            if zoo is None:
                raise FileNotFoundError(
                    "Missing MNIST-Transformers cache. Run "
                    "'python main.py transformer cache --dataset mnist --seed 0 "
                    "--cut_off 0 --data_root data' first."
                )
            layer_label = f"Transformer weight cache ({split}, layer {self.layer_index})"
            show_progress(layer_label, 0, len(zoo))
            matrices = []
            for i, item in enumerate(zoo, 1):
                matrices.append(item["params"][self.layer_name].detach().float().cpu())
                show_progress(layer_label, i, len(zoo))
            self.matrices = torch.stack(matrices)
            self.targets = torch.tensor(
                [float(item["label"]) for item in zoo], dtype=torch.float32
            )
            payload = {"layer_name": self.layer_name, "matrices": self.matrices,
                       "labels": self.targets}
            tmp_file = cache_file.with_suffix(".tmp")
            torch.save(payload, tmp_file)
            os.replace(tmp_file, cache_file)

        if self.matrices.ndim != 3:
            raise ValueError(f"Weight matrices must be [n, h, w], got {self.matrices.shape}")
        self.matrix_shape = tuple(self.matrices.shape[1:])
        if self.matrices.shape[0] != self.targets.numel():
            raise ValueError("Mismatched matrix and target counts")
        print(f"[transformer-weights] {split}: {len(self)} models, "
              f"layer={self.layer_name}, shape={self.matrix_shape}", flush=True)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.matrices[idx], self.targets[idx]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    loss_sum = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss_sum += F.mse_loss(pred, y, reduction="sum").item()
        count += y.numel()
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return {
        "mse": loss_sum / max(count, 1),
        "tau": kendall_tau_b(pred, target),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_variant", choices=["mvprobe", "probex"], default="mvprobe")
    ap.add_argument("--dataset_name", default="MNIST-Transformers", help="Human-readable model-zoo name")
    ap.add_argument("--split_csv", default="unused", help="Unused compatibility flag")
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
    ap.add_argument("--epochs", type=int, default=30)
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

    train_ds = TransformerWeightLayerDataset(args.data_dir, args.split_csv, "train", args.layer_index)
    val_ds = TransformerWeightLayerDataset(args.data_dir, args.split_csv, "val", args.layer_index)
    test_ds = None
    if not args.skip_test_eval:
        test_ds = TransformerWeightLayerDataset(args.data_dir, args.split_csv, "test", args.layer_index)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_ds is not None else None
    )

    model_cls = MVProbeRegression if args.model_variant == "mvprobe" else OriginalProbeXRegression
    model = model_cls(
        input_shape=train_ds.matrix_shape,
        n_probes=args.n_probes,
        proj_dim=args.proj_dim,
        rep_dim=args.rep_dim,
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
        method=("MVProbe" if args.model_variant == "mvprobe" else "ProbeX"),
        task="regression",
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
        f"bs={args.batch_size} wd={args.weight_decay:g} scheduler={args.scheduler} "
        f"factor={args.plateau_factor:g} patience={args.plateau_patience} "
        f"min_lr={args.plateau_min_lr:g}",
        flush=True,
    )

    best_val_tau = -float("inf")
    best_val_mse = float("inf")
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
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.numel()
            train_count += y.numel()

        val = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val["tau"])
        is_best = val["tau"] > best_val_tau
        if is_best:
            best_val_tau = val["tau"]
            best_val_mse = val["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start
        per_epoch = elapsed / epoch
        print_eval(
            task="regression",
            step=epoch,
            epoch=epoch,
            train_loss=train_loss_sum / max(train_count, 1),
            val_value=val["tau"],
            test_value=None,
            elapsed=elapsed,
            remaining=per_epoch * max(0, args.epochs - epoch),
            new_best=is_best,
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    summary = {
        "method": ("MVProbe" if args.model_variant == "mvprobe" else "ProbeX"),
        "model_variant": args.model_variant,
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
        "best_val_tau": best_val_tau,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "skip_test_eval": bool(args.skip_test_eval),
    }

    if test_loader is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test = evaluate(model, test_loader, device)
        summary["final_test_tau"] = test["tau"]
        summary["final_test_mse"] = test["mse"]
        print_seed_result(
            task="regression",
            seed=args.seed,
            best_epoch=best_epoch,
            best_step=best_epoch,
            val_value=best_val_tau,
            test_value=test["tau"],
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
