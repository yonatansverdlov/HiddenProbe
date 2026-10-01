#!/usr/bin/env python3
"""SVHN model-accuracy regression with the official MVProbe four-view encoder.

Data protocol is shared with HiddenProbe/ProbeGen:
  data.regression.svhn/{weights.npy,metrics.csv.gz,layout.csv,split.csv}
  data.make_split() supplies the exact official NFN final-checkpoint split.

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
from models.metrics_cnn import kendall_tau_b
from models.mvprobe import ProbeXRegression as MVProbeRegression
from models.probex import ProbeXRegression as OriginalProbeXRegression


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SVHNWeightLayerDataset(Dataset):
    """One selected Small-CNN-Zoo weight tensor, converted to MVProbe's matrix X."""

    def __init__(self, data_dir: str, split_csv: str, split: str, layer_index: int):
        self.data_dir = data_dir
        self.weights = np.load(os.path.join(data_dir, "weights.npy"), mmap_mode="r")
        layout = pd.read_csv(os.path.join(data_dir, "layout.csv"))

        # Sweep only learned weight tensors, never biases.
        kernels = layout[layout["varname"].str.endswith("/kernel:0")].reset_index(drop=True)
        if not 0 <= layer_index < len(kernels):
            raise ValueError(
                f"layer_index={layer_index} is invalid; available weight layers are 0..{len(kernels)-1}"
            )
        row = kernels.iloc[layer_index]
        self.layer_index = int(layer_index)
        self.layer_name = str(row["varname"])
        self.start = int(row["start_idx"])
        self.end = int(row["end_idx"])
        self.shape = tuple(ast.literal_eval(str(row["shape"])))

        split_path = Path(split_csv).expanduser()
        if split_path.is_absolute():
            split_arg = str(split_path)
        elif split_path.exists():
            # A repo-root relative path such as data/regression/svhn/split.csv.
            split_arg = str(split_path.resolve())
        else:
            # A filename such as split.csv; let make_split resolve it under data_dir.
            split_arg = split_csv

        split_info = make_split(data_dir=data_dir, split_csv=split_arg)[split]
        self.rows = np.asarray(split_info["rows"], dtype=np.int64)
        self.targets = np.asarray(split_info["scores"], dtype=np.float32)

        sample = self._matrix_from_flat(self.weights[int(self.rows[0])])
        self.matrix_shape = tuple(sample.shape)

    def __len__(self):
        return len(self.rows)

    def _matrix_from_flat(self, flat) -> torch.Tensor:
        # Zoo layout is TensorFlow: conv kernels HWIO, dense kernels (in,out).
        w = np.asarray(flat[self.start:self.end], dtype=np.float32).reshape(self.shape)

        if len(self.shape) == 4:
            # Convert to the target model's PyTorch OIHW tensor, then use the
            # official MVProbe ResNet convention. The upstream code calls
            # weight.squeeze().reshape(-1, weight.shape[0]); our grayscale first
            # conv has Cin=1, so we intentionally test dim==4 before squeezing.
            w = np.ascontiguousarray(w.transpose(3, 2, 0, 1))
            t = torch.from_numpy(w)
            t = t.reshape(-1, t.shape[0]).squeeze()
        elif len(self.shape) == 2:
            # Target PyTorch Linear stores (out,in), matching ordinary Model
            # Jungle 2-D weights consumed by the official dataset.
            t = torch.from_numpy(np.ascontiguousarray(w.T))
        elif len(self.shape) == 1:
            t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(1)
        else:
            t = torch.from_numpy(np.ascontiguousarray(w.squeeze()))
            if t.ndim > 2:
                t = t.reshape(t.shape[0], -1)

        if t.ndim != 2:
            raise RuntimeError(
                f"MVProbe requires a matrix; {self.layer_name} produced shape {tuple(t.shape)}"
            )
        return t.float()

    def __getitem__(self, idx):
        matrix = self._matrix_from_flat(self.weights[int(self.rows[idx])])
        return matrix, torch.tensor(self.targets[idx], dtype=torch.float32)


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
    ap.add_argument("--split_csv", default="split.csv")
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

    train_ds = SVHNWeightLayerDataset(args.data_dir, args.split_csv, "train", args.layer_index)
    val_ds = SVHNWeightLayerDataset(args.data_dir, args.split_csv, "val", args.layer_index)
    test_ds = None
    if not args.skip_test_eval:
        test_ds = SVHNWeightLayerDataset(args.data_dir, args.split_csv, "test", args.layer_index)

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
        dataset=f"SVHN / layer {args.layer_index}",
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
        "dataset": "SVHN",
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
