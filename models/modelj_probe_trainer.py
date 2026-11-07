#!/usr/bin/env python3
"""ProbeX/MVProbe training on locally generated Model-J ResNet variants.

This mirrors the original Model-J discriminative protocol:
- one ResNet weight matrix at a time;
- 100-way multi-label target (50 active CIFAR100 classes per target model);
- BCEWithLogitsLoss;
- element-wise binary accuracy at threshold 0.5.

Layer selection should be done on validation only with --skip_test_eval, then the
selected layer can be rerun on several seeds with test evaluation enabled.
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
from safetensors.torch import safe_open
from torch.utils.data import DataLoader

from models.logging_utils import print_eval, print_run_config, print_seed_result
from models.modelj_local_dataset import LocalModelJLayerDataset, list_model_files
from models.mvprobe import ProbeXClassification as MVProbeClassification
from models.probex import ProbeXClassification as OriginalProbeXClassification


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def modelj_probe_layers(root: str | Path, architecture: str) -> list[str]:
    """Return the same ResNet weight family used by the original ProbeX benchmark.

    The official ResNet layer list contains the classifier weight plus convolution
    weights, while excluding normalization scale vectors. Safetensors keys are
    sorted explicitly so layer indices are stable across runs.
    """
    first = list_model_files(root, architecture, "train")[0]
    with safe_open(str(first), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    layers = sorted(
        k for k in keys
        if k == "classifier.1.weight" or k.endswith(".convolution.weight")
    )
    if not layers:
        raise RuntimeError(
            f"No ProbeX-compatible ResNet layers found in {first}. "
            "Expected classifier.1.weight and/or *.convolution.weight."
        )
    return layers


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        loss_sum += float(loss)
        pred = (torch.sigmoid(logits) > 0.5).to(y.dtype)
        correct += int((pred == y).sum())
        total += y.numel()
        samples += y.shape[0]
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data/classification/modelj_cifar100_resnet",
        help="Root containing <architecture>/{train,val,test}/model_idx_*/model.safetensors",
    )
    ap.add_argument("--architecture", default="resnet18")
    ap.add_argument("--model_variant", choices=["probex", "mvprobe"], required=True)
    ap.add_argument("--layer_index", type=int, required=True)
    ap.add_argument("--n_probes", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--rep_dim", type=int, default=512)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument(
        "--eval_every",
        type=int,
        default=25,
        help="Official code evaluates after epoch 1 and then every N epochs.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_test_eval", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.eval_every < 1:
        raise ValueError("--eval_every must be >= 1")

    fix_seed(args.seed)
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = modelj_probe_layers(args.root, args.architecture)
    if not 0 <= args.layer_index < len(layers):
        raise ValueError(
            f"layer_index={args.layer_index} invalid; available indices are 0..{len(layers)-1}"
        )
    layer_name = layers[args.layer_index]

    train_ds = LocalModelJLayerDataset(args.root, args.architecture, "train", layer_name)
    val_ds = LocalModelJLayerDataset(args.root, args.architecture, "val", layer_name)
    test_ds = None if args.skip_test_eval else LocalModelJLayerDataset(
        args.root, args.architecture, "test", layer_name
    )

    first_x, first_y = train_ds[0]
    matrix_shape = tuple(int(x) for x in first_x.shape)
    if len(matrix_shape) != 2:
        raise RuntimeError(f"Expected a 2-D probing matrix, got {matrix_shape} for {layer_name}")
    if tuple(first_y.shape) != (100,):
        raise RuntimeError(f"Expected 100-D Model-J target, got {tuple(first_y.shape)}")

    g = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = None if test_ds is None else DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model_cls = MVProbeClassification if args.model_variant == "mvprobe" else OriginalProbeXClassification
    model = model_cls(
        input_shape=matrix_shape,
        n_probes=args.n_probes,
        proj_dim=args.proj_dim,
        rep_dim=args.rep_dim,
        n_classes=100,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_params = sum(p.numel() for p in model.parameters())
    method = "MVProbe" if args.model_variant == "mvprobe" else "ProbeX"
    print_run_config(
        method=method,
        task="classification",
        dataset=f"Model-J CIFAR100 / {args.architecture} / layer {args.layer_index}",
        seed=args.seed,
        experiment=out_dir.name,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=(0 if test_ds is None else len(test_ds)),
        probes=args.n_probes,
        parameters=total_params,
        trainable=total_params,
        device=str(device),
    )
    print(
        f"Layer: {args.layer_index}/{len(layers)-1} {layer_name} -> X{matrix_shape}; "
        f"proj_dim={args.proj_dim} rep_dim={args.rep_dim} lr={args.lr:g} "
        f"bs={args.batch_size} wd={args.weight_decay:g} epochs={args.epochs} "
        f"eval_every={args.eval_every}",
        flush=True,
    )

    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_train_loss = float("inf")
    best_train_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_targets = y.numel()
                train_loss_sum += float(loss) * batch_targets
                pred = (torch.sigmoid(logits) > 0.5).to(y.dtype)
                train_correct += int((pred == y).sum())
                train_total += batch_targets

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Match the public ProbeX/MVProbe code: eval after epoch 1, 26, 51, ...
        do_eval = (epoch - 1) % args.eval_every == 0
        if not do_eval:
            continue

        val = evaluate(model, val_loader, device)
        is_best = val["acc"] > best_val_acc
        if is_best:
            best_val_acc = val["acc"]
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start
        completed = max(epoch, 1)
        remaining = elapsed / completed * max(args.epochs - epoch, 0)
        print_eval(
            task="classification",
            step=epoch,
            epoch=epoch,
            train_loss=train_loss,
            val_value=val["acc"],
            test_value=None,
            elapsed=elapsed,
            remaining=remaining,
            new_best=is_best,
        )
        print(f"  train_binary_acc={train_acc:.4f} val_loss={val['loss']:.6f}", flush=True)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    summary = {
        "method": method,
        "model_variant": args.model_variant,
        "task": "modelj_multilabel_classification",
        "dataset": "Model-J CIFAR100",
        "architecture": args.architecture,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "layer_name": layer_name,
        "n_layers": len(layers),
        "matrix_shape": list(matrix_shape),
        "n_probes": args.n_probes,
        "proj_dim": args.proj_dim,
        "rep_dim": args.rep_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "params": total_params,
        "best_train_loss": best_train_loss,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "skip_test_eval": bool(args.skip_test_eval),
        "metric": "elementwise binary accuracy at sigmoid threshold 0.5",
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

    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, out_dir / "summary.json")


if __name__ == "__main__":
    main()
