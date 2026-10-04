#!/usr/bin/env python3
"""Generate matched CIFAR100 Model-J variants with smaller ResNets.

For every row in ProbeX/Model-J (ResNet subset), this script keeps the original:
  * train/val/test model split
  * 50 chosen CIFAR100 classes
  * model seed
  * learning rate / scheduler / epochs / batch size / weight decay
  * random-crop and random-flip choices

Only the target-network architecture is changed.

Output layout:
  <output_root>/<architecture>/<split>/model_idx_XXXX/
      model.safetensors
      metadata.json
      config.json

The layout is intentionally simple so weight-space methods can read it directly.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, get_scheduler


ARCHITECTURES = {
    "resnet18": "microsoft/resnet-18",
    "resnet50": "microsoft/resnet-50",
    "resnet101": "microsoft/resnet-101",
}

DEFAULT_OUTPUT_ROOT = "data/classification/modelj_cifar100_resnet"
DEFAULT_CIFAR_ROOT = "data/raw/cifar100"
SOURCE_DATASET = "ProbeX/Model-J"
SOURCE_SUBSET = "ResNet"
TRAIN_PER_CLASS = 425
VAL_PER_CLASS = 75
N_SELECTED_CLASSES = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recreate the Model-J CIFAR100 50-of-100 task with a chosen ResNet."
    )
    p.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--cifar_root", default=DEFAULT_CIFAR_ROOT)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                   default=["train", "val", "test"])
    p.add_argument("--model_idx", type=int, nargs="*", default=None,
                   help="Optional exact Model-J model indices to generate.")
    p.add_argument("--start_position", type=int, default=0,
                   help="Start position after filtering/sorting source rows.")
    p.add_argument("--end_position", type=int, default=None,
                   help="Exclusive end position after filtering/sorting source rows.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the selected source rows across independent jobs.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="This job handles positions where position %% num_shards == shard_id.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional limit for smoke tests.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--split_seed", type=int, default=2025,
                   help="Fixed per-class image split seed shared by every architecture.")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Used only for scheduler names containing 'warmup'.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if a complete output already exists.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_targets(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        out = [str(x) for x in value]
    elif isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(parsed, (list, tuple)):
            out = [str(x) for x in parsed]
        else:
            raise ValueError(f"Could not parse dataset_chosen_targets={value!r}")
    else:
        raise TypeError(f"Unsupported dataset_chosen_targets type: {type(value)}")
    if len(out) != N_SELECTED_CLASSES:
        raise ValueError(f"Expected 50 selected classes, got {len(out)}")
    return out


def source_rows(splits: list[str]) -> list[dict[str, Any]]:
    ds = load_dataset(SOURCE_DATASET, SOURCE_SUBSET)
    rows: list[dict[str, Any]] = []
    for split in splits:
        if split not in ds:
            raise KeyError(f"Source dataset has no split {split!r}")
        for item in ds[split]:
            row = dict(item)
            row["split"] = split
            rows.append(row)
    rows.sort(key=lambda x: int(x["model_idx"]))
    return rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.model_idx:
        wanted = set(args.model_idx)
        rows = [r for r in rows if int(r["model_idx"]) in wanted]
        found = {int(r["model_idx"]) for r in rows}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Requested model_idx values not found: {missing}")

    rows = rows[args.start_position:args.end_position]
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.limit is not None:
        rows = rows[:args.limit]
    return rows


class CIFARSubset(Dataset):
    def __init__(
        self,
        base: CIFAR100,
        indices: list[int],
        label_map: dict[int, int],
        transform,
    ) -> None:
        self.base = base
        self.indices = indices
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, original_label = self.base[self.indices[item]]
        image = self.transform(image)
        return image, self.label_map[int(original_label)]


def make_class_indices(
    train_base: CIFAR100,
    test_base: CIFAR100,
    class_ids: list[int],
    split_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    targets = np.asarray(train_base.targets)
    test_targets = np.asarray(test_base.targets)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in class_ids:
        idx = np.flatnonzero(targets == class_id)
        if len(idx) != 500:
            raise ValueError(f"CIFAR100 class {class_id} has {len(idx)} train images, expected 500")
        rng = np.random.default_rng(split_seed + int(class_id))
        idx = idx.copy()
        rng.shuffle(idx)
        train_idx.extend(idx[:TRAIN_PER_CLASS].tolist())
        val_idx.extend(idx[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS].tolist())

        t_idx = np.flatnonzero(test_targets == class_id)
        if len(t_idx) != 100:
            raise ValueError(f"CIFAR100 class {class_id} has {len(t_idx)} test images, expected 100")
        test_idx.extend(t_idx.tolist())

    return train_idx, val_idx, test_idx


def processor_settings(base_model: str) -> tuple[int, list[float], list[float]]:
    processor = AutoImageProcessor.from_pretrained(base_model)
    size = 224
    crop_size = getattr(processor, "crop_size", None)
    proc_size = getattr(processor, "size", None)
    if isinstance(crop_size, dict):
        size = int(crop_size.get("height", crop_size.get("width", size)))
    elif isinstance(proc_size, dict):
        size = int(proc_size.get("shortest_edge", proc_size.get("height", size)))
    mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
    return size, mean, std


def make_transforms(
    base_model: str,
    random_crop: bool,
    random_flip: bool,
):
    size, mean, std = processor_settings(base_model)
    train_ops = []
    if random_crop:
        train_ops.append(RandomCrop(32, padding=4))
    if random_flip:
        train_ops.append(RandomHorizontalFlip())
    train_ops.extend([Resize((size, size)), ToTensor(), Normalize(mean=mean, std=std)])
    eval_ops = [Resize((size, size)), ToTensor(), Normalize(mean=mean, std=std)]
    return Compose(train_ops), Compose(eval_ops), size


def make_loaders(
    train_base: CIFAR100,
    test_base: CIFAR100,
    selected_names: list[str],
    batch_size: int,
    num_workers: int,
    split_seed: int,
    random_crop: bool,
    random_flip: bool,
    base_model: str,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[int, int], int]:
    name_to_id = {name: i for i, name in enumerate(train_base.classes)}
    missing = sorted(set(selected_names) - set(name_to_id))
    if missing:
        raise ValueError(f"Unknown CIFAR100 classes: {missing}")

    class_ids = sorted(name_to_id[name] for name in selected_names)
    # Match Model-J: keep the original CIFAR100 label IDs and a 100-way head,
    # even though only 50 classes are present in this model's training data.
    label_map = {class_id: class_id for class_id in class_ids}
    train_idx, val_idx, test_idx = make_class_indices(
        train_base, test_base, class_ids, split_seed
    )
    train_tf, eval_tf, image_size = make_transforms(
        base_model, random_crop, random_flip
    )

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        CIFARSubset(train_base, train_idx, label_map, train_tf),
        shuffle=True,
        drop_last=False,
        **kwargs,
    )
    val_loader = DataLoader(
        CIFARSubset(train_base, val_idx, label_map, eval_tf),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    test_loader = DataLoader(
        CIFARSubset(test_base, test_idx, label_map, eval_tf),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader, test_loader, label_map, image_size


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(pixel_values=images, labels=labels)
        batch = labels.numel()
        total_loss += float(out.loss) * batch
        total_correct += int((out.logits.argmax(dim=-1) == labels).sum())
        total += batch
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def is_complete(model_dir: Path, architecture: str, model_idx: int) -> bool:
    weights = model_dir / "model.safetensors"
    metadata = model_dir / "metadata.json"
    if not weights.is_file() or not metadata.is_file():
        return False
    try:
        data = json.loads(metadata.read_text())
    except Exception:
        return False
    return (
        data.get("status") == "complete"
        and data.get("architecture") == architecture
        and int(data.get("model_idx", -1)) == model_idx
        and "test_accuracy" in data
    )


def scheduler_from_row(
    optimizer: AdamW,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float,
):
    name = str(scheduler_name)
    warmup_steps = int(round(total_steps * warmup_ratio)) if "warmup" in name else 0
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    ), warmup_steps


def cpu_state_dict(model, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        out[key] = tensor
    return out


def train_one(
    row: dict[str, Any],
    args: argparse.Namespace,
    train_base: CIFAR100,
    test_base: CIFAR100,
    position: int,
    total_positions: int,
) -> None:
    model_idx = int(row["model_idx"])
    split = str(row["split"])
    architecture = args.architecture
    base_model = ARCHITECTURES[architecture]
    model_dir = Path(args.output_root) / architecture / split / f"model_idx_{model_idx:04d}"

    print()
    print("=" * 88)
    print(f"MODEL {position}/{total_positions} | architecture={architecture} | "
          f"split={split} | model_idx={model_idx:04d}")
    print("=" * 88)

    if not args.force and is_complete(model_dir, architecture, model_idx):
        print(f"Completed: {model_dir} (skipping)")
        return

    selected_names = parse_targets(row["dataset_chosen_targets"])
    seed = int(row.get("seed", model_idx))
    batch_size = int(row["batch_size"])
    learning_rate = float(row["learning_rate"])
    weight_decay = float(row["weight_decay"])
    epochs = int(row["epochs"])
    scheduler_name = str(row["lr_scheduler"])
    random_crop = bool(row["random_crop"])
    random_flip = bool(row["random_flip"])

    seed_everything(seed)
    train_loader, val_loader, test_loader, label_map, image_size = make_loaders(
        train_base=train_base,
        test_base=test_base,
        selected_names=selected_names,
        batch_size=batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        random_crop=random_crop,
        random_flip=random_flip,
        base_model=base_model,
    )

    expected_steps = epochs * len(train_loader)
    source_max_steps = int(row.get("max_train_steps", expected_steps))
    print(
        f"classes=50 train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)} steps/epoch={len(train_loader)}"
    )
    print(
        f"lr={learning_rate:g} scheduler={scheduler_name} epochs={epochs} "
        f"bs={batch_size} wd={weight_decay:g} crop={random_crop} flip={random_flip}"
    )
    if source_max_steps != expected_steps:
        print(
            f"NOTE: source max_train_steps={source_max_steps}; "
            f"our deterministic split gives {expected_steps} total steps."
        )

    id2label = {i: name for i, name in enumerate(train_base.classes)}
    label2id = {name: i for i, name in id2label.items()}

    config = AutoConfig.from_pretrained(base_model)
    config.num_labels = 100
    config.id2label = id2label
    config.label2id = label2id
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler, warmup_steps = scheduler_from_row(
        optimizer, scheduler_name, expected_steps, args.warmup_ratio
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -math.inf
    best_epoch = -1
    best_state = None
    best_train = None
    best_val = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                out = model(pixel_values=images, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch = labels.numel()
            train_loss_sum += float(loss.detach()) * batch
            train_correct += int((out.logits.detach().argmax(dim=-1) == labels).sum())
            train_total += batch

        train_metrics = {
            "loss": train_loss_sum / max(train_total, 1),
            "accuracy": train_correct / max(train_total, 1),
        }
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"EPOCH {epoch:02d}/{epochs:02d} | "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"elapsed={elapsed/60:.1f}m"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_train = dict(train_metrics)
            best_val = dict(val_metrics)
            best_state = cpu_state_dict(model, torch.float32)

    if best_state is None:
        raise RuntimeError("No best state was recorded")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = torch.float32 if args.save_dtype == "float32" else torch.float16
    state_to_save = {
        k: (v.to(dtype=save_dtype) if v.is_floating_point() else v)
        for k, v in best_state.items()
    }

    selected_original_ids = sorted(
        train_base.class_to_idx[name] for name in selected_names
    )
    metadata = {
        "status": "complete",
        "architecture": architecture,
        "base_model": base_model,
        "source_dataset": SOURCE_DATASET,
        "source_subset": SOURCE_SUBSET,
        "source_hf_model_id": row.get("hf_model_id"),
        "source_modelj_row": row,
        "model_idx": model_idx,
        "split": split,
        "seed": seed,
        "dataset": "CIFAR100",
        "dataset_chosen_targets": selected_names,
        "dataset_chosen_target_ids": selected_original_ids,
        "n_selected_classes": N_SELECTED_CLASSES,
        "classifier_num_labels": 100,
        "train_per_class": TRAIN_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "split_seed": args.split_seed,
        "learning_rate": learning_rate,
        "lr_scheduler": scheduler_name,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "random_crop": random_crop,
        "random_flip": random_flip,
        "image_size": image_size,
        "best_epoch": best_epoch,
        "train_loss": best_train["loss"],
        "train_accuracy": best_train["accuracy"],
        "val_loss": best_val["loss"],
        "val_accuracy": best_val["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "source_max_train_steps": source_max_steps,
        "actual_max_train_steps": expected_steps,
        "save_dtype": args.save_dtype,
        "n_model_params": sum(p.numel() for p in model.parameters()),
    }

    tensor_metadata = {
        "dataset_chosen_targets": repr(selected_names),
        "dataset_chosen_target_ids": repr(selected_original_ids),
        "architecture": architecture,
        "base_model": base_model,
        "model_idx": str(model_idx),
        "split": split,
        "seed": str(seed),
        "source_hf_model_id": str(row.get("hf_model_id", "")),
    }
    save_file(state_to_save, str(model_dir / "model.safetensors"), metadata=tensor_metadata)
    model.config.to_json_file(str(model_dir / "config.json"))
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(
        f"SAVED | best_epoch={best_epoch} val_acc={best_val_acc:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} | {model_dir}"
    )


def main() -> None:
    args = parse_args()
    rows = select_rows(source_rows(args.splits), args)
    if not rows:
        print("No source rows selected.")
        return

    print(f"Source: {SOURCE_DATASET}/{SOURCE_SUBSET}")
    print(f"Architecture: {args.architecture} -> {ARCHITECTURES[args.architecture]}")
    print(f"Selected models in this job: {len(rows)}")
    print(f"Output root: {args.output_root}")
    print(
        f"Image split: {TRAIN_PER_CLASS} train + {VAL_PER_CLASS} val per selected class; "
        f"split_seed={args.split_seed}"
    )

    train_base = CIFAR100(root=args.cifar_root, train=True, download=True)
    test_base = CIFAR100(root=args.cifar_root, train=False, download=True)

    for position, row in enumerate(rows, start=1):
        train_one(
            row=row,
            args=args,
            train_base=train_base,
            test_base=test_base,
            position=position,
            total_positions=len(rows),
        )


if __name__ == "__main__":
    main()
