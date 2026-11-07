"""Reader for locally generated Model-J CIFAR100 ResNet variants.

This intentionally mirrors the label construction used by the official ProbeX
ModelsDatasetDiscriminative class, but reads our simple local layout:
  root/<architecture>/<split>/model_idx_XXXX/model.safetensors
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset


CIFAR100_CLASSES = sorted([
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
])
CLASS_TO_META_ID = {name: i for i, name in enumerate(CIFAR100_CLASSES)}


def list_model_files(root: str | Path, architecture: str, split: str) -> list[Path]:
    base = Path(root) / architecture / split
    if not base.is_dir():
        raise FileNotFoundError(base)
    files = sorted(base.glob("model_idx_*/model.safetensors"))
    if not files:
        raise FileNotFoundError(f"No model.safetensors files under {base}")
    return files


def available_weight_layers(model_file: str | Path) -> list[str]:
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        return [k for k in f.keys() if k.endswith(".weight")]


class LocalModelJLayerDataset(Dataset):
    """Read one named weight tensor and the 100-way class-membership target."""

    def __init__(
        self,
        root: str | Path,
        architecture: str,
        split: str,
        layer_name: str,
        flatten_conv: bool = True,
    ) -> None:
        self.files = list_model_files(root, architecture, split)
        self.layer_name = layer_name
        self.flatten_conv = flatten_conv

        with safe_open(str(self.files[0]), framework="pt", device="cpu") as f:
            if layer_name not in f.keys():
                raise KeyError(
                    f"{layer_name!r} not found. Example available layers: "
                    f"{available_weight_layers(self.files[0])[:20]}"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with safe_open(str(path), framework="pt", device="cpu") as f:
            weight = f.get_tensor(self.layer_name)
            metadata = f.metadata() or {}

        chosen = ast.literal_eval(metadata["dataset_chosen_targets"])
        y = np.zeros(100, dtype=np.float32)
        for class_name in chosen:
            y[CLASS_TO_META_ID[class_name]] = 1.0

        if self.flatten_conv and weight.ndim == 4:
            # Same convention as the official ProbeX ResNet loader:
            # [out, in, kh, kw] -> [in*kh*kw, out]
            weight = weight.reshape(weight.shape[0], -1).T.contiguous()
        elif weight.ndim == 1:
            weight = weight[:, None]
        elif weight.ndim > 2:
            weight = weight.squeeze()

        return weight, torch.from_numpy(y)
