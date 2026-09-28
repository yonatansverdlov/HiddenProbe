import json
import os
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import csv

# Note: Dataset Code was partially inspired by: https://github.com/mkofinas/neural-graphs.git


class CIFAR10INRDataset(torch.utils.data.Dataset):
    """CIFAR-10 SIREN dataset in the NFN/NFT directory format.

    Expected directory structure (under ``dataset_dir``)::

        randinit_smaller_0s/net*.pth
        ...
        randinit_smaller_9s/net*.pth
        randinit_smaller_aug0_0s/net*.pth
        ...
        randinit_smaller_aug9_9s/net*.pth

    NFT's augmented training protocol uses the base realization plus aug0..aug9
    for the *training* images only. Therefore:

        train = 45,000 * (1 + extra_aug) = 495,000 for extra_aug=10
        val   = 5,000 base INRs
        test  = 10,000 base INRs

    We intentionally index exact class directories (``<prefix>_<digit>s``) so the
    base prefix cannot accidentally glob augmented directories.
    """

    _IDX_RE = re.compile(r"net(\d+)\.pth$")
    _LABEL_DIR_RE = re.compile(r"_(\d+)s$")

    def __init__(
        self,
        dataset_dir,
        split,
        extra_aug=10,
        base_prefix="randinit_smaller",
        split_points=(45000, 50000),
        cache_models=False,
        num_classes=10,
        dataset_name="CIFAR-10",
    ):
        self.root = Path(dataset_dir).expanduser().resolve()
        self.split = split
        self.extra_aug = int(extra_aug)
        self.base_prefix = base_prefix
        self.split_points = tuple(split_points)
        self.cache_models = bool(cache_models)
        self.num_classes = int(num_classes)
        self.dataset_name = str(dataset_name)

        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} INR directory does not exist: {self.root}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if self.extra_aug < 0:
            raise ValueError("extra_aug must be >= 0")
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        val_point, test_point = self.split_points
        if split == "train":
            prefixes = [base_prefix] + [f"{base_prefix}_aug{i}" for i in range(self.extra_aug)]
            lo, hi = 0, val_point
        elif split == "val":
            prefixes = [base_prefix]
            lo, hi = val_point, test_point
        else:
            prefixes = [base_prefix]
            lo, hi = test_point, None

        self.samples = []
        self.prefix_counts = {}
        for prefix in prefixes:
            records = self._index_prefix(prefix, lo=lo, hi=hi)
            self.samples.extend(records)
            self.prefix_counts[prefix] = len(records)

        # Keep only a lazy cache by default; caching 495K nn.Modules can consume
        # a very large amount of RAM. Set cache_models=True only if you know you
        # have enough host memory and want to trade RAM for filesystem I/O.
        self.all_data = [None] * len(self.samples) if self.cache_models else None

        expected = None
        if split == "train":
            expected = val_point * (1 + self.extra_aug)
        elif split == "val":
            expected = test_point - val_point

        if expected is not None and len(self.samples) != expected:
            raise RuntimeError(
                f"Unexpected {self.dataset_name} INR {split} size: got {len(self.samples):,}, "
                f"expected {expected:,}. Check that the official NFN/NFT archive "
                f"was fully extracted and that extra_aug={self.extra_aug}."
            )
        if split == "test" and len(self.samples) != 10000:
            raise RuntimeError(
                f"Unexpected {self.dataset_name} INR test size: got {len(self.samples):,}, expected 10,000."
            )

    def _index_prefix(self, prefix, lo, hi):
        records = []

        # Exact class directories only: e.g. randinit_smaller_3s or
        # randinit_smaller_aug7_3s. This avoids the overly-broad
        # ``randinit_smaller_*`` glob from also matching augmentation folders.
        for label in range(self.num_classes):
            class_dir = self.root / f"{prefix}_{label}s"
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing {self.dataset_name} INR directory: {class_dir}. "
                    f"For NFT-exact training with extra_aug=10, aug0..aug9 must exist."
                )

            for path in class_dir.glob("net*.pth"):
                m = self._IDX_RE.search(path.name)
                if m is None:
                    continue
                idx = int(m.group(1))
                if idx < lo or (hi is not None and idx >= hi):
                    continue
                records.append((idx, path, label))

        records.sort(key=lambda t: t[0])

        # The net index is global across classes, so every requested index should
        # appear exactly once for a given realization/prefix.
        expected_n = (hi - lo) if hi is not None else 60000 - lo
        if len(records) != expected_n:
            raise RuntimeError(
                f"Prefix {prefix!r}: found {len(records):,} requested INRs, "
                f"expected {expected_n:,} for indices [{lo}, {hi if hi is not None else 'end'})."
            )

        ids = [idx for idx, _, _ in records]
        if ids != list(range(lo, lo + expected_n)):
            raise RuntimeError(
                f"Prefix {prefix!r}: net indices are not the expected contiguous range "
                f"[{lo}, {lo + expected_n})."
            )

        return [(path, label) for _, path, label in records]

    def __len__(self):
        return len(self.samples)

    def n_classes(self):
        return self.num_classes

    @staticmethod
    def _load_model(path):
        state = torch.load(path, map_location="cpu")

        # Official NFN/NFT CIFAR SIREN:
        #   2 -> 32 -> 32 -> 3,
        # sine(30 * .) after the first two Linears, final layer linear.
        converted_state = {
            "seq.0.weight": state["net.0.linear.weight"],
            "seq.0.bias": state["net.0.linear.bias"],
            "seq.1.weight": state["net.1.linear.weight"],
            "seq.1.bias": state["net.1.linear.bias"],
            "seq.2.weight": state["net.2.weight"],
            "seq.2.bias": state["net.2.bias"],
        }

        model = INR_Network(
            in_features=2,
            n_layers=3,
            hidden_features=32,
            out_features=3,
            output_shift=0.0,  # NFN/NFT CIFAR SIRENs output the raw [-1, 1]-scaled RGB signal.
        )
        model.load_state_dict(converted_state)
        return model

    def __getitem__(self, item):
        if self.all_data is not None and self.all_data[item] is not None:
            return self.all_data[item]

        path, label = self.samples[item]
        sample = (self._load_model(path), label)

        if self.all_data is not None:
            self.all_data[item] = sample
        return sample


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class INR_Network(nn.Module):
    def __init__(self, in_features=2, n_layers=3, hidden_features=32, out_features=1, output_shift=0.5):
        super(INR_Network, self).__init__()
        self.output_shift = float(output_shift)
        self.seq = nn.ModuleList([nn.Linear(in_features, hidden_features)])
        for i in range(n_layers - 2):
            self.seq.append(nn.Linear(hidden_features, hidden_features))
        self.seq.append(nn.Linear(hidden_features, out_features))
        self.activation = Sine(w0=30.0)

    def make_coordinates(self, shape=(28, 28), bs=1, coord_range=(-1, 1)):
        x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
        y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
        x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()
        coordinates = np.stack([x_coordinates, y_coordinates]).T
        coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
        return torch.from_numpy(coordinates).type(torch.float)

    def plot_INR_img(self):
        coords = self.make_coordinates(bs=1).to(self.seq[0].weight.device)
        with torch.no_grad():
            out = self.forward(coords)
        out = out.view(28, 28).cpu().numpy()
        return out

    def forward(self, x):
        for layer in self.seq[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.seq[-1](x)
        return x + self.output_shift

    def get_stats(self, acts, quantiles=[0., 0.25, 0.5, 0.75, 1.]):
        """
        activations: shape (bs, **)
        """
        feats = []
        flat_a = acts.flatten(start_dim=1)
        feats.append(flat_a.mean(dim=1))
        feats.append(flat_a.var(dim=1))
        for q in quantiles:
            feats.append(torch.quantile(flat_a, q, dim=1))
        feats = torch.stack(feats, dim=1)
        return feats

    def forward_and_extract_acts(self, x, max_size=3):
        if max_size >= len(self.seq):
            chosen_layers = list(range(len(self.seq)))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.seq), max_size)]

        all_act_feats = []
        for i, layer in enumerate(self.seq):

            is_last_layer = i == len(self.seq) - 1
            x = layer(x)
            if is_last_layer:
                x = x + self.output_shift

            if i in chosen_layers:
                all_act_feats.append(self.get_stats(x))

            if not is_last_layer:
                x = self.activation(x)

        if max_size >= len(self.seq):
            zero_pad = torch.zeros(max_size - len(all_act_feats), all_act_feats[0].shape[1], device=all_act_feats[0].device)
            all_act_feats.append(zero_pad)

        all_act_feats = torch.cat(all_act_feats, dim=0)
        return x, all_act_feats

    def get_weights_stats(self, max_size=3):
        if max_size >= len(self.seq):
            chosen_layers = list(range(len(self.seq)))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.seq), max_size)]
        all_w_stats = []
        all_layer_types = []
        all_act_types = []
        for i, layer in enumerate(self.seq):
            last_layer = i == len(self.seq) - 1
            if i in chosen_layers:
                layer_stats = torch.cat([self.get_stats(layer.weight.unsqueeze(0)),
                                         self.get_stats(layer.bias.unsqueeze(0))], dim=1)
                all_w_stats.append(layer_stats)
                all_layer_types.append(type(layer).__name__)
                all_act_types.append(type(self.activation).__name__ if not last_layer else 'none')
        if max_size >= len(self.seq) + 1:
            zero_pad = torch.zeros(max_size - len(all_w_stats), all_w_stats[0].shape[1], device=all_w_stats[0].device)
            all_w_stats.append(zero_pad)
            all_layer_types.extend(['none'] * (max_size - len(all_layer_types)))
            all_act_types.extend(['none'] * (max_size - len(all_act_types)))
        all_w_stats = torch.cat(all_w_stats, dim=0)
        return all_w_stats, all_layer_types, all_act_types


def infer_inr_arch(state_dict):
    """Derive INR_Network kwargs from a state_dict of sequential Linears.

    Keys are `seq.{i}.weight` / `seq.{i}.bias`. `n_layers` counts Linears
    (in->h, h->h, ..., h->out), so hidden layers = n_layers - 1. Zoos differ in
    depth/width/out_features: fmnist+mnist INRs are 2->32->32->1 (n_layers=3, 2 hidden);
    the official NFN/NFT CIFAR10 INRs are loaded by CIFAR10INRDataset and are
    2->32->32->3 (n_layers=3, 2 hidden, RGB out).
    """
    ws = sorted(
        ((int(k.split(".")[1]), v) for k, v in state_dict.items()
         if k.startswith("seq.") and k.endswith(".weight")),
        key=lambda t: t[0],
    )
    if not ws:
        raise ValueError(
            f"INR state_dict has no `seq.*.weight` entries; keys={list(state_dict)[:8]}")
    return dict(
        in_features=ws[0][1].shape[1],
        n_layers=len(ws),
        hidden_features=ws[0][1].shape[0],
        out_features=ws[-1][1].shape[0],
    )


class INRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train", cache_models=False):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        self.root = self.splits_path.parent
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            Path(dataset_dir) / Path(p) for p in self.dataset["path"]
        ]

        # Keeping tens of thousands of reconstructed INR nn.Modules resident in
        # RAM is extremely expensive and, with DataLoader workers, also creates
        # a large number of shared-memory mappings. The canonical behavior is
        # therefore lazy/no-cache. Small debugging jobs may opt in explicitly.
        self.cache_models = bool(cache_models)
        self.all_data = (
            [None for _ in range(len(self.dataset["label"]))]
            if self.cache_models
            else None
        )

    def __len__(self):
        return len(self.dataset["label"])

    def n_classes(self):
        return len(set(self.dataset["label"]))

    def __getitem__(self, item):
        if self.cache_models and self.all_data[item] is not None:
            return self.all_data[item]

        path = str(self.dataset["path"][item])
        try:
            state_dict = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {path}")
            raise e

        if "label" in state_dict:
            state_dict.pop("label")
        label = int(self.dataset["label"][item])
        model = INR_Network(**infer_inr_arch(state_dict))
        model.load_state_dict(state_dict)
        sample = (model, label)

        if self.cache_models:
            self.all_data[item] = sample

        return sample

class Generic_CNN_Network(nn.Module):
    def __init__(self, n_layers, channels, kernel_sizes, strides, activations, paddings, out_size=10):
        super(Generic_CNN_Network, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(n_layers)])
        self.activations = nn.ModuleList([self.get_act(act) for act in activations])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], out_size)

    def load_weights(self, weights, biases):
        for layer, w, b in zip(self.layers, weights[:-1], biases[:-1]):
            assert layer.weight.data.shape == w.shape
            assert layer.bias.data.shape == b.shape
            layer.weight.data = w.clone()
            layer.bias.data = b.clone()
        self.fc.weight.data = weights[-1].clone()
        self.fc.bias.data = biases[-1].clone()

    def get_act(self, act_type):
        if act_type == 'relu':
            return nn.ReLU()
        elif act_type == 'gelu':
            return nn.GELU()
        elif act_type == 'sine':
            return Sine(w0=30.0)
        elif act_type == 'tanh':
            return nn.Tanh()
        elif act_type == 'sigmoid':
            return nn.Sigmoid()
        elif act_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif act_type == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Activation type {act_type} not recognized.")

    def get_stats(self, acts, quantiles=[0., 0.25, 0.5, 0.75, 1.]):
        """
        activations: shape (bs, **)
        """
        feats = []
        flat_a = acts.flatten(start_dim=1)
        feats.append(flat_a.mean(dim=1))
        feats.append(flat_a.var(dim=1))
        for q in quantiles:
            feats.append(torch.quantile(flat_a, q, dim=1))
        feats = torch.stack(feats, dim=1)
        return feats

    def forward_and_extract_acts(self, x, max_size=3):
        chosen_layers = [int(l) for l in np.linspace(0, len(self.layers), max_size)]
        all_act_feats = [self.get_stats(x)]
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            if i in chosen_layers:
                all_act_feats.append(x.mean(dim=(2, 3)))
            x = act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        all_act_feats.append(x)
        # print([a.shape for a in all_act_feats])
        all_act_feats = torch.cat(all_act_feats, dim=1)
        return x, all_act_feats

    def get_weights_stats(self, max_size=6):
        if max_size >= len(self.layers) + 1:
            chosen_layers = list(range(len(self.layers) + 1))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.layers) + 1, max_size)]
        all_w_stats = []
        all_layer_types = []
        all_act_types = []
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            if i in chosen_layers:
                layer_stats = torch.cat([self.get_stats(layer.weight.unsqueeze(0)),
                                         self.get_stats(layer.bias.unsqueeze(0))], dim=1)
                all_w_stats.append(layer_stats)
                all_layer_types.append(type(layer).__name__)
                all_act_types.append(type(act).__name__)
        assert len(self.layers) in chosen_layers
        layer_stats = torch.cat([self.get_stats(self.fc.weight.unsqueeze(0)),
                                 self.get_stats(self.fc.bias.unsqueeze(0))], dim=1)
        all_w_stats.append(layer_stats)
        all_layer_types.append(type(self.fc).__name__)
        all_act_types.append('none')
        if max_size >= len(self.layers) + 1:
            zero_pad = torch.zeros(max_size - len(all_w_stats), all_w_stats[0].shape[1], device=all_w_stats[0].device)
            all_w_stats.append(zero_pad)
            all_layer_types.extend(['none'] * (max_size - len(all_layer_types)))
            all_act_types.extend(['none'] * (max_size - len(all_act_types)))
        all_w_stats = torch.cat(all_w_stats, dim=0)
        return all_w_stats, all_layer_types, all_act_types

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CachedCNNParkDataset(torch.utils.data.Dataset):
    """Lazy dataset backed by build_cnn_cache.py's Wild Park cache.

    The setup script produces cnn_cache_<split>.pt containing flat, metas and
    scores. Reconstruct each requested target CNN without extracting or
    materializing the entire model zoo.
    """

    def __init__(self, cache_dir, split="train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown Wild Park split: {split}")
        self.cache_file = Path(cache_dir).expanduser().resolve() / (
            f"cnn_cache_{split}.pt"
        )
        if not self.cache_file.is_file():
            raise FileNotFoundError(
                f"Missing Wild Park cache {self.cache_file}. "
                "Run scripts/setup_data/regression_cifar10_wp.sh first."
            )
        # Memory-map the large flat weights on supported PyTorch releases.
        try:
            cache = torch.load(
                self.cache_file, map_location="cpu", weights_only=False,
                mmap=True,
            )
        except TypeError:
            cache = torch.load(
                self.cache_file, map_location="cpu", weights_only=False
            )
        self.flat = cache["flat"]
        self.metas = cache["metas"]
        self.scores = cache["scores"]
        if len(self.metas) != len(self.scores):
            raise ValueError(
                f"Wild Park cache {self.cache_file} has "
                f"{len(self.metas)} metadata rows but {len(self.scores)} scores"
            )

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        offset = int(meta["offset"])
        state_dict = {}
        for key, shape, numel in zip(
            meta["keys"], meta["shapes"], meta["numels"]
        ):
            numel = int(numel)
            state_dict[key] = self.flat[offset:offset + numel].reshape(
                tuple(shape)
            )
            offset += numel

        cfg = meta["config"]
        model = Generic_CNN_Network(
            n_layers=int(cfg["n_layers"]),
            channels=list(cfg["channels"]),
            kernel_sizes=list(cfg["kernel_size"]),
            strides=list(cfg["stride"]),
            activations=list(cfg["activation"]),
            paddings=list(cfg["padding"]),
            out_size=10,
        )
        model.load_state_dict(state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, float(self.scores[index])


class CNN_Park_ModelData(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train"):
        self.split = split
        self.splits_path = os.path.join(dataset_dir, splits_path)
        with open(self.splits_path, "r") as f:
            self.dataset = json.load(f)[self.split]
        # The official CNN Wild Park split JSON stores paths as
        # "cifar10_zooV2/<run>/checkpoint_xxxxxx/checkpoint.pt".
        #
        # Our canonical local layout is:
        #   <dataset_dir>/cnn_wild/<run>/checkpoint_xxxxxx/checkpoint.pt
        #
        # Therefore strip the archive-only "cifar10_zooV2/" prefix before
        # joining with the canonical cnn_wild directory.
        normalized_paths = []
        prefix = "cifar10_zooV2/"
        for p in self.dataset["path"]:
            p = str(p).replace("\\", "/")
            if p.startswith(prefix):
                p = p[len(prefix):]
            normalized_paths.append(
                os.path.join(dataset_dir, "cnn_wild", p)
            )

        self.dataset["path"] = normalized_paths
        self.all_data = [None for _ in range(len(self.dataset["score"]))]

    def __len__(self):
        return len(self.dataset["path"])

    def item_first_load(self, item):

        path = self.dataset["path"][item]
        with open(path, 'rb') as f:
            model_obj = torch.load(f, map_location='cpu', weights_only=False)
        state_dict = model_obj["model"]

        label = self.dataset["score"][item]

        strides = model_obj['config']['stride']
        activations = model_obj['config']['activation']
        kernel_sizes = model_obj['config']['kernel_size']
        channels = model_obj['config']['channels']
        paddings = model_obj['config']['padding']
        n_layers = model_obj['config']['n_layers']
        config = {"n_layers": n_layers, "channels": channels, "kernel_sizes": kernel_sizes, "strides": strides,
                  "activations": activations, "paddings": paddings, "out_size": 10}

        self.all_data[item] = [state_dict, config, label]

    def __getitem__(self, item):
        if self.all_data[item] is None:
            self.item_first_load(item)
        sd, cfg, label = self.all_data[item]
        model = Generic_CNN_Network(**cfg)
        model.load_state_dict(sd)
        return model, label
