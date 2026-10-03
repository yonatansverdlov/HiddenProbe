import ast
import io
import json
import os
import pickle
import random
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
from torch.utils.data import Dataset

# Note: Dataset Code was partially inspired by: https://github.com/mkofinas/neural-graphs.git


class CIFAR10INRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split):
        self.split_dir = Path(dataset_dir) / split

        with (self.split_dir / "labels.csv").open("r") as f:
            self.samples = list(csv.DictReader(f))

        # Cache models after their first loading.
        self.all_data = [None] * len(self.samples)


    def __len__(self):
        return len(self.samples)

    def n_classes(self):
        return 10

    def __getitem__(self, item):
        if self.all_data[item] is None:
            row = self.samples[item]
            path = self.split_dir / row["filename"]
            label = int(row["label"])

            state = torch.load(path, map_location="cpu")

            # Convert downloaded SIREN key names to ProbeGen key names.
            converted_state = {
                "seq.0.weight": state["net.0.linear.weight"],
                "seq.0.bias": state["net.0.linear.bias"],
                "seq.1.weight": state["net.1.linear.weight"],
                "seq.1.bias": state["net.1.linear.bias"],
                "seq.2.weight": state["net.2.weight"],
                "seq.2.bias": state["net.2.bias"],
            }

            # CIFAR-10 INR maps (x,y) to RGB.
            model = INR_Network(
                in_features=2,
                n_layers=3,
                hidden_features=32,
                out_features=3,
            )
            model.load_state_dict(converted_state)

            self.all_data[item] = model, label

        return self.all_data[item]
    

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class INR_Network(nn.Module):
    def __init__(self, in_features=2, n_layers=3, hidden_features=32, out_features=1):
        super(INR_Network, self).__init__()
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
        return x + 0.5

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
                x = x + 0.5

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
    the NFN/DWS CIFAR10 and DNG CIFAR100 INRs used here are 2->32->32->3 (n_layers=3, 2 hidden, RGB out).
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


def _remap_siren_keys(sd):
    """NFN siren_cifar_wts stores SIREN weights as net.{i}.linear.{weight,bias} (hidden layers wrapped in a
    Sine layer) and net.{i}.{weight,bias} (final linear); INR_Network expects seq.{i}.{weight,bias}. Remap
    those keys. No-op when keys are already seq.* (our multiview / mnist / fmnist banks)."""
    if not any(k.startswith("net.") for k in sd):
        return sd
    return {k.replace(".linear.", ".").replace("net.", "seq.", 1): v for k, v in sd.items()}


class INRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train"):
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

        self.all_data = [None for _ in range(len(self.dataset["label"]))]

    def __len__(self):
        return len(self.dataset["label"])

    def n_classes(self):
        return len(set(self.dataset["label"]))

    def __getitem__(self, item):
        if self.all_data[item] is None:
            path = str(self.dataset["path"][item])
            try:
                state_dict = torch.load(path, map_location='cpu')
            except Exception as e:
                print(f"Failed to load {path}")
                raise e
            if "label" in state_dict.keys():
                state_dict.pop("label")
            assert "label" not in state_dict.keys()
            state_dict = _remap_siren_keys(state_dict)   # NFN net.{i}.linear.* -> seq.{i}.* (no-op for seq.* banks)
            label = int(self.dataset["label"][item])
            model = INR_Network(**infer_inr_arch(state_dict))
            model.load_state_dict(state_dict)
            self.all_data[item] = (model, label)

        model, label = self.all_data[item]
        return model, label


# endregion: INRs


# region: CNN datasets



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


class CNN_Park_ModelData(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train"):
        self.split = split
        self.splits_path = os.path.join(dataset_dir, splits_path)
        with open(self.splits_path, "r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [os.path.join(dataset_dir, 'cnn_wild', p) for p in self.dataset["path"]]
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


# endregion: CNN datasets



# =====================================================================================================
# Portable data roots (formerly pgh_paths.py). Every root can be overridden by an environment variable;
# defaults are RELATIVE to this repo and match scripts/setup_data/*.sh (DATA_ROOT=<repo>/data).
#   export DATA_ROOT=/somewhere/data            # base used by the setup scripts and these defaults
#   export PGH_WP_CACHE=/fast/disk/wp_cnn_cache # CIFAR-WP flat-tensor cache dir (cnn_cache_<split>.pt)
#   export PGH_SVHN_GS=/path/to/svhn_cropped    # (or PGH_CIFAR_GS / PGH_MNIST_GS / PGH_FMNIST_GS)
# =====================================================================================================
REPO = os.path.dirname(os.path.abspath(__file__))                       # this repository (portable)
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(REPO, "data")   # same default as scripts/setup_data/_common.sh


def _r(env_name, *rel_parts, base=None):
    """Env override if set/non-empty, else <base or DATA_ROOT>/<rel_parts>."""
    v = os.environ.get(env_name)
    return v if v else os.path.join(base if base is not None else DATA_ROOT, *rel_parts)


WP_DIR   = _r("PGH_WP_DIR", "regression", "cifar10_wp")                 # where regression_cifar10_wp.sh installs
WP_ZIP   = _r("PGH_WP_ZIP", "cnn_wild_park.zip", base=WP_DIR)           # the Zenodo zip (read directly, never extracted)
WP_CACHE = _r("PGH_WP_CACHE", "wp_cnn_cache", base=WP_DIR)              # flat-tensor CNN cache dir built by the WP setup script
# Wild-Park setup downloads the canonical split into the data directory; env can override.
SPLITS = os.environ.get("PGH_SPLITS") or os.path.join(WP_DIR, "splits.json")

# Unterthiner SmallCNN grayscale zoos (weights.npy / metrics.csv.gz / layout.csv), as installed by scripts/setup_data
ZOO_DIRS = {
    "svhn_gs":   _r("PGH_SVHN_GS",   "regression", "svhn"),
    "cifar_gs":  _r("PGH_CIFAR_GS",  "regression", "cifar10_gs"),
    "mnist_gs":  _r("PGH_MNIST_GS",  "regression", "mnist"),
    "fmnist_gs": _r("PGH_FMNIST_GS", "regression", "fmnist"),
}


# ---- CIFAR-10 Wild Park CNN-zoo loader (formerly wp_data.py) -----------------------------------------
"""Self-contained Wild Park CNN-zoo loader (Track B). Mirrors the Track A loader so both tracks use
the IDENTICAL split/protocol, without importing Track A code (train_wp_e2e.py parses args on import).

Each split entry: {path: [zip-internal ckpt paths], score: [test acc], step: [...]}. Checkpoints are
torch.save({"config": {...}, "model": state_dict}); targets are frozen Generic_CNN_Network, out_size=10.
"""

DEFAULT_SPLITS = SPLITS
DEFAULT_ZIP_NAS = WP_ZIP
DEFAULT_ZIP_SHM = os.path.join(os.environ.get("PGH_SHM", "/dev/shm"), "cnn_wild_park.zip")


class _S:                                          # stub for pickled create_random_cnns refs
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _S if "create_random_cnns" in module else super().find_class(module, name)


class _stub_pickle:
    Unpickler = _StubUnpickler; load = pickle.load; loads = pickle.loads
    dump = pickle.dump; dumps = pickle.dumps; HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _build_net(cfg, sd):
    net = Generic_CNN_Network(n_layers=cfg["n_layers"], channels=list(cfg["channels"]),
                              kernel_sizes=list(cfg["kernel_size"]), strides=list(cfg["stride"]),
                              activations=list(cfg["activation"]), paddings=list(cfg["padding"]), out_size=10)
    net.load_state_dict(sd); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _load_cnns_from_cache(cache_file, limit, dev):
    """Fast path: reconstruct CNNs from the flat-tensor cache built by regression_cifar10_wp.sh.
    ~5-6x faster than the zip path (no per-CNN unzip/unpickle); produces bit-identical modules."""
    c = torch.load(cache_file, map_location="cpu", weights_only=False)
    flat, metas, scores = c["flat"], c["metas"], c["scores"]
    N = len(metas) if not limit else min(limit, len(metas))
    nets, ys = [], []; t0 = time.time()
    for i in range(N):
        m = metas[i]; sd = {}; o = m["offset"]
        for k, shape, ne in zip(m["keys"], m["shapes"], m["numels"]):
            sd[k] = flat[o:o + ne].reshape(shape); o += ne
        nets.append(_build_net(m["config"], sd).to(dev)); ys.append(float(scores[i]))
    return nets, torch.tensor(ys)


def load_cnns(split, limit=0, dev="cpu", splits_path=DEFAULT_SPLITS, zip_path=None, cnn_cache=None):
    """Return (list[frozen Generic_CNN_Network on dev], targets tensor of test accuracies).
    If cnn_cache (a dir) is given and cnn_cache/cnn_cache_<split>.pt exists, use the fast cache path;
    else fall back to the (unchanged) serial zip path."""
    if cnn_cache:
        cf = os.path.join(cnn_cache, f"cnn_cache_{split}.pt")
        if os.path.exists(cf):
            return _load_cnns_from_cache(cf, limit, dev)
        print(f"[wp] cnn_cache set but {cf} missing -> serial zip load (run scripts/setup_data/regression_cifar10_wp.sh to build it)", flush=True)
    zp = zip_path or (DEFAULT_ZIP_SHM if os.path.exists(DEFAULT_ZIP_SHM) else DEFAULT_ZIP_NAS)
    zf = zipfile.ZipFile(zp)
    sp = json.load(open(splits_path))[split]
    paths, scores = sp["path"], sp["score"]
    N = len(paths) if not limit else min(limit, len(paths))
    nets, ys = [], []; t0 = time.time()
    for i in range(N):
        obj = torch.load(io.BytesIO(zf.read(paths[i])), map_location="cpu",
                         weights_only=False, pickle_module=_stub_pickle)
        cfg = dict(obj["config"])
        net = Generic_CNN_Network(n_layers=cfg["n_layers"], channels=list(cfg["channels"]),
                                  kernel_sizes=list(cfg["kernel_size"]), strides=list(cfg["stride"]),
                                  activations=list(cfg["activation"]), paddings=list(cfg["padding"]),
                                  out_size=10)
        net.load_state_dict(obj["model"]); net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net.to(dev)); ys.append(float(scores[i]))
    return nets, torch.tensor(ys)


# ---- Unterthiner Small CNN Zoo loader for the *_gs zoos (formerly svhn_gs_data.py) -------------------
"""SVHN-GS (Unterthiner Small CNN Zoo, "CS" collection) adapter for the probe-based trainers.

The released zoo stores FLATTENED 4,970-dim parameter vectors (weights.npy) for a FIXED tiny CNN, plus
metrics.csv.gz (labels/config) and layout.csv (per-variable flatten layout). This module:
  1. SmallCNN — a runnable PyTorch module EXACTLY matching the reference TF architecture
     (dnn_predict_accuracy/train_network.build_cnn): 3x Conv2D(16, k=3, stride=2, VALID, act) + GAP + Dense(10),
     1-channel grayscale input. Feature maps 32->15->7->3.
  2. unflatten_to_state_dict — TF (HWIO conv kernels, (in,out) dense) -> PyTorch (OIHW, (out,in)).
  3. Official NFN split (fixed permutation -> final checkpoint step==86 -> 40/10/50 split).
     All target models are retained; each CNN is reconstructed with its own recorded activation.
Reference: github.com/google-research/google-research/dnn_predict_accuracy ; github.com/jkalogero/scalegmn
"""
DATA_DIR = ZOO_DIRS["svhn_gs"]
_ACT = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": F.selu,
}
# layout varname -> (pytorch state-dict prefix, kind). Uses the .layers/.fc interface capture_features needs.
_MAP = {"sequential/conv2d/kernel:0": ("layers.0", "convk"), "sequential/conv2d/bias:0": ("layers.0", "b"),
        "sequential/conv2d_1/kernel:0": ("layers.1", "convk"), "sequential/conv2d_1/bias:0": ("layers.1", "b"),
        "sequential/conv2d_2/kernel:0": ("layers.2", "convk"), "sequential/conv2d_2/bias:0": ("layers.2", "b"),
        "sequential/dense/kernel:0": ("fc", "densek"), "sequential/dense/bias:0": ("fc", "b")}


class SmallCNN(nn.Module):
    """Runnable SmallCNN exposing the capture_features interface (.layers/.activations/.pool/.flatten/.fc)."""
    def __init__(self, activation="relu"):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Conv2d(1, 16, 3, stride=2, padding=0),
                                     nn.Conv2d(16, 16, 3, stride=2, padding=0),
                                     nn.Conv2d(16, 16, 3, stride=2, padding=0)])
        _a = _ACT[activation]
        self.activations = [_a, _a, _a]                  # per-layer activation (capture_features contract)
        self.pool = nn.AdaptiveAvgPool2d(1)              # global average pool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x, return_hidden=False):
        h = []
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x)); h.append(x)
        out = self.fc(self.flatten(self.pool(x)))
        return (out, tuple(h)) if return_hidden else out


def load_layout(data_dir=DATA_DIR):
    return pd.read_csv(os.path.join(data_dir, "layout.csv"))


def unflatten_to_state_dict(flat, layout_df):
    """flat: np.ndarray[4970] -> torch state_dict for SmallCNN. TF HWIO->OIHW, dense (in,out)->(out,in)."""
    sd = {}
    for _, r in layout_df.iterrows():
        name = r["varname"]; s, e = int(r["start_idx"]), int(r["end_idx"]); shape = ast.literal_eval(r["shape"])
        w = np.asarray(flat[s:e], dtype=np.float32).reshape(shape)
        mod, kind = _MAP[name]
        if kind == "convk":                              # TF (H,W,Cin,Cout) -> torch (Cout,Cin,H,W)
            sd[f"{mod}.weight"] = torch.from_numpy(np.ascontiguousarray(w.transpose(3, 2, 0, 1)))
        elif kind == "densek":                           # TF (in,out) -> torch (out,in)
            sd[f"{mod}.weight"] = torch.from_numpy(np.ascontiguousarray(w.T))
        else:                                            # bias, direct
            sd[f"{mod}.bias"] = torch.from_numpy(np.ascontiguousarray(w))
    return sd


def build_cnn(flat, activation, layout_df):
    m = SmallCNN(activation); m.load_state_dict(unflatten_to_state_dict(flat, layout_df)); m.eval()
    return m


def reference_forward(flat, x, activation, layout_df):
    """Independent 'ground-truth' path from the RAW flat vector (no state_dict), matching TF valid/stride2."""
    sl = {r["varname"]: (int(r["start_idx"]), int(r["end_idx"]), ast.literal_eval(r["shape"]))
          for _, r in layout_df.iterrows()}
    def conv(name, inp):
        s, e, shape = sl[name + "/kernel:0"]; k = torch.tensor(flat[s:e].reshape(shape), dtype=torch.float32)
        k = k.permute(3, 2, 0, 1)                        # HWIO -> OIHW
        s, e, _ = sl[name + "/bias:0"]; b = torch.tensor(flat[s:e], dtype=torch.float32)
        return F.conv2d(inp, k, b, stride=2, padding=0)
    a = _ACT[activation]
    h1 = a(conv("sequential/conv2d", x)); h2 = a(conv("sequential/conv2d_1", h1)); h3 = a(conv("sequential/conv2d_2", h2))
    g = h3.mean(dim=(2, 3))
    s, e, shape = sl["sequential/dense/kernel:0"]; dk = torch.tensor(flat[s:e].reshape(shape), dtype=torch.float32)
    s, e, _ = sl["sequential/dense/bias:0"]; db = torch.tensor(flat[s:e], dtype=torch.float32)
    out = g @ dk + db
    return out, (h1, h2, h3)


def make_split(data_dir=DATA_DIR, split_csv="split.csv"):
    """Return the official NFN 40/10/50 split over final Small-CNN-Zoo checkpoints.

    The split permutation MUST already exist. We never auto-generate a replacement:
    all methods must consume the same official NFN split file.

    Returns dict split -> rows/scores/activation, where rows are indices into
    weights.npy and activation is the per-target activation recorded in metrics.csv.gz.
    """
    metrics = pd.read_csv(os.path.join(data_dir, "metrics.csv.gz"), compression="gzip")
    n_rows = len(metrics)

    scsv = split_csv if os.path.isabs(split_csv) else os.path.join(data_dir, split_csv)
    if not os.path.isfile(scsv):
        raise FileNotFoundError(
            f"Missing official NFN split file: {scsv}. "
            "Run the matching scripts/setup_data/regression_*.sh script."
        )

    shuffled = (
        pd.read_csv(scsv, header=None)
        .values.flatten()
        .astype(np.int64)
    )

    # NFN split CSVs can contain one extra leading 0 line. Mirror the canonical
    # ProbeGen loader exactly: drop ONLY that leading row when the file is N+1.
    if len(shuffled) == n_rows + 1:
        shuffled = shuffled[1:]

    if len(shuffled) != n_rows:
        raise RuntimeError(
            f"Official NFN split has {len(shuffled)} rows but metrics has {n_rows} rows: {scsv}"
        )
    if shuffled.min() < 0 or shuffled.max() >= n_rows:
        raise RuntimeError(f"Official NFN split contains out-of-range indices: {scsv}")

    order = shuffled
    m = metrics.iloc[order].reset_index(drop=True)

    # NFN protocol: final checkpoints only.
    isfinal = (m["step"] == 86).to_numpy()
    finals_rows = order[isfinal]
    mf = m.loc[isfinal].reset_index(drop=True)

    # NFN protocol: second half test; first half -> fixed-seed 80/20 train/val.
    n = len(mf)
    test_split_point = int(0.5 * n)
    test = list(range(test_split_point, n))
    trainval = list(range(test_split_point))
    val_point = int(0.8 * len(trainval))
    import random as _r
    _r.Random(0).shuffle(trainval)
    train = trainval[:val_point]
    val = trainval[val_point:]

    out = {}
    for name, idcs in (("train", train), ("val", val), ("test", test)):
        idcs = np.asarray(idcs, dtype=np.int64)
        out[name] = {
            "rows": finals_rows[idcs],
            "scores": mf["test_accuracy"].to_numpy()[idcs],
            "activation": mf["config.activation"].astype(str).to_numpy()[idcs],
        }
    return out


def load_svhn_cnns(split, activation=None, dev="cpu", data_dir=DATA_DIR, split_csv="split.csv", limit=0):
    """Load ALL target CNNs from an official NFN Small-CNN-Zoo split.

    The activation argument is retained only for backward call compatibility and is
    intentionally ignored. No activation filtering is performed. Each target CNN is
    reconstructed with its own config.activation value from metrics.csv.gz.
    """
    lay = load_layout(data_dir)
    W = np.load(os.path.join(data_dir, "weights.npy"), mmap_mode="r")
    sp = make_split(data_dir, split_csv)[split]

    rows = sp["rows"]
    scores = sp["scores"].astype(np.float32)
    activations = sp["activation"]

    if limit and limit > 0:
        rows = rows[:limit]
        scores = scores[:limit]
        activations = activations[:limit]

    nets = []
    for ri, act in zip(rows, activations):
        act = str(act).lower()
        if act not in _ACT:
            raise ValueError(
                f"Unsupported Small-CNN-Zoo activation {act!r} at weights row {int(ri)}. "
                f"Supported activations: {sorted(_ACT)}"
            )
        m = build_cnn(np.array(W[int(ri)], dtype=np.float32), act, lay)
        for p in m.parameters():
            p.requires_grad_(False)
        nets.append(m.to(dev))

    return nets, torch.tensor(scores)


# =====================================================================================================
# Transformer accuracy prediction (Small Transformer Zoo: MNIST-Transformers / AGNews-Transformers).
# Thin re-export of models/transformer/intake.py: epoch-75 manifest with a seeded, run-disjoint 70/15/15
# split, the safe checkpoint loader, and the consolidated per-split cache used by
# `python main.py transformer ...`. Layout expected under --data_root (see scripts/setup_data/):
#     <data_root>/mnist_transformer/**/<run_id>_<epoch>_<acc>.pt
#     <data_root>/ag_news_transformer/**/<run_id>_<epoch>_<acc>.pt
# =====================================================================================================
from models.transformer.intake import (WRAPPER, EP75_RE, build_manifest, manifest_stats, write_manifest,  # noqa: E402,F401
                                       load_target, load_zoo, cache_path, build_cache, load_zoo_cached)
