from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

from models.probegen_core import ProbeGen
from data_probegen import CIFAR10INRDataset, INRDataset, CNN_Park_ModelData


# Canonical tasks/datasets:
#   Classification: mnist, fmnist, cifar10, cifar10_aug, cifar100, cifar100_aug
#   Regression:     mnist, fmnist, svhn, cifar10_gs, cifar10_wp

# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"Total params:       {total:,}")
    print(f"Trainable params:   {trainable:,}")
    print(f"Non-trainable:      {non_trainable:,}")

    return total, trainable, non_trainable


# =============================================================================
# Small CNN Zoo target network
# =============================================================================

class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SmallCNNZooNetwork(nn.Module):
    """
    Small CNN Zoo architecture used in Predicting Neural Network Accuracy
    from Weights / NFN accuracy-prediction experiments.

    3 convolutional layers:
        channels 1 -> 16 -> 16 -> 16
        kernel_size = 3
        stride      = 2
        padding     = 1

    followed by global average pooling and Linear(16, 10).

    Because global average pooling is used, the learned ProbeGen images may be
    32x32 even for MNIST/Fashion-MNIST.
    """

    def __init__(self, activation: str):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.activations = nn.ModuleList(
            [self._make_activation(activation) for _ in range(3)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = str(name).lower()

        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "sine":
            return Sine(w0=30.0)
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        if name in {"none", "linear", "identity"}:
            return nn.Identity()

        raise ValueError(f"Unsupported CNN Zoo activation: {name}")

    def load_zoo_parameters(self, weights, biases) -> None:
        if len(weights) != 4 or len(biases) != 4:
            raise RuntimeError(
                f"Expected 4 weights + 4 biases, got "
                f"{len(weights)} weights + {len(biases)} biases."
            )

        for layer, w, b in zip(self.layers, weights[:3], biases[:3]):
            if tuple(layer.weight.shape) != tuple(w.shape):
                raise RuntimeError(
                    f"Conv weight mismatch: expected {tuple(layer.weight.shape)}, "
                    f"got {tuple(w.shape)}."
                )
            if tuple(layer.bias.shape) != tuple(b.shape):
                raise RuntimeError(
                    f"Conv bias mismatch: expected {tuple(layer.bias.shape)}, "
                    f"got {tuple(b.shape)}."
                )

            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(b)

        if tuple(self.fc.weight.shape) != tuple(weights[3].shape):
            raise RuntimeError(
                f"FC weight mismatch: expected {tuple(self.fc.weight.shape)}, "
                f"got {tuple(weights[3].shape)}."
            )
        if tuple(self.fc.bias.shape) != tuple(biases[3].shape):
            raise RuntimeError(
                f"FC bias mismatch: expected {tuple(self.fc.bias.shape)}, "
                f"got {tuple(biases[3].shape)}."
            )

        with torch.no_grad():
            self.fc.weight.copy_(weights[3])
            self.fc.bias.copy_(biases[3])

    def forward(self, x):
        # DeepLinearGenerator(n_layers=6) produces [T, 1, 32, 32].
        if x.ndim != 4 or x.shape[1] != 1:
            raise RuntimeError(
                f"SmallCNNZooNetwork expects [T, 1, H, W], got {tuple(x.shape)}."
            )

        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))

        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


# =============================================================================
# Exact NFN Small CNN Zoo regression dataset
# =============================================================================

class CNNZooRegressionDataset(torch.utils.data.Dataset):
    """
    Exact NFN data protocol:

      1. Read official hard-coded permutation CSV.
      2. Reorder all checkpoint rows by that permutation.
      3. Keep only final checkpoints: step == 86.
      4. Test = second 50%.
      5. First 50% becomes train/val pool.
      6. Shuffle train/val positions with random.Random(0).
      7. Train = first 80% of that pool, val = last 20%.

    This is therefore 40% train / 10% val / 50% test of the final models.

    weights.npy is memory-mapped rather than copied into RAM.
    """

    def __init__(self, data_path: str, split: str, idcs_file: str):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.data_path = Path(data_path).expanduser().resolve()
        self.idcs_file = Path(idcs_file).expanduser().resolve()

        weights_path = self.data_path / "weights.npy"
        metrics_path = self.data_path / "metrics.csv.gz"
        layout_path = self.data_path / "layout.csv"

        for p in [weights_path, metrics_path, layout_path, self.idcs_file]:
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        self.data = np.load(weights_path, mmap_mode="r")
        metrics_all = pd.read_csv(metrics_path, compression="gzip")
        self.layout = pd.read_csv(layout_path)

        # NFN split CSV quirk: the file contains one extra leading ``0`` line.
        # Read every line explicitly, then drop ONLY that leading extra row when
        # the CSV has exactly one more entry than metrics.csv.gz. This avoids
        # depending on pandas header inference and preserves a genuine index 0
        # that also appears later in the permutation.
        shuffled_idcs = (
            pd.read_csv(self.idcs_file, header=None)
            .values.flatten()
            .astype(np.int64)
        )
        if len(shuffled_idcs) == len(metrics_all) + 1:
            shuffled_idcs = shuffled_idcs[1:]

        if len(self.data) != len(metrics_all):
            raise RuntimeError(
                f"weights.npy has {len(self.data)} rows but metrics has "
                f"{len(metrics_all)} rows."
            )

        if len(shuffled_idcs) != len(metrics_all):
            raise RuntimeError(
                f"split CSV has {len(shuffled_idcs)} rows but metrics has "
                f"{len(metrics_all)} rows."
            )

        if shuffled_idcs.min() < 0 or shuffled_idcs.max() >= len(metrics_all):
            raise RuntimeError("Official split CSV contains out-of-range indices.")

        # Equivalent to:
        #   data = data[shuffled_idcs]
        #   metrics = metrics.iloc[shuffled_idcs]
        #   isfinal = metrics["step"] == 86
        #
        # without copying the full weight matrix.
        reordered_steps = metrics_all.iloc[shuffled_idcs]["step"].to_numpy()
        final_raw_idcs = shuffled_idcs[reordered_steps == 86]

        split_positions = self._split_indices_iid(len(final_raw_idcs))[split]
        self.raw_idcs = final_raw_idcs[np.asarray(split_positions, dtype=np.int64)]

        self.metrics = metrics_all.iloc[self.raw_idcs].copy()
        self.metrics.index = np.arange(len(self.metrics))

        labels = self.metrics["test_accuracy"].to_numpy(dtype=np.float64)
        if not np.isfinite(labels).all():
            raise RuntimeError("Non-finite test_accuracy labels found.")

        print(
            f"[{self.data_path.name}] {split}: {len(self.raw_idcs)} final models "
            f"(target=test_accuracy)"
        )

    @staticmethod
    def _split_indices_iid(n: int):
        test_split_point = int(0.5 * n)

        splits = {}
        splits["test"] = list(range(test_split_point, n))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))

        rng = random.Random(0)
        rng.shuffle(trainval_idcs)

        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return len(self.raw_idcs)

    def _decode_parameters(self, raw_idx: int):
        flat = np.asarray(self.data[raw_idx], dtype=np.float32)

        weights = []
        biases = []

        for _, row in self.layout.iterrows():
            start = int(row["start_idx"])
            end = int(row["end_idx"])
            shape = ast.literal_eval(str(row["shape"]))
            varname = str(row["varname"])

            arr = flat[start:end].reshape(shape)

            if varname.endswith("kernel:0"):
                # TensorFlow -> PyTorch:
                # Conv:  [H, W, in, out] -> [out, in, H, W]
                # Dense: [in, out]       -> [out, in]
                if arr.ndim == 4:
                    arr = arr.transpose(3, 2, 0, 1)
                elif arr.ndim == 2:
                    arr = arr.transpose(1, 0)
                else:
                    raise RuntimeError(
                        f"Unexpected kernel rank {arr.ndim} for {varname}; "
                        f"shape={arr.shape}."
                    )

                weights.append(
                    torch.from_numpy(np.ascontiguousarray(arr)).float()
                )

            elif varname.endswith("bias:0"):
                biases.append(
                    torch.from_numpy(np.ascontiguousarray(arr)).float()
                )

            else:
                raise ValueError(f"Unrecognized layout variable: {varname}")

        return weights, biases

    def __getitem__(self, idx):
        raw_idx = int(self.raw_idcs[idx])

        weights, biases = self._decode_parameters(raw_idx)

        activation = self.metrics.iloc[idx]["config.activation"]
        model = SmallCNNZooNetwork(activation=activation)
        model.load_zoo_parameters(weights, biases)

        target = float(self.metrics.iloc[idx]["test_accuracy"])
        return model, target


def collate_fn(batch):
    nets = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return nets, labels

# =============================================================================
# CLI / canonical dataset naming
# =============================================================================

parser = argparse.ArgumentParser(description="Canonical ProbeGen backend (all core datasets) and HiddenProbe backend for MNIST/FMNIST classification.")
parser.add_argument("--method", choices=["probegen", "hiddenprobe"], required=True)

parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_seeds", type=int, default=1)

# Canonical interface:
#   classification: mnist, fmnist, cifar10, cifar10_aug, cifar100, cifar100_aug
#   regression:     mnist, fmnist, svhn, cifar10_gs, cifar10_wp
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["classification", "regression"],
)
parser.add_argument("--dataset", type=str, required=True)

# Dataset locations are canonical and selected automatically from
# (task, dataset). Training/sweep scripts never need to pass data paths.
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"

# CIFAR INR options.
parser.add_argument(
    "--cifar_extra_aug",
    type=int,
    default=10,
    help="Additional INR realizations for *_aug classification tasks.",
)
parser.add_argument(
    "--cifar_cache_models",
    type=str2bool,
    default=False,
)
parser.add_argument("--d_hid", type=int, default=314)
parser.add_argument("--mixer_n_layers", type=int, default=6)

# Probe generator. If omitted, choose the natural default for the task:
# classification -> coordinate probes, regression -> image probes.
parser.add_argument(
    "--gen_type",
    type=str,
    default=None,
    choices=[
        "deep_linear_6",
        "deep_linear_5",
        "linear_0_no_acts",
        "linear_2_no_acts",
        "uniform_coords__no_opt",
    ],
)
parser.add_argument("--gen_latent_z", type=int, default=32)
parser.add_argument("--generator_width", type=int, default=16)

parser.add_argument(
    "--per_probe_mlp",
    type=str,
    default="none",
    choices=["none", "linear", "mlp", "mlp2", "mlp3"],
)
parser.add_argument("--per_probe_mlp_width", type=int, default=None)
parser.add_argument("--per_probe_out_dim", type=int, default=4)
parser.add_argument(
    "--per_probe_init",
    type=str,
    default="standard",
    choices=["standard", "inductive"],
)
parser.add_argument("--r_per_hidden", type=int, default=2)
parser.add_argument("--rank", type=int, default=8)

# Optimization
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument(
    "--scheduler",
    type=str,
    default="plateau",
    choices=["cosine", "plateau", "none"],
)
parser.add_argument(
    "--plateau_monitor",
    type=str,
    default=None,
    choices=["val_tau", "val_acc", "val_loss"],
)
parser.add_argument("--plateau_factor", type=float, default=0.7)
parser.add_argument("--plateau_patience", type=int, default=3)
parser.add_argument("--plateau_min_lr", type=float, default=1e-6)

args = parser.parse_args()

# The method selects the model semantics in this backend. ProbeGen is always
# output-only. HiddenProbe is supported here only for MNIST/FMNIST classification
# and always enables hidden features.
if args.method == "probegen":
    args.include_hidden_features = False
elif args.task == "classification" and args.dataset in {"mnist", "fmnist"}:
    args.include_hidden_features = True
else:
    raise ValueError(
        "This backend supports method=hiddenprobe only for classification "
        "datasets {mnist, fmnist}. Other HiddenProbe runs are routed to the "
        "hiddenprobe backend by main.py."
    )

if args.num_seeds < 1:
    raise ValueError("--num_seeds must be >= 1")

CLASSIFICATION_DATASETS = {
    "mnist",
    "fmnist",
    "cifar10",
    "cifar10_aug",
}
REGRESSION_DATASETS = {
    "mnist",
    "fmnist",
    "svhn",
    "cifar10_gs",
    "cifar10_wp",
}

allowed_datasets = (
    CLASSIFICATION_DATASETS
    if args.task == "classification"
    else REGRESSION_DATASETS
)
if args.dataset not in allowed_datasets:
    raise ValueError(
        f"Dataset {args.dataset!r} is not valid for task={args.task!r}. "
        f"Choose one of: {', '.join(sorted(allowed_datasets))}"
    )

if args.gen_type is None:
    args.gen_type = (
        "linear_2_no_acts"
        if args.task == "classification"
        else "deep_linear_6"
    )

if args.plateau_monitor is None:
    args.plateau_monitor = (
        "val_acc" if args.task == "classification" else "val_tau"
    )

if args.task == "regression":
    if args.include_hidden_features:
        raise ValueError(
            "Regression datasets are configured for "
            "--include_hidden_features=false."
        )
    if args.gen_type not in {"deep_linear_5", "deep_linear_6"}:
        raise ValueError(
            f"Regression dataset {args.dataset} uses image probes; "
            "choose deep_linear_5 or deep_linear_6."
        )
    if args.plateau_monitor != "val_tau":
        raise ValueError(
            "Regression selects checkpoints by Kendall tau; "
            "use --plateau_monitor=val_tau."
        )
else:
    if args.gen_type not in {
        "linear_0_no_acts",
        "linear_2_no_acts",
        "uniform_coords__no_opt",
    }:
        raise ValueError(
            f"Classification dataset {args.dataset} is an INR task and expects "
            "2-D coordinate probes. Use linear_2_no_acts (recommended), "
            "linear_0_no_acts, or uniform_coords__no_opt."
        )

# Canonical on-disk layout. Setup scripts write directly to these locations.
# Both model-data directory and split file are selected from (task, dataset).
_DATA_LAYOUT = {
    ("classification", "mnist"): {
        "kind": "mnist_inr",
        "data_rel": "classification/mnist_inr/dataset",
        "split_rel": "classification/mnist_inr/mnist_splits.json",
    },
    ("classification", "fmnist"): {
        "kind": "mnist_inr",
        "data_rel": "classification/fmnist_inr/dataset",
        "split_rel": "classification/fmnist_inr/fmnist_splits.json",
    },
    ("classification", "cifar10"): {
        "kind": "cifar_inr",
        "data_rel": "classification/cifar10_inr",
        "num_classes": 10,
        "augmented": False,
    },
    # CIFAR10-Aug uses the same downloaded CIFAR10 INR data.
    ("classification", "cifar10_aug"): {
        "kind": "cifar_inr",
        "data_rel": "classification/cifar10_inr",
        "num_classes": 10,
        "augmented": True,
    },
    ("regression", "mnist"): {
        "kind": "cnn_zoo",
        "data_rel": "regression/mnist",
        "split_rel": "regression/mnist/split.csv",
    },
    ("regression", "fmnist"): {
        "kind": "cnn_zoo",
        "data_rel": "regression/fmnist",
        "split_rel": "regression/fmnist/split.csv",
    },
    ("regression", "svhn"): {
        "kind": "cnn_zoo",
        "data_rel": "regression/svhn",
        "split_rel": "regression/svhn/split.csv",
    },
    ("regression", "cifar10_gs"): {
        "kind": "cnn_zoo",
        "data_rel": "regression/cifar10_gs",
        "split_rel": "regression/cifar10_gs/split.csv",
    },
    ("regression", "cifar10_wp"): {
        "kind": "cnn_wild_park",
        "data_rel": "regression/cifar10_wp",
        "split_rel": "regression/cifar10_wp/splits.json",
    },
}

torch.multiprocessing.set_sharing_strategy("file_system")


def _require_file(path_like, what):
    p = Path(path_like).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {what}: {p}. "
            f"Run the matching dataset setup script."
        )
    return str(p)


def _require_dir(path_like, what):
    p = Path(path_like).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(
            f"Missing {what}: {p}. "
            f"Run the matching dataset setup script."
        )
    return str(p)


# =============================================================================
# Dataset construction
# =============================================================================


def _selected_dataset_config(args):
    """Resolve canonical paths from (task, dataset) only."""
    cfg = dict(_DATA_LAYOUT[(args.task, args.dataset)])

    data_dir = (DATA_ROOT / cfg["data_rel"]).resolve()

    split_file = None
    if "split_rel" in cfg:
        split_file = (DATA_ROOT / cfg["split_rel"]).resolve()

    cfg["data_dir"] = str(data_dir)
    cfg["resolved_split_file"] = (
        str(split_file) if split_file is not None else None
    )
    return cfg


def build_datasets(args):
    cfg = _selected_dataset_config(args)
    kind = cfg["kind"]
    data_dir = _require_dir(cfg["data_dir"], f"{args.task}/{args.dataset} data directory")

    # ------------------------------------------------------------
    # Regression: Small CNN Zoo (MNIST / FMNIST / SVHN / CIFAR10-GS)
    # ------------------------------------------------------------
    if kind == "cnn_zoo":
        split_file = _require_file(
            cfg["resolved_split_file"],
            f"{args.task}/{args.dataset} split file",
        )

        train_set = CNNZooRegressionDataset(
            data_path=data_dir,
            split="train",
            idcs_file=split_file,
        )
        val_set = CNNZooRegressionDataset(
            data_path=data_dir,
            split="val",
            idcs_file=split_file,
        )
        test_set = CNNZooRegressionDataset(
            data_path=data_dir,
            split="test",
            idcs_file=split_file,
        )

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "task": "regression",
            "d_out": 1,
            "models_c_in": 1,
            "models_c_out": 10,
            "n_hidden_target_layers": 0,
            "data_dir": data_dir,
            "split_file": split_file,
        }

    # ------------------------------------------------------------
    # Regression: CNN Wild Park
    # ------------------------------------------------------------
    if kind == "cnn_wild_park":
        split_file = _require_file(
            cfg["resolved_split_file"],
            f"{args.task}/{args.dataset} split file",
        )

        train_set = CNN_Park_ModelData(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="train",
        )
        val_set = CNN_Park_ModelData(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="val",
        )
        test_set = CNN_Park_ModelData(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="test",
        )

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "task": "regression",
            "d_out": 1,
            "models_c_in": 3,
            "models_c_out": 10,
            "n_hidden_target_layers": 0,
            "data_dir": data_dir,
            "split_file": split_file,
        }

    # ------------------------------------------------------------
    # Classification: MNIST / Fashion-MNIST INR
    # ------------------------------------------------------------
    if kind == "mnist_inr":
        split_file = _require_file(
            cfg["resolved_split_file"],
            f"{args.task}/{args.dataset} split file",
        )

        train_set = INRDataset(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="train",
        )
        val_set = INRDataset(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="val",
        )
        test_set = INRDataset(
            dataset_dir=data_dir,
            splits_path=split_file,
            split="test",
        )

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "task": "classification",
            "d_out": train_set.n_classes(),
            "models_c_in": 2,
            "models_c_out": 1,
            "n_hidden_target_layers": 2,
            "data_dir": data_dir,
            "split_file": split_file,
        }

    # ------------------------------------------------------------
    # Classification: CIFAR-10 / CIFAR-100 INR
    # ------------------------------------------------------------
    if kind == "cifar_inr":
        num_classes = int(cfg["num_classes"])
        is_augmented = bool(cfg["augmented"])
        dataset_name = "CIFAR-100" if num_classes == 100 else "CIFAR-10"
        effective_extra_aug = args.cifar_extra_aug if is_augmented else 0

        common_kwargs = {
            "dataset_dir": data_dir,
            "extra_aug": effective_extra_aug,
            "cache_models": args.cifar_cache_models,
            "num_classes": num_classes,
            "dataset_name": dataset_name,
        }

        train_set = CIFAR10INRDataset(split="train", **common_kwargs)
        val_set = CIFAR10INRDataset(split="val", **common_kwargs)
        test_set = CIFAR10INRDataset(split="test", **common_kwargs)

        if not is_augmented:
            assert len(train_set) == 45000, len(train_set)
            assert len(val_set) == 5000, len(val_set)
            assert len(test_set) == 10000, len(test_set)
        elif effective_extra_aug == 10:
            assert len(train_set) == 495000, len(train_set)
            assert len(val_set) == 5000, len(val_set)
            assert len(test_set) == 10000, len(test_set)

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "task": "classification",
            "d_out": num_classes,
            "models_c_in": 2,
            "models_c_out": 3,
            "n_hidden_target_layers": 2,
            "effective_extra_aug": effective_extra_aug,
            "data_dir": data_dir,
            "num_classes": num_classes,
        }

    raise ValueError(
        f"Unknown dataset configuration: task={args.task}, dataset={args.dataset}, kind={kind}"
    )


# =============================================================================
# Metrics
# =============================================================================

def safe_tau(y_true, y_pred):
    result = kendalltau(y_true, y_pred)
    tau = result.statistic if hasattr(result, "statistic") else result[0]
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {"tau": safe_tau(y_true, y_pred)}


# =============================================================================
# One seed
# =============================================================================

def run_one_seed(args, seed, exp_dir):
    set_seed(seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"run_seed: {seed}\n")

    ds = build_datasets(args)
    train_set = ds["train"]
    val_set = ds["val"]
    test_set = ds["test"]
    task = ds["task"]

    print(
        f"Train set: {len(train_set)}, "
        f"Val set: {len(val_set)}, "
        f"Test set: {len(test_set)}"
    )
    print(f"Task: {task}")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    print(
        "Model config: "
        f"gen_type={args.gen_type}, "
        f"n_tokens={args.n_tokens}, "
        f"d_hid={args.d_hid}, "
        f"generator_width={args.generator_width}, "
        f"include_hidden_features={args.include_hidden_features}, "
        f"n_hidden_target_layers={ds['n_hidden_target_layers']}"
    )

    model = ProbeGen(
        n_tokens=args.n_tokens,
        d_hidden=args.d_hid,
        models_c_in=ds["models_c_in"],
        models_c_out=ds["models_c_out"],
        d_out=ds["d_out"],
        gen_type=args.gen_type,
        gen_latent_z=args.gen_latent_z,
        generator_width=args.generator_width,
        mixer_n_layers=args.mixer_n_layers,
        include_hidden_features=(
            args.include_hidden_features if task == "classification" else False
        ),
        per_probe_mlp=args.per_probe_mlp,
        per_probe_mlp_width=args.per_probe_mlp_width,
        per_probe_out_dim=args.per_probe_out_dim,
        per_probe_init=args.per_probe_init,
        n_hidden_target_layers=ds["n_hidden_target_layers"],
        r_per_hidden=args.r_per_hidden,
        rank=args.rank,
        seed=seed,
    )

    total_params, trainable_params, _ = count_params(model)

    print("Parameter breakdown:")
    for name in [
        "probe_source",
        "hidden_aggregators",
        "per_probe_mlp",
        "points_mixer",
    ]:
        module = getattr(model, name, None)
        if module is not None:
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:20s}: {n:,}")
    print(f"  {'TOTAL':20s}: {total_params:,}")

    meta = {
        "seed": seed,
        "dataset": args.dataset,
        "task": task,
        "selection_metric": "val_tau" if task == "regression" else "val_acc",
        "n_tokens": args.n_tokens,
        "d_hid": args.d_hid,
        "mixer_n_layers": args.mixer_n_layers,
        "gen_type": args.gen_type,
        "gen_latent_z": args.gen_latent_z,
        "generator_width": args.generator_width,
        "include_hidden_features": (
            args.include_hidden_features if task == "classification" else False
        ),
        "models_c_in": ds["models_c_in"],
        "models_c_out": ds["models_c_out"],
        "d_out": ds["d_out"],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "plateau_monitor": args.plateau_monitor,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_min_lr": args.plateau_min_lr,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    meta.update(
        {
            "data_root": str(DATA_ROOT),
            "data_dir": ds.get("data_dir"),
            "split_file": ds.get("split_file"),
        }
    )
    if args.task == "classification" and args.dataset.startswith("cifar"):
        meta.update(
            {
                "cifar_num_classes": ds.get("num_classes"),
                "cifar_extra_aug": ds.get("effective_extra_aug", 0),
                "cifar_cache_models": args.cifar_cache_models,
            }
        )

    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    model = model.float().to(device)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(loader):
        model.eval()

        if task == "regression":
            predicted = []
            gt = []
            loss_sum = 0.0
            total = 0

            for nets, label in loader:
                label = label.to(device=device, dtype=torch.float32)
                for net in nets:
                    net.to(device)

                out = model(nets=nets).reshape(-1)
                loss_sum += F.mse_loss(out, label, reduction="sum").item()
                total += label.numel()
                predicted.extend(out.detach().cpu().numpy().tolist())
                gt.extend(label.detach().cpu().numpy().tolist())

                for net in nets:
                    net.to("cpu")
                    net.zero_grad(set_to_none=True)

            model.train()
            metrics = regression_metrics(gt, predicted)
            metrics["loss"] = loss_sum / max(total, 1)
            metrics["predicted"] = np.asarray(predicted)
            metrics["gt"] = np.asarray(gt)
            return metrics

        loss_sum = 0.0
        correct = 0
        total = 0
        predicted = []
        gt = []

        for nets, label in loader:
            label = label.to(device=device, dtype=torch.long)
            for net in nets:
                net.to(device)

            out = model(nets=nets)
            loss_sum += F.cross_entropy(out, label, reduction="sum").item()

            pred = out.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += label.numel()

            predicted.extend(pred.detach().cpu().numpy().tolist())
            gt.extend(label.detach().cpu().numpy().tolist())

            for net in nets:
                net.to("cpu")
                net.zero_grad(set_to_none=True)

        model.train()
        return {
            "loss": loss_sum / max(total, 1),
            "acc": correct / max(total, 1),
            "predicted": np.asarray(predicted),
            "gt": np.asarray(gt),
        }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
        )
    elif args.scheduler == "plateau":
        plateau_mode = "min" if args.plateau_monitor == "val_loss" else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    global_step = 0
    best_val_metric = -float("inf")
    best_ckpt_path = os.path.join(exp_dir, "best_checkpoint.pth")

    if task == "regression":
        logging = pd.DataFrame(
            columns=[
                "exp_name", "epoch", "global_step", "lr", "train_loss",
                "val_loss", "test_loss", "val_tau", "test_tau", "is_best"
            ]
        )
    else:
        logging = pd.DataFrame(
            columns=[
                "exp_name", "epoch", "global_step", "lr", "train_loss",
                "val_loss", "test_loss", "val_acc", "test_acc", "is_best"
            ]
        )

    def current_lr():
        return optimizer.param_groups[0]["lr"]

    def evaluate_and_log(
        epoch,
        step,
        train_loss=float("nan"),
        save_if_best=True,
        announce_best=False,
    ):
        nonlocal best_val_metric

        # IMPORTANT: checkpoint selection / scheduler use VALIDATION only.
        # TEST is evaluated only for reporting/logging, exactly as in the old logs.
        val_results = evaluate(val_loader)
        test_results = evaluate(test_loader)

        if task == "regression":
            val_metric = val_results["tau"]
            metric_for_selection = (
                val_metric if np.isfinite(val_metric) else -1.0
            )
            plateau_value = metric_for_selection
            metric_label = "val_tau"
        else:
            val_metric = val_results["acc"]
            metric_for_selection = val_metric
            plateau_value = (
                val_results["loss"]
                if args.plateau_monitor == "val_loss"
                else val_metric
            )
            metric_label = "val_acc"

        if args.scheduler == "plateau" and scheduler is not None:
            scheduler.step(plateau_value)

        is_best = False
        if save_if_best and metric_for_selection > best_val_metric:
            is_best = True
            best_val_metric = metric_for_selection
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "epoch": epoch,
                    "global_step": step,
                    "best_val_metric": best_val_metric,
                    "metric_name": metric_label,
                    "args": vars(args),
                    "seed": seed,
                },
                best_ckpt_path,
            )

            if announce_best:
                print(
                    f"step={step} epoch={epoch} NEW_BEST "
                    f"{metric_label}={best_val_metric:.6f} "
                    f"lr={current_lr():.2e}"
                )

        if task == "regression":
            print(
                f"EVAL step={step} epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_results['loss']:.6f} "
                f"test_loss={test_results['loss']:.6f} "
                f"val_tau={val_results['tau']:.6f} "
                f"test_tau={test_results['tau']:.6f} "
                f"lr={current_lr():.2e}"
            )
            return {
                "exp_name": args.exp_name,
                "epoch": epoch,
                "global_step": step,
                "lr": current_lr(),
                "train_loss": train_loss,
                "val_loss": val_results["loss"],
                "test_loss": test_results["loss"],
                "val_tau": val_results["tau"],
                "test_tau": test_results["tau"],
                "is_best": is_best,
            }

        print(
            f"EVAL step={step} epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_results['loss']:.6f} "
            f"test_loss={test_results['loss']:.6f} "
            f"val_acc={val_results['acc']:.6f} "
            f"test_acc={test_results['acc']:.6f} "
            f"lr={current_lr():.2e}"
        )
        return {
            "exp_name": args.exp_name,
            "epoch": epoch,
            "global_step": step,
            "lr": current_lr(),
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "test_loss": test_results["loss"],
            "val_acc": val_results["acc"],
            "test_acc": test_results["acc"],
            "is_best": is_best,
        }

    running_train_loss = 0.0
    running_train_batches = 0

    for epoch in range(args.epochs):
        for nets, label in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            if task == "regression":
                label = label.to(device=device, dtype=torch.float32)
            else:
                label = label.to(device=device, dtype=torch.long)

            for net in nets:
                net.to(device)

            out = model(nets=nets)
            if task == "regression":
                out = out.reshape(-1)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())
            running_train_batches += 1

            if scheduler is not None and args.scheduler == "cosine":
                scheduler.step()

            for net in nets:
                net.to("cpu")
                net.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.eval_every == 0:
                interval_train_loss = (
                    running_train_loss / max(running_train_batches, 1)
                )
                log_row = evaluate_and_log(
                    epoch=epoch,
                    step=global_step,
                    train_loss=interval_train_loss,
                    save_if_best=True,
                    announce_best=True,
                )
                running_train_loss = 0.0
                running_train_batches = 0
                logging = pd.concat(
                    [logging, pd.DataFrame([log_row])],
                    ignore_index=True,
                )
                logging.to_csv(
                    os.path.join(exp_dir, "log.csv"),
                    index=False,
                )

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_dir, "intermediate_checkpoint.pth"),
                )

    torch.save(
        model.state_dict(),
        os.path.join(exp_dir, "epoch_last.pth"),
    )

    # Always evaluate at the end so a late improvement is not missed.
    final_train_loss = (
        running_train_loss / running_train_batches
        if running_train_batches > 0
        else float("nan")
    )
    final_log_row = evaluate_and_log(
        epoch=args.epochs - 1,
        step=global_step,
        train_loss=final_train_loss,
        save_if_best=True,
        announce_best=False,
    )
    logging = pd.concat(
        [logging, pd.DataFrame([final_log_row])],
        ignore_index=True,
    )
    logging.to_csv(
        os.path.join(exp_dir, "log.csv"),
        index=False,
    )

    if not os.path.exists(best_ckpt_path):
        raise RuntimeError("No best checkpoint was created.")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_results = evaluate(val_loader)
    best_test_results = evaluate(test_loader)

    predictions = pd.DataFrame(
        {
            "split": (
                ["val"] * len(best_val_results["gt"])
                + ["test"] * len(best_test_results["gt"])
            ),
            "target": np.concatenate(
                [best_val_results["gt"], best_test_results["gt"]]
            ),
            "prediction": np.concatenate(
                [best_val_results["predicted"], best_test_results["predicted"]]
            ),
        }
    )
    predictions.to_csv(
        os.path.join(exp_dir, "best_predictions.csv"),
        index=False,
    )

    print("\n========== Best checkpoint results ==========")
    print(f"Seed: {seed}")
    print(f"Best epoch: {ckpt['epoch']}")
    print(f"Best global step: {ckpt['global_step']}")

    if task == "regression":
        print(f"Best val Kendall tau:  {best_val_results['tau']:.6f}")
        print(f"Best test Kendall tau: {best_test_results['tau']:.6f}")
        print(f"Best val loss:         {best_val_results['loss']:.6f}")
        print(f"Best test loss:        {best_test_results['loss']:.6f}")
        print("============================================\n")

        return {
            "seed": seed,
            "exp_dir": exp_dir,
            "best_epoch": ckpt["epoch"],
            "best_global_step": ckpt["global_step"],
            "selection_metric": "val_tau",
            "best_val_tau": best_val_results["tau"],
            "best_test_tau": best_test_results["tau"],
            "best_val_loss": best_val_results["loss"],
            "best_test_loss": best_test_results["loss"],
        }

    print(f"Best val accuracy:  {best_val_results['acc']:.6f}")
    print(f"Best test accuracy: {best_test_results['acc']:.6f}")
    print(f"Best val loss:      {best_val_results['loss']:.6f}")
    print(f"Best test loss:     {best_test_results['loss']:.6f}")
    print("============================================\n")

    return {
        "seed": seed,
        "exp_dir": exp_dir,
        "best_epoch": ckpt["epoch"],
        "best_global_step": ckpt["global_step"],
        "selection_metric": "val_acc",
        "best_val_acc": best_val_results["acc"],
        "best_test_acc": best_test_results["acc"],
        "best_val_loss": best_val_results["loss"],
        "best_test_loss": best_test_results["loss"],
    }


# =============================================================================
# Multi-seed summary
# =============================================================================

def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    std = (
        float(np.nanstd(values, ddof=1))
        if np.sum(np.isfinite(values)) > 1
        else 0.0
    )
    return mean, std


def main():
    base_exp_dir = f"experiments/{args.task}/{args.dataset}/runs/{args.exp_name}"
    os.makedirs(base_exp_dir, exist_ok=True)

    results = []

    for seed_idx in range(args.num_seeds):
        run_seed = args.seed + seed_idx

        if args.num_seeds == 1:
            exp_dir = base_exp_dir
        else:
            exp_dir = os.path.join(base_exp_dir, f"seed_{run_seed}")

        print("\n" + "=" * 80)
        print(
            f"Running seed {run_seed} "
            f"({seed_idx + 1}/{args.num_seeds})"
        )
        print(f"Experiment directory: {exp_dir}")
        print("=" * 80 + "\n")

        result = run_one_seed(args, run_seed, exp_dir)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(base_exp_dir, "seeds_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n========== Seeds summary ==========")
    print(f"task: {args.task}")
    print(f"dataset: {args.dataset}")
    print(f"num_seeds: {args.num_seeds}")
    print(f"seeds: {summary_df['seed'].tolist()}")
    print(f"summary csv: {summary_csv}")

    if args.task == "regression":
        val_mean, val_std = mean_std(summary_df["best_val_tau"].values)
        test_mean, test_std = mean_std(summary_df["best_test_tau"].values)
        print(
            f"Best val Kendall tau:  mean={val_mean:.6f}, std={val_std:.6f}"
        )
        print(
            f"Best test Kendall tau: mean={test_mean:.6f}, std={test_std:.6f}"
        )
    else:
        val_mean, val_std = mean_std(summary_df["best_val_acc"].values)
        test_mean, test_std = mean_std(summary_df["best_test_acc"].values)
        print(
            f"Best val accuracy:  mean={val_mean:.6f}, std={val_std:.6f}"
        )
        print(
            f"Best test accuracy: mean={test_mean:.6f}, std={test_std:.6f}"
        )

    print("===================================\n")


if __name__ == "__main__":
    main()
