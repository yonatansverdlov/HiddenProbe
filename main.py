"""Unified entry point for the merged HiddenProbe repository.

Core rule:
  * method=probegen always uses the canonical implementation imported from
    inr_classification_branch. This is true for BOTH classification and regression.
  * method=hiddenprobe uses that same backend only for MNIST/FMNIST classification.
  * all remaining HiddenProbe runs use the hiddenprobe branch implementations.
  * transformer experiments remain on the hiddenprobe branch and keep their existing
    "python main.py transformer ..." interface.

The routing layer owns method/task/dataset selection. Backend-only names such as
nfn_cifar_inr, *_gs and wp are intentionally hidden from the public interface.
"""
from __future__ import annotations

import os
import runpy
import sys
from typing import List, Optional, Set


CLASSIFICATION_DATASETS = {"mnist", "fmnist", "cifar10", "cifar10_aug"}
REGRESSION_DATASETS = {"mnist", "fmnist", "svhn", "cifar10_gs", "cifar10_wp"}

_DATASET_ALIASES = {
    "svh": "svhn",
    "mnist_inr": "mnist",
    "fmnist_inr": "fmnist",
    "cifar10_inr": "cifar10",
    "mnist_reg": "mnist",
    "fmnist_reg": "fmnist",
    "svhn_reg": "svhn",
    "svh_reg": "svhn",
    "cifar-aug": "cifar10_aug",
    "cifar10-aug": "cifar10_aug",
    "cifar_aug": "cifar10_aug",
    "cifar-gs": "cifar10_gs",
    "cifar10-gs": "cifar10_gs",
    "cifar_gs": "cifar10_gs",
    "cifar-wp": "cifar10_wp",
    "cifar10-wp": "cifar10_wp",
    "cifar_wp": "cifar10_wp",
}

_REGRESSION_ZOO = {
    "mnist": "mnist_gs",
    "fmnist": "fmnist_gs",
    "svhn": "svhn_gs",
    "cifar10_gs": "cifar_gs",
    "cifar10_wp": "wp",
}

USAGE = """usage:
  python main.py [--method {probegen,hiddenprobe}] [--task {classification,regression}] --dataset DATASET [args...]

method defaults to probegen; task is inferred when the dataset is unambiguous.
Specify --task for mnist and fmnist.

core datasets:
  classification: mnist, fmnist, cifar10, cifar10_aug
  regression:     mnist, fmnist, svhn, cifar10_gs, cifar10_wp

routing:
  ProbeGen                         -> canonical inr_classification backend (always)
  HiddenProbe MNIST/FMNIST class. -> canonical inr_classification backend
  HiddenProbe CIFAR class.        -> hiddenprobe INR backend
  HiddenProbe regression          -> hiddenprobe CNN-zoo/Wild-Park backend

transformers (kept from hiddenprobe):
  python main.py transformer {train,count,cache,smoke,manifest,verify} [args...]
"""


def _usage(msg: Optional[str] = None) -> None:
    if msg:
        print(f"main.py: {msg}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    raise SystemExit(2)


def _get_flag(argv: List[str], name: str) -> Optional[str]:
    key = f"--{name}"
    prefix = key + "="
    for i, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == key:
            if i + 1 >= len(argv):
                _usage(f"{key} needs a value")
            return argv[i + 1]
    return None


def _drop_flags(argv: List[str], names: Set[str]) -> List[str]:
    out: List[str] = []
    i = 0
    keys = {f"--{name}" for name in names}
    prefixes = tuple(f"--{name}=" for name in names)
    while i < len(argv):
        arg = argv[i]
        if arg.startswith(prefixes):
            i += 1
            continue
        if arg in keys:
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def _normalize_dataset(dataset: str) -> str:
    dataset = dataset.lower().strip()
    return _DATASET_ALIASES.get(dataset, dataset)


def _available_cpu_workers() -> int:
    """Number of CPU workers actually available to this process.

    On Linux, sched_getaffinity respects Slurm/cgroup/cpuset restrictions, so this
    matches the CPUs the current job can really use rather than the host total.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _run(module: str, argv: List[str]) -> None:
    sys.argv = [sys.argv[0]] + argv
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def _route_core(method: str, task: str, dataset: str, argv: List[str]) -> None:
    # This backend owns ALL ProbeGen runs, plus HiddenProbe MNIST/FMNIST classification.
    #
    # Every core dataset reconstructs target networks as nn.Module objects inside
    # Dataset.__getitem__. Passing those modules through multiprocessing queues
    # creates many shared-memory tensor mappings and can exhaust mmap/shared-memory
    # resources even when plenty of GPU memory is available. Keep core loading in
    # the main process for all datasets.
    n_workers = 0

    forwarded = _drop_flags(argv, {"method", "task", "dataset", "n_workers"})
    forwarded = [
        "--method", method,
        "--task", task,
        "--dataset", dataset,
        "--n_workers", str(n_workers),
    ] + forwarded
    _run("models.probegen_core_trainer", forwarded)


def _route_hiddenprobe_cifar(dataset: str, argv: List[str]) -> None:
    # HiddenProbe CIFAR uses the NFN/NFT 2-hidden-layer SIRENs from the hiddenprobe branch.
    forwarded = _drop_flags(argv, {"method", "task", "dataset"})

    split_name = (
        "nfn_cifar_split_noaug.json"
        if dataset == "cifar10"
        else "nfn_cifar_split.json"
    )

    # Structural dataset parameters are fixed by the dataset, not user-facing knobs.
    forwarded += [
        "--dataset", "nfn_cifar_inr",
        "--dataset_dir", "data/classification/cifar10_inr",
        "--splits", split_name,
        "--L", "2",
        "--H", "32",
        "--out_dim", "3",
        "--n_classes", "10",
        "--models_c_in", "2",
    ]
    _run("models.inr_hiddenprobe_trainer", forwarded)


def _route_hiddenprobe_regression(dataset: str, argv: List[str]) -> None:
    n_probes = _get_flag(argv, "n_probes")
    forwarded = _drop_flags(
        argv,
        {"method", "task", "dataset", "zoo", "hidden_mode", "n_probes"},
    )
    forwarded += [
        "--zoo", _REGRESSION_ZOO[dataset],
        "--hidden_mode", "on",
    ]

    # Unified public name: --n_probes. The legacy regression backend calls this
    # n_out_probes; in shared-probe mode it is also the hidden-probe count.
    if n_probes is not None:
        forwarded += ["--n_out_probes", n_probes]

    # Grayscale SmallCNN zoos use one input channel; Wild Park uses RGB.
    if dataset != "cifar10_wp" and _get_flag(forwarded, "models_c_in") is None:
        forwarded += ["--models_c_in", "1"]

    _run("models.cnn_zoo_trainer", forwarded)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        _usage()

    # Transformer work is intentionally retained from the second branch unchanged.
    if argv[0] == "transformer":
        _run("models.transformer.train", argv[1:])
        return

    method = (_get_flag(argv, "method") or "probegen").lower()
    task_flag = _get_flag(argv, "task")
    dataset_raw = _get_flag(argv, "dataset")
    if dataset_raw is None:
        _usage("--dataset is required")

    dataset = _normalize_dataset(dataset_raw)
    raw = dataset_raw.lower().strip()
    legacy_class = {"mnist_inr", "fmnist_inr", "cifar10_inr"}
    legacy_reg = {"mnist_reg", "fmnist_reg", "svhn_reg", "svh_reg"}
    if task_flag is not None:
        task = task_flag.lower()
        if raw in legacy_class and task != "classification":
            _usage(f"{dataset_raw!r} denotes a classification dataset")
        if raw in legacy_reg and task != "regression":
            _usage(f"{dataset_raw!r} denotes a regression dataset")
    elif raw in legacy_class or dataset in CLASSIFICATION_DATASETS - REGRESSION_DATASETS:
        task = "classification"
    elif raw in legacy_reg or dataset in REGRESSION_DATASETS - CLASSIFICATION_DATASETS:
        task = "regression"
    else:
        _usage(f"--task is required for ambiguous dataset {dataset_raw!r}")

    if method not in {"probegen", "hiddenprobe"}:
        _usage(f"unknown method {method!r}")
    if task not in {"classification", "regression"}:
        _usage(f"unknown task {task!r}")

    allowed = CLASSIFICATION_DATASETS if task == "classification" else REGRESSION_DATASETS
    if dataset not in allowed:
        _usage(
            f"dataset {dataset_raw!r} is not valid for task={task!r}; "
            f"choose one of {sorted(allowed)}"
        )

    # IMPORTANT: ProbeGen ALWAYS comes from the user's canonical backend.
    if method == "probegen":
        _route_core(method, task, dataset, argv)
        return

    # HiddenProbe MNIST/FMNIST classification also comes from the canonical backend.
    if task == "classification" and dataset in {"mnist", "fmnist"}:
        _route_core(method, task, dataset, argv)
        return

    # Remaining HiddenProbe runs come from the second branch.
    if task == "classification":
        _route_hiddenprobe_cifar(dataset, argv)
        return

    _route_hiddenprobe_regression(dataset, argv)


if __name__ == "__main__":
    main()
