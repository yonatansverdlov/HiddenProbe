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

import runpy
import sys


CLASSIFICATION_DATASETS = {"mnist", "fmnist", "cifar10", "cifar10_aug"}
REGRESSION_DATASETS = {"mnist", "fmnist", "svhn", "cifar10_gs", "cifar10_wp"}

_DATASET_ALIASES = {
    "svh": "svhn",
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
  python main.py --method {probegen,hiddenprobe} --task {classification,regression} --dataset DATASET [args...]

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


def _usage(msg: str | None = None) -> None:
    if msg:
        print(f"main.py: {msg}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    raise SystemExit(2)


def _get_flag(argv: list[str], name: str) -> str | None:
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


def _drop_flags(argv: list[str], names: set[str]) -> list[str]:
    out: list[str] = []
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


def _run(module: str, argv: list[str]) -> None:
    sys.argv = [sys.argv[0]] + argv
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def _route_core(method: str, task: str, dataset: str, argv: list[str]) -> None:
    # This backend owns ALL ProbeGen runs, plus HiddenProbe MNIST/FMNIST classification.
    forwarded = _drop_flags(argv, {"method", "task", "dataset"})
    forwarded = [
        "--method", method,
        "--task", task,
        "--dataset", dataset,
    ] + forwarded
    _run("models.probegen_core_trainer", forwarded)


def _route_hiddenprobe_cifar(dataset: str, argv: list[str]) -> None:
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


def _route_hiddenprobe_regression(dataset: str, argv: list[str]) -> None:
    forwarded = _drop_flags(argv, {"method", "task", "dataset", "zoo"})
    forwarded += ["--zoo", _REGRESSION_ZOO[dataset]]

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

    method = _get_flag(argv, "method")
    task = _get_flag(argv, "task")
    dataset_raw = _get_flag(argv, "dataset")

    if method is None:
        _usage("--method {probegen,hiddenprobe} is required")
    if task is None:
        _usage("--task {classification,regression} is required")
    if dataset_raw is None:
        _usage("--dataset is required")

    method = method.lower()
    task = task.lower()
    dataset = _normalize_dataset(dataset_raw)

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
