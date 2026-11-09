#!/usr/bin/env python3
"""Paper-style probe-count plots from benchmark summary.csv files.

Supports checkpoints/probe_count_3seeds/<method>/<dataset>/summary.csv
and checkpoints/probe_count_5seeds/<dataset>/summary.csv.

Figure 3's exact colors: ProbeGen orange #D55E00; HiddenProbe blue #0072B2.
Default: each dataset's curve is mean over seeds with ±SEM error bars.
No invented/interpolated experimental values.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
Q_VALUES = (16, 32, 64, 128, 256)
DATASETS = ("fmnist_inr", "svhn", "mnist_transformer")
LABELS = {
    "fmnist_inr": "FMNIST INR",
    "svhn": "SVHN",
    "mnist_transformer": "MNIST transformer",
}
METRICS = {
    "fmnist_inr": ("test_acc", "Test accuracy"),
    "svhn": ("test_tau", r"Test Kendall $\tau$"),
    "mnist_transformer": ("test_tau", r"Test Kendall $\tau$"),
}
METHODS = ("probegen", "hiddenprobe")
STYLE = {
    "probegen": dict(label="ProbeGen (output-only)", color="#D55E00",
                     linestyle="--", marker="o"),
    "hiddenprobe": dict(label="HiddenProbe (hidden-layers)", color="#0072B2",
                        linestyle="-", marker="s"),
}


@dataclass(frozen=True)
class Point:
    mean: float
    sd: float
    sem: float
    n_seeds: int


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#333333",
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finite_value(row: dict, key: str, path: Path) -> float:
    try:
        v = float(row[key])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Missing or invalid {key} in {path}: {row}") from exc
    if not math.isfinite(v):
        raise ValueError(f"Nonfinite {key} in {path}: {row}")
    return v


def read_summary(path: Path, dataset: str, method: str) -> dict[int, Point]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "method", "n_probes", "n_seeds", "test_metric_name",
                    "test_metric_mean", "test_metric_sd", "test_metric_sem"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)

    result: dict[int, Point] = {}
    for row in rows:
        if row["method"].strip().lower() != method:
            continue
        if row["dataset"].strip().lower() != dataset:
            raise ValueError(f"Wrong dataset in {path}: {row['dataset']}")
        if row["test_metric_name"].strip() != METRICS[dataset][0]:
            raise ValueError(f"Wrong metric in {path}: {row['test_metric_name']}")
        q, n = int(row["n_probes"]), int(row["n_seeds"])
        if q not in Q_VALUES or n < 2 or q in result:
            raise ValueError(f"Bad Q={q} or seed count n={n}, or duplicate, in {path}")
        pt = Point(
            finite_value(row, "test_metric_mean", path),
            finite_value(row, "test_metric_sd", path),
            finite_value(row, "test_metric_sem", path), n)
        if pt.sd < 0 or pt.sem < 0:
            raise ValueError(f"Negative uncertainty in {path}: Q={q}")
        result[q] = pt
    return result


def find_summary(root: Path, dataset: str, method: str) -> tuple[Path, dict[int, Point]]:
    paths = (
        root / dataset / "summary.csv",
        root / method / dataset / "summary.csv",
        root / dataset / method / "summary.csv",
    )
    for path in paths:
        if path.is_file():
            rows = read_summary(path, dataset, method)
            if rows:
                return path, rows
    raise FileNotFoundError(
        f"No {method} summary for {dataset}. Checked: " +
        ", ".join(str(p) for p in paths))


def load_dataset(root: Path, dataset: str) -> dict[str, dict[int, Point]]:
    found = {}
    for method in METHODS:
        path, found[method] = find_summary(root, dataset, method)
        print(f"[data] {dataset}/{method}: {len(found[method])} Q values from {path}")
    n_seeds = {p.n_seeds for group in found.values() for p in group.values()}
    if len(n_seeds) != 1:
        raise ValueError(f"Mixed seed counts in {dataset}: {n_seeds}")
    for method in METHODS:
        missing = sorted(set(Q_VALUES) - set(found[method]))
        if missing:
            print(f"[warning] missing {dataset}/{method} Q={missing}; "
                  "the curve will not interpolate", file=sys.stderr)
    return found


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], label=STYLE[m]["label"], color=STYLE[m]["color"],
               linestyle=STYLE[m]["linestyle"], marker=STYLE[m]["marker"],
               markerfacecolor="white", markeredgewidth=0.9,
               markersize=4.0, linewidth=1.5)
        for m in METHODS
    ]


def draw_panel(ax, dataset: str, data: dict[str, dict[int, Point]],
               uncertainty: str) -> None:
    values = []
    for method in METHODS:
        points = data[method]
        ys = [points[q].mean if q in points else float("nan") for q in Q_VALUES]
        errs = ([getattr(points[q], uncertainty) if q in points else float("nan")
                 for q in Q_VALUES] if uncertainty != "none" else None)
        for i, y in enumerate(ys):
            if math.isfinite(y):
                e = errs[i] if errs is not None else 0
                values.extend((y-e, y+e))
        kw = STYLE[method]
        ax.plot(range(len(Q_VALUES)), ys, color=kw["color"],
                linestyle=kw["linestyle"], marker=kw["marker"],
                markersize=4, markerfacecolor="white",
                markeredgewidth=0.9, linewidth=1.5, zorder=3)
        if errs is not None:
            idx = [i for i, y in enumerate(ys) if math.isfinite(y)]
            ax.errorbar(idx, [ys[i] for i in idx], yerr=[errs[i] for i in idx],
                        fmt="none", ecolor=kw["color"], elinewidth=0.8,
                        capsize=1.8, capthick=0.8, zorder=4)

    # Figure 3 uses evenly spaced categorical probe counts, not log-scaled x.
    ax.set_xticks(range(len(Q_VALUES)), [str(q) for q in Q_VALUES])
    ax.set_xlim(-0.2, len(Q_VALUES)-0.8)
    ax.set_xlabel(r"Number of probes $N$", labelpad=4)
    ax.set_ylabel(METRICS[dataset][1], labelpad=4)
    ax.grid(axis="y", color="#d2d2d2", alpha=0.55, linewidth=0.65)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    if values:
        lo, hi = min(values), max(values)
        span = max(hi-lo, 0.025 if dataset == "fmnist_inr" else 0.012)
        ax.set_ylim(max(-1, lo-0.12*span), min(1, hi+0.16*span))
    ax.tick_params(axis="both", direction="out", pad=2)


def save(fig, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = base.with_suffix("." + fmt)
        fig.savefig(path, dpi=dpi, format=fmt,
                    bbox_inches="tight", pad_inches=0.045)
        print(f"[saved] {path}")
    plt.close(fig)


def combined(data: dict, order: list[str], base: Path,
             uncertainty: str, dpi: int) -> None:
    count = len(order)
    fig, axs = plt.subplots(1, count, figsize=(2.30*count+0.35, 2.30),
                            squeeze=False)
    for i, dataset in enumerate(order):
        ax = axs[0][i]
        draw_panel(ax, dataset, data[dataset], uncertainty)
        ax.text(0.5, -0.37, f"({chr(97+i)}) {LABELS[dataset]}",
                ha="center", va="top", fontsize=9, transform=ax.transAxes)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.015),
               columnspacing=1.5, handlelength=2.3, handletextpad=0.45)
    fig.subplots_adjust(left=0.08 if count > 1 else 0.18, right=0.99,
                        bottom=0.30, top=0.79, wspace=0.53 if count > 1 else 0.2)
    save(fig, base, dpi)


def individual(dataset: str, data: dict, base: Path,
               uncertainty: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.60, 2.70))
    draw_panel(ax, dataset, data, uncertainty)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.2, handlelength=2.1, handletextpad=0.35,
               prop={"size": 7.3})
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.24, top=0.82)
    save(fig, base, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "checkpoints/probe_count_3seeds")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None,
                        help="Default: automatically include datasets with both methods present")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--layout", choices=("both", "combined", "individual"),
                        default="both")
    parser.add_argument("--uncertainty", choices=("sem", "sd", "none"), default="sem",
                        help="SEM default; 'none' matches the no-errorbars seed-0 figure")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be >= 72")
    name = args.name or args.input_root.name
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
        parser.error("Invalid output base name")
    if not args.input_root.exists():
        parser.error(f"Input root not found: {args.input_root}")

    requested = args.datasets or DATASETS
    data = {}
    for dataset in requested:
        try:
            data[dataset] = load_dataset(args.input_root, dataset)
        except FileNotFoundError as exc:
            if args.datasets is not None:
                parser.error(str(exc))
            print(f"[skip] {dataset}: {exc}", file=sys.stderr)
    if not data:
        parser.error(f"No paired summaries in {args.input_root}")
    set_paper_style()
    order = [d for d in requested if d in data]
    if args.layout in ("both", "combined"):
        combined(data, order, args.output_dir / name, args.uncertainty, args.dpi)
    if args.layout in ("both", "individual"):
        for dataset in order:
            individual(dataset, data[dataset],
                       args.output_dir / f"{name}_{dataset}",
                       args.uncertainty, args.dpi)
    print(f"[done] {len(order)} dataset(s), uncertainty={args.uncertainty.upper()}")


if __name__ == "__main__":
    main()
