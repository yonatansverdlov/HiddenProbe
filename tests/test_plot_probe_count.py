"""Unit tests for probe-count figure input parsing (no real results needed)."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_probe_count", ROOT / "scripts" / "plot_probe_count.py")
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)

FIELDS = ["dataset", "method", "n_probes", "n_seeds", "test_metric_name",
          "test_metric_mean", "test_metric_sd", "test_metric_sem"]


def write_summary(path, dataset, methods, n_seeds=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for method in methods:
            for q in plot.Q_VALUES:
                writer.writerow({
                    "dataset": dataset, "method": method,
                    "n_probes": q, "n_seeds": n_seeds,
                    "test_metric_name": plot.METRICS[dataset][0],
                    "test_metric_mean": .9 if method == "hiddenprobe" else .88,
                    "test_metric_sd": .004,
                    "test_metric_sem": .004 / n_seeds**.5,
                })


class PlotProbeCountTests(unittest.TestCase):
    def test_split_3seed_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for m in plot.METHODS:
                write_summary(root / m / "mnist_transformer" / "summary.csv",
                              "mnist_transformer", [m])
            d = plot.load_dataset(root, "mnist_transformer")
            self.assertEqual(len(d["hiddenprobe"]), 5)
            self.assertEqual(d["hiddenprobe"][256].n_seeds, 3)
            self.assertEqual(d["probegen"][256].mean, .88)

    def test_combined_5seed_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_summary(root / "fmnist_inr" / "summary.csv",
                          "fmnist_inr", plot.METHODS, 5)
            d = plot.load_dataset(root, "fmnist_inr")
            self.assertEqual(d["hiddenprobe"][128].n_seeds, 5)

    def test_reject_mixed_seed_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_summary(root / "hiddenprobe" / "svhn" / "summary.csv",
                          "svhn", ["hiddenprobe"], 3)
            write_summary(root / "probegen" / "svhn" / "summary.csv",
                          "svhn", ["probegen"], 5)
            with self.assertRaisesRegex(ValueError, "Mixed seed counts"):
                plot.load_dataset(root, "svhn")


if __name__ == "__main__":
    unittest.main()
