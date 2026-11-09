"""Dry-run tests for MNIST/SVHN regression hidden-layer sweeps (CPU only)."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "HiddenProbe" / "regression"
MASKS = ("0", "1", "2", "01", "02", "12")


def load_driver(dataset):
    path = SCRIPTS / f"{dataset}_hidden_layer_ablation_sweep.py"
    spec = importlib.util.spec_from_file_location(f"ablation_{dataset}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegressionLayerSweepsTests(unittest.TestCase):
    def test_24_sweep_configs_include_published_baseline(self):
        for dataset in ("svhn", "mnist"):
            with self.subTest(dataset=dataset):
                configs = load_driver(dataset).candidates()
                self.assertEqual(len(configs), 24)
                self.assertEqual(len(set(map(str, configs))), 24)
                self.assertTrue(any(
                    cfg["probe_lr"] == "6e-4" and cfg["factor"] == "0.7"
                    and cfg["patience"] == 4 for cfg in configs
                ))

    def test_every_mask_has_correct_dry_run_command(self):
        for dataset in ("svhn", "mnist"):
            for mask in MASKS:
                with self.subTest(dataset=dataset, mask=mask):
                    runner = SCRIPTS / f"run_{dataset}_layers_{mask}_sweep.sh"
                    self.assertTrue(runner.is_file())
                    proc = subprocess.run(
                        [sys.executable, str(SCRIPTS / f"{dataset}_hidden_layer_ablation_sweep.py"),
                         "--layers", *mask, "--dry-run"],
                        cwd=ROOT, text=True, capture_output=True, check=True,
                    )
                    lines = [l for l in proc.stdout.splitlines() if "--hidden_layer_ids" in l]
                    self.assertEqual(len(lines), 24)
                    self.assertTrue(all("--skip_test_eval 1" in l for l in lines))
                    for line in lines:
                        self.assertIn("--hidden_layer_ids " + " ".join(mask), line)
                        self.assertIn("--dataset " + dataset, line)
                        if dataset == "svhn":
                            self.assertIn("--adapter_preset expressive", line)
                            self.assertIn("--interaction_rank 96", line)
                            self.assertIn("--mixer_hidden 384", line)
                        else:
                            self.assertIn("--adapter_preset compact", line)
                            self.assertIn("--interaction_rank 48", line)
                            self.assertIn("--mixer_hidden 256", line)


if __name__ == "__main__":
    unittest.main()
