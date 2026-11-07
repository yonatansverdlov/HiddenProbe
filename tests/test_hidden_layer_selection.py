"""CPU smoke tests for selective target-INR activation capture.

Run from the repository root:
    python -m unittest discover -s tests -p test_hidden_layer_selection.py
"""
import argparse
import unittest

from models.hidden_layer_selection import parse_bool_argument, resolve_hidden_layer_ids


class HiddenLayerSelectionTests(unittest.TestCase):
    def test_default_all_hidden(self):
        self.assertEqual(resolve_hidden_layer_ids(None, 2, True), (0, 1))
        self.assertEqual(resolve_hidden_layer_ids(["None"], 2, True), (0, 1))

    def test_default_output_only(self):
        self.assertEqual(resolve_hidden_layer_ids(None, 2, False), ())
        self.assertEqual(resolve_hidden_layer_ids(["none"], 2, False), ())

    def test_explicit_layer_selections(self):
        self.assertEqual(resolve_hidden_layer_ids(["0"], 2, True), (0,))
        self.assertEqual(resolve_hidden_layer_ids(["1"], 2, True), (1,))
        self.assertEqual(resolve_hidden_layer_ids(["0", "1"], 2, True), (0, 1))
        self.assertEqual(resolve_hidden_layer_ids(["0,1"], 2, True), (0, 1))
        self.assertEqual(resolve_hidden_layer_ids(["[0, 1]"], 2, True), (0, 1))
        self.assertEqual(resolve_hidden_layer_ids(["1", "0"], 2, True), (1, 0))

    def test_invalid_indices(self):
        for raw in (["-1"], ["2"], ["0,0"], ["x"], ["0", "2"], ["[]"]):
            with self.subTest(ids=raw), self.assertRaises(ValueError):
                resolve_hidden_layer_ids(raw, 2, True)

    def test_selection_while_disabled_is_invalid(self):
        with self.assertRaises(ValueError):
            resolve_hidden_layer_ids(["0"], 2, False)

    def test_boolean_argument(self):
        for value in (True, "1", "yes", "TRUE", "on"):
            self.assertTrue(parse_bool_argument(value))
        for value in (False, "0", "no", "FALSE", "off"):
            self.assertFalse(parse_bool_argument(value))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_bool_argument("maybe")


class CoreProbeSelectionTests(unittest.TestCase):
    """Exercise the actual ProbeGen hook and token construction on CPU."""

    def test_selected_layers_change_token_width(self):
        import torch
        import torch.nn as nn

        from models.probegen_core import ProbeGen

        class TinyINR(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.ModuleList([
                    nn.Linear(2, 32),
                    nn.Linear(32, 32),
                    nn.Linear(32, 1),
                ])

            def forward(self, coords):
                h = torch.sin(30.0 * self.seq[0](coords))
                h = torch.sin(30.0 * self.seq[1](h))
                return self.seq[2](h) + 0.5

        net = TinyINR()
        for raw, include, selected_count in [
            (None, False, 0),
            (None, True, 2),
            ([0], True, 1),
            ([1], True, 1),
            ([0, 1], True, 2),
        ]:
            with self.subTest(raw=raw, include=include):
                model = ProbeGen(
                    n_tokens=4,
                    d_hidden=16,
                    models_c_in=2,
                    models_c_out=1,
                    d_out=10,
                    gen_type="linear_0_no_acts",
                    n_hidden_target_layers=2,
                    include_hidden_features=include,
                    hidden_layer_ids=raw,
                    per_probe_mlp="none",
                    r_per_hidden=2,
                    rank=4,
                    seed=0,
                )
                probes = model.generate_probes()
                tokens = model._build_probe_tokens(net, probes)
                self.assertEqual(tuple(tokens.shape), (4, 1 + 2 * selected_count))
                self.assertEqual(len(model.hidden_layer_ids), selected_count)


if __name__ == "__main__":
    unittest.main()
