"""Single-block Transformer ablations retain only selected block + norm + logits."""
import unittest

import torch

from models.transformer.readout_channeltraj import ChannelTrajectoryReadout


def fake_routes():
    torch.manual_seed(137)
    routes = []
    n0 = 0
    for route, (q, tokens) in enumerate(((12, 17), (4, 49))):
        n1 = n0 + q
        routes.append({
            "b1": torch.randn(1, q, tokens, 32),
            "b2": torch.randn(1, q, tokens, 32),
            "fn": torch.randn(1, q, tokens, 32),
            "logits": torch.randn(1, q, 10),
            "pids": torch.arange(n0, n1),
            "route": route,
            "mask": None,
        })
        n0 = n1
    return routes


def changed(routes, key):
    return [{**r, key: r[key] + 10.0} for r in routes]


class SelectedBlockTests(unittest.TestCase):
    def make_model(self, block):
        return ChannelTrajectoryReadout(
            n_probes=16, n_classes=10, n_routes=2,
            route_lens=[17, 49], dropout=0.0, ffn=384,
            hidden_block_only=block,
        ).eval()

    def test_block1_never_sees_block2(self):
        model = self.make_model(1)
        routes = fake_routes()
        a = model.tokens(routes)
        self.assertTrue(torch.equal(a, model.tokens(changed(routes, "b2"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "b1"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "fn"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "logits"))))

    def test_block2_never_sees_block1(self):
        model = self.make_model(2)
        routes = fake_routes()
        a = model.tokens(routes)
        self.assertTrue(torch.equal(a, model.tokens(changed(routes, "b1"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "b2"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "fn"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "logits"))))

    def test_ablation_architectures_have_equal_capacity(self):
        a = self.make_model(1)
        b = self.make_model(2)
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in b.parameters()),
        )
        self.assertEqual(tuple(a.tokens(fake_routes()).shape), (1, 81, 192))
        self.assertEqual(tuple(b.tokens(fake_routes()).shape), (1, 81, 192))

    def test_full_model_still_uses_both_blocks(self):
        model = self.make_model(0)
        routes = fake_routes()
        a = model.tokens(routes)
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "b1"))))
        self.assertFalse(torch.equal(a, model.tokens(changed(routes, "b2"))))
        self.assertEqual(model.arch_version, "channel_trajectory_v1")


if __name__ == "__main__":
    unittest.main()
