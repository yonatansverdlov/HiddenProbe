"""CPU unit tests for SVHN-style masked CNN hidden activation capture."""
import unittest
import torch
import torch.nn as nn
from models.lr_dipt_capture import capture_features, pad_batch, META_DIM


class ToyThreeLayerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 4, 3, 2, 1),
            nn.Conv2d(4, 4, 3, 2, 1),
            nn.Conv2d(4, 4, 3, 2, 1),
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        for conv, activation in zip(self.layers, self.activations):
            x = activation(conv(x))
        return self.fc(self.flatten(self.pool(x)))


class HiddenLayerSelectionTests(unittest.TestCase):
    def test_six_selections_preserve_final_logits_and_layer_metadata(self):
        torch.manual_seed(1)
        net = ToyThreeLayerCNN()
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        baseline, all_records = capture_features(net, x)
        self.assertEqual(len(all_records), 3)
        torch.testing.assert_close(baseline, net(x), rtol=0, atol=0)

        for ids in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
            with self.subTest(ids=ids):
                logits, records = capture_features(net, x, layer_ids=ids)
                torch.testing.assert_close(logits, baseline, rtol=0, atol=0)
                self.assertEqual(len(records), len(ids))
                for i, record in zip(ids, records):
                    self.assertEqual(record["meta"], all_records[i]["meta"])
                    torch.testing.assert_close(record["cells"], all_records[i]["cells"])
                    torch.testing.assert_close(record["glob"], all_records[i]["glob"])
                batch = pad_batch([records], META_DIM, P=2, device="cpu")
                self.assertEqual(batch["lmask"].shape, (1, len(ids)))
                self.assertTrue(bool(batch["lmask"].all()))
                grad = torch.autograd.grad(
                    logits.sum() + sum(rec["glob"].sum() for rec in records), x,
                    retain_graph=True,
                )[0]
                self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_invalid_masks_rejected(self):
        net = ToyThreeLayerCNN()
        x = torch.zeros(2, 1, 32, 32)
        for ids in ([], [0, 0], [2, 1], [-1], [3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                capture_features(net, x, layer_ids=ids)


if __name__ == "__main__":
    unittest.main()
