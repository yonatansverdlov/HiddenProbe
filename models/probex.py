"""ProbeX single-view weight encoder.

Architecture follows the official ProbeX implementation from:
  Horwitz et al., "Learning on Model Weights using Tree Experts", CVPR 2025.

For X in R^{d_H x d_W}, ProbeX uses one learned first-order view:
  XU -> shared projection -> ReLU -> flatten -> representation encoder.

Task heads below adapt the encoder to single-label classification and scalar
regression while preserving the original ProbeX encoder computation.
"""
from __future__ import annotations

import torch
from torch import nn


class ProbeXCore(nn.Module):
    def __init__(self, input_shape, n_probes: int, proj_dim: int, rep_dim: int):
        super().__init__()
        d_h, d_w = int(input_shape[0]), int(input_shape[1])

        # Official ProbeX parameterization: weight has shape (d_w, n_probes).
        self.probes = nn.Linear(n_probes, d_w, bias=False)
        self.shared_probe_proj = nn.Linear(d_h, proj_dim, bias=False)
        self.per_probe_encoder = nn.Linear(proj_dim * n_probes, rep_dim)
        self.rep_dim_out = int(rep_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        probe_responses = x @ self.probes.weight
        projected = self.shared_probe_proj(probe_responses.transpose(1, 2))
        projected = torch.relu(projected)
        flat = projected.reshape(projected.shape[0], -1)
        return self.per_probe_encoder(flat)


class ProbeXClassification(ProbeXCore):
    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        n_classes: int,
    ):
        super().__init__(input_shape, n_probes, proj_dim, rep_dim)
        self.classification_head = nn.Linear(self.rep_dim_out, int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.encode(x))


class ProbeXRegression(ProbeXCore):
    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
    ):
        super().__init__(input_shape, n_probes, proj_dim, rep_dim)
        self.regression_head = nn.Linear(self.rep_dim_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.encode(x)).squeeze(-1)
