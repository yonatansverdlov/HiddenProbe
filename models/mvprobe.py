"""MVProbe / ProbeX four-view weight encoder.

Adapted from the official MVProbe implementation:
  https://github.com/AI-hew-math/MVProbe
  "What Linear Probes Miss: Multi-View Probing for Weight-Space Learning"
  arXiv:2605.23410 / ICML 2026.

The four-view encoder below preserves the released implementation semantics.
Only a scalar regression head is added for our model-accuracy prediction tasks.

Copyright (c) 2026 The MVProbe Authors.
Released under the MIT License; see LICENSES/MVProbe_LICENSE.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _standardize_per_sample(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Official MVProbe per-sample standardization over (N_points, n_probes)."""
    mean = S.mean(dim=(1, 2), keepdim=True)
    std = S.std(dim=(1, 2), keepdim=True)
    return (S - mean) / (std + eps)


class ProbeXCore(nn.Module):
    """Official four-view MVProbe encoder: XU, XX^TU, X^TU, X^TXU."""

    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__()
        self.x_center = x_center
        self.x_row_norm = x_row_norm

        d_H, d_W = int(input_shape[0]), int(input_shape[1])

        self.probes_xu = nn.Linear(n_probes, d_W, bias=False)
        self.probes_xxtu = nn.Linear(n_probes, d_H, bias=False)
        self.probes_xtu = nn.Linear(n_probes, d_H, bias=False)
        self.probes_xtxu = nn.Linear(n_probes, d_W, bias=False)

        self.shared_probe_proj_row = nn.Linear(d_H, proj_dim, bias=False)
        self.shared_probe_proj_row2 = nn.Linear(d_H, proj_dim, bias=False)
        self.shared_probe_proj_col = nn.Linear(d_W, proj_dim, bias=False)
        self.shared_probe_proj_col2 = nn.Linear(d_W, proj_dim, bias=False)

        # This intentionally matches the released code: each branch keeps
        # n_probes * proj_dim features before the final encoder.
        self.concat_all4_encoder = nn.Linear(4 * proj_dim * n_probes, rep_dim)
        self.rep_dim_out = int(rep_dim)

    def _preprocess_points(self, Xpc: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.x_center:
            Xpc = Xpc - Xpc.mean(dim=1, keepdim=True)
        if self.x_row_norm:
            Xpc = F.normalize(Xpc, dim=-1, eps=eps)
        return Xpc

    def encode(self, x: torch.Tensor, active_branches: Optional[List[str]] = None):
        X_row = self._preprocess_points(x)
        X_col = self._preprocess_points(x.transpose(1, 2))

        def proj_flat(pr: torch.Tensor, proj_layer: nn.Linear) -> torch.Tensor:
            pr_t = pr.transpose(1, 2)
            pr_proj = torch.relu(proj_layer(pr_t))
            return pr_proj.reshape(pr_proj.shape[0], -1)

        pr_xu = _standardize_per_sample(X_row @ self.probes_xu.weight)
        pr_xxtu = _standardize_per_sample(
            X_row @ (X_row.transpose(1, 2) @ self.probes_xxtu.weight)
        )
        pr_xtu = _standardize_per_sample(X_col @ self.probes_xtu.weight)
        pr_xtxu = _standardize_per_sample(
            X_col @ (X_col.transpose(1, 2) @ self.probes_xtxu.weight)
        )

        f_xu = proj_flat(pr_xu, self.shared_probe_proj_row)
        f_xxtu = proj_flat(pr_xxtu, self.shared_probe_proj_row2)
        f_xtu = proj_flat(pr_xtu, self.shared_probe_proj_col)
        f_xtxu = proj_flat(pr_xtxu, self.shared_probe_proj_col2)

        keep = (
            set(active_branches)
            if active_branches is not None
            else {"xu", "xxtu", "xtu", "xtxu"}
        )
        if "xu" not in keep:
            f_xu = f_xu * 0.0
        if "xxtu" not in keep:
            f_xxtu = f_xxtu * 0.0
        if "xtu" not in keep:
            f_xtu = f_xtu * 0.0
        if "xtxu" not in keep:
            f_xtxu = f_xtxu * 0.0

        f = torch.cat([f_xu, f_xxtu, f_xtu, f_xtxu], dim=-1)
        rep = self.concat_all4_encoder(f)

        pr_ret = pr_xu
        if active_branches is not None:
            pr_map = {
                "xu": pr_xu,
                "xxtu": pr_xxtu,
                "xtu": pr_xtu,
                "xtxu": pr_xtxu,
            }
            for branch in ("xu", "xxtu", "xtu", "xtxu"):
                if branch in keep:
                    pr_ret = pr_map[branch]
                    break
        return pr_ret, rep


class ProbeXRegression(ProbeXCore):
    """Minimal task adaptation: official MVProbe encoder + scalar accuracy head."""

    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__(
            input_shape=input_shape,
            n_probes=n_probes,
            proj_dim=proj_dim,
            rep_dim=rep_dim,
            x_center=x_center,
            x_row_norm=x_row_norm,
        )
        self.regression_head = nn.Linear(self.rep_dim_out, 1)

    def forward(self, x):
        _, representation = self.encode(x)
        return self.regression_head(representation).squeeze(-1)


class ProbeXClassification(ProbeXCore):
    """Official MVProbe encoder + linear classification head."""

    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        n_classes: int,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__(
            input_shape=input_shape,
            n_probes=n_probes,
            proj_dim=proj_dim,
            rep_dim=rep_dim,
            x_center=x_center,
            x_row_norm=x_row_norm,
        )
        self.classification_head = nn.Linear(self.rep_dim_out, int(n_classes))

    def forward(self, x):
        _, representation = self.encode(x)
        return self.classification_head(representation)
