"""Low-rank factorized Linear for parameter-matched PAT ablations.

A shared Linear W:(in->out) is replaced by W = U @ V, i.e. down:(in->r) then up:(r->out).
Params: in*r + r*out (+ out bias)  vs  in*out (+ out) for the full Linear.

CRITICAL (invariance): PAT's S_N x prod S_{d_l} equivariance comes from every per-position
transform being SHARED across the probe (N) and neuron (D) axes. Factoring a shared weight
into two shared weights keeps it shared -> the factorized module is still applied identically
at every position, so the exact-invariance identity is preserved (verified by the fp64 suite).

`make_linear` is a drop-in for nn.Linear that factorizes ONLY when a positive rank is given
AND the factorization is actually smaller (rank < in*out/(in+out)); otherwise it returns a
plain nn.Linear. rank<=0 => full Linear => bit-for-bit identical to the un-factored model.
"""
from __future__ import annotations

import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True,
                 variance_preserving: bool = False):
        super().__init__()
        if rank <= 0:
            raise ValueError("LowRankLinear needs rank>0")
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False)   # U^T : in -> r
        self.up = nn.Linear(rank, out_features, bias=bias)     # V   : r  -> out
        if variance_preserving:
            # Default nn.Linear init makes the PRODUCT W=up@down ~3x smaller-variance than a dense
            # Linear, so deep low-rank stacks (e.g. the 8-layer ST head, no LayerNorm) vanish to a
            # constant output at init (init_out_std=0 -> stuck at the ln(K) plateau). Fix: init so
            # Var(W_eff)=2/in (He): Var(down)=1/in, Var(up)=2/rank -> rank*(2/rank)*(1/in)=2/in.
            nn.init.normal_(self.down.weight, std=(1.0 / in_features) ** 0.5)
            nn.init.normal_(self.up.weight, std=(2.0 / rank) ** 0.5)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.down(x))


def make_linear(in_features: int, out_features: int, rank: int = 0, bias: bool = True) -> nn.Module:
    """nn.Linear, or LowRankLinear when rank>0 and it saves parameters."""
    full = in_features * out_features
    lr = rank * (in_features + out_features)
    if rank and rank > 0 and lr < full:
        return LowRankLinear(in_features, out_features, rank, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
