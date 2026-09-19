"""Low-rank building blocks for LR-DIPT (Low-Rank Dual-Interaction Probe Transformer).

Every big dense matrix is factorized U@V so the whole readout stays within ~1.05x Kahana's 591,361.
Reuses the U@V pattern of pat.lowrank.LowRankLinear; adds an optional mid-nonlinearity, zero-init
up-projection (identity-start residual blocks), masked low-rank attention (Q/K/V: d->r, O: r->d),
attention pooling, low-rank bilinear, and a Tucker probe pool. No dense d*d projections anywhere.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedLinear(nn.Module):
    """W in R^{out x in} ~= U @ V with rank r. x -> V(x) -> [act] -> U(.). Params = r*(in+out)+bias.
    zero_init_up makes the block start at ~0 (for identity-start residuals)."""
    def __init__(self, in_features, out_features, rank, bias=True, act=None, zero_init_up=False):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)          # V: in -> r
        self.up = nn.Linear(rank, out_features, bias=bias)            # U: r  -> out
        self.act = act
        # He-style init on the effective product (see models.lowrank rationale)
        nn.init.normal_(self.down.weight, std=(1.0 / in_features) ** 0.5)
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.normal_(self.up.weight, std=(2.0 / rank) ** 0.5)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.down(x)
        if self.act is not None:
            h = self.act(h)
        return self.up(h)


class LowRankResidualBlock(nn.Module):
    """Pre-norm residual FF with a rank-r bottleneck; up-proj zero-init => starts at identity."""
    def __init__(self, width, rank, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = FactorizedLinear(width, width, rank, bias=True, act=nn.GELU(), zero_init_up=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ff(self.norm(x)))


class LowRankAttention(nn.Module):
    """Masked scaled-dot-product attention with rank-r Q/K/V and O: r->d. One head over the rank dim
    (rank plays the role of the projected width). Supports query_len != key_len, key padding mask
    (True=pad), and an additive relation bias [.., Lq, Lk]."""
    def __init__(self, d_model, rank, out_dim=None):
        super().__init__()
        out_dim = out_dim or d_model
        self.q = nn.Linear(d_model, rank, bias=False)
        self.k = nn.Linear(d_model, rank, bias=False)
        self.v = nn.Linear(d_model, rank, bias=False)
        self.o = nn.Linear(rank, out_dim, bias=True)
        self.scale = 1.0 / math.sqrt(rank)
        nn.init.zeros_(self.o.bias)

    def forward(self, query, key, value=None, key_padding_mask=None, rel_bias=None):
        # query [.., Lq, d], key/value [.., Lk, d]. key_padding_mask [.., Lk] True=pad.
        value = key if value is None else value
        q, k, v = self.q(query), self.k(key), self.v(value)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale               # [.., Lq, Lk]
        if rel_bias is not None:
            att = att + rel_bias
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask.unsqueeze(-2), float("-inf"))
        # rows that are fully masked -> uniform-zero (avoid NaN); caller masks such outputs downstream
        allpad = torch.isneginf(att).all(dim=-1, keepdim=True)
        att = att.masked_fill(allpad, 0.0)
        w = F.softmax(att, dim=-1)
        w = torch.nan_to_num(w, 0.0)
        return self.o(torch.matmul(w, v))                                     # [.., Lq, out_dim]


class LowRankAttentionPool(nn.Module):
    """Pool a set [.., L, d] -> [.., out_dim] via a learned seed query + low-rank attention."""
    def __init__(self, d_model, rank, out_dim=None):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.attn = LowRankAttention(d_model, rank, out_dim)

    def forward(self, x, key_padding_mask=None):
        lead = x.shape[:-2]
        q = self.seed.expand(*lead, 1, -1)
        return self.attn(q, x, x, key_padding_mask=key_padding_mask).squeeze(-2)


class LowRankBilinear(nn.Module):
    """Low-rank bilinear interaction: (A x) elementwise* (B y), each -> rank. Output dim = rank."""
    def __init__(self, dx, dy, rank):
        super().__init__()
        self.a = nn.Linear(dx, rank, bias=False)
        self.b = nn.Linear(dy, rank, bias=False)

    def forward(self, x, y):
        return self.a(x) * self.b(y)                                          # [.., rank]


class TuckerPool(nn.Module):
    """Pool fixed-role tokens [.., P, feat] -> flat [.., r_probe*r_feat] via T = U_p^T @ X @ V_f.
    Preserves probe-slot identity without a huge P*feat dense layer."""
    def __init__(self, num_probes, feat, r_probe, r_feat):
        super().__init__()
        self.up = nn.Parameter(torch.randn(num_probes, r_probe) * (1.0 / num_probes) ** 0.5)
        self.vf = nn.Parameter(torch.randn(feat, r_feat) * (1.0 / feat) ** 0.5)
        self.out_dim = r_probe * r_feat

    def forward(self, x):                                                     # x [.., P, feat]
        t = torch.einsum("...pf,pr->...rf", x, self.up)                       # [.., r_probe, feat]
        t = torch.einsum("...rf,fs->...rs", t, self.vf)                       # [.., r_probe, r_feat]
        return t.flatten(-2)                                                  # [.., r_probe*r_feat]


def count_params(module, trainable_only=True):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
