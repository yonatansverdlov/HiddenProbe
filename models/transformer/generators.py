"""Shared UNCONDITIONED probe generators — G3 (nonlinear) and glin (deep-linear) — Q-parameterized.
The probes are identical for every target (no weight access, no target conditioning).

Banks (Q = n_probes):
  AGNews: encoder_codes [Q,17,32]                                          -> Q encoder probes.
  MNIST : encoder_codes [n_enc,17,32] + image_patch_codes [n_nat,49,16]    -> n_enc+n_nat=Q (3:1 enc:native).
          Q64 -> 48+16 ; Q128 -> 96+32. Global probe IDs 0..n_enc-1 encoder, n_enc..Q-1 native.
G3: x = z + linear_out(GELU(linear_in(z))) (weights shared across codes).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

ENC_TOK = 17
PATCH_TOK = 49
PATCH_DIM = 16
IMG_DIM = 32


def probe_split(dataset: str, n_probes: int):
    """-> (n_enc, n_nat). MNIST uses a 3:1 encoder:native split; AGNews is all-encoder."""
    if dataset == "mnist":
        n_nat = n_probes // 4
        return n_probes - n_nat, n_nat
    return n_probes, 0


def unpatchify(p: torch.Tensor) -> torch.Tensor:
    *lead, n_img, npatch, pdim = p.shape
    assert npatch == PATCH_TOK and pdim == PATCH_DIM, p.shape
    x = p.reshape(*lead, n_img, 7, 7, 4, 4)
    x = x.permute(*range(len(lead)), -5, -4, -2, -3, -1)
    return x.reshape(*lead, n_img, 1, 28, 28)


class _Route(nn.Module):
    def __init__(self, c_in: int, hidden: int):
        super().__init__()
        self.linear_in = nn.Linear(c_in, hidden); self.linear_out = nn.Linear(hidden, c_in)
        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1); nn.init.zeros_(self.linear_out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.linear_in(z))
        return z + self.linear_out(h)


class ProbeGeneratorG3(nn.Module):
    def __init__(self, dataset: str, n_probes: int = 64, seed: int = 0):
        super().__init__()
        self.dataset = dataset; self.n_probes = n_probes
        self.n_enc, self.n_nat = probe_split(dataset, n_probes)
        g = torch.Generator().manual_seed(seed)
        self.encoder_codes = nn.Parameter(torch.randn(self.n_enc, ENC_TOK, IMG_DIM, generator=g))
        self.enc = _Route(IMG_DIM, 64)
        if self.n_nat:
            self.image_patch_codes = nn.Parameter(torch.randn(self.n_nat, PATCH_TOK, PATCH_DIM, generator=g))
            self.patch = _Route(PATCH_DIM, 32)
        else:
            self.patch = None

    def forward(self, meta: int) -> Dict[str, torch.Tensor]:
        enc = self.enc(self.encoder_codes).unsqueeze(0).expand(meta, -1, -1, -1)
        out = {"encoder_probes": enc}
        if self.patch is not None:
            out["native_images"] = unpatchify(self.patch(self.image_patch_codes)).unsqueeze(0).expand(meta, -1, -1, -1, -1)
        return out


class _DeepLinear(nn.Module):
    """§9 shared bias-free deep-linear decoder cin->h->h->cin. NO activation/norm/sigmoid/clip/residual.
    Identity-composite init via factored (A, I, A^T) with A^T A = I_cin (h>=cin) copied into UNtied storage —
    initial output == input codes; tests factorization/optimization, not extra expressivity."""

    def __init__(self, cin: int, h: int):
        super().__init__()
        self.l1 = nn.Linear(cin, h, bias=False)     # W1 = A     [h,cin]
        self.l2 = nn.Linear(h, h, bias=False)       # W2 = I_h   [h,h]
        self.l3 = nn.Linear(h, cin, bias=False)     # W3 = A^T   [cin,h]
        A = torch.empty(h, cin); nn.init.orthogonal_(A)          # columns orthonormal -> A^T A = I_cin
        with torch.no_grad():
            self.l1.weight.copy_(A); self.l2.weight.copy_(torch.eye(h)); self.l3.weight.copy_(A.t())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l2(self.l1(z)))         # composite = A^T I A = I at init


class ProbeGeneratorGLinear(nn.Module):
    """§9 shared UNCONDITIONED deep-linear generator (G-linear-token). Same codes/routes/lengths/Q as G3;
    replaces G3's nonlinear _Route with a factored linear decoder. Encoder 32->64->64->32 (8,192 params),
    native nonspatial 16->32->32->16 (2,048) -> verified unpatchify. Unconditioned, like G3."""

    def __init__(self, dataset: str, n_probes: int = 64, seed: int = 0):
        super().__init__()
        self.dataset = dataset; self.n_probes = n_probes
        self.n_enc, self.n_nat = probe_split(dataset, n_probes)
        g = torch.Generator().manual_seed(seed)
        self.encoder_codes = nn.Parameter(torch.randn(self.n_enc, ENC_TOK, IMG_DIM, generator=g))
        self.enc_dec = _DeepLinear(IMG_DIM, 64)                  # 32->64->64->32
        if self.n_nat:
            self.image_patch_codes = nn.Parameter(torch.randn(self.n_nat, PATCH_TOK, PATCH_DIM, generator=g))
            self.nat_dec = _DeepLinear(PATCH_DIM, 32)            # 16->32->32->16
        else:
            self.nat_dec = None

    def forward(self, meta: int) -> Dict[str, torch.Tensor]:
        enc = self.enc_dec(self.encoder_codes).unsqueeze(0).expand(meta, -1, -1, -1)
        out = {"encoder_probes": enc}
        if self.nat_dec is not None:
            out["native_images"] = unpatchify(self.nat_dec(self.image_patch_codes)).unsqueeze(0).expand(meta, -1, -1, -1, -1)
        return out
