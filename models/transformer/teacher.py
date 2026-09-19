"""Frozen SmallZoo transformer teachers (Transformer-NFN vision/text zoos) — reproduce, not approximate.

Verbatim architecture (github.com/MathematicalAI-NUS/Transformer-NFN @ fcb4663):
  * embed_dim d=32, 2 heads (head_dim 16), 2 blocks, forward_mul 2 (fc 32->64->32, biased, ReLU).
  * attention: bias-free queries/keys/values/out_projection; attn = softmax(QK^T / sqrt(head_dim), dim=-1).
  * NO residual additions (each block line overwrites x): block = FFN(norm2(attn(norm1(x)))).
  * non-affine LayerNorm (elementwise_affine=False, eps=1e-5) at norm1, norm2, and the final norm.
  * classifier: token-mean pool -> fc1(32,32) ReLU -> fc2(32,C), biased.
  * MNIST embed: conv1 (1->32, k=4, s=4) -> [B,49,32] + non-learnable sinusoidal pos_embedding[49,32].
  * AGNews embed: shared frozen Word2Vec lookup + sinusoidal pos (per-checkpoint embedding omitted).

This module is FUNCTIONAL: the core operates on a plain params dict keyed by the checkpoint's own
state_dict names, so it runs released checkpoints directly and supports per-target meta-execution via
distinct frozen tensor sets (no repeated load_state_dict on one mutable module). Gradients flow through
the INPUT (probes) while teacher tensors are treated as constants (detached at the leaves by the caller).
The `SmallZooTransformer` nn.Module is an INDEPENDENT reference implementation used only for parity tests.
"""
from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5            # nn.LayerNorm default; teacher uses non-affine LN
EMBED_DIM = 32
N_HEADS = 2
N_BLOCKS = 2
FORWARD_MUL = 2       # fc1: 32 -> 64


def sinusoidal_pos(embed_dim: int, length: int) -> torch.Tensor:
    """Non-learnable sinusoidal PE, verbatim from the source (getPositionalEncoding)."""
    position = torch.arange(length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=EPS)


def _attention(p: Dict[str, torch.Tensor], i: int, x: torch.Tensor, n_heads: int = N_HEADS) -> torch.Tensor:
    """Bias-free multi-head self-attention, arbitrary leading (meta, query, ...) dims. Matches source math."""
    *lead, n, e = x.shape
    hd = e // n_heads
    q = F.linear(x, p[f"encoder.{i}.attention.queries.weight"]).reshape(*lead, n, n_heads, hd)
    k = F.linear(x, p[f"encoder.{i}.attention.keys.weight"]).reshape(*lead, n, n_heads, hd)
    v = F.linear(x, p[f"encoder.{i}.attention.values.weight"]).reshape(*lead, n, n_heads, hd)
    q = q.movedim(-2, -3); k = k.movedim(-2, -3); v = v.movedim(-2, -3)     # [..., H, n, hd]
    attn = torch.matmul(q, k.transpose(-1, -2)) / float(hd) ** 0.5           # [..., H, n, n]
    attn = torch.softmax(attn, dim=-1)
    o = torch.matmul(attn, v).movedim(-3, -2).reshape(*lead, n, e)           # [..., n, e]
    return F.linear(o, p[f"encoder.{i}.attention.out_projection.weight"])


def _block(p: Dict[str, torch.Tensor], i: int, x: torch.Tensor, n_heads: int = N_HEADS) -> torch.Tensor:
    """One encoder block. NO residual (verbatim): x <- attn(norm1(x)); x <- fc2(relu(fc1(norm2(x))))."""
    a = _attention(p, i, _ln(x), n_heads)                                    # dropout is identity in eval
    h = F.relu(F.linear(_ln(a), p[f"encoder.{i}.fc1.weight"], p[f"encoder.{i}.fc1.bias"]))
    return F.linear(h, p[f"encoder.{i}.fc2.weight"], p[f"encoder.{i}.fc2.bias"])


def _classifier(p: Dict[str, torch.Tensor], final_norm: torch.Tensor) -> torch.Tensor:
    pooled = final_norm.mean(dim=-2)                                         # token-mean pool
    h = F.relu(F.linear(pooled, p["classifier.fc1.weight"], p["classifier.fc1.bias"]))
    return F.linear(h, p["classifier.fc2.weight"], p["classifier.fc2.bias"])


def encoder_route(p: Dict[str, torch.Tensor], x: torch.Tensor, n_heads: int = N_HEADS,
                  n_blocks: int = N_BLOCKS) -> Dict[str, torch.Tensor]:
    """THE locked observation boundary: x is [..., n, 32] AFTER embedding+position, immediately before
    block 1. Runs both blocks, final norm, mean-pool classifier. Returns the four observation channels."""
    assert x.shape[-1] == EMBED_DIM, f"encoder route expects last dim {EMBED_DIM}, got {x.shape}"
    b1 = _block(p, 0, x, n_heads)
    b2 = _block(p, 1, b1, n_heads)
    fn = _ln(b2)
    logits = _classifier(p, fn)
    return {"block1": b1, "block2": b2, "final_norm": fn, "logits": logits}


def embed_vision(p: Dict[str, torch.Tensor], image: torch.Tensor) -> torch.Tensor:
    """MNIST native route: conv patchify (k=s=4) -> [.., tokens, 32] + non-learnable sinusoidal pos.
    Differentiable w.r.t. `image` (F.conv2d only supports one leading batch, so flatten/unflatten)."""
    *lead, c, hh, ww = image.shape
    w = p["embedding.conv1.weight"]; b = p["embedding.conv1.bias"]
    ph = w.shape[-1]
    flat = image.reshape(-1, c, hh, ww)
    z = F.conv2d(flat, w, b, stride=ph)                                      # [B, 32, H/4, W/4]
    z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)               # [B, tokens, 32]
    pos = p["embedding.pos_embedding"]                                       # non-learnable [tokens, 32]
    z = z + pos
    return z.reshape(*lead, z.shape[-2], z.shape[-1])


def full_vision(p: Dict[str, torch.Tensor], image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Full native MNIST forward: embed then encoder route. logits must equal the reference model."""
    return encoder_route(p, embed_vision(p, image))


def classifier_out_dim(p: Dict[str, torch.Tensor]) -> int:
    return p["classifier.fc2.weight"].shape[0]                              # infer C from the checkpoint


# --------------------------------------------------------------------------------------------------
# Independent reference nn.Module (parity target only; must NOT share forward code with the functional
# core above so a shared bug cannot validate itself). Structure mirrors the verbatim source.
# --------------------------------------------------------------------------------------------------
class _RefAttention(nn.Module):
    def __init__(self, e, h):
        super().__init__(); self.e, self.h, self.hd = e, h, e // h
        self.queries = nn.Linear(e, e, bias=False); self.keys = nn.Linear(e, e, bias=False)
        self.values = nn.Linear(e, e, bias=False); self.out_projection = nn.Linear(e, e, bias=False)

    def forward(self, x):
        b, s, e = x.shape
        xq = self.queries(x).reshape(b, s, self.h, self.hd).permute(0, 2, 1, 3)
        xk = self.keys(x).reshape(b, s, self.h, self.hd).permute(0, 2, 1, 3).permute(0, 1, 3, 2)
        xv = self.values(x).reshape(b, s, self.h, self.hd).permute(0, 2, 1, 3)
        a = torch.softmax(torch.matmul(xq, xk) / float(self.hd) ** 0.5, dim=-1)
        x = torch.matmul(a, xv).permute(0, 2, 1, 3).reshape(b, s, e)
        return self.out_projection(x)


class _RefBlock(nn.Module):
    def __init__(self, e, h, fmul):
        super().__init__()
        self.norm1 = nn.LayerNorm(e, elementwise_affine=False); self.attention = _RefAttention(e, h)
        self.norm2 = nn.LayerNorm(e, elementwise_affine=False)
        self.fc1 = nn.Linear(e, e * fmul); self.activation = nn.ReLU(); self.fc2 = nn.Linear(e * fmul, e)

    def forward(self, x):
        x = self.attention(self.norm1(x))
        x = self.fc2(self.activation(self.fc1(self.norm2(x))))
        return x


class _RefClassifier(nn.Module):
    def __init__(self, e, c):
        super().__init__(); self.fc1 = nn.Linear(e, e); self.activation = nn.ReLU(); self.fc2 = nn.Linear(e, c)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x.mean(dim=1))))


class SmallZooTransformer(nn.Module):
    """Independent reference: load a real/fixture state_dict and forward from the after-embedding boundary
    (encoder route) or, for MNIST, from a native image. Used ONLY by parity tests."""
    def __init__(self, n_classes: int, image_size: int = 28, patch: int = 4, n_channels: int = 1,
                 e: int = EMBED_DIM, h: int = N_HEADS, n_blocks: int = N_BLOCKS, fmul: int = FORWARD_MUL):
        super().__init__()
        tokens = (image_size // patch) ** 2
        self.conv1 = nn.Conv2d(n_channels, e, kernel_size=patch, stride=patch)
        self.pos_embedding = nn.Parameter(sinusoidal_pos(e, tokens), requires_grad=False)
        self.encoder = nn.ModuleList([_RefBlock(e, h, fmul) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(e, elementwise_affine=False)
        self.classifier = _RefClassifier(e, n_classes)

    def load_teacher(self, sd: Dict[str, torch.Tensor]):
        remap = {"embedding.conv1.weight": "conv1.weight", "embedding.conv1.bias": "conv1.bias",
                 "embedding.pos_embedding": "pos_embedding"}
        own = {remap.get(k, k): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(own, strict=False)
        # embedding keys (conv1/pos) are absent for encoder-only teachers (AGNews) -> optional;
        # they're only used by the native MNIST route, never by encoder_route.
        optional = {"pos_embedding", "conv1.weight", "conv1.bias"}
        hard_missing = [m for m in missing if m not in optional]
        assert not hard_missing, f"missing teacher keys: {hard_missing}"
        assert not unexpected, f"unexpected teacher keys: {unexpected}"
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def encoder_route(self, x):
        b1 = self.encoder[0](x); b2 = self.encoder[1](b1); fn = self.norm(b2)
        return {"block1": b1, "block2": b2, "final_norm": fn, "logits": self.classifier(fn)}

    def embed(self, image):
        b = image.shape[0]
        z = self.conv1(image).reshape(b, EMBED_DIM, -1).permute(0, 2, 1) + self.pos_embedding
        return z

    def forward(self, image):
        return self.encoder_route(self.embed(image))


# --------------------------------------------------------------------------------------------------
# Synthetic fixtures (NOT released checkpoints — for plumbing/parity tests only; release parity stays
# UNVERIFIED until real SmallZoo checkpoints are present).
# --------------------------------------------------------------------------------------------------
def make_fixture_mnist(n_classes: int = 10, seed: int = 0, dtype=torch.float64) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    e, h, fmul, tokens = EMBED_DIM, N_HEADS, FORWARD_MUL, 49
    def rn(*s):
        return torch.randn(*s, generator=g, dtype=dtype)
    p: Dict[str, torch.Tensor] = {}
    for i in range(N_BLOCKS):
        for nm in ("queries", "keys", "values", "out_projection"):
            p[f"encoder.{i}.attention.{nm}.weight"] = rn(e, e) * 0.3
        p[f"encoder.{i}.fc1.weight"] = rn(e * fmul, e) * 0.3; p[f"encoder.{i}.fc1.bias"] = rn(e * fmul) * 0.1
        p[f"encoder.{i}.fc2.weight"] = rn(e, e * fmul) * 0.3; p[f"encoder.{i}.fc2.bias"] = rn(e) * 0.1
    p["classifier.fc1.weight"] = rn(e, e) * 0.3; p["classifier.fc1.bias"] = rn(e) * 0.1
    p["classifier.fc2.weight"] = rn(n_classes, e) * 0.3; p["classifier.fc2.bias"] = rn(n_classes) * 0.1
    p["embedding.conv1.weight"] = rn(e, 1, 4, 4) * 0.3; p["embedding.conv1.bias"] = rn(e) * 0.1
    p["embedding.pos_embedding"] = sinusoidal_pos(e, tokens).to(dtype)
    return p
