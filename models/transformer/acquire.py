"""Acquisition + prediction: generator -> Q64 probes -> frozen teacher responses -> tokenizer -> scalar.

Serial per-target execution is the correctness reference (spec §3.4). Teacher tensors are constants
(requires_grad=False leaves); gradients flow loss->predictor->responses->probes->generator.
The response-only boundary is enforced structurally: the predictor sees ONLY tokenized responses + fixed
semantic IDs (probe/channel/route), never weights/inputs/labels.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
from . import teacher as T

ROUTE_ENC, ROUTE_NAT = 0, 1
CHANNELS = ("block1", "block2", "final_norm")


def _tokenize_target(tok, enc_out: Dict[str, torch.Tensor], nat_out: Optional[Dict[str, torch.Tensor]]):
    """-> [256, w] summary tokens for one target (3 hidden channels * 64 probes + 64 logit)."""
    dev = enc_out["block1"].device
    n_enc = enc_out["block1"].shape[0]
    enc_p = torch.arange(n_enc, device=dev); enc_r = torch.full((n_enc,), ROUTE_ENC, device=dev)
    if nat_out is not None:
        n_nat = nat_out["block1"].shape[0]
        nat_p = torch.arange(n_enc, n_enc + n_nat, device=dev)
        nat_r = torch.full((n_nat,), ROUTE_NAT, device=dev)   # global IDs after encoder
    chan = []
    for ci, ch in enumerate(CHANNELS):
        parts = [tok.tokenize_hidden(enc_out[ch], ci, enc_p, enc_r)]
        if nat_out is not None:
            parts.append(tok.tokenize_hidden(nat_out[ch], ci, nat_p, nat_r))
        chan.append(torch.cat(parts, 0))                                  # [64, w]
    lparts = [tok.tokenize_logits(enc_out["logits"], enc_p, enc_r)]
    if nat_out is not None:
        lparts.append(tok.tokenize_logits(nat_out["logits"], nat_p, nat_r))
    chan.append(torch.cat(lparts, 0))                                     # [64, w] logit tokens
    return torch.cat(chan, 0)                                             # [256, w]


def _const_responses(n_probes: int, n_tok: int, C: int, dtype, like: torch.Tensor):
    """Fixed constant responses (severed-response fixture): independent of teacher weights/context."""
    g = torch.Generator().manual_seed(777)
    def c(*s):
        return (torch.randn(*s, generator=g, dtype=dtype)).to(like.device)
    return {"block1": c(n_probes, n_tok, T.EMBED_DIM), "block2": c(n_probes, n_tok, T.EMBED_DIM),
            "final_norm": c(n_probes, n_tok, T.EMBED_DIM), "logits": c(n_probes, C)}


def compute_probes(system, teachers: List[Dict[str, torch.Tensor]]):
    """Return the probe banks (shared unconditioned generator: identical across targets)."""
    return system.generator(len(teachers))


def _maybe(d, fn):
    return {k: fn(v) for k, v in d.items()} if d is not None else None


def _build_contract(enc, nat, dev):
    """Assemble the response-only contract (spec §2/readouts.py) from the batched teacher outputs.
    enc/nat: {block1,block2,final_norm:[meta,Qr,n,32], logits:[meta,Qr,C]}. No weights/ids/context leak."""
    n_enc = enc["block1"].shape[1]
    routes = [{"b1": enc["block1"], "b2": enc["block2"], "fn": enc["final_norm"], "logits": enc["logits"],
               "pids": torch.arange(n_enc, device=dev), "route": 0, "mask": None}]
    if nat is not None:
        n_nat = nat["block1"].shape[1]
        routes.append({"b1": nat["block1"], "b2": nat["block2"], "fn": nat["final_norm"], "logits": nat["logits"],
                       "pids": torch.arange(n_enc, n_enc + n_nat, device=dev), "route": 1, "mask": None})
    return routes


def _predict_batched(system, teachers, probes, dev) -> torch.Tensor:
    """Fast path: run the frozen teacher on ALL targets at once via vmap over the target axis, instead of a
    serial per-target python loop (that loop capped GPU util ~27%). Numerically equal to the serial path
    (pure-functional teacher core). Used only for the plain training/eval case; controls use the serial loop.
    All targets in a batch share the SmallZoo architecture (C is fixed per dataset via the C-filter)."""
    meta = len(teachers)
    # stack only keys present in EVERY teacher (real checkpoints have inconsistent extra keys, e.g. a spurious
    # queries.bias in some); the intersection still contains every key the teacher core reads (else the serial
    # path would fail too), and the extras are unused.
    common = set(teachers[0].keys())
    for t in teachers[1:]:
        common &= t.keys()
    ps = {k: torch.stack([t[k] for t in teachers]).to(dev) for k in teachers[0].keys() if k in common}
    enc = torch.func.vmap(T.encoder_route)(ps, probes["encoder_probes"])                  # dict of [meta, ...]
    nat = None
    if "native_images" in probes:
        # conv patch-embed per target in a loop (cheap; vmap-over-conv2d is unsupported on CUDA), THEN vmap
        # only the encoder over the embedded tokens (the expensive part, which vmaps fine on CUDA).
        emb = torch.stack([T.embed_vision({k: v[m] for k, v in ps.items()}, probes["native_images"][m])
                           for m in range(meta)])
        nat = torch.func.vmap(T.encoder_route)(ps, emb)
    if getattr(system.cfg, "readout_arch", "r0") != "r0":            # architecture readouts consume raw responses
        return system.readout(_build_contract(enc, nat, dev))
    toks = []
    for m in range(len(teachers)):
        enc_m = {k: v[m] for k, v in enc.items()}
        nat_m = {k: v[m] for k, v in nat.items()} if nat is not None else None
        toks.append(_tokenize_target(system.tokenizer, enc_m, nat_m))
    return system.predictor(torch.stack(toks))


def predict(system, teachers: List[Dict[str, torch.Tensor]], sever: bool = False,
            detach_responses: bool = False, no_grad_teacher: bool = False,
            force_serial: bool = False) -> torch.Tensor:
    """-> raw scalar prediction [meta].
    Fast batched teacher (vmap) by default; the serial loop is used for the controls and for force_serial
    (parity test). sever=True: replace teacher responses with fixed constants (severed-response test).
    detach_responses / no_grad_teacher: negative controls that must break the probe/generator grad path."""
    probes = compute_probes(system, teachers)
    prm = next(system.parameters()); dtype = prm.dtype; dev = prm.device
    if not (sever or detach_responses or no_grad_teacher or force_serial):
        return _predict_batched(system, teachers, probes, dev)
    toks = []
    for m, p in enumerate(teachers):
        p = {k: v.to(dev) for k, v in p.items()}          # move this target's frozen weights to the model device
        C = T.classifier_out_dim(p)
        if sever:
            like = probes["encoder_probes"]
            n_enc, n_tok = probes["encoder_probes"].shape[1], probes["encoder_probes"].shape[2]
            enc_out = _const_responses(n_enc, n_tok, C, dtype, like)
            nat_out = (_const_responses(probes["native_images"].shape[1], 49, C, dtype, like)
                       if "native_images" in probes else None)
        else:
            ctx = torch.no_grad() if no_grad_teacher else torch.enable_grad()
            with ctx:
                enc_out = T.encoder_route(p, probes["encoder_probes"][m])
                nat_out = T.full_vision(p, probes["native_images"][m]) if "native_images" in probes else None
            if detach_responses:
                enc_out = _maybe(enc_out, lambda t: t.detach()); nat_out = _maybe(nat_out, lambda t: t.detach())
        toks.append(_tokenize_target(system.tokenizer, enc_out, nat_out))
    return system.predictor(torch.stack(toks))                            # [meta]
