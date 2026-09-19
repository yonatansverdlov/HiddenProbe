"""Phase-2 architecture readouts (spec §3-4). Response-ONLY: consume raw teacher responses (the contract
built in acquire) + fixed semantic ids; never weights, sketches, context, probe coordinates, target ids,
filenames, metadata or labels.

Contract (per route r): dict with
  b1,b2,fn : [meta, Qr, n_r, 32]   the three hidden views (block1, block2, final_norm)
  logits   : [meta, Qr, C]          actual classifier logits (kept separate from hidden)
  pids     : [Qr] long              GLOBAL probe ids (0..Q-1), for the fixed probe embedding
  route    : int                    route id (0=encoder, 1=native)
  mask     : [meta, Qr, n_r] or None  1=valid token (padding excluded from pooling + stats)

R0  = the existing tokenizer+predictor, unchanged (registry passthrough, lives in predictor.py).
R2  = cross-layer fusion: concat b1|b2|fn per (probe,token) -> 96 feat -> proj -> 4 local queries -> 4
      separate summary tokens/probe + 1 logit token/probe = 5Q+1 global tokens -> pre-LN Transformer.
StatsBypass (§4) = response-only summary statistics -> 32 feats concatenated after final-norm, before head.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

DSTATE = 32
VIEWS = ("b1", "b2", "fn")
HEADS = 6


# ----------------------------------------------------------------------------------------------------
def _masked_token_stats(x: torch.Tensor, mask: Optional[torch.Tensor]):
    """x [meta,Q,n,C] -> per-probe channelwise token mean & population std over VALID tokens: each [meta,Q,C]."""
    x = x.float()
    if mask is None:
        m = x.mean(dim=2)
        v = x.var(dim=2, unbiased=False)
    else:
        w = mask.float().unsqueeze(-1)                    # [meta,Q,n,1]
        cnt = w.sum(dim=2).clamp_min(1.0)                 # [meta,Q,1]
        m = (x * w).sum(dim=2) / cnt
        v = ((x - m.unsqueeze(2)) ** 2 * w).sum(dim=2) / cnt
    return m, v.clamp_min(0.0).sqrt()


def _agg(v: torch.Tensor):
    """aggregate a per-probe quantity [meta,Q,...] across probes -> mean & population std, concatenated on last."""
    return torch.cat([v.mean(dim=1), v.std(dim=1, unbiased=False)], dim=-1)


class StatsBypass(nn.Module):
    """§4 response-only stats -> `out` feats. Per route kept separate; fragile reductions in FP32; masked."""
    ROUTE_FEATS = 3 * (4 * DSTATE + 2) + 6                # per route: 3 views x(mean/std of per-probe mean & std =4*32, + logrms mean/std=2) + logits(rms,ent,margin)*(mean,std)=6

    def __init__(self, n_routes: int, out: int = 32):
        super().__init__()
        self.n_routes = n_routes
        self.fdim = n_routes * self.ROUTE_FEATS
        self.mlp = nn.Sequential(nn.Linear(self.fdim, 64), nn.GELU(), nn.Linear(64, out))

    def _route_feats(self, route: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = route.get("mask")
        feats = []
        for v in VIEWS:                                   # hidden views
            pm, ps = _masked_token_stats(route[v], mask)  # per-probe channel mean/std [meta,Q,32]
            feats.append(_agg(pm)); feats.append(_agg(ps))            # 2*(2*32)=128
            x = route[v].float()
            if mask is None:
                lrms = 0.5 * torch.log((x ** 2).mean(dim=(2, 3)) + 1e-8)      # [meta,Q]
            else:
                w = mask.float().unsqueeze(-1); cnt = w.sum(dim=(2, 3)).clamp_min(1.0)
                lrms = 0.5 * torch.log((x ** 2 * w).sum(dim=(2, 3)) / cnt + 1e-8)
            feats.append(_agg(lrms.unsqueeze(-1)))               # lrms [meta,Q] -> [meta,2]
        lg = route["logits"].float()                      # [meta,Q,C]
        rms = (lg ** 2).mean(dim=-1).clamp_min(0).sqrt()  # [meta,Q]
        p = torch.softmax(lg, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)      # [meta,Q]
        top2 = lg.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1])            # [meta,Q]
        for q in (rms, ent, margin):
            feats.append(torch.stack([q.mean(dim=1), q.std(dim=1, unbiased=False)], dim=-1))  # [meta,2]
        return torch.cat([f.reshape(f.shape[0], -1) for f in feats], dim=-1)                  # [meta, ROUTE_FEATS]

    def forward(self, contract: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        parts = [self._route_feats(r) for r in contract]
        return self.mlp(torch.cat(parts, dim=-1))         # [meta, out]


class PerProbeMoments(nn.Module):
    """§7 per-probe moments -> `out` feats PER PROBE (never aggregated across probes). Raw width 291 for
    teacher width 32: 3 view channel means (96) + 3 view channel population vars (96) + 3 diagonal cross-view
    covariances b1b2/b1fn/b2fn (96) + logit rms/entropy/top2-margin (3). Two-pass centered pop moments, FP32,
    masked before arithmetic; one-token var/cov = 0. Deterministic transforms: asinh(means,covs), log1p(vars,
    margin), stabilized log-rms(logit rms), raw entropy."""
    RAW = 3 * DSTATE + 3 * DSTATE + 3 * DSTATE + 3         # 291

    def __init__(self, out: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(self.RAW, 64), nn.GELU(), nn.Linear(64, out))

    def route_raw(self, r: Dict[str, torch.Tensor]) -> torch.Tensor:
        """-> [meta, Qr, 291] raw per-probe moment features (pre-MLP)."""
        mask = r.get("mask")
        views = [r["b1"].float(), r["b2"].float(), r["fn"].float()]        # each [meta,Qr,n,32]
        if mask is None:
            def mmean(x):
                return x.mean(dim=2)
        else:
            w = mask.float().unsqueeze(-1); cnt = w.sum(dim=2).clamp_min(1.0)
            def mmean(x):
                return (x * w).sum(dim=2) / cnt
        means = [mmean(v) for v in views]                                  # [meta,Qr,32]
        cent = [v - m.unsqueeze(2) for v, m in zip(views, means)]
        vars = [mmean(c * c) for c in cent]                                # population var (0 for one token)
        covs = [mmean(cent[0] * cent[1]), mmean(cent[0] * cent[2]), mmean(cent[1] * cent[2])]
        lg = r["logits"].float()                                          # [meta,Qr,C]
        rms = (lg ** 2).mean(dim=-1).clamp_min(0).sqrt()                  # [meta,Qr]
        logp = torch.log_softmax(lg, dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)                            # [meta,Qr] stable
        top2 = lg.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).clamp_min(0)               # [meta,Qr]
        feats = ([torch.asinh(m) for m in means] + [torch.log1p(v.clamp_min(0)) for v in vars]
                 + [torch.asinh(c) for c in covs]
                 + [(0.5 * torch.log(rms ** 2 + 1e-8)).unsqueeze(-1), ent.unsqueeze(-1),
                    torch.log1p(margin).unsqueeze(-1)])
        return torch.cat(feats, dim=-1)                                   # [meta,Qr,291]

    def forward(self, r: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.route_raw(r))                                # [meta,Qr,out]


# ----------------------------------------------------------------------------------------------------
class R2Readout(nn.Module):
    """Cross-layer fusion before pooling: 4 separate local summaries per probe (fixed slots, not compressed
    to 1) + 1 logit token per probe -> 5Q+1 tokens -> pre-LN Transformer. Optional StatsBypass."""
    N_QUERIES = 4

    def __init__(self, n_probes: int, n_classes: int, n_routes: int, nmax: int, ffn: int,
                 w: int = 192, n_blocks: int = 2, readout: str = "multi", stats: bool = True, dropout: float = 0.1,
                 token_mlp: bool = False, moments: bool = False, signed_mix: bool = False, route_lens=None):
        super().__init__()
        self.w = w; self.readout = readout; self.token_mlp = token_mlp; self.moments = moments
        self.signed_mix = signed_mix
        self.fuse = nn.Linear(len(VIEWS) * DSTATE, w)                    # 96 -> w
        if token_mlp:                                                    # §6 pre-pool nonlinear map (~25k params)
            self.tmlp = nn.Sequential(nn.Linear(w, 64), nn.GELU(), nn.Linear(64, w))
            self.tmlp_alpha = nn.Parameter(torch.tensor(0.1))
        if moments:                                                     # §7 per-probe moments -> concat into logit token
            self.moments_mod = PerProbeMoments(32)
        if signed_mix:                                                   # §8 signed token mixing: learn A[4,n] per bucket
            assert route_lens is not None and len(route_lens) == n_routes, "signed_mix needs route_lens per route"
            self.mix_A = nn.ParameterList([nn.Parameter(torch.empty(self.N_QUERIES, int(L))) for L in route_lens])
            for A in self.mix_A:
                nn.init.normal_(A, std=1.0 / (A.shape[1] ** 0.5))       # std 1/sqrt(n); no softmax, no sign/row constraints
            self.mix_gate = nn.Parameter(torch.tensor(0.1))             # learned scalar gate, init 0.1
        self.tok_pos = nn.Parameter(torch.zeros(nmax, w))
        self.local_q = nn.Parameter(torch.zeros(self.N_QUERIES, w))     # 4 learned local queries
        self.local_mha = nn.MultiheadAttention(w, HEADS, batch_first=True)
        self.slot_emb = nn.Parameter(torch.zeros(self.N_QUERIES, w))    # fixed slot ids for the 4 summaries
        self.logit_proj = nn.Linear(n_classes + (32 if moments else 0), w)
        self.logit_slot = nn.Parameter(torch.zeros(w))
        self.probe_emb = nn.Parameter(torch.zeros(n_probes, w))
        self.route_emb = nn.Parameter(torch.zeros(n_routes, w))
        self.cls = nn.Parameter(torch.zeros(1, 1, w))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(w, HEADS, ffn, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(w)
        self.stats = StatsBypass(n_routes) if stats else None
        for p in (self.tok_pos, self.local_q, self.slot_emb, self.probe_emb, self.route_emb, self.cls):
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.logit_slot, std=0.02)
        ro = (3 * w if readout == "multi" else w) + (32 if stats else 0)
        self.head = nn.Sequential(nn.Linear(ro, w), nn.GELU(), nn.Linear(w, 1))

    def _route_tokens(self, r: Dict[str, torch.Tensor]) -> torch.Tensor:
        b1, b2, fn = r["b1"], r["b2"], r["fn"]
        meta, Qr, n, _ = b1.shape
        h = self.fuse(torch.cat([b1, b2, fn], dim=-1))                        # [meta,Qr,n,w] content, fused BEFORE pooling
        if self.token_mlp:                                                    # §6: nonlinear map on content, pre semantic ids
            h = h + self.tmlp_alpha * self.tmlp(h)
        content = h                                                           # §8 Z: content before semantic ids (post-tmlp if combined)
        h = h + self.tok_pos[:n]
        hflat = h.reshape(meta * Qr, n, self.w)
        q = self.local_q.unsqueeze(0).expand(meta * Qr, -1, -1)
        kpm = None
        if r.get("mask") is not None:
            kpm = (~r["mask"].bool()).reshape(meta * Qr, n)
        summ, _ = self.local_mha(q, hflat, hflat, key_padding_mask=kpm, need_weights=False)  # [meta*Qr,4,w]
        summ = summ.reshape(meta, Qr, self.N_QUERIES, self.w)
        if self.signed_mix:                                                   # §8: add 4 gated signed summaries A@Z to the 4 attn slots
            Zc = content if r.get("mask") is None else content * r["mask"].unsqueeze(-1)  # mask BEFORE mixing
            A = self.mix_A[int(r["route"])][:, :n]                            # [4,n] selected by logical bucket (not padded length)
            summ = summ + self.mix_gate * torch.einsum("sn,mqnw->mqsw", A, Zc)
        summ = summ + self.slot_emb + self.probe_emb[r["pids"]].unsqueeze(1) + self.route_emb[r["route"]]
        summ = summ.reshape(meta, Qr * self.N_QUERIES, self.w)                 # 4 separate tokens/probe
        lg_in = r["logits"]
        if self.moments:                                                       # §7: concat 32 moment feats into logit token
            lg_in = torch.cat([lg_in, self.moments_mod(r)], dim=-1)
        lg = self.logit_proj(lg_in) + self.logit_slot \
            + self.probe_emb[r["pids"]] + self.route_emb[r["route"]]           # [meta,Qr,w] one logit token/probe
        return torch.cat([summ, lg], dim=1)

    def forward(self, contract: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        meta = contract[0]["b1"].shape[0]
        toks = torch.cat([self._route_tokens(r) for r in contract], dim=1)     # [meta, 5Q, w]
        x = torch.cat([self.cls.expand(meta, -1, -1), toks], dim=1)            # +CLS -> 5Q+1
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        if self.readout == "multi":
            rep = torch.cat([x[:, 0], x.mean(1), x.max(1).values], dim=-1)
        else:
            rep = x[:, 0]
        if self.stats is not None:
            rep = torch.cat([rep, self.stats(contract)], dim=-1)               # bypass AFTER final-norm, before head
        return self.head(rep).squeeze(-1)


# ----------------------------------------------------------------------------------------------------
class _R3Block(nn.Module):
    """Pre-LN latent block: (1) latent->memory cross-attn, (2) latent self-attn, (3) FFN — all residual.
    Reads the ORIGINAL encoded memory (passed in every block); memory is never overwritten."""
    def __init__(self, w, ffn, dropout):
        super().__init__()
        self.ln_c = nn.LayerNorm(w); self.cross = nn.MultiheadAttention(w, HEADS, batch_first=True, dropout=dropout)
        self.ln_s = nn.LayerNorm(w); self.selfa = nn.MultiheadAttention(w, HEADS, batch_first=True, dropout=dropout)
        self.ln_f = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, w))

    def forward(self, lat, mem, kpm):
        q = self.ln_c(lat)
        lat = lat + self.cross(q, mem, mem, key_padding_mask=kpm, need_weights=False)[0]  # read ORIGINAL memory
        s = self.ln_s(lat)
        lat = lat + self.selfa(s, s, s, need_weights=False)[0]
        return lat + self.ffn(self.ln_f(lat))


class R3Readout(nn.Module):
    """Full-token response memory + repeated latent readout (spec §5). No local pooling of hidden responses;
    K learned latents cross-attend the ORIGINAL memory in every block. Avoids full self-attn over memory."""
    def __init__(self, n_probes, n_classes, n_routes, nmax, ffn,
                 w=192, n_slots=16, n_blocks=2, stats=True, dropout=0.1):
        super().__init__()
        self.w = w; self.n_slots = n_slots
        self.fuse = nn.Linear(len(VIEWS) * DSTATE, w)                    # 96 -> w (fused before any pooling)
        self.tok_pos = nn.Parameter(torch.zeros(nmax, w))
        self.probe_emb = nn.Parameter(torch.zeros(n_probes, w))
        self.route_emb = nn.Parameter(torch.zeros(n_routes, w))
        self.logit_proj = nn.Linear(n_classes, w)
        self.logit_slot = nn.Parameter(torch.zeros(w))
        self.mem_ln = nn.LayerNorm(w)                                    # encode memory once
        self.latents = nn.Parameter(torch.zeros(n_slots, w))
        self.blocks = nn.ModuleList([_R3Block(w, ffn, dropout) for _ in range(n_blocks)])
        self.stats = StatsBypass(n_routes) if stats else None
        for p in (self.tok_pos, self.probe_emb, self.route_emb, self.latents):
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.logit_slot, std=0.02)
        head_in = n_slots * w + (32 if stats else 0)                     # flatten the latent bank (counted)
        self.head = nn.Sequential(nn.Linear(head_in, w), nn.GELU(), nn.Linear(w, 1))

    def _memory(self, contract):
        toks, masks = [], []
        for r in contract:
            b1, b2, fn = r["b1"], r["b2"], r["fn"]
            meta, Qr, n, _ = b1.shape
            h = self.fuse(torch.cat([b1, b2, fn], dim=-1)) + self.tok_pos[:n]           # [meta,Qr,n,w]
            h = h + self.probe_emb[r["pids"]].unsqueeze(1) + self.route_emb[r["route"]]
            toks.append(h.reshape(meta, Qr * n, self.w))
            m = (r["mask"].reshape(meta, Qr * n) if r.get("mask") is not None
                 else torch.ones(meta, Qr * n, device=b1.device))
            masks.append(m)
            lg = self.logit_proj(r["logits"]) + self.logit_slot \
                + self.probe_emb[r["pids"]] + self.route_emb[r["route"]]                # [meta,Qr,w] one logit tok/probe
            toks.append(lg)
            masks.append(torch.ones(meta, Qr, device=b1.device))
        mem = self.mem_ln(torch.cat(toks, dim=1))                                       # [meta, M, w]
        kpm = (torch.cat(masks, dim=1) < 0.5)                                           # True = padded (ignored)
        return mem, kpm

    def forward(self, contract):
        meta = contract[0]["b1"].shape[0]
        mem, kpm = self._memory(contract)
        lat = self.latents.unsqueeze(0).expand(meta, -1, -1)
        for b in self.blocks:
            lat = b(lat, mem, kpm)                                                      # every block re-reads mem
        rep = lat.reshape(meta, self.n_slots * self.w)
        if self.stats is not None:
            rep = torch.cat([rep, self.stats(contract)], dim=-1)
        return self.head(rep).squeeze(-1)


class ROutReadout(nn.Module):
    """Output-only (ProbeGen-style) baseline: the predictor observes ONLY the target's output logits per probe.
    block1/block2/final_norm are NOT fed to the predictor (the teacher still forwards them; they are simply not
    observed). Q+1 tokens (one logit token/probe + CLS) -> pre-LN Transformer -> head. Same cap/MSE/splits."""

    def __init__(self, n_probes: int, n_classes: int, n_routes: int, nmax: int, ffn: int,
                 w: int = 192, n_blocks: int = 2, readout: str = "multi", dropout: float = 0.1):
        super().__init__()
        self.w = w; self.readout = readout
        self.logit_proj = nn.Linear(n_classes, w)                        # ONLY the output logits are observed
        self.probe_emb = nn.Parameter(torch.zeros(n_probes, w))
        self.route_emb = nn.Parameter(torch.zeros(n_routes, w))
        self.cls = nn.Parameter(torch.zeros(1, 1, w))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(w, HEADS, ffn, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(w)
        for p in (self.probe_emb, self.route_emb, self.cls):
            nn.init.normal_(p, std=0.02)
        ro = 3 * w if readout == "multi" else w
        self.head = nn.Sequential(nn.Linear(ro, w), nn.GELU(), nn.Linear(w, 1))

    def _route_tokens(self, r: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logit_proj(r["logits"]) + self.probe_emb[r["pids"]] + self.route_emb[r["route"]]  # [meta,Qr,w]

    def forward(self, contract: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        meta = contract[0]["logits"].shape[0]
        toks = torch.cat([self._route_tokens(r) for r in contract], dim=1)   # [meta, Q, w] — one token/probe
        x = torch.cat([self.cls.expand(meta, -1, -1), toks], dim=1)          # +CLS -> Q+1
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        rep = torch.cat([x[:, 0], x.mean(1), x.max(1).values], dim=-1) if self.readout == "multi" else x[:, 0]
        return self.head(rep).squeeze(-1)


def build_readout(arch: str, *, n_probes, n_classes, n_routes, nmax, ffn, readout="multi",
                  n_blocks=2, stats=False, dropout=0.1, n_slots=16, token_mlp=False, moments=False,
                  signed_mix=False, route_lens=None):
    """Registry. r0 is built in system.py (tokenizer+predictor); here: r2 (cross-layer fusion), r3 (latent memory),
    rout (output/logits-only ProbeGen-style baseline)."""
    if arch == "rout":
        for bad, name in [(stats, "stats_bypass"), (token_mlp, "token_mlp"), (moments, "moments"), (signed_mix, "signed_mix")]:
            if bad:
                raise ValueError(f"{name} is a hidden-response module; not supported with output-only readout 'rout'")
        return ROutReadout(n_probes, n_classes, n_routes, nmax, ffn, n_blocks=n_blocks, readout=readout, dropout=dropout)
    if arch == "channel_trajectory":                               # v1, readout_channeltraj.py: mnist/agnews, Q256 only
        for bad, name in [(stats, "stats_bypass"), (token_mlp, "token_mlp"), (moments, "moments"), (signed_mix, "signed_mix")]:
            if bad:
                raise ValueError(f"{name} is not part of channel_trajectory (no bypass/moments/mixing modules)")
        if n_blocks != 2:
            raise ValueError("channel_trajectory fixes 2 global Transformer blocks")
        from .readout_channeltraj import ChannelTrajectoryReadout
        return ChannelTrajectoryReadout(n_probes, n_classes, n_routes, route_lens=route_lens, dropout=dropout,
                                        readout=readout, ffn=ffn)
    if arch == "r2":
        return R2Readout(n_probes, n_classes, n_routes, nmax, ffn, n_blocks=n_blocks,
                         readout=readout, stats=stats, dropout=dropout, token_mlp=token_mlp, moments=moments,
                         signed_mix=signed_mix, route_lens=route_lens)
    if arch == "r3":
        if signed_mix:
            raise ValueError("signed_mix (§8) is an R2-only readout switch; not supported with r3")
        return R3Readout(n_probes, n_classes, n_routes, nmax, ffn, n_slots=n_slots,
                         n_blocks=n_blocks, stats=stats, dropout=dropout)
    raise ValueError(f"unknown readout arch {arch!r} (r0 is built in system.py)")
