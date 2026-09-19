"""LR-DIPT Phase B2/B3: spatial hidden tokenizer + early output-conditioned interaction branch.

B2 tokenizer: each map's 4x4x6 cells + Fourier position + FiLM(layer metadata) -> low-rank map-query
pool -> concat global stats -> 64-d hidden token. Shared across all layers/architectures; layer
semantics enter only through metadata. Chunked over maps to bound memory.

B3 early branch: a probe's OUTPUT token conditions (queries) that same probe's hidden CHANNEL tokens
(low-rank cross-attn, channel-permutation-invariant), then an ordered low-rank layer transformer with a
relative-layer-distance bias, an output-conditioned layer pool, and a Tucker pool over fixed probe
slots -> z_early. All early-branch VALUES are hidden-derived (the output token only conditions).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lr_dipt_lowrank import (FactorizedLinear, LowRankAttention, LowRankAttentionPool, TuckerPool)


def fourier_grid(G, n_freq):
    """Fixed 2D Fourier position for a GxG grid, coords in [-1,1]. Returns [G*G, 4*n_freq]."""
    lin = torch.linspace(-1, 1, G)
    v, u = torch.meshgrid(lin, lin, indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)                                   # [G*G]
    feats = []
    for j in range(n_freq):
        w = (2 ** j) * math.pi
        feats += [torch.sin(w * u), torch.cos(w * u), torch.sin(w * v), torch.cos(w * v)]
    return torch.stack(feats, dim=-1)                                     # [G*G, 4*n_freq]


class LayerMetadataEncoder(nn.Module):
    def __init__(self, meta_dim, d_meta=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(meta_dim, d_meta), nn.GELU(), nn.Linear(d_meta, d_meta))

    def forward(self, meta):                                              # [.., meta_dim] -> [.., d_meta]
        return self.net(meta)


class SpatialTokenizer(nn.Module):
    """[.,G,G,6] cells + [.,14] globals + [.,d_meta] metadata -> [.,d_hidden] token."""
    def __init__(self, d_spatial=32, d_hidden=64, d_meta=32, n_global=14, G=4, n_freq=4, attn_rank=8):
        super().__init__()
        self.G = G
        self.cell_proj = nn.Linear(6, d_spatial)
        self.register_buffer("pos", fourier_grid(G, n_freq))             # [G*G, 4*n_freq]
        self.pos_proj = nn.Linear(4 * n_freq, d_spatial)
        self.film = nn.Linear(d_meta, 2 * d_spatial)
        self.map_pool = LowRankAttentionPool(d_spatial, attn_rank, out_dim=d_spatial)
        self.out = FactorizedLinear(d_spatial + n_global, d_hidden, rank=min(32, d_hidden))

    def forward(self, cells, glob, meta):
        # cells [N,G,G,6], glob [N,14], meta [N,d_meta]
        N = cells.shape[0]
        t = self.cell_proj(cells.reshape(N, self.G * self.G, 6)) + self.pos_proj(self.pos)  # [N,GG,d_sp]
        gamma, beta = self.film(meta).chunk(2, dim=-1)                    # [N,d_sp]
        t = t * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)             # FiLM per cell
        m = self.map_pool(t)                                             # [N,d_sp]
        return self.out(torch.cat([m, glob], dim=-1))                   # [N,d_hidden]


def tokenize_maps(tok, cells, glob, meta_emb, cmask, chunk=2048):
    """Apply the shared tokenizer to all (b,l,p,c) maps in chunks -> hidden_tokens [B,L,P,C,d_hidden].
    cells [B,L,P,C,G,G,6], glob [B,L,P,C,14], meta_emb [B,L,d_meta], cmask [B,L,C]."""
    B, L, P, C = cells.shape[:4]
    dh = tok.out.up.out_features
    G = tok.G
    meta_full = meta_emb[:, :, None, None, :].expand(B, L, P, C, -1)      # broadcast meta per map
    cflat = cells.reshape(-1, G, G, 6); gflat = glob.reshape(-1, glob.shape[-1]); mflat = meta_full.reshape(-1, meta_emb.shape[-1])
    outs = []
    for i in range(0, cflat.shape[0], chunk):
        outs.append(tok(cflat[i:i + chunk], gflat[i:i + chunk], mflat[i:i + chunk]))
    H = torch.cat(outs, 0).reshape(B, L, P, C, dh)
    return H * cmask[:, :, None, :, None].to(H.dtype)                    # zero padded channels


def output_probe_token(logits, d=64):
    """Per-probe feature from the 10 logits + confidence stats (order/scale preserved). -> [B,P,d_raw]."""
    lp = F.log_softmax(logits, dim=-1); p = lp.exp()
    ent = -(p * lp).sum(-1, keepdim=True)
    maxp = p.max(-1, keepdim=True).values
    top2 = p.topk(2, dim=-1).values
    margin = (top2[..., :1] - top2[..., 1:2])
    l2 = logits.norm(dim=-1, keepdim=True)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.cat([logits, ent, maxp, margin, l2, lse], dim=-1)       # [B,P,15]


class EarlyBranch(nn.Module):
    def __init__(s, d=64, P=128, attn_rank=16, r_probe=8, r_feat=32, z_dim=128, max_rel=4):
        super().__init__()
        s.out_tok = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        s.probe_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        s.chan_attn = LowRankAttention(d, attn_rank)                     # output-query over channels
        s.rel_bias = nn.Parameter(torch.zeros(2 * max_rel + 1))          # relative-layer-distance bias
        s.max_rel = max_rel
        s.layer_attn = LowRankAttention(d, attn_rank)                    # ordered layer self-attn
        s.layer_ff = FactorizedLinear(d, d, attn_rank, act=nn.GELU(), zero_init_up=True)
        s.layer_norm = nn.LayerNorm(d)
        s.layer_pool = LowRankAttention(d, attn_rank)                    # output-query over layers
        s.tucker = TuckerPool(P, d, r_probe, r_feat)
        s.proj = FactorizedLinear(s.tucker.out_dim, z_dim, rank=32)

    def forward(s, logits, H, cmask, lmask):
        # logits [B,P,10]; H hidden tokens [B,L,P,C,d]; cmask [B,L,C]; lmask [B,L]
        B, L, P, C, d = H.shape
        ot = s.out_tok(output_probe_token(logits)) + s.probe_emb.unsqueeze(0)   # [B,P,d]
        # (1) output-conditioned channel pooling per (b,l,p): query=ot[b,p], keys=H[b,l,p,:]
        q = ot.permute(0, 1, 2)[:, None].expand(B, L, P, d).reshape(B * L * P, 1, d)
        keys = H.permute(0, 1, 2, 3, 4).reshape(B * L * P, C, d)
        kpm = (~cmask)[:, :, None, :].expand(B, L, P, C).reshape(B * L * P, C)
        e = s.chan_attn(q, keys, keys, key_padding_mask=kpm).reshape(B, L, P, d)  # [B,L,P,d]
        # (2) ordered layer transformer per probe (relative-distance bias, masked)
        el = e.permute(0, 2, 1, 3).reshape(B * P, L, d)                  # [B*P,L,d]
        idx = torch.arange(L, device=H.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-s.max_rel, s.max_rel) + s.max_rel
        bias = s.rel_bias[rel][None]                                     # [1,L,L]
        lpm = (~lmask)[:, None, :].expand(B, P, L).reshape(B * P, L)
        el = el + s.layer_attn(el, el, el, key_padding_mask=lpm, rel_bias=bias)
        el = el + s.layer_ff(s.layer_norm(el))
        # (3) output-conditioned layer pooling: query = probe output token
        oq = ot.reshape(B * P, 1, d)
        ep = s.layer_pool(oq, el, el, key_padding_mask=lpm).reshape(B, P, d)     # [B,P,d]
        # (4) Tucker pool over fixed probe slots -> z_early
        return s.proj(s.tucker(ep))                                      # [B,z_dim]


def layer_glob_stats(glob, cmask, probe_mask=None):
    """Channel-perm-invariant per-layer stats from global map stats: masked mean+max over channels,
    then mean over probes. glob [B,L,P,C,G], cmask [B,L,C] -> [B,L,2G].
    probe_mask [B,P] (True=active): probe-mean uses ONLY active probes (active-count normalized).
    None -> plain mean over all probes (UNCHANGED)."""
    m = cmask[:, :, None, :, None].to(glob.dtype)                        # [B,L,1,C,1]
    denom = m.sum(3).clamp(min=1)
    mean = (glob * m).sum(3) / denom                                    # [B,L,P,G]
    mx = glob.masked_fill(m == 0, float("-inf")).amax(3)                # [B,L,P,G]
    mx = torch.nan_to_num(mx, nan=0.0, posinf=0.0, neginf=0.0)          # fully-padded layer -> 0 (not -3e38)
    stat = torch.cat([mean, mx], dim=-1)                                 # [B,L,P,2G]
    if probe_mask is None:
        return stat.mean(2)                                             # [B,L,2G]  plain probe mean (UNCHANGED)
    pm = probe_mask[:, None, :, None].to(stat.dtype)                     # [B,1,P,1]
    return (stat * pm).sum(2) / pm.sum(2).clamp(min=1.0)                 # [B,L,2G]  active-probe mean


class LateBranch(nn.Module):
    """Output-INDEPENDENT global hidden summary z_hidden (channel-perm-invariant, ragged)."""
    def __init__(s, d=64, P=128, attn_rank=16, z_dim=128, glob_dim=28, rep_dim=9, use_stats=False, max_rel=4,
                 hidden_agg="neuron_collapse", probe_mixer="none"):
        super().__init__()
        s.use_stats = use_stats
        s.hidden_agg = hidden_agg          # "neuron_collapse" (recipe ii) | "neuron_profile" (recipe i)
        s.probe_mixer = probe_mixer        # recipe-(i) only: "none" | "attn" | "tokenmix"
        s.chan_pool = LowRankAttentionPool(d, attn_rank)                 # channels -> per (b,l,p)
        s.probe_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        s.probe_pool = LowRankAttentionPool(d, attn_rank)               # probes -> per (b,l)
        if hidden_agg == "neuron_profile" and probe_mixer == "attn":    # (i): probes exchange info before readout
            s.pmix = LowRankAttention(d, attn_rank)
        elif hidden_agg == "neuron_profile" and probe_mixer == "tokenmix":
            s.pmix_norm = nn.LayerNorm(d)
            s.pmix = nn.Linear(P, P)                                     # token-mix over the (identified) probe axis
        s.stat_proj = nn.Linear(glob_dim + (rep_dim if use_stats else 0), d)  # invariant glob [+ opt SVD rep]
        s.rel_bias = nn.Parameter(torch.zeros(2 * max_rel + 1)); s.max_rel = max_rel
        s.layer_attn = LowRankAttention(d, attn_rank)
        s.layer_ff = FactorizedLinear(d, d, attn_rank, act=nn.GELU(), zero_init_up=True)
        s.norm = nn.LayerNorm(d)
        s.layer_pool = LowRankAttentionPool(d, attn_rank)               # layers -> per b
        s.proj = FactorizedLinear(d, z_dim, rank=32)

    def forward(s, H, glob, cmask, lmask, rep=None, probe_mask=None):
        B, L, P, C, d = H.shape
        if s.hidden_agg == "neuron_profile":
            # RECIPE (i): each channel = its cross-probe PROFILE (probe-identified, probes may mix), then
            # set-pool over channels LAST. Channel identity survives across probes -> quotient (S_C)^L.
            Hp = H + s.probe_emb[None, None, :, None, :]                      # probe identity, broadcast over C
            x = Hp.permute(0, 1, 3, 2, 4).reshape(B * L * C, P, d)           # [B*L*C, P, d]
            ppm = ((~probe_mask)[:, None, None, :].expand(B, L, C, P).reshape(B * L * C, P)
                   if probe_mask is not None else None)
            if s.probe_mixer == "attn":                                      # probes EXCHANGE info (residual)
                x = x + s.pmix(x, x, x, key_padding_mask=ppm)
            elif s.probe_mixer == "tokenmix":
                if ppm is not None: x = x.masked_fill(ppm.unsqueeze(-1), 0.0)  # zero dropped probes pre-mix
                x = x + s.pmix(s.pmix_norm(x).transpose(-1, -2)).transpose(-1, -2)  # Linear(P->P) over probes
            prof = s.probe_pool(x, key_padding_mask=ppm).reshape(B, L, C, d)  # reduce P -> per-channel profile
            kpm = (~cmask).reshape(B * L, C)
            gl = s.chan_pool(prof.reshape(B * L, C, d), key_padding_mask=kpm).reshape(B, L, d)  # channels LAST
        else:
            # RECIPE (ii) [DEFAULT, UNCHANGED]: collapse channels per (b,l,p), then pool probes -> (S_C)^{L*P}.
            kpm = (~cmask)[:, :, None, :].expand(B, L, P, C).reshape(B * L * P, C)
            g = s.chan_pool(H.reshape(B * L * P, C, d), key_padding_mask=kpm).reshape(B, L, P, d)
            g = g + s.probe_emb.unsqueeze(0)
            # probe dropout: -inf attention on dropped probes (active probes' probe_emb unchanged). None=UNCHANGED.
            ppm = (~probe_mask)[:, None, :].expand(B, L, P).reshape(B * L, P) if probe_mask is not None else None
            gl = s.probe_pool(g.reshape(B * L, P, d), key_padding_mask=ppm).reshape(B, L, d)   # [B,L,d]
        stats = layer_glob_stats(glob, cmask, probe_mask=probe_mask)     # baseline v1: invariant glob stats
        if s.use_stats and rep is not None: stats = torch.cat([stats, rep], -1)   # opt-in SVD rep stats
        gl = gl + s.stat_proj(stats)
        idx = torch.arange(L, device=H.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-s.max_rel, s.max_rel) + s.max_rel
        gl = gl + s.layer_attn(gl, gl, gl, key_padding_mask=~lmask, rel_bias=s.rel_bias[rel][None])
        gl = gl + s.layer_ff(s.norm(gl))
        z = s.layer_pool(gl, key_padding_mask=~lmask)                    # [B,d]
        return s.proj(z)                                                # [B,z_dim]
