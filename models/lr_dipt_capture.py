"""LR-DIPT feature capture (Phase B). Summary-in-forward: for each conv->activation pair, immediately
reduce the [P,C,H,W] pre- and post-activation maps to compact 4x4 (mean/std/max) cells + global stats,
then release the full-resolution map. Keeps the computation graph so learned probes get gradients;
never detaches summaries. Target weights are frozen (no weight grad needed). No skips (audited).

Returns per-net records; pad_batch() collates a list of nets into padded [B,L,P,C,...] tensors + masks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# activation-type ids (mirrors Generic_CNN_Network.get_act; 'Sine' is the SIREN activation)
_ACT_IDS = {"ReLU": 0, "GELU": 1, "Sine": 2, "Tanh": 3, "Sigmoid": 4, "LeakyReLU": 5, "Identity": 6}
N_ACT = 7
N_GLOBAL = 7                      # per pre/post: mean,std,min,max,log1p|.|2,pos_frac,nearzero_frac
CELL_STATS = 3                   # mean,std,max per cell (per pre/post)


def _act_id(act_module):
    return _ACT_IDS.get(type(act_module).__name__, 6)


def _summarize(x, G):
    """x [P,C,H,W] -> cells [P,C,G,G,3] (mean,std,max) + glob [P,C,7]. Stats in float32."""
    x = x.float()
    mean = F.adaptive_avg_pool2d(x, G)                                   # [P,C,G,G]
    sec = F.adaptive_avg_pool2d(x * x, G)
    std = torch.sqrt(torch.clamp(sec - mean * mean, min=1e-6))
    mx = F.adaptive_max_pool2d(x, G)
    cells = torch.stack([mean, std, mx], dim=-1)                          # [P,C,G,G,3]
    xf = x.flatten(2)                                                     # [P,C,HW]
    glob = torch.stack([xf.mean(-1), xf.std(-1), xf.amin(-1), xf.amax(-1),
                        torch.log1p(xf.norm(dim=-1)),
                        (xf > 0).float().mean(-1), (xf.abs() < 1e-3).float().mean(-1)], dim=-1)  # [P,C,7]
    return cells, glob


@torch.no_grad()
def _layer_meta(layer, act, l, L, H, W):
    """Continuous+categorical layer metadata (no gradient needed). Returns a dict of python scalars."""
    kh, kw = (layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,) * 2)
    sh, sw = (layer.stride if isinstance(layer.stride, tuple) else (layer.stride,) * 2)
    dh, dw = (layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,) * 2)
    return dict(fwd_depth=l / max(L - 1, 1), rev_depth=(L - 1 - l) / max(L - 1, 1),
                log2_cin=math.log2(max(layer.in_channels, 1)), log2_cout=math.log2(max(layer.out_channels, 1)),
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw, groups=layer.groups,
                log2_h=math.log2(max(H, 1)), log2_w=math.log2(max(W, 1)),
                act_id=_act_id(act), pos=(0 if l == 0 else (2 if l == L - 1 else 1)))


N_REP = 9   # stable_rank, effective_rank, top5 singular values, mean_channel_cos, max_channel_corr

@torch.no_grad()
def _repstats(post):
    """Detached representation stats of the channel x probe response matrix R (fixed side features).
    R[c,p] = channel c's post-activation GAP-mean on probe p. -> [N_REP]. No grad (SVD)."""
    R = post.float().mean(dim=(-2, -1)).transpose(0, 1)                  # [C, P]
    C = R.shape[0]; dev = R.device
    try:
        sv = torch.linalg.svdvals(R).clamp(min=0)
    except Exception:
        sv = torch.zeros(min(R.shape), device=dev)
    ssum = sv.sum().clamp(min=1e-12); smax = sv[0].clamp(min=1e-12)
    stable_rank = (sv ** 2).sum() / (smax ** 2)
    p = (sv / ssum).clamp(min=1e-12); eff_rank = torch.exp(-(p * p.log()).sum())
    top5 = torch.zeros(5, device=dev); k = min(5, sv.numel()); top5[:k] = sv[:k] / smax
    Rn = R / R.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = Rn @ Rn.t() - torch.eye(C, device=dev)
    mean_cos = cos.sum() / max(C * (C - 1), 1); max_corr = cos.abs().max() if C > 1 else torch.zeros((), device=dev)
    return torch.stack([stable_rank, eff_rank, *top5.unbind(), mean_cos, max_corr]).nan_to_num_(0.0)


def capture_features(net, X, G=4, stats=False):
    """Run net on probes X [P,3,32,32] with summary-in-forward. Returns (logits [P,10], layer_records).
    Each record: cells [P,C,G,G,6], glob [P,C,14], meta dict, C. If stats=True also rep [N_REP] (detached
    SVD statistics; gated behind --use_hidden_statistics so the baseline v1 late is unchanged)."""
    x = X
    L = len(net.layers)
    recs = []
    for l, (layer, act) in enumerate(zip(net.layers, net.activations)):
        pre = layer(x)                                                    # [P,C,H,W] pre-activation
        H, W = pre.shape[-2:]
        pre_cells, pre_glob = _summarize(pre, G)
        post = act(pre)                                                   # post-activation
        post_cells, post_glob = _summarize(post, G)
        rec = dict(cells=torch.cat([pre_cells, post_cells], dim=-1),      # [P,C,G,G,6]
                   glob=torch.cat([pre_glob, post_glob], dim=-1),         # [P,C,14]
                   meta=_layer_meta(layer, act, l, L, H, W), C=pre.shape[1])
        if stats: rec["rep"] = _repstats(post)                           # [N_REP] detached, opt-in
        recs.append(rec)
        x = post
    logits = net.fc(net.flatten(net.pool(x)))                            # [P,10]
    return logits, recs


def pad_batch(records_list, meta_dim, P, G=4, device="cuda", dtype=torch.float32, MAXC=None, MAXL=None):
    """Collate a list (over B nets) of layer_records into padded tensors.
    Returns dict: cells [B,L,P,C,G,G,6], glob [B,L,P,C,14], meta [B,L,meta_dim],
                  cmask [B,L,C] (True=valid), lmask [B,L] (True=valid)."""
    B = len(records_list)
    L = MAXL or max(len(r) for r in records_list)
    C = MAXC or max((rec["C"] for r in records_list for rec in r), default=1)
    has_rep = bool(records_list and records_list[0] and "rep" in records_list[0][0])
    cells = torch.zeros(B, L, P, C, G, G, 6, device=device, dtype=dtype)
    glob = torch.zeros(B, L, P, C, 2 * N_GLOBAL, device=device, dtype=dtype)
    rep = torch.zeros(B, L, N_REP, device=device, dtype=dtype) if has_rep else None
    meta = torch.zeros(B, L, meta_dim, device=device, dtype=dtype)
    cmask = torch.zeros(B, L, C, dtype=torch.bool, device=device)
    lmask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b, recs in enumerate(records_list):
        for l, rec in enumerate(recs):
            if l >= L:
                break
            c = min(rec["C"], C)
            cells[b, l, :, :c] = rec["cells"][:, :c].to(dtype)
            glob[b, l, :, :c] = rec["glob"][:, :c].to(dtype)
            if has_rep: rep[b, l] = rec["rep"].to(dtype)
            meta[b, l] = encode_meta(rec["meta"], meta_dim, device, dtype)
            cmask[b, l, :c] = True
            lmask[b, l] = True
    out = dict(cells=cells, glob=glob, meta=meta, cmask=cmask, lmask=lmask)
    if has_rep: out["rep"] = rep
    return out


# continuous metadata packed into a fixed vector (categorical act handled via one-hot slots)
_CONT_KEYS = ["fwd_depth", "rev_depth", "log2_cin", "log2_cout", "kh", "kw", "sh", "sw",
              "dh", "dw", "groups", "log2_h", "log2_w"]
META_DIM = len(_CONT_KEYS) + N_ACT + 3          # continuous + act one-hot + position one-hot(3)


def encode_meta(meta, meta_dim, device, dtype):
    v = torch.zeros(meta_dim, device=device, dtype=dtype)
    for i, k in enumerate(_CONT_KEYS):
        v[i] = meta[k]
    v[len(_CONT_KEYS) + int(meta["act_id"])] = 1.0
    v[len(_CONT_KEYS) + N_ACT + int(meta["pos"])] = 1.0
    return v
