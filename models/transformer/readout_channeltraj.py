"""Channel-trajectory readout — arch 'channel_trajectory' (version 1). Q=256, FFN fixed 384.

Hypothesis under test: encode each hidden CHANNEL's responses across the COMPLETE probe bank (one long
"trajectory" per channel per view), instead of summarizing each probe separately (R2's per-probe local
attention pooling). Not claimed to be better; this module makes the comparison possible.

Consumes the RAW response contract from acquire._build_contract — response-only: b1/b2/fn/logits + the fixed
semantic ids (probe ids, route id, mask). Never weights, sketches, probe coordinates, target ids, labels.

Fixed acquisition geometries (one per zoo; the shared 256-probe bank is unchanged):
    mnist  : route 0 = 192 encoder probes x 17 tokens (ids   0..191)      C = 10 logits
             route 1 =  64 native  probes x 49 tokens (ids 192..255)      trajectory = 3264 + 3136 = 6,400 coords
    agnews : route 0 = 256 encoder probes x 17 tokens (ids   0..255)      C = 4 logits
             trajectory = 4,352 coords
Per hidden view, channel c's trajectory = concatenation over routes of (probe j, token t) -> coordinate
off_r + j*L_r + t. The teacher hidden width is 32 channels in both zoos (three views: block1, block2, final_norm).

Architecture (widths verbatim from the spec; all Linear layers biased; LayerNorm affine, eps 1e-5; plain GELU):
    E  (ONE shared trajectory encoder)  : Linear(T,128) - GELU - Linear(128,192) - LN(192)      T = trajectory length
        U1 = E(A1)  U2 = E(A2)  Un = E(An)                      each [M,32,192]   (A* = [M,32,T] trajectories)
    F  (block2 / final_norm fusion)      : Linear(384,192) - GELU - Linear(192,192) - LN(192)
        V2 = F(cat(U2, Un))                                     [M,32,192]   (block1 stays separate)
    tokens = CLS | 32 x (U1 + type[0]) | 32 x (V2 + type[1]) | 256 x (Linear(C,192)(logits) + probe_id + type[2])
           = 321 tokens of width 192.   NO channel-index / position / route embeddings (probe ids cover routes).
    2 x pre-LN TransformerEncoderLayer(192, 6 heads, FFN 384, GELU, dropout 0.1; biased QKV/out; 2 affine LN each)
    final LN(192) ; rep = cat[CLS, mean(320 non-CLS), max(320 non-CLS)] = 576 ; Linear(576,192)-GELU-Linear(192,1)
Predicts STANDARDIZED accuracy (no sigmoid). Pooling never crosses the target axis.

Analytic trainable parameters: readout mnist 1,713,281 (+ G3 159,888 = 1,873,169 <= 1,880,000 cap);
agnews 1,449,985 (+ encoder-only G3 143,456 = 1,593,441). Identical readout
widths on both zoos — only the first Linear's fan-in (T) and the logit projection's fan-in (C) follow the geometry.
FFN: fixed at 384 on mnist (spec; no auto-fit). On agnews the readout is far under the cap, so --ffn 0 auto-fits
the FFN to the 1,880,000 cap (param-fair with r2tm, which always filled the cap); --ffn 384 keeps the mnist widths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

CT_ARCH_VERSION = "channel_trajectory_v1"
CT_W = 192            # token width
CT_HEADS = 6
CT_FFN = 384          # fixed (spec); NOT auto-fit
CT_ENC_HID = 128      # trajectory-encoder hidden width
CT_DSTATE = 32        # teacher hidden width (channels), both zoos
TYPE_B1, TYPE_B2N, TYPE_LOGIT = 0, 1, 2
_LN_EPS = 1e-5


@dataclass(frozen=True)
class CTGeometry:
    name: str
    routes: Tuple[Tuple[int, int], ...]     # ((probes, tokens) per route, canonical route order)
    n_classes: int

    @property
    def n_probes(self) -> int:
        return sum(q for q, _ in self.routes)

    @property
    def route_lens(self) -> List[int]:
        return [L for _, L in self.routes]

    @property
    def traj(self) -> int:                  # trajectory length per channel per view
        return sum(q * L for q, L in self.routes)

    def offsets(self) -> List[int]:         # coordinate offset of each route inside the trajectory
        out, off = [], 0
        for q, L in self.routes:
            out.append(off); off += q * L
        return out


CT_GEOMETRIES: Dict[str, CTGeometry] = {
    "mnist": CTGeometry("mnist", ((192, 17), (64, 49)), 10),
    "agnews": CTGeometry("agnews", ((256, 17),), 4),
}


def geometry_for(n_probes: int, n_classes: int, route_lens) -> CTGeometry:
    """Resolve the (unique) supported geometry from the readout's construction arguments."""
    rl = [int(L) for L in route_lens]
    for g in CT_GEOMETRIES.values():
        if g.n_probes == n_probes and g.n_classes == n_classes and g.route_lens == rl:
            return g
    sup = [(g.name, g.n_probes, g.n_classes, g.route_lens) for g in CT_GEOMETRIES.values()]
    raise ValueError(f"channel_trajectory supports only {sup}; got n_probes={n_probes}, n_classes={n_classes}, route_lens={rl}")


# --- MNIST constants kept as module-level names (tests / trainer bind to them) ---
_MN = CT_GEOMETRIES["mnist"]
CT_N_CLASSES = _MN.n_classes                            # 10
(CT_N_ENC, CT_L_ENC), (CT_N_NAT, CT_L_NAT) = _MN.routes  # 192,17 / 64,49
CT_N_PROBES = _MN.n_probes                              # 256
CT_TRAJ = _MN.traj                                      # 6400
CT_ROUTE_LENS = _MN.route_lens                          # [17, 49]


class CTContractError(ValueError):
    """Raised when the response contract is not the fixed Q256 geometry this readout requires."""


def _logical_view(x: torch.Tensor, mask, L: int, what: str) -> torch.Tensor:
    """x: [M,Qr,n,32] (n may include computational padding >= L). Returns [M,Qr,L,32] with the fixed logical
    length L and masked (invalid) logical entries ZEROED IN PLACE (no shifting). Uses torch.where so NaN/garbage in
    masked slots cannot propagate (x*mask would turn NaN*0 into NaN)."""
    M, Qr, n, d = x.shape
    if n < L:
        raise CTContractError(f"{what}: token length {n} < logical length {L}")
    if mask is None:
        if n != L:
            raise CTContractError(f"{what}: unmasked contract must have exactly {L} tokens, got {n}")
        return x
    mask = mask.bool()
    if mask.shape != (M, Qr, n):
        raise CTContractError(f"{what}: mask shape {tuple(mask.shape)} != {(M, Qr, n)}")
    if n > L and bool(mask[..., L:].any()):
        raise CTContractError(f"{what}: valid tokens beyond the logical length {L} (padding must be trailing)")
    xl = x[..., :L, :]
    ml = mask[..., :L].unsqueeze(-1)
    return torch.where(ml, xl, torch.zeros((), dtype=x.dtype, device=x.device))


def assemble_trajectories(contract: List[Dict], geom: CTGeometry = _MN) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contract (routes in canonical order) -> (A1, A2, An) each [M,32,T] channel trajectories + logits [M,Q,C].

    Channel-first layout: for each view and route, [M,Qr,Lr,32] -> permute(0,3,1,2) -> [M,32,Qr,Lr] -> [M,32,Qr*Lr];
    routes concatenated along the last dim. (A bare reshape of [M,Q,n,32] would mix channels — never do that.)
    The target axis M is never mixed."""
    if len(contract) != len(geom.routes):
        raise CTContractError(f"channel_trajectory[{geom.name}] needs exactly {len(geom.routes)} route(s); got {len(contract)}")
    dev = contract[0]["b1"].device
    M = contract[0]["b1"].shape[0]
    pid0 = 0
    for ri, (r, (Qr, L)) in enumerate(zip(contract, geom.routes)):
        if int(r["route"]) != ri:
            raise CTContractError(f"routes must be in canonical order 0..{len(geom.routes) - 1}; route {ri} carries id {r['route']}")
        if not torch.equal(r["pids"].to(dev), torch.arange(pid0, pid0 + Qr, device=dev)):
            raise CTContractError(f"route {ri} probe ids must be exactly {pid0}..{pid0 + Qr - 1} in order")
        for v in ("b1", "b2", "fn"):
            if r[v].shape[1] != Qr or r[v].shape[-1] != CT_DSTATE:
                raise CTContractError(f"route{ri}.{v}: expected [M,{Qr},n,{CT_DSTATE}], got {tuple(r[v].shape)}")
        if r["logits"].shape[1] != Qr or r["logits"].shape[-1] != geom.n_classes:
            raise CTContractError(f"route{ri}.logits: expected [M,{Qr},{geom.n_classes}], got {tuple(r['logits'].shape)}")
        if r["b1"].shape[0] != M:
            raise CTContractError("routes carry different numbers of targets")
        pid0 += Qr

    def traj(view: str) -> torch.Tensor:
        parts = []
        for ri, (r, (Qr, L)) in enumerate(zip(contract, geom.routes)):
            x = _logical_view(r[view], r.get("mask"), L, f"route{ri}.{view}")           # [M,Qr,L,32]
            parts.append(x.permute(0, 3, 1, 2).reshape(M, CT_DSTATE, Qr * L))           # [M,32,Qr*L]
        return torch.cat(parts, dim=-1)                                                 # [M,32,T]

    A1, A2, An = traj("b1"), traj("b2"), traj("fn")
    logits = torch.cat([r["logits"] for r in contract], dim=1)                          # [M,Q,C] canonical order
    return A1, A2, An, logits


class ChannelTrajectoryReadout(nn.Module):
    ARCH_VERSION = CT_ARCH_VERSION

    def __init__(self, n_probes: int = CT_N_PROBES, n_classes: int = CT_N_CLASSES, n_routes: int = 2,
                 route_lens=None, dropout: float = 0.1, readout: str = "multi", ffn: int = CT_FFN):
        super().__init__()
        route_lens = list(route_lens) if route_lens is not None else CT_ROUTE_LENS
        bad = []
        if n_routes != len(route_lens): bad.append(f"n_routes={n_routes} != len(route_lens)={len(route_lens)}")
        if readout != "multi": bad.append(f"readout={readout!r} (spec fixes concat[CLS,mean,max])")
        if int(ffn) <= 0: bad.append(f"ffn={ffn}")
        if bad:
            raise ValueError("channel_trajectory: " + "; ".join(bad))
        self.geom = geometry_for(n_probes, n_classes, route_lens)     # raises for unsupported zoos / Q / C
        self.ffn = int(ffn)          # mnist: fixed 384 (system gate); agnews: may be auto-fit to the 1.88M cap
        w = CT_W
        # B. ONE shared trajectory encoder (all channels, all three views)
        self.enc = nn.Sequential(nn.Linear(self.geom.traj, CT_ENC_HID), nn.GELU(), nn.Linear(CT_ENC_HID, w),
                                 nn.LayerNorm(w, eps=_LN_EPS))
        # C. block2 / final_norm fusion (shared across channels); no residual, no gate
        self.fuse = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(), nn.Linear(w, w), nn.LayerNorm(w, eps=_LN_EPS))
        self.type_emb = nn.Parameter(torch.zeros(3, w))               # block1 / paired block2+norm / logits
        # D. logit tokens: Linear(C,192) + probe-id table [256,192] (+ logit type)
        self.logit_proj = nn.Linear(n_classes, w)
        self.probe_emb = nn.Parameter(torch.zeros(n_probes, w))
        self.cls = nn.Parameter(torch.zeros(1, 1, w))
        # E. two independently initialised pre-LN blocks (separate objects), one final LN, 576 -> 192 -> 1 head
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(w, CT_HEADS, self.ffn, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(2)])
        self.final_norm = nn.LayerNorm(w, eps=_LN_EPS)
        self.head = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Linear(w, 1))
        for p in (self.type_emb, self.probe_emb, self.cls):           # repo convention for id tables / CLS
            nn.init.normal_(p, std=0.02)

    # ---- pieces exposed for tests ----
    def tokens(self, contract: List[Dict]) -> torch.Tensor:
        A1, A2, An, logits = assemble_trajectories(contract, self.geom)
        U1, U2, Un = self.enc(A1), self.enc(A2), self.enc(An)                          # [M,32,192] each
        V2 = self.fuse(torch.cat([U2, Un], dim=-1))                                    # [M,32,192]
        h1 = U1 + self.type_emb[TYPE_B1]
        h2 = V2 + self.type_emb[TYPE_B2N]
        lg = self.logit_proj(logits) + self.probe_emb + self.type_emb[TYPE_LOGIT]     # [M,256,192]
        M = A1.shape[0]
        return torch.cat([self.cls.expand(M, -1, -1), h1, h2, lg], dim=1)             # [M,321,192]

    def forward(self, contract: List[Dict]) -> torch.Tensor:
        x = self.tokens(contract)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        body = x[:, 1:]                                                                # 320 non-CLS tokens
        rep = torch.cat([x[:, 0], body.mean(dim=1), body.max(dim=1).values], dim=-1)  # [M,576]; never across M
        return self.head(rep).squeeze(-1)


def expected_param_count(dataset: str = "mnist", ffn: int = CT_FFN) -> Dict[str, int]:
    """Analytic trainable-parameter breakdown of the readout (spec table), for verification against the model."""
    g = CT_GEOMETRIES[dataset]
    w, h, T, C, Q = CT_W, CT_ENC_HID, g.traj, g.n_classes, g.n_probes
    lin = lambda i, o: i * o + o
    ln = 2 * w
    enc = lin(T, h) + lin(h, w) + ln                                    # mnist: 819,328 + 24,768 + 384 = 844,480
    fuse = lin(2 * w, w) + lin(w, w) + ln                                # 73,920 + 37,056 + 384 = 111,360
    ids = lin(C, w) + Q * w + 3 * w + w                                  # mnist: 2,112 + 49,152 + 576 + 192 = 52,032
    block = (3 * w * w + 3 * w) + lin(w, w) + lin(w, ffn) + lin(ffn, w) + 2 * ln   # 297,024 at ffn 384
    head = ln + lin(3 * w, w) + lin(w, 1)                                # 384 + 110,784 + 193 = 111,361
    d = {"trajectory_encoder": enc, "fusion": fuse, "logit_proj+ids+cls": ids,
         "transformer_blocks(2)": 2 * block, "final_norm+head": head}
    d["readout_total"] = sum(d.values())                                 # mnist 1,713,281 ; agnews 1,449,985
    return d
