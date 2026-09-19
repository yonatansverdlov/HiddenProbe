"""LearnedSystem: shared unconditioned probe generator (G3 / glin) + response readout.
Knobs: n_probes (Q), readout {cls,multi,pma}, pma_seeds, readout_arch. Ceiling <= 1.88M.
Frozen teachers are external (out of optimizer). Param accounting = unique trainable identities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn

from .generators import ProbeGeneratorG3, ProbeGeneratorGLinear
from .predictor import ResponseTokenizer, CrossResponsePredictor
from .readout_channeltraj import CT_FFN, CT_ARCH_VERSION, CT_N_PROBES, CT_GEOMETRIES   # channel_trajectory v1 (additive)

PARAM_CEILING = 1_880_000       # spec cap (unchanged unless user says so)
NMAX = {"mnist": 49, "agnews": 17}
NROUTES = {"mnist": 2, "agnews": 1}
ROUTE_LENS = {"mnist": [17, 49], "agnews": [17]}   # response tokens per route (enc=17, native=49) for §8 signed mixing


# spec §9 analytic totals for the LOCKED Q64/cls configs (sanity for the default path)
ANALYTIC_Q64 = {
    ("mnist", "g3", 10, 944): 1_874_401,
    ("agnews", "g3", 10, 944): 1_863_153,
    ("agnews", "g3", 4, 944): 1_862_001,
}


@dataclass
class TPConfig:
    dataset: str
    generator: str
    n_classes: int
    ffn: int
    n_probes: int = 64
    readout: str = "cls"
    pma_seeds: int = 1
    seed: int = 0
    readout_arch: str = "r0"           # r0 = existing tokenizer+predictor ; r2 = cross-layer fusion (readouts.py)
    stats_bypass: bool = False         # §4 response-stats bypass (r2 only for now)
    dropout: float = 0.1               # predictor/readout transformer dropout (count-invariant)
    n_slots: int = 16                  # r3 latent slots
    token_mlp: bool = False            # §6 r2 pre-pool nonlinear feature map
    moments: bool = False              # §7 r2 per-probe moments -> logit token
    signed_mix: bool = False           # §8 r2 signed token mixing (A[4,n] per bucket, gated)

    @property
    def key(self):
        return (self.dataset, self.generator, self.n_classes, self.ffn)


class LearnedSystem(nn.Module):
    def __init__(self, cfg: TPConfig):
        super().__init__()
        self.cfg = cfg
        nroutes = NROUTES[cfg.dataset]; nmax = NMAX[cfg.dataset]; route_lens = ROUTE_LENS[cfg.dataset]
        self._nroutes, self._nmax, self._route_lens = nroutes, nmax, route_lens
        self.arch_version = "legacy"
        if cfg.readout_arch == "channel_trajectory":              # v1: mnist / agnews, shared g3 probes, Q256, fixed FFN
            bad = []
            if cfg.dataset not in CT_GEOMETRIES: bad.append(f"dataset={cfg.dataset} (supported: {list(CT_GEOMETRIES)})")
            if cfg.generator != "g3": bad.append(f"generator={cfg.generator} (needs the shared unconditioned g3)")
            if cfg.n_probes != CT_N_PROBES: bad.append(f"n_probes={cfg.n_probes} (needs {CT_N_PROBES})")
            if cfg.n_classes != CT_GEOMETRIES.get(cfg.dataset, CT_GEOMETRIES["mnist"]).n_classes:
                bad.append(f"n_classes={cfg.n_classes} (dataset {cfg.dataset} logits)")
            if cfg.dataset == "mnist" and cfg.ffn != CT_FFN: bad.append(f"ffn={cfg.ffn} (mnist: fixed {CT_FFN}; no auto-fit)")
            if cfg.readout != "multi": bad.append(f"readout={cfg.readout} (spec fixes concat[CLS,mean,max])")
            for flag in ("stats_bypass", "token_mlp", "moments", "signed_mix"):
                if getattr(cfg, flag): bad.append(flag)
            if bad:
                raise ValueError("readout_arch=channel_trajectory rejects: " + ", ".join(bad))
            self.arch_version = CT_ARCH_VERSION
        if cfg.generator == "g3":
            self.generator = ProbeGeneratorG3(cfg.dataset, n_probes=cfg.n_probes, seed=cfg.seed)
        elif cfg.generator == "glin":
            self.generator = ProbeGeneratorGLinear(cfg.dataset, n_probes=cfg.n_probes, seed=cfg.seed)   # §9 shared deep-linear
        else:
            raise ValueError(cfg.generator)
        if cfg.readout_arch == "r0":
            self.tokenizer = ResponseTokenizer(cfg.n_probes, NMAX[cfg.dataset], cfg.n_classes, NROUTES[cfg.dataset])
            self.predictor = CrossResponsePredictor(cfg.ffn, readout=cfg.readout, pma_seeds=cfg.pma_seeds,
                                                    dropout=cfg.dropout)
            self.readout = None
        else:
            from .readouts import build_readout
            self.tokenizer = None; self.predictor = None
            kw = dict(n_probes=cfg.n_probes, n_classes=cfg.n_classes, n_routes=nroutes, nmax=nmax, ffn=cfg.ffn,
                      readout=cfg.readout, stats=cfg.stats_bypass, dropout=cfg.dropout, n_slots=cfg.n_slots,
                      token_mlp=cfg.token_mlp, moments=cfg.moments, signed_mix=cfg.signed_mix, route_lens=route_lens)
            if cfg.readout_arch == "channel_trajectory":
                # Separate init stream (spec §6): the readout's construction neither consumes nor depends on the
                # global RNG that G3's decoders used above (G3 codes already use their own seeded generator), so
                # changing the readout can never change the initial G3 bank, and vice versa. CPU RNG only: the
                # system is built on CPU and moved to the device afterwards.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(cfg.seed * 7919 + 13)
                    self.readout = build_readout(cfg.readout_arch, **kw)
            else:
                self.readout = build_readout(cfg.readout_arch, **kw)

    def component_counts(self) -> Dict[str, int]:
        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad) if m is not None else 0
        return {"generator": n(self.generator),
                "tokenizer": n(self.tokenizer), "predictor": n(self.predictor), "readout": n(self.readout)}

    def n_trainable(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); total += p.numel()
        return total


def fit_ffn(dataset, generator, n_classes, n_probes, readout="cls", pma_seeds=1,
            readout_arch="r0", stats_bypass=False, token_mlp=False, moments=False, signed_mix=False,
            ceiling=PARAM_CEILING) -> int:
    """Largest predictor/readout FFN width (mult of 8) keeping total trainable <= ceiling — fills the budget."""
    if readout_arch == "channel_trajectory" and dataset == "mnist":
        return CT_FFN                                   # fixed by the spec on mnist; agnews falls through to the cap fit
    lo, hi, best = 64, 3072, 64
    while lo <= hi:
        mid = ((lo + hi) // 2) // 8 * 8
        cfg = TPConfig(dataset, generator, n_classes, mid, n_probes, readout, pma_seeds,
                       readout_arch=readout_arch, stats_bypass=stats_bypass, token_mlp=token_mlp, moments=moments,
                       signed_mix=signed_mix)
        if LearnedSystem(cfg).n_trainable() <= ceiling:
            best = mid; lo = mid + 8
        else:
            hi = mid - 8
    return best


def verify_config(cfg: TPConfig, verbose: bool = True):
    sys = LearnedSystem(cfg)
    comp = sys.component_counts(); total = sys.n_trainable()
    ok_ceiling = total <= PARAM_CEILING
    analytic = ANALYTIC_Q64.get(cfg.key) if (cfg.n_probes == 64 and cfg.readout == "cls") else None
    if verbose:
        print(f"== {cfg.dataset}-{cfg.generator.upper()} C={cfg.n_classes} FFN{cfg.ffn} Q{cfg.n_probes} "
              f"readout={cfg.readout}{('x'+str(cfg.pma_seeds)) if cfg.readout=='pma' else ''} arch={cfg.readout_arch} ==")
        for k, v in comp.items():
            print(f"    {k:12s} {v:>10,}")
        print(f"    {'TOTAL':12s} {total:>10,}   <={PARAM_CEILING:,}: {'OK' if ok_ceiling else 'FAIL'}"
              + (f"   analytic {analytic:,} {'OK' if total==analytic else 'DIFF'}" if analytic else ""))
    return total, comp, ok_ceiling