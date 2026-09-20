"""END-TO-END LEARNED-PROBE neuron-profile trainer for INR classification.
LIVES IN / RUNS FROM the ProbeGen-multilayer repo (imports ProbeGen/data/pat from there).

Same downstream model as the frozen 0.57 winner (scripts/train_np.py's NeuronProfileNet: each neuron ->
its N-probe response vector; shared set-transformer pools neurons per layer (+layer emb); concat L
layer-vectors (+ y feature) -> MLP head), BUT the probes are now LEARNED end-to-end instead of a frozen
cached bank. Every step: generate probes from a trainable generator, forward each frozen INR at those
probes, capture pre-activations WITH grad, predict, cross-entropy; gradients flow through each frozen INR
back into the probe generator (Kahana's learned-probe protocol) AND the head.

Differences vs train_np.py:
  * probe_source (latents + generator) is TRAINABLE (own --probe_lr group); learned coords squashed to
    the INR domain via --domain_tanh (like main.py Arm-B).
  * NO activation cache (probes change every step) -> INRs are preloaded to GPU and forwarded live.
Single vs multi view is chosen by --splits (e.g. cifar10_splits vs cifar10_splits_multiview), like train_np.py.

Usage (reproduce-then-learn on CIFAR-10, head fixed at the 0.57 winner):
  PYTHONPATH=. python scripts/train_np_e2e.py --dataset cifar10_inr --splits cifar10_splits_multiview \
    --L 4 --H 32 --out_dim 3 --n_classes 10 --models_c_in 2 \
    --d 128 --nenc 2 --lr 5e-4 --probe_lr 1e-2 --domain_tanh 1 \
    --plateau_factor 0.2 --plateau_patience 6 --plateau_min_lr 1e-5 \
    --batch_size 32 --warmup 300 --epochs 50 --eval_every 1000 --exp_name e2e_cifar
"""
import argparse, math, os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root — portable, no hardcoded path
sys.path.insert(0, ROOT)   # so data.py and the models/ package import WITHOUT needing PYTHONPATH=.
from models.lowrank import make_linear
from models.probegen_utils import LowRankEncoderLayer, find_hidden_linear_layers, run_with_linear_activation_hooks
from models.hiddenprobe_probe_source import LearnedProbeSource
from data import INRDataset

NP = 128
SIREN_W0 = 30.0

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10_inr")
ap.add_argument("--splits", default="cifar10_splits_multiview", help="splits prefix (no .json); *_multiview = MV regime")
ap.add_argument("--L", type=int, default=4, help="# hidden layers of the target INR")
ap.add_argument("--H", type=int, default=32, help="hidden width of the target INR")
ap.add_argument("--out_dim", type=int, default=3, help="target INR output dim (fmnist=1, cifar=3)")
ap.add_argument("--n_classes", type=int, default=10)
ap.add_argument("--models_c_in", type=int, default=2, help="INR input coord dim (2=image INRs, 3=3D)")
ap.add_argument("--dataset_dir", default=None, help="root holding the INR checkpoints + splits json "
                "(default <repo>/experiments/inr_classification/dataset). Relative paths in the splits json "
                "resolve under this; pass an absolute --splits *.json to point anywhere.")
ap.add_argument("--runs_dir", default=None, help="output dir for this run (default <repo>/experiments/<dataset>/runs)")
# ---- head / aggregator (identical knobs to train_np.py) ----
ap.add_argument("--d", type=int, default=128); ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--nheads", type=int, default=4); ap.add_argument("--nenc", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--neuron_pool", choices=["mean", "attn"], default="mean",
                help="how to pool the H neuron tokens per layer: mean (default) | attn (learned-query "
                     "attention pool, +~d params)")
ap.add_argument("--cross_layer", type=int, default=0, help="1 = mix the L per-layer tokens with a low-rank "
                "self-attention before the head (cross-layer structure). +~4*d*xlayer_rank params.")
ap.add_argument("--xlayer_rank", type=int, default=32, help="rank of the cross-layer self-attention")
ap.add_argument("--head", choices=["neuron_profile", "set_transformer"], default="neuron_profile",
                help="neuron_profile (default, per-layer pool) | set_transformer (attention-through: all L*H "
                     "neuron tokens + y + CLS in one deep transformer, permutation-invariant over neurons, "
                     "late CLS readout — no early pooling). nenc controls depth; watch [e2e] head_params.")
ap.add_argument("--ema_decay", type=float, default=0.0, help="0 = off; >0 (e.g. 0.999) = keep an EMA of "
                "{head, probe_source} and eval/report with it (zero params, +~0.5-1%%)")
# ---- probe generator (NEW: learned) ----
ap.add_argument("--gen_type", default="linear_2_no_acts"); ap.add_argument("--gen_latent_z", type=int, default=32)
ap.add_argument("--generator_width", type=int, default=16)
ap.add_argument("--n_probes", type=int, default=128, help="number of learned probes (was hardcoded NP=128); model dims auto-scale.")
ap.add_argument("--domain_tanh", type=int, default=1, help="1 = squash learned probe coords to (-1,1)^d (recommended)")
ap.add_argument("--probe_lr", type=float, default=1e-2, help="LR for the learned probe generator (the big lever)")
# ---- optimization ----
ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--warmup_start", type=float, default=1e-5)
ap.add_argument("--plateau_factor", type=float, default=0.2); ap.add_argument("--plateau_patience", type=int, default=6)
ap.add_argument("--plateau_min_lr", type=float, default=1e-5)
ap.add_argument("--head_wd", type=float, default=0.0, help="decoupled (AdamW) weight decay on the HEAD group "
                "only (the probe generator stays wd=0). Regularizes the readout without shrinking the probes.")
ap.add_argument("--scheduler", choices=["plateau", "cosine", "cosine_restart"], default="plateau",
                help="plateau (default, ReduceLROnPlateau on val) | cosine (anneal to plateau_min_lr over the run) "
                     "| cosine_restart (warm restarts). Both LR groups scale together; warmup still applies first.")
ap.add_argument("--cosine_restarts", type=int, default=1, help="number of cosine cycles for cosine_restart")
ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--eval_bs", type=int, default=256)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--exp_name", required=True)
ap.add_argument("--ensemble_ckpts", default="", help="comma-separated best.pt paths: skip training, load each "
                "(must match this arch), average softmax over TEST, report per-seed + ensemble accuracy, exit.")
ap.add_argument("--init_ckpt", default="", help="WARM-START: init head+probe_source weights from this best.pt "
                "(must match this arch), then keep training with a fresh optimizer/scheduler. Weights only "
                "(best.pt has no optimizer state), so use a reduced --lr to continue without perturbing them.")
ap.add_argument("--feat_fourier", type=int, default=0, help="set_transformer: add K multi-scale Fourier features "
                "(sin+cos at geometric freqs around w0) of each probe-profile. 0=off. Costs 2*K*NP input dims.")
ap.add_argument("--feat_order", type=int, default=0, help="set_transformer: add M order-statistic features "
                "(M evenly-spaced quantiles of the sorted probe-profile; permutation-invariant). 0=off, cheap.")
ap.add_argument("--readout", default="cls", choices=["cls", "pma", "multi"], help="set_transformer readout: "
                "'cls' token (default); 'pma' = K learned seed-query attention pools (see --pma_seeds); "
                "'multi' = concat[CLS, mean-pool, max-pool] (widens the single-token readout bottleneck).")
ap.add_argument("--pma_seeds", type=int, default=1, help="readout=pma: number of learned pooling seed queries "
                "(K); readout is K*d wide -> less information bottleneck than a single CLS/PMA vector.")
args = ap.parse_args()
NP = args.n_probes   # override the module default (128) with the CLI value
DEV = "cuda"; L, H, OUT = args.L, args.H, args.out_dim
DS_DIR = args.dataset_dir or os.path.join(ROOT, "experiments", "inr_classification", "dataset")
SPLITS_JSON = args.splits if args.splits.endswith(".json") else args.splits + ".json"
RUNS_DIR = args.runs_dir or os.path.join(ROOT, "experiments", args.dataset, "runs")
EXP = os.path.join(RUNS_DIR, args.exp_name); os.makedirs(EXP, exist_ok=True)
LOG = os.path.join(EXP, "log.csv")
print(f"[e2e] repo_root={ROOT}", flush=True)
print(f"[e2e] dataset_dir={DS_DIR}  splits={SPLITS_JSON}  runs_dir={RUNS_DIR}", flush=True)


# ---- trainable probe generator (NOT frozen, NOT cached) ----
def build_probe_model():
    torch.manual_seed(0)  # same init as the frozen bank; then it LEARNS from here
    m = LearnedProbeSource(
        n_probes=NP,
        models_c_in=args.models_c_in,
        gen_type=args.gen_type,
        gen_latent_z=args.gen_latent_z,
        generator_width=args.generator_width,
        domain_tanh=bool(args.domain_tanh),
    )
    return m.to(DEV)

PM = build_probe_model()


def _capture_pre_activations(net, x):
    """Capture hidden Linear outputs before the following SIREN activation.

    This replaces the stale PATAdapter dependency that was no longer present in
    models/ProbeGen.py. Forward hooks preserve the autograd graph, so gradients
    still flow from the HiddenProbe loss through the frozen target INR to x and
    therefore into the learned probe generator.
    """
    hidden_layers = find_hidden_linear_layers(net, expected_count=L)
    y, acts = run_with_linear_activation_hooks(net, x, hidden_layers)
    return acts, y
PROBE_MOD = PM.probe_source
n_probe_params = sum(p.numel() for p in PROBE_MOD.parameters() if p.requires_grad)


def load_split(split):
    ds = INRDataset(dataset_dir=DS_DIR, splits_path=SPLITS_JSON, split=split)
    idx = list(range(len(ds)))
    nets, ys, t0 = [], [], time.time()
    for j, i in enumerate(idx):
        net, label = ds[i]
        for p in net.parameters(): p.requires_grad_(False)
        nets.append(net.to(DEV).eval()); ys.append(int(label))
        if (j + 1) % 20000 == 0: print(f"[e2e] preload {split} {j+1}/{len(idx)} ({(j+1)/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[e2e] {split}: {len(nets)} INRs on GPU in {time.time()-t0:.0f}s", flush=True)
    return nets, torch.tensor(ys, device=DEV)


def capture_batch(nets_b, x):
    """Live capture -> A[B,NP,L*H,1], f[B,NP,OUT]; keeps grad graph to x (the learned probes)."""
    Al, fl = [], []
    for net in nets_b:
        acts, y = _capture_pre_activations(net, x)   # grad: loss->acts->net(x)->x->generator
        Al.append(torch.cat(acts, dim=-1)); fl.append(y)
    A = torch.stack(Al, 0).unsqueeze(-1)                     # [B,NP,L*H,1]
    f = torch.stack(fl, 0)                                   # [B,NP,OUT]
    return A, f


def feats(nets_b):
    return capture_batch(nets_b, PM.generate_probes())


class _LowRankSelfAttn(nn.Module):   # cheap self-attention over the L layer-tokens (cross-layer mixing)
    def __init__(s, d, rank):
        super().__init__()
        s.q = nn.Linear(d, rank, bias=False); s.k = nn.Linear(d, rank, bias=False)
        s.v = nn.Linear(d, rank, bias=False); s.o = nn.Linear(rank, d)
        s.scale = rank ** -0.5; nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)  # identity-start residual
    def forward(s, x):                                   # [B,L,d] -> [B,L,d]
        w = torch.softmax((s.q(x) @ s.k(x).transpose(-1, -2)) * s.scale, dim=-1)
        return s.o(w @ s.v(x))


class NeuronProfileNet(nn.Module):   # neuron-profile head (probe-based; pre + post SIREN activations)
    def __init__(s, d, rank, nheads, nenc, L, out, nclass, dropout=0.0, neuron_pool="mean",
                 cross_layer=False, xlayer_rank=32):
        super().__init__(); s.L = L; s.neuron_pool = neuron_pool
        s.cross_layer = cross_layer
        s.n_stats = 7
        in_dim = 2 * NP + s.n_stats
        s.in_proj = make_linear(in_dim, d, rank=rank)
        if rank > 0:
            s.enc = nn.ModuleList([LowRankEncoderLayer(d, nheads, 4*d, rank, dropout) for _ in range(nenc)]); s.lr = True
        else:
            el = nn.TransformerEncoderLayer(d, nheads, 4*d, dropout, activation="relu", batch_first=True, norm_first=True)
            s.enc = nn.TransformerEncoder(el, nenc); s.lr = False
        if neuron_pool == "attn":                       # learned-query attention pool over the H neurons (+~d params)
            s.pool_q = nn.Parameter(torch.randn(d) * (1.0 / d) ** 0.5); s.pool_scale = 1.0 / math.sqrt(d)
        if cross_layer:
            s.xnorm = nn.LayerNorm(d); s.xlayer = _LowRankSelfAttn(d, xlayer_rank)
        s.layer_emb = nn.Embedding(L, d)
        s.y_proj = make_linear(NP*out, d, rank=rank)
        s.head = nn.Sequential(make_linear((L+1)*d, d, rank=rank), nn.ReLU(), make_linear(d, d, rank=rank), nn.ReLU(), make_linear(d, nclass, rank=rank))
    def _pool(s, z):                                     # z [B,H,d] -> [B,d]
        if s.neuron_pool == "attn":
            w = torch.softmax((z * s.pool_q).sum(-1) * s.pool_scale, dim=1)   # [B,H] attention over neurons
            return (w.unsqueeze(-1) * z).sum(1)
        return z.mean(1)
    def forward(s, A, f):
        B = A.shape[0]; A = A[..., 0].reshape(B, NP, s.L, H); per = []
        for l in range(s.L):
            z = A[:, :, l, :].transpose(1, 2)             # [B,H,NP] raw pre-activations
            feats = [z, torch.sin(SIREN_W0 * z)]
            r = z
            feats.append(torch.stack([r.mean(-1), r.std(-1), r.amax(-1), r.amin(-1),
                                      r.abs().mean(-1), (r > 0).float().mean(-1), r.norm(dim=-1)], dim=-1))
            z = s.in_proj(torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0])
            if s.lr:
                for lay in s.enc: z = lay(z)
            else: z = s.enc(z)
            per.append(s._pool(z) + s.layer_emb(torch.tensor(l, device=A.device)))
        lt = torch.stack(per, dim=1)                      # [B,L,d]
        if s.cross_layer:
            lt = lt + s.xlayer(s.xnorm(lt))               # mix the L layer-tokens
        return s.head(torch.cat([lt.reshape(B, s.L * lt.shape[-1]), s.y_proj(f.reshape(B, -1))], -1))


class SetTransformerHead(nn.Module):
    """Attention-through head: ALL (layer,neuron) probe-profiles as ONE set (permutation-invariant over
    neurons; layer identity via layer_emb, NO neuron-id) + the y token + a learned CLS, run through a deep
    joint transformer, read out once via CLS. No early pooling — the part NFT gets right. Probe-based, Q=128."""
    def __init__(s, d, nheads, nenc, L, out, nclass, dropout=0.0,
                 feat_fourier=0, feat_order=0, readout="cls", pma_seeds=1):
        super().__init__(); s.L = L
        s.n_stats = 7
        s.feat_fourier = feat_fourier; s.feat_order = feat_order; s.readout = readout; s.pma_seeds = pma_seeds
        if feat_fourier > 0:                              # geometric multi-scale freqs centered on w0
            s.register_buffer("ffreqs", SIREN_W0 * (2.0 ** torch.linspace(-2.0, 1.0, feat_fourier)))
        in_dim = 2 * NP + s.n_stats + (2 * feat_fourier * NP) + feat_order
        s.in_proj = make_linear(in_dim, d)
        s.layer_emb = nn.Embedding(L, d)
        s.y_proj = make_linear(NP * out, d)
        el = nn.TransformerEncoderLayer(d, nheads, 4 * d, dropout, activation="relu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(el, nenc)
        if readout == "pma":                             # K learned seed-query attention pools -> K*d readout
            s.pma_seed = nn.Parameter(torch.randn(1, pma_seeds, d) * 0.02)
            s.pma = nn.MultiheadAttention(d, nheads, dropout=dropout, batch_first=True)
            ro_dim = pma_seeds * d
        elif readout == "multi":                         # concat[CLS, mean-pool, max-pool] -> 3d readout
            s.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02); ro_dim = 3 * d
        else:                                            # single CLS token
            s.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02); ro_dim = d
        s.head = nn.Sequential(make_linear(ro_dim, d), nn.ReLU(), make_linear(d, nclass))
    def _feats(s, z):                                     # z [.., NP] -> [.., in_dim]
        feats = [z, torch.sin(SIREN_W0 * z)]
        if s.feat_fourier > 0:                            # multi-scale Fourier (sin+cos at each freq)
            for fk in s.ffreqs: feats.append(torch.sin(fk * z)); feats.append(torch.cos(fk * z))
        r = z
        feats.append(torch.stack([r.mean(-1), r.std(-1), r.amax(-1), r.amin(-1),
                                  r.abs().mean(-1), (r > 0).float().mean(-1), r.norm(dim=-1)], dim=-1))
        if s.feat_order > 0:                              # M quantiles of the sorted profile (perm-invariant)
            sz = torch.sort(z, dim=-1).values
            idx = torch.linspace(0, z.shape[-1] - 1, s.feat_order, device=z.device).round().long()
            feats.append(sz.index_select(-1, idx))
        return torch.cat(feats, -1) if len(feats) > 1 else feats[0]
    def forward(s, A, f):
        B = A.shape[0]; A = A[..., 0].reshape(B, NP, s.L, H)     # [B,NP,L,H]
        z = A.permute(0, 2, 3, 1)                                # [B,L,H,NP] each (l,h) = its NP-profile
        z = s.in_proj(s._feats(z))                              # [B,L,H,d]
        z = z + s.layer_emb(torch.arange(s.L, device=A.device))[None, :, None, :]   # layer id only
        tokens = z.reshape(B, s.L * H, -1)                      # [B, L*H, d]  (unordered neuron set)
        yt = s.y_proj(f.reshape(B, -1)).unsqueeze(1)            # [B,1,d]
        if s.readout == "pma":                                 # pool L*H+1 tokens with K learned seed queries
            x = s.enc(torch.cat([yt, tokens], dim=1))
            pooled, _ = s.pma(s.pma_seed.expand(B, -1, -1), x, x)   # [B,K,d]
            return s.head(pooled.reshape(B, -1))               # K*d readout (K seeds -> wider bottleneck)
        cls = s.cls.expand(B, -1, -1)                          # [B,1,d]
        x = s.enc(torch.cat([cls, yt, tokens], dim=1))         # deep joint transformer over L*H+2 tokens
        if s.readout == "multi":                               # concat CLS + mean + max over all tokens
            return s.head(torch.cat([x[:, 0], x.mean(1), x.max(1).values], dim=-1))
        return s.head(x[:, 0])                                  # single-CLS readout


if args.ensemble_ckpts:                                  # ensemble mode: only need the test split
    trN, trY, vaN, vaY = [], None, [], None; teN, teY = load_split("test")
else:
    trN, trY = load_split("train"); vaN, vaY = load_split("val"); teN, teY = load_split("test")
torch.manual_seed(args.seed)
if args.head == "set_transformer":
    head = SetTransformerHead(args.d, args.nheads, args.nenc, L, OUT, args.n_classes, args.dropout,
                              feat_fourier=args.feat_fourier, feat_order=args.feat_order,
                              readout=args.readout, pma_seeds=args.pma_seeds).to(DEV)
else:
    head = NeuronProfileNet(args.d, args.rank, args.nheads, args.nenc, L, OUT, args.n_classes, args.dropout,
                            neuron_pool=args.neuron_pool,
                            cross_layer=bool(args.cross_layer), xlayer_rank=args.xlayer_rank,
                            ).to(DEV)
Ph = sum(p.numel() for p in head.parameters())
print(f"[e2e] {args.exp_name} head_params={Ph:,} + learned_probe_params={n_probe_params:,} = {Ph+n_probe_params:,} "
      f"(domain_tanh={bool(args.domain_tanh)} probe_lr={args.probe_lr} "
      f"cross_layer={bool(args.cross_layer)} ema_decay={args.ema_decay})", flush=True)
if args.init_ckpt and not args.ensemble_ckpts:           # WARM-START: init live weights from a saved best.pt
    st = torch.load(args.init_ckpt, map_location=DEV)
    head.load_state_dict({k: v.to(DEV) for k, v in st["head"].items()})
    PROBE_MOD.load_state_dict({k: v.to(DEV) for k, v in st["probe_source"].items()})
    print(f"[e2e] warm-start: loaded head+probe_source from {args.init_ckpt}", flush=True)
# ---- EMA of {head, probe_source} (eval/report with it) ----  (seeds from warm-started weights if --init_ckpt)
ema = None
if args.ema_decay > 0:
    ema = {"head": {k: v.detach().clone() for k, v in head.state_dict().items()},
           "probe_source": {k: v.detach().clone() for k, v in PROBE_MOD.state_dict().items()}}
def ema_update():
    dcy = args.ema_decay
    for k, v in head.state_dict().items(): ema["head"][k].mul_(dcy).add_(v.detach(), alpha=1 - dcy)
    for k, v in PROBE_MOD.state_dict().items(): ema["probe_source"][k].mul_(dcy).add_(v.detach(), alpha=1 - dcy)
if args.ensemble_ckpts:                                  # load each ckpt, average softmax over TEST, report, exit
    ckpts = [c for c in args.ensemble_ckpts.split(",") if c]
    print(f"[ensemble] {args.exp_name}: {len(ckpts)} checkpoints over {len(teN)} test INRs", flush=True)
    sum_probs = None; per_seed = []
    for cp in ckpts:
        st = torch.load(cp, map_location=DEV)
        head.load_state_dict({k: v.to(DEV) for k, v in st["head"].items()})
        PROBE_MOD.load_state_dict({k: v.to(DEV) for k, v in st["probe_source"].items()})
        head.eval(); probs = []
        with torch.no_grad():
            for i in range(0, len(teN), args.eval_bs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    A, f = feats(teN[i:i+args.eval_bs]); o = head(A, f)
                probs.append(torch.softmax(o.float(), dim=-1))
        probs = torch.cat(probs, 0); acc = (probs.argmax(-1) == teY).float().mean().item()
        per_seed.append(acc); sum_probs = probs if sum_probs is None else sum_probs + probs
        print(f"[ensemble]   {os.path.basename(os.path.dirname(cp))} test_acc={acc:.4f}", flush=True)
    ens = (sum_probs.argmax(-1) == teY).float().mean().item()
    print(f"[ensemble] {args.exp_name} per_seed={['%.4f' % a for a in per_seed]} "
          f"mean={sum(per_seed)/len(per_seed):.4f} ENSEMBLE={ens:.4f}", flush=True)
    sys.exit(0)

opt = torch.optim.AdamW([                                 # AdamW: decoupled wd on the head group only
    {"params": list(head.parameters()), "lr": args.lr, "weight_decay": args.head_wd},
    {"params": [p for p in PROBE_MOD.parameters() if p.requires_grad], "lr": args.probe_lr, "weight_decay": 0.0},
])
sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.plateau_factor,
         patience=args.plateau_patience, min_lr=args.plateau_min_lr) if args.scheduler == "plateau" else None)
total_steps = args.epochs * ((len(trN) + args.batch_size - 1) // args.batch_size)   # for cosine schedules

@torch.no_grad()
def ev(nets, y, swap_ema=True):
    head.eval()
    swap = ema is not None and swap_ema
    if swap:                                             # eval with the EMA weights, then restore live weights
        bak_h = {k: v.detach().clone() for k, v in head.state_dict().items()}
        bak_p = {k: v.detach().clone() for k, v in PROBE_MOD.state_dict().items()}
        head.load_state_dict(ema["head"]); PROBE_MOD.load_state_dict(ema["probe_source"])
    c = 0
    for i in range(0, len(nets), args.eval_bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            A, f = feats(nets[i:i+args.eval_bs]); o = head(A, f)
        c += (o.argmax(-1) == y[i:i+args.eval_bs]).sum().item()
    if swap:
        head.load_state_dict(bak_h); PROBE_MOD.load_state_dict(bak_p)
    head.train(); return c / len(nets)

def log(r):
    new = not os.path.exists(LOG)
    with open(LOG, "a") as fp:
        if new: fp.write("exp_name,epoch,global_step,lr,probe_lr,train_loss,val_acc,test_acc,is_best\n")
        fp.write(r + "\n")

N = len(trN); step = 0; best = 0.0; t0 = time.time(); best_state = None
for ep in range(args.epochs):
    perm = torch.randperm(N)
    for bi in range(0, N, args.batch_size):
        idx = perm[bi:bi+args.batch_size].tolist()
        nets_b = [trN[i] for i in idx]; yb = trY[idx]
        if args.warmup > 0 and step < args.warmup:
            fac = (step + 1) / args.warmup
            opt.param_groups[0]["lr"] = args.warmup_start + (args.lr - args.warmup_start) * fac
            opt.param_groups[1]["lr"] = args.warmup_start + (args.probe_lr - args.warmup_start) * fac
        elif args.scheduler in ("cosine", "cosine_restart"):   # anneal both groups together each step (post-warmup)
            denom = max(1, total_steps - args.warmup)
            p = (step - args.warmup) / denom
            if args.scheduler == "cosine_restart":
                T = max(1, denom // max(1, args.cosine_restarts)); p = ((step - args.warmup) % T) / T
            fac = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
            opt.param_groups[0]["lr"] = args.plateau_min_lr + (args.lr - args.plateau_min_lr) * fac
            opt.param_groups[1]["lr"] = args.plateau_min_lr + (args.probe_lr - args.plateau_min_lr) * fac
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            A, f = feats(nets_b); out = head(A, f)
            loss = F.cross_entropy(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) +
                                       [p for p in PROBE_MOD.parameters() if p.requires_grad], 1.0)
        opt.step(); step += 1
        if ema is not None: ema_update()
        if step % args.eval_every == 0:
            va = ev(vaN, vaY); te = ev(teN, teY)   # EMA-weighted eval when ema_decay>0
            if args.scheduler == "plateau" and step >= args.warmup: sched.step(va)
            isb = va > best
            if isb:
                best = va
                src_h = ema["head"] if ema is not None else head.state_dict()   # save the EMA snapshot when on
                src_p = ema["probe_source"] if ema is not None else PROBE_MOD.state_dict()
                best_state = {"head": {k: v.detach().cpu().clone() for k, v in src_h.items()},
                              "probe_source": {k: v.detach().cpu().clone() for k, v in src_p.items()}}
                torch.save(best_state, os.path.join(EXP, "best.pt"))
            log(f"{args.exp_name},{ep},{step},{opt.param_groups[0]['lr']:.2e},{opt.param_groups[1]['lr']:.2e},{loss.item():.4f},{va:.4f},{te:.4f},{isb}")
            print(f"[e2e] step={step} ({step/(time.time()-t0):.2f}/s) loss={loss.item():.3f} val={va:.4f} test={te:.4f} best={best:.4f}", flush=True)
if best_state is not None:
    head.load_state_dict({k: v.to(DEV) for k, v in best_state["head"].items()})
    PROBE_MOD.load_state_dict({k: v.to(DEV) for k, v in best_state["probe_source"].items()})
fva = ev(vaN, vaY, swap_ema=False); fte = ev(teN, teY, swap_ema=False)   # best_state already holds the EMA snapshot
print(f"[FINAL] {args.exp_name} params={Ph+n_probe_params:,} best_val={best:.4f} val={fva:.4f} test={fte:.4f}", flush=True)
log(f"FINAL,{args.epochs},{step},,,,{fva:.4f},{fte:.4f},best")
