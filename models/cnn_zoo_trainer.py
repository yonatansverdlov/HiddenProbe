"""Track B trainer: ProbeGen-H (and Kahana = hidden_mode=off) on Wild Park, matched protocol.

Reuses the official backbone (via probegen_h.ProbeGenH), the validated late hidden encoder, the WP
split/loader (data.py), and the shared metrics (models/metrics_cnn.py). Kahana rows are ProbeGenH(hidden_mode=off)
so the ONLY difference from a ProbeGen-H row is the hidden pathway.

Efficiency is the headline: every run prints total target-CNN queries (= P_out + P_hidden).

Example (compact ProbeGen-H, 32 output + 32 hidden probes):
  python main.py cnn_zoo --adapter_preset compact \
     --n_out_probes 32 --n_hidden_probes 32 --hidden_mode on --n_train 0 --epochs 30 \
     --exp_name pgh_c_32+32_s0 --out_dir checkpoints/track_b/pgh_c_32+32_s0
Kahana baseline (128 output probes):
  python main.py cnn_zoo --hidden_mode off --n_out_probes 128 \
     --exp_name kahana_128_s0 --out_dir checkpoints/track_b/kahana_128_s0
"""
import argparse, os, time, json, random, subprocess, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from models.probegen_h import ProbeGenH
from data import load_cnns, DEFAULT_SPLITS
from models.metrics_cnn import acc_to_logit, all_metrics


def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:                       # reproducibility diagnostic: also force deterministic non-cuDNN kernels
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede the first cuBLAS call
        torch.use_deterministic_algorithms(True, warn_only=True)      # warn (not fail) on ops with no deterministic impl
    print(f"[SEED] random={seed} numpy={seed} torch={seed} cuda={seed} "
          f"cudnn.deterministic=True cudnn.benchmark=False deterministic_algorithms={deterministic}", flush=True)


def _rng_capture():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _rng_restore(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"].cpu() if torch.is_tensor(s["torch_cpu"]) else s["torch_cpu"])
    if s.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["torch_cuda"]])


def save_training_state(path, model, opt, sched, epoch, step, best, bstate):
    """Full resume state saved at an EPOCH BOUNDARY: everything needed to continue bit-for-bit."""
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "global_step": step, "best_val_tau": best,
                "best_state": bstate, "rng": _rng_capture()}, path)


def write_run_metadata(out_dir, args):
    """resolved_config.yaml + command.txt + git_commit.txt + environment.txt (reproducibility §9)."""
    with open(os.path.join(out_dir, "resolved_config.yaml"), "w") as f:
        for k in sorted(vars(args)):
            f.write(f"{k}: {getattr(args, k)}\n")
    with open(os.path.join(out_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    open(os.path.join(out_dir, "git_commit.txt"), "w").write(sha + "\n")
    with open(os.path.join(out_dir, "environment.txt"), "w") as f:
        f.write(f"python={sys.version.split()[0]}\ntorch={torch.__version__}\n"
                f"cuda={torch.version.cuda}\ndevice={args.device}\n"
                f"target_space={args.target_space}\nscheduler={args.scheduler}\n"
                f"metrics_mode={args.target_space}\nseed={args.seed}\nsplits={args.splits}\n")

PRESETS = {  # hidden adapter sizes; interaction_rank overridable via CLI
    "compact":    dict(hidden_dim=64,  z_hidden_dim=64,  fusion_hidden=128, fusion_out=64),
    "expressive": dict(hidden_dim=128, z_hidden_dim=128, fusion_hidden=256, fusion_out=128),
}

ap = argparse.ArgumentParser()
ap.add_argument("--hidden_mode", choices=["on", "off"], default="on")
ap.add_argument("--probe_sharing", choices=["shared", "separate"], default="shared",
                help="shared (PRIMARY: Q unique inputs, one forward gives logits+hidden) | separate (ablation)")
ap.add_argument("--adapter_preset", choices=["compact", "expressive"], default="compact")
ap.add_argument("--n_out_probes", type=int, default=64)
ap.add_argument("--n_hidden_probes", type=int, default=32)
ap.add_argument("--interaction_rank", type=int, default=32)
ap.add_argument("--hidden_agg", choices=["neuron_collapse", "neuron_profile"], default="neuron_collapse",
                help="hidden aggregation: neuron_collapse (recipe ii, DEFAULT = current model) | "
                     "neuron_profile (recipe i: channel=cross-probe profile, set-pool channels last)")
ap.add_argument("--probe_mixer", choices=["none", "attn", "tokenmix"], default="none",
                help="recipe-(i) only: probe-axis mixer before readout (none | low-rank self-attn | Linear(P->P))")
ap.add_argument("--mixer_hidden", type=int, default=256, help="Kahana classifier width (raise for capacity-matched control)")
ap.add_argument("--hidden_dim", type=int, default=0, help="override preset hidden_dim (0=use preset)")
# training
ap.add_argument("--n_train", type=int, default=0, help="0=all 113586")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4, help="readout lr (Kahana WP default 3e-4)")
ap.add_argument("--probe_lr", type=float, default=3e-4, help="lr for latents+generator")
ap.add_argument("--hidden_lr", type=float, default=0.0,
                help="0 = no separation (hidden-branch readout uses --lr). >0 = SEPARATE lr for the hidden "
                     "branch (tokenizer/encoder/interaction/fusion) vs the output/Kahana readout (--lr). "
                     "Optimizer-only; no architecture change.")
ap.add_argument("--rank_loss_w", type=float, default=0.0)
ap.add_argument("--target_space", choices=["raw", "logit"], default="raw",
                help="raw = regress on accuracy (Kahana official); logit = regress on acc_to_logit(acc)")
ap.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine",
                help="cosine = CosineAnnealingLR every step (Kahana official); plateau = ReduceLROnPlateau on val tau")
ap.add_argument("--weight_decay", type=float, default=0.0)
ap.add_argument("--warmup", type=int, default=0, help="0 = no warmup (Kahana official)")
ap.add_argument("--grad_clip", type=float, default=0.0, help="0 = no clipping (Kahana official); >0 clips grad norm")
ap.add_argument("--plateau_factor", type=float, default=0.5)
ap.add_argument("--plateau_patience", type=int, default=6)
ap.add_argument("--plateau_min_lr", type=float, default=1e-5)
ap.add_argument("--init_kahana_checkpoint", default="", help="load Kahana backbone weights, then joint train")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_subset", type=int, default=1485)
ap.add_argument("--val_limit", type=int, default=0, help="0=all val CNNs; >0 loads only N (smoke/debug)")
ap.add_argument("--test_limit", type=int, default=0, help="0=all test CNNs; >0 loads only N (smoke/debug)")
ap.add_argument("--eval_every", type=int, default=2500)
ap.add_argument("--eval_cnn_bs", type=int, default=256)
ap.add_argument("--splits", default=DEFAULT_SPLITS); ap.add_argument("--zip", default="")
ap.add_argument("--cnn_cache", default="", help="dir with prebuilt cnn_cache_<split>.pt (build_cnn_cache.py) for "
                "~5-6x faster CIFAR-WP loading (bit-identical CNNs). '' = serial zip load (default, unchanged).")
ap.add_argument("--models_c_in", type=int, default=3, help="target-CNN input channels: 3=CIFAR/WP, 1=grayscale SVHN-GS")
ap.add_argument("--gen_type", default="deep_linear_5", help="probe generator: deep_linear_5 (WP canonical), "
                "deep_linear_6 (CIFAR-GS / Kahana nfn_cnn_zoo official), etc. (build_probe_source gen_type).")
ap.add_argument("--zoo", default="wp", choices=["wp", "svhn_gs", "cifar_gs", "mnist_gs", "fmnist_gs"],
                help="wp=CIFAR/Wild-Park (data.load_cnns); *_gs=Unterthiner SmallCNN Zoo (svhn/cifar/mnist/fmnist grayscale), "
                     "loaded via data.load_svhn_cnns (identical 4970-dim SmallCNN, only the training data differs).")
ap.add_argument("--zoo_data_dir", default="", help="override the SmallCNN-zoo data dir for a *_gs zoo (weights.npy/metrics.csv.gz/layout.csv).")
ap.add_argument("--zoo_split", default="", help="override the split csv filename for a *_gs zoo (e.g. svhn_split_nfn.csv). Auto-generated seeded permutation if absent.")
ap.add_argument("--activation", default="relu", choices=["relu", "tanh"], help="svhn_gs only: which activation subset (ScaleGMN benches ReLU=0.8689, Tanh=0.8736)")
ap.add_argument("--dataset_name", default="", help="label printed at startup (e.g. SVHN); hard-fails if it "
                "is SVHN but --splits looks like a CIFAR/Wild-Park path (leakage guard).")
ap.add_argument("--exp_name", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--assert_canonical", type=int, default=0,
                help="1 = fail loudly unless canonical primary protocol (target_space=raw, shared probes, "
                     "canonical deep_linear_5 probe source). Set by the Stage-2/3 generators.")
ap.add_argument("--allow_sparse_eval", type=int, default=0,
                help="1 = permit <6 planned periodic evals (else the run aborts: short jobs must have a "
                     "cadence that actually evaluates). Set only for deliberate long-cadence full-data runs.")
ap.add_argument("--eval_only_ckpt", default="",
                help="SALVAGE mode: load this final-epoch checkpoint, eval final-epoch VAL(+test) with the "
                     "identical model/protocol, write summary.json (final_val_*), and exit — NO training. "
                     "Only valid for cosine jobs (cadence-independent training).")
ap.add_argument("--resume_training_state", default="",
                help="resume from a training_state.pt (model+probes+optimizer+scheduler+epoch+step+best+RNG). "
                     "Continues from the saved epoch up to --epochs (total target), NOT +epochs more.")
ap.add_argument("--sched_total_epochs", type=int, default=0,
                help="cosine T_max horizon in epochs (0=use --epochs). Set to the FINAL total (e.g. 30) when a "
                     "run stops early (e.g. 12) but will later resume, so the cosine schedule stays continuous.")
ap.add_argument("--dump_preds", default="",
                help="with --eval_only_ckpt: also save per-example val/test (true,pred) to this .npz "
                     "(for the positive-affine calibration audit).")
ap.add_argument("--probe_warmup_steps", type=int, default=0,
                help="WAVE-2 knob: for the first W optimizer steps, freeze the probe_source (latents+generator "
                     "= optimizer group1) LR at 0 so the readout/hidden encoder stabilizes first, then switch "
                     "directly to --probe_lr. group0 (readout) LR is unaffected. Default 0 = OFF (UNCHANGED).")
ap.add_argument("--deterministic", type=int, default=0,
                help="1 = torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG (reproducibility diagnostic; slower)")
ap.add_argument("--skip_test_eval", type=int, default=0,
                help="TUNING MODE: 1 = skip ALL test-set eval at finalization (final-epoch test, best-val "
                     "test, hidden-zero/shuffle diagnostics, dump_preds/calibration) to save compute and "
                     "prevent test leakage. Still saves best/final/training_state and a validation-only "
                     "summary.json (with val trajectory). Default 0 = full eval (behavior UNCHANGED).")
ap.add_argument("--probe_dropout", type=float, default=0.0,
                help="TRAINING-TIME probe dropout (shared mode only): each training batch, drop each of "
                     "the Q probes per-example with prob p (keep_prob=1-p). Dropped probes' logit blocks "
                     "contribute ZERO to the Kahana path and get -inf softmax / active-count mean in the "
                     "hidden path. Eval uses ALL Q probes (no dropout). Default 0.0 = OFF (UNCHANGED).")
ap.add_argument("--init_state_dict", default="",
                help="load a FULL model state_dict (e.g. the nested Q128-from-Q64 construction) as the "
                     "initialization, then train with a FRESH optimizer+scheduler from epoch 0. Unlike "
                     "--init_kahana_checkpoint (backbone-only + re-init kahana-preserving), this loads ALL "
                     "tensors verbatim (strict). Mutually exclusive with --resume_training_state.")
ap.add_argument("--eval_at_init", type=int, default=0,
                help="1 = evaluate VAL at step 0 (before any training) and seed it as the historical "
                     "best-val checkpoint (best.pt). Ensures a strong warm/nested init remains eligible as "
                     "best-val even if fine-tuning transiently degrades. Default 0 = OFF (UNCHANGED).")
ap.add_argument("--probe_min_lr", type=float, default=-1.0,
                help="plateau only: per-group min_lr floor for the PROBE group (latents+generator, the last "
                     "optimizer group). Use when probe_lr < plateau_min_lr so ReduceLROnPlateau does not "
                     "raise the probe LR up to plateau_min_lr. <0 = use scalar --plateau_min_lr (UNCHANGED).")
args = ap.parse_args()
if args.probe_dropout > 0.0 and args.probe_sharing != "shared":
    print(f"[ABORT] --probe_dropout {args.probe_dropout} requires --probe_sharing shared (got "
          f"{args.probe_sharing}): masks must align across the shared output/hidden probe bank.", flush=True)
    raise SystemExit(5)
if args.init_state_dict and args.resume_training_state:
    print("[ABORT] --init_state_dict and --resume_training_state are mutually exclusive.", flush=True)
    raise SystemExit(5)

if args.assert_canonical:
    _viol = []
    if args.target_space != "raw":      _viol.append(f"target_space={args.target_space} (must be raw)")
    if args.probe_sharing != "shared":  _viol.append(f"probe_sharing={args.probe_sharing} (must be shared)")
    # PGH always builds build_probe_source(deep_linear_5); assert the code path is present.
    from models.probegen_h import ProbeGenH as _PGH
    import inspect as _insp
    if "build_probe_source" not in _insp.getsource(_PGH.__init__):
        _viol.append("ProbeGenH no longer constructs build_probe_source (canonical probe path modified)")
    if _viol:
        print(f"[PROTOCOL-ASSERT] FAIL exp={args.exp_name}: " + "; ".join(_viol), flush=True)
        raise SystemExit(2)
    print(f"[PROTOCOL-ASSERT] PASS exp={args.exp_name} model=pgh target_space=raw probe_sharing=shared "
          f"probe_source=build_probe_source({args.gen_type})", flush=True)

if args.dataset_name:
    # Leakage guard only applies to the wp loader (splits/zip). zoo=svhn_gs bypasses the WP loader
    # entirely (SmallCNN adapter), so --splits is unused and not a leakage vector.
    if args.zoo == "wp":
        _sp = os.path.basename(args.splits).lower()
        if args.dataset_name.upper() == "SVHN" and any(k in _sp for k in ("cnn_park", "wild", "cifar")):
            print(f"[DATASET] FAIL exp={args.exp_name}: dataset_name=SVHN but --splits={args.splits} is a "
                  f"CIFAR/Wild-Park path — refusing to run (leakage guard).", flush=True)
            raise SystemExit(4)
        print(f"[DATASET] dataset={args.dataset_name} splits={os.path.basename(args.splits)} "
              f"zip={os.path.basename(args.zip) if args.zip else '(default)'}", flush=True)
    else:
        print(f"[DATASET] dataset={args.dataset_name} zoo={args.zoo} (wp splits/zip bypassed)", flush=True)

seed_everything(args.seed, deterministic=bool(args.deterministic))
DEV = args.device
os.makedirs(args.out_dir, exist_ok=True)
# ---- idempotency guard: skip (exit 0) a FROM-SCRATCH run whose out_dir already holds a completed run.
# Prevents duplicate/clobbering reruns (e.g. the same config queued on two hosts). Exempts resume/eval.
# Delete summary.json to force a rerun.
if (not args.resume_training_state and not args.eval_only_ckpt
        and os.path.exists(os.path.join(args.out_dir, "summary.json"))):
    print(f"[SKIP] {args.exp_name}: {args.out_dir}/summary.json already exists (run complete) — "
          f"exiting WITHOUT retraining. Delete summary.json to force a rerun.", flush=True)
    raise SystemExit(0)
write_run_metadata(args.out_dir, args)
LOG = os.path.join(args.out_dir, "log.csv")

# ---- data (identical split across all methods) --------------------------------------------------
# Unterthiner SmallCNN grayscale zoos (all share the 4970-dim fixed CNN; only data_dir + split differ).
from data import ZOO_DIRS as _ZOO_DIRS   # portable, env-overridable (see data.py)
_SMALLCNN_ZOOS = {
    "svhn_gs":  (_ZOO_DIRS["svhn_gs"],  "svhn_split.csv"),
    "cifar_gs": (_ZOO_DIRS["cifar_gs"], "cifar_gs_split.csv"),
    "mnist_gs": (_ZOO_DIRS["mnist_gs"], "mnist_split.csv"),
    "fmnist_gs":(_ZOO_DIRS["fmnist_gs"],"fashion_mnist_split.csv"),
}
if args.zoo in _SMALLCNN_ZOOS:
    from data import load_svhn_cnns
    _ddir, _dsplit = _SMALLCNN_ZOOS[args.zoo]
    _ddir = args.zoo_data_dir or _ddir
    _dsplit = args.zoo_split or _dsplit
    print(f"[DATASET] zoo={args.zoo} data_dir={_ddir} split={_dsplit} activation={args.activation} "
          f"(Unterthiner SmallCNN zoo; NO INR/CIFAR-WP weights)", flush=True)
    trN, trY = load_svhn_cnns("train", args.activation, DEV, data_dir=_ddir, split_csv=_dsplit, limit=args.n_train)
    vaN, vaY = load_svhn_cnns("val", args.activation, DEV, data_dir=_ddir, split_csv=_dsplit, limit=args.val_limit)
    teN, teY = load_svhn_cnns("test", args.activation, DEV, data_dir=_ddir, split_csv=_dsplit, limit=args.test_limit)
else:
    zp = args.zip or None
    _cc = args.cnn_cache or None
    trN, trY = load_cnns("train", args.n_train, DEV, args.splits, zp, cnn_cache=_cc)
    vaN, vaY = load_cnns("val", args.val_limit, DEV, args.splits, zp, cnn_cache=_cc)
    teN, teY = load_cnns("test", args.test_limit, DEV, args.splits, zp, cnn_cache=_cc)
# regression target: raw accuracy (Kahana official) or logit-space
trTGT = (trY if args.target_space == "raw" else acc_to_logit(trY)).to(DEV)

# ---- model ---------------------------------------------------------------------------------------
preset = dict(PRESETS[args.adapter_preset])
if args.hidden_dim > 0:
    preset["hidden_dim"] = args.hidden_dim
model = ProbeGenH(n_out_probes=args.n_out_probes, n_hidden_probes=args.n_hidden_probes,
                  interaction_rank=args.interaction_rank, mixer_hidden=args.mixer_hidden,
                  hidden_mode=args.hidden_mode, probe_sharing=args.probe_sharing, models_c_in=args.models_c_in,
                  gen_type=args.gen_type, hidden_agg=args.hidden_agg, probe_mixer=args.probe_mixer,
                  **preset).to(DEV)

if args.init_kahana_checkpoint:                       # warm-start: load backbone, keep training ALL
    sd = torch.load(args.init_kahana_checkpoint, map_location=DEV)
    bk = {k: v for k, v in sd.items() if k.startswith(("probe_source", "kahana_mlp"))}
    model.load_state_dict(bk, strict=False)
    if args.hidden_mode == "on":
        model._init_kahana_preserving()               # re-assert y_hat==y_K at (warm) init
    print(f"[warm-start] loaded {len(bk)} Kahana tensors from {args.init_kahana_checkpoint}", flush=True)

if args.init_state_dict:                              # full constructed init (e.g. nested Q128-from-Q64)
    isd = torch.load(args.init_state_dict, map_location=DEV)
    isd = isd.get("model", isd) if isinstance(isd, dict) and "model" in isd else isd
    model.load_state_dict({k: v.to(DEV) for k, v in isd.items()})   # strict: exact constructed init
    print(f"[INIT-STATE] {args.exp_name} loaded FULL model state_dict from {args.init_state_dict} "
          f"({len(isd)} tensors) — fresh optimizer+schedule from epoch 0.", flush=True)
if args.probe_dropout > 0.0:
    print(f"[PROBEDROP] exp={args.exp_name} probe_dropout={args.probe_dropout} keep_prob={1-args.probe_dropout:.3f} "
          f"(per-example Bernoulli mask over {args.n_out_probes} probes each train batch; eval uses ALL "
          f"{args.n_out_probes}; default-0 path is bit-identical to pre-dropout).", flush=True)

rep = model.param_report()
print(f"[PARAMS] exp={args.exp_name} mode={args.hidden_mode} preset={args.adapter_preset} "
      f"P_out={args.n_out_probes} P_hidden={args.n_hidden_probes} | "
      + " ".join(f"{k}={v:,}" for k, v in rep.items() if not k.startswith("queries_")), flush=True)
_q = model.n_target_queries()
print(f"[QUERIES] exp={args.exp_name} probe_sharing={_q['probe_sharing']} "
      f"output_probe_count={_q['out_probes']} hidden_probe_count={_q['hidden_probes']} "
      f"unique_query_count={_q['unique_query_count']} target_forward_input_count={_q['target_forward_input_count']}", flush=True)
print(f"[PROTOCOL] exp={args.exp_name} target_space={args.target_space} scheduler={args.scheduler} "
      f"warmup={args.warmup} grad_clip={args.grad_clip} rank_loss_w={args.rank_loss_w} "
      f"lr={args.lr} probe_lr={args.probe_lr} epochs={args.epochs} bs={args.batch_size}", flush=True)
from models.lr_dipt_lowrank import count_params as _cp
print(f"[PROBES] type={args.gen_type} Q={args.n_out_probes} latent_dim={model.gen_latent_z} "
      f"generator_params={_cp(model.probe_source.generator)} latent_params={model.probe_source.input.numel()} "
      f"trainable_generator={all(p.requires_grad for p in model.probe_source.generator.parameters())} "
      f"trainable_latents={model.probe_source.input.requires_grad} unique_query_count={_q['unique_query_count']} "
      f"latent_init_std={model.probe_source.input.std().item():.3f} probe_source=build_probe_source({args.gen_type})", flush=True)
print(f"[RECORD] model=pgh Q={args.n_out_probes} target_space={args.target_space} probe_type={args.gen_type} "
      f"latent_dim={model.gen_latent_z} latent_init_std={model.probe_source.input.std().item():.3f} "
      f"unique_query_count={_q['unique_query_count']} lr={args.lr} probe_lr={args.probe_lr} "
      f"scheduler={args.scheduler} batch_size={args.batch_size} rank_loss={args.rank_loss_w} "
      f"warmup={args.warmup} grad_clip={args.grad_clip} "
      f"capacity[adapter_preset={args.adapter_preset} hidden_dim={preset.get('hidden_dim')} "
      f"interaction_rank={args.interaction_rank} mixer_hidden={args.mixer_hidden}]", flush=True)

# ---- optim: probe source (latents+generator) @ probe_lr vs readout @ lr -------------------------
# For canonical Kahana, set probe_lr=lr=3e-4 -> ALL params (latents, generator, readout) get one lr.
_HIDDEN_PREFIXES = ("meta_enc", "tokenizer", "late_branch", "inter_A", "inter_B", "fusion_body", "fusion_final")
probe_params, out_params, hid_params = [], [], []
for n, p in model.named_parameters():
    if n.startswith(("probe_source", "hidden_probe_source")):
        probe_params.append(p)
    elif n.startswith(_HIDDEN_PREFIXES):
        hid_params.append(p)          # hidden branch (tokenizer/encoder/interaction/fusion)
    else:
        out_params.append(p)          # output/Kahana readout (kahana_mlp)
if args.hidden_lr > 0:                 # 3-group SEPARATED optimizer
    groups = [{"params": out_params, "lr": args.lr, "weight_decay": args.weight_decay},
              {"params": hid_params, "lr": args.hidden_lr},
              {"params": probe_params, "lr": args.probe_lr}]
    _base_lrs = (args.lr, args.hidden_lr, args.probe_lr)
    print(f"[OPTIM] SEPARATED | group0(output/kahana) lr={args.lr} n={len(out_params)} | "
          f"group1(hidden-branch) lr={args.hidden_lr} n={len(hid_params)} | "
          f"group2(probe_source) lr={args.probe_lr} n={len(probe_params)}", flush=True)
else:                                  # 2-group (backward compatible: readout = out+hidden @ --lr)
    groups = [{"params": out_params + hid_params, "lr": args.lr, "weight_decay": args.weight_decay},
              {"params": probe_params, "lr": args.probe_lr}]
    _base_lrs = (args.lr, args.probe_lr)
    print(f"[OPTIM] group0(readout=output+hidden) lr={args.lr} wd={args.weight_decay} n={len(out_params)+len(hid_params)} | "
          f"group1(probe_source=latents+generator) lr={args.probe_lr} n={len(probe_params)} | "
          f"all_one_lr={args.lr == args.probe_lr}", flush=True)
opt = torch.optim.Adam(groups)
import math as _math
_steps_per_epoch = _math.ceil(len(trN) / args.batch_size)
_total_steps = args.epochs * _steps_per_epoch
_sched_epochs = args.sched_total_epochs or args.epochs               # cosine horizon (>= --epochs for early-stop+resume)
_sched_total_steps = _sched_epochs * _steps_per_epoch
_planned_evals = _total_steps // max(1, args.eval_every)             # periodic (mid-training) evals
print(f"[EVALPLAN] exp={args.exp_name} total_steps={_total_steps} steps_per_epoch={_steps_per_epoch} "
      f"eval_every={args.eval_every} planned_periodic_evals={_planned_evals} "
      f"(+1 guaranteed final-epoch val)", flush=True)
if _planned_evals < 6 and not args.allow_sparse_eval and not args.eval_only_ckpt:
    print(f"[EVALPLAN] ABORT exp={args.exp_name}: planned_periodic_evals={_planned_evals} < 6 "
          f"(eval_every={args.eval_every} vs total_steps={_total_steps}). This is the Round-1 bug: a short job "
          f"with too-coarse cadence gets ~0 evals and a corrupted plateau trajectory. Lower --eval_every or "
          f"pass --allow_sparse_eval 1 for a deliberate long-cadence full-data run.", flush=True)
    raise SystemExit(3)
if args.scheduler == "cosine":                                        # Kahana official: step every batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, _sched_total_steps - args.warmup))
    print(f"[SCHED] cosine T_max={_sched_total_steps - args.warmup} (horizon {_sched_epochs} epochs; "
          f"run stops at --epochs {args.epochs})", flush=True)
else:                                                                 # plateau: step on val tau at eval
    if args.probe_min_lr >= 0:
        # per-group min_lr: probe group (ALWAYS the last optimizer group) floored at --probe_min_lr so its
        # intended smaller relative LR is preserved (a scalar min_lr would RAISE probe_lr up to plateau_min_lr
        # on the first reduction when probe_lr < plateau_min_lr).
        _min_lrs = [args.plateau_min_lr] * (len(groups) - 1) + [args.probe_min_lr]
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.plateau_factor,
                                                           patience=args.plateau_patience, min_lr=_min_lrs)
        print(f"[SCHED] plateau per-group min_lr={_min_lrs} (readout floor={args.plateau_min_lr}, "
              f"probe floor={args.probe_min_lr}); base lrs={_base_lrs}", flush=True)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.plateau_factor,
                                                           patience=args.plateau_patience, min_lr=args.plateau_min_lr)


def rank_loss(pred, tgt):
    di = pred[:, None] - pred[None, :]; dt = tgt[:, None] - tgt[None, :]
    m = (dt.abs() > 1e-6)
    return F.softplus(-di * torch.sign(dt))[m].mean() if m.any() else pred.sum() * 0.0


@torch.no_grad()
def ev(nets, y, zero_hidden=False, shuffle_hidden=False, nmax=None):
    model.eval(); preds = []
    n = len(nets) if nmax is None else min(nmax, len(nets))
    for i in range(0, n, args.eval_cnn_bs):
        b = nets[i:min(i + args.eval_cnn_bs, n)]        # clip to n (nets may be longer than nmax)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            p = model(b, device=DEV, zero_hidden=zero_hidden, shuffle_hidden=shuffle_hidden)
        preds.append(p.float().cpu())
    model.train()
    return all_metrics(torch.cat(preds), y[:n], space=args.target_space)


def logrow(r):
    new = not os.path.exists(LOG)
    with open(LOG, "a") as fp:
        if new: fp.write("exp,step,lr,loss,val_tau,val_accmse,val_accmae,test_tau,test_accmse,test_accmae,queries,is_best\n")
        fp.write(r + "\n")


Q = model.n_target_queries()["total_queries"]

if args.eval_only_ckpt:                                       # SALVAGE / pred-dump: score a ckpt, no training
    sd = torch.load(args.eval_only_ckpt, map_location=DEV)
    model.load_state_dict(sd)
    if args.dump_preds:                                        # per-example (true,pred) for calibration
        @torch.no_grad()
        def _preds(nets):
            model.eval(); ps = []
            for i in range(0, len(nets), args.eval_cnn_bs):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                    ps.append(model(nets[i:i + args.eval_cnn_bs], device=DEV).float().cpu())
            return torch.cat(ps).numpy()
        np.savez(args.dump_preds, val_true=vaY.detach().cpu().numpy(), val_pred=_preds(vaN),
                 test_true=teY.detach().cpu().numpy(), test_pred=_preds(teN))
        print(f"[DUMP] {args.exp_name} wrote per-example preds -> {args.dump_preds}", flush=True)
    fv = ev(vaN, vaY); ff = ev(teN, teY)
    print(f"[FINAL-VAL] {args.exp_name} final_val_tau={fv['tau_b']:.4f} final_val_mse={fv['acc_mse']*1e5:.2f} "
          f"final_val_mae={fv['acc_mae']:.4f}  (salvaged final.pt, cosine)", flush=True)
    json.dump({"exp": args.exp_name, "seed": args.seed, "queries": Q, "target_space": args.target_space,
               "scheduler": args.scheduler, "salvaged_from": args.eval_only_ckpt, "mode": "eval_only",
               "final_val_tau": fv["tau_b"], "final_val_mse": fv["acc_mse"] * 1e5, "final_val_mae": fv["acc_mae"],
               "finalepoch_test_tau": ff["tau_b"], "finalepoch_test_accmse_x1e5": ff["acc_mse"] * 1e5,
               "finalepoch_test_accmae": ff["acc_mae"]},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    raise SystemExit(0)

N = len(trN); step = 0; best = -1.0; t0 = time.time(); bstate = None; start_epoch = 0
if args.resume_training_state:
    rs = torch.load(args.resume_training_state, map_location=DEV)
    model.load_state_dict({k: v.to(DEV) for k, v in rs["model"].items()})
    opt.load_state_dict(rs["optimizer"]); sched.load_state_dict(rs["scheduler"])
    start_epoch = rs["epoch"]; step = rs["global_step"]; best = rs["best_val_tau"]; bstate = rs["best_state"]
    _rng_restore(rs["rng"])                                            # reproduce data order from the resume point
    print(f"[RESUME] {args.exp_name} from {args.resume_training_state}: epoch {start_epoch} step {step} "
          f"best_val_tau {best:.4f} -> continuing to --epochs {args.epochs}", flush=True)
    if start_epoch >= args.epochs:
        print(f"[RESUME] start_epoch {start_epoch} >= --epochs {args.epochs}: nothing to train.", flush=True)
STATE_PATH = os.path.join(args.out_dir, "training_state.pt")
if args.eval_at_init and not args.resume_training_state:
    vm0 = ev(vaN, vaY, nmax=args.val_subset)                          # score the (warm/nested) init at step 0
    best = vm0["tau_b"]
    bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(bstate, os.path.join(args.out_dir, "best.pt"))
    logrow(f"{args.exp_name},0,{opt.param_groups[0]['lr']:.2e},nan,{vm0['tau_b']:.4f},"
           f"{vm0['acc_mse']*1e5:.2f},{vm0['acc_mae']:.4f},nan,nan,nan,{Q},True")
    print(f"[INIT-EVAL] {args.exp_name} step=0 val_tau={vm0['tau_b']:.4f} -> seeded as best-val floor "
          f"(the init stays eligible as the historical best checkpoint).", flush=True)
for ep in range(start_epoch, args.epochs):
    perm = torch.randperm(N)
    for bi in range(0, N, args.batch_size):
        idx = perm[bi:bi + args.batch_size].tolist()
        if args.warmup > 0 and step < args.warmup:
            fr = (step + 1) / args.warmup
            for g, base in zip(opt.param_groups, _base_lrs): g["lr"] = base * fr
        if args.probe_warmup_steps > 0 and len(opt.param_groups) > 1:   # freeze probe_source LR for first W steps
            if step < args.probe_warmup_steps:      opt.param_groups[1]["lr"] = 0.0
            elif step == args.probe_warmup_steps:   opt.param_groups[1]["lr"] = args.probe_lr   # release once
        opt.zero_grad()
        pmask = None
        if args.probe_dropout > 0.0:                                   # per-example probe dropout (train only)
            keep = 1.0 - args.probe_dropout
            pmask = torch.rand(len(idx), model.n_out_probes, device=DEV) < keep    # [B,Q] True=keep
            j = torch.randint(0, model.n_out_probes, (len(idx),), device=DEV)      # force >=1 active per row
            pmask[torch.arange(len(idx), device=DEV), j] = True
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            pred = model([trN[i] for i in idx], device=DEV, probe_mask=pmask).float()
        tb = trTGT[idx]
        loss = F.mse_loss(pred, tb) + (args.rank_loss_w * rank_loss(pred, tb) if args.rank_loss_w > 0 else 0.0)
        loss.backward()
        if args.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); step += 1
        if args.scheduler == "cosine" and step > args.warmup:          # cosine steps every batch post-warmup
            sched.step()
        if step % args.eval_every == 0:
            vm = ev(vaN, vaY, nmax=args.val_subset)
            if args.scheduler == "plateau" and step >= args.warmup:
                _lr_before = opt.param_groups[0]["lr"]
                sched.step(vm["tau_b"])
                _lr_after = opt.param_groups[0]["lr"]
                if _lr_after != _lr_before:
                    print(f"[PLATEAU] exp={args.exp_name} step={step} val_tau={vm['tau_b']:.4f} "
                          f"LR {_lr_before:.2e} -> {_lr_after:.2e} (readout group)", flush=True)
            isb = vm["tau_b"] > best
            tm = ev(teN, teY, nmax=args.val_subset) if isb else {"tau_b": float("nan"), "acc_mse": float("nan"), "acc_mae": float("nan")}
            if isb:
                best = vm["tau_b"]
                bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(bstate, os.path.join(args.out_dir, "best.pt"))
            logrow(f"{args.exp_name},{step},{opt.param_groups[0]['lr']:.2e},{loss.item():.4f},"
                   f"{vm['tau_b']:.4f},{vm['acc_mse']*1e5:.2f},{vm['acc_mae']:.4f},"
                   f"{tm['tau_b']:.4f},{tm['acc_mse']*1e5:.2f},{tm['acc_mae']:.4f},{Q},{isb}")
            print(f"[pgh] step={step} ({step/(time.time()-t0):.1f}/s) loss={loss.item():.3f} "
                  f"val_tau={vm['tau_b']:.4f} val_accMSE={vm['acc_mse']*1e5:.2f} best={best:.4f} Q={Q}", flush=True)
    # end of epoch: save resume-safe training state (RNG captured HERE = just before next epoch's perm)
    save_training_state(STATE_PATH, model, opt, sched, ep + 1, step, best, bstate)

# ---- checkpoint policy (§8): save FINAL-epoch ckpt + report both final-epoch and best-val test ----
torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
           os.path.join(args.out_dir, "final.pt"))
fv = ev(vaN, vaY)                                            # ALWAYS eval final-epoch VAL (cheap, uniform metric)
print(f"[FINAL-VAL] {args.exp_name} final_val_tau={fv['tau_b']:.4f} final_val_mse={fv['acc_mse']*1e5:.2f} "
      f"final_val_mae={fv['acc_mae']:.4f}", flush=True)
if bstate is not None:                                       # load best-val checkpoint (== best.pt on disk)
    model.load_state_dict({k: v.to(DEV) for k, v in bstate.items()})
ff = ft = None; diag = {}
if args.skip_test_eval:                                      # TUNING MODE: no test eval, no diagnostics, no dump
    print(f"[FINAL] {args.exp_name} best_val_tau={best:.4f} | TEST SKIPPED (tuning_mode) "
          f"| queries={Q} params={rep['total_trainable']:,}", flush=True)
    args.dump_preds = ""                                     # ensure no test preds are written in tuning mode
else:
    ff = ev(teN, teY)                                        # final-epoch checkpoint, full test
    ft = ev(teN, teY)                                        # best-val checkpoint, full test (reported)
    print(f"[FINAL-EPOCH] {args.exp_name} TEST tau={ff['tau_b']:.4f} accMSE={ff['acc_mse']*1e5:.2f} accMAE={ff['acc_mae']:.4f}", flush=True)
    print(f"[FINAL] {args.exp_name} best_val_tau={best:.4f} | TEST tau={ft['tau_b']:.4f} "
          f"accMSE={ft['acc_mse']*1e5:.2f} accMSEx1e5={ft['acc_mse']*1e5:.2f} accMAE={ft['acc_mae']:.4f} "
          f"| queries={Q} params={rep['total_trainable']:,}", flush=True)
    diag = {"tau_full": ft["tau_b"], "mse_full": ft["acc_mse"]}
    if args.hidden_mode == "on":
        z = ev(teN, teY, zero_hidden=True); s = ev(teN, teY, shuffle_hidden=True)
        diag.update(tau_hidden_zero=z["tau_b"], tau_hidden_shuffle=s["tau_b"],
                    hidden_zero_drop=ft["tau_b"] - z["tau_b"], hidden_shuffle_drop=ft["tau_b"] - s["tau_b"])
        print(f"[DIAG] {args.exp_name} tau_full={ft['tau_b']:.4f} tau_hidden_zero={z['tau_b']:.4f} "
              f"tau_hidden_shuffle={s['tau_b']:.4f} | zero_drop={diag['hidden_zero_drop']:+.4f} "
              f"shuffle_drop={diag['hidden_shuffle_drop']:+.4f} (positive => hidden IS used)", flush=True)
if args.dump_preds:                                          # Stage-3: per-example preds + positive-affine cal
    @torch.no_grad()
    def _pe(nets):
        model.eval(); ps = []
        for i in range(0, len(nets), args.eval_cnn_bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                ps.append(model(nets[i:i + args.eval_cnn_bs], device=DEV).float().cpu())
        return torch.cat(ps).numpy()
    _vp, _tp = _pe(vaN), _pe(teN); _vt = vaY.detach().cpu().numpy(); _tt = teY.detach().cpu().numpy()
    _vv = _vp.var(); _a = max(float(np.cov(_vp, _vt, bias=True)[0, 1] / _vv) if _vv > 0 else 1.0, 1e-6)
    _b = float(_vt.mean() - _a * _vp.mean())                  # fit on VALIDATION ONLY
    np.savez(args.dump_preds, val_true=_vt, val_pred=_vp, test_true=_tt, test_pred=_tp, affine_a=_a, affine_b=_b)
    diag.update(cal_affine_a=_a, cal_affine_b=_b,
                cal_val_mse=float(np.mean((_a * _vp + _b - _vt) ** 2)) * 1e5, cal_val_mae=float(np.mean(np.abs(_a * _vp + _b - _vt))),
                cal_test_mse=float(np.mean((_a * _tp + _b - _tt) ** 2)) * 1e5, cal_test_mae=float(np.mean(np.abs(_a * _tp + _b - _tt))))
    print(f"[CAL] {args.exp_name} a={_a:.3f} b={_b:.4f} val_mse {fv['acc_mse']*1e5:.1f}->{diag['cal_val_mse']:.1f} "
          f"test_mse {ft['acc_mse']*1e5:.1f}->{diag['cal_test_mse']:.1f}", flush=True)
# ---- validation trajectory (step, val_tau, lr) read back from log.csv for summary.json ----
_vtraj = []
try:
    import csv as _csv
    for _r in _csv.DictReader(open(LOG)):
        try: _vtraj.append([int(float(_r["step"])), float(_r["val_tau"]), float(_r["lr"])])
        except Exception: pass
except Exception:
    pass
_summ = {"exp": args.exp_name, "seed": args.seed, "queries": Q, "params": rep["total_trainable"],
         "target_space": args.target_space, "scheduler": args.scheduler,
         "lr": args.lr, "probe_lr": args.probe_lr, "plateau_factor": args.plateau_factor,
         "plateau_patience": args.plateau_patience, "plateau_min_lr": args.plateau_min_lr,
         "skip_test_eval": bool(args.skip_test_eval),
         "best_val_tau": best,
         "final_val_tau": fv["tau_b"], "final_val_mse": fv["acc_mse"] * 1e5, "final_val_mae": fv["acc_mae"],
         "val_trajectory": _vtraj}
if ft is not None:                                           # full-eval mode only: test fields present
    _summ.update({"final_test_tau": ft["tau_b"], "final_test_acc_mse": ft["acc_mse"],
                  "final_test_accmse_x1e5": ft["acc_mse"] * 1e5, "final_test_accmae": ft["acc_mae"],
                  "finalepoch_test_tau": ff["tau_b"], "finalepoch_test_acc_mse": ff["acc_mse"],
                  "finalepoch_test_accmse_x1e5": ff["acc_mse"] * 1e5, "finalepoch_test_accmae": ff["acc_mae"]})
_summ.update(diag)
json.dump(_summ, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
