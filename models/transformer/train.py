"""Transformer-probe trainer/CLI (spec §10-12): smoke | train | predict | evaluate | count.
Invoked through the repository dispatcher:  python main.py transformer {train,count,cache,smoke,manifest,verify} ...

Real SmallZoo checkpoint zoos are ABSENT on this cluster -> a synthetic FixtureZoo drives plumbing/smoke
(labels are a learnable function of teacher weights, surfaced through responses). The real manifest schema
is documented in build_manifest(); benchmark numbers stay UNVERIFIED until the zoos + AGNews w2v are staged.

Loss = standardized-accuracy MSE (train-only mean/std, persisted, inverse before metrics). AdamW groups:
gen/codes 1e-3 wd0 ; predictor+tokenizer 3e-4 wd1e-3.
Microbatch accumulation to effective batch 32 (loss-sum / effective_batch); grad-clip 1 once/update.
"""
import argparse, json, math, os, sys, time, hashlib, subprocess
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root (this file lives in models/transformer/)
from models.transformer import teacher as T
from models.transformer import intake as IN
from models.transformer.system import LearnedSystem, TPConfig, PARAM_CEILING, fit_ffn
from models.transformer.readout_channeltraj import CT_FFN                # channel_trajectory v1: fixed FFN (no auto-fit)
from models.transformer.acquire import predict
from models.logging_utils import print_run_config, print_eval, print_seed_result
try:
    from scipy.stats import kendalltau, spearmanr
except Exception:
    kendalltau = spearmanr = None


# ---------------- fixture zoo (synthetic; NOT released checkpoints) ----------------
def build_fixture_zoo(dataset, C, n, seed=0):
    """n fixture teachers + labels in [0,1]. Label q scales the teacher's logits, so responses carry it
    (learnable smoke signal). Frozen tensors (requires_grad=False)."""
    g = torch.Generator().manual_seed(1000 + seed)
    zoo = []
    for i in range(n):
        q = torch.rand(1, generator=g).item()
        p = T.make_fixture_mnist(n_classes=C, seed=10_000 + seed * 100 + i, dtype=torch.float32)
        p["classifier.fc2.weight"] = p["classifier.fc2.weight"] * (0.4 + 1.2 * q)   # logit scale ~ q
        for v in p.values():
            v.requires_grad_(False)
        zoo.append({"id": f"{dataset}_fix_{seed}_{i}", "params": p, "label": q})
    return zoo


# ---------------- normalization + optimizer groups ----------------
class Norm:
    def __init__(self, ys):
        t = torch.tensor(ys); self.mean = t.mean().item()
        self.std = t.std(unbiased=False).item() or 1e-8
    def fwd(self, y):  # to standardized
        return (y - self.mean) / (self.std + 1e-8)
    def inv(self, z):  # back to accuracy
        return z * (self.std + 1e-8) + self.mean
    def state(self):
        return {"mean": self.mean, "std": self.std}


def param_groups(system, gen_lr=1e-3, pred_lr=3e-4, weight_decay=1e-3):
    """ProbeGen-style split: probe generator (gen_lr, wd0) vs decoder/predictor (pred_lr, wd)."""
    gen, pred = [], []
    for name, p in system.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("generator."):
            gen.append(p)
        else:                                    # tokenizer + predictor / readout
            pred.append(p)
    return [{"params": gen, "lr": gen_lr, "weight_decay": 0.0},
            {"params": pred, "lr": pred_lr, "weight_decay": weight_decay}]


def build_scheduler(opt, kind, max_updates, patience=5, factor=0.5, warmup=0):
    """kind: none | cosine (T_max=max_updates, optional linear warmup) | plateau (ReduceLROnPlateau on val_tau)."""
    if kind == "cosine":
        if warmup > 0:
            def lam(step):
                if step < warmup:
                    return float(step + 1) / warmup
                prog = (step - warmup) / max(1, max_updates - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return torch.optim.lr_scheduler.LambdaLR(opt, lam), "step"
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_updates), "step"
    if kind == "plateau":
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=factor, patience=patience), "eval")
    return None, "none"


# ---------------- train / eval core ----------------
def kendall(pred, true):
    if kendalltau is None:
        return float("nan")
    return kendalltau(pred, true).correlation


def run_eval(system, zoo, norm, micro=8):
    system.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for i in range(0, len(zoo), micro):
            chunk = zoo[i:i + micro]
            out = predict(system, [z["params"] for z in chunk])
            preds += [norm.inv(v.item()) for v in out]
            trues += [z["label"] for z in chunk]; ids += [z["id"] for z in chunk]
    tau = kendall(preds, trues)
    mae = sum(abs(p - t) for p, t in zip(preds, trues)) / len(preds)
    return {"kendall_tau": tau, "mae": mae, "n": len(preds)}, (ids, preds, trues)


def _snapshot_params(system):
    return {n: p.detach().clone() for n, p in system.named_parameters()}


def _load_params(system, shadow):
    with torch.no_grad():
        for n, p in system.named_parameters():
            p.copy_(shadow[n])                            # in-place: preserves tensor identity -> optimizer state valid


# ---------------- reproducibility state (additive; all default-off so legacy runs are byte-identical) ----------------
_COMPAT_KEYS = ("dataset", "generator", "n_classes", "n_probes", "readout", "readout_arch", "pma_seeds",
                "stats_bypass", "token_mlp", "moments", "signed_mix",
                "dropout", "n_slots", "seed")


def _check_resume_compat(saved: dict, now: dict):
    """Refuse to resume a training_state.pt whose ARCHITECTURE / data population differs (e.g. an r2tm state
    into a channel_trajectory run). Compares the config keys that define the model, the resolved FFN, the
    architecture version and the train/val manifest hash."""
    sc, nc = saved.get("cfg", {}), now.get("cfg", {})
    diffs = [f"{k}: saved={sc.get(k)!r} now={nc.get(k)!r}" for k in _COMPAT_KEYS if sc.get(k) != nc.get(k)]
    for k in ("resolved_ffn", "arch_version", "manifest_hash"):
        if saved.get(k) != now.get(k):
            diffs.append(f"{k}: saved={saved.get(k)!r} now={now.get(k)!r}")
    if diffs:
        raise SystemExit("[resume] INCOMPATIBLE training_state — refusing to resume:\n  " + "\n  ".join(diffs))


def _split_hash(tr, va) -> str:
    """Hash of the exact train/val population (ids, after C-filter / cut_off) — the 'manifest hash' saved with runs."""
    h = hashlib.sha256()
    for z in sorted(tr, key=lambda z: z["id"]):
        h.update(f"tr:{z['id']}\n".encode())
    for z in sorted(va, key=lambda z: z["id"]):
        h.update(f"va:{z['id']}\n".encode())
    return h.hexdigest()[:16]


def _code_rev() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return "unknown"


def train_loop(system, train_zoo, val_zoo, max_updates, eval_every, micro=8, eff=32, clip=1.0, log=print,
               gen_lr=1e-3, pred_lr=3e-4, scheduler="none", patience=5, factor=0.5,
               weight_decay=1e-3, warmup=0, seed=0, ema_decay=0.0,
               exact_accum=False, state_every=0, state_path=None, resume=None, meta=None, profile=False,
               stop_after=0):
    """exact_accum: rescale accumulated grads by eff/count when a window holds count != eff models (exact mean over
    the models actually accumulated; partial epoch tails are otherwise over-weighted). state_every/state_path:
    write a resumable training_state.pt every N optimizer updates (model+opt+sched+RNG+data order+best+norm+meta).
    resume: a loaded training_state dict -> exact continuation from a step boundary. stop_after: stop after this
    many updates while KEEPING max_updates as the schedule horizon (time-boxed run, resumed later). All default-off."""
    norm = Norm([z["label"] for z in train_zoo])
    train_probe = train_zoo[:min(256, len(train_zoo))]        # fixed train subset for a train-vs-val gap curve
    groups = param_groups(system, gen_lr=gen_lr, pred_lr=pred_lr, weight_decay=weight_decay)
    opt = torch.optim.AdamW(groups)
    sched, sched_when = build_scheduler(opt, scheduler, max_updates, patience=patience, factor=factor, warmup=warmup)
    all_params = [p for gp in groups for p in gp["params"]]
    upd, step_in_acc = 0, 0
    g = torch.Generator().manual_seed(seed)              # data order varies by training seed (split fixed elsewhere)
    best_tau, best_state, best_which, hist = -2.0, None, "online", []
    best_step, best_epoch = 0, 0
    epoch_num = 1
    train_started = time.time()
    # §11 full-system EMA: coherent single-trajectory weight average of ALL learned params (generator,
    # codes, embeddings, readout/predictor, heads). Teacher is external (not in system) so
    # never averaged; buffers are left as the online system's (copied, not averaged). Shadow = training overhead.
    ema = _snapshot_params(system) if (ema_decay and ema_decay > 0) else None
    opt.zero_grad()
    resume_perm, resume_pos = None, 0
    if resume is not None:                                   # exact continuation: state was saved at a step boundary
        dev_ = next(system.parameters()).device
        system.load_state_dict(resume["model"]); opt.load_state_dict(resume["optimizer"])
        if sched is not None and resume.get("scheduler") is not None:
            sched.load_state_dict(resume["scheduler"])
        g.set_state(resume["data_rng"]); torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        upd = int(resume["upd"]); best_tau = resume["best_tau"]; best_which = resume["best_which"]
        best_state = resume["best_state"]; hist = list(resume["hist"])
        best_step = int(resume.get("best_step", upd))
        best_epoch = int(resume.get("best_epoch", max(1, int((upd * eff) // max(1, len(train_zoo))))))
        epoch_num = int(resume.get("epoch_num", max(1, best_epoch)))
        norm.mean, norm.std = resume["norm"]["mean"], resume["norm"]["std"]       # train-only stats, as saved
        if ema is not None and resume.get("ema") is not None:
            ema = {n: v.to(dev_) for n, v in resume["ema"].items()}
        resume_perm, resume_pos = resume.get("epoch_perm"), int(resume.get("pos", 0))
        log(f"  [resume] continuing at upd {upd} (best val_tau {best_tau:.4f}) from {resume.get('path', '?')}")

    def _save_state(perm_, pos_):
        if not state_path:
            return
        st = {"upd": upd, "model": system.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict() if sched is not None else None,
              "data_rng": g.get_state(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
              "epoch_perm": perm_, "pos": pos_, "best_tau": best_tau, "best_which": best_which,
              "best_state": best_state, "best_step": best_step, "best_epoch": best_epoch,
              "epoch_num": epoch_num, "hist": hist, "norm": norm.state(),
              "ema": ({n: v.detach().cpu() for n, v in ema.items()} if ema is not None else None),
              "scaler": None,                                # no AMP/GradScaler in this trainer (fp32 policy)
              "meta": meta}
        tmp = state_path + ".tmp"; torch.save(st, tmp); os.replace(tmp, state_path)   # atomic replace

    step_times = []
    stopped = lambda: bool(stop_after) and upd >= stop_after
    while upd < max_updates and not stopped():
        if resume_perm is not None:                          # finish the interrupted epoch from the saved position
            perm, start = list(resume_perm), resume_pos; resume_perm = None
        else:
            perm, start = torch.randperm(len(train_zoo), generator=g).tolist(), 0
        for i in range(start, len(perm), micro):
            idx = perm[i:i + micro]
            chunk = [train_zoo[j] for j in idx]
            if profile and step_in_acc == 0:                 # start of an update window (ALL its microbatches)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_win = time.time()
            system.train()
            out = predict(system, [z["params"] for z in chunk])
            y = torch.tensor([z["label"] for z in chunk])
            z = torch.tensor([norm.fwd(v) for v in y.tolist()], device=out.device)
            loss = ((out - z) ** 2).sum() / eff             # separable: sum / effective batch
            loss.backward(); step_in_acc += len(idx)
            if step_in_acc >= eff:
                if exact_accum and step_in_acc != eff:      # exact mean over the models actually in this window
                    scale = eff / step_in_acc
                    for p in all_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(all_params, clip)
                opt.step(); opt.zero_grad(); step_in_acc = 0; upd += 1
                if profile:                                  # full update = fwd+bwd of every microbatch + clip + step
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    step_times.append(time.time() - t_win)
                    if upd == 1 or upd % 10 == 0:
                        log(f"  [profile] upd {upd} loss {loss.item() * eff / max(1, len(idx)):.4f}")
                if ema is not None:                          # update ONCE per successful optimizer step (not per microbatch)
                    with torch.no_grad():
                        for n, p in system.named_parameters():
                            ema[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
                if sched_when == "step":
                    sched.step()
                if upd % eval_every == 0 or upd == max_updates:
                    m, _ = run_eval(system, val_zoo, norm, micro=max(32, micro))         # ONLINE val
                    mt, _ = run_eval(system, train_probe, norm, micro=max(32, micro))    # train-subset (gap curve)
                    hist.append((upd, m["kendall_tau"], m["mae"], mt["kendall_tau"], mt["mae"]))
                    otau = m["kendall_tau"] if m["kendall_tau"] == m["kendall_tau"] else -2
                    etau = None
                    if ema is not None:                      # EMA val via coherent snapshot swap (EMA gen->teacher->EMA pred)
                        online_bak = _snapshot_params(system); _load_params(system, ema)
                        me, _ = run_eval(system, val_zoo, norm, micro=max(32, micro))
                        _load_params(system, online_bak)     # restore online params (buffers untouched)
                        etau = me["kendall_tau"] if me["kendall_tau"] == me["kendall_tau"] else -2
                    is_best = False
                    if otau > best_tau:                       # BEST-val over {online, EMA} -> one coherent deployed system
                        best_tau, best_which = otau, "online"
                        best_step, best_epoch = upd, epoch_num
                        is_best = True
                        best_state = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}
                    if etau is not None and etau > best_tau:
                        best_tau, best_which = etau, "ema"
                        best_step, best_epoch = upd, epoch_num
                        is_best = True
                        sd = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}
                        for n in ema:
                            sd[n] = ema[n].detach().cpu().clone()          # EMA params + online buffers
                        best_state = sd
                    if sched_when == "eval":
                        sched.step(otau)                     # scheduler follows the ONLINE training metric
                    elapsed = time.time() - train_started
                    remaining = (elapsed / max(upd, 1)) * max(0, max_updates - upd)
                    train_loss = loss.item() * eff / max(1, len(idx))
                    print_eval(
                        task="regression", step=upd, epoch=epoch_num,
                        train_loss=train_loss, val_value=m["kendall_tau"], test_value=None,
                        elapsed=elapsed, remaining=remaining, new_best=is_best,
                    )
                if state_every and (upd % state_every == 0 or upd == max_updates or stopped()):
                    _save_state(perm, i + micro)             # next microbatch index of this epoch (exact resume)
                if upd >= max_updates or stopped():
                    break
        epoch_num += 1
    if profile and step_times:
        warm = step_times[5:] if len(step_times) > 5 else step_times
        msg = (f"  [profile] {len(step_times)} updates | update time mean {sum(warm)/len(warm)*1e3:.1f} ms "
               f"(fwd+bwd+step, excl. 5 warm-up) | est. 40k updates {sum(warm)/len(warm)*40000/3600:.2f} h")
        if torch.cuda.is_available():
            msg += f" | peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        log(msg)
    if best_state is None:                                   # no eval happened (tiny smoke) -> fall back to final
        best_state = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}
    if ema is not None:
        log(f"  [select] best val_tau {best_tau:.4f} via {best_which}{'  (EMA won)' if best_which == 'ema' else ''}")
    return norm, hist, best_tau, best_state, best_step, best_epoch


# ---------------- CLI actions ----------------
def _mkcfg(a):
    if a.readout_arch == "channel_trajectory" and a.dataset == "mnist":   # spec: mnist FFN fixed at 384 — never auto-fit
        if a.ffn not in (0, CT_FFN):
            print(f"[cfg] channel_trajectory/mnist ignores --ffn {a.ffn}: FFN is fixed at {CT_FFN}")
        ffn = CT_FFN
    else:                                                    # (channel_trajectory/agnews: --ffn 384 or 0 = fit to cap)
        ffn = a.ffn if a.ffn > 0 else fit_ffn(a.dataset, a.generator, a.n_classes, a.n_probes,
                                              a.readout, a.pma_seeds,
                                              readout_arch=a.readout_arch, stats_bypass=a.stats_bypass,
                                              token_mlp=a.token_mlp, moments=a.moments, signed_mix=a.signed_mix)
    return TPConfig(a.dataset, a.generator, a.n_classes, ffn, n_probes=a.n_probes,
                    readout=a.readout, pma_seeds=a.pma_seeds, seed=a.seed,
                    readout_arch=a.readout_arch, stats_bypass=a.stats_bypass, dropout=a.dropout, n_slots=a.n_slots,
                    token_mlp=a.token_mlp, moments=a.moments, signed_mix=a.signed_mix)


def cmd_count(a):
    from models.transformer.system import verify_config
    verify_config(_mkcfg(a))


def cmd_smoke(a):
    assert a.n_fixtures <= 16 and a.max_updates <= 100, "smoke caps: <=16 fixtures, <=100 updates (spec §0)"
    cfg = _mkcfg(a)
    sys_ = LearnedSystem(cfg)
    assert sys_.n_trainable() <= PARAM_CEILING
    sys_.to("cuda" if torch.cuda.is_available() else "cpu")
    tr = build_fixture_zoo(a.dataset, a.n_classes, a.n_fixtures, seed=0)
    print(f"[smoke] {cfg.key} params={sys_.n_trainable():,}  overfit set={len(tr)}  (fixtures; UNVERIFIED benchmark)")
    norm0 = Norm([z["label"] for z in tr])
    pre, _ = run_eval(sys_, tr, norm0, micro=4)              # random-init baseline on the SAME set
    t0 = time.time()
    _, hist, best, _ = train_loop(sys_, tr, tr, a.max_updates, eval_every=max(1, a.max_updates // 5), micro=4, eff=8,
                                  ema_decay=(a.ema_decay if a.ema else 0.0))    # smoke --ema exercises the §11 path
    dt = time.time() - t0
    print(f"[smoke] {dt:.1f}s  TRAIN-set tau {pre['kendall_tau']:.3f} (init) -> {best:.3f} (best)  "
          f"mae {pre['mae']:.3f} -> {hist[-1][2]:.3f}  [clear progress => pipeline optimizes; NOT benchmark evidence]")
    assert best > pre["kendall_tau"] + 0.15, "smoke did not show clear optimization progress"


def _filter_C(zoo, C):
    """Keep only targets whose classifier output dim == C (logit tokenizer is fixed-C; per-C bucketing)."""
    keep = [z for z in zoo if T.classifier_out_dim(z["params"]) == C]
    return keep, len(zoo) - len(keep)


def cmd_manifest(a):
    man = IN.build_manifest(a.data_root, a.dataset, seed=a.seed, cut_off=a.cut_off)
    assert man, f"no epoch-75 checkpoints under {a.data_root} for {a.dataset} at cut_off={a.cut_off}"
    tag = IN._cut_tag(a.cut_off)
    out = a.out or os.path.join(a.data_root, f"manifest_{a.dataset}_{tag}_s{a.seed}.json")
    h = IN.write_manifest(man, out, overwrite=a.overwrite)
    print(f"[manifest] {a.dataset}: {len(man)} epoch-75 runs  hash={h}")
    for s, st in IN.manifest_stats(man).items():
        print(f"    {s:5s} n={st['n']:6d}  acc mean={st['acc_mean']} [{st['acc_min']},{st['acc_max']}]")
    print(f"    written -> {out}")


def cmd_cache(a):
    assert a.data_root and os.path.isdir(a.data_root), f"--data_root missing: {a.data_root!r}"
    t0 = time.time()
    counts = IN.build_cache(
        a.data_root, a.dataset, seed=a.seed, cut_off=a.cut_off, overwrite=a.overwrite
    )
    print(
        f"[cache] {a.dataset} cut_off={a.cut_off} seed={a.seed} "
        f"built in {time.time()-t0:.1f}s -> {counts}"
    )
    for split in ("train", "val", "test"):
        print(
            f"    {split:5s} -> "
            f"{IN.cache_path(a.data_root, a.dataset, split, a.seed, cut_off=a.cut_off)}"
        )


def cmd_verify(a):
    # released-checkpoint parity: functional teacher core vs independent reference, on REAL checkpoints.
    man = IN.build_manifest(a.data_root, a.dataset, seed=0)
    import random as _r; _r.Random(0).shuffle(man)
    n = min(a.n, len(man)); maxerr = 0.0; cdist = {}
    for r in man[:n]:
        p = IN.load_target(r["checkpoint_path"])
        C = T.classifier_out_dim(p); cdist[C] = cdist.get(C, 0) + 1
        ref = T.SmallZooTransformer(n_classes=C).double().load_teacher({k: v.double() for k, v in p.items()})
        if a.dataset == "mnist":
            img = torch.randn(3, 1, 28, 28, dtype=torch.float64)
            fo = T.full_vision({k: v.double() for k, v in p.items()}, img); ro = ref(img)
        else:
            x = torch.randn(3, 17, 32, dtype=torch.float64)
            fo = T.encoder_route({k: v.double() for k, v in p.items()}, x); ro = ref.encoder_route(x)
        for ch in ("block1", "block2", "final_norm", "logits"):
            maxerr = max(maxerr, (fo[ch] - ro[ch]).abs().max().item())
    print(f"[verify] {a.dataset}: {n} real checkpoints | functional-vs-reference max err = {maxerr:.2e} "
          f"({'PASS' if maxerr < 1e-6 else 'FAIL'}) | C distribution = {cdist}")


def cmd_train(a):
    cfg = _mkcfg(a)
    torch.manual_seed(a.seed)                          # vary model init per training seed (split stays FIXED below)
    sys_ = LearnedSystem(cfg)
    assert sys_.n_trainable() <= PARAM_CEILING, f"over {PARAM_CEILING:,} param ceiling"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys_.to(dev)
    assert a.data_root and os.path.isdir(a.data_root), f"--data_root missing: {a.data_root!r}"
    # SPLIT IS FIXED (seed 0) for every training seed — else multi-seed runs get shifted partitions that leak
    # into the seed-0 test set used at eval. Training seed only varies model init + data order, never the split.
    tr = IN.load_zoo_cached(
        a.data_root, a.dataset, "train", seed=0, cut_off=a.cut_off, limit=a.limit
    )
    va = IN.load_zoo_cached(
        a.data_root, a.dataset, "val", seed=0, cut_off=a.cut_off, limit=a.limit
    )
    if tr is None or va is None:                       # no cache -> slow per-file path
        print(
            "[train] WARNING: no threshold-specific consolidated cache found -> "
            "per-file load (slow). Run `python main.py transformer cache ... --cut_off ...` first."
        )
        man = IN.build_manifest(a.data_root, a.dataset, seed=0, cut_off=a.cut_off)
        tr = IN.load_zoo(man, "train", limit=a.limit)
        va = IN.load_zoo(man, "val", limit=a.limit)
    tr, dtr = _filter_C(tr, cfg.n_classes)
    va, dva = _filter_C(va, cfg.n_classes)
    print(
        f"[train] threshold-specific split cut_off={a.cut_off}: "
        f"train={len(tr)} val={len(va)}"
    )
    print(f"[train] {cfg.key} params={sys_.n_trainable():,} | train={len(tr)}(-{dtr} offC) val={len(va)}(-{dva} offC)")
    assert tr and va, "empty split after C-filter — check --n_classes vs the zoo's classifier dim"
    runs = a.runs_dir or f"checkpoints/tp_{a.dataset}_{a.generator}_s{a.seed}"
    os.makedirs(runs, exist_ok=True)
    eff = max(32, a.micro)                                  # eff-batch >= micro so loss mean-scaling stays correct
    # reproducibility metadata (saved in last.pt and training_state.pt; also the resume-compatibility key)
    meta = {"cfg": vars(a), "resolved_ffn": cfg.ffn, "eff_batch": eff, "arch_version": sys_.arch_version,
            "code_rev": _code_rev(), "manifest_hash": _split_hash(tr, va), "split_seed": 0,
            "param_total": sys_.n_trainable(), "components": sys_.component_counts()}
    print(f"[train] arch_version={meta['arch_version']} code_rev={meta['code_rev']} manifest_hash={meta['manifest_hash']} "
          f"eff_batch={eff} exact_accum={a.exact_accum}")
    resume = None
    if a.resume_state:
        resume = torch.load(a.resume_state, map_location="cpu", weights_only=False)
        _check_resume_compat(resume.get("meta") or {}, meta); resume["path"] = a.resume_state
    state_path = os.path.join(runs, "training_state.pt") if a.save_state_every > 0 else None
    norm, hist, best, best_state = train_loop(sys_, tr, va, a.max_updates, eval_every=a.eval_every, micro=a.micro,
                                  eff=eff, gen_lr=a.gen_lr, pred_lr=a.pred_lr,
                                  scheduler=a.scheduler, patience=a.plateau_patience, factor=a.plateau_factor,
                                  weight_decay=a.weight_decay, warmup=a.warmup, seed=a.seed,
                                  ema_decay=(a.ema_decay if a.ema else 0.0),
                                  exact_accum=a.exact_accum, state_every=a.save_state_every, state_path=state_path,
                                  resume=resume, meta=meta, profile=a.profile, stop_after=a.stop_after)
    torch.save({"state_dict": best_state, "cfg": vars(a), "norm": norm.state(),   # BEST-val checkpoint, not final
                "best_val_tau": best, "history": hist,
                "arch_version": meta["arch_version"], "code_rev": meta["code_rev"],
                "manifest_hash": meta["manifest_hash"],
                "resolved": {"ffn": cfg.ffn, "eff_batch": eff, "exact_accum": a.exact_accum,
                             "param_total": meta["param_total"], "components": meta["components"]}},
               os.path.join(runs, "last.pt"))
    print(f"[train] done | best val_tau={best:.4f} | saved {runs}/last.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "smoke", "train", "manifest", "verify", "cache"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", choices=["mnist", "agnews"], required=True)
        s.add_argument("--seed", type=int, default=0)
        if name in ("count", "smoke", "train"):
            s.add_argument("--generator", choices=["g3", "glin"], required=True,
                           help="shared unconditioned probe generator: g3 (nonlinear) | glin (deep-linear)")
            s.add_argument("--n_classes", type=int, default=10)
            s.add_argument("--ffn", type=int, default=944, help="predictor FFN width; <=0 -> auto-fit to <=1.9M")
            s.add_argument("--n_probes", type=int, default=64, help="Q (phase-2: up to 128)")
            s.add_argument("--readout", choices=["cls", "multi", "pma"], default="cls")
            s.add_argument("--pma_seeds", type=int, default=1)
            s.add_argument("--readout_arch", choices=["r0", "r2", "r3", "rout", "channel_trajectory"], default="r0",
                           help="r0=tokenizer+predictor; r2=cross-layer fusion; r3=full-memory latent; rout=output/logits-only "
                                "baseline; channel_trajectory=v1 channel-first trajectory readout (mnist / agnews, shared g3 probes, Q256 only, mnist FFN fixed 384)")
            s.add_argument("--stats_bypass", action="store_true", help="add the §4 response-stats bypass")
            s.add_argument("--dropout", type=float, default=0.1, help="predictor/readout transformer dropout")
            s.add_argument("--n_slots", type=int, default=16, help="r3 latent slots")
            s.add_argument("--token_mlp", action="store_true", help="§6 r2 pre-pool nonlinear feature map")
            s.add_argument("--moments", action="store_true", help="§7 r2 per-probe moments into the logit token")
            s.add_argument("--signed_mix", action="store_true", help="§8 r2 signed token mixing (A[4,n]/bucket, gated)")
            s.add_argument("--ema", action="store_true", help="§11 full-system EMA (coherent single-trajectory weight avg)")
            s.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (default 0.999)")
        if name == "smoke":
            s.add_argument("--n_fixtures", type=int, default=12); s.add_argument("--max_updates", type=int, default=60)
        if name in ("manifest", "verify", "train", "cache"):
            s.add_argument("--data_root", required=True)
        if name in ("manifest", "cache"):
            s.add_argument(
                "--cut_off",
                type=float,
                default=0.0,
                help="Absolute accuracy threshold applied BEFORE the 70/15/15 split.",
            )
        if name == "cache":
            s.add_argument("--overwrite", action="store_true")
        if name == "manifest":
            s.add_argument("--out", default=None); s.add_argument("--overwrite", action="store_true")
        if name == "verify":
            s.add_argument("--n", type=int, default=20)
        if name == "train":
            s.add_argument("--runs_dir", default=None); s.add_argument("--limit", type=int, default=0)
            s.add_argument("--max_updates", type=int, default=20000); s.add_argument("--eval_every", type=int, default=500)
            s.add_argument("--micro", type=int, default=8)
            # ProbeGen-style optimizer split (defaults = the frozen phase-2 values) + scheduler
            s.add_argument("--gen_lr", type=float, default=1e-3, help="probe-generator lr (ProbeGen probe_lr)")
            s.add_argument("--pred_lr", type=float, default=3e-4, help="decoder/predictor lr (ProbeGen lr)")
            s.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
            s.add_argument("--plateau_patience", type=int, default=5)
            s.add_argument("--plateau_factor", type=float, default=0.5)
            s.add_argument("--weight_decay", type=float, default=1e-3, help="AdamW wd for predictor/readout (probe codes + generator use wd 0)")
            s.add_argument("--warmup", type=int, default=0, help="linear warmup updates before cosine")
            s.add_argument(
                "--cut_off",
                type=float,
                default=0.0,
                help="Absolute accuracy threshold. Population is filtered before the fixed-seed 70/15/15 split.",
            )
            # additive reproducibility / accumulation knobs (all default-off => legacy behaviour unchanged)
            s.add_argument("--exact_accum", action="store_true",
                           help="rescale accumulated grads by eff/count when a window holds != eff models (exact mean; partial tails)")
            s.add_argument("--save_state_every", type=int, default=0,
                           help="write <runs_dir>/training_state.pt every N optimizer updates (0 = off)")
            s.add_argument("--resume_state", type=str, default="",
                           help="exact continuation from a training_state.pt (architecture/data must match; else refused)")
            s.add_argument("--profile", action="store_true", help="synchronized per-update step time + peak GPU memory")
            s.add_argument("--stop_after", type=int, default=0,
                           help="stop after N updates but keep --max_updates as the cosine horizon (time-boxed; resume later)")
    a = ap.parse_args()
    {"count": cmd_count, "smoke": cmd_smoke, "train": cmd_train,
     "manifest": cmd_manifest, "verify": cmd_verify, "cache": cmd_cache}[a.cmd](a)


if __name__ == "__main__":
    main()
