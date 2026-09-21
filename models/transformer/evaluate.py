#!/usr/bin/env python3
# TEST evaluation of finished runs. Given the run dir(s) of ONE config (one per seed), load each trained model,
# predict on the held-out TEST split, and report the per-seed test Kendall tau (the reported single-model
# protocol; runs trained with --cut_off are scored on the matching acc>=cut_off test subset). A seed-averaged
# prediction ensemble is printed as an auxiliary line only.
#   python -m models.transformer.evaluate "checkpoints/<run_dir_prefix>*_s*"
import sys, os, glob, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.transformer.system import TPConfig, LearnedSystem, fit_ffn
from models.transformer.acquire import predict
from models.transformer import intake as IN
from models.transformer import teacher as T
try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None

pats = sys.argv[1:] if len(sys.argv) > 1 else ["checkpoints/tpf_*"]   # >1 pattern => cross-config ensemble
pat = " + ".join(pats)
dirs = sorted({d for p in pats for d in glob.glob(p) if os.path.isfile(os.path.join(d, "last.pt"))})
if not dirs:
    print(f"  (no completed runs for {pat} yet — skipping)")
    sys.exit(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def build(a, sd):
    # FFN read straight from the checkpoint tensor shapes (exact; independent of fit_ffn/ceiling changes).
    ffn = None
    for k in ("predictor.blocks.0.linear1.weight",     # r0
              "readout.blocks.0.linear1.weight",        # r2 (TransformerEncoderLayer)
              "readout.blocks.0.ffn.0.weight"):         # r3 (_R3Block)
        if k in sd:
            ffn = sd[k].shape[0]; break
    assert ffn is not None, "cannot infer FFN width from checkpoint"
    return TPConfig(a["dataset"], a["generator"], a["n_classes"], ffn, n_probes=a["n_probes"],
                    readout=a["readout"], pma_seeds=a["pma_seeds"], seed=a["seed"],
                    readout_arch=a.get("readout_arch", "r0"), stats_bypass=a.get("stats_bypass", False),
                    dropout=a.get("dropout", 0.1), n_slots=a.get("n_slots", 16),
                    token_mlp=a.get("token_mlp", False), moments=a.get("moments", False),
                    signed_mix=a.get("signed_mix", False))


def test_preds(d):
    # CACHE: a finished checkpoint's test predictions never change -> cache them keyed on last.pt mtime + limit,
    # so re-running ens_all/ens_tp recomputes ONLY new/changed groups (was: full 11k-target eval every call).
    ckpath = os.path.join(d, "last.pt")
    lim = int(os.environ.get("TP_TESTLIMIT", "0"))
    mt = os.path.getmtime(ckpath)
    cachep = os.path.join(d, f"test_preds_cache_lim{lim}.pt")
    if os.path.isfile(cachep):
        try:
            c = torch.load(cachep, map_location="cpu", weights_only=False)
            if c.get("mtime") == mt:
                return c["a"], c["preds"], c["trues"]
        except Exception:
            pass                                                # stale/corrupt cache -> recompute
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    a = ck["cfg"]; cfg = build(a, ck["state_dict"])
    sysm = LearnedSystem(cfg); sysm.load_state_dict(ck["state_dict"]); sysm.to(dev).eval()
    mean, std = ck["norm"]["mean"], ck["norm"]["std"]
    cut = float(a.get("cut_off", 0.0))
    te = IN.load_zoo_cached(
        a["data_root"], a["dataset"], "test", seed=0, cut_off=cut
    )
    assert te is not None, (
        f"no threshold-specific test cache for {a['dataset']} cut_off={cut} "
        "(run the `cache` subcommand with the same --cut_off)"
    )
    te = [z for z in te if T.classifier_out_dim(z["params"]) == a["n_classes"]]
    lim = int(os.environ.get("TP_TESTLIMIT", "0"))          # fast approximate ranking on CPU; 0 = full test
    if lim:
        te = te[:lim]
    ids, preds, trues = [], [], []
    with torch.no_grad():
        for i in range(0, len(te), 8):
            ch = te[i:i + 8]
            out = predict(sysm, [z["params"] for z in ch])
            preds += [v.item() * (std + 1e-8) + mean for v in out]
            trues += [z["label"] for z in ch]; ids += [z["id"] for z in ch]
    preds_d, trues_d = dict(zip(ids, preds)), dict(zip(ids, trues))
    try:
        torch.save({"mtime": mt, "a": a, "preds": preds_d, "trues": trues_d}, cachep)
    except Exception:
        pass                                                    # cache best-effort; never fail the eval on it
    return a, preds_d, trues_d


def tau(pred_by_id, true_by_id):
    if kendalltau is None:
        return float("nan")
    ids = sorted(pred_by_id)
    return kendalltau([pred_by_id[i] for i in ids], [true_by_id[i] for i in ids]).correlation


import statistics
acc, cfg0, trues0, seed_taus, seed_preds = {}, None, None, [], []
for d in dirs:
    a, p, t = test_preds(d)
    cfg0 = cfg0 or (a["dataset"], a["generator"], a["n_probes"], a["readout"], a["pred_lr"], a["gen_lr"], a["scheduler"])
    trues0 = trues0 or t
    st = tau(p, t); seed_taus.append(st); seed_preds.append(p)
    for i, v in p.items():
        acc.setdefault(i, []).append(v)
n = len(seed_taus)
cut = float(a.get("cut_off", 0.0))
mean = sum(seed_taus) / n
std = statistics.stdev(seed_taus) if n > 1 else 0.0
ens = {i: sum(vs) / len(vs) for i, vs in acc.items()}
et = tau(ens, trues0)
threshold_pct = int(round(cut * 100))
dataset_label = str(a["dataset"]).upper()
print(f"{dataset_label} Transformer — threshold {threshold_pct}")
print(f"Test Kendall tau: {mean:.4f} ± {std:.4f}")
print("Seeds: " + ", ".join(f"{v:.4f}" for v in seed_taus))

# Auxiliary ensemble is intentionally hidden from the normal result summary.
if os.environ.get("TP_SHOW_AUX"):
    print(f"Auxiliary {n}-model ensemble tau: {et:.4f}")

# TP_THRESH=1 is an auxiliary post-hoc analysis only. It does NOT reproduce
# the Transformer-NFN retrain-per-threshold protocol; use separate --cut_off runs for that.
if os.environ.get("TP_THRESH"):
    ids = sorted(ens)
    # Quasi-NFN / Transformer-NFN protocol: keep checkpoints with TRUE test-accuracy >= an ABSOLUTE cutoff
    # (20/40/60/80%), evaluate tau on that harder high-acc TEST subset. Train-once, eval-only (predictor is
    # NOT retrained per threshold). Labels are accuracy in [0,1], so the cutoff is p/100 directly.
    print("  -- accuracy-thresholded test tau (Transformer/Quasi-NFN protocol; keep true-acc >= cutoff) --")
    print("     thresh   acc>=      n   SINGLE-MODEL(mean+/-std)   [ensemble]")
    for p in (0, 20, 40, 60, 80):
        thr = p / 100.0                                     # ABSOLUTE accuracy cutoff (not a percentile)
        keep = [i for i in ids if trues0[i] >= thr]
        tv = {i: trues0[i] for i in keep}
        per_seed = [tau({i: sp[i] for i in keep}, tv) for sp in seed_preds]         # single-model, per seed
        m = sum(per_seed) / len(per_seed)
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        ekt = tau({i: ens[i] for i in keep}, tv)                                    # K-model ensemble (aux)
        print(f"     {p:2d}%     {thr:.4f}  {len(keep):5d}   {m:.4f} +/- {sd:.4f}        {ekt:.4f}")
