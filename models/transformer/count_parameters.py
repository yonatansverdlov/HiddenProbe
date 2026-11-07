"""count_parameters command (spec §12): instantiate the shipped configs, report components, enforce the
1,880,000 ceiling, and check the Q64 configs against the analytic table. No data / checkpoints needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.transformer.system import TPConfig, verify_config, fit_ffn


def p2(dataset, gen, C, n_probes, readout, pma_seeds=1, readout_arch="r0", stats_bypass=False):
    ffn = fit_ffn(dataset, gen, C, n_probes, readout, pma_seeds, readout_arch, stats_bypass)  # <=1.88M
    return TPConfig(dataset, gen, C, ffn, n_probes=n_probes, readout=readout, pma_seeds=pma_seeds,
                    readout_arch=readout_arch, stats_bypass=stats_bypass)

CONFIGS = [
    # locked Q64 spec configs (must match analytic)
    TPConfig("mnist", "g3", 10, 944),
    TPConfig("agnews", "g3", 10, 944),
    TPConfig("agnews", "g3", 4, 944),
    # phase-2 (Q128 x readout) — FFN auto-fit to <=1.88M
    p2("mnist", "g3", 10, 128, "multi"),
    p2("mnist", "g3", 10, 128, "pma", pma_seeds=4),
    p2("agnews", "g3", 4, 128, "multi"),
    # architecture round: R2 cross-layer fusion (Q256), +/- stats bypass
    p2("mnist", "g3", 10, 256, "multi", readout_arch="r2"),
    p2("mnist", "g3", 10, 256, "multi", readout_arch="r2", stats_bypass=True),
    p2("agnews", "g3", 4, 256, "multi", readout_arch="r2"),
    p2("agnews", "g3", 4, 256, "multi", readout_arch="r2", stats_bypass=True),
    # R3 latent-memory readout (Q256)
    p2("mnist", "g3", 10, 256, "multi", readout_arch="r3"),
    p2("mnist", "g3", 10, 256, "multi", readout_arch="r3", stats_bypass=True),
    p2("agnews", "g3", 4, 256, "multi", readout_arch="r3", stats_bypass=True),
    # output-only ProbeGen baseline (Q256)
    p2("mnist", "g3", 10, 256, "multi", readout_arch="rout"),
    p2("agnews", "g3", 4, 256, "multi", readout_arch="rout"),
    # channel-trajectory readout (Q256; mnist FFN fixed at 384, agnews cap-fit)
    TPConfig("mnist", "g3", 10, 384, n_probes=256, readout="multi", readout_arch="channel_trajectory"),
    TPConfig("agnews", "g3", 4, 384, n_probes=256, readout="multi", readout_arch="channel_trajectory"),
]

if __name__ == "__main__":
    all_ok = True
    for cfg in CONFIGS:
        total, comp, ok_ceiling = verify_config(cfg)
        all_ok = all_ok and ok_ceiling
        print()
    print("RESULT:", "ALL CONFIGS UNDER 1.88M CEILING" if all_ok else "SOME CONFIG OVER 1.88M — diagnose")
    sys.exit(0 if all_ok else 1)
