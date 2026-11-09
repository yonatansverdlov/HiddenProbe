# SVHN HiddenProbe hidden-layer ablation

SVHN-GS target CNNs have **three** convolutional hidden layers indexed 0, 1, 2.
Each ablation selects exactly one or two of them; the final output is always used.
The frozen target still executes ALL layers to obtain unchanged final logits.
Only the **observed hidden summaries** are filtered by --hidden_layer_ids.

The six new variants are:

```bash
bash scripts/HiddenProbe/regression/run_svhn_layers_0_sweep.sh
bash scripts/HiddenProbe/regression/run_svhn_layers_1_sweep.sh
bash scripts/HiddenProbe/regression/run_svhn_layers_2_sweep.sh
bash scripts/HiddenProbe/regression/run_svhn_layers_01_sweep.sh
bash scripts/HiddenProbe/regression/run_svhn_layers_02_sweep.sh
bash scripts/HiddenProbe/regression/run_svhn_layers_12_sweep.sh
```

Output-only (ProbeGen) and all-three-layer HiddenProbe already exist; their
canonical scripts are unchanged. In particular, omitting --hidden_layer_ids
preserves the previous HiddenProbe architecture.

Each wrapper calls the shared SVHN layer-ablation sweep driver, which uses
the original SVHN protocol (128 shared image probes, raw-accuracy regression,
expressive late encoder, interaction rank 96, mixer width 384, batch 32,
probe LR 6e-4, Kendall tau, final 150 epochs). As in the INR ablations, we
search over 27 hyperparameter choices with seed 0 (20 epochs each) using
only best validation tau; we then retrain the selected config for five
independent seeds (0–4), 150 epochs. During the sweep, test evaluation is
disabled. The original SVHN setting lr=3e-4, plateau patience=4,
factor=0.7 appears in the grid.

The current repository's run_svhn_regression.sh does five fixed-hyperparameter
seeds, not an HPO sweep. This script adopts the 27-config validation-sweep
pattern used for FMNIST/CIFAR layer selection, with SVHN-specific ranges:

- lr: 2e-4, 3e-4, 5e-4
- plateau patience: 3, 4
- plateau factor: 0.5, 0.7, 0.8, 0.9
- plus a cosine schedule for each lr

Each mask writes to
`checkpoints/hidden_layer_ablation/svhn/layers_<mask>/`.
The sweep produces `sweep_results.csv` and `selected_config.json`;
the final phase produces `five_seeds.csv` and `summary.json` with
mean/std/SEM. Valid completed runs are reused. Partial runs are restarted.

Overrides:

```bash
# Preview all commands, without downloading data:
bash scripts/HiddenProbe/regression/run_svhn_layers_02_sweep.sh --dry-run

# Normal cluster runs (data already present):
SKIP_DATA_SETUP=1 bash scripts/HiddenProbe/regression/run_svhn_layers_02_sweep.sh

# Override epochs / seed count when experimenting:
SKIP_DATA_SETUP=1 SWEEP_EPOCHS=20 FINAL_EPOCHS=150 NUM_SEEDS=5 \
  bash scripts/HiddenProbe/regression/run_svhn_layers_12_sweep.sh
```

Run the CPU capture regression test using
`python -m unittest tests/test_svhn_hidden_layer_ablation.py`.
