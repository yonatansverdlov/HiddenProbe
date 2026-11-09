# MNIST HiddenProbe hidden-layer ablation

MNIST-GS target CNNs have **three** convolutional hidden layers indexed 0, 1, 2.
Each ablation selects exactly one or two of them; the final output is always used.
The frozen target still executes ALL layers to obtain unchanged final logits.
Only the **observed hidden summaries** are filtered by --hidden_layer_ids.

The six new variants are:

```bash
bash scripts/HiddenProbe/regression/run_mnist_layers_0_sweep.sh
bash scripts/HiddenProbe/regression/run_mnist_layers_1_sweep.sh
bash scripts/HiddenProbe/regression/run_mnist_layers_2_sweep.sh
bash scripts/HiddenProbe/regression/run_mnist_layers_01_sweep.sh
bash scripts/HiddenProbe/regression/run_mnist_layers_02_sweep.sh
bash scripts/HiddenProbe/regression/run_mnist_layers_12_sweep.sh
```

Output-only (ProbeGen) and all-three-layer HiddenProbe already exist; their
canonical scripts are unchanged. In particular, omitting --hidden_layer_ids
preserves the previous HiddenProbe architecture.

Each wrapper calls the shared MNIST layer-ablation sweep driver, using
the original MNIST protocol: 128 shared image probes, raw-accuracy regression,
compact late encoder, interaction rank 48, mixer width 256, batch 32,
Kendall tau, and final 150 epochs. **Readout LR stays fixed at 3e-4.**

We use a **24-config MNIST layer-ablation sweep**, varying **probe_lr**
(not hidden_lr or readout lr), Plateau factor, and patience:

- probe_lr: 1e-4, 3e-4, 6e-4
- plateau factor: 0.2, 0.3, 0.5, 0.7
- plateau patience: 4, 5

Each config is trained for 20 epochs with seed 0, ranked by best validation
Kendall tau only (no test evaluation). The winning configuration is then
retrained for 150 epochs with five independent seeds 0–4.

Each mask writes to
`checkpoints/hidden_layer_ablation/mnist/layers_<mask>/`.
The sweep produces `sweep_results.csv` and `selected_config.json`;
the final phase produces `five_seeds.csv` and `summary.json` with
mean/std/SEM. Valid completed runs are reused. Partial runs are restarted.

Overrides:

```bash
# Preview all commands, without downloading data:
bash scripts/HiddenProbe/regression/run_mnist_layers_02_sweep.sh --dry-run

# Normal cluster runs (data already present):
SKIP_DATA_SETUP=1 bash scripts/HiddenProbe/regression/run_mnist_layers_02_sweep.sh

# Override epochs / seed count when experimenting:
SKIP_DATA_SETUP=1 SWEEP_EPOCHS=20 FINAL_EPOCHS=150 NUM_SEEDS=5 \
  bash scripts/HiddenProbe/regression/run_mnist_layers_12_sweep.sh
```

Run the CPU capture regression test using
`python -m unittest discover -s tests -p 'test_mnist_hidden_layer_ablation.py'`.
