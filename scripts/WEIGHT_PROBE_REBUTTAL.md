# ProbeX / MVProbe five-benchmark rebuttal sweeps

This branch restores the weight-space ProbeX (single-view) and MVProbe (four-view)
encoders and runs **the same hyperparameter grid independently for each method**.

These scripts are **prepared, not GPU-validated yet**. They start by invoking
the same `scripts/setup_data/*` download/preparation scripts as HiddenProbe.

## Commands

| Benchmark | ProbeX | MVProbe | Configs per method |
| --- | --- | --- | ---: |
| MNIST-Transformers (accuracy regression) | `bash scripts/ProbeX/regression/run_mnist_transformers.sh` | `bash scripts/MVProbe/regression/run_mnist_transformers.sh` | 1728 |
| SVHN (accuracy regression) | `bash scripts/ProbeX/regression/run_svhn_regression.sh` | `bash scripts/MVProbe/regression/run_svhn_regression.sh` | 864 |
| MNIST (accuracy regression) | `bash scripts/ProbeX/regression/run_mnist_regression.sh` | `bash scripts/MVProbe/regression/run_mnist_regression.sh` | 864 |
| FMNIST (INR classification) | `bash scripts/ProbeX/classification/run_fmnist_inr.sh` | `bash scripts/MVProbe/classification/run_fmnist_inr.sh` | 576 |
| CIFAR-10 (INR classification, non-augmented) | `bash scripts/ProbeX/classification/run_cifar10_inr_nonaug.sh` | `bash scripts/MVProbe/classification/run_cifar10_inr_nonaug.sh` | 576 |

Total: **4608 hyperparameter configurations per method**, **9216 across
both methods**, before final five-seed evaluation. This is computationally
expensive; consider reducing the grid before production runs.

## Common sweep grid

- Observed weight matrix (selected layer): INR layers 0/1;
  CNN conv layers 0/1/2; Transformers six selected weight matrices below
- Learned weight probes: `n_probes = {64, 128}`
- Probe projection dimension: `proj_dim = {64, 128}`
- Representation dimension: `rep_dim = 512` (fixed, overridable with `REP_DIM`)
- Adam learning rate: `lr = {1e-4, 3e-4, 5e-4}`
- Batch size: `{64, 128}`
- Weight decay: `{0, 1e-5, 1e-4}`
- `ReduceLROnPlateau` factor: `{0.2, 0.3, 0.5, 0.7}`
- Scheduler patience: 5; minimum learning rate: `1e-6`
- Sweep: seed 0, 60 epochs, **validation-only model selection**
- Final: selected hyperparameters, seeds 0–4, 150 epochs; test only after
  selecting the best epoch on validation
- `SWEEP_EPOCHS`, `FINAL_EPOCHS`, `REP_DIM`, `OUT_DIR`, and
  `DATA_ROOT` are environmental overrides.

The output is a per-experiment `summary.json`, aggregate
`sweep_results.csv`, `selected_config.json`, and a five-seed summary.
Completed configurations are skipped on re-run.

**Probe-count interpretation:** ProbeX has one first-order view per probe;
MVProbe computes *four* weight-space views. `n_probes` is therefore not a
matched target-network query count against HiddenProbe and should not be
described as one.

## MNIST-Transformers weight choices (0-based indices)

| Layer index | Matrix |
| ---: | --- |
| 0 | `encoder.0.attention.queries.weight` |
| 1 | `encoder.0.attention.values.weight` |
| 2 | `encoder.0.fc1.weight` |
| 3 | `encoder.1.attention.queries.weight` |
| 4 | `encoder.1.attention.values.weight` |
| 5 | `encoder.1.fc1.weight` |

The Transformer runners call `scripts/setup_data/transformer_mnist.sh`,
then `python main.py transformer cache --dataset mnist --seed 0
--cut_off 0 --data_root "$DATA_ROOT"`. The weight-space trainer uses the
same epoch-75, threshold-0, 70/15/15 split as the existing Transformer
experiment. Per-layer matrix caches avoid reloading full checkpoints for
every sweep configuration.

## Before publication

- Run a short smoke test for one configuration per benchmark/method, then
  confirm complete multi-seed runs.
- Verify sample splits, tensor shapes, test isolation and the chosen layer
  for the Transformer zoo.
- Report best-epoch selection, per-seed Kendall tau / classification
  accuracy, and total trainable parameter counts.
- Do **not** merge into `main` until these checks pass.
