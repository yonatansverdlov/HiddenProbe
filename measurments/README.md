# Measurement benchmarks

This folder was restored from the September 2026 history. The historical
MNIST-INR and CIFAR-WP scripts are preserved for reference; they may require
third-party benchmarking modules that are no longer checked into the repo.
The new Fashion-MNIST INR benchmark uses only the current repository code.

## FMNIST-INR: probe count vs runtime and GPU memory

Use an allocated CUDA GPU and the existing FMNIST-INR data installation.
No retraining or downloading is required.
The original FMNIST INR trainer uses FP32 (no autocast); this benchmark
therefore defaults to FP32. The historical MNIST benchmark used BF16.

~~~bash
# 16/32/64/128/256 probes, both methods, eval-mode forward costs.
# Defaults to the ENTIRE train split, batch size 32, FP32 and 1 measured eval-mode pass.
python measurments/benchmark_fmnist_inr_probe_cost.py

# Forward-only on the entire Fashion-MNIST INR TRAIN split,
# analogous to the old full-split forward-timing comparison:
python measurments/benchmark_fmnist_inr_probe_cost.py --phase forward --limit 0

# Smaller smoke run or GPU with limited memory:
python measurments/benchmark_fmnist_inr_probe_cost.py --phase both --limit 32

# Optional training-step cost (forward + backward + Adam) instead of inference:
python measurments/benchmark_fmnist_inr_probe_cost.py --phase train_step

# More robust inference timing on the entire train split:
python measurments/benchmark_fmnist_inr_probe_cost.py --repeats 3
~~~

Output: measurments/fmnist_inr_probe_cost.csv, written one configuration
at a time. The default full training split and 5 probe counts x 2 methods x 1 phase give 10 rows.
Using --phase both gives 20 rows.

### What the measurements mean

- model_compute_s: synchronized model forward-only elapsed time, or a
  forward + backward + Adam optimizer step for phase=train_step.
  Disk loading and CPU-to-GPU transfer are excluded.
- target_transfer_s: elapsed time for the frozen target INRs to move to GPU.
  The probes and prediction model are already on GPU.
- data_load_s: time reading/building target networks from CPU checkpoints.
  Keep this separate from compute; filesystem cache may alter it.
- end_to_end_s: model setup plus the full measured data-loading, transfer,
  compute, and cleanup wall time, excluding warmup.
- compute_ms_per_network: model compute time divided by the number of target
  INRs and passes. Useful for cross-Q and cross-method comparisons.
- cuda_peak_allocated_mib: peak live PyTorch GPU allocation while
  processing each configuration; includes predictor, optimizer (train mode),
  and the current batch of frozen target networks.
- cuda_peak_reserved_mib: peak PyTorch CUDA caching allocator reservation.
  Not identical to nvidia-smi usage.

Each configuration launches in a fresh Python process to prevent previous
CUDA allocations/reservations from contaminating the comparison.
All variants use exactly the same ordered set of target INRs and batch size.
The methods differ in whether hidden states are included; no prediction
accuracy is measured. Model weights start randomly initialized, and the
train-step phase benchmarks optimization steps, not convergence or
whole-training duration.

This benchmark deliberately does not claim that Q=64 matches Q=128 in
accuracy: consult the independently run probe-count accuracy experiments
before comparing equal-accuracy configurations.

### Historical reference

- benchmark_mnist_regression_forward.py (despite its name, its final
  version benchmarks MNIST-INR classification forward cost)
- benchmark_cifar10_wp_regression_forward.py
- mnist_inr_forward.csv: recorded September 2026 results, Q=128, 55,000
  MNIST-INR target models, BF16, batch size 32. The CSV's total_seconds
  includes disk loading and model construction, while forward_seconds
  excludes those setup costs.

The historical measurements did not record GPU peak-memory statistics.


## Figure-3 cost sweeps: SVHN, MNIST-Transformers and CIFAR10-WP

These are **standalone inference-cost benchmarks**, complementary to the
accuracy-vs-probe-count experiments. The FMNIST-INR cost sweep above is
unchanged. For each of the other three tasks, both methods run on the same
canonical **TRAIN** target models with `Q={16,32,64,128,256}`, in isolated
processes (10 configurations per task). Each run writes one CSV row immediately
and a rerun skips existing rows with matching settings.

```bash
# Full train splits; Eval mode, FP32, batch size 32, 2 warmup batches,
# one measured pass. Requires an allocated CUDA GPU and existing caches.
bash measurments/run_svhn_probe_cost.sh
bash measurments/run_mnist_transformer_probe_cost.sh
bash measurments/run_cifar10_wp_probe_cost.sh

# Preview all 10 configurations without loading data or touching outputs:
bash measurments/run_svhn_probe_cost.sh --dry-run

# Quick sanity check, without wiping existing full-run results:
bash measurments/run_svhn_probe_cost.sh --limit 32 --csv /tmp/svhn_probe_cost_smoke.csv

# Three measured passes for more stable compute times (new output file):
bash measurments/run_mnist_transformer_probe_cost.sh --repeats 3 \
  --csv measurments/mnist_transformer_probe_cost_3passes.csv
```

Results:
- `measurments/svhn_probe_cost.csv`
- `measurments/mnist_transformer_probe_cost.csv`
- `measurments/cifar10_wp_probe_cost.csv`

Common implementation: `measurments/benchmark_other_probe_cost.py`.
The `--probes`, `--methods`, `--limit`, `--batch-size`, `--precision`,
`--warmup`, `--repeats`, `--seed`, `--data-root`, `--device`,
`--csv` flags are available. `--overwrite` explicitly resets the selected
CSV. Without it, existing results are never overwritten. Use a distinct CSV
name for smoke tests or different settings.

**Definitions and limitations**

- `model_compute_s` = GPU-synchronized forward for all target models; does
  not include batch CPU->GPU transfers. `compute_ms_per_network` is normalized
  by target model count and repetitions.
- `target_transfer_s` = all current-target transfers CPU->GPU.
- `data_load_s` = one-time load and CPU reconstruction of the canonical
  target network population. `model_setup_s` = predictor construction and
  placement on GPU.
- `end_to_end_s` = CPU data load + predictor setup + measured full pass wall
  time (forward, transfers, cleanup). Warmup and subprocess startup excluded.
- `cuda_peak_allocated_mib` and `cuda_peak_reserved_mib` are PyTorch CUDA
  allocator metrics, **not** the whole device's `nvidia-smi` utilization.
- Target CNNs are CPU-resident between batches for **both** methods to avoid
  unfairly counting an entire zoo as HiddenProbe's inference memory. For the
  Transformer benchmark, target parameter dictionaries are also moved to GPU
  only for the current batch. Thus costs do **not** match the complete training
  pipeline's original GPU-resident SVHN HiddenProbe implementation.
- Default **FP32** provides a controlled same-precision comparison, like
  FMNIST. The SVHN/WP HiddenProbe original CNN trainer uses BF16 autocast,
  whereas core ProbeGen and the Transformer trainer use FP32. An optional
  `--precision bf16` measures both methods with BF16 on a capable GPU.
- These are **randomly initialized model inference benchmarks**, not trained
  checkpoint evaluations. No accuracy or test metrics are produced. Pair them
  with the separate probe-count experiments before making any performance-per-
  cost claim. CIFAR10-WP is not part of `probe_count_5seeds.py` in this repo:
  there is no five-seed Q-sweep accuracy result for it from that script.
- Use the same GPU and keep it otherwise idle when comparing methods/Q.
- Data must already exist. For the Transformer cache, run
  `python main.py transformer cache --dataset mnist --seed 0 --cut_off 0 --data_root data`;
  for other datasets, use their existing `scripts/setup_data/regression_*.sh`
  setup scripts. This benchmark never downloads or modifies datasets.
