# Measurement benchmarks

This folder was restored from the September 2026 history. The historical
MNIST-INR and CIFAR-WP scripts are preserved for reference; they may require
third-party benchmarking modules that are no longer checked into the repo.
The new Fashion-MNIST INR benchmark uses only the current repository code.

## FMNIST-INR: probe count vs runtime and GPU memory

Use an allocated CUDA GPU and the existing FMNIST-INR data installation.
No retraining or downloading is required.

~~~bash
# 16/32/64/128/256 probes, both methods, both forward and training-step costs.
# Defaults to the ENTIRE train split, batch size 32, BF16 and 1 measured pass.
python measurments/benchmark_fmnist_inr_probe_cost.py

# Forward-only on the entire Fashion-MNIST INR TRAIN split,
# analogous to the old full-split forward-timing comparison:
python measurments/benchmark_fmnist_inr_probe_cost.py --phase forward --limit 0

# Smaller smoke run or GPU with limited memory:
python measurments/benchmark_fmnist_inr_probe_cost.py --phase both --limit 32

# More robust timing on the entire train split:
python measurments/benchmark_fmnist_inr_probe_cost.py --repeats 3
~~~

Output: measurments/fmnist_inr_probe_cost.csv, written one configuration
at a time. The default full training split and 5 probe counts x 2 methods x 2 phases give 20 rows.
Using --phase forward gives the 10-row comparison.

### What the measurements mean

- model_compute_s: synchronized model forward-only elapsed time, or a
  forward + backward + Adam optimizer step for phase=train_step.
  Disk loading and CPU-to-GPU transfer are excluded.
- target_transfer_s: elapsed time for the frozen target INRs to move to GPU.
  The probes and prediction model are already on GPU.
- data_load_s: time reading/building target networks from CPU checkpoints.
  Keep this separate from compute; filesystem cache may alter it.
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
