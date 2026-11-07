# Figure 3: Probe Count × 5 Seeds (HiddenProbe vs ProbeGen)

A resumable benchmark for the three datasets in Figure 3: **Fashion-MNIST INR
(classification accuracy)**, **SVHN CNN zoo (test Kendall tau)** and
**MNIST-Transformers (test Kendall tau)**.

The driver **reuses each method's saved best hyperparameters** from
`scripts/HiddenProbe/` and `scripts/ProbeGen/`. Only `n_probes` and the
random initialization/training seed vary. The default experiment grid is:
`N={16,32,64,128,256}`, `seed={0,1,2,3,4}`, two methods = **50 runs
per dataset**. The fixed training/validation/test split is unchanged.

## Commands

Use the existing CUDA + Conda environment on an allocated GPU node:

```bash
cd ~/HiddenProbe
git pull --ff-only

# Review full launch commands without training.
python scripts/probe_count_5seeds.py --dataset fmnist_inr --dry-run
python scripts/probe_count_5seeds.py --dataset svhn --dry-run
python scripts/probe_count_5seeds.py --dataset mnist_transformer --dry-run

# Smoke-test one full training + evaluation run.
python scripts/probe_count_5seeds.py --dataset svhn --max-runs 1

# Run all remaining configurations and seeds, sequentially on ONE GPU.
python scripts/probe_count_5seeds.py --dataset fmnist_inr
python scripts/probe_count_5seeds.py --dataset svhn
python scripts/probe_count_5seeds.py --dataset mnist_transformer
```

Optional flags: `--probes 16 32`, `--seeds 0 1`,
`--methods hiddenprobe`, `--skip-setup`, `--eval-batch-size 32` (applies
to FMNIST), `--max-runs 1`, `--out-root <path>`. Completed profiled runs
are skipped; the driver refuses to silently reuse a completed training run
without benchmark metrics. For the canonical ProbeGen/FMNIST core backend,
checkpoints remain at `experiments/{classification,regression}/<dataset>/runs/`
as required by the existing trainer.

## What is measured

Each training run selects its best checkpoint **on validation only**. The
normal, full held-out TEST evaluation of that checkpoint is synchronized and
timed once after training (no extra test pass). The following fields are
recorded for each (dataset, method, N, seed):

- `test_metric`: TEST accuracy in [0,1] for Fashion-MNIST INR; TEST Kendall
  tau for SVHN and MNIST-Transformers.
- `eval_time_sec`: wall time for the **full test evaluation**, including
  method-specific input pipeline and metric calculation, with CUDA
  synchronization at both boundaries.
- `eval_n_examples` and `eval_examples_per_sec`: evaluation throughput.
- `eval_peak_gpu_mb`: CUDA **peak live allocated** memory during the test
  evaluation. This **includes tensors already live at evaluation start**.
- `eval_gpu_baseline_mb` and `eval_incremental_gpu_mb`: live allocated
  memory before evaluation, and extra peak above that baseline.
- `train_time_sec`: time from the beginning of training to selection of the
  final best-validation checkpoint, excluding dataset setup and including
  periodic validation.
- `trainable_params`, `gpu_name`, `torch_version`, `eval_batch_size`.

**Important memory caveat:** The existing CNN-zoo HiddenProbe backend keeps
the target-network zoo resident on GPU, while the core ProbeGen backend
streams target networks in batches. Thus the absolute GPU peaks include
**different baseline data residency** and must not be interpreted as
architecture-only inference memory. Both peak and incremental allocation are
reported transparently. For an exact, standalone inference-memory comparison,
run evaluation in isolated processes with identical batch-level target
materialization.

FMNIST training batch sizes differ between saved best configs (64 for
HiddenProbe, 32 for ProbeGen), but **final TEST evaluation uses batch size 32
for both**. SVHN retains its saved training/validation batch sizes, but **final
TEST evaluation uses batch size 64 for both** (instead of HiddenProbe\'s
original 256). Transformer test evaluation uses batch size 32 for both. For meaningful runtime comparisons, use the same GPU,
avoid other GPU jobs and repeat timing if results are noisy.

## Outputs

- `checkpoints/probe_count_5seeds/<dataset>/per_seed.csv`: individual run
  results, updated after every finished run.
- `checkpoints/probe_count_5seeds/<dataset>/summary.csv`: **mean, sample SD
  and SEM** for each metric and each (method, N), written only after all
  requested seeds finish.
- `checkpoints/probe_count_5seeds/<dataset>/comparison.csv`: paired
  HiddenProbe minus ProbeGen test metric, ProbeGen/HiddenProbe evaluation
  speedup (greater than 1 means HiddenProbe is faster), and
  HiddenProbe/ProbeGen absolute GPU memory ratio. Interpret the memory
  ratio cautiously because the two backends retain different data tensors.
- `checkpoints/probe_count_5seeds/<dataset>/logs/*.log`: stdout/stderr of
  each run.
- Each trainer's `probe_count_eval.json`: machine-readable benchmark data.

In MNIST-Transformers, the original HiddenProbe channel-trajectory readout
only supported N=256. The benchmark branch generalizes its input trajectory
geometry for N={16,32,64,128} with the original 3:1 split of encoder/native
probes; N=256 is preserved exactly.

**Status:** Repository edits have been committed, but the full experiment and
CUDA evaluation have **not** been executed on the cluster. Start with one
`--max-runs 1` smoke test.
