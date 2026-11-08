#!/bin/bash
#SBATCH --job-name=pc3_mnist_transformer_probegen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -f scripts/probe_count_5seeds.py || { echo "Submit this job from the HiddenProbe repository root." >&2; exit 1; }

python -u scripts/probe_count_5seeds.py \
  --dataset mnist_transformer \
  --methods probegen \
  --seeds 0 1 2 \
  --out-root checkpoints/probe_count_3seeds/probegen \
  --skip-setup
