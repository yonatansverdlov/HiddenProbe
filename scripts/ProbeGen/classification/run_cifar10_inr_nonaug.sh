#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_cifar10.sh

MAIN_PY="${MAIN_PY:-main.py}"


DATASET="cifar10"
EXP_NAME="cifar10_nonaug_mlp2_dhid318_FINAL_5SEEDS_30E_plateau_lr7e-4_pat3_fac0.2"

N_TOKENS=128
D_HID=318
MIXER_LAYERS=6

GEN_TYPE="linear_2_no_acts"
GEN_LATENT_Z=32
GENERATOR_WIDTH=16

PER_PROBE_MLP="mlp2"
PER_PROBE_MLP_WIDTH=256
PER_PROBE_OUT_DIM=4
R_PER_HIDDEN=2

BATCH_SIZE=64
EPOCHS=30
NUM_SEEDS=5
START_SEED=0

SCHEDULER="plateau"
LR=7e-4
PLATEAU_MONITOR="val_acc"
PLATEAU_PATIENCE=3
PLATEAU_FACTOR=0.2
PLATEAU_MIN_LR=1e-6

WD=0.0
EVAL_EVERY=500
N_WORKERS=0
DEVICE="cuda"

CIFAR_EXTRA_AUG=0
CIFAR_CACHE_MODELS=false

python "${MAIN_PY}" \
  --method probegen \
  --task classification \
    --exp_name="${EXP_NAME}" \
    --dataset="${DATASET}" \
    --seed="${START_SEED}" \
    --num_seeds="${NUM_SEEDS}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --n_probes="${N_TOKENS}" \
    --d_hid="${D_HID}" \
    --mixer_n_layers="${MIXER_LAYERS}" \
    --gen_type="${GEN_TYPE}" \
    --gen_latent_z="${GEN_LATENT_Z}" \
    --generator_width="${GENERATOR_WIDTH}" \
    --per_probe_mlp="${PER_PROBE_MLP}" \
    --per_probe_mlp_width="${PER_PROBE_MLP_WIDTH}" \
    --per_probe_out_dim="${PER_PROBE_OUT_DIM}" \
    --r_per_hidden="${R_PER_HIDDEN}" \
    --scheduler="${SCHEDULER}" \
    --plateau_monitor="${PLATEAU_MONITOR}" \
    --plateau_patience="${PLATEAU_PATIENCE}" \
    --plateau_factor="${PLATEAU_FACTOR}" \
    --plateau_min_lr="${PLATEAU_MIN_LR}" \
    --lr="${LR}" \
    --weight_decay="${WD}" \
    --eval_every="${EVAL_EVERY}" \
    --n_workers="${N_WORKERS}" \
    --cifar_extra_aug="${CIFAR_EXTRA_AUG}" \
    --cifar_cache_models="${CIFAR_CACHE_MODELS}" \
    --device="${DEVICE}"

SUMMARY="experiments/classification/${DATASET}/runs/${EXP_NAME}/seeds_summary.csv"

python - <<PY
import pandas as pd
path = "${SUMMARY}"
df = pd.read_csv(path)
print(df.to_string(index=False))

val = df["best_val_acc"].astype(float)
test = df["best_test_acc"].astype(float)

print()
print(f"VAL accuracy:  {val.mean():.6f} ± {val.std(ddof=1):.6f}")
print(f"TEST accuracy: {test.mean():.6f} ± {test.std(ddof=1):.6f}")
print(f"Saved to: {path}")
PY
