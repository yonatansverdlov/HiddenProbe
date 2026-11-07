# HiddenProbe INR hidden-layer selection

Both HiddenProbe INR classification backends now support the same public flags
through `python main.py`:

- `--include_hidden_features true|false`
- `--hidden_layer_ids ID [ID ...]` (0-based, optional; default **None**)

Hidden layer indices exclude the final output Linear. IDs select which target
hidden pre-activations are observed; all target layers still run normally.
Both the learned probes and the downstream classification head remain trainable.

| Include hidden | Hidden IDs | Effective hidden layers |
| --- | --- | --- |
| true (HiddenProbe default) | omitted / `None` | all available hidden layers |
| true | `0` | first hidden layer only |
| true | `1` | second hidden layer only |
| true | `0 1` or `0,1` | both, in specified order |
| false | omitted / `None` | output only |
| false | explicit indices | error (conflicting flags) |

For both Fashion-MNIST INR and CIFAR-10 INR, there are exactly **two**
hidden layers: 0 (2->32) and 1 (32->32), with a SIREN activation.
The final output layer has one channel for Fashion-MNIST and three RGB
channels for CIFAR-10, and is **never** a hidden-layer ID.

## Examples

Keep all existing training options for the intended dataset and supply
the selected layer as an extra CLI option:

```bash
# FMNIST: first hidden layer and output
python main.py --method hiddenprobe --task classification --dataset fmnist \
  --exp_name fmnist_layer0 --hidden_layer_ids 0

# FMNIST: second hidden layer and output
python main.py --method hiddenprobe --task classification --dataset fmnist \
  --exp_name fmnist_layer1 --hidden_layer_ids 1

# CIFAR-10 INR (non-augmented): first hidden layer and output
python main.py --method hiddenprobe --task classification --dataset cifar10 \
  --exp_name cifar10_layer0 --hidden_layer_ids 0

# CIFAR-10 INR (non-augmented): second hidden layer and output
python main.py --method hiddenprobe --task classification --dataset cifar10 \
  --exp_name cifar10_layer1 --hidden_layer_ids 1

# Preserve the existing HiddenProbe baseline (uses both hidden layers)
python main.py --method hiddenprobe --task classification --dataset fmnist \
  --exp_name fmnist_full

# Optional output-only HiddenProbe (does not change the ProbeGen baseline)
python main.py --method hiddenprobe --task classification --dataset cifar10 \
  --exp_name cifar10_output_only --include_hidden_features false
```

**Important:** The short invocations above use trainer defaults. For
publication-grade ablations, also pass the exact hyperparameters from the
corresponding existing dataset runner/sweep. The FMNIST and CIFAR HiddenProbe
backends have different architectures and hyperparameter sets; do not assume
their defaults or heads are interchangeable.

FMNIST uses the canonical `ProbeGen` class with per-probe hidden encoders;
selecting one hidden layer changes the token width from 5 to 3
(`r_per_hidden=2`). CIFAR-10 uses the neuron-profile / Set Transformer
head; selecting one hidden layer changes the hidden-token count from 64 to
32 (for `H=32`) while retaining the original layer's embedding ID.

Selected indices are stored in the FMNIST `meta.json` and CIFAR
`summary.json`. For quick CPU sanity checks:

```bash
python -m unittest discover -s tests -p test_hidden_layer_selection.py
```

The existing output-only ProbeGen baseline and full HiddenProbe baseline do
not need to be re-run for the layer-selection ablation.

## Four ablation sweep scripts (FMNIST / CIFAR-10 non-augmented)

These are the only *new* layer combinations: output + layer 0, and output +
layer 1 for each dataset. The output-only and both-hidden-layers results
are taken from the existing baseline table.

```bash
bash scripts/HiddenProbe/classification/run_fmnist_inr_layer0_sweep.sh
bash scripts/HiddenProbe/classification/run_fmnist_inr_layer1_sweep.sh
bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug_layer0_sweep.sh
bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug_layer1_sweep.sh
```

All four scripts automatically set up their dataset (unless
`SKIP_DATA_SETUP=1`) and support `--dry-run`, which skips dataset setup.
They share the implementation in
`scripts/HiddenProbe/classification/hidden_layer_ablation_sweep.py`.
Unfinished configurations are rerun; completed ones are skipped.

- **FMNIST**: 3 learning rates `{3e-4,5e-4,7e-4}` ×
  2 Plateau patiences `{3,5}` × 4 factors `{0.2,0.3,0.5,0.7}` =
  **24 configs per layer**. Sweep: seed 0, 20 epochs; final: seeds 0–4,
  30 epochs. Uses the FMNIST HiddenProbe MLP head, batch size 64,
  128 probes and the dataset's existing model settings.
- **CIFAR-10 non-augmented**: the same 24 Plateau combinations plus
  3 cosine schedules, one per learning rate = **27 configs per layer**.
  Sweep: seed 0, 20 epochs; final: seeds 0–4, **60 epochs** to match
  the currently checked-in CIFAR-10 HiddenProbe baseline runner
  (the earlier paper description used 30 final epochs).
  Uses the existing CIFAR Set Transformer head (d=120, 2 layers, 8 heads),
  batch size 32, EMA 0.999, probe LR 7e-4 and the other dataset-specific
  options from the checked-in runner.

Hyperparameter selection uses **validation accuracy only**.
The existing trainers may also log test metrics during the sweep, but
they are not used for ranking or selection.

Results are in
`checkpoints/hidden_layer_ablation/{fmnist,cifar10}/layer{0,1}/`;
individual FMNIST runs remain under
`experiments/classification/fmnist/runs/`.
Each sweep produces `sweep_results.csv` and `selected_config.json`;
the final phase produces `five_seeds.csv` and `summary.json`.

```bash
# Preview the 24 or 27 configs and commands without running training:
bash scripts/HiddenProbe/classification/run_fmnist_inr_layer0_sweep.sh --dry-run

# Optional overrides (defaults are 20, 30/60, 5):
SWEEP_EPOCHS=20 FINAL_EPOCHS=60 NUM_SEEDS=5 \
  bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug_layer0_sweep.sh
```
