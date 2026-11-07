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
