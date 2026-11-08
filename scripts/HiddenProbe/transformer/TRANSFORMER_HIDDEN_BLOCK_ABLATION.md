# Transformer hidden-block ablation

Two target model zoos are tested: MNIST-Transformers and AGNews-Transformers.
Both target networks have two Transformer encoder blocks. The ablation runs
**HiddenProbe** with precisely the following information:

| Condition | Block 1 | Block 2 | Final teacher LayerNorm | Classifier logits |
|---|:---:|:---:|:---:|:---:|
| Existing ProbeGen (unchanged) | | | | yes |
| Block 1 only | yes | | yes | yes |
| Block 2 only | | yes | yes | yes |
| Existing HiddenProbe (unchanged) | yes | yes | yes | yes |

For Block 1 and Block 2, the selected hidden representation is processed
with the same channel-trajectory encoder and fused with the final LayerNorm
representation using the same readout architecture. Both ablations have
identical trainable parameter counts and the same input token shape.
**No selected-block ablation ever reads the other block's response** (though
the frozen target network itself must still execute both blocks normally).

## Four independent sweeps

From the repository root, in an environment with GPU and prepared dependencies:

```bash
bash scripts/HiddenProbe/transformer/run_mnist_transformer_block1_sweep.sh
bash scripts/HiddenProbe/transformer/run_mnist_transformer_block2_sweep.sh
bash scripts/HiddenProbe/transformer/run_agnews_transformer_block1_sweep.sh
bash scripts/HiddenProbe/transformer/run_agnews_transformer_block2_sweep.sh
```

Each script first prepares the dataset, caches the fixed zero-threshold split
(seed 0, 70/15/15), then performs a 27-config validation sweep followed by
five independent final seeds.

Each script also accepts `--dry-run` (prints commands, does **not** download
data or train) and `--only-sweep` (skips five-seed final training).
Set `SKIP_DATA_SETUP=1` to bypass preparation/cache when already done.

## Search space and protocol

The same 27-config scheduler/learning-rate grid used in the INR hidden-layer
ablation is applied **separately** for each dataset and selected block:

- Predictor learning rate: `{3e-4, 5e-4, 7e-4}`.
- ReduceLROnPlateau: patience `{3,5}` x factor `{0.2,0.3,0.5,0.7}` (24 configs).
- Cosine schedule: one for each learning rate (3 configs).

Fixed to the original full HiddenProbe Transformer runner: G3 learned probes,
Q=256, channel-trajectory readout, hidden width=192 and two readout blocks;
teacher output classes 10 for MNIST / 4 for AGNews, FFN=384,
generator LR=5e-4 for MNIST / 1e-3 for AGNews, AdamW predictor weight
decay 1e-3, dropout 0.1, micro=32, exact_accum, eval every 1,000 updates.

Training uses **40,000 optimizer updates** (not epochs) for the sweep and
**40,000** for each of the five final seeds, preserving the original
Transformer runner's training budget. Override with `SWEEP_UPDATES` and/or
`FINAL_UPDATES` to conduct a shorter trial. Example:

```bash
SWEEP_UPDATES=4000 FINAL_UPDATES=40000 \
  bash scripts/HiddenProbe/transformer/run_mnist_transformer_block1_sweep.sh
```

Sweep runs use seed 0. Configurations are ranked **exclusively by best
validation Kendall's tau**; test tau is *never used* to select hyperparameters.
The best config is rerun for seeds 0--4, each with independent
initialization and data order, using the same train/val/test split.
Test Kendall tau is reported as mean +/− **standard error** over five seeds.

Output is under
`checkpoints/transformer_hidden_layer_ablation/{mnist,agnews}/block{1,2}/`.

- `sweep_40000updates/sweep_results.csv` — validation results by config
- `sweep_40000updates/selected_config.json` — chosen hyperparameters
- `final_40000updates/five_seeds.csv` — seed-level validation and test tau
- `final_40000updates/summary.json` — mean, SD, and SE

Completed runs are reused if the checkpoint configuration matches; incomplete
ones are restarted. Old unmodified ProbeGen and full HiddenProbe results
are taken from the baseline table; no baseline retraining is performed.

## Tests

```bash
python -m unittest tests.test_transformer_hidden_block_ablation
```

Note: these scripts prepare GPU experiments; creating them in GitHub does not
run the sweeps on the lab cluster.
