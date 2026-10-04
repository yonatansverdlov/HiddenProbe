# Matched Model-J CIFAR100 ResNet variants

This directory recreates the **same 50-of-100 CIFAR100 meta-task** used by
ProbeX Model-J while changing only the target ResNet architecture.

For every source Model-J ResNet row we reuse:
- the original model index and meta train/val/test split;
- the same 50 selected CIFAR100 classes;
- seed, learning rate, scheduler, epochs, batch size, weight decay;
- random-crop and random-flip choices.

The generated target networks use the Hugging Face Microsoft ResNet family so
their state-dict layer names follow the same convention as the original
Model-J ResNet weights.

## Output

```
data/classification/modelj_cifar100_resnet/
  resnet18/
    train/model_idx_XXXX/
      model.safetensors
      metadata.json
      config.json
    val/...
    test/...
  resnet50/
    ...
```

A completed model is detected by both `model.safetensors` and a valid
`metadata.json` with `status=complete`. Re-running the generator therefore
skips completed models and retries interrupted ones.

## Install the extra dependencies

```bash
pip install -r scripts/model_j_resnet/requirements.txt
```

## Smoke test

Train only one matched source model:

```bash
bash scripts/model_j_resnet/run_resnet18.sh --limit 1
```

or pick exact Model-J indices:

```bash
bash scripts/model_j_resnet/run_resnet18.sh --model_idx 12 55 817
```

## Full datasets

```bash
bash scripts/model_j_resnet/run_resnet18.sh
bash scripts/model_j_resnet/run_resnet50.sh
```

By default this covers all source splits (701 train, 100 val, 201 test).

## Parallel jobs / sharding

The generator can be split into independent jobs. Example with four GPUs/jobs:

```bash
NUM_SHARDS=4 SHARD_ID=0 bash scripts/model_j_resnet/run_resnet18.sh
NUM_SHARDS=4 SHARD_ID=1 bash scripts/model_j_resnet/run_resnet18.sh
NUM_SHARDS=4 SHARD_ID=2 bash scripts/model_j_resnet/run_resnet18.sh
NUM_SHARDS=4 SHARD_ID=3 bash scripts/model_j_resnet/run_resnet18.sh
```

Every shard writes distinct model directories, so jobs do not share checkpoint
files. The same scheme works for ResNet50.

## Build a manifest after generation

```bash
python scripts/model_j_resnet/build_manifest.py \
  --architecture resnet18

python scripts/model_j_resnet/build_manifest.py \
  --architecture resnet50
```

This produces `manifest.jsonl` inside each architecture directory.

## Reading weights later

`models/modelj_local_dataset.py` contains `LocalModelJLayerDataset`.
It returns one named weight matrix plus the 100-dimensional multi-label vector
of classes used to train that target model.

Example:

```python
from models.modelj_local_dataset import (
    LocalModelJLayerDataset,
    available_weight_layers,
    list_model_files,
)

files = list_model_files(
    "data/classification/modelj_cifar100_resnet",
    "resnet18",
    "train",
)
layers = available_weight_layers(files[0])
ds = LocalModelJLayerDataset(
    root="data/classification/modelj_cifar100_resnet",
    architecture="resnet18",
    split="train",
    layer_name=layers[0],
)
X, y = ds[0]
```

## Matching details

CIFAR100 has 500 training images per class. We use a fixed stratified split of
425 train + 75 validation images per selected class, giving 21,250 training
images and exactly 333 batches/epoch at batch size 64, matching the step count
reported by the original Model-J ResNet cards.

The split is fixed by `--split_seed` and is shared across architectures.
This is deliberate: ResNet18 and ResNet50 see the exact same images for a
given Model-J row.

The target classifier keeps a **100-way CIFAR100 head** and original CIFAR100
label IDs even though each model only sees 50 classes. This matches the
parameter count and setup of the published Model-J ResNet models.

Scheduler names are copied from Model-J. For scheduler variants containing
`warmup`, the source metadata does not expose a warmup length, so this
pipeline uses `--warmup_ratio 0.1` by default. That choice is recorded in
every generated `metadata.json`.

ResNet101 is also accepted by the generator as a calibration option, although
the intended new datasets are ResNet18 and ResNet50.
