# HiddenProbe

HiddenProbe learns representations of neural networks by evaluating learned probes on a frozen target network and using responses from its hidden layers for downstream prediction. This repository contains the implementation and experiment scripts for INR classification and model accuracy prediction.

<p align="center">
  <img src="figs/hiddenprobe_overview.png" width="95%" alt="HiddenProbe overview">
</p>

## Installation

```bash
git clone https://github.com/yonatansverdlov/HiddenProbe.git
cd HiddenProbe

conda create -n hiddenprobe python=3.11
conda activate hiddenprobe
pip install -r requirements.txt
```

Run all commands from the repository root. Each experiment script automatically downloads or prepares the required dataset before training.

Every listed HiddenProbe `run_*.sh` experiment has a corresponding ProbeGen runner at the same relative path under `scripts/ProbeGen/`, including all five MNIST-/AGNews-Transformer thresholds. The ProbeGen runners preserve their own reference-branch hyperparameters and seed settings. For example, run `./scripts/ProbeGen/classification/run_mnist_inr.sh` or `./scripts/ProbeGen/regression/run_cifar10_wp_regression.sh`.

ProbeGen Wild Park loads the **same** prebuilt `cnn_cache_{train,val,test}.pt` files as HiddenProbe, from `$PGH_WP_CACHE` or `data/regression/cifar10_wp/wp_cnn_cache/`; setup builds these directly from the archive without extracting individual checkpoints. The ProbeGen augmented CIFAR-10 run preserves its reference configuration of 20 extra augmentations.

## Running the experiments

### INR classification

**MNIST**

```bash
./scripts/HiddenProbe/classification/run_mnist_inr.sh
```

**Fashion-MNIST**

```bash
./scripts/HiddenProbe/classification/run_fmnist_inr.sh
```

**CIFAR-10**

```bash
./scripts/HiddenProbe/classification/run_cifar10_inr_nonaug.sh
```

**CIFAR-10 Augmented**

```bash
./scripts/HiddenProbe/classification/run_cifar10_inr_aug.sh
```

### Model accuracy prediction

**MNIST**

```bash
./scripts/HiddenProbe/regression/run_mnist_regression.sh
```

**Fashion-MNIST**

```bash
./scripts/HiddenProbe/regression/run_fmnist_regression.sh
```

**SVHN**

```bash
./scripts/HiddenProbe/regression/run_svhn_regression.sh
```

**CIFAR-10 Gray Scale**

```bash
./scripts/HiddenProbe/regression/run_cifar10_gs_regression.sh
```

**CIFAR-10 Wild Park**

```bash
./scripts/HiddenProbe/regression/run_cifar10_wp_regression.sh
```

### Transformer accuracy prediction

For MNIST-Transformers and AGNews-Transformers we report five accuracy-threshold settings. For each threshold, target models below the threshold are removed first, and the remaining population is then split into 70% train, 15% validation, and 15% test. Each script runs five training seeds.

**MNIST-Transformers**

```bash
./scripts/HiddenProbe/regression/run_mnist_transformer_thresh0.sh
./scripts/HiddenProbe/regression/run_mnist_transformer_thresh20.sh
./scripts/HiddenProbe/regression/run_mnist_transformer_thresh40.sh
./scripts/HiddenProbe/regression/run_mnist_transformer_thresh60.sh
./scripts/HiddenProbe/regression/run_mnist_transformer_thresh80.sh
```

**AGNews-Transformers**

```bash
./scripts/HiddenProbe/regression/run_agnews_transformer_thresh0.sh
./scripts/HiddenProbe/regression/run_agnews_transformer_thresh20.sh
./scripts/HiddenProbe/regression/run_agnews_transformer_thresh40.sh
./scripts/HiddenProbe/regression/run_agnews_transformer_thresh60.sh
./scripts/HiddenProbe/regression/run_agnews_transformer_thresh80.sh
```

## Repository structure

```text
main.py                         Main experiment entry point
models/                         HiddenProbe models and trainers
scripts/HiddenProbe/            HiddenProbe experiments and sweeps
scripts/ProbeGen/               ProbeGen experiment runners
scripts/setup_data/             Automatic dataset download and preparation
```

## Citation

Coming soon.
