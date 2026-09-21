# HiddenProbe

HiddenProbe learns representations of neural networks by evaluating learned probes on a frozen target network and using responses from its hidden layers for downstream prediction. This repository contains the implementation and experiment scripts for INR classification and model accuracy prediction.

<p align="center">
  <img src="assets/hiddenprobe_overview.png" width="95%" alt="HiddenProbe overview">
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
scripts/HiddenProbe/            Final experiment runners
scripts/setup_data/             Automatic dataset download and preparation
sweeps/                         Hyperparameter sweeps
```

## Citation

Coming soon.
