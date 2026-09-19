# Dataset setup scripts

This folder contains setup scripts for the 8 datasets used by the project.

Classification:
- `classification_cifar10.sh` — CIFAR-10 INRs (`siren_cifar_wts`, arXiv:2305.13546) -> `$DATA_ROOT/classification/cifar10_inr`

Regression (CNN zoos):
- `regression_mnist.sh` — MNIST-GS Small CNN Zoo -> `$DATA_ROOT/regression/mnist`
- `regression_fmnist.sh` — Fashion-MNIST-GS Small CNN Zoo -> `$DATA_ROOT/regression/fmnist`
- `regression_svhn.sh` — SVHN Small CNN Zoo -> `$DATA_ROOT/regression/svhn`
- `regression_cifar10_gs.sh` — CIFAR10-GS Small CNN Zoo -> `$DATA_ROOT/regression/cifar10_gs`
- `regression_cifar10_wp.sh` — CNN Wild Park -> `$DATA_ROOT/regression/cifar10_wp`

Regression (transformer zoos, Small Transformer Zoo on the HuggingFace Hub):
- `regression_mnist_transformer.sh` — MNIST-Transformers -> `$DATA_ROOT/mnist_transformer`
- `regression_agnews_transformer.sh` — AGNews-Transformers -> `$DATA_ROOT/ag_news_transformer`

`setup_all_datasets.sh` runs all eight in sequence. `_common.sh` holds the shared helpers (resumable
downloads, extraction with progress, restart-safe checks, and the Small-Transformer-Zoo helpers).
`DATA_ROOT` defaults to `<repo>/data`.

There is intentionally NO separate CIFAR10-Aug setup script. CIFAR10 and its augmented training configuration
use the same downloaded CIFAR10 INR release; `classification_cifar10.sh` builds both split files from it with
`build_nfn_cifar_splits.py` (`nfn_cifar_split.json` — augmented, `nfn_cifar_split_noaug.json` — non-augmented;
labels from the sub-directory name, split by net index).

`splits/` holds the CNN-zoo train / val / test splits shipped with the repository (`cnn_park_splits.json` for
CNN Wild Park, `gs_splits/*.csv` for the Small CNN Zoos); the setup scripts copy them next to the installed
data and the run scripts pass them by absolute path. CNN Wild Park is special: its Zenodo zip is never
extracted; `regression_cifar10_wp.sh` builds the flat-tensor CNN cache (`wp_cnn_cache/cnn_cache_{train,val,test}.pt`)
from the zip with `build_cnn_cache.py` (needs `torch` and `omegaconf`), which is what training reads.

The transformer scripts download the zoo's zip archive from the HuggingFace Hub, extract it under the wrapper
directory above, and then build the consolidated per-split cache with `python main.py transformer cache`
(epoch-75 checkpoints, seeded run-disjoint train/val/test split; cache files
`$DATA_ROOT/tpcache_<dataset>_{train,val,test}_s<seed>.pt`).

## Behavior

Every script is restart-safe:

1. If the final dataset is already present, it prints that the dataset exists and exits successfully.
2. If an archive has already been downloaded, it skips download and tries extraction directly.
3. There is NO pre-extraction archive validation (`tar -t`, etc.).
4. If extraction fails, the archive is treated as bad/incomplete, deleted, downloaded again, and extraction is retried once.
5. Extraction is performed in a temporary directory. Stale temporary extraction directories from interrupted runs are deleted on the next run.
6. Partial HTTP downloads use `.part` files and are resumed when possible.
7. If derived files (split files, caches) already exist, their build step is skipped.
8. Every significant operation prints a `[setup] ...` message.
9. Successful setup removes the archive by default to save disk space. Set `KEEP_ARCHIVES=1` to keep downloaded archives.

Run from the repository root, for example:

```bash
bash scripts/setup_data/regression_svhn.sh
```

The scripts can safely be called at the beginning of training scripts (the INR classification run scripts do so).
Environment knobs: `DATA_ROOT`, `DOWNLOAD_DIR`, `PYTHON` (interpreter used for the transformer cache build),
`SPLIT_SEED` (split seed of the transformer cache; the training scripts use split seed 0), `KEEP_ARCHIVES`.
