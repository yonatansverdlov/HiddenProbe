import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import kendalltau

from data import (
    INRDataset,
    NFNZooDataset,
    CNN_Park_ModelData,
    CIFAR10INRDataset,
    ModelNet40INRDataset,
    ModelNet40INRDataset,
    ModelJResNetProbeGenDataset,
    build_modelj_cifar100_class_to_idx,
)
from models.ProbeGen import ProbeGen


# Note: Structure of directories is similar to:
# https://github.com/mkofinas/neural-graphs.git


def set_seed(seed):
    """for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# region: setup
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"Total params:       {total:,}")
    print(f"Trainable params:   {trainable:,}")
    print(f"Non-trainable:      {non_trainable:,}")

    return total, trainable, non_trainable

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_seeds", type=int, default=1)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist_inr",
    choices=["mnist_inr", "fmnist_inr", "cifar10_inr", "nfn_cifar_inr", "nfn_cnn_zoo", "cnn_park", "cifar_inr", "tiny_imagenet_inr", "modelnet40_inr", 'model_j_resnet'],
)
# architecture
parser.add_argument("--n_tokens", type=int, default=128)
parser.add_argument("--d_hid", type=int, default=256)
parser.add_argument("--mixer_n_layers", type=int, default=6)
# cross-probe aggregator: "mlp" (default flatten->points_mixer) or "pat" (Probe-Activation Transformer)
parser.add_argument("--aggregator", type=str, default="mlp",
                    choices=["mlp", "pat", "deepsets", "set_transformer"],
                    help="Cross-probe aggregator: 'mlp' (flatten->head), 'pat' (Probe-Activation "
                         "Transformer, needs a vector generator), or the invariant baselines "
                         "'deepsets' / 'set_transformer'.")
parser.add_argument("--hidden_aggregator", type=str, default="set_transformer",
                    choices=["set_transformer", "deepsets"],
                    help="Per-hidden-layer neuron aggregator when --include_hidden_features (default "
                         "set_transformer = merged behavior; 'deepsets' = pre-merge baseline).")
parser.add_argument("--pat_d", type=int, default=256,
                    help="PAT transformer width (PATConfig.d). Shrink for a capacity-matched arm (e.g. 112 -> ~2.2M).")
parser.add_argument("--pat_n_blocks", type=int, default=6, help="PAT depth (PATConfig.n_blocks).")
parser.add_argument("--pat_rank", type=int, default=0,
                    help="PAT low-rank factorization rank (PATConfig.pat_rank). 0 = full. Small values "
                         "(e.g. 7) match Kahana's ~435k params while keeping width d and exact invariance.")
parser.add_argument("--st_rank", type=int, default=0,
                    help="set_transformer low-rank factorization rank (0 = full). Matches ~435k with with_hidden.")
parser.add_argument("--probe_pos_encoding", type=str2bool, default=False,
                    help="PAT STAGE C: per-probe-ROW positional embedding (breaks S_N probe invariance).")
parser.add_argument("--freeze_generator", type=str2bool, default=False,
                    help="Freeze the probe generator at its seeded init (fixed probe bank).")
parser.add_argument("--splits_path", type=str, default=None,
                    help="Override the per-dataset splits json (e.g. cifar10_splits_multiview.json).")
parser.add_argument("--dataset_dir", type=str, default=None,
                    help="Override the INR dataset root (default experiments/inr_classification/dataset). "
                         "Relative paths in the splits json resolve under this.")

# probe generator
parser.add_argument("--gen_type", type=str, default="linear_2_no_acts")
parser.add_argument("--gen_latent_z", type=int, default=32)
parser.add_argument("--generator_width", type=int, default=16)

# per-probe representation
parser.add_argument(
    "--include_hidden_features",
    type=str2bool,
    default=False,
    help=(
        "False: token=[x, f(x)]. True: append a permutation-invariant "
        "HiddenNeuronSetTransformer representation from every hidden Linear layer."
    ),
)
parser.add_argument(
    "--per_probe_mlp",
    type=str,
    default="none",
    choices=["none", "linear", "mlp", "mlp2", "mlp3"],
    help="Optional per-probe projection psi. 'mlp' is an alias for 'mlp2'.",
)
parser.add_argument(
    "--per_probe_mlp_width",
    type=int,
    default=None,
    help="Hidden width of psi. Defaults to --d_hid.",
)
parser.add_argument(
    "--per_probe_out_dim",
    type=int,
    default=None,
    help=(
        "Output width of psi before probe tokens are flattened. None uses "
        "--per_probe_mlp_width, or --d_hid when that is also None."
    ),
)
parser.add_argument(
    "--per_probe_init",
    type=str,
    default="standard",
    choices=["standard", "inductive"],
    help=(
        "Initialization of psi for INR hidden features. CNN hidden features "
        "always use an exact [0 | I] hidden/output projection before psi."
    ),
)
parser.add_argument(
    "--r_per_hidden",
    type=int,
    default=1,
    help=(
        "Output width of the HiddenNeuronSetTransformer for each hidden target layer. "
        "Used only when --include_hidden_features=true."
    ),
)
parser.add_argument(
    "--r_per_conv",
    type=int,
    default=1,
    help=(
        "Output width of the Set Transformer for each intermediate Conv layer. "
        "The current CNN path expects exactly three Conv2d layers."
    ),
)


# optimization
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--inference_only_benchmark", type=str, default="False")
parser.add_argument("--inference_only_batches", type=int, default=-1)
parser.add_argument("--inference_only_out_csv", type=str, default="")
parser.add_argument("--device", type=str, default="cuda")
# Optional mixed precision + grad clipping (opt-in; defaults preserve the fp32/no-clip behavior).
# Our protocol (PAT/baseline sweeps) uses: --amp bf16 --grad_clip 1.0
parser.add_argument("--amp", type=str, default="off", choices=["off", "bf16"],
                    help="Autocast dtype for forward/loss. 'off' = fp32 (default).")
parser.add_argument("--grad_clip", type=float, default=0.0,
                    help="Max grad norm (clip_grad_norm_) after backward; 0 = no clipping (default).")

parser.add_argument(
    "--scheduler",
    type=str,
    default="cosine",
    choices=["cosine", "plateau", "none"],
)

parser.add_argument(
    "--plateau_monitor",
    type=str,
    default="val_metric",
    choices=["val_metric", "val_loss"],
    help="For ReduceLROnPlateau: monitor validation metric or validation loss.",
)

parser.add_argument("--plateau_factor", type=float, default=0.7)
parser.add_argument("--plateau_patience", type=int, default=3)
parser.add_argument("--plateau_min_lr", type=float, default=1e-6)
# Optional linear LR warmup (used by PAT: ramp lr_warmup_start -> lr over lr_warmup_steps).
parser.add_argument("--lr_warmup_steps", type=int, default=0,
                    help="Linear LR warmup over this many optimizer steps (0 = off).")
parser.add_argument("--lr_warmup_start", type=float, default=1e-4,
                    help="Starting LR for warmup (ramps to --lr over --lr_warmup_steps).")
# [Arm-B] learned-probe recipe: freeze the generator for the first N steps (aggregator warms up on a
# fixed bank), then unfreeze via add_param_group ONTO THE EXISTING optimizer (never rebuilt -> main-group
# Adam moments preserved) at a reduced generator LR = lr * gen_lr_mult.
parser.add_argument("--gen_freeze_steps", type=int, default=0,
                    help="[Arm-B] freeze probe generator for this many steps, then unfreeze (0 = off).")
parser.add_argument("--gen_lr_mult", type=float, default=0.1,
                    help="[Arm-B] generator LR = lr * this, applied when the generator unfreezes.")
parser.add_argument("--hidden_agg_lr_mult", type=float, default=1.0,
                    help="LR of the per-hidden-layer aggregators = lr * this. <1 decouples them so their "
                         "gradients don't destabilize joint training (with_hidden fix; e.g. 0.05).")
parser.add_argument("--domain_tanh", type=str2bool, default=False,
                    help="[Arm-B] squash learned probe coords through tanh into the INR domain (-1,1)^d.")
parser.add_argument("--pat_token_scheme", type=str, default="lite", choices=["lite", "typed"],
                    help="PAT INR output container: 'lite' (broadcast V(f), bit-compat) | 'typed' (F_out token).")
parser.add_argument("--eval_test", action="store_true",
                    help="Contract: compute TEST only once at the end (best-val ckpt); skip inline test evals.")

args = parser.parse_args()

if args.num_seeds < 1:
    raise ValueError("--num_seeds must be >= 1")

torch.multiprocessing.set_sharing_strategy("file_system")

# endregion: setup


# region: data
def collate_fn(batch):
    models = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    if torch.is_tensor(labels[0]):
        labels = torch.stack(labels, dim=0)
    else:
        labels = torch.tensor(labels)

    return models, labels

# endregion: data


# Number of hidden Linear layers in each target-network family.
# The INR networks have two hidden Linear layers. CNN datasets can still use
# output-only ProbeGen, but do not support include_hidden_features=True.
_N_HIDDEN_TARGET_LAYERS = {
    "mnist_inr": 2,
    "fmnist_inr": 2,
    "cifar10_inr": 4,   # our multiview zoo: 2->32->32->32->32->3 SIREN (4 hidden layers)
    "nfn_cifar_inr": 2,  # NFN/NFT data: 2->32->32->3 SIREN (2 hidden layers) — matches train_np_e2e
    "cifar_inr": 2,
    "tiny_imagenet_inr": 4,
    "modelnet40_inr": 4,
    "nfn_cnn_zoo": 0,
    "cnn_park": 0,
    "model_j_resnet": 0,
}


def build_datasets(args):
    if args.dataset in ["mnist_inr", "fmnist_inr", "cifar10_inr"]:
        dataset_dir = args.dataset_dir or "experiments/inr_classification/dataset"
        # --splits_path overrides the default (e.g. cifar10_splits_multiview.json for the 5-view zoo).
        splits_path = args.splits_path or {
            "mnist_inr": 'mnist_splits.json',
            "fmnist_inr": 'fmnist_splits.json',
            "cifar10_inr": 'cifar10_splits.json',
        }[args.dataset]
        train_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="train")
        val_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="val")
        test_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="test")
        d_out = train_set.n_classes()
        # cifar10 INRs regress RGB (3 channels); mnist/fmnist INRs regress grayscale (1).
        models_c_out = 3 if args.dataset == "cifar10_inr" else 1
        models_c_in = 2
        is_regr = False
    elif args.dataset == "nfn_cifar_inr":       # NFN/NFT CIFAR-INR (2-hidden SIREN) — Kahana baseline data
        dataset_dir = args.dataset_dir or "experiments/inr_classification/dataset"
        splits_path = args.splits_path or "nfn_cifar_split.json"
        train_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="train")
        val_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="val")
        test_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="test")
        d_out = train_set.n_classes()
        models_c_out = 3
        models_c_in = 2
        is_regr = False
    elif args.dataset in ['cifar_inr']:
        dataset_dir = (
                "/home/yonatans/ProbeGen/experiments/"
                "inr_classification/dataset/cifar-inr")

        train_set = CIFAR10INRDataset(dataset_dir, split="train")
        val_set = CIFAR10INRDataset(dataset_dir, split="val")
        test_set = CIFAR10INRDataset(dataset_dir, split="test")

        d_out = 10
        models_c_in = 2
        models_c_out = 3
        is_regr = False

    elif args.dataset == "modelnet40_inr":
        dataset_dir = (
                "/home/yonatans/ProbeGen/experiments/"
                "inr_classification/dataset/modelnet40-inrs")

        train_set = ModelNet40INRDataset(dataset_dir, split="train")
        val_set = ModelNet40INRDataset(dataset_dir, split="val")
        test_set = ModelNet40INRDataset(dataset_dir, split="test")

        d_out = 40
        models_c_in = 3
        models_c_out = 1
        is_regr = False

    elif args.dataset == "model_j_resnet":
        dataset_dir = (
            "/home/yonatans/ProbeGen/experiments/"
            "inr_classification/dataset/model-j-resnet"
        )

        class_to_idx = (
            build_modelj_cifar100_class_to_idx(dataset_dir)
        )

        train_set = ModelJResNetProbeGenDataset(
            root_dir=dataset_dir,
            split="train",
            class_to_idx=class_to_idx,
        )

        val_set = ModelJResNetProbeGenDataset(
            root_dir=dataset_dir,
            split="val",
            class_to_idx=class_to_idx,
        )

        test_set = ModelJResNetProbeGenDataset(
            root_dir=dataset_dir,
            split="test",
            class_to_idx=class_to_idx,
        )

        # Generated probes are RGB images.
        models_c_in = 3

        # Each hidden ResNet returns 50 logits.
        models_c_out = 50

        # Predict 100 independent class-membership logits.
        d_out = 100
        is_regr = False

    elif args.dataset == "nfn_cnn_zoo":
        base_dataset_dir = "experiments/cnn_generalization/dataset"
        dataset_dir = os.path.join(base_dataset_dir, "small-zoo-cifar10")
        splits_path = os.path.join(base_dataset_dir, "nfn_cifar10_split.csv")
        train_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="train")
        val_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="val")
        test_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="test")
        d_out = 1
        models_c_out = 10
        models_c_in = 1
        is_regr = True

    elif args.dataset == "cnn_park":
        dataset_dir = "experiments/cnn_generalization/dataset"
        splits_path = "cnn_park_splits.json"
        train_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="train")
        val_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="val")
        test_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="test")
        d_out = 1
        models_c_out = 10
        models_c_in = 3
        is_regr = True

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_set, val_set, test_set, d_out, models_c_out, models_c_in, is_regr



@torch.no_grad()
def run_inference_only_benchmark(model, loader, device, max_batches=-1):
    """
    Pure inference benchmark:
      - no gradients
      - no optimizer
      - no training
      - no caching
    """
    import time
    import torch

    model.eval()

    n_batches = 0
    n_items = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, y = batch
        elif isinstance(batch, dict):
            inputs = batch
            y = None
        else:
            raise RuntimeError(f"Unsupported batch type in inference benchmark: {type(batch)}")

        if isinstance(inputs, dict):
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
            out = model(**inputs)
        else:
            if torch.is_tensor(inputs):
                inputs = inputs.to(device)
            out = model(inputs)

        # Force actual compute.
        if torch.is_tensor(out):
            _ = out.detach()

        if y is not None and torch.is_tensor(y):
            n_items += y.shape[0]
        else:
            n_items += 1

        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    total_sec = t1 - t0

    return {
        "inference_sec": total_sec,
        "n_batches": n_batches,
        "n_items": n_items,
        "sec_per_batch": total_sec / max(n_batches, 1),
        "sec_per_item": total_sec / max(n_items, 1),
    }


def run_one_seed(args, seed, exp_dir):
    set_seed(seed)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(exp_dir, exist_ok=True)

    # save args in exp dir
    with open(f"{exp_dir}/args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"run_seed: {seed}\n")

    # region: data
    train_set, val_set, test_set, d_out, models_c_out, models_c_in, is_regr = build_datasets(args)


    is_multilabel = args.dataset == "model_j_resnet"
    is_cnn_target = args.dataset in {
        "nfn_cnn_zoo",
        "cnn_park",
        "model_j_resnet",
    }
    print(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    # endregion: data

    # region: model
    n_hidden_target_layers = _N_HIDDEN_TARGET_LAYERS[args.dataset]

    # In the current datasets, regression means the target networks are CNNs.
    # Therefore include_hidden_features automatically selects Conv2d feature maps
    # for regression and Linear hidden activations for INR classification.
    use_conv_hidden_features = args.include_hidden_features and is_cnn_target

    n_conv_target_layers = 0
    if use_conv_hidden_features:
        sample_net, _ = train_set[0]
        n_conv_target_layers = sum(
            isinstance(module, nn.Conv2d)
            for module in sample_net.modules()
        )

        if n_conv_target_layers == 0:
            raise ValueError(
                f"Dataset {args.dataset!r} did not provide a target network "
                "with Conv2d layers."
            )

        # The dataset owns the target models; this local reference is no longer
        # needed after architecture inspection.
        del sample_net

    if (
        args.include_hidden_features
        and not use_conv_hidden_features
        and n_hidden_target_layers == 0
    ):
        raise ValueError(
            "--include_hidden_features=true requires hidden Linear layers for "
            f"non-regression targets; dataset {args.dataset!r} has none."
        )

    print(
        "Model config: "
        f"include_hidden_features={args.include_hidden_features}, "
        f"use_conv_hidden_features={use_conv_hidden_features}, "
        f"per_probe_mlp={args.per_probe_mlp}, "
        f"per_probe_mlp_width={args.per_probe_mlp_width}, "
        f"per_probe_out_dim={args.per_probe_out_dim}, "
        f"per_probe_init={args.per_probe_init}, "
        f"n_hidden_target_layers={n_hidden_target_layers}, "
        f"r_per_hidden={args.r_per_hidden}, "
        f"r_per_conv={args.r_per_conv}, "
        f"n_conv_target_layers={n_conv_target_layers}"
    )

    model = ProbeGen(
        n_tokens=args.n_tokens,
        d_hidden=args.d_hid,
        models_c_in=models_c_in,
        models_c_out=models_c_out,
        d_out=d_out,
        gen_type=args.gen_type,
        gen_latent_z=args.gen_latent_z,
        generator_width=args.generator_width,
        mixer_n_layers=args.mixer_n_layers,
        include_hidden_features=args.include_hidden_features,
        per_probe_mlp=args.per_probe_mlp,
        per_probe_mlp_width=args.per_probe_mlp_width,
        per_probe_out_dim=args.per_probe_out_dim,
        per_probe_init=args.per_probe_init,
        n_hidden_target_layers=n_hidden_target_layers,
        r_per_hidden=args.r_per_hidden,
        hidden_aggregator=args.hidden_aggregator,
        is_cnn_target=is_cnn_target,
        r_per_conv=args.r_per_conv,
        n_conv_target_layers=n_conv_target_layers,
        aggregator=args.aggregator,
        pat_d=args.pat_d,
        pat_n_blocks=args.pat_n_blocks,
        pat_rank=args.pat_rank,
        st_rank=args.st_rank,
        probe_pos_encoding=args.probe_pos_encoding,
        domain_tanh=args.domain_tanh,
        pat_token_scheme=args.pat_token_scheme,
    )

    # Probe-bank fingerprint: sha1 of the generated probe coordinates for this seed (parity check
    # for frozen-bank arms). Works for any aggregator; probe_source() = generator(input).
    import hashlib
    with torch.no_grad():
        _pc0 = model.generate_probes().detach().cpu().contiguous().float()   # post-domain_tanh coords
        _pc = _pc0.numpy().tobytes()
    _fp = hashlib.sha1(_pc).hexdigest()[:16]
    _inb = float(((_pc0 >= -1.0) & (_pc0 <= 1.0)).float().mean())
    print(f"[probe_bank_fp] seed={seed} fingerprint={_fp} coord_min={_pc0.min():.3f} "
          f"coord_max={_pc0.max():.3f} frac_in_domain={_inb:.3f} domain_tanh={args.domain_tanh}", flush=True)

    # Freeze the probe generator at its seeded init (fixed probe bank) when requested.
    if args.freeze_generator:
        n_frozen = 0
        if isinstance(model.probe_source.input, torch.nn.Parameter):
            model.probe_source.input.requires_grad_(False)
            n_frozen += model.probe_source.input.numel()
        for p in model.probe_source.generator.parameters():
            p.requires_grad_(False)
            n_frozen += p.numel()
        print(f"[freeze_generator] probe generator frozen at seeded init "
              f"({n_frozen:,} params, fixed probe bank).", flush=True)

    total_params, trainable_params, _ = count_params(model)

    meta = {
        "seed": seed,
        "dataset": args.dataset,
        "include_hidden_features": args.include_hidden_features,
        "use_conv_hidden_features": use_conv_hidden_features,
        "per_probe_mlp": args.per_probe_mlp,
        "per_probe_mlp_width": args.per_probe_mlp_width,
        "per_probe_out_dim": args.per_probe_out_dim,
        "per_probe_init": args.per_probe_init,
        "n_hidden_target_layers": n_hidden_target_layers,
        "r_per_hidden": args.r_per_hidden,
        "r_per_conv": args.r_per_conv,
        "n_conv_target_layers": n_conv_target_layers,
        "n_tokens": args.n_tokens,
        "d_hid": args.d_hid,
        "mixer_n_layers": args.mixer_n_layers,
        "aggregator": args.aggregator,
        "pat_d": args.pat_d,
        "pat_n_blocks": args.pat_n_blocks,
        "pat_rank": args.pat_rank,
        "probe_pos_encoding": args.probe_pos_encoding,
        "freeze_generator": args.freeze_generator,
        "probe_bank_fp": _fp,
        "splits_path": args.splits_path,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_warmup_start": args.lr_warmup_start,
        "gen_type": args.gen_type,
        "gen_latent_z": args.gen_latent_z,
        "generator_width": args.generator_width,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "plateau_monitor": args.plateau_monitor,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_min_lr": args.plateau_min_lr,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    model = model.float()
    model = model.to(device)

    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()

        loss_sum = 0.0
        correct = 0
        total = 0
        predicted = []
        gt = []

        for batch in loader:
            label = batch[1].to(device)

            # freeze target nets: they are fixed data, never optimized -> no need to build/backprop
            # their autograd graph (grad to a learned probe generator flows through x, not target params).
            [net.requires_grad_(False).to(device) for net in batch[0]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.amp == "bf16")):
                out = model(nets=batch[0])

            if is_regr:
                label = label.float().unsqueeze(1)

                batch_loss = F.mse_loss(
                    out,
                    label,
                    reduction="sum",
                )

                loss_sum += batch_loss.item()
                total += label.numel()

                predicted.extend(
                    out.detach().flatten().cpu().tolist()
                )
                gt.extend(
                    label.detach().flatten().cpu().tolist()
                )

            elif is_multilabel:
                label = label.float()

                if out.shape != label.shape:
                    raise RuntimeError(
                        "Multilabel shape mismatch: "
                        f"output={tuple(out.shape)}, "
                        f"target={tuple(label.shape)}"
                    )

                batch_loss = (
                    F.binary_cross_entropy_with_logits(
                        out,
                        label,
                        reduction="sum",
                    )
                )

                # Same accuracy used by the ProbeX Model-J code:
                # sigmoid threshold 0.5 over all 100 decisions.
                pred = (torch.sigmoid(out) > 0.5).float()

                loss_sum += batch_loss.item()
                correct += pred.eq(label).sum().item()
                total += label.numel()

                predicted.extend(
                    pred.detach().flatten().cpu().tolist()
                )
                gt.extend(
                    label.detach().flatten().cpu().tolist()
                )

            else:
                batch_loss = F.cross_entropy(
                    out,
                    label,
                    reduction="sum",
                )

                pred = out.argmax(dim=1)

                loss_sum += batch_loss.item()
                correct += pred.eq(label).sum().item()
                total += label.shape[0]

                predicted.extend(
                    pred.detach().cpu().tolist()
                )
                gt.extend(
                    label.detach().cpu().tolist()
                )

            [net.to("cpu") for net in batch[0]]
            [net.zero_grad() for net in batch[0]]

        predicted = np.asarray(predicted)
        gt = np.asarray(gt)

        model.train()

        res_d = {
            "avg_loss": loss_sum / max(total, 1),
            "predicted": predicted,
            "gt": gt,
        }

        if is_regr:
            res_d["kendalltau"] = (
                kendalltau(predicted, gt).statistic
            )
        else:
            res_d["avg_acc"] = correct / max(total, 1)

        return res_d

    if str(args.inference_only_benchmark).lower() == "true":
        bench = run_inference_only_benchmark(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.inference_only_batches,
        )

        bench.update({
            "dataset": args.dataset,
            "seed": seed,
            "n_tokens": args.n_tokens,
            "gen_type": getattr(args, "gen_type", ""),
            "include_hidden_features": args.include_hidden_features,
            "use_conv_hidden_features": use_conv_hidden_features,
            "r_per_conv": args.r_per_conv,
                "per_probe_mlp": getattr(args, "per_probe_mlp", ""),
        })

        if args.inference_only_out_csv:
            from pathlib import Path

            out_csv = Path(args.inference_only_out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)

            row = pd.DataFrame([bench])
            if out_csv.exists():
                row.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                row.to_csv(out_csv, index=False)

        print("INFERENCE_ONLY_BENCHMARK:", bench)
        return bench

    # [Arm-B] partition generator params so they can unfreeze mid-run at a reduced LR. Group 0 excludes
    # them; they are add_param_group'd onto THIS optimizer at gen_freeze_steps (never rebuilt).
    arm_b = (not args.freeze_generator) and args.gen_freeze_steps > 0
    gen_params = []
    if arm_b:
        if isinstance(model.probe_source.input, torch.nn.Parameter):
            gen_params.append(model.probe_source.input)
        gen_params += list(model.probe_source.generator.parameters())
        for _p in gen_params:
            _p.requires_grad_(False)                     # frozen during the warmup window
        _gen_ids = {id(p) for p in gen_params}
        base_params = [p for p in model.parameters() if id(p) not in _gen_ids and p.requires_grad]
        print(f"[arm_b] generator frozen for first {args.gen_freeze_steps} steps "
              f"({sum(p.numel() for p in gen_params):,} params); will unfreeze at "
              f"gen_lr={args.lr * args.gen_lr_mult:.2e}, gen_min_lr="
              f"{args.plateau_min_lr * args.gen_lr_mult:.2e}", flush=True)
    else:
        base_params = [p for p in model.parameters() if p.requires_grad]
    # [hidden-agg fix] the 4 HiddenNeuronSetTransformer aggregators destabilize joint training at full LR
    # (diagnosed: detaching them lets the model train 0.116->0.207; global low LR does NOT help). Decouple
    # them into a low-LR group so they contribute gently instead of poisoning the optimization.
    hid_params = []
    if getattr(args, "hidden_agg_lr_mult", 1.0) != 1.0 and getattr(model, "hidden_aggregators", None) is not None:
        hid_params = [p for p in model.hidden_aggregators.parameters() if p.requires_grad]
        _hid_ids = {id(p) for p in hid_params}
        base_params = [p for p in base_params if id(p) not in _hid_ids]
        print(f"[hidden_agg] {sum(p.numel() for p in hid_params):,} aggregator params -> low-LR group "
              f"lr={args.lr * args.hidden_agg_lr_mult:.2e} ({args.hidden_agg_lr_mult}x)", flush=True)
    optimizer = torch.optim.Adam(params=base_params, lr=args.lr, weight_decay=args.wd)
    if hid_params:
        optimizer.add_param_group({"params": hid_params, "lr": args.lr * args.hidden_agg_lr_mult,
                                   "weight_decay": args.wd})
    gen_added = not arm_b                                  # nothing to add later when not Arm-B

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
        )

    elif args.scheduler == "plateau":
        if args.plateau_monitor == "val_loss":
            plateau_mode = "min"
        else:
            plateau_mode = "max"

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    elif args.scheduler == "none":
        scheduler = None

    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


    if is_regr:


        criterion = nn.MSELoss()


    elif is_multilabel:


        criterion = nn.BCEWithLogitsLoss()


    else:


        criterion = nn.CrossEntropyLoss()
    global_step = 0

    logging = pd.DataFrame(columns=[
        "exp_name",
        "epoch",
        "global_step",
        "lr",
        "train_loss",
        "val_loss",
        "test_loss",
        "val_acc",
        "test_acc",
        "val_kendalltau",
        "test_kendalltau",
        "is_best",
    ])

    best_val_metric = -float("inf")
    best_epoch = None
    best_global_step = None
    best_ckpt_path = os.path.join(exp_dir, "best_checkpoint.pth")

    epoch_iter = tqdm(range(args.epochs), ncols=150)

    iters_train_loss = 0.0
    iters_since_eval = 0

    def current_lr():
        return optimizer.param_groups[0]["lr"]

    def evaluate_and_log(epoch, global_step, train_loss, save_if_best=True):
        nonlocal best_val_metric, best_epoch, best_global_step

        val_results_dict = evaluate(model, val_loader, device)
        if args.eval_test:              # contract: TEST computed once at the end (best-val ckpt) only
            test_results_dict = {"avg_loss": float("nan"), "avg_acc": float("nan"),
                                 "kendalltau": float("nan")}
        else:
            test_results_dict = evaluate(model, test_loader, device)

        # [Arm-B] per-eval probe-coordinate diagnostics (drift/collapse watch).
        with torch.no_grad():
            _c = model.generate_probes().detach().float()
            _frac = float(((_c >= -1.0) & (_c <= 1.0)).float().mean())
            print(f"[coord_stats] step={global_step} min={_c.min():.3f} max={_c.max():.3f} "
                  f"mean={_c.mean():.3f} std={_c.std():.4f} frac_in_domain={_frac:.3f}", flush=True)

        val_loss = val_results_dict["avg_loss"]
        test_loss = test_results_dict["avg_loss"]

        if is_regr:
            val_main_metric = val_results_dict["kendalltau"]
            test_main_metric = test_results_dict["kendalltau"]

            log_row = {
                "exp_name": args.exp_name,
                "epoch": epoch,
                "global_step": global_step,
                "lr": current_lr(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "val_acc": None,
                "test_acc": None,
                "val_kendalltau": val_main_metric,
                "test_kendalltau": test_main_metric,
            }

            metric_name = "val_kendalltau"

        else:
            val_main_metric = val_results_dict["avg_acc"]
            test_main_metric = test_results_dict["avg_acc"]

            log_row = {
                "exp_name": args.exp_name,
                "epoch": epoch,
                "global_step": global_step,
                "lr": current_lr(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "val_acc": val_main_metric,
                "test_acc": test_main_metric,
                "val_kendalltau": None,
                "test_kendalltau": None,
            }

            metric_name = "val_acc"

        # Step ReduceLROnPlateau on the validation-time signal — NOT every
        # training iteration. Metric chosen by --plateau_monitor.
        if args.scheduler == "plateau" and scheduler is not None:
            if args.plateau_monitor == "val_loss":
                scheduler.step(val_loss)
            else:
                scheduler.step(val_main_metric)

        is_best = False

        if save_if_best and val_main_metric > best_val_metric:
            is_best = True
            best_val_metric = val_main_metric
            best_epoch = epoch
            best_global_step = global_step

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_metric": best_val_metric,
                "metric_name": metric_name,
                "args": vars(args),
                "seed": seed,
            }, best_ckpt_path)

            print(
                f"\nNew best checkpoint saved: "
                f"epoch={epoch}, step={global_step}, {metric_name}={best_val_metric:.6f}"
            )

        log_row["is_best"] = is_best

        return log_row

    # ---------------------------------------------------------------------
    # Training loop: unchanged
    # ---------------------------------------------------------------------
    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            # Linear LR warmup: ramp lr_warmup_start -> lr over the first lr_warmup_steps steps,
            # then the plateau/cosine scheduler owns the LR. Aggregator-neutral (used by PAT).
            if args.lr_warmup_steps > 0 and global_step < args.lr_warmup_steps:
                warmup_lr = args.lr_warmup_start + (args.lr - args.lr_warmup_start) * (
                    global_step / args.lr_warmup_steps)
                optimizer.param_groups[0]["lr"] = warmup_lr        # base group only (gen group unadded)

            # [Arm-B] unfreeze the generator once the freeze window elapses: add_param_group onto the
            # EXISTING optimizer at reduced LR (main-group Adam moments preserved); extend the plateau
            # floor list so ReduceLROnPlateau stays index-aligned.
            if not gen_added and global_step >= args.gen_freeze_steps:
                for _p in gen_params:
                    _p.requires_grad_(True)
                gen_lr = args.lr * args.gen_lr_mult
                optimizer.add_param_group({"params": gen_params, "lr": gen_lr,
                                           "weight_decay": args.wd})
                if scheduler is not None and args.scheduler == "plateau":
                    scheduler.min_lrs.append(args.plateau_min_lr * args.gen_lr_mult)
                gen_added = True
                print(f"[arm_b] unfroze generator at step {global_step}: gen_lr={gen_lr:.2e}", flush=True)

            label = batch[1].to(device)

            # This is okay if x is an nn.Module / custom object with in-place .to().
            # If x is a plain Tensor, use: batch[0] = [x.to(device) for x in batch[0]]
            [x.to(device) for x in batch[0]]

            inputs = {"nets": batch[0]}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.amp == "bf16")):
                out = model(**inputs)

                if is_regr:
                    label = label.float().unsqueeze(1)
                elif is_multilabel:
                    label = label.float()

                    if out.shape != label.shape:
                        raise RuntimeError(
                            "Multilabel shape mismatch: "
                            f"output={tuple(out.shape)}, "
                            f"target={tuple(label.shape)}"
                        )

                loss = criterion(out, label)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # MODEL_J_DEBUG_AFTER_BACKWARD
            if args.dataset == "model_j_resnet":
                if not hasattr(model, "_modelj_debug_counter"):
                    model._modelj_debug_counter = 0

                model._modelj_debug_counter += 1

            optimizer.step()

            if scheduler is not None and args.scheduler == "cosine":
                scheduler.step()
            # plateau scheduler is stepped inside evaluate_and_log on val metric.

            [x.to("cpu") for x in batch[0]]
            [x.zero_grad() for x in batch[0]]

            iters_train_loss += loss.item()
            iters_since_eval += 1

            if global_step % args.eval_every == 0 and global_step > 0:

                train_loss = iters_train_loss / max(1, iters_since_eval)

                log_row = evaluate_and_log(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    save_if_best=True,
                )

                logging = pd.concat(
                    [logging, pd.DataFrame(log_row, index=[0])],
                    ignore_index=True,
                )
                logging.to_csv(f"{exp_dir}/log.csv", index=False)

                iters_train_loss = 0.0
                iters_since_eval = 0

                torch.save(model.state_dict(), os.path.join(exp_dir, "intermediate_checkpoint.pth"))

            global_step += 1


    # Save last model
    torch.save(model.state_dict(), os.path.join(exp_dir, "epoch_last.pth"))


    # Final evaluation, in case the last evaluation did not happen exactly at the end
    print("\nRunning final evaluation...")

    final_train_loss = iters_train_loss / max(1, iters_since_eval)

    final_log_row = evaluate_and_log(
        epoch=args.epochs - 1,
        global_step=global_step,
        train_loss=final_train_loss,
        save_if_best=True,
    )

    logging = pd.concat(
        [logging, pd.DataFrame(final_log_row, index=[0])],
        ignore_index=True,
    )
    logging.to_csv(f"{exp_dir}/log.csv", index=False)


    # Load best checkpoint and print final test result of best model
    print("\nLoading best checkpoint and evaluating on test set...")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Final probe-bank fingerprint (of the best-val checkpoint's learned coords).
    with torch.no_grad():
        _fpc = model.generate_probes().detach().cpu().contiguous().float()
    _fp_final = hashlib.sha1(_fpc.numpy().tobytes()).hexdigest()[:16]
    _fin_frac = float(((_fpc >= -1.0) & (_fpc <= 1.0)).float().mean())
    print(f"[probe_bank_fp_final] seed={seed} fingerprint={_fp_final} coord_min={_fpc.min():.3f} "
          f"coord_max={_fpc.max():.3f} frac_in_domain={_fin_frac:.3f}", flush=True)

    best_val_results = evaluate(model, val_loader, device)
    best_test_results = evaluate(model, test_loader, device)

    print("\n========== Best checkpoint results ==========")
    print(f"Seed: {seed}")
    print(f"Best epoch: {ckpt['epoch']}")
    print(f"Best global step: {ckpt['global_step']}")
    print(f"Selection metric: {ckpt['metric_name']}")
    print(f"Best validation metric: {ckpt['best_val_metric']:.6f}")

    print(f"Best val loss: {best_val_results['avg_loss']:.6f}")
    print(f"Best test loss: {best_test_results['avg_loss']:.6f}")

    if is_regr:
        print(f"Best val kendalltau: {best_val_results['kendalltau']:.6f}")
        print(f"Best test kendalltau: {best_test_results['kendalltau']:.6f}")
        best_val_main = best_val_results["kendalltau"]
        best_test_main = best_test_results["kendalltau"]
        metric_name = "kendalltau"
    else:
        print(f"Best val acc: {best_val_results['avg_acc']:.6f}")
        print(f"Best test acc: {best_test_results['avg_acc']:.6f}")
        best_val_main = best_val_results["avg_acc"]
        best_test_main = best_test_results["avg_acc"]
        metric_name = "acc"

    print("============================================\n")


    result = {
        "seed": seed,
        "exp_dir": exp_dir,
        "best_epoch": ckpt["epoch"],
        "best_global_step": ckpt["global_step"],
        "selection_metric": ckpt["metric_name"],
        "best_val_metric": ckpt["best_val_metric"],
        "best_val_loss": best_val_results["avg_loss"],
        "best_test_loss": best_test_results["avg_loss"],
        "metric_name": metric_name,
        "best_val_main": best_val_main,
        "best_test_main": best_test_main,
    }

    return result


def mean_std(values):
    values = np.array(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def main():
    base_exp_dir = f"experiments/{args.dataset}/runs/{args.exp_name}"
    os.makedirs(base_exp_dir, exist_ok=True)

    results = []

    for seed_idx in range(args.num_seeds):
        run_seed = args.seed + seed_idx

        if args.num_seeds == 1:
            exp_dir = base_exp_dir
        else:
            exp_dir = os.path.join(base_exp_dir, f"seed_{run_seed}")

        print("\n" + "=" * 80)
        print(f"Running seed {run_seed} ({seed_idx + 1}/{args.num_seeds})")
        print(f"Experiment directory: {exp_dir}")
        print("=" * 80 + "\n")

        result = run_one_seed(args, run_seed, exp_dir)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(base_exp_dir, "seeds_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    test_metric_mean, test_metric_std = mean_std(summary_df["best_test_main"].values)
    val_metric_mean, val_metric_std = mean_std(summary_df["best_val_main"].values)
    test_loss_mean, test_loss_std = mean_std(summary_df["best_test_loss"].values)
    val_loss_mean, val_loss_std = mean_std(summary_df["best_val_loss"].values)

    metric_name = summary_df["metric_name"].iloc[0]

    print("\n========== Seeds summary ==========")
    print(f"num_seeds: {args.num_seeds}")
    print(f"seeds: {summary_df['seed'].tolist()}")
    print(f"summary csv: {summary_csv}")
    print(f"Best val {metric_name}:  mean={val_metric_mean:.6f}, std={val_metric_std:.6f}")
    print(f"Best test {metric_name}: mean={test_metric_mean:.6f}, std={test_metric_std:.6f}")
    print(f"Best val loss:  mean={val_loss_mean:.6f}, std={val_loss_std:.6f}")
    print(f"Best test loss: mean={test_loss_mean:.6f}, std={test_loss_std:.6f}")
    print("===================================\n")


if __name__ == "__main__":
    main()
