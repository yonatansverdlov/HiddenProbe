from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.lowrank import make_linear


class LowRankEncoderLayer(nn.Module):
    """Pre-norm (norm_first) transformer encoder layer with low-rank q/k/v/out + FF, matching
    nn.TransformerEncoderLayer(activation='relu', norm_first=True) semantics but with every big
    Linear factorized as U@V (make_linear). Shared across tokens -> permutation-equivariant; the
    aggregator's mean-pool then makes it invariant (low-rank preserves both)."""

    def __init__(self, d_model, nhead, dim_ff, rank, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead, self.dh = nhead, d_model // nhead
        self.q = make_linear(d_model, d_model, rank=rank)
        self.k = make_linear(d_model, d_model, rank=rank)
        self.v = make_linear(d_model, d_model, rank=rank)
        self.o = make_linear(d_model, d_model, rank=rank)
        self.l1 = make_linear(d_model, dim_ff, rank=rank)
        self.l2 = make_linear(dim_ff, d_model, rank=rank)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _sa(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.nhead, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.nhead, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.nhead, self.dh).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:                              # kpm: True = PAD -> SDPA bool mask True = KEEP
            attn_mask = (~key_padding_mask)[:, None, None, :]         # [B,1,1,T], broadcast over heads/queries
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, src_key_padding_mask=None):
        x = x + self.drop(self._sa(self.n1(x), src_key_padding_mask))  # pre-norm self-attn
        x = x + self.drop(self.l2(self.drop(F.relu(self.l1(self.n2(x))))))  # pre-norm FF
        return x


class DeepLinearGenerator(nn.Module):
    """Probe-image generator used by the original ProbeGen implementation."""

    def __init__(
        self,
        out_channels: int,
        latent_dim: int = 100,
        width_mult: int = 16,
        n_layers: int = 6,
    ):
        super().__init__()

        if n_layers == 6:
            layers = [
                nn.ConvTranspose2d(latent_dim, width_mult * 8, 4, 1, 0),
                nn.ConvTranspose2d(width_mult * 8, width_mult * 4, 4, 2, 1),
                nn.ConvTranspose2d(width_mult * 4, width_mult * 2, 4, 2, 1),
                nn.ConvTranspose2d(width_mult * 2, width_mult * 2, 3, 1, 1),
                nn.ConvTranspose2d(width_mult * 2, width_mult, 4, 2, 1),
                nn.ConvTranspose2d(width_mult, out_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        elif n_layers == 5:
            layers = [
                nn.ConvTranspose2d(latent_dim, width_mult * 8, 4, 1, 0),
                nn.ConvTranspose2d(width_mult * 8, width_mult * 4, 4, 2, 1),
                nn.ConvTranspose2d(width_mult * 4, width_mult * 2, 4, 2, 1),
                nn.ConvTranspose2d(width_mult * 2, width_mult, 4, 2, 1),
                nn.ConvTranspose2d(width_mult, out_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        elif n_layers == 0:
            layers = [nn.Tanh()]
        else:
            raise ValueError(
                f"DeepLinearGenerator supports n_layers in {{0, 5, 6}}, "
                f"got {n_layers}."
            )

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class HiddenNeuronSetTransformer(nn.Module):
    """
    Permutation-invariant encoder over the hidden neurons of one Linear layer.

    Input:  [T, H]
    Output: [T, out_dim]
    """

    def __init__(
        self,
        d_model: int = 64,
        out_dim: int = 1,
        num_heads: int = 2,
        num_layers: int = 1,
        pool: str = "mean",
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        if pool not in {"mean", "sum", "max"}:
            raise ValueError(f"Unknown pool: {pool}")

        self.pool = pool
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h: Tensor) -> Tensor:
        if h.ndim != 2:
            raise ValueError(
                f"Expected hidden activations [T, H], got {tuple(h.shape)}."
            )

        x = self.input_proj(h.unsqueeze(-1))  # [T, H, d_model]
        x = self.encoder(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        else:
            x = x.max(dim=1).values

        return self.output_proj(x)


class HiddenNeuronDeepSets(nn.Module):
    """Per-hidden-layer DeepSets aggregator (ported verbatim from the pre-merge baseline family).

    Same interface as HiddenNeuronSetTransformer: [T, H] -> [T, out_dim], permutation-invariant over
    the H hidden neurons. Selected by --hidden_aggregator deepsets.
    """

    def __init__(self, in_dim=1, hidden_dim=64, out_dim=8, pool="mean"):
        super().__init__()

        self.pool = pool

        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.phi:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        for i, m in enumerate(self.rho):
            if isinstance(m, nn.Linear):
                if i < len(self.rho) - 1:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h):
        """h: [T, H] -> [T, out_dim]."""
        h = h.unsqueeze(-1)          # [T, H, 1]
        z = self.phi(h)              # [T, H, hidden_dim]
        if self.pool == "mean":
            z = z.mean(dim=1)
        elif self.pool == "sum":
            z = z.sum(dim=1)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")
        return self.rho(z)           # [T, out_dim]


class DeepSetsEncoder(nn.Module):
    """Cross-probe DeepSets aggregator (ported verbatim from the pre-merge baseline family).

    Maps per-probe tokens [B, T, in_dim] -> [B, out_dim] via a shared phi MLP, mean/sum pool
    over the probe axis (permutation-invariant), then rho. Used for --aggregator deepsets.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, pool="mean"):
        super().__init__()

        self.pool = pool

        phi_layers = []
        last_dim = in_dim
        for _ in range(n_layers):
            phi_layers.append(nn.Linear(last_dim, hidden_dim))
            phi_layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming for ReLU-followed Linears, Xavier for the output Linear, zero biases."""
        for m in self.phi:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.rho[0].weight, nonlinearity="relu")
        if self.rho[0].bias is not None:
            nn.init.zeros_(self.rho[0].bias)
        nn.init.xavier_uniform_(self.rho[2].weight)
        if self.rho[2].bias is not None:
            nn.init.zeros_(self.rho[2].bias)

    def forward(self, z):
        """z: [B, T, D] -> [B, out_dim]."""
        h = self.phi(z)
        if self.pool == "sum":
            h = h.sum(dim=1)
        elif self.pool == "mean":
            h = h.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")
        return self.rho(h)


class InvariantProbeTransformer(nn.Module):
    """Cross-probe set-transformer aggregator (ported verbatim from the pre-merge baseline family).

    No positional encoding -> permutation-equivariant over probes; pooling makes it invariant.
    Maps [B, T, in_dim] -> [B, out_dim]. Used for --aggregator set_transformer.
    """

    def __init__(
        self,
        in_dim,
        d_model=128,
        out_dim=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.0,
        pool="mean",
        rank=0,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.pool = pool
        self.rank = int(rank)

        self.input_proj = make_linear(in_dim, d_model, rank=self.rank)

        if self.rank > 0:
            # low-rank encoder stack (custom pre-norm layers) — param-matched ablation
            self.encoder = nn.ModuleList([
                LowRankEncoderLayer(d_model, num_heads, dim_feedforward, self.rank, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            make_linear(d_model, d_model, rank=self.rank),
            nn.ReLU(),
            make_linear(d_model, out_dim, rank=self.rank),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # only re-init plain Linears (LowRankLinear keeps its default factored init)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        for m in self.output_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        """z: [B, T, in_dim] -> [B, out_dim]."""
        h = self.input_proj(z)
        if self.rank > 0:
            for layer in self.encoder:      # low-rank ModuleList
                h = layer(h)
        else:
            h = self.encoder(h)
        if self.pool == "mean":
            h = h.mean(dim=1)
        elif self.pool == "sum":
            h = h.sum(dim=1)
        elif self.pool == "max":
            h = h.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {self.pool}")
        return self.output_proj(h)


class ProbeSource(nn.Module):
    """Own the trainable/fixed probe tensor and its generator."""

    def __init__(self, initial_input: Tensor, generator: nn.Module, trainable: bool):
        super().__init__()
        self.generator = generator

        if trainable:
            self.input = nn.Parameter(initial_input)
        else:
            self.register_buffer("input", initial_input)

    def forward(self) -> Tensor:
        return self.generator(self.input)


def build_probe_source(
    *,
    n_tokens: int,
    models_c_in: int,
    gen_type: str,
    gen_latent_z: int,
    generator_width: int,
) -> ProbeSource:
    """
    Build probes while preserving the original implementation's RNG order.

    The original code first sampled a latent image tensor for every optimized
    generator and then overwrote it in deep_linear_0/linear_0/linear_2. Those
    seemingly redundant samples are intentionally retained so existing seeds
    initialize identically.
    """
    initial: Optional[Tensor] = None
    if "no_opt" not in gen_type:
        initial = torch.randn(n_tokens, gen_latent_z, 1, 1)

    if gen_type == "deep_linear_6":
        assert initial is not None
        return ProbeSource(
            initial,
            DeepLinearGenerator(
                out_channels=models_c_in,
                latent_dim=gen_latent_z,
                width_mult=generator_width,
                n_layers=6,
            ),
            trainable=True,
        )

    if gen_type == "deep_linear_5":
        assert initial is not None
        return ProbeSource(
            initial,
            DeepLinearGenerator(
                out_channels=models_c_in,
                latent_dim=gen_latent_z,
                width_mult=generator_width,
                n_layers=5,
            ),
            trainable=True,
        )

    if gen_type == "deep_linear_0":
        image_input = torch.randn(n_tokens, models_c_in, 32, 32)
        return ProbeSource(
            image_input,
            DeepLinearGenerator(
                out_channels=models_c_in,
                latent_dim=gen_latent_z,
                width_mult=generator_width,
                n_layers=0,
            ),
            trainable=True,
        )

    if gen_type == "linear_0_no_acts":
        vector_input = torch.randn(n_tokens, models_c_in)
        return ProbeSource(vector_input, nn.Identity(), trainable=True)

    if gen_type == "linear_2_no_acts":
        latent_input = torch.randn(n_tokens, gen_latent_z)
        generator = nn.Sequential(
            nn.Linear(gen_latent_z, gen_latent_z),
            nn.Linear(gen_latent_z, models_c_in),
        )
        return ProbeSource(latent_input, generator, trainable=True)

    if gen_type == "uniform_coords__no_opt":
        fixed_input = torch.rand(n_tokens, models_c_in) * 2.0 - 1.0
        return ProbeSource(fixed_input, nn.Identity(), trainable=False)

    raise ValueError(f"Generator type '{gen_type}' is not recognized.")


def build_hidden_aggregators(
    *,
    n_hidden_target_layers: int,
    r_per_hidden: int,
    kind: str = "set_transformer",
) -> nn.ModuleList:
    """Per-hidden-layer neuron aggregators. 'set_transformer' (default, merged behavior) or
    'deepsets' (ported pre-merge baseline). Both map [T, H] -> [T, r_per_hidden]."""
    def make():
        if kind == "deepsets":
            return HiddenNeuronDeepSets(in_dim=1, out_dim=r_per_hidden)
        if kind == "set_transformer":
            return HiddenNeuronSetTransformer(
                d_model=64, out_dim=r_per_hidden, num_heads=2, num_layers=1, pool="mean")
        raise ValueError(f"Unknown hidden_aggregator: {kind}")
    return nn.ModuleList([make() for _ in range(n_hidden_target_layers)])


def build_per_probe_mlp(
    *,
    input_dim: int,
    d_hidden: int,
    kind: str,
    hidden_width: Optional[int],
    output_dim: Optional[int],
) -> Tuple[Optional[nn.Module], int]:
    if kind == "mlp":
        kind = "mlp2"

    valid = {"none", "linear", "mlp2", "mlp3"}
    if kind not in valid:
        raise ValueError(f"Unknown per_probe_mlp='{kind}'. Expected one of {valid}.")

    width = d_hidden if hidden_width is None else hidden_width
    out_dim = width if output_dim is None else output_dim

    if kind == "none":
        return None, input_dim

    if kind == "linear":
        return nn.Linear(input_dim, out_dim), out_dim

    if kind == "mlp2":
        return (
            nn.Sequential(
                nn.Linear(input_dim, width),
                nn.ReLU(),
                nn.Linear(width, out_dim),
            ),
            out_dim,
        )

    return (
        nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, out_dim),
        ),
        out_dim,
    )



def build_zero_hidden_correction_projection(
    *,
    hidden_dim: int,
    output_dim: int,
) -> nn.Linear:
    """
    Build a zero-initialized additive correction

        hidden_features -> correction in f(x)-space.

    The CNN token is then:

        f(x) + correction(hidden_features).

    Since the projection is initialized to zero, the model begins with exactly
    the same token f(x) as the output-only baseline. The f(x) path itself is
    never projected or modified.
    """
    if hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("hidden_dim and output_dim must be positive.")

    projection = nn.Linear(
        hidden_dim,
        output_dim,
        bias=False,
    )

    with torch.no_grad():
        projection.weight.zero_()

    return projection


def apply_inductive_per_probe_init(
    *,
    module: Optional[nn.Module],
    kind: str,
    input_dim: int,
    passthrough_columns: Sequence[int],
) -> None:
    if module is None:
        return

    if kind == "mlp":
        kind = "mlp2"

    pass_set = set(passthrough_columns)
    hidden_columns = [i for i in range(input_dim) if i not in pass_set]

    with torch.no_grad():
        if kind == "linear":
            if not isinstance(module, nn.Linear):
                raise TypeError("Expected nn.Linear for kind='linear'.")

            module.weight.zero_()
            for row, column in enumerate(passthrough_columns):
                if row < module.weight.shape[0]:
                    module.weight[row, column] = 1.0
            if module.bias is not None:
                module.bias.zero_()
            return

        if kind not in {"mlp2", "mlp3"}:
            raise ValueError(
                "Inductive initialization requires per_probe_mlp to be "
                "'linear', 'mlp2', or 'mlp3'."
            )

        first = module[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("The first per-probe MLP module must be nn.Linear.")

        first.weight[:, hidden_columns] = 0.0
        for row, column in enumerate(passthrough_columns):
            if row < first.weight.shape[0]:
                first.weight[row, column] = 1.0
        if first.bias is not None:
            first.bias.zero_()


def build_mlp_head(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    rank: int = 0,
) -> nn.Sequential:
    if n_layers < 2:
        raise ValueError(f"mixer_n_layers must be at least 2, got {n_layers}.")

    layers = [make_linear(input_dim, hidden_dim, rank=rank), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([make_linear(hidden_dim, hidden_dim, rank=rank), nn.ReLU()])
    layers.append(make_linear(hidden_dim, output_dim, rank=rank))
    return nn.Sequential(*layers)


def find_hidden_linear_layers(
    net: nn.Module,
    expected_count: int,
) -> list[nn.Linear]:
    linear_layers = [
        module for module in net.modules()
        if isinstance(module, nn.Linear)
    ]
    hidden_layers = linear_layers[:-1]

    if len(hidden_layers) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} hidden Linear layers in the target "
            f"network, but found {len(hidden_layers)}. "
            f"Total Linear layers: {len(linear_layers)}."
        )
    return hidden_layers


def run_with_linear_activation_hooks(
    net: nn.Module,
    x: Tensor,
    hidden_layers: Iterable[nn.Linear],
) -> Tuple[Tensor, list[Tensor]]:
    hidden_layers = list(hidden_layers)
    activations: list[Optional[Tensor]] = [None] * len(hidden_layers)
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            activations[index] = output
        return hook

    for index, layer in enumerate(hidden_layers):
        handles.append(layer.register_forward_hook(make_hook(index)))

    try:
        y = net(x)
    finally:
        for handle in handles:
            handle.remove()

    if any(activation is None for activation in activations):
        raise RuntimeError("At least one hidden Linear activation was not captured.")

    return y, [a for a in activations if a is not None]


def run_with_conv_activation_hooks(
    net: nn.Module,
    x: Tensor,
) -> Tuple[Tensor, list[Tensor]]:
    """
    Run net(x) once and capture Conv2d outputs in actual forward-execution order.

    A module reused multiple times contributes one entry for every invocation.
    """
    conv_layers = [module for module in net.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError(
            "Conv hidden features require a target network with Conv2d layers."
        )

    activations: list[Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError("A hooked Conv2d layer returned a non-tensor output.")
        activations.append(output)

    for layer in conv_layers:
        handles.append(layer.register_forward_hook(hook))

    try:
        y = net(x)
    finally:
        for handle in handles:
            handle.remove()

    if not activations:
        raise RuntimeError(
            "Conv2d modules were found, but none ran during the target forward pass."
        )

    return y, activations


def flatten_probe_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(
            f"Expected tokens [B, T, D], got {tuple(tokens.shape)}."
        )
    return tokens.reshape(tokens.shape[0], -1)