from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .probegen_core_utils import (
    apply_inductive_per_probe_init,
    build_hidden_aggregators,
    build_mlp_head,
    build_per_probe_mlp,
    build_probe_source,
    find_hidden_linear_layers,
    flatten_probe_tokens,
    run_with_linear_activation_hooks,
    seeded_initialization,
)


class ProbeGen(nn.Module):
    """
    ProbeGen supporting both vector probes and image probes.

    Output-only mode (used for the CNN Zoo regression experiments):
        x -> target network -> f(x)
        [B, T, models_c_out]
        -> optional per-probe MLP
        -> flatten across probes
        -> mixer MLP
        -> prediction

    For deep_linear_5 / deep_linear_6, build_probe_source may return image
    probes [T, C, H, W]. The target CNN receives those images directly.

    Hidden-feature mode remains supported for vector probes. It is intentionally
    disabled for the CNN Zoo image-probe experiments.
    """

    def __init__(
        self,
        n_tokens: int,
        d_hidden: int,
        models_c_in: int,
        models_c_out: int,
        d_out: int,
        gen_type: str = "deep_linear_6",
        gen_latent_z: int = 32,
        generator_width: int = 16,
        mixer_n_layers: int = 6,
        include_hidden_features: bool = False,
        per_probe_mlp: str = "none",
        per_probe_mlp_width: Optional[int] = None,
        per_probe_out_dim: Optional[int] = None,
        per_probe_init: str = "standard",
        n_hidden_target_layers: int = 0,
        r_per_hidden: int = 2,
        rank: int = 8,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive.")
        if models_c_in <= 0 or models_c_out <= 0:
            raise ValueError("models_c_in and models_c_out must be positive.")
        if n_hidden_target_layers < 0:
            raise ValueError("n_hidden_target_layers cannot be negative.")
        if include_hidden_features and n_hidden_target_layers == 0:
            raise ValueError(
                "include_hidden_features=True requires at least one hidden Linear layer."
            )
        if r_per_hidden <= 0:
            raise ValueError("r_per_hidden must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if per_probe_init not in {"standard", "inductive"}:
            raise ValueError(
                "per_probe_init must be either 'standard' or 'inductive'."
            )

        self.n_tokens = n_tokens
        self.models_c_in = models_c_in
        self.models_c_out = models_c_out
        self.include_hidden_features = include_hidden_features
        self.n_hidden_target_layers = n_hidden_target_layers
        self.r_per_hidden = r_per_hidden
        self.rank = rank

        with seeded_initialization(seed):
            self.probe_source = build_probe_source(
                n_tokens=n_tokens,
                models_c_in=models_c_in,
                gen_type=gen_type,
                gen_latent_z=gen_latent_z,
                generator_width=generator_width,
                seed=seed,
            )

            if include_hidden_features:
                self.hidden_aggregators = build_hidden_aggregators(
                    n_hidden_target_layers=n_hidden_target_layers,
                    r_per_hidden=r_per_hidden,
                    rank=rank,
                )

                # Hidden-feature representation is currently defined for vector
                # probes: [x, hidden_features..., f(x)].
                token_input_dim = (
                    n_hidden_target_layers * r_per_hidden
                    + models_c_out
                )
            else:
                self.hidden_aggregators = None

                # Output-only tokens contain only f(x).
                token_input_dim = models_c_out

            self.per_probe_mlp, token_dim = build_per_probe_mlp(
                input_dim=token_input_dim,
                d_hidden=d_hidden,
                kind=per_probe_mlp,
                hidden_width=per_probe_mlp_width,
                output_dim=per_probe_out_dim,
            )

            if per_probe_init == "inductive":
                self._initialize_per_probe_mlp_inductively(
                    kind=per_probe_mlp,
                    token_input_dim=token_input_dim,
                )

            self.points_mixer = build_mlp_head(
                input_dim=n_tokens * token_dim,
                hidden_dim=d_hidden,
                output_dim=d_out,
                n_layers=mixer_n_layers,
            )

    def _initialize_per_probe_mlp_inductively(
        self,
        *,
        kind: str,
        token_input_dim: int,
    ) -> None:
        if self.per_probe_mlp is None:
            print(
                "[ProbeGen] per_probe_init='inductive' ignored because "
                "per_probe_mlp='none'."
            )
            return

        if not self.include_hidden_features:
            print(
                "[ProbeGen] per_probe_init='inductive' is a no-op because "
                "include_hidden_features=False."
            )
            return

        passthrough_columns = (
            list(range(self.models_c_in))
            + list(
                range(
                    token_input_dim - self.models_c_out,
                    token_input_dim,
                )
            )
        )

        apply_inductive_per_probe_init(
            module=self.per_probe_mlp,
            kind=kind,
            input_dim=token_input_dim,
            passthrough_columns=passthrough_columns,
        )

    def generate_probes(self) -> Tensor:
        x = self.probe_source()
        
        # Vector probes: [T, D]
        # Image probes:  [T, C, H, W]
        if x.ndim not in {2, 4}:
            raise RuntimeError(
                "Expected vector probes [T, D] or image probes [T, C, H, W], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[0] != self.n_tokens:
            raise RuntimeError(
                f"Expected {self.n_tokens} probes, got {x.shape[0]}."
            )

        return x

    def _validate_output(self, y: Tensor) -> None:
        if y.ndim != 2:
            raise RuntimeError(
                f"Expected target output [T, models_c_out], got {tuple(y.shape)}."
            )
        if y.shape[-1] != self.models_c_out:
            raise RuntimeError(
                f"Expected target output width {self.models_c_out}, "
                f"got {y.shape[-1]}."
            )

    def _build_output_only_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        y = net(x)
        self._validate_output(y)
        return y

    def _build_hidden_feature_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        if self.hidden_aggregators is None:
            raise RuntimeError("Hidden aggregators were not initialized.")

        if x.ndim != 2:
            raise RuntimeError(
                "Hidden-feature mode is currently defined only for vector probes. "
                "Use include_hidden_features=False for image probes."
            )

        hidden_layers = find_hidden_linear_layers(
            net,
            expected_count=self.n_hidden_target_layers,
        )

        y, hidden_activations = run_with_linear_activation_hooks(
            net,
            x,
            hidden_layers,
        )
        self._validate_output(y)

        if len(hidden_activations) != self.n_hidden_target_layers:
            raise RuntimeError(
                f"Expected {self.n_hidden_target_layers} hidden activations, "
                f"got {len(hidden_activations)}."
            )

        parts = []

        for aggregator, activation in zip(
            self.hidden_aggregators,
            hidden_activations,
        ):
            parts.append(aggregator(activation))

        parts.append(y)
        return torch.cat(parts, dim=-1)

    def _build_probe_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        if self.include_hidden_features:
            return self._build_hidden_feature_tokens(net, x)
        return self._build_output_only_tokens(net, x)

    def forward_generator(self, nets) -> Tensor:
        x = self.generate_probes()

        outputs = []
        for net in nets:
            y = net(x)
            self._validate_output(y)
            outputs.append(y)

        return torch.stack(outputs, dim=0)

    def forward(self, nets) -> Tensor:
        x = self.generate_probes()

        tokens = torch.stack(
            [self._build_probe_tokens(net, x) for net in nets],
            dim=0,
        )  # [B, T, D]

        if self.per_probe_mlp is not None:
            tokens = self.per_probe_mlp(tokens)

        mixed = flatten_probe_tokens(tokens)
        return self.points_mixer(mixed)


class ProbingGenAdapter(nn.Module):
    """Compatibility adapter for already-computed probe outputs."""

    def __init__(
        self,
        n_tokens: int,
        d_hidden: int,
        models_c_out: int,
        d_out: int,
        mixer_n_layers: int = 6,
    ):
        super().__init__()

        self.points_mixer = build_mlp_head(
            input_dim=n_tokens * models_c_out,
            hidden_dim=d_hidden,
            output_dim=d_out,
            n_layers=mixer_n_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.points_mixer(x.reshape(x.shape[0], -1))
