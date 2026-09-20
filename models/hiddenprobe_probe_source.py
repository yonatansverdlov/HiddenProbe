"""Internal probe-source support for the HiddenProbe CIFAR backend.

This is NOT the canonical ProbeGen implementation. All public method=probegen
runs are routed to models/probegen_core.py (from inr_classification_branch).
This module is retained only because the HiddenProbe CIFAR trainer reuses its
learned probe generator.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.probegen_utils import (
    DeepSetsEncoder,
    HiddenNeuronSetTransformer,
    InvariantProbeTransformer,
    apply_inductive_per_probe_init,
    build_hidden_aggregators,
    build_mlp_head,
    build_per_probe_mlp,
    build_probe_source,
    build_zero_hidden_correction_projection,
    find_hidden_linear_layers,
    flatten_probe_tokens,
    run_with_conv_activation_hooks,
    run_with_linear_activation_hooks,
)


class ProbeGen(nn.Module):
    """
    ProbeGen with fixed cross-probe aggregation:

        [B, T, D] -> flatten -> mixer MLP -> [B, d_out]

    Hidden-feature modes:
      - Linear targets (INRs): one Set Transformer over hidden neurons per layer.
      - CNN targets: GAP each Conv2d feature map, then one Set Transformer over
        channels per Conv layer. The three layer representations are concatenated
        with f(x), projected back to f(x)-space with a [0 | I] initialization,
        and passed through the ordinary per-probe MLP.
    """

    def __init__(
        self,
        n_tokens: int,
        d_hidden: int,
        models_c_in: int,
        models_c_out: int,
        d_out: int,
        gen_type: str = "deep_linear_5",
        gen_latent_z: int = 32,
        generator_width: int = 16,
        mixer_n_layers: int = 6,
        include_hidden_features: bool = False,
        per_probe_mlp: str = "none",
        per_probe_mlp_width: Optional[int] = None,
        per_probe_out_dim: Optional[int] = None,
        per_probe_init: str = "standard",
        n_hidden_target_layers: int = 2,
        r_per_hidden: int = 1,
        hidden_aggregator: str = "set_transformer",  # per-hidden-layer agg: "set_transformer" | "deepsets"
        is_cnn_target: bool = False,
        r_per_conv: int = 1,
        n_conv_target_layers: int = 0,
        aggregator: str = "mlp",             # "mlp" | "pat" (PAT = Probe-Activation Transformer)
        pat_d: int = 256,                    # PAT transformer width (PATConfig.d); default = full 9.8M model
        pat_n_blocks: int = 6,               # PAT depth (PATConfig.n_blocks)
        pat_rank: int = 0,                   # PAT low-rank factorization rank (PATConfig.pat_rank; 0 = full)
        st_rank: int = 0,                    # set_transformer low-rank factorization rank (0 = full)
        probe_pos_encoding: bool = False,    # PAT STAGE C: per-row probe positional encoding (breaks S_N)
        domain_tanh: bool = False,           # [Arm-B] squash learned probe coords to (-1,1)^d (INR domain)
        pat_token_scheme: str = "lite",      # PAT INR output container: "lite" (broadcast V(f)) | "typed" (F_out)
    ):
        super().__init__()
        self.st_rank = int(st_rank)          # set_transformer low-rank (used in _build_cross_probe_aggregator)

        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive.")
        if models_c_in <= 0 or models_c_out <= 0:
            raise ValueError("models_c_in and models_c_out must be positive.")
        if n_hidden_target_layers < 0:
            raise ValueError("n_hidden_target_layers cannot be negative.")
        if r_per_hidden <= 0:
            raise ValueError("r_per_hidden must be positive.")
        if r_per_conv <= 0:
            raise ValueError("r_per_conv must be positive.")
        if n_conv_target_layers < 0:
            raise ValueError("n_conv_target_layers cannot be negative.")
        if (
            include_hidden_features
            and is_cnn_target
            and n_conv_target_layers == 0
        ):
            raise ValueError(
                "CNN hidden features require n_conv_target_layers > 0."
            )
        if per_probe_init not in {"standard", "inductive"}:
            raise ValueError(
                "per_probe_init must be either 'standard' or 'inductive'."
            )

        self.n_tokens = n_tokens
        self.models_c_in = models_c_in
        self.models_c_out = models_c_out
        self.domain_tanh = bool(domain_tanh)
        self.include_hidden_features = include_hidden_features
        self.is_cnn_target = is_cnn_target
        self.use_conv_hidden_features = include_hidden_features and is_cnn_target
        self.n_hidden_target_layers = n_hidden_target_layers
        self.r_per_conv = r_per_conv
        self.n_conv_target_layers = n_conv_target_layers

        self.probe_source = build_probe_source(
            n_tokens=n_tokens,
            models_c_in=models_c_in,
            gen_type=gen_type,
            gen_latent_z=gen_latent_z,
            generator_width=generator_width,
        )

        # PAT replaces the flatten->points_mixer cross-probe path with one joint axial transformer
        # over the raw (B, T, D, 1) probe-activation tensor; it owns its own head to d_out classes.
        # Defaults (pat_d=256, pat_n_blocks=6) reproduce the full 9.8M model; smaller values give a
        # capacity-matched arm. PATAdapter captures target pre-activations itself, so it needs only
        # (nets, probe_coords). Requires a vector-valued generator (x.dim()==2), e.g. linear_2_no_acts.
        self.aggregator_kind = aggregator
        if aggregator == "pat":
            raise NotImplementedError("aggregator='pat' (PATAdapter / axial transformer) is not included in this branch; use the default aggregator")

        self.hidden_aggregators: Optional[nn.ModuleList] = None
        self.conv_hidden_aggregators: Optional[nn.ModuleList] = None
        self.conv_hidden_projection: Optional[nn.Linear] = None
        # Cross-probe aggregator (deepsets/set_transformer): sits between the per-probe tokens and
        # the points_mixer head, replacing the flatten step. None for the default mlp path.
        self.aggregator_module: Optional[nn.Module] = None

        if self.use_conv_hidden_features:
            # ------------------------------------------------------------
            # CNN hidden mode
            # ------------------------------------------------------------
            # The ordinary CNN ProbeGen representation is f(x). Therefore the
            # existing per-probe MLP is built exactly on models_c_out inputs.
            # The newly added hidden information is projected back to this same
            # space before that MLP, so any requested MLP kind remains unchanged.
            self.per_probe_mlp, token_dim = build_per_probe_mlp(
                input_dim=models_c_out,
                d_hidden=d_hidden,
                kind=per_probe_mlp,
                hidden_width=per_probe_mlp_width,
                output_dim=per_probe_out_dim,
            )

            head_in = self._build_cross_probe_aggregator(
                aggregator, token_dim, n_tokens, models_c_out, d_hidden)
            self.points_mixer = build_mlp_head(
                input_dim=head_in,
                hidden_dim=d_hidden,
                output_dim=d_out,
                n_layers=mixer_n_layers,
            )

            # Build added modules without advancing the process-wide RNG. This
            # preserves the baseline probes, per-probe MLP, mixer, and subsequent
            # DataLoader shuffle when the same seed is used.
            with torch.random.fork_rng(devices=[]):
                self.conv_hidden_aggregators = nn.ModuleList(
                    [
                        HiddenNeuronSetTransformer(
                            d_model=64,
                            out_dim=r_per_conv,
                            num_heads=2,
                            num_layers=1,
                            pool="mean",
                        )
                        for _ in range(self.n_conv_target_layers)
                    ]
                )

                hidden_dim_total = self.n_conv_target_layers * r_per_conv
                self.conv_hidden_projection = (
                    build_zero_hidden_correction_projection(
                        hidden_dim=hidden_dim_total,
                        output_dim=models_c_out,
                    )
                )

        else:
            # ------------------------------------------------------------
            # Existing INR hidden mode or ordinary output-only mode
            # ------------------------------------------------------------
            if include_hidden_features:
                self.hidden_aggregators = build_hidden_aggregators(
                    n_hidden_target_layers=n_hidden_target_layers,
                    r_per_hidden=r_per_hidden,
                    kind=hidden_aggregator,
                )
                token_input_dim = (
                    models_c_in
                    + n_hidden_target_layers * r_per_hidden
                    + models_c_out
                )
            elif is_cnn_target:
                # For CNNs, the original ProbeGen token is f(x), not GAP(x).
                token_input_dim = models_c_out
            else:
                token_input_dim = models_c_in + models_c_out

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

            head_in = self._build_cross_probe_aggregator(
                aggregator, token_dim, n_tokens, models_c_out, d_hidden)
            self.points_mixer = build_mlp_head(
                input_dim=head_in,
                hidden_dim=d_hidden,
                output_dim=d_out,
                n_layers=mixer_n_layers,
                rank=int(getattr(self, "st_rank", 0)),
            )

    def _build_cross_probe_aggregator(self, aggregator, token_dim, n_tokens, models_c_out, d_hidden):
        """deepsets/set_transformer: build the cross-probe aggregator ([B,T,token_dim]->[B,head_in])
        and return head_in = n_tokens*models_c_out (kept equal to the mlp head width for parity).
        mlp: leave aggregator_module=None and return the flattened width n_tokens*token_dim.
        Ported from the pre-merge baseline family (probegen_utils.DeepSetsEncoder / InvariantProbeTransformer).
        """
        if aggregator not in ("deepsets", "set_transformer"):
            return n_tokens * token_dim
        head_in = n_tokens * models_c_out
        if aggregator == "deepsets":
            self.aggregator_module = DeepSetsEncoder(
                in_dim=token_dim, hidden_dim=d_hidden * 2, out_dim=head_in,
                n_layers=6, pool="mean")
        else:
            self.aggregator_module = InvariantProbeTransformer(
                in_dim=token_dim, d_model=d_hidden, out_dim=head_in,
                num_heads=2, num_layers=2, dim_feedforward=d_hidden, dropout=0.0, pool="mean",
                rank=int(getattr(self, "st_rank", 0)))
        return head_in

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
        if self.domain_tanh:                 # [Arm-B] keep learned coords in the INR domain (-1,1)^d
            x = torch.tanh(x)
        return x

    def _validate_target_output(self, y: Tensor) -> None:
        if y.ndim != 2 or y.shape[-1] != self.models_c_out:
            raise RuntimeError(
                f"Expected target output [T, {self.models_c_out}], "
                f"got {tuple(y.shape)}."
            )

    def _build_output_only_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        y = net(x)
        self._validate_target_output(y)

        if self.is_cnn_target:
            return y
        return torch.cat([x, y], dim=-1)

    def _build_linear_hidden_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise RuntimeError(
                "Linear hidden features expect vector/coordinate probes [T, C], "
                f"got {tuple(x.shape)}."
            )
        if self.hidden_aggregators is None:
            raise RuntimeError("Linear hidden aggregators were not initialized.")

        hidden_layers = find_hidden_linear_layers(
            net,
            expected_count=self.n_hidden_target_layers,
        )
        y, hidden_activations = run_with_linear_activation_hooks(
            net,
            x,
            hidden_layers,
        )
        self._validate_target_output(y)

        parts = [x]
        for aggregator, activation in zip(
            self.hidden_aggregators,
            hidden_activations,
        ):
            parts.append(aggregator(activation))
        parts.append(y)
        return torch.cat(parts, dim=-1)

    def _build_conv_hidden_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        """
        CNN path for one target network.

        Every executed Conv2d output is used:

            [T, C_l, H_l, W_l]
                --GAP over H_l,W_l-->
            [T, C_l]
                --Set Transformer over channels-->
            [T, r_per_conv]

        The representations of all Conv layers are concatenated and mapped to
        an additive correction in f(x)-space:

            token = f(x) + hidden_projection(hidden_features).

        hidden_projection is initialized to zero, so the model begins exactly
        with the output-only token f(x). The direct f(x) path is never modified.
        """
        if x.ndim != 4:
            raise RuntimeError(
                "Conv hidden features expect image probes [T, C, H, W], "
                f"got {tuple(x.shape)}. Use an image-producing gen_type such as "
                "'deep_linear_5'."
            )
        if self.conv_hidden_aggregators is None:
            raise RuntimeError("Conv hidden aggregators were not initialized.")
        if self.conv_hidden_projection is None:
            raise RuntimeError("Conv hidden projection was not initialized.")

        y, conv_activations = run_with_conv_activation_hooks(net, x)
        self._validate_target_output(y)

        if len(conv_activations) != self.n_conv_target_layers:
            raise RuntimeError(
                f"Expected {self.n_conv_target_layers} executed Conv2d layers, "
                f"but captured {len(conv_activations)}."
            )

        hidden_parts = []
        for aggregator, activation in zip(
            self.conv_hidden_aggregators,
            conv_activations,
        ):
            if activation.ndim != 4:
                raise RuntimeError(
                    "Expected Conv2d activation [T, C, H, W], got "
                    f"{tuple(activation.shape)}."
                )

            # One scalar per channel, for every probe.
            channel_values = activation.mean(dim=(2, 3))
            hidden_parts.append(aggregator(channel_values))

        hidden_features = torch.cat(hidden_parts, dim=-1)
        hidden_correction = self.conv_hidden_projection(hidden_features)

        return y + hidden_correction

    def _build_probe_tokens(self, net: nn.Module, x: Tensor) -> Tensor:
        if not self.include_hidden_features:
            return self._build_output_only_tokens(net, x)
        if self.use_conv_hidden_features:
            return self._build_conv_hidden_tokens(net, x)
        return self._build_linear_hidden_tokens(net, x)

    def forward_generator(self, nets) -> Tensor:
        # Preserve the original method's direct generator behavior.
        x = self.probe_source()
        return torch.stack([net(x) for net in nets], dim=0)

    def forward(self, nets) -> Tensor:
        x = self.generate_probes()

        if self.aggregator_kind == "pat":
            return self.pat_adapter(nets, x)

        tokens = torch.stack(
            [self._build_probe_tokens(net, x) for net in nets],
            dim=0,
        )

        if self.per_probe_mlp is not None:
            tokens = self.per_probe_mlp(tokens)

        # deepsets/set_transformer aggregate over the probe axis before the head; mlp flattens.
        if self.aggregator_module is not None:
            return self.points_mixer(self.aggregator_module(tokens))
        return self.points_mixer(flatten_probe_tokens(tokens))


class ProbingGenAdapter(nn.Module):
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
