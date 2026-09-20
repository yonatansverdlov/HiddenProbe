from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@contextmanager
def seeded_initialization(seed: Optional[int]):
    """Temporarily seed PyTorch without changing the caller's RNG state."""
    if seed is None:
        yield
        return

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def make_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


class LowRankLinear(nn.Module):
    """
    Linear map constrained to rank <= rank.

    A dense weight W in R^{out_features x in_features} is represented as

        W = U V^T,

    where
        U in R^{out_features x r}
        V in R^{in_features  x r}.

    The bias is left unfactorized. The factor width is always exactly
    the requested `rank`; we do not clamp it to min(in_features, out_features).
    (The mathematical rank of UV^T is still naturally bounded by the matrix
    dimensions.)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def weight(self) -> Tensor:
        """Materialized effective weight, shape [out_features, in_features]."""
        return self.U @ self.V.transpose(0, 1)

    def reset_parameters(self) -> None:
        # Initialize an ordinary Xavier dense matrix, then project it to rank r.
        # This gives the factorized layer the same initial scale as nn.Linear.
        with torch.no_grad():
            dense = torch.empty(self.out_features, self.in_features)
            nn.init.xavier_uniform_(dense)
            self.set_effective_weight(dense)
            if self.bias is not None:
                bound = 1.0 / math.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_effective_weight(self, weight: Tensor) -> None:
        """Set UV^T to the best rank-r SVD approximation of `weight`."""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected weight {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}."
            )

        # SVD is used only during initialization, not in the forward pass.
        # k is only the number of singular directions available in this
        # particular matrix. The PARAMETERS still keep exactly `self.rank`
        # columns: U:[out, rank], V:[in, rank].
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        k = min(self.rank, s.numel())

        self.U.zero_()
        self.V.zero_()

        sqrt_s = s[:k].clamp_min(0).sqrt()
        self.U[:, :k].copy_(
            (u[:, :k] * sqrt_s.unsqueeze(0)).to(
                dtype=self.U.dtype, device=self.U.device
            )
        )
        self.V[:, :k].copy_(
            (vh[:k, :].transpose(0, 1) * sqrt_s.unsqueeze(0)).to(
                dtype=self.V.dtype, device=self.V.device
            )
        )

        # If rank exceeds the maximum possible matrix rank, keep the effective
        # initialization exact while avoiding permanently dead redundant
        # columns: U extras are zero, V extras are small random values.
        if k < self.rank:
            nn.init.normal_(self.V[:, k:], mean=0.0, std=1.0 / math.sqrt(self.in_features))

    def forward(self, x: Tensor) -> Tensor:
        # x V U^T is algebraically identical to x W^T for W = U V^T,
        # while avoiding materializing the dense matrix in the forward pass.
        x = F.linear(x, self.V.transpose(0, 1), bias=None)
        return F.linear(x, self.U, bias=self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


def make_linear(
    in_features: int,
    out_features: int,
    *,
    rank: int,
    bias: bool = True,
) -> LowRankLinear:
    """Low-rank linear used only inside the added small hidden branch."""
    return LowRankLinear(in_features, out_features, rank=rank, bias=bias)


class SmallLowRankHiddenEncoder(nn.Module):
    """
    Fast permutation-invariant encoder for one hidden target layer.

    Input:
        h: [T, H]  (T probes, H hidden neurons)

    Output:
        z: [T, out_dim]

    Each neuron activation is processed independently by a small pointwise MLP,
    then we mean-pool over neurons. Therefore the encoder is invariant to hidden
    neuron permutations and has O(T * H) set-processing cost rather than the
    O(T * H^2) attention cost of the larger Set Transformer.

    All matrix-valued maps are represented as W = U V^T with factor width
    `rank`. `r_per_hidden` is only the final output width and is independent of
    `rank`.
    """

    def __init__(
        self,
        out_dim: int = 2,
        rank: int = 8,
        latent_dim: int = 16,
        pool: str = "mean",
    ):
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if pool not in {"mean", "sum", "max"}:
            raise ValueError(f"Unknown pool: {pool}")

        self.out_dim = out_dim
        self.rank = rank
        self.latent_dim = latent_dim
        self.pool = pool

        # Shared across all hidden neurons. No interaction/attention between
        # neuron pairs: this is the small/fast hidden encoder.
        self.phi1 = LowRankLinear(1, latent_dim, rank=rank)
        self.phi2 = LowRankLinear(latent_dim, latent_dim, rank=rank)
        self.rho = LowRankLinear(latent_dim, out_dim, rank=rank)

    def forward(self, h: Tensor) -> Tensor:
        if h.ndim != 2:
            raise ValueError(
                f"Expected hidden activations [T, H], got {tuple(h.shape)}."
            )

        # [T,H] -> [T,H,1] -> [T,H,D]
        z = F.relu(self.phi1(h.unsqueeze(-1)))
        z = F.relu(self.phi2(z))

        # Permutation-invariant reduction over the hidden-neuron dimension H.
        if self.pool == "mean":
            z = z.mean(dim=1)
        elif self.pool == "sum":
            z = z.sum(dim=1)
        else:
            z = z.max(dim=1).values

        return self.rho(z)


class ProbeSource(nn.Module):
    """Own a trainable or fixed set of vector probes."""

    def __init__(self, initial_input: Tensor, generator: nn.Module, trainable: bool):
        super().__init__()
        self.generator = generator

        if trainable:
            self.input = nn.Parameter(initial_input)
        else:
            self.register_buffer("input", initial_input)

    def forward(self) -> Tensor:
        return self.generator(self.input)


from typing import Optional

import torch
import torch.nn as nn


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
                nn.ConvTranspose2d(
                    latent_dim,
                    width_mult * 8,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    width_mult * 8,
                    width_mult * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult * 4,
                    width_mult * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult * 2,
                    width_mult * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult * 2,
                    width_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Tanh(),
            ]

        elif n_layers == 5:
            layers = [
                nn.ConvTranspose2d(
                    latent_dim,
                    width_mult * 8,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    width_mult * 8,
                    width_mult * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult * 4,
                    width_mult * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult * 2,
                    width_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    width_mult,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Tanh(),
            ]

        elif n_layers == 0:
            layers = [
                nn.Tanh(),
            ]

        else:
            raise ValueError(
                "DeepLinearGenerator supports "
                f"n_layers in {{0, 5, 6}}, got {n_layers}."
            )

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)


class ProbeSource(nn.Module):
    def __init__(
        self,
        initial: torch.Tensor,
        generator: nn.Module,
        trainable: bool = True,
    ):
        super().__init__()

        if trainable:
            self.initial = nn.Parameter(initial)
        else:
            self.register_buffer("initial", initial)

        self.generator = generator

    def forward(self):
        return self.generator(self.initial)


def make_torch_generator(seed: Optional[int]):
    if seed is None:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class seeded_initialization:
    """
    Context manager that temporarily fixes torch's global RNG.

    Useful because nn.Linear / nn.ConvTranspose2d initialize their
    parameters from the global torch RNG.
    """

    def __init__(self, seed: Optional[int]):
        self.seed = seed
        self.cpu_rng_state = None
        self.cuda_rng_state = None

    def __enter__(self):
        if self.seed is None:
            return

        self.cpu_rng_state = torch.get_rng_state()

        if torch.cuda.is_available():
            self.cuda_rng_state = torch.cuda.get_rng_state_all()

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.seed is None:
            return

        torch.set_rng_state(self.cpu_rng_state)

        if (
            torch.cuda.is_available()
            and self.cuda_rng_state is not None
        ):
            torch.cuda.set_rng_state_all(
                self.cuda_rng_state
            )


def build_probe_source(
    *,
    n_tokens: int,
    models_c_in: int,
    gen_type: str,
    gen_latent_z: int,
    generator_width: int,
    seed: Optional[int],
) -> ProbeSource:
    """
    Build the probe source.

    Supported generators:

    linear_0_no_acts
        Directly optimize the probes themselves.

    linear_2_no_acts
        Trainable latent probes followed by two Linear layers.

    deep_linear_5
        Original ProbeGen-style image generator with five
        ConvTranspose2d layers followed by Tanh.

    deep_linear_6
        Six-layer version of the same generator.

    uniform_coords__no_opt
        Fixed uniformly sampled coordinates in [-1, 1].
    """

    rng = make_torch_generator(seed)

    with seeded_initialization(seed):

        # --------------------------------------------------------------
        # Directly optimized probes
        # --------------------------------------------------------------
        if gen_type == "linear_0_no_acts":

            initial = torch.randn(
                n_tokens,
                models_c_in,
                generator=rng,
            )

            return ProbeSource(
                initial=initial,
                generator=nn.Identity(),
                trainable=True,
            )


        # --------------------------------------------------------------
        # Two-layer fully-linear generator
        # --------------------------------------------------------------
        if gen_type == "linear_2_no_acts":

            initial = torch.randn(
                n_tokens,
                gen_latent_z,
                generator=rng,
            )

            generator = nn.Sequential(
                nn.Linear(
                    gen_latent_z,
                    gen_latent_z,
                ),
                nn.Linear(
                    gen_latent_z,
                    models_c_in,
                ),
            )

            return ProbeSource(
                initial=initial,
                generator=generator,
                trainable=True,
            )


        # --------------------------------------------------------------
        # Original ProbeGen 5-layer image generator
        #
        # z:
        #   [n_tokens, latent_dim, 1, 1]
        #
        # output:
        #   [n_tokens, models_c_in, 32, 32]
        # --------------------------------------------------------------
        if gen_type == "deep_linear_5":

            initial = torch.randn(
                n_tokens,
                gen_latent_z,
                1,
                1,
                generator=rng,
            )

            generator = DeepLinearGenerator(
                out_channels=models_c_in,
                latent_dim=gen_latent_z,
                width_mult=generator_width,
                n_layers=5,
            )

            return ProbeSource(
                initial=initial,
                generator=generator,
                trainable=True,
            )


        # --------------------------------------------------------------
        # Original ProbeGen 6-layer image generator
        # --------------------------------------------------------------
        if gen_type == "deep_linear_6":

            initial = torch.randn(
                n_tokens,
                gen_latent_z,
                1,
                1,
                generator=rng,
            )

            generator = DeepLinearGenerator(
                out_channels=models_c_in,
                latent_dim=gen_latent_z,
                width_mult=generator_width,
                n_layers=6,
            )

            return ProbeSource(
                initial=initial,
                generator=generator,
                trainable=True,
            )


        # --------------------------------------------------------------
        # Fixed uniform probes
        # --------------------------------------------------------------
        if gen_type == "uniform_coords__no_opt":

            initial = torch.rand(
                n_tokens,
                models_c_in,
                generator=rng,
            )

            initial = (
                initial
                .mul(2.0)
                .sub(1.0)
            )

            return ProbeSource(
                initial=initial,
                generator=nn.Identity(),
                trainable=False,
            )


    raise ValueError(
        f"Generator type '{gen_type}' is not supported."
    )

def build_hidden_aggregators(
    *,
    n_hidden_target_layers: int,
    r_per_hidden: int,
    rank: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            SmallLowRankHiddenEncoder(
                out_dim=r_per_hidden,
                rank=rank,
                latent_dim=16,
                pool="mean",
            )
            for _ in range(n_hidden_target_layers)
        ]
    )


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


def _set_linear_effective_weight(module: nn.Module, weight: Tensor) -> None:
    if isinstance(module, LowRankLinear):
        module.set_effective_weight(weight)
        return
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            module.weight.copy_(weight.to(module.weight))
        return
    raise TypeError(f"Expected LowRankLinear or nn.Linear, got {type(module).__name__}.")


def apply_inductive_per_probe_init(
    *,
    module: Optional[nn.Module],
    kind: str,
    input_dim: int,
    passthrough_columns: Sequence[int],
) -> None:
    """
    Initialize psi so the old [x, f(x)] coordinates pass through initially
    while hidden coordinates start with zero influence.

    For low-rank layers the desired dense initialization is factorized by SVD.
    """
    if module is None:
        return

    if kind == "mlp":
        kind = "mlp2"

    pass_set = set(passthrough_columns)
    hidden_columns = [i for i in range(input_dim) if i not in pass_set]

    with torch.no_grad():
        if kind == "linear":
            if not isinstance(module, (LowRankLinear, nn.Linear)):
                raise TypeError("Expected a linear module for kind='linear'.")
            W = torch.zeros(module.out_features, input_dim)
            for row, column in enumerate(passthrough_columns):
                if row < W.shape[0]:
                    W[row, column] = 1.0
            _set_linear_effective_weight(module, W)
            if module.bias is not None:
                module.bias.zero_()
            return

        if kind not in {"mlp2", "mlp3"}:
            raise ValueError(
                "Inductive initialization requires per_probe_mlp to be "
                "'linear', 'mlp2', or 'mlp3'."
            )

        first = module[0]
        if not isinstance(first, (LowRankLinear, nn.Linear)):
            raise TypeError("The first per-probe MLP module must be linear.")

        # Start from its current effective weight, zero hidden columns, and put
        # explicit passthrough entries in the first available rows.
        W = first.weight.detach().clone()
        W[:, hidden_columns] = 0.0
        for row, column in enumerate(passthrough_columns):
            if row < W.shape[0]:
                W[row, :] = 0.0
                W[row, column] = 1.0
        _set_linear_effective_weight(first, W)
        if first.bias is not None:
            first.bias.zero_()


def build_mlp_head(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
) -> nn.Sequential:
    if n_layers < 2:
        raise ValueError(f"mixer_n_layers must be at least 2, got {n_layers}.")

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def find_hidden_linear_layers(
    net: nn.Module,
    expected_count: int,
) -> list[nn.Linear]:
    # These are the target network's Linear layers.  We do NOT alter/factorize
    # the target networks themselves; only ProbeGen's learned matrices are low rank.
    linear_layers = [module for module in net.modules() if isinstance(module, nn.Linear)]
    hidden_layers = linear_layers[:-1]

    if len(hidden_layers) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} hidden Linear layers in the target network, "
            f"but found {len(hidden_layers)}. Total Linear layers: {len(linear_layers)}."
        )

    return hidden_layers


def run_with_linear_activation_hooks(
    net: nn.Module,
    x: Tensor,
    hidden_layers: Iterable[nn.Linear],
) -> Tuple[Tensor, list[Tensor]]:
    """Run net(x) once and capture outputs of the requested hidden Linear layers."""
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
        raise RuntimeError("At least one hidden activation was not captured.")

    return y, [a for a in activations if a is not None]


def flatten_probe_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B, T, D], got {tuple(tokens.shape)}.")
    return tokens.reshape(tokens.shape[0], -1)
