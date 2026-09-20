"""Minimal learned-probe source used only by HiddenProbe CIFAR classification.

Public ProbeGen runs never use this module. They are routed to
models.probegen_core / models.probegen_core_trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.probegen_utils import build_probe_source


class LearnedProbeSource(nn.Module):
    def __init__(
        self,
        *,
        n_probes: int,
        models_c_in: int,
        gen_type: str = "linear_2_no_acts",
        gen_latent_z: int = 32,
        generator_width: int = 16,
        domain_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.n_probes = int(n_probes)
        self.domain_tanh = bool(domain_tanh)
        self.probe_source = build_probe_source(
            n_tokens=self.n_probes,
            models_c_in=int(models_c_in),
            gen_type=gen_type,
            gen_latent_z=int(gen_latent_z),
            generator_width=int(generator_width),
        )

    def generate_probes(self) -> torch.Tensor:
        x = self.probe_source()
        if self.domain_tanh:
            x = torch.tanh(x)
        if x.shape[0] != self.n_probes:
            raise RuntimeError(
                f"Expected {self.n_probes} probes, got {x.shape[0]}."
            )
        return x
