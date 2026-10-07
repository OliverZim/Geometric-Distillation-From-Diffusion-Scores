"""Off-manifold training point generation."""

from __future__ import annotations

from typing import Self

import torch
from pydantic import Field, model_validator

from ..config import ComponentConfig


class OffManifoldNoiseConfig(ComponentConfig):
    """Configuration for :func:`sample_off_manifold`."""

    tau_min: float = Field(
        default=0.02,
        gt=0.0,
        description="Lower bound of the log-uniform perturbation scale τ.",
    )
    tau_max: float = Field(
        default=0.15,
        description="Upper bound of the log-uniform perturbation scale τ.",
    )

    @model_validator(mode="after")
    def _check_tau_range(self) -> Self:
        if self.tau_max <= self.tau_min:
            raise ValueError(
                f"tau_max ({self.tau_max}) must be > tau_min ({self.tau_min})"
            )
        return self


def sample_off_manifold(
    x_on: torch.Tensor,
    config: OffManifoldNoiseConfig,
) -> torch.Tensor:
    """Sample ``x_off = x + τ·ε`` with ``τ ~ LogUniform`` and ``ε ~ N(0, I)``."""
    batch_size = x_on.shape[0]
    device = x_on.device
    dtype = x_on.dtype

    log_tau_min = torch.log(torch.tensor(config.tau_min, device=device, dtype=dtype))
    log_tau_max = torch.log(torch.tensor(config.tau_max, device=device, dtype=dtype))

    u = torch.rand(batch_size, device=device, dtype=dtype)
    log_tau = log_tau_min + u * (log_tau_max - log_tau_min)
    tau = torch.exp(log_tau)

    epsilon = torch.randn(
        batch_size,
        x_on.shape[1],
        device=device,
        dtype=dtype,
    )

    return x_on + tau.unsqueeze(1) * epsilon
