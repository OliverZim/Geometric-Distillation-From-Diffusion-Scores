"""Pseudo-projection targets from a frozen score network."""

from __future__ import annotations

from typing import Self

import torch
from pydantic import Field, model_validator

from geometry_distill.score.network import ScoreNetwork

from ..config import ComponentConfig


class FlowConfig(ComponentConfig):
    """Configuration for :class:`FlowPseudoProjector`."""

    sigmas: tuple[float, ...] = Field(
        default=(0.2, 0.1, 0.05, 0.02),
        description=(
            "Noise levels in strictly descending order; a single value gives "
            "the single-step pseudo-projection."
        ),
    )

    @model_validator(mode="after")
    def _check_sigmas(self) -> Self:
        if len(self.sigmas) == 0:
            raise ValueError("sigmas must be non-empty")
        for i, s in enumerate(self.sigmas):
            if s <= 0.0:
                raise ValueError(f"sigmas[{i}] must be positive, got {s}")
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] <= self.sigmas[i + 1]:
                raise ValueError(
                    f"sigmas must be strictly descending, "
                    f"got sigmas[{i}]={self.sigmas[i]} <= "
                    f"sigmas[{i + 1}]={self.sigmas[i + 1]}"
                )
        return self


class FlowPseudoProjector:
    """Pseudo-projection via a chain of score evaluations at descending σ."""

    _config: FlowConfig
    _score_network: ScoreNetwork

    def __init__(self, config: FlowConfig, score_network: ScoreNetwork):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_score_network", score_network)

    @property
    def config(self) -> FlowConfig:
        return self._config

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        self._score_network.eval()
        device = next(self._score_network.parameters()).device
        x = x.to(device)

        for sigma_val in self._config.sigmas:
            sigma = torch.full(
                (x.shape[0],),
                sigma_val,
                device=device,
                dtype=x.dtype,
            )
            score = self._score_network(x, sigma)
            x = x + (sigma_val**2) * score

        return x
