"""Score network ``s_θ(x, σ)`` predicting ``∇log p_σ(x)`` for noisy inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import Field

from ..config import ComponentConfig


class ScoreNetworkConfig(ComponentConfig):
    """Configuration for :class:`ScoreNetwork`."""

    x_dim: int = Field(
        default=3,
        ge=1,
        description="Ambient space dimension (input and output).",
    )
    hidden_dim: int = Field(
        default=128,
        ge=1,
        description="Width of hidden layers.",
    )
    n_layers: int = Field(
        default=3,
        ge=1,
        description="Number of hidden layers.",
    )


def _sigma_embedding(sigma: torch.Tensor) -> torch.Tensor:
    """[log σ, 1/σ] embedding."""
    s = sigma.clamp_min(1e-8)
    return torch.stack([torch.log(s), torch.reciprocal(s)], dim=-1)


class ScoreNetwork(nn.Module):
    """Score model ``s_θ(x, σ): R^d × R → R^d``."""

    _config: ScoreNetworkConfig

    def __init__(self, config: ScoreNetworkConfig):
        super().__init__()
        object.__setattr__(self, "_config", config)

        dim = config.x_dim + 2
        layers: list[nn.Module] = []
        for _ in range(config.n_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.SiLU())
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, config.x_dim))
        self.net = nn.Sequential(*layers)

    @property
    def config(self) -> ScoreNetworkConfig:
        return self._config

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict scores for noisy points, shape ``(batch, x_dim)``."""
        inp = torch.cat([x, _sigma_embedding(sigma)], dim=-1)
        out = self.net(inp)
        return out / sigma.clamp_min(1e-8).unsqueeze(-1)
