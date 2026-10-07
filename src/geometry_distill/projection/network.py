"""Single-chart encoder–decoder projector ``x̂ = D(E(x))``."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from pydantic import Field

from ..config import ComponentConfig


class SingleChartConfig(ComponentConfig):
    """Configuration for :class:`SingleChartProjector`."""

    x_dim: int = Field(
        default=3,
        ge=1,
        description="Ambient space dimension (input and output).",
    )
    k: int = Field(
        default=2,
        ge=1,
        description="Latent (intrinsic) dimension of the manifold.",
    )
    hidden_dim: int = Field(
        default=128,
        ge=1,
        description="Width of hidden layers in both encoder and decoder.",
    )
    enc_layers: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of hidden layers in the encoder MLP; "
            "the output layer is added automatically."
        ),
    )
    dec_layers: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of hidden layers in the decoder MLP; "
            "the output layer is added automatically."
        ),
    )


class SingleChartProjector(nn.Module):
    """Learned nearest-point projection."""

    _is_atlas: ClassVar[bool] = False
    _config: SingleChartConfig

    def __init__(self, config: SingleChartConfig):
        super().__init__()
        self._config = config

        enc: list[nn.Module] = []
        dim = config.x_dim
        for _ in range(config.enc_layers):
            enc.append(nn.Linear(dim, config.hidden_dim))
            enc.append(nn.SiLU())
            dim = config.hidden_dim
        enc.append(nn.Linear(dim, config.k))
        self.encoder: nn.Module = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        dim = config.k
        for _ in range(config.dec_layers):
            dec.append(nn.Linear(dim, config.hidden_dim))
            dec.append(nn.SiLU())
            dim = config.hidden_dim
        dec.append(nn.Linear(dim, config.x_dim))
        self.decoder: nn.Module = nn.Sequential(*dec)

    @property
    def config(self) -> SingleChartConfig:
        return self._config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ambient points to latent codes, ``(batch, x_dim)`` → ``(batch, k)``."""
        return self.encoder(x)  # type: ignore[return-value]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent codes to ambient space, ``(batch, k)`` → ``(batch, x_dim)``."""
        return self.decoder(z)  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the learned manifold."""
        return self.decode(self.encode(x))
