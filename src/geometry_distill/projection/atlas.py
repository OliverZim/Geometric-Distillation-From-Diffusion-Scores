"""Multi-chart soft atlas projector."""

from __future__ import annotations

from typing import ClassVar, Self, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, model_validator

from ..config import ComponentConfig


class SoftAtlasConfig(ComponentConfig):
    """Configuration for :class:`SoftAtlasProjector`."""

    x_dim: int = Field(
        default=3,
        ge=1,
        description="Ambient space dimension (input and output).",
    )
    k: int = Field(
        default=2,
        ge=1,
        description="Latent (intrinsic) dimension per chart.",
    )
    n_charts: int = Field(
        default=2,
        ge=1,
        description="Number of charts in the atlas.",
    )
    hidden_dim: int = Field(
        default=128,
        ge=1,
        description="Width of hidden layers throughout the network.",
    )
    trunk_layers: int = Field(
        default=2,
        ge=1,
        description="Number of hidden layers in the shared trunk MLP.",
    )
    head_layers: int = Field(
        default=2,
        ge=1,
        description="Number of hidden layers in the weight and latent heads.",
    )
    dec_layers: int = Field(
        default=2,
        ge=0,
        description=(
            "Number of hidden layers per chart decoder; "
            "0 means the same as head_layers."
        ),
    )

    @model_validator(mode="after")
    def _default_dec_layers(self) -> Self:
        """Resolve the ``dec_layers=0`` sentinel to ``head_layers``."""
        if self.dec_layers == 0:
            object.__setattr__(self, "dec_layers", self.head_layers)
        return self


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
) -> nn.Sequential:
    """MLP with *n_layers* hidden SiLU layers and a linear output (no activation)."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SoftAtlasProjector(nn.Module):
    r"""Learned projection as a softmax-weighted blend of per-chart decoders."""

    _is_atlas: ClassVar[bool] = True
    _config: SoftAtlasConfig

    def __init__(self, config: SoftAtlasConfig):
        super().__init__()
        self._config = config

        trunk_layers: list[nn.Module] = []
        dim = config.x_dim
        for _ in range(config.trunk_layers):
            trunk_layers.append(nn.Linear(dim, config.hidden_dim))
            trunk_layers.append(nn.SiLU())
            dim = config.hidden_dim
        self.trunk: nn.Module = nn.Sequential(*trunk_layers)

        self.weight_head: nn.Module = _build_mlp(
            in_dim=config.hidden_dim,
            out_dim=config.n_charts,
            hidden_dim=config.hidden_dim,
            n_layers=config.head_layers,
        )

        self.latent_head: nn.Module = _build_mlp(
            in_dim=config.hidden_dim,
            out_dim=config.n_charts * config.k,
            hidden_dim=config.hidden_dim,
            n_layers=config.head_layers,
        )

        decoders: list[nn.Module] = []
        for _ in range(config.n_charts):
            dec = _build_mlp(
                in_dim=config.k,
                out_dim=config.x_dim,
                hidden_dim=config.hidden_dim,
                n_layers=config.dec_layers,
            )
            decoders.append(dec)
        self.decoders = nn.ModuleList(decoders)

    @property
    def config(self) -> SoftAtlasConfig:
        return self._config

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return assignment weights and per-chart latent codes."""
        h = self.trunk(x)  # type: ignore[return-value]
        weights = F.softmax(self.weight_head(h), dim=-1)  # type: ignore[return-value]
        all_z = self.latent_head(
            h
        ).reshape(  # type: ignore[return-value]
            *x.shape[:-1], self.config.n_charts, self.config.k
        )  # vmap-safe: keep leading dims generic.
        return weights, all_z

    def decode(self, z: torch.Tensor, chart_idx: int) -> torch.Tensor:
        """Decode latent codes through chart *chart_idx*'s decoder, ``(batch, k)`` → ``(batch, x_dim)``."""
        if chart_idx < 0 or chart_idx >= self.config.n_charts:
            raise IndexError(
                f"chart_idx {chart_idx} out of range [0, {self.config.n_charts})"
            )
        return self.decoders[chart_idx](z)  # type: ignore[return-value]

    def soft_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Weight-averaged latent code ``sum_j w_j z_j``, shape ``(batch, k)``."""
        weights, all_z = self.encode(x)
        return (weights.unsqueeze(-1) * all_z).sum(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the learned manifold: :math:`\\hat{x} = \\sum_j w_j D_j(z_j)`."""
        weights, all_z = self.encode(x)
        x_hat = torch.zeros_like(x)
        for j in range(self.config.n_charts):
            x_j = self.decoders[j](all_z[..., j, :])
            x_hat = x_hat + weights[..., j : j + 1] * x_j
        return x_hat
