"""Analytic toy-manifold definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from pydantic import Field, field_validator

from geometry_distill.config import ComponentConfig

#: Names of the toy manifolds.
ManifoldName = Literal["hemisphere", "saddle", "paraboloid"]


class Manifold(ComponentConfig, ABC):
    """Base configuration and analytic geometry of a toy manifold."""

    name: ManifoldName = Field(description="Manifold identifier.")
    x_dim: int = Field(
        default=3,
        ge=1,
        description="Ambient space dimension.",
    )
    k: int = Field(
        default=2,
        ge=1,
        description="Intrinsic (latent) dimension.",
    )
    u_bounds: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="``(u_min, u_max)`` of the parameter domain.",
    )
    v_bounds: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="``(v_min, v_max)`` of the parameter domain.",
    )
    plot_domain: Literal["disk", "square"] = Field(
        default="square",
        description="Shape of the parameter-space figure plot domain.",
    )
    plot_half_width: float = Field(
        default=2.0,
        gt=0.0,
        description="Half-width of the figure plot domain (radius or half-side).",
    )

    @field_validator("u_bounds", "v_bounds")
    @classmethod
    def _increasing_bounds(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        if bounds[0] >= bounds[1]:
            raise ValueError(f"bounds must be increasing, got {bounds}")
        return bounds

    @abstractmethod
    def parametrization(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Map batched 1-D parameters ``(u, v)`` to ``(batch, 3)`` points."""

    @abstractmethod
    def volume_element(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Analytic volume element at ``(u, v)``, shape ``(batch,)``."""

    @abstractmethod
    def inverse(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Map batched ambient ``(x, y, z)`` to parameters ``(batch, 2)``."""

    @abstractmethod
    def boundary_mask(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Boolean ``(batch,)`` mask of parameters inside the KDE boundary."""


class Hemisphere(Manifold):
    """Upper unit hemisphere ``z = sqrt(1 - u² - v²)`` on the unit disk."""

    name: Literal["hemisphere"] = "hemisphere"
    u_bounds: tuple[float, float] = (-1.0, 1.0)
    v_bounds: tuple[float, float] = (-1.0, 1.0)
    plot_domain: Literal["disk"] = "disk"
    plot_half_width: float = 1.0

    def parametrization(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z = torch.sqrt(torch.clamp(1.0 - u**2 - v**2, min=0.0))
        return torch.stack([u, v, z], dim=-1)

    def volume_element(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(torch.clamp(1.0 - u**2 - v**2, min=1e-10))

    def inverse(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([x, y], dim=-1)

    def boundary_mask(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u**2 + v**2) < 0.8


class Saddle(Manifold):
    """Hyperbolic paraboloid ``z = u² - v²`` over the square domain."""

    name: Literal["saddle"] = "saddle"

    def parametrization(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.stack([u, v, u**2 - v**2], dim=-1)

    def volume_element(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 + 4.0 * u**2 + 4.0 * v**2)

    def inverse(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([x, y], dim=-1)

    def boundary_mask(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u.abs() < 1.6) & (v.abs() < 1.6)


class Paraboloid(Manifold):
    """Elliptic paraboloid ``z = u² + v²`` over the square domain."""

    name: Literal["paraboloid"] = "paraboloid"

    def parametrization(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.stack([u, v, u**2 + v**2], dim=-1)

    def volume_element(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 + 4.0 * u**2 + 4.0 * v**2)

    def inverse(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([x, y], dim=-1)

    def boundary_mask(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u.abs() < 1.6) & (v.abs() < 1.6)


#: Ready-to-use manifold geometries, keyed by :data:`ManifoldName`.
MANIFOLDS: dict[ManifoldName, Manifold] = {
    "hemisphere": Hemisphere(),
    "saddle": Saddle(),
    "paraboloid": Paraboloid(),
}
