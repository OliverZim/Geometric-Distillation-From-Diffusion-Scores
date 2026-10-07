"""Warped pullback metric from the extended feature map ``f⁺``."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch.func import jvp, vmap

from ..config import ComponentConfig


class WarpedPullbackMetricConfig(ComponentConfig):
    """Configuration for the warped pullback metric :math:`f^+`."""

    beta: float = Field(
        default=10.0,
        gt=0.0,
        description="Warp scale of the off-manifold residual signal.",
    )
    space: Literal["latent", "ambient"] = Field(
        default="latent",
        description=(
            "On-manifold feature map: 'latent' = encoder; 'ambient' = projector."
        ),
    )


# Keeps the residual norm differentiable at on-manifold points.
_RESIDUAL_FLOOR = 1e-8


def compute_f_plus(
    x: torch.Tensor,
    projector: nn.Module,
    config: WarpedPullbackMetricConfig,
) -> torch.Tensor:
    r"""Extended feature map :math:`f^+(x) = [f(x);\ \beta\,s_{\mathrm{proj}}(x)]`."""
    if hasattr(projector, "soft_latent"):
        z = projector.soft_latent(x)
        x_hat = projector(x)
    else:
        z = projector.encode(x)
        x_hat = projector.decode(z)
    residual_sq = ((x - x_hat) ** 2).sum(dim=-1)
    # Python float breaks MPS jvp; keep the floor device/dtype-matched.
    floor = torch.as_tensor(
        _RESIDUAL_FLOOR, dtype=residual_sq.dtype, device=residual_sq.device
    )
    s = torch.sqrt(residual_sq + floor)
    if config.space == "ambient":
        return torch.cat([x_hat, config.beta * s.unsqueeze(-1)], dim=-1)
    return torch.cat([z, config.beta * s.unsqueeze(-1)], dim=-1)


def pullback_velocity(
    x: torch.Tensor,
    v: torch.Tensor,
    projector: nn.Module,
    config: WarpedPullbackMetricConfig,
) -> torch.Tensor:
    r"""Pull back velocities with the Jacobian of :math:`f^+`."""

    def f_plus(xx: torch.Tensor) -> torch.Tensor:
        return compute_f_plus(xx, projector, config)

    return vmap(lambda xx, vv: jvp(f_plus, (xx,), (vv,))[1])(x, v)
