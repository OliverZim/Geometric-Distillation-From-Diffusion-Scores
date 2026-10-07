"""Langevin dynamics for manifold sampling."""

from __future__ import annotations

from typing import Callable

import torch
from pydantic import Field

from ..config import ComponentConfig


class LangevinConfig(ComponentConfig):
    """Configuration for :func:`sample_langevin`."""

    step_size: float = Field(
        default=1e-3,
        gt=0.0,
        description="Euler–Maruyama step size η.",
    )
    n_steps: int = Field(
        default=1200,
        ge=1,
        description="Number of Langevin steps.",
    )
    noise_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Noise scale γ (√(2η)·γ is the injected standard deviation).",
    )
    grad_clip: float = Field(
        default=5.0,
        ge=0.0,
        description="Per-sample L2 gradient clipping threshold; 0 disables clipping.",
    )
    detach_every_step: bool = Field(
        default=True,
        description="Detach the state after every step.",
    )
    return_trajectory: bool = Field(
        default=False,
        description="Return all intermediate states instead of only the final state.",
    )


def _clip_grad(grad: torch.Tensor, grad_clip: float) -> torch.Tensor:
    if grad_clip <= 0.0:
        return grad
    gnorm = torch.norm(grad, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(grad_clip / gnorm, max=1.0)
    return grad * scale


def sample_langevin(
    x_init: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    config: LangevinConfig,
) -> torch.Tensor:
    """Run overdamped Langevin dynamics (Euler–Maruyama).

    Returns the final state, or the full trajectory when configured.
    """
    sqrt_2eta = float((2.0 * config.step_size) ** 0.5) * float(config.noise_scale)

    x = x_init
    trajectory: list[torch.Tensor] = (
        [x_init.detach().clone()] if config.return_trajectory else []
    )

    for _ in range(config.n_steps):
        x.requires_grad_(True)
        energy = energy_fn(x)
        grad = torch.autograd.grad(
            energy.sum(), x, create_graph=False, retain_graph=False
        )[0]
        grad = _clip_grad(grad, config.grad_clip)

        xi = torch.randn_like(x)

        x = x - config.step_size * grad + sqrt_2eta * xi

        if config.detach_every_step:
            x = x.detach()

        if config.return_trajectory:
            trajectory.append(
                x.detach().clone() if config.detach_every_step else x.clone()
            )

    if config.return_trajectory:
        return torch.stack(trajectory, dim=0)

    return x
