"""Non-amortized path refinement under the warped metric."""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import Field

from ..config import ComponentConfig
from .metric import WarpedPullbackMetricConfig, compute_f_plus


class PathRefinementConfig(ComponentConfig):
    """Configuration for :func:`refine_paths`."""

    metric: WarpedPullbackMetricConfig = Field(
        default_factory=WarpedPullbackMetricConfig,
        description="Warped pullback metric built from the projector.",
    )
    steps: int = Field(
        default=1000,
        ge=0,
        description="Adam iterations (0 = return the initialisation).",
    )
    lr: float = Field(
        default=3e-3,
        gt=0.0,
        description="Adam learning rate (in ambient units).",
    )


def chord_paths(
    x0, x1, n_waypoints: int, device: torch.device | None = None
) -> torch.Tensor:
    """Straight chords between two endpoint clouds, shape ``(n, n_waypoints, d)``."""
    if n_waypoints < 2:
        raise ValueError(f"n_waypoints must be at least 2, got {n_waypoints}")
    t = torch.linspace(0.0, 1.0, n_waypoints, device=device).view(1, -1, 1)
    a = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(1)
    b = torch.as_tensor(x1, dtype=torch.float32, device=device).unsqueeze(1)
    return a + (b - a) * t


def arc_length_midpoints(paths: torch.Tensor, fraction: float = 0.5) -> torch.Tensor:
    """Point on each polyline at a given fraction of its arc length."""
    if paths.ndim != 3:
        raise ValueError(f"paths must be 3-D (n, m, d), got shape {paths.shape}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n, m, d = paths.shape
    if m < 2:
        raise ValueError(f"need at least 2 waypoints, got {m}")

    segment = paths[:, 1:] - paths[:, :-1]
    seg_len = segment.norm(dim=-1)
    cum = torch.cumsum(seg_len, dim=1)
    target = fraction * cum[:, -1:]
    idx = torch.searchsorted(cum, target).clamp(max=m - 2)

    seg_start = cum.gather(1, idx) - seg_len.gather(1, idx)
    denom = seg_len.gather(1, idx)
    weight = torch.where(
        denom > 0, (target - seg_start) / denom, torch.zeros_like(denom)
    ).clamp(0.0, 1.0)

    idx_pt = idx.unsqueeze(-1).expand(-1, -1, d)
    p0 = paths.gather(1, idx_pt).squeeze(1)
    p1 = paths.gather(1, (idx + 1).unsqueeze(-1).expand(-1, -1, d)).squeeze(1)
    return p0 + weight * (p1 - p0)


def discrete_energy(
    paths: torch.Tensor,
    projector: nn.Module,
    config: WarpedPullbackMetricConfig,
) -> torch.Tensor:
    """Per-path discrete pullback energy, shape ``(batch,)`` for ``paths``
    of shape ``(batch, n_points, d)``."""
    b, n, d = paths.shape
    f = compute_f_plus(paths.reshape(-1, d), projector, config).reshape(b, n, -1)
    return ((f[:, 1:] - f[:, :-1]) ** 2).sum(dim=-1).sum(dim=-1)


def refine_paths(
    paths: torch.Tensor,
    projector: nn.Module,
    config: PathRefinementConfig,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Refine waypoint paths by gradient descent on :func:`discrete_energy`.

    Endpoints are fixed. Returns the lowest-energy iterate per path and
    the energy trace.
    """
    paths = paths.detach().clone()
    p0, p1 = paths[:, :1], paths[:, -1:]
    interior = paths[:, 1:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([interior], lr=config.lr)
    with torch.no_grad():
        e0 = discrete_energy(paths, projector, config.metric)
    best_e = e0.clone()
    best = paths.clone()
    trace: list[float] = [float(e0.mean())]
    for _ in range(config.steps):
        full = torch.cat([p0, interior, p1], dim=1)
        e = discrete_energy(full, projector, config.metric)
        opt.zero_grad(set_to_none=True)
        e.sum().backward()
        opt.step()
        with torch.no_grad():
            improved = e < best_e
            best_e = torch.where(improved, e.detach(), best_e)
            best[improved] = full.detach()[improved]
            trace.append(float(e.mean()))
    return best.detach(), {
        "energy_init": e0.tolist(),
        "energy_best": best_e.tolist(),
        "energy_trace": trace[:: max(1, config.steps // 50)],
    }
