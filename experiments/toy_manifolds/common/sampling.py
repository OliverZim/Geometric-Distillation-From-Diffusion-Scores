"""Imbalanced and uniform data generation for toy-manifold experiments."""

from __future__ import annotations

from typing import Tuple

import torch

from .manifolds import Manifold


def sample_imbalanced(
    manifold: Manifold,
    n_points: int = 3000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate imbalanced training data on a toy manifold (GAGA protocol)."""
    points_3d_list: list[torch.Tensor] = []
    uv_list: list[torch.Tensor] = []
    collected = 0

    mean = torch.tensor([1.0, 1.0])
    cov = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    L = torch.linalg.cholesky(cov)

    while collected < n_points:
        batch_size = int(1.5 * (n_points - collected))
        batch_size = max(batch_size, 1)

        raw = torch.randn(batch_size, 2)
        uv = mean.unsqueeze(0) + raw @ L.T

        # Hemisphere: keep off the z = 0 clamp boundary.
        in_bounds = (
            (uv[:, 0] >= -2.0)
            & (uv[:, 0] <= 2.0)
            & (uv[:, 1] >= -2.0)
            & (uv[:, 1] <= 2.0)
        )
        if manifold.name == "hemisphere":
            in_bounds &= (uv[:, 0] ** 2 + uv[:, 1] ** 2) < 1.0
        uv = uv[in_bounds]

        if uv.shape[0] == 0:
            continue

        u, v = uv[:, 0], uv[:, 1]
        points_3d = manifold.parametrization(u, v)

        remaining = n_points - collected
        take = min(uv.shape[0], remaining)
        points_3d_list.append(points_3d[:take])
        uv_list.append(uv[:take])
        collected += take

    return torch.cat(points_3d_list, dim=0), torch.cat(uv_list, dim=0)


def sample_uniform(
    manifold: Manifold,
    n_points: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate points uniformly on a toy manifold's parameter domain."""
    u_min, u_max = manifold.u_bounds
    v_min, v_max = manifold.v_bounds

    u = torch.rand(n_points) * (u_max - u_min) + u_min
    v = torch.rand(n_points) * (v_max - v_min) + v_min

    if manifold.name == "hemisphere":
        uv_list: list[torch.Tensor] = []
        points_list: list[torch.Tensor] = []

        all_u = u
        all_v = v
        uv_candidates = torch.stack([all_u, all_v], dim=-1)
        valid = uv_candidates[:, 0] ** 2 + uv_candidates[:, 1] ** 2 < 1.0
        uv_valid = uv_candidates[valid]
        points_valid = manifold.parametrization(uv_valid[:, 0], uv_valid[:, 1])

        if uv_valid.shape[0] >= n_points:
            uv_list.append(uv_valid[:n_points])
            points_list.append(points_valid[:n_points])
        else:
            uv_list.append(uv_valid)
            points_list.append(points_valid)
            n_remaining = n_points - uv_valid.shape[0]
            while n_remaining > 0:
                batch_u = torch.rand(n_remaining) * (u_max - u_min) + u_min
                batch_v = torch.rand(n_remaining) * (v_max - v_min) + v_min
                uv_batch = torch.stack([batch_u, batch_v], dim=-1)
                valid_batch = uv_batch[:, 0] ** 2 + uv_batch[:, 1] ** 2 < 1.0
                uv_batch = uv_batch[valid_batch]
                points_batch = manifold.parametrization(uv_batch[:, 0], uv_batch[:, 1])
                take = min(n_remaining, uv_batch.shape[0])
                uv_list.append(uv_batch[:take])
                points_list.append(points_batch[:take])
                n_remaining -= take

        return torch.cat(points_list, dim=0), torch.cat(uv_list, dim=0)

    uv = torch.stack([u, v], dim=-1)
    points_3d = manifold.parametrization(u, v)
    return points_3d, uv
