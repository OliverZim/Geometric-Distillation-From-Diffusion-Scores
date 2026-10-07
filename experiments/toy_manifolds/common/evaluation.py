"""Uniformity evaluation for volume-guided generation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr

from .kde import fit_kde
from .manifolds import Manifold


def evaluate_uniformity(
    points_3d: torch.Tensor,
    manifold: Manifold,
    return_arrays: bool = False,
    fit_mask_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate uniformity of generated points against the analytic volume element.

    With *fit_mask_only*, the KDE is fit on the masked points only;
    otherwise it is fit on all points and evaluated on the masked subset.
    """
    n_total = points_3d.shape[0]

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    uv = manifold.inverse(x, y, z)

    u, v = uv[:, 0], uv[:, 1]
    mask = manifold.boundary_mask(u, v)
    uv_masked = uv[mask]
    n_masked = uv_masked.shape[0]

    if n_masked < 2:
        return _empty_result(n_total, n_masked, return_arrays)

    kde_fn = fit_kde(uv_masked if fit_mask_only else uv)
    density = kde_fn(uv_masked)

    vol_elem = manifold.volume_element(uv_masked[:, 0], uv_masked[:, 1])
    vol_elem_np = vol_elem.detach().cpu().numpy()

    finite_mask = np.isfinite(density) & np.isfinite(vol_elem_np)
    if finite_mask.sum() < 2:
        return _empty_result(n_total, n_masked, return_arrays)

    R, _ = pearsonr(density[finite_mask], vol_elem_np[finite_mask])
    R2 = R**2

    result: Dict[str, Any] = {
        "R": float(R),
        "R2": float(R2),
        "n_masked": int(n_masked),
        "n_total": int(n_total),
    }

    if return_arrays:
        result["uv_masked"] = uv_masked.cpu()
        result["density"] = density
        result["vol_elem"] = vol_elem_np

    return result


def manifold_distance(
    points_3d: torch.Tensor,
    manifold: Manifold,
    n_per_axis: int = 250,
) -> np.ndarray:
    """Per-point Euclidean distance to the analytic manifold surface."""
    x = points_3d.detach().cpu().numpy()

    if manifold.name == "hemisphere":
        bounds = (-1.0, 1.0)
    else:
        bounds = (-2.5, 2.5)
    u = torch.linspace(*bounds, n_per_axis)
    v = torch.linspace(*bounds, n_per_axis)
    U, V = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([U.ravel(), V.ravel()], dim=-1)
    if manifold.name == "hemisphere":
        uv = uv[(uv**2).sum(-1) <= 1.0]
    surface = manifold.parametrization(uv[:, 0], uv[:, 1])

    x_t = torch.from_numpy(x).float()
    init = np.empty((x.shape[0], 2))
    chunk = 512
    for i in range(0, x.shape[0], chunk):
        d2 = ((x_t[i : i + chunk, None, :] - surface[None, :, :]) ** 2).sum(-1)
        init[i : i + chunk] = uv[d2.argmin(-1)].numpy()

    def surface_at(uvv: np.ndarray) -> np.ndarray:
        return manifold.parametrization(
            torch.tensor(uvv[0]), torch.tensor(uvv[1])
        ).numpy()

    dists = np.empty(x.shape[0])
    for i, xi in enumerate(x):
        res = minimize(
            lambda uvv: float(((surface_at(uvv) - xi) ** 2).sum()),
            init[i],
            method="L-BFGS-B",
            bounds=[bounds, bounds],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        dists[i] = np.sqrt(res.fun)
    return dists


def summarize_manifold_distances(d: np.ndarray) -> Dict[str, float]:
    """Summarize per-point distances to the manifold surface as scalar metrics."""
    d = np.asarray(d, dtype=np.float64)
    pcts = np.percentile(d, [25, 50, 75, 90, 95, 99])
    metrics: Dict[str, float] = {
        "dist_mean": float(d.mean()),
        "dist_median": float(pcts[1]),
        "dist_p25": float(pcts[0]),
        "dist_p75": float(pcts[2]),
        "dist_p90": float(pcts[3]),
        "dist_p95": float(pcts[4]),
        "dist_p99": float(pcts[5]),
        "dist_max": float(d.max()),
    }
    for thr in (0.01, 0.05, 0.1, 0.3):
        metrics[f"frac_dist_lt_{thr:g}"] = float((d < thr).mean())
    return metrics


def evaluate_manifold_distance(
    points_3d: torch.Tensor,
    manifold: Manifold,
) -> Dict[str, float]:
    """Summarize the distance of generated points to the manifold surface."""
    return summarize_manifold_distances(manifold_distance(points_3d, manifold))


def _empty_result(
    n_total: int,
    n_masked: int,
    return_arrays: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "R": 0.0,
        "R2": 0.0,
        "n_masked": int(n_masked),
        "n_total": int(n_total),
    }
    if return_arrays:
        result["uv_masked"] = torch.empty(0, 2)
        result["density"] = np.array([])
        result["vol_elem"] = np.array([])
    return result
