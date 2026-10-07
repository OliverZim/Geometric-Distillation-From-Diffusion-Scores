"""Metric utilities for the synthetic scRNA-seq pipeline."""

from __future__ import annotations

import numpy as np
import ot


def wasserstein_1(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact 1-Wasserstein distance with uniform marginals and Euclidean cost."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    cost = np.sqrt(ot.dist(X, Y, metric="sqeuclidean"))
    return float(ot.emd2(np.ones(n) / n, np.ones(m) / m, cost))


def off_manifold_residual(X: np.ndarray, projector, *, chunk: int = 512) -> np.ndarray:
    """Per-point residual ``||x - P(x)||`` under a frozen projector."""
    import torch

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    device = None
    if hasattr(projector, "parameters"):
        try:
            device = next(projector.parameters()).device
        except StopIteration:
            device = None
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            xb = torch.from_numpy(X[start : start + chunk])
            if device is not None:
                xb = xb.to(device)
            out[start : start + chunk] = (xb - projector(xb)).cpu().norm(dim=-1).numpy()
    return out


def check_lengths(tree_gt: np.ndarray, lengths: dict[str, np.ndarray]) -> None:
    """Validate shape, finiteness and non-negativity of aligned length arrays."""
    arrays = {"L_tree": tree_gt, **lengths}
    for name, values in arrays.items():
        if values.shape != tree_gt.shape:
            raise ValueError(
                f"{name} has shape {values.shape}, expected {tree_gt.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"{name} contains negative lengths")


def summarize_lengths(tree_gt: np.ndarray, lengths: dict[str, np.ndarray]) -> dict:
    """Return the per-method mean squared length error against the tree GT."""
    summary: dict = {
        "n": int(tree_gt.size),
        "gt_mean": float(tree_gt.mean()),
    }
    for name, length in lengths.items():
        summary[name] = {"mse": float(np.mean((length - tree_gt) ** 2))}
    return summary
