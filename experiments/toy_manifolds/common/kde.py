"""KDE wrappers around ``scipy.stats.gaussian_kde``."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from scipy.stats import gaussian_kde


def fit_kde(
    uv_points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], np.ndarray]:
    """Fit a Gaussian KDE to parameter-space points ``(N, D)``."""
    data_np = uv_points.detach().cpu().numpy().T
    w_np = None
    if weights is not None:
        w_np = weights.detach().cpu().numpy()

    kde = gaussian_kde(data_np, weights=w_np)

    def evaluate_fn(uv_query: torch.Tensor) -> np.ndarray:
        query_np = uv_query.detach().cpu().numpy().T
        return kde.evaluate(query_np)

    return evaluate_fn


def evaluate_kde(
    kde: Callable[[torch.Tensor], np.ndarray],
    uv_query: torch.Tensor,
) -> np.ndarray:
    return kde(uv_query)
