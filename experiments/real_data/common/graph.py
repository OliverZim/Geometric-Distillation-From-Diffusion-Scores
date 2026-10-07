"""Graph helpers for the real-data transport pipeline."""

from __future__ import annotations

import numpy as np


def subsample_indices(n: int, max_cells: int, seed: int) -> np.ndarray:
    """Return a deterministic subsample of ``max_cells`` indices out of ``n``."""
    if max_cells <= 0 or max_cells >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))
