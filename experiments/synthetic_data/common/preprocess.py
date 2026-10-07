"""Preprocessing for the synthetic scRNA-seq analysis."""

from __future__ import annotations

import numpy as np
from scipy import sparse

MIN_CELLS = 10
LIB_PERCENTILE = 20


def preprocess(
    counts: sparse.csr_matrix,
    mode: str = "notebook",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform the cells×genes count matrix for downstream analysis.

    Returns ``(X, kept_cells, kept_genes)`` with dense log1p-normalized values
    in ``notebook`` mode and raw counts in ``raw`` mode.
    """
    if mode not in ("notebook", "raw"):
        raise ValueError(f"Unknown preprocess mode: {mode!r} (use 'notebook' or 'raw')")

    if mode == "raw":
        return (
            counts.toarray(),
            np.ones(counts.shape[0], dtype=bool),
            np.ones(counts.shape[1], dtype=bool),
        )

    gene_mask = counts.getnnz(axis=0) >= MIN_CELLS
    counts = counts[:, gene_mask]

    lib = np.asarray(counts.sum(axis=1)).ravel()
    cutoff = np.percentile(lib, LIB_PERCENTILE)
    cell_mask = lib >= cutoff
    counts = counts[cell_mask]

    norm = sparse.diags(1.0 / lib[cell_mask]) @ counts
    return np.log1p(norm.toarray()), cell_mask, gene_mask
