"""Graph helpers for the synthetic scRNA-seq analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def fit_pca(X: np.ndarray, n_pcs: int, seed: int) -> tuple[np.ndarray, PCA]:
    """Fit PCA on the preprocessed matrix; returns (PCs, fitted PCA)."""
    pca = PCA(n_components=n_pcs, random_state=seed)
    return pca.fit_transform(X), pca


def knn_graph(X: np.ndarray, k: int) -> sparse.csr_matrix:
    """Undirected kNN graph on ``X`` with Euclidean edge weights."""
    return knn_graphs(X, (k,))[k]


def knn_graphs(X: np.ndarray, k_values: Sequence[int]) -> dict[int, sparse.csr_matrix]:
    """Return the kNN graphs for several ``k`` from one neighbour search."""
    kmax = max(k_values)
    nn = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, idx = nn.kneighbors(X)

    n = X.shape[0]
    rows2d = np.repeat(np.arange(n), kmax + 1).reshape(n, kmax + 1)
    graphs = {}
    for k in k_values:
        dk, ik = dists[:, : k + 1], idx[:, : k + 1]
        rows_k = rows2d[:, : k + 1]
        keep = ik != rows_k
        directed = sparse.coo_matrix(
            (
                dk.ravel()[keep.ravel()],
                (rows_k.ravel()[keep.ravel()], ik.ravel()[keep.ravel()]),
            ),
            shape=(n, n),
        ).tocsr()
        graphs[k] = _symmetrize_min(directed)
    return graphs


def _symmetrize_min(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """Symmetrize by union, weight = min of the two directed weights."""
    adj_t = adj.T.tocsr()
    both_min = (adj + adj_t - abs(adj - adj_t)) * 0.5
    single = (adj != 0) != (adj_t != 0)
    return both_min + single.multiply(adj + adj_t)


def shortest_paths(
    adj: sparse.csr_matrix, indices: np.ndarray | None = None
) -> np.ndarray:
    """All-pairs or multi-source weighted Dijkstra distances."""
    return csgraph_dijkstra(adj, directed=False, indices=indices, unweighted=False)


def n_connected_components(adj: sparse.csr_matrix) -> int:
    """Number of connected components of the (undirected) kNN graph."""
    return int(connected_components(adj, directed=False)[0])


def subsample_indices(n: int, max_cells: int, seed: int) -> np.ndarray:
    """Deterministic subsample of ``max_cells`` cell indices; 0 or ``>= n`` returns all."""
    if max_cells <= 0 or max_cells >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))
