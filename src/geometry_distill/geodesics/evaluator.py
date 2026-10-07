"""Path evaluation against a Dijkstra baseline on a background point cloud."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pydantic import Field
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..config import ComponentConfig


class ManifoldPathEvaluatorConfig(ComponentConfig):
    """Configuration for :class:`ManifoldPathEvaluator`."""

    n_neighbors: int = Field(
        default=15,
        ge=1,
        description="Number of neighbours for the k-NN graph edges.",
    )


class ManifoldPathEvaluator:
    """Evaluate predicted paths against a Dijkstra baseline on a point cloud."""

    _config: ManifoldPathEvaluatorConfig

    def __init__(self, config: ManifoldPathEvaluatorConfig, X_background):
        X = torch.as_tensor(X_background, dtype=torch.float32).numpy()
        if X.ndim != 2:
            raise ValueError(
                f"X_background must be 2-D (n_points, x_dim), got shape {X.shape}"
            )
        n_points, x_dim = X.shape
        n_neighbors = config.n_neighbors
        if n_points < n_neighbors:
            raise ValueError(
                f"X_background must have at least n_neighbors ({n_neighbors}) "
                f"points, got {n_points}"
            )
        self._config = config
        self._X = X
        self._n_neighbors = n_neighbors
        self._tree = cKDTree(X)

        dists, idx = self._tree.query(X, k=n_neighbors + 1)
        self._rows = np.repeat(np.arange(n_points), n_neighbors)
        self._cols = idx[:, 1:].ravel()
        self._data = dists[:, 1:].ravel().astype(np.float64)

    @property
    def config(self) -> ManifoldPathEvaluatorConfig:
        return self._config

    def dijkstra_distance(
        self,
        x0,
        x1,
    ) -> float:
        """Shortest path between two points on the k-NN graph."""
        n_points = self._X.shape[0]
        a0 = np.asarray(x0, dtype=np.float32).reshape(1, -1)
        a1 = np.asarray(x1, dtype=np.float32).reshape(1, -1)
        if a0.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"x0 has dimension {a0.shape[1]}, expected {self._X.shape[1]}"
            )

        d0, i0 = self._tree.query(a0, k=self._n_neighbors)
        d1, i1 = self._tree.query(a1, k=self._n_neighbors)
        d0 = np.asarray(d0).ravel().astype(np.float64)
        d1 = np.asarray(d1).ravel().astype(np.float64)

        rows = np.concatenate(
            [
                self._rows,
                np.full(self._n_neighbors, n_points, dtype=np.int64),
                np.full(self._n_neighbors, n_points + 1, dtype=np.int64),
            ]
        )
        cols = np.concatenate([self._cols, i0.ravel(), i1.ravel()])
        data = np.concatenate([self._data, d0, d1])

        graph = coo_matrix(
            (data, (rows, cols)), shape=(n_points + 2, n_points + 2)
        ).tocsr()
        dist = dijkstra(graph, directed=False, indices=n_points)
        return float(dist[n_points + 1])

    @torch.no_grad()
    def evaluate_path(
        self,
        path: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        projector: nn.Module,
    ) -> dict[str, float]:
        """Evaluate one predicted path.

        Returns:
            ``L_straight``, ``L_ambient``, ``L_dijkstra``, ``rho`` and
            ``M_off``.
        """
        path = torch.as_tensor(path, dtype=torch.float32)
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        x1 = torch.as_tensor(x1, dtype=torch.float32)

        L_straight = float(torch.linalg.vector_norm(x1 - x0).item())
        segments = torch.linalg.vector_norm(path[1:] - path[:-1], dim=-1)
        L_ambient = float(segments.sum().item())
        L_dijkstra = self.dijkstra_distance(x0.numpy(), x1.numpy())

        rho = L_ambient / L_dijkstra if L_dijkstra > 0.0 else float("nan")

        interior = path[1:-1]
        if projector is not None and interior.shape[0] > 0:
            residual_sq = ((interior - projector(interior)) ** 2).sum(dim=-1)
            M_off = float(residual_sq.mean().item())
        else:
            M_off = float("nan")

        return {
            "L_straight": L_straight,
            "L_ambient": L_ambient,
            "L_dijkstra": L_dijkstra,
            "rho": rho,
            "M_off": M_off,
        }
