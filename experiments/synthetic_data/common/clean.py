"""Clean-target embeddings and tree ground truth for the synthetic scRNA-seq data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from experiments.synthetic_data.common.dataset import load_simulation
from experiments.synthetic_data.common.preprocess import preprocess

__all__ = [
    "CLEAN_MATRICES",
    "CleanTree",
    "build_clean_tree",
    "load_clean_tree",
    "load_clean_working_space",
    "load_clean_working_spaces",
    "save_clean_tree",
    "tree_distances",
    "tree_segments",
    "distance_to_tree",
]

CLEAN_MATRICES: tuple[str, ...] = ("base_cell_means", "cell_means")


class _CleanEmbeddingSetup(NamedTuple):
    """Masks and PCA fit shared by every clean-matrix embedding."""

    kept_cells: np.ndarray
    kept_genes: np.ndarray
    kept_gene_pos: np.ndarray
    components: np.ndarray
    mean: np.ndarray


def _clean_embedding_setup(
    scenario: str, data_root: str | Path, geometry: dict
) -> _CleanEmbeddingSetup:
    """Recompute the notebook masks and validate the stored PCA fit."""
    data_dir = Path(data_root) / scenario

    ds = load_simulation(scenario, data_root)
    X_noisy, kept_cells, kept_genes = preprocess(ds.counts, mode="notebook")
    del X_noisy

    cell_index = np.asarray(geometry["cell_index"])
    assert np.array_equal(cell_index, np.flatnonzero(kept_cells)), (
        "geometry['cell_index'] does not match the notebook preprocessing of "
        f"{data_dir}: the stored Stage I artifacts were built with different "
        "masks (or a different dataset revision)"
    )

    pca = geometry["pca"]
    components = pca.components_
    mean = pca.mean_
    kept_gene_pos = np.flatnonzero(kept_genes)
    if components.shape[1] != len(kept_gene_pos) or mean.shape[0] != len(kept_gene_pos):
        raise ValueError(
            f"PCA fit does not match the preprocessing masks: components "
            f"{components.shape} / mean {mean.shape} vs {len(kept_gene_pos)} "
            "kept genes"
        )

    return _CleanEmbeddingSetup(
        kept_cells=np.asarray(kept_cells),
        kept_genes=np.asarray(kept_genes),
        kept_gene_pos=kept_gene_pos,
        components=components,
        mean=mean,
    )


def _clean_matrix_path(scenario: str, data_root: str | Path, matrix: str) -> Path:
    """Return the path of one exported clean matrix."""
    path = Path(data_root) / scenario / f"{matrix}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean targets not found: {path} "
            "(regenerate with "
            "experiments/synthetic_data/synthetic/data_generation/simulate_splatter.R)"
        )
    return path


def _validate_clean_matrix(matrix: str) -> None:
    if matrix not in CLEAN_MATRICES:
        raise ValueError(f"matrix must be one of {CLEAN_MATRICES}, got {matrix!r}")


def _embed_clean_matrix(
    clean: np.ndarray,
    matrix: str,
    setup: _CleanEmbeddingSetup,
    chunk: int = 1000,
) -> np.ndarray:
    """Stream one clean matrix into the PCA working space using its own row sums.

    ``clean`` is the dense genes×cells array (or an ``mmap_mode="r"`` view);
    only the kept genes and cells are touched.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk}")

    expected_shape = (len(setup.kept_genes), len(setup.kept_cells))
    if clean.shape != expected_shape:
        raise ValueError(
            f"{matrix} shape {clean.shape} does not match the counts "
            f"matrix {expected_shape} (genes, cells)"
        )

    n_cells = int(setup.kept_cells.sum())
    n_pcs = setup.components.shape[0]

    rowsum = np.zeros(n_cells, dtype=np.float64)
    for g0 in range(0, len(setup.kept_gene_pos), chunk):
        rows = setup.kept_gene_pos[g0 : g0 + chunk]
        rowsum += clean[rows][:, setup.kept_cells].sum(axis=0)
    if not np.all(rowsum > 0):
        bad = int((rowsum <= 0).sum())
        raise ValueError(
            f"{bad} kept cells have zero clean expression over the kept genes "
            f"({matrix})"
        )

    scores = np.zeros((n_cells, n_pcs), dtype=np.float64)
    for g0 in range(0, len(setup.kept_gene_pos), chunk):
        g1 = min(g0 + chunk, len(setup.kept_gene_pos))
        rows = setup.kept_gene_pos[g0:g1]
        x = np.log1p(clean[rows][:, setup.kept_cells] / rowsum)
        scores += x.T @ setup.components[:, g0:g1].T
    scores -= setup.components @ setup.mean

    if not np.isfinite(scores).all():
        raise ValueError(
            f"clean working space for {matrix!r} contains non-finite values"
        )
    return scores.astype(np.float32)


def load_clean_working_space(
    scenario: str,
    data_root: str | Path,
    geometry: dict,
    *,
    matrix: str = "base_cell_means",
    chunk: int = 1000,
) -> np.ndarray:
    """Embed one clean matrix into the geometry's PCA working space.

    The result is row-aligned with ``geometry["X_pca"]``. Normalization uses
    the cell's own filtered row sum of the clean matrix, mirroring the
    notebook preprocessing over the kept genes.
    """
    _validate_clean_matrix(matrix)
    clean_path = _clean_matrix_path(scenario, data_root, matrix)
    setup = _clean_embedding_setup(scenario, data_root, geometry)
    return _embed_clean_matrix(
        np.load(clean_path, mmap_mode="r"), matrix, setup, chunk=chunk
    )


def load_clean_working_spaces(
    scenario: str,
    data_root: str | Path,
    geometry: dict,
    *,
    matrices: tuple[str, ...] = CLEAN_MATRICES,
    chunk: int = 1000,
) -> dict[str, np.ndarray]:
    """Embed several clean matrices, sharing one mask/PCA setup pass."""
    if isinstance(matrices, str):
        raise TypeError("matrices must be a sequence of names, not a string")
    matrices = tuple(matrices)
    for matrix in matrices:
        _validate_clean_matrix(matrix)

    paths = {
        matrix: _clean_matrix_path(scenario, data_root, matrix) for matrix in matrices
    }
    setup = _clean_embedding_setup(scenario, data_root, geometry)
    return {
        matrix: _embed_clean_matrix(
            np.load(paths[matrix], mmap_mode="r"), matrix, setup, chunk=chunk
        )
        for matrix in matrices
    }


def _natural_key(name: str) -> tuple[tuple[int, object], ...]:
    """Sort key that orders ``Path2`` before ``Path10``."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", str(name))
    )


@dataclass(frozen=True)
class CleanTree:
    """Discretised clean manifold tree of the Splatter paths.

    Each path is an ordered polyline through its unique ``(Path, Step)`` clean
    points, children attach to their parent's final point, and all arc
    coordinates are cumulative Euclidean lengths in the PCA working space.
    """

    path_names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    points: np.ndarray
    point_start: np.ndarray
    n_points: np.ndarray
    cum_arc: np.ndarray
    cell_path: np.ndarray
    cell_arc: np.ndarray
    junction_max_dist: float
    step_spread_max: float
    junction_tol: float

    @property
    def root(self) -> int:
        """Index of the root path."""
        return int(np.flatnonzero(self.parents == -1)[0])

    def path_points(self, path: int) -> np.ndarray:
        """Polyline points of one path, ordered by ``Step``."""
        start = int(self.point_start[path])
        return self.points[start : start + int(self.n_points[path])]


def build_clean_tree(
    X_clean_pca: np.ndarray,
    labels: pd.DataFrame,
    junction_tol_factor: float = 1e-6,
) -> CleanTree:
    """Build the ground-truth tree from clean points and per-cell labels.

    Each ``(Path, Step)`` group is collapsed to its mean point; a path attaches
    to the path whose final point is nearest to its first point within
    ``junction_tol_factor * extent``, and exactly one root is required.
    """
    X = np.asarray(X_clean_pca, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X_clean_pca must be 2-D, got shape {X.shape}")
    if X.shape[0] != len(labels):
        raise ValueError(
            f"X_clean_pca has {X.shape[0]} rows but labels has {len(labels)}"
        )
    if X.shape[0] == 0:
        raise ValueError("X_clean_pca is empty")
    if not np.isfinite(X).all():
        raise ValueError("X_clean_pca contains non-finite values")
    if not junction_tol_factor > 0:
        raise ValueError(
            f"junction_tol_factor must be positive, got {junction_tol_factor}"
        )
    missing = [c for c in ("Path", "Step") if c not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required columns: {missing}")
    if np.asarray(labels["Path"].isna()).any():
        raise ValueError("labels['Path'] contains missing values")
    if np.asarray(labels["Step"].isna()).any():
        raise ValueError("labels['Step'] contains missing values")

    path_values = labels["Path"].astype(str).to_numpy()
    step_values = np.asarray(labels["Step"], dtype=np.float64)
    unique_paths = sorted(set(path_values.tolist()), key=_natural_key)

    points_list: list[np.ndarray] = []
    cum_arc_list: list[np.ndarray] = []
    spread_max = 0.0
    cell_path = np.empty(X.shape[0], dtype=np.int64)
    cell_arc = np.empty(X.shape[0], dtype=np.float64)

    for path_idx, name in enumerate(unique_paths):
        rows = np.flatnonzero(path_values == name)
        steps = step_values[rows]
        if not np.all(steps == np.floor(steps)):
            raise ValueError(f"path {name!r} has non-integral Step values")
        unique_steps, inverse = np.unique(steps.astype(np.int64), return_inverse=True)
        if unique_steps.size < 2:
            raise ValueError(
                f"path {name!r} has {unique_steps.size} unique Step value(s); "
                "at least 2 are needed to form a polyline"
            )

        group = X[rows]
        step_points = np.zeros((unique_steps.size, X.shape[1]), dtype=np.float64)
        np.add.at(step_points, inverse, group)
        counts = np.bincount(inverse, minlength=unique_steps.size)
        step_points /= counts[:, None]

        spread = np.linalg.norm(group - step_points[inverse], axis=1)
        per_step_spread = np.zeros(unique_steps.size, dtype=np.float64)
        np.maximum.at(per_step_spread, inverse, spread)
        spread_max = max(spread_max, float(per_step_spread.max()))

        segment = np.linalg.norm(np.diff(step_points, axis=0), axis=1)
        cum_arc = np.concatenate([[0.0], np.cumsum(segment)])
        points_list.append(step_points)
        cum_arc_list.append(cum_arc)
        cell_path[rows] = path_idx
        cell_arc[rows] = cum_arc[inverse]

    n_paths = len(unique_paths)
    n_points = np.array([p.shape[0] for p in points_list], dtype=np.int64)
    point_start = np.concatenate([[0], np.cumsum(n_points)[:-1]]).astype(np.int64)
    points = np.concatenate(points_list, axis=0).astype(np.float32)
    cum_arc = np.concatenate(cum_arc_list)
    lengths = np.array([c[-1] for c in cum_arc_list], dtype=np.float64)

    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent <= 0:
        raise ValueError("clean points have zero extent")
    tol = junction_tol_factor * extent

    starts = points[point_start].astype(np.float64)
    ends = points[point_start + n_points - 1].astype(np.float64)
    parents = np.full(n_paths, -1, dtype=np.int64)
    junction_dist = np.full(n_paths, np.inf, dtype=np.float64)
    for path_idx in range(n_paths):
        d = np.linalg.norm(ends - starts[path_idx], axis=1)
        d[path_idx] = np.inf
        candidate = int(np.argmin(d))
        if d[candidate] <= tol:
            parents[path_idx] = candidate
            junction_dist[path_idx] = d[candidate]

    roots = np.flatnonzero(parents == -1)
    if roots.size != 1:
        raise ValueError(
            "expected exactly one root path (a path with no other path's final "
            f"point within junction tolerance {tol:.3e}), found {roots.size}: "
            f"{[unique_paths[int(r)] for r in roots]}"
        )
    for path_idx in range(n_paths):
        seen: set[int] = set()
        node = path_idx
        while node != -1:
            if node in seen:
                raise ValueError(
                    "path attachment graph contains a cycle through "
                    f"{[unique_paths[k] for k in seen]}"
                )
            seen.add(node)
            node = int(parents[node])

    children: dict[int, list[int]] = {p: [] for p in range(n_paths)}
    for path_idx in range(n_paths):
        if parents[path_idx] != -1:
            children[int(parents[path_idx])].append(path_idx)
    offsets = np.zeros(n_paths, dtype=np.float64)
    order = [int(roots[0])]
    for path_idx in order:
        for child in children[path_idx]:
            offsets[child] = offsets[path_idx] + lengths[path_idx]
            order.append(child)
    if len(order) != n_paths:
        raise ValueError(
            "path attachment graph is disconnected: only "
            f"{len(order)}/{n_paths} paths are reachable from the root "
            f"{unique_paths[int(roots[0])]!r}"
        )

    non_root = parents != -1
    return CleanTree(
        path_names=tuple(unique_paths),
        parents=parents,
        offsets=offsets,
        lengths=lengths,
        points=points,
        point_start=point_start,
        n_points=n_points,
        cum_arc=cum_arc,
        cell_path=cell_path,
        cell_arc=cell_arc,
        junction_max_dist=float(junction_dist[non_root].max())
        if non_root.any()
        else 0.0,
        step_spread_max=spread_max,
        junction_tol=tol,
    )


def _path_depths(parents: np.ndarray) -> np.ndarray:
    """Depth of each path in the attachment tree (root depth 0)."""
    depths = np.zeros(parents.shape[0], dtype=np.int64)
    for path_idx in range(parents.shape[0]):
        node, depth = path_idx, 0
        while parents[node] != -1:
            node = int(parents[node])
            depth += 1
        depths[path_idx] = depth
    return depths


def _lca_paths(
    parents: np.ndarray, depths: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Deepest common ancestor path index per pair ``(a[k], b[k])``."""
    a = a.copy()
    b = b.copy()
    for _ in range(int(parents.shape[0]) + 1):
        same = a == b
        if same.all():
            return a
        depth_a = depths[a]
        depth_b = depths[b]
        climb_a = (~same) & (depth_a >= depth_b)
        climb_b = (~same) & (depth_b >= depth_a)
        a[climb_a] = parents[a[climb_a]]
        b[climb_b] = parents[b[climb_b]]
    raise RuntimeError("LCA did not converge — the attachment graph is not a tree")


def tree_distances(tree: CleanTree, cells: np.ndarray) -> np.ndarray:
    """Return the exact tree arc length for each pair of cell indices."""
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError(
            f"cells must have shape (n, 2), got {cells.shape if cells.ndim else ()}"
        )
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError(f"cells must have an integer dtype, got {cells.dtype}")
    n_cells = tree.cell_path.shape[0]
    if cells.size == 0:
        return np.zeros(0, dtype=np.float64)
    if cells.min() < 0 or cells.max() >= n_cells:
        raise ValueError(
            f"cell indices must be in [0, {n_cells}), got "
            f"[{cells.min()}, {cells.max()}]"
        )

    path_i = tree.cell_path[cells[:, 0]]
    path_j = tree.cell_path[cells[:, 1]]
    t_i = tree.offsets[path_i] + tree.cell_arc[cells[:, 0]]
    t_j = tree.offsets[path_j] + tree.cell_arc[cells[:, 1]]

    depths = _path_depths(tree.parents)
    lca = _lca_paths(tree.parents, depths, path_i, path_j)
    t_lca = np.where(
        lca == path_i,
        t_i,
        np.where(lca == path_j, t_j, tree.offsets[lca] + tree.lengths[lca]),
    )
    distances = np.where(path_i == path_j, np.abs(t_i - t_j), t_i + t_j - 2 * t_lca)
    distances = np.maximum(distances, 0.0)
    if not np.isfinite(distances).all():
        raise ValueError("tree_distances produced non-finite values")
    return distances


def tree_segments(tree: CleanTree) -> tuple[np.ndarray, np.ndarray]:
    """Return the clean-tree polyline segments as ``(seg_a, seg_b)``."""
    seg_a: list[np.ndarray] = []
    seg_b: list[np.ndarray] = []
    for path in range(len(tree.path_names)):
        points = tree.path_points(path).astype(np.float64)
        seg_a.append(points[:-1])
        seg_b.append(points[1:])
    return np.concatenate(seg_a), np.concatenate(seg_b)


def distance_to_tree(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    chunk: int = 20000,
) -> np.ndarray:
    """Return the exact Euclidean distance of each point to the tree polyline."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, points.shape[-1])
    best = np.full(len(points), np.inf)
    for a, b in zip(seg_a, seg_b):
        ab = b - a
        length_sq = ab @ ab
        for start in range(0, len(points), chunk):
            query = points[start : start + chunk]
            t = np.clip((query - a) @ ab / length_sq, 0.0, 1.0)
            d_sq = ((query - (a + t[:, None] * ab)) ** 2).sum(1)
            np.minimum(
                best[start : start + chunk], d_sq, out=best[start : start + chunk]
            )
    return np.sqrt(best)


def save_clean_tree(tree: CleanTree, path: str | Path) -> None:
    """Serialise a :class:`CleanTree` to a compressed ``.npz`` artifact."""
    np.savez_compressed(
        path,
        path_names=np.asarray(tree.path_names, dtype=str),
        parents=tree.parents,
        offsets=tree.offsets,
        lengths=tree.lengths,
        points=tree.points,
        point_start=tree.point_start,
        n_points=tree.n_points,
        cum_arc=tree.cum_arc,
        cell_path=tree.cell_path,
        cell_arc=tree.cell_arc,
        junction_max_dist=np.float64(tree.junction_max_dist),
        step_spread_max=np.float64(tree.step_spread_max),
        junction_tol=np.float64(tree.junction_tol),
    )


def load_clean_tree(path: str | Path) -> CleanTree:
    """Load a :class:`CleanTree` written by :func:`save_clean_tree`."""
    with np.load(path) as data:
        return CleanTree(
            path_names=tuple(str(name) for name in data["path_names"]),
            parents=data["parents"].astype(np.int64),
            offsets=data["offsets"].astype(np.float64),
            lengths=data["lengths"].astype(np.float64),
            points=data["points"].astype(np.float32),
            point_start=data["point_start"].astype(np.int64),
            n_points=data["n_points"].astype(np.int64),
            cum_arc=data["cum_arc"].astype(np.float64),
            cell_path=data["cell_path"].astype(np.int64),
            cell_arc=data["cell_arc"].astype(np.float64),
            junction_max_dist=float(data["junction_max_dist"]),
            step_spread_max=float(data["step_spread_max"]),
            junction_tol=float(data["junction_tol"]),
        )
