"""Working space, splits, variants and ground-truth artifacts for the synthetic scRNA-seq pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from experiments.synthetic_data.common.dataset import load_simulation
from experiments.synthetic_data.common.graph import (
    knn_graphs,
    n_connected_components,
    shortest_paths,
    subsample_indices,
)
from experiments.synthetic_data.common.preprocess import preprocess

BIN_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_BINS = len(BIN_EDGES) - 1

# Held-out window of the population-transport protocol: the middle bin
# [0.4, 0.6) of every path.
HOLDOUT_MID_BINS: tuple[int, ...] = (2,)


def prepare_working_space(
    scenario: str, data_root: str | Path, n_pcs: int = 50, seed: int = 42
) -> dict:
    """Load, preprocess and PCA-project one scenario.

    Returns a dict with ``X_pca``, ``cell_index`` (kept cells in the raw cell
    order), ``labels`` (aligned with ``X_pca`` rows) and the fitted ``pca``.
    """
    ds = load_simulation(scenario, data_root)
    X, kept_cells, _ = preprocess(ds.counts, mode="notebook")
    labels = ds.labels.loc[kept_cells].reset_index(drop=True)
    cell_index = np.flatnonzero(kept_cells).astype(np.int64)

    n_components = min(n_pcs, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X).astype(np.float32)

    assert X_pca.shape[0] == len(labels) == len(cell_index), (
        f"cell count mismatch: {X_pca.shape[0]} PCA rows, {len(labels)} labels, "
        f"{len(cell_index)} cell indices"
    )
    assert (labels["Cell"].values == ds.cell_ids[cell_index]).all(), (
        "labels/Cell misalignment with the filtered cell order"
    )

    return {"X_pca": X_pca, "cell_index": cell_index, "labels": labels, "pca": pca}


def make_splits(labels: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 90/10 train/val split by ``Path``; returns sorted index arrays."""
    n = len(labels)
    train_idx, val_idx = train_test_split(
        np.arange(n),
        test_size=0.1,
        stratify=labels["Path"].values,
        random_state=seed,
    )
    return np.sort(train_idx).astype(np.int64), np.sort(val_idx).astype(np.int64)


def make_bins_and_roots(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-path pseudotime bins (5 fixed bins) and the global root cell indices."""
    n = len(labels)
    bin_id = np.zeros(n, dtype=np.int8)
    edges = np.asarray(BIN_EDGES)
    for path in pd.unique(labels["Path"]):
        mask = (labels["Path"] == path).values
        pt = labels.loc[mask, "Pseudotime"].values
        b = np.searchsorted(edges[1:], pt, side="right")
        bin_id[mask] = np.minimum(b, N_BINS - 1)

    pg = labels["PseudotimeGlobal"].values
    root_idx = np.flatnonzero(pg == pg.min()).astype(np.int64)
    return bin_id, root_idx


def make_variants(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    bin_id: np.ndarray,
) -> dict:
    """Return the ``full`` and ``holdout_mid`` train/val/held-out index masks."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)

    variants: dict[str, dict[str, np.ndarray]] = {
        "full": {"train": train_idx, "val": val_idx, "held_out": empty},
    }

    held = np.isin(bin_id[train_idx], HOLDOUT_MID_BINS)
    variants["holdout_mid"] = {
        "train": np.sort(train_idx[~held]).astype(np.int64),
        "val": val_idx,
        "held_out": np.sort(train_idx[held]).astype(np.int64),
    }

    return variants


def seeded_subsample(idx: np.ndarray, n: int) -> np.ndarray:
    """Seeded subsample of ``n`` entries from ``idx``; fewer entries returns all."""
    idx = np.asarray(idx, dtype=np.int64)
    n = min(int(n), len(idx))
    if n == len(idx):
        return idx
    return np.random.choice(idx, size=n, replace=False).astype(np.int64)


def build_train_val(
    geom: dict, variant: str, n_val: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, held_out)`` for one data variant.

    ``val_idx`` is a seeded subsample of the val split for ``full`` and of the
    variant's kept train cells otherwise. Raises if the variant is unknown or
    if any split leaks held-out cells.
    """
    variants = geom["variants"]
    if variant not in variants:
        raise ValueError(
            f"variant '{variant}' not found in variants.npz "
            f"(scenario has: {sorted(variants)})"
        )
    mask = variants[variant]
    train_idx = np.asarray(mask["train"], dtype=np.int64)
    held_out = np.asarray(mask["held_out"], dtype=np.int64)
    variant_val = np.asarray(mask["val"], dtype=np.int64)

    if variant == "full":
        val_source = np.asarray(geom["val_idx"], dtype=np.int64)
    else:
        val_source = train_idx
    val_idx = seeded_subsample(val_source, n_val)

    assert len(train_idx) == len(mask["train"]), (
        "train size differs from the variant mask"
    )
    assert set(train_idx.tolist()) == set(
        np.asarray(mask["train"], dtype=np.int64).tolist()
    ), "train set differs from the variant mask"
    assert not np.isin(held_out, train_idx).any(), "held_out cells leaked into train"
    assert not np.isin(held_out, val_idx).any(), "held_out cells leaked into val"
    assert len(val_idx) == min(int(n_val), len(val_source)), (
        "val size must be min(n_val, len(val source))"
    )
    if variant == "full":
        assert not np.isin(val_idx, train_idx).any(), (
            "train/val overlap for the full variant"
        )
        assert set(train_idx.tolist()) == set(
            np.asarray(geom["train_idx"], dtype=np.int64).tolist()
        ), "full variant train must equal split.npz train_idx"
    else:
        assert np.isin(val_idx, train_idx).all(), (
            "variant val must be a subsample of the variant's kept train cells"
        )
    assert not np.isin(held_out, variant_val).any(), (
        "held_out cells leaked into variant val"
    )
    return train_idx, val_idx, held_out


def build_ground_truth(
    X_pca: np.ndarray,
    out_dir: str | Path,
    k_values: tuple[int, ...] = (10, 15, 30),
    max_cells: int = 4000,
    seed: int = 42,
) -> dict:
    """Build and save the kNN graphs and bulk all-pairs Dijkstra ground truth."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_pca = np.asarray(X_pca)
    n = X_pca.shape[0]

    graphs = knn_graphs(X_pca, k_values)
    connectivity: dict[str, int] = {}
    for k in k_values:
        adj = graphs[k]
        sparse.save_npz(out_dir / f"knn_k{k}.npz", adj)
        connectivity[str(k)] = n_connected_components(adj)

    idx = subsample_indices(n, max_cells, seed)
    sub = X_pca[idx]
    sub_graphs = graphs if len(idx) == n else knn_graphs(sub, k_values)
    dijkstra: dict[str, dict] = {}
    for k in k_values:
        d = shortest_paths(sub_graphs[k])
        tri = _upper_triangle(d)
        np.savez(
            out_dir / f"dijkstra_subset_{max_cells}_k{k}.npz",
            dist=tri,
            indices=idx.astype(np.int64),
        )
        dijkstra[str(k)] = {
            "n_subset": len(idx),
            "fraction_infinite": float(np.isinf(d).mean()),
        }
        del d

    return {
        "n_cells": n,
        "n_subset": len(idx),
        "connectivity": connectivity,
        "dijkstra": dijkstra,
    }


def load_geometry(scenario: str, data_root: str | Path) -> dict:
    """Load the Section 1 artifacts under ``{data_root}/{scenario}/geometry``."""
    g = Path(data_root) / scenario / "geometry"
    if not g.is_dir():
        raise FileNotFoundError(
            f"geometry artifacts not found: {g} (run prepare_geometry.py first)"
        )

    X_pca = np.load(g / "X_pca.npy")
    df = pd.read_csv(g / "labels_aligned.csv")
    cell_index = df["cell_index"].values.astype(np.int64)
    labels = df.drop(columns=["cell_index"])

    fit = np.load(g / "pca_fit.npz")
    pca = PCA(n_components=int(fit["n_pcs"]))
    pca.n_components_ = int(fit["n_pcs"])
    pca.n_features_in_ = fit["mean"].shape[0]
    pca.components_ = fit["components"]
    pca.mean_ = fit["mean"]
    pca.explained_variance_ratio_ = fit["explained_variance_ratio_"]
    pca.explained_variance_ = fit["explained_variance_"]

    split = np.load(g / "split.npz")
    variants: dict[str, dict[str, np.ndarray]] = {}
    v = np.load(g / "variants.npz")
    for key in v.files:
        name, part = key.split("/", 1)
        variants.setdefault(name, {})[part] = v[key]

    graphs: dict[int, sparse.csr_matrix] = {}
    for f in sorted(g.glob("knn_k*.npz")):
        graphs[int(f.stem.split("_k")[1])] = sparse.load_npz(f)

    dijkstra: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for f in sorted(g.glob("dijkstra_subset_*_k*.npz")):
        rest = f.stem.removeprefix("dijkstra_subset_")
        max_cells, k = (int(x) for x in rest.split("_k"))
        d = np.load(f)
        dijkstra.setdefault(max_cells, {})[k] = {
            "dist": d["dist"],
            "indices": d["indices"],
        }

    return {
        "X_pca": X_pca,
        "cell_index": cell_index,
        "labels": labels,
        "pca": pca,
        "train_idx": split["train_idx"],
        "val_idx": split["val_idx"],
        "bin_id": np.load(g / "bins.npy"),
        "root_idx": np.load(g / "roots.npy"),
        "variants": variants,
        "graphs": graphs,
        "dijkstra": dijkstra,
    }


@torch.no_grad()
def project_cloud(
    projector: torch.nn.Module,
    x: np.ndarray,
    device: torch.device | None = None,
    chunk: int = 500,
) -> np.ndarray:
    """Project a point cloud through a frozen projector in batches."""
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), chunk):
        batch = torch.from_numpy(x[start : start + chunk]).to(
            device=device, dtype=torch.float32
        )
        out[start : start + chunk] = projector(batch).detach().cpu().numpy()
    return out


def _upper_triangle(d: np.ndarray) -> np.ndarray:
    """Flattened strictly-upper-triangle entries as float32 (row by row)."""
    n = d.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    pos = 0
    for i in range(n):
        row = d[i, i + 1 :]
        out[pos : pos + len(row)] = row.astype(np.float32, copy=False)
        pos += len(row)
    return out
