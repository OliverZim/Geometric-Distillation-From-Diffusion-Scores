"""Geometry artifacts and calibration for the real-data transport pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BIN_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_BINS = len(BIN_EDGES) - 1


def make_bins(labels: pd.DataFrame) -> np.ndarray:
    """Return the per-path pseudotime bin of every cell.

    Bin 0 = [0, 0.2), ..., bin 4 = [0.8, 1.0] (last bin inclusive).
    """
    n = len(labels)
    bin_id = np.zeros(n, dtype=np.int8)
    edges = np.asarray(BIN_EDGES)
    for path in pd.unique(labels["Path"]):
        mask = (labels["Path"] == path).values
        pt = labels.loc[mask, "Pseudotime"].values
        b = np.searchsorted(edges[1:], pt, side="right")
        bin_id[mask] = np.minimum(b, N_BINS - 1)
    return bin_id


def load_geometry(scenario: str, data_root: str | Path) -> dict:
    """Return the working space, labels, split, bins and variant masks."""
    g = Path(data_root) / scenario / "geometry"
    if not g.is_dir():
        raise FileNotFoundError(
            f"geometry artifacts not found: {g} "
            "(run experiments/real_data/scripts/prepare_eb_geometry.py first)"
        )

    X_pca = np.load(g / "X_pca.npy")
    labels = pd.read_csv(g / "labels_aligned.csv")

    split = np.load(g / "split.npz")
    variants: dict[str, dict[str, np.ndarray]] = {}
    v = np.load(g / "variants.npz")
    for key in v.files:
        name, part = key.split("/", 1)
        variants.setdefault(name, {})[part] = v[key]

    return {
        "X_pca": X_pca,
        "labels": labels,
        "train_idx": split["train_idx"],
        "val_idx": split["val_idx"],
        "bin_id": np.load(g / "bins.npy"),
        "variants": variants,
    }


def load_calibration(scenario: str, data_root: str | Path) -> dict:
    """Return the scenario's ``geometry/calibration.json`` (raw + derived)."""
    path = Path(data_root) / scenario / "geometry" / "calibration.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"calibration file not found: {path} — run "
            f"experiments/real_data/scripts/calibrate_scales.py first"
        )
    with open(path) as f:
        cal = json.load(f)
    if not isinstance(cal.get("derived"), dict):
        raise ValueError(
            f"calibration file {path} has no 'derived' section — run "
            f"experiments/real_data/scripts/calibrate_scales.py first"
        )
    return cal


def seeded_subsample(idx: np.ndarray, n: int) -> np.ndarray:
    """Return a seeded subsample of ``n`` entries from ``idx`` without replacement."""
    idx = np.asarray(idx, dtype=np.int64)
    n = min(int(n), len(idx))
    if n == len(idx):
        return idx
    return np.random.choice(idx, size=n, replace=False).astype(np.int64)


def build_train_val(
    geom: dict, variant: str, n_val: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, held_out)`` for one variant.

    Non-``full`` variants validate on a seeded subsample of their kept train
    cells; the returned arrays are asserted to keep held-out cells out of both.
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

    val_source = (
        np.asarray(geom["val_idx"], dtype=np.int64) if variant == "full" else train_idx
    )
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
