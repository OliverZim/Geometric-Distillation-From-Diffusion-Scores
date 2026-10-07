"""Load the real-data working spaces into the geometry-distill label contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EB_TIMEPOINTS: tuple[str, ...] = ("T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E")
EB_TIMEPOINT_PT: dict[str, float] = {
    tp: i / (len(EB_TIMEPOINTS) - 1) for i, tp in enumerate(EB_TIMEPOINTS)
}

DEFAULT_CSV = "experiments/real_data/eb/data/eb_after_pca_log1p.csv"


def _labels_frame(
    *,
    cells: np.ndarray,
    path: np.ndarray,
    timepoint: np.ndarray,
    pseudotime: np.ndarray,
) -> pd.DataFrame:
    """Build the shared label frame with the 20-step discretisation."""
    n_steps = 20
    step = np.clip(np.ceil(pseudotime * n_steps).astype(np.int64), 1, n_steps)
    step[pseudotime <= 0] = 1
    return pd.DataFrame(
        {
            "Cell": np.asarray(cells, dtype=object),
            "Path": np.asarray(path, dtype=object),
            "Timepoint": np.asarray(timepoint, dtype=object),
            "Pseudotime": pseudotime,
            "PseudotimeGlobal": pseudotime.copy(),
            "Step": step,
        }
    )


def load_eb_pca_csv(csv_path: str | Path, *, n_pcs: int = 50) -> dict[str, Any]:
    """Load the EB PCA table and build labels for a single developmental path.

    Expects ``barcode``, ``timepoint`` and ``PC*`` columns.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"EB PCA CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"barcode", "timepoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    pc_cols = sorted(pc_cols, key=lambda c: int(c[2:]))
    if not pc_cols:
        raise ValueError(f"{path}: no PC* columns")
    n_keep = min(int(n_pcs), len(pc_cols))
    X_pca = np.asarray(df[pc_cols[:n_keep]].to_numpy(), dtype=np.float32)
    if not np.isfinite(X_pca).all():
        raise ValueError(f"{path}: non-finite PCA entries")

    timepoint = df["timepoint"].astype(str).values
    unknown = sorted(set(timepoint) - set(EB_TIMEPOINTS))
    if unknown:
        raise ValueError(f"{path}: unexpected timepoints {unknown}")

    pt = np.asarray([EB_TIMEPOINT_PT[t] for t in timepoint], dtype=np.float64)
    n = X_pca.shape[0]
    labels = _labels_frame(
        cells=df["barcode"].astype(str).values,
        path=np.full(n, "EB"),
        timepoint=timepoint,
        pseudotime=pt,
    )
    return {
        "X_pca": X_pca,
        "labels": labels,
        "cell_index": np.arange(n, dtype=np.int64),
        "source_path": path,
        "attrs": {
            "n_cells": int(n),
            "n_pcs": int(n_keep),
            "n_pcs_available": int(len(pc_cols)),
            "timepoints": {tp: int((timepoint == tp).sum()) for tp in EB_TIMEPOINTS},
            "duplicate_barcodes": int(df["barcode"].duplicated().sum()),
        },
    }


def load_day_npz(npz_path: str | Path, *, path_label: str) -> dict[str, Any]:
    """Load a ``{data, day, cell_id}`` npz into the label contract.

    ``Timepoint`` is ``day<d>`` and ``Pseudotime`` is the day's rank in [0, 1].
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"npz not found: {path}")
    z = np.load(path, allow_pickle=True)
    X_pca = np.asarray(z["data"], dtype=np.float32)
    if not np.isfinite(X_pca).all():
        raise ValueError(f"{path}: non-finite entries")
    day = np.asarray(z["day"]).astype(int)
    days = sorted(set(day.tolist()))
    pt_of = {d: i / (len(days) - 1) for i, d in enumerate(days)}
    pt = np.asarray([pt_of[d] for d in day], dtype=np.float64)
    n = X_pca.shape[0]
    cell = (
        np.asarray(z["cell_id"]).astype(str)
        if "cell_id" in z.files
        else np.array([f"c{i}" for i in range(n)])
    )
    labels = _labels_frame(
        cells=cell,
        path=np.full(n, path_label),
        timepoint=np.asarray([f"day{d}" for d in day]),
        pseudotime=pt,
    )
    return {
        "X_pca": X_pca,
        "labels": labels,
        "cell_index": np.arange(n, dtype=np.int64),
        "source_path": path,
        "attrs": {
            "n_cells": int(n),
            "n_pcs": int(X_pca.shape[1]),
            "n_pcs_available": int(X_pca.shape[1]),
            "timepoints": {f"day{d}": int((day == d).sum()) for d in days},
            "duplicate_barcodes": int(pd.Series(cell).duplicated().sum()),
        },
    }
