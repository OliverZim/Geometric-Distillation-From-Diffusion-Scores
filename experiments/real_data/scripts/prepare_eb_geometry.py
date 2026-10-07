#!/usr/bin/env python
"""Build geometry artifacts from the real-data working space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from sklearn.model_selection import train_test_split

from experiments.real_data.common.eb import DEFAULT_CSV, load_day_npz, load_eb_pca_csv
from experiments.real_data.common.space import make_bins
from experiments.real_data.scripts.calibrate_scales import (
    CalibrationConfig,
    Dataset,
    calibrate_scenario,
)
from geometry_distill.config import ExperimentConfig
from geometry_distill.seeding import set_seed

REPO_ROOT = Path(__file__).resolve().parents[3]


class GeometryConfig(ExperimentConfig):
    """Configuration for the real-data geometry preparation script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Dataset = Field(
        default="eb",
        description="Geometry folder under data_root (eb, cite50, ...).",
    )
    data_root: Path = Field(
        default=Path("experiments/real_data/eb/data"),
        description="Directory to write <scenario>/geometry/ into.",
    )
    csv: Path = Field(
        default=Path(DEFAULT_CSV),
        description="Path to the EB PCA CSV (ignored when npz is set).",
    )
    npz: Path | None = Field(
        default=None,
        description="Day-labelled {data, day, cell_id} npz instead of the CSV.",
    )
    n_pcs: int = Field(
        default=50,
        ge=1,
        description="Number of principal components to keep.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for the splits and calibration.",
    )


def _sanitize(obj: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def prepare_geometry(config: GeometryConfig) -> Path:
    """Write X_pca, labels, splits, variants and calibration under data_root."""
    set_seed(config.seed)
    if config.npz is not None:
        print(f"\n=== {config.scenario}: load day-labelled npz ===", flush=True)
        art = load_day_npz(_resolve(config.npz), path_label=config.scenario.upper())
    else:
        print(
            f"\n=== {config.scenario}: load PCA CSV (n_pcs={config.n_pcs}) ===",
            flush=True,
        )
        art = load_eb_pca_csv(_resolve(config.csv), n_pcs=config.n_pcs)
    X_pca = art["X_pca"]
    labels = art["labels"]
    print(
        f"  {X_pca.shape[0]} cells x {X_pca.shape[1]} PCs  "
        f"timepoints={art['attrs']['timepoints']}",
        flush=True,
    )

    out_dir = _resolve(config.data_root) / config.scenario / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_pca.npy", X_pca)
    labels.to_csv(out_dir / "labels_aligned.csv", index=False)

    print(f"=== {config.scenario}: splits / bins / variants ===", flush=True)
    n = len(labels)
    train_idx, val_idx = train_test_split(
        np.arange(n),
        test_size=0.1,
        stratify=labels["Timepoint"].values,
        random_state=config.seed,
    )
    train_idx = np.sort(train_idx.astype(np.int64))
    val_idx = np.sort(val_idx.astype(np.int64))
    bin_id = make_bins(labels)

    empty = np.empty(0, dtype=np.int64)
    variants = {
        "full": {"train": train_idx, "val": val_idx, "held_out": empty},
    }
    tp = labels["Timepoint"].values
    for held in sorted(pd.unique(tp)):
        keep = np.flatnonzero(tp != held)
        tr, va = train_test_split(
            keep, test_size=0.1, stratify=tp[keep], random_state=config.seed
        )
        variants[f"holdout_{held}"] = {
            "train": np.sort(tr.astype(np.int64)),
            "val": np.sort(va.astype(np.int64)),
            "held_out": np.sort(np.flatnonzero(tp == held).astype(np.int64)),
        }
    print(
        "  variants: "
        + ", ".join(
            f"{k} (train {len(v['train'])}, held_out {len(v['held_out'])})"
            for k, v in variants.items()
        )
    )
    np.savez(out_dir / "split.npz", train_idx=train_idx, val_idx=val_idx)
    np.save(out_dir / "bins.npy", bin_id)
    np.savez(
        out_dir / "variants.npz",
        **{
            f"{name}/{part}": arr
            for name, parts in variants.items()
            for part, arr in parts.items()
        },
    )

    geometry_config = {
        "source": "day_npz" if config.npz is not None else "eb_pca_csv",
        "scenario": config.scenario,
        "source_path": str(art["source_path"]),
        "n_pcs": int(X_pca.shape[1]),
        "seed": int(config.seed),
        "n_cells": int(X_pca.shape[0]),
        "attrs": art["attrs"],
    }
    (out_dir / "config.json").write_text(
        json.dumps(_sanitize(geometry_config), indent=2) + "\n"
    )

    print(f"=== {config.scenario}: calibrate scales ===", flush=True)
    calib_cfg = CalibrationConfig(
        scenario=config.scenario,
        seed=config.seed,
        data_root=config.data_root,
        _cli_parse_args=False,
    )
    calib = calibrate_scenario(calib_cfg, config.scenario)
    (out_dir / "calibration.json").write_text(
        json.dumps(_sanitize(calib), indent=2) + "\n"
    )
    print(f"  wrote {out_dir}", flush=True)
    return out_dir


def main() -> None:
    config = GeometryConfig()
    prepare_geometry(config)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
