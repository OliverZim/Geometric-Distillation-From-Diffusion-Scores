#!/usr/bin/env python
"""Calibrate noise-scale hyperparameters for the synthetic scRNA-seq pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from scipy.spatial import cKDTree

from experiments.synthetic_data.common.graph import subsample_indices
from experiments.synthetic_data.common.space import prepare_working_space
from geometry_distill.config import ExperimentConfig
from geometry_distill.seeding import set_seed

SCENARIOS = ("linear", "bifurcation", "tree")
SPACING_SUBSET = 3000
NOISE_GROUP_MIN_CELLS = 20
STEP_EARLY = 3
STEP_LATE = 97


class CalibrationConfig(ExperimentConfig):
    """Configuration for the scale calibration script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["all", "linear", "bifurcation", "tree"] = Field(
        default="all",
        description="Which scenario(s) to calibrate.",
    )
    n_pcs: int = Field(
        default=50,
        ge=1,
        description="Number of principal components.",
    )
    seed: int = Field(
        default=42,
        description="Random seed.",
    )
    data_root: Path = Field(
        default=Path("experiments/synthetic_data/synthetic/data"),
        description="Directory with the simulation outputs and geometry artifacts.",
    )

    @property
    def scenarios(self) -> tuple[str, ...]:
        return SCENARIOS if self.scenario == "all" else (self.scenario,)


def measure_spacing(X: np.ndarray, seed: int) -> float:
    """Median 5th-NN distance on a seeded 3000-cell subset."""
    idx = subsample_indices(X.shape[0], SPACING_SUBSET, seed)
    dists, _ = cKDTree(X).query(X[idx], k=6)
    return float(np.median(dists[:, 5]))


def measure_noise_floor(X: np.ndarray, labels: pd.DataFrame) -> float:
    """Median per-(Path, Step)-group spread around the group centroid."""
    values: list[float] = []
    for _, group in labels.groupby(["Path", "Step"]):
        if len(group) < NOISE_GROUP_MIN_CELLS:
            continue
        pts = X[group.index.values]
        centroid = pts.mean(axis=0)
        values.append(float(np.linalg.norm(pts - centroid, axis=1).mean()))
    if not values:
        return float("nan")
    return float(np.median(values))


def measure_data_extent(
    X: np.ndarray, labels: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    """Median over paths of the early/late endpoint distance; per-path values."""
    step_max = int(np.asarray(labels["Step"]).max())
    step_early = max(1, int(round(0.03 * step_max)))
    step_late = max(step_early + 1, int(round(0.97 * step_max)))
    per_path: dict[str, float] = {}
    for path, group in labels.groupby("Path"):
        early = X[group.index.values[group["Step"] <= step_early]]
        late = X[group.index.values[group["Step"] >= step_late]]
        if len(early) == 0 or len(late) == 0:
            continue
        dists, _ = cKDTree(late).query(early, k=len(late))
        per_path[str(path)] = float(dists.mean())
    if not per_path:
        return float("nan"), per_path
    return float(np.median(list(per_path.values()))), per_path


def calibrate_scenario(config: CalibrationConfig, scenario: str) -> dict[str, Any]:
    """Return the raw measurements and derived defaults for one scenario."""
    X_pca, labels = _load_working_space(config, scenario)
    X = np.asarray(X_pca, dtype=np.float64)
    n_cells, n_pcs_ambient = int(X.shape[0]), int(X.shape[1])

    spacing = measure_spacing(X, config.seed)
    noise_floor = measure_noise_floor(X, labels)
    data_extent, per_path_lengths = measure_data_extent(X, labels)
    pc1_std = float(np.std(X_pca[:, 0]))

    sigma_min = round(max(0.1 * spacing, 0.0005), 4)
    extent_cap_bound = 5.0 * noise_floor > 0.5 * data_extent
    sigma_max = round(min(5.0 * noise_floor, 0.5 * data_extent), 3)
    pseudo_sigma = round(noise_floor, 4)
    # Flow levels must be strictly descending; the extent cap can leave
    # sigma_max below pseudo_sigma, so sort explicitly.
    if sigma_max > 2.0 * pseudo_sigma:
        flow_sigmas = (
            sigma_max,
            round(sigma_max / 2, 3),
            pseudo_sigma,
            round(pseudo_sigma / 2, 3),
        )
        flow_fallback = False
    elif sigma_max > pseudo_sigma:
        flow_sigmas = (sigma_max, pseudo_sigma)
        flow_fallback = True
    elif pseudo_sigma > sigma_max:
        flow_sigmas = (pseudo_sigma, sigma_max)
        flow_fallback = True
    else:
        flow_sigmas = (sigma_max, round(max(sigma_max / 2.0, sigma_min), 4))
        flow_fallback = True
    tau_range = (round(0.5 * noise_floor, 4), round(1.5 * noise_floor, 4))
    jitter = round(0.5 * noise_floor, 4)

    if extent_cap_bound:
        print(
            f"  warning: sigma_max capped by data_extent "
            f"(5*noise_floor={5 * noise_floor:.4f} > 0.5*data_extent={0.5 * data_extent:.4f})",
            flush=True,
        )
    if flow_fallback and np.isfinite(sigma_max) and np.isfinite(pseudo_sigma):
        print(
            f"  warning: sigma_max <= 2*pseudo_sigma ({sigma_max} <= "
            f"{2 * pseudo_sigma:.4f}) — flow sigmas fall back to "
            f"{flow_sigmas}",
            flush=True,
        )

    return {
        "scenario": scenario,
        "n_cells": int(n_cells),
        "n_pcs": n_pcs_ambient,
        "raw": {
            "spacing": spacing,
            "noise_floor": noise_floor,
            "data_extent": data_extent,
            "pc1_std": pc1_std,
            "per_path_lengths": per_path_lengths,
            "subspace": "ambient",
            "n_pcs": n_pcs_ambient,
        },
        "derived": {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "pseudo_sigma": pseudo_sigma,
            "flow_sigmas": list(flow_sigmas),
            "tau_range": list(tau_range),
            "jitter": jitter,
        },
        "warnings": {
            "extent_cap_bound": bool(extent_cap_bound),
            "flow_fallback": bool(flow_fallback),
        },
    }


def _load_working_space(
    config: CalibrationConfig, scenario: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load ``geometry/X_pca.npy`` and labels if present, else recompute them."""
    g = config.data_root / scenario / "geometry"
    if (g / "X_pca.npy").exists() and (g / "labels_aligned.csv").exists():
        return np.load(g / "X_pca.npy"), pd.read_csv(g / "labels_aligned.csv")
    art = prepare_working_space(scenario, config.data_root, config.n_pcs, config.seed)
    return art["X_pca"], art["labels"]


def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':<12} {'spacing':>8} {'noise':>8} {'extent':>8} {'pc1_std':>8} "
        f"{'sigma_min':>9} {'sigma_max':>9} {'pseudo':>8} {'flow_sigmas':<24} "
        f"{'tau_range':<16} {'jitter':>7}"
    )
    print("\n=== scale calibration ===")
    print(header)
    for r in results:
        raw, d = r["raw"], r["derived"]
        flow = "(" + ", ".join(f"{v:.4f}" for v in d["flow_sigmas"]) + ")"
        tau = f"[{d['tau_range'][0]:.4f}, {d['tau_range'][1]:.4f}]"
        print(
            f"{r['scenario']:<12} {raw['spacing']:>8.4f} {raw['noise_floor']:>8.4f} "
            f"{raw['data_extent']:>8.4f} {raw['pc1_std']:>8.4f} "
            f"{d['sigma_min']:>9.4f} {d['sigma_max']:>9.4f} {d['pseudo_sigma']:>8.4f} "
            f"{flow:<24} {tau:<16} {d['jitter']:>7.4f}"
        )
    print()


def _sanitize(obj: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    config = CalibrationConfig()
    set_seed(config.seed)

    results = [calibrate_scenario(config, s) for s in config.scenarios]
    _print_table(results)
    for result in results:
        out_path = (
            config.data_root / result["scenario"] / "geometry" / "calibration.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_sanitize(result), indent=2) + "\n")
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
