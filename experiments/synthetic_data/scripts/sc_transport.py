#!/usr/bin/env python
"""Pooled population transport on the synthetic scRNA-seq manifold."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Self

import numpy as np
import ot
import torch
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from experiments.synthetic_data.common.clean import (
    CleanTree,
    distance_to_tree,
    load_clean_tree,
    tree_segments,
)
from experiments.synthetic_data.common.eval_metrics import (
    off_manifold_residual,
    wasserstein_1,
)
from experiments.synthetic_data.common.space import (
    HOLDOUT_MID_BINS,
    load_geometry,
    project_cloud,
)
from geometry_distill.config import ExperimentConfig
from geometry_distill.device import resolve_device
from geometry_distill.geodesics import (
    PathRefinementConfig,
    WarpedPullbackMetricConfig,
    arc_length_midpoints,
    chord_paths,
    refine_paths,
)
from geometry_distill.projection import projector_from_checkpoint
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)

PROTOCOL = "pooled"
TARGET = "strict"
WINDOW_MODE = "bin"
OT_COUPLING_SPACE = "raw"
GEODESIC_COUPLING_SPACE = "projected"

METHODS = (
    "bin_prev",
    "bin_next",
    "ot_midpoint",
    "geodesic_midpoint",
    "geodesic_midpoint_projected",
)


class TransportExperimentConfig(ExperimentConfig):
    """Configuration for the synthetic scRNA-seq population-transport script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["linear", "bifurcation", "tree"] = Field(
        default="linear",
        description="Synthetic scenario to transport.",
    )
    variant: Literal["holdout_mid"] = Field(
        default="holdout_mid",
        description="Geometry variant to transport.",
    )
    data_root: Path = Field(
        default=Path("experiments/synthetic_data/synthetic/data"),
        description="Directory with the simulation outputs and geometry artifacts.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (coupling, subsampling).",
    )
    projector_checkpoint: Path = Field(
        description="Path to the projector checkpoint (stage 2 output).",
    )

    held_out_bin: int = Field(
        default=HOLDOUT_MID_BINS[0],
        description="Per-path pseudotime bin predicted by transport.",
    )
    n_pairs: int = Field(
        default=0,
        ge=0,
        description="Cap on coupled pairs (0 = all source cells).",
    )
    refine: PathRefinementConfig = Field(
        default_factory=lambda: PathRefinementConfig(
            metric=WarpedPullbackMetricConfig(space="ambient"),
            steps=1000,
            lr=1e-4,
        ),
        description="Chord refinement under the warped metric.",
    )
    n_waypoints: int = Field(
        default=31,
        ge=2,
        description="Waypoints of the chord polyline.",
    )
    pair_chunk: int = Field(
        default=500,
        ge=1,
        description="Batch size for projector forward passes.",
    )
    n_eval: int = Field(
        default=2000,
        ge=0,
        description="Per-cloud cap for the exact W1 (0 = all cells).",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @model_validator(mode="after")
    def _check_held_out_bin(self) -> Self:
        if self.held_out_bin not in HOLDOUT_MID_BINS:
            raise ValueError(
                f"variant 'holdout_mid' holds out bins {HOLDOUT_MID_BINS}; "
                f"held_out_bin {self.held_out_bin} is not blind"
            )
        if not 0 < self.held_out_bin < 4:
            raise ValueError(
                f"held_out_bin must be 1..3 (needs both neighbours), "
                f"got {self.held_out_bin}"
            )
        return self


def build_windows(
    geom: dict, config: TransportExperimentConfig
) -> dict[str, np.ndarray]:
    """Return the source, target and withheld rows of the pooled transport."""
    bin_id = geom["bin_id"]
    variant = geom["variants"][config.variant]
    train = np.asarray(variant["train"], dtype=np.int64)
    held = np.asarray(variant["held_out"], dtype=np.int64)
    if held.size == 0:
        raise ValueError(
            f"variant {config.variant!r} has no held_out cells; transport "
            "needs a blind holdout variant"
        )

    source = np.sort(train[bin_id[train] == config.held_out_bin - 1])
    target = np.sort(train[bin_id[train] == config.held_out_bin + 1])
    withheld = np.sort(held[bin_id[held] == config.held_out_bin])
    if source.size == 0 or target.size == 0 or withheld.size == 0:
        raise ValueError(
            f"empty transport window: source {source.size}, target "
            f"{target.size}, withheld {withheld.size}"
        )
    for a, b in ((source, target), (source, withheld), (target, withheld)):
        if np.intersect1d(a, b).size:
            raise ValueError("transport windows overlap")
    return {"source": source, "target": target, "withheld": withheld}


def _cap_indices(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded cap on a window; ``cap <= 0`` or ``cap >= len`` keeps all rows."""
    if cap <= 0 or cap >= len(idx):
        return idx
    return np.sort(idx[rng.choice(len(idx), size=cap, replace=False)])


def plan_targets(plan: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one target index per source row from a transport plan."""
    row_sums = plan.sum(axis=1, keepdims=True)
    rows = np.divide(
        plan, row_sums, out=np.zeros_like(plan, dtype=np.float64), where=row_sums > 0
    )
    u = rng.random((len(rows), 1))
    return (u >= rows.cumsum(axis=1)).sum(axis=1).clip(max=plan.shape[1] - 1)


def coupling_features(
    X: np.ndarray,
    idx: np.ndarray,
    projector: torch.nn.Module,
    space: str,
    device: torch.device,
) -> np.ndarray:
    """Return the features in which the exact OT is solved."""
    if space == "raw":
        return np.asarray(X[idx], dtype=np.float64)
    if space == "projected":
        return project_cloud(projector, X[idx], device).astype(np.float64)
    raise ValueError(f"unknown coupling space {space!r}")


def couple(
    X: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    projector: torch.nn.Module,
    space: str,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one exact-OT partner per source row in ``space``."""
    src = coupling_features(X, source_idx, projector, space, device)
    tgt = coupling_features(X, target_idx, projector, space, device)
    cost = ot.dist(src, tgt, metric="sqeuclidean")
    plan = ot.emd(
        np.ones(len(src)) / len(src),
        np.ones(len(tgt)) / len(tgt),
        cost,
    )
    return target_idx[plan_targets(plan, rng)]


def geodesic_midpoints(
    projector: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    config: TransportExperimentConfig,
    device: torch.device,
) -> dict:
    """Refine each chord and return its arc-length midpoint.

    Returns ``{"midpoint_raw", "midpoint_projected", "paths", "energy_init",
    "energy_best"}``; chunks of ``config.pair_chunk`` pairs are refined
    independently, so the chunk size does not change the result.
    """
    n, d = x0.shape
    m = config.n_waypoints
    midpoint_raw = np.empty((n, d), dtype=np.float32)
    paths_out = np.empty((n, m, d), dtype=np.float32)
    energy_init = np.empty(n, dtype=np.float64)
    energy_best = np.empty(n, dtype=np.float64)

    for start in range(0, n, config.pair_chunk):
        stop = min(start + config.pair_chunk, n)
        chords = chord_paths(x0[start:stop], x1[start:stop], m, device)
        refined, info = refine_paths(chords, projector, config.refine)
        midpoint_raw[start:stop] = (
            arc_length_midpoints(refined, fraction=0.5).detach().cpu().numpy()
        )
        paths_out[start:stop] = refined.detach().cpu().numpy().astype(np.float32)
        energy_init[start:stop] = info["energy_init"]
        energy_best[start:stop] = info["energy_best"]
        logger.info("  [geodesic] %s/%s pairs refined", stop, n)

    return {
        "midpoint_raw": midpoint_raw,
        "midpoint_projected": project_cloud(projector, midpoint_raw, device),
        "paths": paths_out,
        "energy_init": energy_init,
        "energy_best": energy_best,
    }


def _subsample(cloud: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded subsample of ``cloud``; ``cap <= 0`` or ``cap >= len`` keeps all."""
    if cap <= 0 or cap >= len(cloud):
        return cloud
    return cloud[np.sort(rng.choice(len(cloud), size=cap, replace=False))]


def evaluate(
    *,
    config: TransportExperimentConfig,
    predictions: dict[str, np.ndarray],
    ref: np.ndarray,
    projector: torch.nn.Module,
    tree: CleanTree,
) -> dict:
    """Return W1, tree-distance and projector-residual metrics per method."""
    seg_a, seg_b = tree_segments(tree)
    ref_eval = _subsample(ref, config.n_eval, np.random.default_rng(config.seed + 1))
    metrics: dict[str, dict[str, float]] = {}
    for name, pred in predictions.items():
        pred_eval = _subsample(pred, config.n_eval, np.random.default_rng(config.seed))
        gt_dist = distance_to_tree(pred, seg_a, seg_b)
        metrics[name] = {
            "W1": wasserstein_1(pred_eval, ref_eval),
            "gt_tree_dist_mean": float(gt_dist.mean()),
            "offmanifold_resid_mean": float(
                off_manifold_residual(pred_eval, projector).mean()
            ),
        }
    gt_ref = distance_to_tree(ref, seg_a, seg_b)
    diagnostics = {
        "gt_tree_dist_reference_clean": {"mean": float(gt_ref.mean())},
    }
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "n_ref_eval": int(len(ref_eval)),
    }


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-method scalars for ``metrics.jsonl``."""
    return {
        f"{name}/{key}": float(value)
        for name, row in metrics.items()
        for key, value in row.items()
    }


def _load_projector(
    config: TransportExperimentConfig, device: torch.device
) -> torch.nn.Module:
    """Load the Stage II checkpoint and check the blind-variant pairing."""
    ckpt_path = Path(config.projector_checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"projection checkpoint not found: {ckpt_path} (set --projector-checkpoint)"
        )
    config_path = ckpt_path.parent / "config.json"
    proj_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if proj_config and proj_config.get("variant") != config.variant:
        raise ValueError(
            f"projection run at {ckpt_path.parent} has variant "
            f"{proj_config.get('variant')!r}, expected {config.variant!r}; the "
            "transport protocol requires a window-blind projector"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    projector = projector_from_checkpoint(ckpt, device=device)
    projector.requires_grad_(False)
    return projector


def _main_impl(config: TransportExperimentConfig, run: RunWriter) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)
    logger.info(
        "Protocol: %s  target: %s  window: %s  coupling: OT=%s, geodesic=%s",
        PROTOCOL,
        TARGET,
        WINDOW_MODE,
        OT_COUPLING_SPACE,
        GEODESIC_COUPLING_SPACE,
    )

    geom = load_geometry(config.scenario, config.data_root)
    X = geom["X_pca"].astype(np.float32)
    geometry_dir = config.data_root / config.scenario / "geometry"
    X_cell_means = np.load(geometry_dir / "X_cell_means_pca.npy")
    if X_cell_means.shape != X.shape:
        raise ValueError(
            "clean target embedding is not row-aligned with X_pca: "
            f"{X_cell_means.shape} vs {X.shape}"
        )
    paths = [str(p) for p in geom["labels"]["Path"].unique()]
    tree = load_clean_tree(geometry_dir / "clean_tree.npz")

    projector = _load_projector(config, device)

    rng = np.random.default_rng(config.seed)
    windows = build_windows(geom, config)
    source = _cap_indices(windows["source"], config.n_pairs, rng)
    x0 = X[source]
    partner_raw = couple(
        X, source, windows["target"], projector, OT_COUPLING_SPACE, device, rng
    )
    x1 = X[partner_raw]
    preds = {
        "bin_prev": x0,
        "bin_next": x1,
        "ot_midpoint": (x0 + x1) / 2.0,
    }

    partner_projected: np.ndarray | None = None
    coherent_paths: list[np.ndarray] = []
    geo_energy_init: list[np.ndarray] = []
    geo_energy_best: list[np.ndarray] = []
    if config.refine.steps > 0:
        partner_projected = couple(
            X,
            source,
            windows["target"],
            projector,
            GEODESIC_COUPLING_SPACE,
            device,
            rng,
        )
        geo = geodesic_midpoints(projector, x0, X[partner_projected], config, device)
        preds["geodesic_midpoint"] = geo["midpoint_raw"]
        preds["geodesic_midpoint_projected"] = geo["midpoint_projected"]
        coherent_paths.append(geo["paths"])
        geo_energy_init.append(geo["energy_init"])
        geo_energy_best.append(geo["energy_best"])
    logger.info(
        "Pooled group: %s source, %s target, %s withheld, %s coupled pairs",
        len(source),
        len(windows["target"]),
        len(windows["withheld"]),
        len(partner_raw),
    )

    withheld_idx = windows["withheld"]
    ref = X_cell_means[withheld_idx].astype(np.float32)

    result = evaluate(
        config=config, predictions=preds, ref=ref, projector=projector, tree=tree
    )

    if config.refine.steps > 0:
        energy_init = np.concatenate(geo_energy_init)
        energy_best = np.concatenate(geo_energy_best)
        geodesic_info: dict = {
            "enabled": True,
            "space": config.refine.metric.space,
            "beta": config.refine.metric.beta,
            "n_waypoints": config.n_waypoints,
            "refine_steps": config.refine.steps,
            "refine_lr": config.refine.lr,
            "energy_init_mean": float(energy_init.mean()),
            "energy_best_mean": float(energy_best.mean()),
            "energy_drop_frac": float(1.0 - energy_best.mean() / energy_init.mean()),
        }
    else:
        geodesic_info = {"enabled": False}

    evaluation = {
        "scenario": config.scenario,
        "variant": config.variant,
        "blind": True,
        "protocol": PROTOCOL,
        "target": TARGET,
        "window_mode": WINDOW_MODE,
        "coupling": {
            "ot_midpoint": OT_COUPLING_SPACE,
            "geodesic_midpoint": GEODESIC_COUPLING_SPACE,
        },
        "projector_checkpoint": str(config.projector_checkpoint),
        "t_star": 0.5,
        "held_out_bin": config.held_out_bin,
        "source_bin": config.held_out_bin - 1,
        "target_bin": config.held_out_bin + 1,
        "paths": paths,
        "n_pairs_cap": config.n_pairs,
        "n_source": int(len(source)),
        "n_target": int(len(windows["target"])),
        "n_withheld": int(len(withheld_idx)),
        "n_pairs": int(len(source)),
        "n_ref": int(result["n_ref_eval"]),
        "geodesic": geodesic_info,
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
    }
    run.save_json("evaluation.json", evaluation)
    run.save_checkpoint(
        "paths.pt",
        {
            "predictions": {
                name: torch.from_numpy(pred.astype(np.float32))
                for name, pred in preds.items()
            },
            "source_idx": source,
            "target_idx_raw": partner_raw,
            "target_idx_projected": partner_projected,
            "withheld_idx": withheld_idx,
            "geodesic_paths": (
                torch.from_numpy(np.concatenate(coherent_paths))
                if coherent_paths
                else None
            ),
        },
    )
    run.log_metrics(0, _flatten_metrics(result["metrics"]))

    for name, row in result["metrics"].items():
        logger.info(
            "%-28s W1 %.5f  GT-tree %.5f  off-manifold %.5f",
            name,
            row["W1"],
            row["gt_tree_dist_mean"],
            row["offmanifold_resid_mean"],
        )
    ref_gt = result["diagnostics"]["gt_tree_dist_reference_clean"]
    logger.info("Withheld clean population GT distance: mean %.5f", ref_gt["mean"])
    logger.info("Evaluation saved to: %s", run.out_dir / "evaluation.json")
    logger.info("Done.")


def main() -> None:
    config = TransportExperimentConfig()

    set_seed(config.seed)

    try:
        run = RunWriter(config.run)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run)


if __name__ == "__main__":
    main()
