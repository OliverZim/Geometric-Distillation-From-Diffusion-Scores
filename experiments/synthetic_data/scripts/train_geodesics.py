#!/usr/bin/env python
"""Evaluate chord baselines and non-amortised refinement on the scRNA-seq manifold."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

from experiments.synthetic_data.common.clean import (
    distance_to_tree,
    load_clean_tree,
    tree_distances,
    tree_segments,
)
from experiments.synthetic_data.common.eval_metrics import (
    check_lengths,
    summarize_lengths,
)
from experiments.synthetic_data.common.graph import knn_graphs
from experiments.synthetic_data.common.space import load_geometry, project_cloud
from geometry_distill.config import ExperimentConfig
from geometry_distill.device import resolve_device
from geometry_distill.geodesics import (
    PathRefinementConfig,
    WarpedPullbackMetricConfig,
    chord_paths,
    refine_paths,
)
from geometry_distill.projection import projector_from_checkpoint
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)

GRAPH_K = 15
GRAPH_METHOD = f"graph_noisy_k{GRAPH_K}"
GRAPH_WAYPOINTS = 31
LENGTH_KEYS = (
    "chord_raw",
    "chord_proj",
    "refine_raw",
    "refine_proj",
    GRAPH_METHOD,
)


class GeodesicsExperimentConfig(ExperimentConfig):
    """Configuration for the synthetic scRNA-seq geodesics script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["linear", "bifurcation", "tree"] = Field(
        default="linear",
        description="Synthetic scenario to evaluate.",
    )
    variant: Literal["full", "holdout_mid"] | None = Field(
        default=None,
        description=(
            "Stage II variant; default: auto-detect from the projector run's "
            "config.json."
        ),
    )
    data_root: Path = Field(
        default=Path("experiments/synthetic_data/synthetic/data"),
        description="Directory with the simulation outputs and geometry artifacts.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (endpoint pairs, refinement).",
    )
    projector_checkpoint: Path = Field(
        description="Path to the projector checkpoint (stage 2 output).",
    )
    pairs_path: Path | None = Field(
        default=None,
        description=(
            "Path to the Stage II pairs.npz evaluation pool; default: the "
            "checkpoint's run directory."
        ),
    )

    refine: PathRefinementConfig = Field(
        default_factory=lambda: PathRefinementConfig(
            metric=WarpedPullbackMetricConfig(space="ambient"),
            steps=0,
            lr=1e-4,
        ),
        description="Non-amortised chord refinement under the warped metric.",
    )
    n_waypoints: int = Field(
        default=31,
        ge=3,
        description="Waypoints of the chord polyline.",
    )
    n_eval_pairs: int = Field(
        default=0,
        ge=0,
        description="Number of pair-pool entries to evaluate (0 = all).",
    )
    pair_chunk: int = Field(
        default=500,
        ge=1,
        description="Batch size for projector forward passes.",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )


def _resolve_projector_checkpoint(config: GeodesicsExperimentConfig) -> Path:
    """Return the configured projector checkpoint path, failing if missing."""
    path = Path(config.projector_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"projector checkpoint not found: {path}\n"
            "  (set --projector-checkpoint or train one with "
            "experiments/synthetic_data/scripts/train_projection.py)"
        )
    return path


def _resolve_pairs_path(
    config: GeodesicsExperimentConfig, projector_path: Path
) -> Path:
    """Return the configured pair pool path, failing if it is missing."""
    path = (
        Path(config.pairs_path)
        if config.pairs_path
        else projector_path.parent / "pairs.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"pair pool not found: {path}\n"
            "  The Stage II pair draw is not reproducible outside its "
            "training run; point --pairs-path at an existing run."
        )
    return path


def _subset_indices(n_pairs: int, n_eval: int) -> np.ndarray:
    """Deterministic evenly spaced subset of ``range(n_pairs)``."""
    if n_eval <= 0 or n_eval >= n_pairs:
        return np.arange(n_pairs, dtype=np.int64)
    return np.unique(np.linspace(0, n_pairs - 1, n_eval, dtype=np.int64))


def _variant_from_projection_run(projector_path: Path) -> str | None:
    """Read the variant recorded next to a projector checkpoint, if any."""
    config_path = projector_path.parent / "config.json"
    if not config_path.is_file():
        return None
    try:
        variant = json.loads(config_path.read_text()).get("variant")
    except (OSError, json.JSONDecodeError):
        return None
    return str(variant) if variant else None


def _pair_pool_cells(geom: dict, variant: str) -> np.ndarray:
    """Return the cells the Stage II pair pool was drawn from for ``variant``."""
    if variant == "full":
        return np.asarray(geom["val_idx"], dtype=np.int64)
    variants = geom.get("variants", {})
    if variant not in variants:
        raise ValueError(
            f"variant {variant!r} not found in the geometry variants "
            f"(available: {sorted(variants)})"
        )
    return np.asarray(variants[variant]["train"], dtype=np.int64)


def _validate_categories(categories: np.ndarray, labels, cells: np.ndarray) -> None:
    """Cross-check ``same:``/``cross:`` categories against the cell labels."""
    path_values = labels["Path"].astype(str).to_numpy()
    for k, category in enumerate(categories):
        path_a = path_values[cells[k, 0]]
        path_b = path_values[cells[k, 1]]
        if category.startswith("same:"):
            expected = category.removeprefix("same:")
            if path_a != expected or path_b != expected:
                raise ValueError(
                    f"pair {k}: category {category!r} disagrees with the "
                    f"labels {path_a!r}/{path_b!r}"
                )
        elif category.startswith("cross:"):
            try:
                left, right = category.removeprefix("cross:").split("|")
                expected_a, expected_b = left.rsplit("-", 1)[0], right.rsplit("-", 1)[0]
            except ValueError as exc:
                raise ValueError(
                    f"pair {k}: malformed cross category {category!r}"
                ) from exc
            if path_a != expected_a or path_b != expected_b:
                raise ValueError(
                    f"pair {k}: category {category!r} disagrees with the "
                    f"labels {path_a!r}/{path_b!r}"
                )
        else:
            raise ValueError(f"pair {k}: unknown category prefix in {category!r}")


def _load_pairs(
    pairs_path: Path,
    labels,
    allowed_cells: np.ndarray,
    n_cells: int,
    n_eval_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load and validate the Stage II pair pool.

    Returns ``(cells, categories, pair_idx, subset, n_pool)`` for the
    selected evaluation subset.
    """
    with np.load(pairs_path) as pool:
        missing = [k for k in ("cells", "category", "pair_idx") if k not in pool.files]
        if missing:
            raise ValueError(
                f"pair pool {pairs_path} is missing entries {missing}; found "
                f"{sorted(pool.files)}"
            )
        cells_all = pool["cells"].astype(np.int64)
        categories_all = pool["category"].astype(str)
        pair_idx_all = pool["pair_idx"].astype(np.int64)

    if cells_all.ndim != 2 or cells_all.shape[1] != 2:
        raise ValueError(f"cells must have shape (n, 2), got {cells_all.shape}")
    if categories_all.shape != (len(cells_all),):
        raise ValueError(
            f"category has shape {categories_all.shape}, expected ({len(cells_all)},)"
        )
    if pair_idx_all.shape != cells_all.shape:
        raise ValueError(
            f"pair_idx has shape {pair_idx_all.shape}, expected {cells_all.shape}"
        )
    if cells_all.size and (cells_all.min() < 0 or cells_all.max() >= n_cells):
        raise ValueError(
            f"pair cells must be in [0, {n_cells}), got "
            f"[{cells_all.min()}, {cells_all.max()}]"
        )
    if not np.isin(cells_all, np.asarray(allowed_cells)).all():
        raise ValueError(
            "pair cells are not a subset of the cells the pool should have "
            "been drawn from for this variant; the pair pool and the current "
            "geometry artifacts disagree"
        )
    _validate_categories(categories_all, labels, cells_all)

    subset = _subset_indices(len(cells_all), n_eval_pairs)
    return (
        cells_all[subset],
        categories_all[subset],
        pair_idx_all[subset],
        subset,
        int(len(cells_all)),
    )


def _norms(x: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norms in float64."""
    return np.linalg.norm(x.astype(np.float64), axis=1)


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Straight interpolation between two clouds."""
    t = np.linspace(0.0, 1.0, n)
    return a[:, None, :] * (1 - t)[None, :, None] + b[:, None, :] * t[None, :, None]


def _interior_tree_mean(
    polyline: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> np.ndarray:
    """Per-pair mean interior waypoint distance to the clean tree."""
    m = polyline.shape[1]
    if m < 3:
        raise ValueError("need at least 3 waypoints for interior off-GT stats")
    dist = distance_to_tree(polyline, seg_a, seg_b).reshape(len(polyline), m)
    return dist[:, 1:-1].mean(axis=1)


def _resample_polyline(polyline: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` equal-arc-length points."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0.0:
        return np.repeat(polyline[:1], n, axis=0)
    query = np.linspace(0.0, arc[-1], n)
    return np.stack(
        [np.interp(query, arc, polyline[:, d]) for d in range(polyline.shape[1])],
        axis=1,
    )


def _graph_noisy_baseline(
    X_pca: np.ndarray,
    cells: np.ndarray,
    k: int = GRAPH_K,
    n_waypoints: int = GRAPH_WAYPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kNN-Dijkstra lengths and resampled shortest-path polylines."""
    cloud = X_pca.astype(np.float64, copy=False)
    adj = knn_graphs(cloud, (k,))[k]
    sources, targets = cells[:, 0], cells[:, 1]
    lengths = np.full(len(cells), np.nan, dtype=np.float64)
    polylines = np.zeros((len(cells), n_waypoints, cloud.shape[1]), dtype=np.float64)
    for source in np.unique(sources):
        rows = np.flatnonzero(sources == source)
        dist, pred = csgraph_dijkstra(
            adj,
            directed=False,
            indices=np.array([int(source)]),
            return_predecessors=True,
        )
        predecessor = pred[0]
        for row in rows:
            target = int(targets[row])
            lengths[row] = dist[0, target]
            path = [target]
            current = target
            while current != int(source):
                current = int(predecessor[current])
                if current < 0:
                    path = []
                    break
                path.append(current)
            if path:
                polylines[row] = _resample_polyline(
                    cloud[np.array(path[::-1])], n_waypoints
                )
    return lengths, polylines


@dataclass
class RefinementResult:
    """Per-pair outputs of the non-amortised chord refinement."""

    paths_raw: np.ndarray
    paths_proj: np.ndarray
    L_raw: np.ndarray
    L_proj: np.ndarray
    m_off_raw: np.ndarray
    m_off_proj: np.ndarray
    energy_init: np.ndarray
    energy_best: np.ndarray


def _path_lengths(paths: np.ndarray) -> np.ndarray:
    """Euclidean polyline length per path, ``(n, M, d)`` -> ``(n,)``."""
    steps = np.diff(paths.astype(np.float64), axis=1)
    return np.linalg.norm(steps, axis=-1).sum(axis=1)


def _interior_residual_sq(paths: np.ndarray, images: np.ndarray) -> np.ndarray:
    """Mean squared residual of the interior waypoints, shape ``(n,)``."""
    if paths.shape[1] <= 2:
        raise ValueError("need at least 3 waypoints for interior diagnostics")
    residual_sq = ((paths.astype(np.float64) - images.astype(np.float64)) ** 2).sum(-1)
    return residual_sq[:, 1:-1].mean(axis=1)


def _refine_all(
    projector: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    config: GeodesicsExperimentConfig,
    device: torch.device,
) -> RefinementResult:
    """Refine every pair's straight chord under the configured metric."""
    n, d = x0.shape
    m = config.n_waypoints
    paths_raw = np.empty((n, m, d), dtype=np.float32)
    paths_proj = np.empty((n, m, d), dtype=np.float32)
    m_off_raw = np.empty(n, dtype=np.float64)
    m_off_proj = np.empty(n, dtype=np.float64)
    energy_init = np.empty(n, dtype=np.float64)
    energy_best = np.empty(n, dtype=np.float64)

    for start in range(0, n, config.pair_chunk):
        stop = min(start + config.pair_chunk, n)
        chords = chord_paths(x0[start:stop], x1[start:stop], m, device)
        refined, info = refine_paths(chords, projector, config.refine)
        raw = refined.detach().cpu().numpy().astype(np.float32)
        images = project_cloud(
            projector, raw.reshape(-1, d), device, config.pair_chunk * m
        ).reshape(raw.shape)
        images_of_images = project_cloud(
            projector, images.reshape(-1, d), device, config.pair_chunk * m
        ).reshape(raw.shape)

        paths_raw[start:stop] = raw
        paths_proj[start:stop] = images
        m_off_raw[start:stop] = _interior_residual_sq(raw, images)
        m_off_proj[start:stop] = _interior_residual_sq(images, images_of_images)
        energy_init[start:stop] = info["energy_init"]
        energy_best[start:stop] = info["energy_best"]
        logger.info("  [refine] %s/%s pairs refined", stop, n)

    return RefinementResult(
        paths_raw=paths_raw,
        paths_proj=paths_proj,
        L_raw=_path_lengths(paths_raw),
        L_proj=_path_lengths(paths_proj),
        m_off_raw=m_off_raw,
        m_off_proj=m_off_proj,
        energy_init=energy_init,
        energy_best=energy_best,
    )


def _summarize_refinement(refinement: RefinementResult) -> dict:
    """Return the mean off-manifold residuals of a refinement run."""
    if (refinement.energy_best > refinement.energy_init + 1e-12).any():
        raise ValueError(
            "refinement returned a best energy above its initialisation; "
            "best-iterate tracking in refine_paths should never increase"
        )
    return {
        "enabled": True,
        "m_off_raw_mean": float(refinement.m_off_raw.mean()),
        "m_off_proj_mean": float(refinement.m_off_proj.mean()),
    }


def main() -> None:
    config = GeodesicsExperimentConfig()

    set_seed(config.seed)

    try:
        projector_path = _resolve_projector_checkpoint(config)
        pairs_path = _resolve_pairs_path(config, projector_path)
        if config.variant is None:
            detected = _variant_from_projection_run(projector_path)
            if detected is not None:
                config = config.model_copy(update={"variant": detected})
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    variant = config.variant or "full"
    if variant != config.variant:
        config = config.model_copy(update={"variant": variant})

    with run:
        run.write_config(config)
        _main_impl(config, run, projector_path, pairs_path)


def _main_impl(
    config: GeodesicsExperimentConfig,
    run: RunWriter,
    projector_path: Path,
    pairs_path: Path,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    geom = load_geometry(config.scenario, config.data_root)
    X_pca = geom["X_pca"]
    labels = geom["labels"]
    n_cells = X_pca.shape[0]

    geometry_dir = config.data_root / config.scenario / "geometry"
    clean_path = geometry_dir / "X_clean_pca.npy"
    tree_path = geometry_dir / "clean_tree.npz"
    if not clean_path.is_file() or not tree_path.is_file():
        raise FileNotFoundError(
            f"clean ground-truth artifacts not found under {geometry_dir} — "
            "run `python -m experiments.synthetic_data.scripts.prepare_geometry "
            f"--task clean --scenario {config.scenario}` first"
        )
    X_clean = np.load(clean_path)
    tree = load_clean_tree(tree_path)
    if X_clean.shape != X_pca.shape:
        raise ValueError(
            f"clean embedding shape {X_clean.shape} does not match X_pca "
            f"{X_pca.shape}; regenerate the clean artifacts"
        )

    ckpt = torch.load(projector_path, map_location="cpu", weights_only=False)
    projector = projector_from_checkpoint(ckpt, device=device)

    cells, categories, pair_idx, subset, n_pool = _load_pairs(
        pairs_path,
        labels,
        _pair_pool_cells(geom, config.variant),
        n_cells,
        config.n_eval_pairs,
    )
    logger.info("Scenario: %s  variant: %s", config.scenario, config.variant)
    logger.info("Projector:  %s", projector_path)
    logger.info(
        "Pair pool:  %s (%s pairs, evaluating %s)", pairs_path, n_pool, len(cells)
    )
    logger.info(
        "Clean tree: %s paths, junction max dist %.2e, step spread max %.2e",
        len(tree.path_names),
        tree.junction_max_dist,
        tree.step_spread_max,
    )
    n_same = int(np.char.startswith(categories, "same").sum())
    logger.info(
        "Evaluating %s same-branch + %s cross-branch", n_same, len(cells) - n_same
    )

    x0, x1 = X_pca[cells[:, 0]], X_pca[cells[:, 1]]
    x0_clean, x1_clean = X_clean[cells[:, 0]], X_clean[cells[:, 1]]

    p0 = project_cloud(projector, x0, device, config.pair_chunk)
    p1 = project_cloud(projector, x1, device, config.pair_chunk)

    lengths = {
        "chord_raw": _norms(x1 - x0),
        "chord_proj": _norms(p1 - p0),
    }
    tree_gt = tree_distances(tree, cells).astype(np.float64)
    check_lengths(tree_gt, lengths)

    seg_a, seg_b = tree_segments(tree)
    off_gt_mean: dict[str, np.ndarray] = {}
    for name, polyline in {
        "chord_raw": chord_paths(x0, x1, config.n_waypoints, device)
        .detach()
        .cpu()
        .numpy(),
        "chord_proj": _lerp(p0, p1, config.n_waypoints),
    }.items():
        off_gt_mean[name] = _interior_tree_mean(polyline, seg_a, seg_b)

    refinement: RefinementResult | None = None
    refine_summary: dict | None = None
    if config.refine.steps > 0:
        logger.info(
            "Refining %s chords: %s Adam steps, lr %s, %s waypoints, %s space, beta %s",
            len(cells),
            config.refine.steps,
            config.refine.lr,
            config.n_waypoints,
            config.refine.metric.space,
            config.refine.metric.beta,
        )
        refinement = _refine_all(projector, x0, x1, config, device)
        lengths["refine_raw"] = refinement.L_raw
        lengths["refine_proj"] = refinement.L_proj
        check_lengths(tree_gt, lengths)
        refine_summary = _summarize_refinement(refinement)
        for name, polyline in {
            "refine_raw": refinement.paths_raw,
            "refine_proj": refinement.paths_proj,
        }.items():
            off_gt_mean[name] = _interior_tree_mean(polyline, seg_a, seg_b)
        logger.info(
            "Refinement: M_off raw %.2e -> projected %.2e",
            refine_summary["m_off_raw_mean"],
            refine_summary["m_off_proj_mean"],
        )

    logger.info("Computing %s on the noisy kNN graph (k=%s)...", GRAPH_METHOD, GRAPH_K)
    graph_lengths, graph_polylines = _graph_noisy_baseline(X_pca, cells)
    lengths[GRAPH_METHOD] = graph_lengths
    check_lengths(tree_gt, lengths)
    off_gt_mean[GRAPH_METHOD] = _interior_tree_mean(graph_polylines, seg_a, seg_b)

    off_gt_methods = [key for key in LENGTH_KEYS if key in off_gt_mean]
    for method in off_gt_methods:
        logger.info("Off-GT %-12s mean %.5f", method, off_gt_mean[method].mean())
    off_gt_eval = {
        "metric": "mean interior Euclidean distance to the clean-tree polyline",
        "n_waypoints": config.n_waypoints,
        "methods": off_gt_methods,
        "mean": {
            method: float(off_gt_mean[method].mean()) for method in off_gt_methods
        },
    }

    length_summary = summarize_lengths(tree_gt, lengths)
    methods = [key for key in LENGTH_KEYS if key in lengths]
    logger.info(
        "Length MSE (n=%s, GT mean %.4f): %s",
        length_summary["n"],
        length_summary["gt_mean"],
        {key: round(length_summary[key]["mse"], 6) for key in methods},
    )

    primary_method = "refine_proj" if refinement is not None else "chord_proj"
    evaluation = {
        "scenario": config.scenario,
        "variant": config.variant,
        "seed": config.seed,
        "device": str(device),
        "inputs": {
            "projector_checkpoint": str(projector_path),
            "pairs_path": str(pairs_path),
            "n_pool_pairs": n_pool,
            "n_eval_pairs": int(len(cells)),
        },
        "settings": {
            "beta": config.refine.metric.beta,
            "space": config.refine.metric.space,
            "n_waypoints": config.n_waypoints,
            "refine_steps": config.refine.steps,
            "refine_lr": config.refine.lr,
            "pair_chunk": config.pair_chunk,
            "protocol": (
                "chord baselines only (refine.steps=0)"
                if refinement is None
                else f"chord + {config.refine.steps}-step non-amortised "
                f"refinement ({config.refine.metric.space} space)"
            ),
        },
        "primary": {
            "method": primary_method,
            "metric": "LengthMSE vs clean-tree ground truth",
            "value": length_summary[primary_method]["mse"],
        },
        "refinement": refine_summary or {"enabled": False},
        "off_gt": off_gt_eval,
        "diagnostics": {
            "clean_tree": {
                "path_names": list(tree.path_names),
                "parents": tree.parents.tolist(),
                "lengths": tree.lengths.tolist(),
                "junction_max_dist": tree.junction_max_dist,
                "step_spread_max": tree.step_spread_max,
                "junction_tol": tree.junction_tol,
            },
        },
        "length_mse": length_summary,
    }

    run.log_metrics(
        0,
        {f"{method}/mse": float(length_summary[method]["mse"]) for method in methods},
    )
    run.log_metrics(
        0,
        {
            f"off_gt/{method}/mean": value
            for method, value in off_gt_eval["mean"].items()
        },
    )
    run.save_json("evaluation.json", _sanitize(evaluation))
    np.savez(
        run.out_dir / "off_gt.npz",
        category=categories,
        cells=cells,
        methods=np.asarray(off_gt_methods),
        **{f"mean__{m}": off_gt_mean[m] for m in off_gt_methods},
    )
    paths_payload = {
        "cells": cells,
        "pair_idx": pair_idx,
        "category": categories,
        "subset_idx": subset,
        "x0": x0,
        "x1": x1,
        "x0_clean": x0_clean,
        "x1_clean": x1_clean,
        "L_tree": tree_gt,
        "L_chord_raw": lengths["chord_raw"],
        "L_chord_proj": lengths["chord_proj"],
        f"L_{GRAPH_METHOD}": lengths[GRAPH_METHOD],
    }
    if refinement is not None:
        paths_payload.update(
            {
                "L_ref_raw": refinement.L_raw,
                "L_ref_proj": refinement.L_proj,
                "M_off_raw": refinement.m_off_raw,
                "M_off_proj": refinement.m_off_proj,
                "energy_init": refinement.energy_init,
                "energy_best": refinement.energy_best,
                "paths_raw": torch.from_numpy(refinement.paths_raw),
                "paths_proj": torch.from_numpy(refinement.paths_proj),
            }
        )
    run.save_checkpoint("paths.pt", paths_payload)

    logger.info("Evaluation saved to: %s", run.out_dir / "evaluation.json")
    logger.info("Done.")


def _sanitize(obj):
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


if __name__ == "__main__":
    main()
