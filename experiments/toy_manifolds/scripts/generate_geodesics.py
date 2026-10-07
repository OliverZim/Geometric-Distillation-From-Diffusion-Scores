#!/usr/bin/env python
"""Chord-refine and project geodesics on toy manifolds."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Self

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pydantic import Field, model_validator  # noqa: E402
from pydantic_settings import SettingsConfigDict  # noqa: E402

from experiments.toy_manifolds.common import (  # noqa: E402
    MANIFOLDS,
    Manifold,
    ManifoldName,
    sample_uniform,
)
from geometry_distill.config import ExperimentConfig  # noqa: E402
from geometry_distill.device import resolve_device  # noqa: E402
from geometry_distill.geodesics import (  # noqa: E402
    ManifoldPathEvaluator,
    ManifoldPathEvaluatorConfig,
    PathRefinementConfig,
    WarpedPullbackMetricConfig,
    chord_paths,
    refine_paths,
)
from geometry_distill.projection import (  # noqa: E402
    projector_config_from_checkpoint,
    projector_from_checkpoint,
)
from geometry_distill.run import RunConfig, RunWriter  # noqa: E402
from geometry_distill.seeding import set_seed  # noqa: E402

logger = logging.getLogger(__name__)


class GeodesicExperimentConfig(ExperimentConfig):
    """Configuration for the toy-manifold geodesics script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    manifold: ManifoldName = Field(
        default="hemisphere",
        description="Toy manifold to refine on.",
    )
    projector_checkpoint: Path = Field(
        description="Path to the projector checkpoint (stage 2 output).",
    )
    n_eval_pairs: int = Field(
        default=20,
        ge=1,
        description="Number of random endpoint pairs to evaluate.",
    )
    refine: PathRefinementConfig = Field(
        default_factory=lambda: PathRefinementConfig(
            metric=WarpedPullbackMetricConfig(space="ambient"),
        ),
        description="Non-amortised chord refinement under the warped metric.",
    )
    n_waypoints: int = Field(
        default=31,
        ge=2,
        description="Waypoints of the chord polyline used for refinement.",
    )
    evaluator: ManifoldPathEvaluatorConfig = Field(
        default_factory=ManifoldPathEvaluatorConfig,
        description="Dijkstra-baseline evaluator over the clean background cloud.",
    )
    n_background: int = Field(
        default=100000,
        ge=1,
        description="Number of points in the Dijkstra background cloud.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (data, endpoint pairs, refinement).",
    )
    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @model_validator(mode="after")
    def _check_refine_steps(self) -> Self:
        # This script always refines, so a no-op refinement is an error.
        if self.refine.steps < 1:
            raise ValueError(f"refine.steps must be >= 1, got {self.refine.steps}")
        return self


def resolve_projector_checkpoint(config: GeodesicExperimentConfig) -> Path:
    """Return the configured projector checkpoint path, failing if missing."""
    path = Path(config.projector_checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Projector checkpoint not found at: {path}\n"
            "  (set --projector-checkpoint or train one with "
            "experiments/toy_manifolds/scripts/train_projection.py)"
        )
    return path


def _sample_pairs_uniform(
    manifold: Manifold,
    n_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n_pairs`` random endpoint pairs on the clean manifold."""
    points, _ = sample_uniform(manifold, n_points=2 * n_pairs)
    return points[:n_pairs], points[n_pairs:]


def _refine_eval_chords(
    projector: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    config: GeodesicExperimentConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Refine chords under the warped metric and project the waypoints."""
    chords = chord_paths(x0, x1, config.n_waypoints, device)
    refined, info = refine_paths(chords, projector, config.refine)
    raw = refined.detach()
    with torch.no_grad():
        proj = projector(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape)
    return {
        "chord": chords.detach(),
        "raw": raw,
        "proj": proj.detach(),
        "energy_init": torch.as_tensor(info["energy_init"], dtype=torch.float32),
        "energy_best": torch.as_tensor(info["energy_best"], dtype=torch.float32),
        "energy_trace": torch.as_tensor(info["energy_trace"], dtype=torch.float32),
    }


def _summarise_refinement(
    refined: dict[str, torch.Tensor],
    rows_raw: list[dict[str, float]],
    rows_proj: list[dict[str, float]],
    config: GeodesicExperimentConfig,
) -> dict:
    """Aggregate refinement diagnostics and acceptance flags."""
    L_chord = np.array([m["L_straight"] for m in rows_proj])
    L_dijkstra = np.array([m["L_dijkstra"] for m in rows_proj])
    L_ref_raw = np.array([m["L_ambient"] for m in rows_raw])
    L_ref_proj = np.array([m["L_ambient"] for m in rows_proj])
    m_off_raw = np.array([m["M_off"] for m in rows_raw])
    m_off_proj = np.array([m["M_off"] for m in rows_proj])
    energy_init = refined["energy_init"].numpy().astype(np.float64)
    energy_best = refined["energy_best"].numpy().astype(np.float64)
    if (energy_best > energy_init + 1e-12).any():
        raise ValueError(
            "refinement returned a best energy above its initialisation; "
            "best-iterate tracking in refine_paths should never increase"
        )
    drop = 1.0 - float(energy_best.mean() / energy_init.mean())
    rel_err = (L_ref_proj - L_dijkstra) / L_dijkstra
    return {
        "steps": config.refine.steps,
        "lr": config.refine.lr,
        "space": config.refine.metric.space,
        "n_waypoints": config.n_waypoints,
        "energy_init_mean": float(energy_init.mean()),
        "energy_best_mean": float(energy_best.mean()),
        "energy_drop_frac": drop,
        "L_chord_mean": float(L_chord.mean()),
        "L_ref_raw_mean": float(L_ref_raw.mean()),
        "L_ref_proj_mean": float(L_ref_proj.mean()),
        "L_dijkstra_mean": float(L_dijkstra.mean()),
        "LengthMSE": float(np.mean((L_ref_proj - L_dijkstra) ** 2)),
        "LengthMSE_chord": float(np.mean((L_chord - L_dijkstra) ** 2)),
        "rel_err_mean": float(np.mean(rel_err)),
        "rel_err_std": float(np.std(rel_err)),
        "rho_mean": float(np.mean(L_ref_proj / L_dijkstra)),
        "m_off_raw_mean": float(m_off_raw.mean()),
        "m_off_proj_mean": float(m_off_proj.mean()),
        "flags": {
            "energy_drop_ok": bool(drop >= 0.8),
            "m_off_raw_ok": bool(m_off_raw.mean() <= 1e-1),
            "m_off_proj_ok": bool(m_off_proj.mean() <= 1e-4),
            "length_sanity_ok": bool(L_ref_proj.mean() <= 2.0 * L_chord.mean()),
        },
    }


def _plot_refinement_3d(
    X_background: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    chords: torch.Tensor,
    refined_proj: torch.Tensor,
    L_chord: np.ndarray,
    L_dijkstra: np.ndarray,
    path: Path,
) -> None:
    """Chord (dashed) vs refine+project (solid) for the worst chord pairs."""
    n_show = min(5, chords.shape[0])
    worst = np.argsort(-np.abs(L_chord - L_dijkstra))[:n_show]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    n_bg = min(4000, X_background.shape[0])
    bg = X_background[:n_bg].numpy()
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=1.0, alpha=0.3, c="0.6")
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_show))
    for j, i in enumerate(worst):
        chord = chords[i].numpy()
        ref = refined_proj[i].numpy()
        ax.plot(chord[:, 0], chord[:, 1], chord[:, 2], lw=1.2, ls="--", color="0.35")
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], lw=2.0, color=colors[j])
        ax.scatter(
            [x0[i, 0].item(), x1[i, 0].item()],
            [x0[i, 1].item(), x1[i, 1].item()],
            [x0[i, 2].item(), x1[i, 2].item()],
            marker="o",
            s=25,
            color=colors[j],
            edgecolors="k",
            linewidths=0.5,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("chord (dashed) vs refine+project (solid)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = GeodesicExperimentConfig()

    set_seed(config.seed)

    try:
        projector_ckpt_path = resolve_projector_checkpoint(config)
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run, projector_ckpt_path)


def _main_impl(
    config: GeodesicExperimentConfig,
    run: RunWriter,
    projector_ckpt_path: Path,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    manifold = MANIFOLDS[config.manifold]
    logger.info("Manifold: %s", manifold.name)

    logger.info("Loading trained projector from: %s", projector_ckpt_path)
    ckpt = torch.load(projector_ckpt_path, map_location=device, weights_only=False)
    projector = projector_from_checkpoint(ckpt, device=device)
    projector.requires_grad_(False)
    logger.info(
        "  Loaded projector: %s", projector_config_from_checkpoint(ckpt["config"])
    )

    logger.info("Sampling endpoint pairs...")
    x0_eval, x1_eval = _sample_pairs_uniform(manifold, config.n_eval_pairs)
    logger.info("  Eval pairs: %s", x0_eval.shape[0])

    logger.info("Building Dijkstra baseline...")
    X_background, _ = sample_uniform(manifold, n_points=config.n_background)
    evaluator = ManifoldPathEvaluator(config.evaluator, X_background)
    projector_cpu = copy.deepcopy(projector).to("cpu")

    logger.info(
        "Chord refinement + projection: %d Adam steps, lr=%s, space=%s, %d waypoints",
        config.refine.steps,
        config.refine.lr,
        config.refine.metric.space,
        config.n_waypoints,
    )
    refined = _refine_eval_chords(projector, x0_eval, x1_eval, config, device)
    rows_raw = [
        evaluator.evaluate_path(
            refined["raw"][i].cpu(),
            x0_eval[i].cpu(),
            x1_eval[i].cpu(),
            projector_cpu,
        )
        for i in range(config.n_eval_pairs)
    ]
    rows_proj = [
        evaluator.evaluate_path(
            refined["proj"][i].cpu(),
            x0_eval[i].cpu(),
            x1_eval[i].cpu(),
            projector_cpu,
        )
        for i in range(config.n_eval_pairs)
    ]
    refinement = _summarise_refinement(refined, rows_raw, rows_proj, config)

    L_chord = np.array([r["L_straight"] for r in rows_proj])
    L_dijkstra = np.array([r["L_dijkstra"] for r in rows_proj])
    L_ref_raw = np.array([r["L_ambient"] for r in rows_raw])
    L_ref_proj = np.array([r["L_ambient"] for r in rows_proj])
    rho = np.array([r["rho"] for r in rows_proj])
    M_off_raw = np.array([r["M_off"] for r in rows_raw])
    M_off_proj = np.array([r["M_off"] for r in rows_proj])

    logger.info("=" * 90)
    logger.info(
        "%4s %11s %11s %11s %11s %7s %10s %11s",
        "pair",
        "L_chord",
        "L_dijkstra",
        "L_ref_raw",
        "L_ref_proj",
        "rho",
        "M_off_raw",
        "M_off_proj",
    )
    logger.info("-" * 90)
    for i in range(config.n_eval_pairs):
        logger.info(
            "%4d %11.4f %11.4f %11.4f %11.4f %7.3f %10.3e %11.3e",
            i,
            L_chord[i],
            L_dijkstra[i],
            L_ref_raw[i],
            L_ref_proj[i],
            rho[i],
            M_off_raw[i],
            M_off_proj[i],
        )
    logger.info("-" * 90)
    logger.info("LengthMSE (refine+proj vs Dijkstra): %.6f", refinement["LengthMSE"])
    logger.info(
        "LengthMSE (chord vs Dijkstra):       %.6f", refinement["LengthMSE_chord"]
    )
    logger.info(
        "Relative length error:               %.4f ± %.4f",
        refinement["rel_err_mean"],
        refinement["rel_err_std"],
    )
    logger.info("rho = L_ref_proj / L_dijkstra:       %.4f", refinement["rho_mean"])
    logger.info(
        "M_off raw %.3e, projected %.3e; energy %.3e -> %.3e (drop %.1f%%); flags %s",
        refinement["m_off_raw_mean"],
        refinement["m_off_proj_mean"],
        refinement["energy_init_mean"],
        refinement["energy_best_mean"],
        refinement["energy_drop_frac"] * 100.0,
        refinement["flags"],
    )
    logger.info("=" * 90)

    eval_metrics = {
        "manifold": manifold.name,
        "n_eval_pairs": config.n_eval_pairs,
        "LengthMSE": refinement["LengthMSE"],
        "LengthMSE_straight": refinement["LengthMSE_chord"],
        "rel_err_mean": refinement["rel_err_mean"],
        "rel_err_std": refinement["rel_err_std"],
        "rho_mean": refinement["rho_mean"],
        "L_ref_raw_mean": refinement["L_ref_raw_mean"],
        "L_ref_proj_mean": refinement["L_ref_proj_mean"],
        "L_dijkstra_mean": refinement["L_dijkstra_mean"],
        "L_chord_mean": refinement["L_chord_mean"],
        "projector_checkpoint": str(config.projector_checkpoint),
        "beta": config.refine.metric.beta,
        "space": config.refine.metric.space,
        "n_background": config.n_background,
        "k_neighbors": config.evaluator.n_neighbors,
        "refinement": refinement,
    }
    evaluation_path = run.save_json("evaluation.json", eval_metrics)
    logger.info("Evaluation saved to: %s", evaluation_path)

    paths_path = run.save_checkpoint(
        "paths.pt",
        {
            "x0": x0_eval.cpu(),
            "x1": x1_eval.cpu(),
            "refined_chord": refined["chord"].cpu(),
            "refined_raw": refined["raw"].cpu(),
            "refined_proj": refined["proj"].cpu(),
            "L_chord": torch.from_numpy(L_chord),
            "L_ref_raw": torch.from_numpy(L_ref_raw),
            "L_ref_proj": torch.from_numpy(L_ref_proj),
            "L_dijkstra": torch.from_numpy(L_dijkstra),
            "M_off_ref_raw": torch.from_numpy(M_off_raw),
            "M_off_ref_proj": torch.from_numpy(M_off_proj),
            "rho_ref_proj": torch.from_numpy(rho),
            "energy_init": refined["energy_init"],
            "energy_best": refined["energy_best"],
            "energy_trace": refined["energy_trace"],
        },
    )
    logger.info("Paths saved to: %s", paths_path)

    plot_path = run.out_dir / "refinement_3d.png"
    _plot_refinement_3d(
        X_background,
        x0_eval.cpu(),
        x1_eval.cpu(),
        refined["chord"].cpu(),
        refined["proj"].cpu(),
        L_chord,
        L_dijkstra,
        plot_path,
    )
    logger.info("Refinement plot saved to: %s", plot_path)

    run.log_metrics(
        config.refine.steps,
        {
            "LengthMSE": refinement["LengthMSE"],
            "LengthMSE_straight": refinement["LengthMSE_chord"],
            "rho_mean": refinement["rho_mean"],
            "m_off_proj_mean": refinement["m_off_proj_mean"],
        },
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
