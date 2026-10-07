#!/usr/bin/env python
"""Volume-guided Langevin sampling on toy manifolds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from experiments.toy_manifolds.common import (
    MANIFOLDS,
    Manifold,
    ManifoldName,
    evaluate_uniformity,
    fit_kde,
    manifold_distance,
    sample_imbalanced,
    sample_uniform,
    summarize_manifold_distances,
)
from geometry_distill.config import ExperimentConfig
from geometry_distill.device import resolve_device
from geometry_distill.projection import projector_from_checkpoint
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.sampling import (
    LangevinConfig,
    encoder_jacobian,
    log_volume_from_jacobian,
    projection_residual_squared,
    sample_langevin,
)
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)


class GenerationExperimentConfig(ExperimentConfig):
    """Configuration for the volume-guided generation script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    manifold: ManifoldName = Field(
        default="hemisphere",
        description="Toy manifold to generate samples for.",
    )
    projector_checkpoint: Path = Field(
        description="Path to the projector checkpoint (stage 2 output).",
    )

    lambda_pull: float = Field(
        default=100.0,
        ge=0.0,
        description="Weight of the manifold-pulling term in the energy.",
    )
    langevin: LangevinConfig = Field(
        default_factory=lambda: LangevinConfig(n_steps=10000),
        description="Langevin dynamics parameters.",
    )

    filter_multiple: float = Field(
        default=0.0,
        description=(
            "Residual filter threshold as a multiple of the mean training "
            "residual; <= 0 keeps all samples."
        ),
    )
    final_project: bool = Field(
        default=False,
        description="Project the samples once more after Langevin dynamics.",
    )

    save_figure_data: bool = Field(
        default=False,
        description="Save figure_data.npz with the raw per-point arrays for figures.",
    )
    figure_bias_n: int = Field(
        default=5000,
        ge=1,
        description="Number of imbalanced points saved for the biased-density panel.",
    )
    figure_plot_domain: Literal["disk", "square"] | None = Field(
        default=None,
        description="Figure plot domain in parameter space; defaults to the manifold's.",
    )
    figure_plot_half_width: float | None = Field(
        default=None,
        gt=0.0,
        description="Half-width of the figure plot domain; defaults to the manifold's.",
    )

    n_train: int = Field(
        default=3000,
        ge=1,
        description="Number of imbalanced points for residual-threshold calibration.",
    )
    n_generate: int = Field(
        default=3000,
        ge=1,
        description="Number of independently sampled manifold points to initialize.",
    )
    jitter_scale: float = Field(
        default=0.005,
        ge=0.0,
        description="Gaussian noise scale for jittering initial samples.",
    )
    kde_fit_mask_only: bool = Field(
        default=False,
        description="Fit the uniformity KDE on the masked (u,v) points only.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (data, jitter, Langevin noise).",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @model_validator(mode="after")
    def _fill_figure_plot_domain(self) -> "GenerationExperimentConfig":
        manifold = MANIFOLDS[self.manifold]
        if self.figure_plot_domain is None:
            self.figure_plot_domain = manifold.plot_domain
        if self.figure_plot_half_width is None:
            self.figure_plot_half_width = manifold.plot_half_width
        return self


def resolve_projector_checkpoint(config: GenerationExperimentConfig) -> Path:
    """Return the configured projector checkpoint path, failing if it is missing."""
    path = Path(config.projector_checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Projector checkpoint not found at: {path}\n"
            "  (set --projector-checkpoint or train one with "
            "experiments/toy_manifolds/scripts/train_projection.py)"
        )
    return path


_GAGA_BASELINES: dict[str, dict[str, float]] = {
    "hemisphere": {"R": 0.85, "R2": 0.71},
    "saddle": {"R": 0.60, "R2": 0.36},
    "paraboloid": {"R": 0.66, "R2": 0.44},
}

_ORIGINAL_BASELINES: dict[str, dict[str, float]] = {
    "hemisphere": {"R": 0.03, "R2": 0.00},
    "saddle": {"R": 0.04, "R2": 0.00},
    "paraboloid": {"R": 0.04, "R2": 0.00},
}


def make_energy_fn(
    projector: nn.Module,
    lambda_pull: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the volume-guided Langevin energy function."""

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        z = projector.encode(x)
        jacobian = encoder_jacobian(projector, x, z=z)
        x_hat = projector.decode(z)
        pull_term = 0.5 * ((x - x_hat) ** 2).sum(dim=-1)
        return lambda_pull * pull_term - log_volume_from_jacobian(jacobian)

    return energy_fn


_PLOT_EPS = 1e-6


def _plot_domain_mask(
    uv: np.ndarray,
    domain: str,
    half_width: float,
) -> np.ndarray:
    """Boolean mask of parameter points inside the figure plot domain."""
    if domain == "disk":
        return (uv[:, 0] ** 2 + uv[:, 1] ** 2) <= half_width**2 + _PLOT_EPS
    return (np.abs(uv[:, 0]) <= half_width + _PLOT_EPS) & (
        np.abs(uv[:, 1]) <= half_width + _PLOT_EPS
    )


def _save_figure_data(
    run: RunWriter,
    config: GenerationExperimentConfig,
    manifold: Manifold,
    points: torch.Tensor,
    dists_all: np.ndarray,
    eval_result: dict,
) -> Path:
    """Save ``figure_data.npz`` with the raw arrays used for figures."""
    set_seed(config.seed)
    _, uv_biased = sample_imbalanced(manifold, n_points=config.figure_bias_n)
    uv_gen = manifold.inverse(points[:, 0], points[:, 1], points[:, 2])

    kde_biased = fit_kde(uv_biased)
    kde_gen = fit_kde(uv_gen)
    c_biased = np.asarray(kde_biased(uv_biased), dtype=np.float64)
    c_gen = np.asarray(kde_gen(uv_gen), dtype=np.float64)

    uv_gen_np = uv_gen.detach().cpu().numpy()
    points_biased = manifold.parametrization(uv_biased[:, 0], uv_biased[:, 1])
    R_biased = float(evaluate_uniformity(points_biased, manifold)["R"])

    arrays: dict[str, np.ndarray] = {
        "uv_biased": uv_biased.detach().cpu().numpy(),
        "c_biased": c_biased,
        "uv_gen": uv_gen_np,
        "c_gen": c_gen,
        "dists_all": np.asarray(dists_all, dtype=np.float64),
        "plot_mask": _plot_domain_mask(
            uv_gen_np, config.figure_plot_domain, config.figure_plot_half_width
        ),
        "plot_half_width": np.asarray(config.figure_plot_half_width),
        "plot_domain": np.asarray(config.figure_plot_domain),
        "R_biased": np.asarray(R_biased, dtype=np.float64),
        "R_langevin": np.asarray(float(eval_result["R"]), dtype=np.float64),
    }
    if "density" in eval_result:
        arrays["uv_masked"] = eval_result["uv_masked"].detach().cpu().numpy()
        arrays["density"] = np.asarray(eval_result["density"], dtype=np.float64)
        arrays["vol_elem"] = np.asarray(eval_result["vol_elem"], dtype=np.float64)

    path = run.out_dir / "figure_data.npz"
    np.savez(path, **arrays)
    return path


def main() -> None:
    config = GenerationExperimentConfig()

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
    config: GenerationExperimentConfig,
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
    logger.info("  Loaded projector: %s", projector.config)

    filter_active = config.filter_multiple > 0.0
    if filter_active:
        logger.info("Computing filter threshold...")
        points_train, _uv_train = sample_imbalanced(manifold, n_points=config.n_train)
        points_train = points_train.to(device)

        with torch.no_grad():
            residuals_train = projection_residual_squared(projector, points_train)
            mean_residual = float(residuals_train.mean().item())
        filter_threshold = config.filter_multiple * mean_residual
        logger.info("  Mean training residual: %.6e", mean_residual)
        logger.info(
            "  Filter threshold (ε = %.1f×): %.6e",
            config.filter_multiple,
            filter_threshold,
        )
    else:
        mean_residual = None
        filter_threshold = None
        logger.info(
            "Residual filtering disabled; skipping filter-threshold calibration."
        )

    logger.info("Initialising Langevin samples...")
    points_init, _ = sample_uniform(
        manifold,
        n_points=config.n_generate,
    )
    points_init = points_init.to(device)

    # Jitter so initial points are not exactly stationary under the
    # manifold-pulling term.
    jitter = config.jitter_scale * torch.randn(
        points_init.shape,
        device=device,
        dtype=points_init.dtype,
    )
    x_init = points_init + jitter
    logger.info("  Initial samples: %s", x_init.shape)

    energy_fn = make_energy_fn(projector, lambda_pull=config.lambda_pull)

    logger.info(
        "Running Langevin dynamics (n_steps=%d, η=%s)...",
        config.langevin.n_steps,
        config.langevin.step_size,
    )
    x_final = sample_langevin(x_init, energy_fn, config=config.langevin)
    logger.info("  Final samples: %s", x_final.shape)

    if config.final_project:
        logger.info("Applying final projection...")
        with torch.no_grad():
            x_final = projector(x_final)

    n_total = x_final.shape[0]

    if filter_active:
        logger.info("Filtering samples by residual threshold...")
        with torch.no_grad():
            residuals_final = projection_residual_squared(projector, x_final)
            keep_mask = residuals_final < filter_threshold
            x_filtered = x_final[keep_mask]
        n_kept = int(x_filtered.shape[0])
        logger.info(
            "  Kept: %d/%d (%.1f%%)",
            n_kept,
            n_total,
            100.0 * n_kept / max(n_total, 1),
        )
        if n_kept < 10:
            logger.warning(
                "Very few samples passed the filter. Consider increasing "
                "--filter-multiple or checking the projector quality."
            )
    else:
        logger.info(
            "Residual filtering disabled (--filter-multiple <= 0.0), "
            "keeping all samples."
        )
        x_filtered = x_final
        n_kept = n_total

    logger.info("Evaluating uniformity...")
    eval_result = evaluate_uniformity(
        x_filtered,
        manifold,
        return_arrays=config.save_figure_data,
        fit_mask_only=config.kde_fit_mask_only,
    )

    R = float(eval_result["R"])
    R2 = float(eval_result["R2"])
    n_masked = int(eval_result["n_masked"])
    logger.info("  R  = %.4f", R)
    logger.info("  R² = %.4f", R2)
    logger.info("  Masked: %d/%d", n_masked, n_kept)

    logger.info("Evaluating distance to manifold...")
    dists_all = manifold_distance(x_filtered, manifold)
    dist_metrics = summarize_manifold_distances(dists_all)
    logger.info(
        "  Distance to manifold: mean %.4f, median %.4f, p90 %.4f, max %.4f; "
        "frac<0.1 %.3f, frac<0.3 %.3f",
        dist_metrics["dist_mean"],
        dist_metrics["dist_median"],
        dist_metrics["dist_p90"],
        dist_metrics["dist_max"],
        dist_metrics["frac_dist_lt_0.1"],
        dist_metrics["frac_dist_lt_0.3"],
    )

    log_metrics: dict[str, float] = {
        "R": R,
        "R2": R2,
        "n_total": float(n_total),
        "n_kept": float(n_kept),
        "n_masked": float(n_masked),
        "dist_mean": dist_metrics["dist_mean"],
        "dist_median": dist_metrics["dist_median"],
        "dist_p90": dist_metrics["dist_p90"],
        "frac_dist_lt_0.1": dist_metrics["frac_dist_lt_0.1"],
    }
    if mean_residual is not None:
        log_metrics["mean_training_residual"] = mean_residual
    run.log_metrics(0, log_metrics)

    orig = _ORIGINAL_BASELINES[manifold.name]
    gaga = _GAGA_BASELINES[manifold.name]
    logger.info("=" * 72)
    logger.info("%-20s %8s %8s %20s", "Benchmark", "R", "R²", "Source")
    logger.info("-" * 72)
    logger.info(
        "%-20s %8.2f %8.2f %20s",
        "Original data",
        orig["R"],
        orig["R2"],
        "GAGA Table 2",
    )
    logger.info("%-20s %8.2f %8.2f %20s", "GAGA", gaga["R"], gaga["R2"], "GAGA Table 2")
    logger.info("%-20s %8.4f %8.4f %20s", "Ours", R, R2, "(this run)")
    logger.info("=" * 72)

    points_path = run.save_checkpoint(
        "generated_points.pt",
        {
            "points_3d": x_filtered.cpu(),
            "n_total": n_total,
            "n_kept": n_kept,
            "filter_threshold": filter_threshold,
        },
    )
    logger.info("Generated points saved to: %s", points_path)

    evaluation = {
        "manifold": manifold.name,
        "R": R,
        "R2": R2,
        "n_masked": n_masked,
        "n_total_masked_candidates": n_kept,
        "n_generated": n_total,
        "n_filtered": n_kept,
        "filter_threshold": filter_threshold,
        "mean_training_residual": mean_residual,
        "final_project": config.final_project,
        "kde_fit_mask_only": config.kde_fit_mask_only,
        "lambda_pull": config.lambda_pull,
        "step_size": config.langevin.step_size,
        "n_steps": config.langevin.n_steps,
        "noise_scale": config.langevin.noise_scale,
        "seed": config.seed,
        "gaga_R": gaga["R"],
        "gaga_R2": gaga["R2"],
        "original_data_R": orig["R"],
        "original_data_R2": orig["R2"],
        **dist_metrics,
    }
    if config.save_figure_data:
        figure_path = _save_figure_data(
            run=run,
            config=config,
            manifold=manifold,
            points=x_filtered,
            dists_all=dists_all,
            eval_result=eval_result,
        )
        evaluation["figure_data"] = figure_path.name
        logger.info("Figure data saved to: %s", figure_path)
    evaluation_path = run.save_json("evaluation.json", evaluation)
    logger.info("Evaluation saved to: %s", evaluation_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
