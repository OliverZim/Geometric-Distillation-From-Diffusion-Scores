#!/usr/bin/env python
"""Train a score model on the synthetic scRNA-seq working space."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from experiments.synthetic_data.common.space import build_train_val, load_geometry
from geometry_distill.checkpoint import model_checkpoint
from geometry_distill.config import ExperimentConfig
from geometry_distill.device import resolve_device
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.score import (
    LogUniformSigmaSchedule,
    LogUniformSigmaScheduleConfig,
    ScoreNetwork,
    ScoreNetworkConfig,
    ScoreTrainer,
    ScoreTrainerConfig,
)
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)

RESIDUAL_N_CELLS = 1000


class ScoreExperimentConfig(ExperimentConfig):
    """Configuration for the synthetic scRNA-seq score training script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["linear", "bifurcation", "tree"] = Field(
        default="linear",
        description="Synthetic scenario to train on.",
    )
    variant: Literal["full", "holdout_mid"] = Field(
        default="full",
        description=(
            "Task-4 data variant; non-full variants validate on their kept train cells."
        ),
    )
    data_root: Path = Field(
        default=Path("experiments/synthetic_data/synthetic/data"),
        description="Directory with the simulation outputs and geometry artifacts.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (weights, noise, shuffling, subsamples).",
    )
    n_val: int = Field(
        default=1500,
        ge=1,
        description="Number of validation cells (capped by availability).",
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="Training batch size.",
    )
    pseudo_sigma: float | None = Field(
        default=None,
        description="Pseudo-target noise level for the residual sanity; None skips it.",
    )

    schedule: LogUniformSigmaScheduleConfig = Field(
        default_factory=lambda: LogUniformSigmaScheduleConfig(
            sigma_min=0.0035,
            sigma_max=0.05,
        ),
        description="Training noise schedule.",
    )
    score: ScoreNetworkConfig = Field(
        default_factory=lambda: ScoreNetworkConfig(hidden_dim=256, n_layers=3),
        description="Score network; x_dim is derived from the working space.",
    )
    trainer: ScoreTrainerConfig = Field(
        default_factory=lambda: ScoreTrainerConfig(
            epochs=1000,
            patience=300,
            lr_schedule="cosine",
        ),
        description="Trainer hyperparameters.",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )


@torch.no_grad()
def residual_sanity(
    model: torch.nn.Module,
    x_cells: np.ndarray,
    X_full: np.ndarray,
    sigma: float,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Return the mean 5th-NN distance of the cells before and after one
    pseudo-projection step ``x + sigma^2 * s(x, sigma)``."""
    model.eval()
    x_t = torch.as_tensor(x_cells, dtype=torch.float32, device=device)
    sigma_t = torch.full((x_t.shape[0],), float(sigma), device=device)
    s = model(x_t, sigma_t)
    proj = x_t + float(sigma) ** 2 * s
    X_full = np.asarray(X_full, dtype=np.float64)
    tree = cKDTree(X_full)
    d_before, _ = tree.query(np.asarray(x_cells, dtype=np.float64), k=6)
    d_after, _ = tree.query(proj.cpu().numpy().astype(np.float64), k=6)
    return float(d_before[:, 5].mean()), float(d_after[:, 5].mean())


def main() -> None:
    config = ScoreExperimentConfig()

    set_seed(config.seed)

    try:
        geom = load_geometry(config.scenario, config.data_root)
        config = config.model_copy(
            update={
                "score": config.score.model_copy(
                    update={"x_dim": geom["X_pca"].shape[1]}
                ),
            }
        )
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run, geom)


def _main_impl(
    config: ScoreExperimentConfig,
    run: RunWriter,
    geom: dict,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    X_pca = geom["X_pca"]
    logger.info("Scenario: %s  variant: %s", config.scenario, config.variant)
    logger.info("Working space: %s cells x %s PCs", *X_pca.shape)

    train_idx, val_idx, held_out = build_train_val(geom, config.variant, config.n_val)
    logger.info("Train cells: %s (variant '%s')", len(train_idx), config.variant)
    logger.info(
        "Val cells: %s (%s)",
        len(val_idx),
        "val split" if config.variant == "full" else "variant kept train cells",
    )
    logger.info("Held-out cells (not used): %s", len(held_out))

    train_dataset = TensorDataset(torch.from_numpy(X_pca[train_idx]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_dataset = TensorDataset(torch.from_numpy(X_pca[val_idx]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    score_config = config.score
    logger.info("Building score network...")
    score_net = ScoreNetwork(score_config).to(device)
    n_params = sum(p.numel() for p in score_net.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    schedule = LogUniformSigmaSchedule(config.schedule)
    logger.info(
        "Sigma schedule: log-uniform [%s, %s]",
        config.schedule.sigma_min,
        config.schedule.sigma_max,
    )

    trainer = ScoreTrainer(config.trainer, score_net, schedule)

    logger.info("Training...")
    history = trainer.fit(train_loader, val_loader=val_loader)

    n_epochs = len(history["train_loss"])
    for epoch in range(n_epochs):
        run.log_metrics(epoch, {key: v[epoch] for key, v in history.items()})

    monitor_key = "val_bin_mean" if "val_bin_mean" in history else "val_loss"
    best_epoch = int(np.argmin(history[monitor_key]))
    best_val_loss = float(history[monitor_key][best_epoch])
    best_val_aggregate = float(history["val_loss"][best_epoch])
    final_loss = float(history["train_loss"][-1])
    bin_keys = sorted(k for k in history if k.startswith("val_loss_sigma_"))
    per_bin = {k: float(history[k][best_epoch]) for k in bin_keys}
    logger.info("  Trained %d/%d epochs", n_epochs, config.trainer.epochs)
    logger.info("  Final train loss: %.6f", final_loss)
    logger.info(
        "  Best val loss:    %.6f (epoch %d, monitor '%s')",
        best_val_loss,
        best_epoch,
        monitor_key,
    )
    logger.info(
        "  Per-sigma val losses (best epoch): %s",
        ", ".join(f"σ={k.rsplit('_', 1)[-1]}={v:.4f}" for k, v in per_bin.items()),
    )

    logger.info("Running residual sanity...")
    residual_row: dict = {"sigma": config.pseudo_sigma, "n_cells": 0}
    if config.pseudo_sigma is not None:
        residual_idx = np.random.choice(
            train_idx, size=min(RESIDUAL_N_CELLS, len(train_idx)), replace=False
        ).astype(np.int64)
        x_res = np.asarray(X_pca[residual_idx], dtype=np.float32)
        nn5_before, nn5_after = residual_sanity(
            score_net, x_res, X_pca, config.pseudo_sigma, device=device
        )
        residual_row = {
            "sigma": config.pseudo_sigma,
            "n_cells": int(len(residual_idx)),
            "nn5_before": nn5_before,
            "nn5_after": nn5_after,
        }
        logger.info("  Mean 5-NN %.4f -> %.4f", nn5_before, nn5_after)
    else:
        logger.info("  Skipped (pseudo_sigma not set)")

    evaluation = {
        "scenario": config.scenario,
        "variant": config.variant,
        "n_train_cells": int(len(train_idx)),
        "n_val_cells": int(len(val_idx)),
        "n_held_out_cells": int(len(held_out)),
        "n_epochs_trained": int(n_epochs),
        "best_epoch": best_epoch,
        "val_monitor": monitor_key,
        "best_val_loss": best_val_loss,
        "best_val_loss_aggregate": best_val_aggregate,
        "final_train_loss": final_loss,
        "per_sigma_bin_val_losses": per_bin,
        "residual_sanity": residual_row,
    }
    run.save_json("evaluation.json", evaluation)

    checkpoint_path = run.save_checkpoint(
        "score.pt", model_checkpoint(score_net, score_config)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
