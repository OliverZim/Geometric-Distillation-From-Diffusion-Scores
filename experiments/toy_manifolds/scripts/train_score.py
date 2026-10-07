#!/usr/bin/env python
"""Train a score model on a toy manifold."""

from __future__ import annotations

import logging
from typing import Literal, Self

import numpy as np
import torch
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from torch.utils.data import DataLoader, TensorDataset

from experiments.toy_manifolds.common import (
    MANIFOLDS,
    ManifoldName,
    sample_imbalanced,
    sample_uniform,
)
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


class ScoreExperimentConfig(ExperimentConfig):
    """Configuration for the toy-manifold score training script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    manifold: ManifoldName = Field(
        default="hemisphere",
        description="Toy manifold to train on.",
    )

    n_train: int = Field(
        default=20000,
        ge=1,
        description="Number of training points.",
    )
    n_val: int = Field(
        default=1000,
        ge=1,
        description="Number of validation points.",
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="Training batch size.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (data, weights, noise, shuffling).",
    )
    val_distribution: Literal["imbalanced", "uniform"] = Field(
        default="imbalanced",
        description=(
            "'imbalanced' = same GAGA protocol distribution as the training "
            "data; 'uniform' = uniform over the parameter domain."
        ),
    )

    schedule: LogUniformSigmaScheduleConfig = Field(
        default_factory=LogUniformSigmaScheduleConfig,
        description="Log-uniform training noise schedule.",
    )

    score: ScoreNetworkConfig = Field(
        default_factory=ScoreNetworkConfig,
        description="Score network; x_dim is derived from the manifold.",
    )
    trainer: ScoreTrainerConfig = Field(
        default_factory=ScoreTrainerConfig,
        description="Trainer hyperparameters.",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @model_validator(mode="after")
    def _apply_manifold_x_dim(self) -> Self:
        """Force the score network's x_dim to the manifold's ambient dimension."""
        self.score = self.score.model_copy(
            update={"x_dim": MANIFOLDS[self.manifold].x_dim}
        )
        return self


def main() -> None:
    config = ScoreExperimentConfig()

    set_seed(config.seed)

    try:
        run = RunWriter(config.run)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run)


def _main_impl(config: ScoreExperimentConfig, run: RunWriter) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    manifold = MANIFOLDS[config.manifold]
    logger.info("Manifold: %s", manifold.name)

    logger.info("Generating training data...")
    points_train, uv_train = sample_imbalanced(manifold, n_points=config.n_train)
    train_dataset = TensorDataset(points_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    logger.info("  Training points: %s", points_train.shape)

    logger.info("Generating validation data...")
    if config.val_distribution == "imbalanced":
        points_val, uv_val = sample_imbalanced(manifold, n_points=config.n_val)
    else:
        points_val, uv_val = sample_uniform(manifold, n_points=config.n_val)
    logger.info(
        "  Validation points: %s (%s distribution)",
        points_val.shape,
        config.val_distribution,
    )

    logger.info("Building score network...")
    score_net = ScoreNetwork(config.score).to(device)
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
    val_dataset = TensorDataset(points_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    history = trainer.fit(train_loader, val_loader=val_loader)

    n_epochs = len(history["train_loss"])
    for epoch in range(n_epochs):
        run.log_metrics(epoch, {k: v[epoch] for k, v in history.items()})

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
    per_bin_str = ", ".join(
        f"σ={k.rsplit('_', 1)[-1]}={v:.4f}" for k, v in per_bin.items()
    )
    logger.info("  Per-sigma val losses (best epoch): %s", per_bin_str)

    logger.info("Running diagnostic (score pointing toward manifold)...")
    score_net.eval()
    z_change_rows: list[dict] = []
    with torch.no_grad():
        diag_x = points_val[:10].to(device)
        for sigma_val in [0.05, 0.1, 0.2, 0.4]:
            sigma_t = torch.full((diag_x.shape[0],), sigma_val, device=device)
            score = score_net(diag_x, sigma_t)
            pseudo = diag_x + sigma_val**2 * score
            z_change = pseudo[:, 2] - diag_x[:, 2]
            z_change_rows.append(
                {"sigma": sigma_val, "mean_z_change": float(z_change.mean().item())}
            )
            logger.info(
                "    σ=%.3f: mean z-change = %.6f", sigma_val, z_change.mean().item()
            )

    evaluation = {
        "manifold": config.manifold,
        "val_distribution": config.val_distribution,
        "n_train": int(points_train.shape[0]),
        "n_val_cells": int(points_val.shape[0]),
        "n_epochs_trained": int(n_epochs),
        "best_epoch": best_epoch,
        "val_monitor": monitor_key,
        "best_val_loss": best_val_loss,
        "best_val_loss_aggregate": best_val_aggregate,
        "final_train_loss": final_loss,
        "per_sigma_bin_val_losses": per_bin,
        "z_change_diagnostic": z_change_rows,
    }
    run.save_json("evaluation.json", evaluation)

    checkpoint_path = run.save_checkpoint(
        "score.pt", model_checkpoint(score_net, config.score)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
