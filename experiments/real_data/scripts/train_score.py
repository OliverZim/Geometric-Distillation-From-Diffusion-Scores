#!/usr/bin/env python
"""Train a score model on a real scRNA-seq working space (EB / Cite / Multi)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from torch.utils.data import DataLoader, TensorDataset

from experiments.real_data.common.space import (
    build_train_val,
    load_geometry,
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
    """Configuration for the real-data score training script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["eb", "cite50", "cite100", "multi50", "multi100"] = Field(
        default="eb",
        description="Real dataset to train on.",
    )
    variant: str = Field(
        default="full",
        description=(
            "Key of variants.npz; non-full variants are day-blind holdouts and "
            "validate on their kept train cells."
        ),
    )
    data_root: Path = Field(
        default=Path("experiments/real_data/eb/data"),
        description="Data root containing <scenario>/geometry/.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (weights, noise, shuffling, subsamples).",
    )
    n_val: int = Field(
        default=1500,
        ge=1,
        description="Validation cells (capped by availability).",
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="Training batch size.",
    )

    schedule: LogUniformSigmaScheduleConfig = Field(
        default_factory=lambda: LogUniformSigmaScheduleConfig(
            sigma_min=0.005,
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
    logger.info(
        "Sigma schedule: log-uniform [%s, %s]",
        config.schedule.sigma_min,
        config.schedule.sigma_max,
    )

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

    logger.info("Building score network...")
    score_net = ScoreNetwork(config.score).to(device)
    n_params = sum(p.numel() for p in score_net.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    schedule = LogUniformSigmaSchedule(config.schedule)
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
    }
    run.save_json("evaluation.json", evaluation)

    checkpoint_path = run.save_checkpoint(
        "score.pt", model_checkpoint(score_net, config.score)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
