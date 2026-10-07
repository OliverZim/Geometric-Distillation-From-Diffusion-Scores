#!/usr/bin/env python
"""Train a projection model by distilling from a frozen score network."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Self

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict
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
from geometry_distill.projection import (
    FlowConfig,
    FlowPseudoProjector,
    ProjectionLossConfig,
    ProjectionTrainer,
    ProjectionTrainerConfig,
    SingleChartConfig,
    SingleChartProjector,
)
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.score import score_from_checkpoint
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)


class ProjectionExperimentConfig(ExperimentConfig):
    """Configuration for the toy-manifold projection distillation script."""

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

    score_checkpoint: Path = Field(
        description="Path to the score checkpoint (stage 1 output).",
    )

    flow_sigmas: Annotated[tuple[float, ...], NoDecode] = Field(
        default=(0.05,),
        description=(
            "Score-noise levels for the pseudo-projection; a single value "
            "gives the single-step pseudo-projection."
        ),
    )

    projector: SingleChartConfig = Field(
        default_factory=SingleChartConfig,
        description="Projector network; x_dim and k are derived from the manifold.",
    )
    loss: ProjectionLossConfig = Field(
        default_factory=lambda: ProjectionLossConfig(
            lambda_fix=10.0,
            lambda_idem=1.0,
            lambda_fiber=1.0,
            lambda_overlap=0.0,
            lambda_orth=1.0,
        ),
        description="Loss weights.",
    )
    trainer: ProjectionTrainerConfig = Field(
        default_factory=lambda: ProjectionTrainerConfig(
            epochs=1000,
            lr_schedule="cosine",
        ),
        description="Trainer hyperparameters.",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @field_validator("flow_sigmas", mode="before")
    @classmethod
    def _parse_flow_sigmas(cls, value):
        if isinstance(value, str):
            return tuple(float(part) for part in value.split(","))
        return value

    @model_validator(mode="after")
    def _apply_manifold_dims(self) -> Self:
        """Force the projector's x_dim and k to the manifold's values."""
        manifold = MANIFOLDS[self.manifold]
        self.projector = self.projector.model_copy(
            update={"x_dim": manifold.x_dim, "k": manifold.k}
        )
        return self


def resolve_score_checkpoint(config: ProjectionExperimentConfig) -> Path:
    """Return the configured score checkpoint path, failing if it is missing."""
    path = Path(config.score_checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Score checkpoint not found at: {path}\n"
            "  (set --score-checkpoint or train one with "
            "experiments/toy_manifolds/scripts/train_score.py)"
        )
    return path


def main() -> None:
    config = ProjectionExperimentConfig()

    set_seed(config.seed)

    try:
        score_ckpt_path = resolve_score_checkpoint(config)
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run, score_ckpt_path)


def _main_impl(
    config: ProjectionExperimentConfig,
    run: RunWriter,
    score_ckpt_path: Path,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    manifold = MANIFOLDS[config.manifold]
    logger.info("Manifold: %s", manifold.name)

    logger.info("Loading frozen score model from: %s", score_ckpt_path)
    ckpt = torch.load(score_ckpt_path, map_location=device, weights_only=False)
    score_net = score_from_checkpoint(ckpt, device=device)
    score_net.requires_grad_(False)
    logger.info("  Loaded score network: %s", score_net.config)

    logger.info("Generating training data...")
    points_train, _ = sample_imbalanced(manifold, n_points=config.n_train)
    train_dataset = TensorDataset(points_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    logger.info("  Training points: %s", points_train.shape)

    logger.info("Generating validation data...")
    points_val, _ = sample_uniform(manifold, n_points=config.n_val)
    val_dataset = TensorDataset(points_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    logger.info("  Validation points: %s", points_val.shape)

    logger.info("Building pseudo-projector...")
    pseudo_config = FlowConfig(sigmas=config.flow_sigmas)
    pseudo_projector = FlowPseudoProjector(pseudo_config, score_net)
    logger.info(
        "  Pseudo-projector: %s %s",
        type(pseudo_projector).__name__,
        pseudo_config,
    )

    logger.info("Building projector network...")
    projector = SingleChartProjector(config.projector).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    loss_config = config.loss

    logger.info("Building trainer...")
    trainer = ProjectionTrainer(
        config.trainer,
        projector,
        score_net,
        pseudo_projector,
        loss_config,
    )

    logger.info("Training...")
    history = trainer.fit(train_loader, val_loader=val_loader)

    n_epochs_trained = len(next(iter(history.values()))) if history else 0
    for epoch_idx in range(n_epochs_trained):
        step_metrics = {}
        for key, values in history.items():
            step_metrics[key] = values[epoch_idx]
        run.log_metrics(epoch_idx, step_metrics)

    for key, values in history.items():
        final_val = values[-1] if values else float("nan")
        logger.info("  Final %s: %.6f", key, final_val)

    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(val_loader)
    proj_mse = float(val_metrics["proj"])
    fixed_mse = float(val_metrics.get("fix", float("nan")))
    idem_mse = float(val_metrics.get("idem", float("nan")))

    eval_metrics = {
        "proj_mse": proj_mse,
        "fixed_point_mse": fixed_mse,
        "idempotence_mse": idem_mse,
        "total_validation_loss": float(val_metrics["total"]),
    }
    logger.info("  Projection MSE:  %.6e", proj_mse)
    logger.info("  Fixed-point MSE: %.6e", fixed_mse)
    logger.info("  Idempotence MSE: %.6e", idem_mse)

    eval_step = n_epochs_trained
    run.log_metrics(eval_step, eval_metrics)

    checkpoint_path = run.save_checkpoint(
        "projector.pt", model_checkpoint(projector, config.projector)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)

    evaluation_path = run.save_json(
        "evaluation.json",
        {
            "manifold": config.manifold,
            "proj_mse": proj_mse,
            "fixed_point_mse": fixed_mse,
            "idempotence_mse": idem_mse,
            "total_validation_loss": float(val_metrics["total"]),
            "n_epochs_trained": n_epochs_trained,
            "score_checkpoint": str(config.score_checkpoint),
        },
    )
    logger.info("  Evaluation saved to: %s", evaluation_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
