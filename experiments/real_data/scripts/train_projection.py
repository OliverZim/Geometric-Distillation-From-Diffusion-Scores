#!/usr/bin/env python
"""Train a projection model on a real scRNA-seq working space (EB / Cite / Multi)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict
from torch.utils.data import DataLoader, TensorDataset

from experiments.real_data.common.space import (
    build_train_val,
    load_calibration,
    load_geometry,
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
    SoftAtlasConfig,
    projector_from_config,
)
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.score import score_from_checkpoint
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)


class ProjectionExperimentConfig(ExperimentConfig):
    """Configuration for the real-data projection distillation script."""

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
        description="Master seed (weights, noise, shuffling).",
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

    score_checkpoint: Path = Field(
        description="Path to the score checkpoint (stage 1 output).",
    )

    flow_sigmas: Annotated[tuple[float, ...], NoDecode] | None = Field(
        default=None,
        description=(
            "Score-noise levels for the pseudo-projection; default: the "
            "scenario's calibration.json derived value."
        ),
    )
    tau_min: float | None = Field(
        default=None,
        description=(
            "Lower off-manifold perturbation scale; default: the scenario's "
            "calibration.json derived tau range."
        ),
    )
    tau_max: float | None = Field(
        default=None,
        description=(
            "Upper off-manifold perturbation scale; default: the scenario's "
            "calibration.json derived tau range."
        ),
    )
    tau_as_norm: bool = Field(
        default=False,
        description="Divide the tau band by sqrt(x_dim) to read it as a norm.",
    )

    n_charts: int = Field(
        default=1,
        ge=1,
        description="1 selects the single-chart projector, >1 the soft atlas.",
    )
    projector: SingleChartConfig | None = Field(
        default=None,
        description=(
            "Single-chart projector; x_dim is derived from the working space."
        ),
    )
    atlas: SoftAtlasConfig | None = Field(
        default=None,
        description=("Soft-atlas projector; x_dim is derived from the working space."),
    )
    loss: ProjectionLossConfig = Field(
        default_factory=lambda: ProjectionLossConfig(
            lambda_fix=10.0,
            lambda_idem=1.0,
            lambda_orth=1.0,
        ),
        description="Loss weights.",
    )
    trainer: ProjectionTrainerConfig = Field(
        default_factory=lambda: ProjectionTrainerConfig(
            epochs=1000,
            patience=300,
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
    def _resolve_projector(self) -> Self:
        """Select the projector variant and zero the atlas orthogonality loss."""
        if self.n_charts > 1:
            if self.atlas is None:
                base = self.projector or SingleChartConfig()
                self.atlas = SoftAtlasConfig(
                    x_dim=base.x_dim,
                    k=base.k,
                    n_charts=self.n_charts,
                    hidden_dim=base.hidden_dim,
                    trunk_layers=base.enc_layers,
                    head_layers=max(1, base.enc_layers - 1),
                    dec_layers=base.dec_layers,
                )
            self.projector = None
        else:
            if self.projector is None:
                self.projector = SingleChartConfig(k=5, hidden_dim=256)
            self.atlas = None
        if self.n_charts > 1 and self.loss.lambda_orth:
            self.loss = self.loss.model_copy(update={"lambda_orth": 0.0})
        return self

    @property
    def projector_config(self) -> SingleChartConfig | SoftAtlasConfig:
        """Return the projector configuration selected by ``n_charts``."""
        if self.n_charts > 1:
            assert self.atlas is not None
            return self.atlas
        assert self.projector is not None
        return self.projector

    def with_projector_x_dim(self, x_dim: int) -> Self:
        active = self.projector_config.model_copy(update={"x_dim": x_dim})
        key = "atlas" if self.n_charts > 1 else "projector"
        return self.model_copy(update={key: active})


def resolve_score_checkpoint(config: ProjectionExperimentConfig) -> Path:
    """Return the configured score checkpoint path, failing if it is missing."""
    path = Path(config.score_checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Score checkpoint not found at: {path}\n"
            "  (set --score-checkpoint or train one with "
            "experiments/real_data/scripts/train_score.py)"
        )
    return path


def resolve_flow_sigmas(
    config: ProjectionExperimentConfig, calibration: dict, calib_path: Path
) -> tuple[float, ...]:
    """Return the configured flow sigmas, falling back to the calibration."""
    value = (
        config.flow_sigmas
        if config.flow_sigmas is not None
        else calibration["derived"].get("flow_sigmas")
    )
    sigmas = tuple(value) if value is not None else ()
    if not sigmas or not all(np.isfinite(s) for s in sigmas):
        raise ValueError(
            f"--flow-sigmas is not set and the derived 'flow_sigmas' in "
            f"{calib_path} (scenario {config.scenario}) is missing/NaN — run "
            f"prepare_eb_geometry.py / calibrate_scales.py first, or pass "
            f"--flow-sigmas explicitly"
        )
    return tuple(float(s) for s in sigmas)


def resolve_tau_range(
    config: ProjectionExperimentConfig, calibration: dict, calib_path: Path
) -> tuple[float, float]:
    """Return the configured tau band, falling back per endpoint to the calibration."""
    calib_range = calibration["derived"].get("tau_range")
    if calib_range is None or len(calib_range) != 2:
        raise ValueError(
            f"--tau-min/--tau-max are not set and the derived 'tau_range' in "
            f"{calib_path} (scenario {config.scenario}) is missing — run "
            f"prepare_eb_geometry.py / calibrate_scales.py first, or pass "
            f"--tau-min/--tau-max explicitly"
        )
    tau_min = config.tau_min if config.tau_min is not None else float(calib_range[0])
    tau_max = config.tau_max if config.tau_max is not None else float(calib_range[1])
    for name, value in (("tau-min", tau_min), ("tau-max", tau_max)):
        if not np.isfinite(value):
            raise ValueError(
                f"--{name} is not set and the derived 'tau_range' in "
                f"{calib_path} (scenario {config.scenario}) is missing/NaN — run "
                f"prepare_eb_geometry.py / calibrate_scales.py first, or pass "
                f"--{name} explicitly"
            )
    return float(tau_min), float(tau_max)


def tau_as_norm(tau_min: float, tau_max: float, n_pcs: int) -> tuple[float, float]:
    """Divide a distance-derived tau band by sqrt(D) to a per-coordinate scale.

    ``sample_off_manifold`` treats tau as a per-coordinate scale, so a band
    derived from distances is inflated by sqrt(D).
    """
    scale = float(np.sqrt(n_pcs))
    return tau_min / scale, tau_max / scale


def main() -> None:
    config = ProjectionExperimentConfig()

    set_seed(config.seed)

    try:
        geom = load_geometry(config.scenario, config.data_root)
        calibration = load_calibration(config.scenario, config.data_root)
        calib_path = (
            config.data_root / config.scenario / "geometry" / "calibration.json"
        )
        flow_sigmas = resolve_flow_sigmas(config, calibration, calib_path)
        tau_min, tau_max = resolve_tau_range(config, calibration, calib_path)
        if config.tau_as_norm:
            tau_min, tau_max = tau_as_norm(tau_min, tau_max, geom["X_pca"].shape[1])
        score_ckpt_path = resolve_score_checkpoint(config)
        noise = config.trainer.noise.model_copy(
            update={"tau_min": tau_min, "tau_max": tau_max}
        )
        config = config.model_copy(
            update={
                "flow_sigmas": flow_sigmas,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "trainer": config.trainer.model_copy(update={"noise": noise}),
            }
        ).with_projector_x_dim(geom["X_pca"].shape[1])
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run, geom, score_ckpt_path)


def _main_impl(
    config: ProjectionExperimentConfig,
    run: RunWriter,
    geom: dict,
    score_ckpt_path: Path,
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

    logger.info("Loading frozen score model from: %s", score_ckpt_path)
    ckpt = torch.load(score_ckpt_path, map_location=device, weights_only=False)
    score_net = score_from_checkpoint(ckpt, device=device)
    if score_net.config.x_dim != X_pca.shape[1]:
        raise ValueError(
            f"score checkpoint x_dim {score_net.config.x_dim} does not match "
            f"the working space ({X_pca.shape[1]} PCs)"
        )
    score_net.requires_grad_(False)
    logger.info("  Loaded score network: %s", score_net.config)

    logger.info("Building pseudo-projector...")
    pseudo_config = FlowConfig(sigmas=config.flow_sigmas)
    pseudo_projector = FlowPseudoProjector(pseudo_config, score_net)
    logger.info(
        "  Pseudo-projector: %s %s",
        type(pseudo_projector).__name__,
        pseudo_config,
    )

    logger.info("Building projector network...")
    projector = projector_from_config(config.projector_config).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    logger.info("  Architecture: %s", type(config.projector_config).__name__)
    logger.info("  Parameters: %s", f"{n_params:,}")

    logger.info("Building trainer...")
    trainer = ProjectionTrainer(
        config.trainer,
        projector,
        score_net,
        pseudo_projector,
        config.loss,
    )

    logger.info("Training...")
    history = trainer.fit(train_loader, val_loader=val_loader)

    n_epochs_trained = len(history["proj"]) if history else 0
    for epoch_idx in range(n_epochs_trained):
        run.log_metrics(epoch_idx, {key: v[epoch_idx] for key, v in history.items()})

    for key, values in history.items():
        final_val = values[-1] if values else float("nan")
        logger.info("  Final %s: %.6f", key, final_val)

    best_epoch = int(np.argmin(history["val_total"]))
    best_val_total = float(history["val_total"][best_epoch])
    best_at_best_epoch = {
        f"{key}_best": float(history[key][best_epoch])
        for key in ("val_proj", "val_fix", "val_idem", "val_fiber", "val_orth")
        if key in history
    }
    logger.info(
        "  Trained %d/%d epochs; best val_total %.6f at epoch %d",
        n_epochs_trained,
        config.trainer.epochs,
        best_val_total,
        best_epoch,
    )

    evaluation = {
        "scenario": config.scenario,
        "variant": config.variant,
        "n_train_cells": int(len(train_idx)),
        "n_val_cells": int(len(val_idx)),
        "n_held_out_cells": int(len(held_out)),
        "n_epochs_trained": int(n_epochs_trained),
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        **best_at_best_epoch,
        "losses": config.loss.model_dump(),
        "pseudo": {
            "flow_sigmas": list(config.flow_sigmas),
            "tau_range": [config.tau_min, config.tau_max],
        },
        "score_checkpoint": str(score_ckpt_path),
    }
    run.save_json("evaluation.json", evaluation)

    checkpoint_path = run.save_checkpoint(
        "projector.pt", model_checkpoint(projector, config.projector_config)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
