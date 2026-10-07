#!/usr/bin/env python
"""Train a projection model on the synthetic scRNA-seq working space."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import pandas as pd
import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict
from torch.utils.data import DataLoader, TensorDataset

from experiments.synthetic_data.common.space import build_train_val, load_geometry
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

K_DEFAULTS = {
    "linear": 2,
    "bifurcation": 3,
    "tree": 3,
}

PAIR_N_SAME_BRANCH = 1000
PAIR_N_CROSS_BRANCH = 1000
PAIR_MIN_PT_GAP = 0.3
PAIR_MIN_PAIRS_PER_PATH = 20

_BIN_LATE = (4,)
_BIN_MID = (2, 3)
_BIN_EARLY = (0,)
_BIN_NAMES = {_BIN_LATE: "late", _BIN_MID: "mid", _BIN_EARLY: "early"}

CROSS_CATEGORIES: dict[
    str, list[tuple[str, tuple[int, ...], str, tuple[int, ...], int]]
] = {
    "linear": [("Path1", _BIN_LATE, "Path2", _BIN_EARLY, 1000)],
    "bifurcation": [
        ("Path1", _BIN_LATE, "Path2", _BIN_MID, 333),
        ("Path1", _BIN_LATE, "Path3", _BIN_MID, 333),
        ("Path2", _BIN_MID, "Path3", _BIN_MID, 334),
    ],
    "tree": [
        ("Path1", _BIN_LATE, "Path4", _BIN_MID, 100),
        ("Path1", _BIN_LATE, "Path5", _BIN_MID, 100),
        ("Path1", _BIN_LATE, "Path6", _BIN_MID, 100),
        ("Path1", _BIN_LATE, "Path7", _BIN_MID, 100),
        ("Path4", _BIN_MID, "Path5", _BIN_MID, 100),
        ("Path6", _BIN_MID, "Path7", _BIN_MID, 100),
        ("Path4", _BIN_MID, "Path6", _BIN_MID, 100),
        ("Path4", _BIN_MID, "Path7", _BIN_MID, 100),
        ("Path5", _BIN_MID, "Path6", _BIN_MID, 100),
        ("Path5", _BIN_MID, "Path7", _BIN_MID, 100),
    ],
}


class ProjectionExperimentConfig(ExperimentConfig):
    """Configuration for the synthetic scRNA-seq projection training script."""

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
        description="Master seed (weights, noise, shuffling, pair pool).",
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

    score_checkpoint: Path = Field(
        description="Path to the score checkpoint (stage 1 output).",
    )
    latent_dim: int | None = Field(
        default=None,
        description="Latent dimension; default: linear 2, bifurcation/tree 3.",
    )
    flow_sigmas: Annotated[tuple[float, ...], NoDecode] = Field(
        default=(0.05, 0.035),
        description="Score-noise levels for the pseudo-projection.",
    )
    tau_min: float = Field(
        default=0.02,
        description="Lower off-manifold perturbation scale.",
    )
    tau_max: float = Field(
        default=0.06,
        description="Upper off-manifold perturbation scale.",
    )

    projector: SingleChartConfig = Field(
        default_factory=SingleChartConfig,
        description="Projector network; x_dim and k are derived.",
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
    def _apply_scenario_k(self) -> Self:
        """Fill the latent dimension from the scenario default when unset."""
        k = (
            self.latent_dim
            if self.latent_dim is not None
            else K_DEFAULTS[self.scenario]
        )
        self.latent_dim = k
        self.projector = self.projector.model_copy(update={"k": k})
        return self


def resolve_score_checkpoint(config: ProjectionExperimentConfig) -> Path:
    """Return the configured score checkpoint path, failing if it is missing."""
    path = Path(config.score_checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Score checkpoint not found at: {path}\n"
            "  (set --score-checkpoint or train one with "
            "experiments/synthetic_data/scripts/train_score.py)"
        )
    return path


def _bin_name(bins: tuple[int, ...]) -> str:
    return _BIN_NAMES.get(bins, "-".join(str(b) for b in bins))


def same_branch_pairs(
    labels: pd.DataFrame,
    val_idx: np.ndarray,
    n_pairs: int = PAIR_N_SAME_BRANCH,
    min_gap: float = PAIR_MIN_PT_GAP,
    min_per_path: int = PAIR_MIN_PAIRS_PER_PATH,
) -> tuple[np.ndarray, list[str], dict]:
    """Draw same-branch val-cell pairs with ``|ΔPseudotime| >= min_gap``.

    Quotas are stratified across paths by val population with a per-path
    floor; paths that cannot fill their quota give up their share to paths
    with spare capacity. Returns val-relative row pairs, per-pair category
    labels and the quotas/caps info dict.
    """
    path_of = labels["Path"].values
    pt = labels["Pseudotime"].values
    paths = sorted(pd.unique(path_of[val_idx]))
    groups = {p: np.sort(val_idx[path_of[val_idx] == p]) for p in paths}
    val_pos = np.full(len(labels), -1, dtype=np.int64)
    val_pos[val_idx] = np.arange(len(val_idx))

    elig: dict[str, np.ndarray] = {}
    for p in paths:
        cells = groups[p]
        if len(cells) < 2:
            elig[p] = np.empty((0, 2), dtype=np.int64)
            continue
        dpt = np.abs(pt[cells, None] - pt[cells][None, :])
        iu = np.triu_indices(len(cells), k=1)
        keep = dpt[iu] >= min_gap
        elig[p] = np.stack([cells[iu[0][keep]], cells[iu[1][keep]]], axis=1)

    n_elig = {p: len(elig[p]) for p in paths}
    pops = np.array([len(groups[p]) for p in paths], dtype=float)
    caps = np.array([n_elig[p] for p in paths])
    floors = np.full(len(paths), min_per_path, dtype=int)

    raw = n_pairs * pops / pops.sum() if pops.sum() > 0 else np.zeros(len(paths))
    quotas = np.maximum(np.floor(raw), floors).astype(int)
    quotas = np.minimum(quotas, caps).astype(int)

    frac_order = np.argsort(-(raw - np.floor(raw)))
    excess = int(quotas.sum()) - n_pairs
    i = 0
    while excess > 0 and i < 4 * len(paths):
        idx = frac_order[i % len(paths)]
        if quotas[idx] > floors[idx]:
            quotas[idx] -= 1
            excess -= 1
        i += 1
    leftover = n_pairs - int(quotas.sum())
    i = 0
    while leftover > 0 and i < 8 * len(paths):
        idx = frac_order[i % len(paths)]
        if quotas[idx] < caps[idx]:
            quotas[idx] += 1
            leftover -= 1
        i += 1

    capped = leftover > 0
    if capped:
        logger.info(
            "  [pair pool] same-branch pool exhausted: only %s/%s pairs "
            "available across the val paths",
            int(quotas.sum()),
            n_pairs,
        )

    pair_idx_list: list[np.ndarray] = []
    category: list[str] = []
    for j, p in enumerate(paths):
        q = int(quotas[j])
        if q <= 0 or n_elig[p] == 0:
            continue
        pick = np.random.choice(n_elig[p], size=min(q, n_elig[p]), replace=False)
        pair_idx_list.append(val_pos[elig[p][pick]])
        category.extend([f"same:{p}"] * len(pick))

    info = {
        "quotas": {p: int(quotas[j]) for j, p in enumerate(paths)},
        "eligible": {p: int(n_elig[p]) for p in paths},
        "drawn": {},
        "capped": bool(capped),
    }
    for j, p in enumerate(paths):
        info["drawn"][p] = (
            int(quotas[j]) if not capped else int(min(quotas[j], n_elig[p]))
        )

    if not pair_idx_list:
        return np.empty((0, 2), dtype=np.int64), [], info

    pair_idx = np.concatenate(pair_idx_list, axis=0).astype(np.int64)
    return pair_idx, category, info


def cross_branch_pairs(
    labels: pd.DataFrame,
    bin_id: np.ndarray,
    val_idx: np.ndarray,
    categories: list[tuple[str, tuple[int, ...], str, tuple[int, ...], int]],
) -> tuple[np.ndarray, list[str], dict]:
    """Draw cross-branch val-cell pairs from the per-scenario category table."""
    path_of = labels["Path"].values
    val_pos = np.full(len(labels), -1, dtype=np.int64)
    val_pos[val_idx] = np.arange(len(val_idx))
    pair_idx_list: list[np.ndarray] = []
    category: list[str] = []
    info: dict = {"drawn": {}, "capped": []}

    for path_a, bins_a, path_b, bins_b, n in categories:
        key = f"cross:{path_a}-{_bin_name(bins_a)}|{path_b}-{_bin_name(bins_b)}"
        a = val_idx[(path_of[val_idx] == path_a) & np.isin(bin_id[val_idx], bins_a)]
        b = val_idx[(path_of[val_idx] == path_b) & np.isin(bin_id[val_idx], bins_b)]
        n_available = len(a) * len(b)
        n_drawn = min(int(n), n_available)
        if n_drawn < int(n):
            logger.info(
                "  [pair pool] category %s: only %s/%s pairs available",
                key,
                n_drawn,
                n,
            )
            info["capped"].append(key)
        if n_drawn > 0:
            pairs = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
            pick = np.random.choice(len(pairs), size=n_drawn, replace=False)
            pair_idx_list.append(val_pos[pairs[pick]])
            category.extend([key] * n_drawn)
        info["drawn"][key] = n_drawn

    if not pair_idx_list:
        return np.empty((0, 2), dtype=np.int64), [], info

    pair_idx = np.concatenate(pair_idx_list, axis=0).astype(np.int64)
    return pair_idx, category, info


def build_pair_pool(
    labels: pd.DataFrame,
    bin_id: np.ndarray,
    val_idx: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build the geodesics evaluation pair pool of same- and cross-branch pairs.

    Returns val-relative pair rows, the corresponding global cell indices, the
    category labels and the per-category quota info.
    """
    same_idx, same_cat, same_info = same_branch_pairs(labels, val_idx)
    cross_idx, cross_cat, cross_info = cross_branch_pairs(
        labels, bin_id, val_idx, CROSS_CATEGORIES[scenario]
    )
    pair_idx = np.concatenate([same_idx, cross_idx], axis=0).astype(np.int64)
    category = np.asarray(same_cat + cross_cat)
    cells = val_idx[pair_idx]
    info = {
        "n_same_branch": int(len(same_idx)),
        "n_cross_branch": int(len(cross_idx)),
        "n_total": int(len(pair_idx)),
        "n_target_total": PAIR_N_SAME_BRANCH + PAIR_N_CROSS_BRANCH,
        "same_branch": same_info,
        "cross_branch": cross_info,
    }
    return pair_idx, cells, category, info


@torch.no_grad()
def fixed_point_errors(
    projector: torch.nn.Module, X_pca: np.ndarray, val_idx: np.ndarray, device
) -> dict:
    """Return the clean-val fixed-point and idempotence MSEs."""
    projector.eval()
    x = torch.from_numpy(np.asarray(X_pca[val_idx], dtype=np.float32)).to(device)
    p = projector(x)
    fixed = torch.linalg.vector_norm(p - x, dim=1).pow(2)
    idem = torch.linalg.vector_norm(projector(p) - p, dim=1).pow(2)
    return {
        "fixed_point_mse": float(fixed.mean()),
        "idempotence_mse": float(idem.mean()),
    }


@torch.no_grad()
def residual_distribution(
    projector: torch.nn.Module,
    X_pca: np.ndarray,
    labels: pd.DataFrame,
    val_idx: np.ndarray,
    device,
) -> dict:
    """Return overall and per-path median residual norms ``||x - P(x)||``."""
    projector.eval()
    x = torch.from_numpy(np.asarray(X_pca[val_idx], dtype=np.float32)).to(device)
    p = projector(x)
    res = torch.linalg.vector_norm(p - x, dim=1).cpu().numpy()
    path_of = labels["Path"].values[val_idx]
    per_path = {}
    for path in sorted(pd.unique(path_of)):
        mask = path_of == path
        per_path[path] = {
            "median_residual": float(np.median(res[mask])),
            "n_cells": int(mask.sum()),
        }
    return {"median_residual": float(np.median(res)), "per_path": per_path}


def main() -> None:
    config = ProjectionExperimentConfig()

    set_seed(config.seed)

    try:
        geom = load_geometry(config.scenario, config.data_root)
        score_ckpt_path = resolve_score_checkpoint(config)
        noise = config.trainer.noise.model_copy(
            update={"tau_min": config.tau_min, "tau_max": config.tau_max}
        )
        config = config.model_copy(
            update={
                "projector": config.projector.model_copy(
                    update={"x_dim": geom["X_pca"].shape[1]}
                ),
                "trainer": config.trainer.model_copy(update={"noise": noise}),
            }
        )
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _main_impl(config, run, score_ckpt_path, geom)


def _main_impl(
    config: ProjectionExperimentConfig,
    run: RunWriter,
    score_ckpt_path: Path,
    geom: dict,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    X_pca = geom["X_pca"]
    labels = geom["labels"]
    bin_id = geom["bin_id"]
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
    projector = SingleChartProjector(config.projector).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
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
    logger.info(
        "  Trained %d/%d epochs; best val_total %.6f at epoch %d",
        n_epochs_trained,
        config.trainer.epochs,
        best_val_total,
        best_epoch,
    )

    val_metrics = trainer.evaluate(val_loader)
    proj_mse = float(val_metrics["proj"])
    fixed_mse = float(val_metrics.get("fix", float("nan")))
    idem_mse = float(val_metrics.get("idem", float("nan")))
    logger.info(
        "  Val (fresh): proj %s  fix %s  idem %s",
        f"{proj_mse:.6e}",
        f"{fixed_mse:.6e}",
        f"{idem_mse:.6e}",
    )

    logger.info("Running diagnostics...")
    fp = fixed_point_errors(projector, X_pca, val_idx, device)
    logger.info("  Fixed-point MSE: %.3e", fp["fixed_point_mse"])

    logger.info("Building geodesics pair pool...")
    pair_idx, cells, categories, pool_info = build_pair_pool(
        labels, bin_id, val_idx, config.scenario
    )
    logger.info(
        "  Pair pool: %s pairs (%s same-branch, %s cross-branch)",
        pool_info["n_total"],
        pool_info["n_same_branch"],
        pool_info["n_cross_branch"],
    )

    residuals = residual_distribution(projector, X_pca, labels, val_idx, device)
    logger.info("  Median residual ||x - P(x)||: %.4f", residuals["median_residual"])

    evaluation = {
        "scenario": config.scenario,
        "variant": config.variant,
        "n_train_cells": int(len(train_idx)),
        "n_val_cells": int(len(val_idx)),
        "n_held_out_cells": int(len(held_out)),
        "n_epochs_trained": int(n_epochs_trained),
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        "final_proj_mse": proj_mse,
        "final_fix_mse": fixed_mse,
        "final_idem_mse": idem_mse,
        "final_total": float(val_metrics["total"]),
        "losses": config.loss.model_dump(),
        "fixed_point": fp,
        "residuals": residuals,
        "pair_pool": pool_info,
        "pseudo": {
            "flow_sigmas": list(config.flow_sigmas),
            "tau_range": [config.trainer.noise.tau_min, config.trainer.noise.tau_max],
        },
        "score_checkpoint": str(score_ckpt_path),
    }
    run.save_json("evaluation.json", evaluation)

    np.savez(
        run.out_dir / "pairs.npz",
        pair_idx=pair_idx,
        cells=cells,
        category=categories,
    )

    checkpoint_path = run.save_checkpoint(
        "projector.pt", model_checkpoint(projector, config.projector)
    )
    logger.info("  Checkpoint saved to: %s", checkpoint_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
