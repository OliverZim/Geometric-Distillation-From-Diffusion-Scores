"""Projection distillation training loop."""

from __future__ import annotations

import copy
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from pydantic import Field
from torch.utils.data import DataLoader

from ..config import ComponentConfig
from ..sampling.noise import OffManifoldNoiseConfig, sample_off_manifold
from ..score.network import ScoreNetwork
from .losses import ProjectionLossConfig, compute_losses
from .pseudo import FlowPseudoProjector


class ProjectionTrainerConfig(ComponentConfig):
    """Configuration for :class:`ProjectionTrainer`."""

    lr: float = Field(
        default=1e-3,
        description="Learning rate for AdamW.",
    )
    weight_decay: float = Field(
        default=0.0,
        description="Weight decay for AdamW.",
    )
    epochs: int = Field(
        default=200,
        description="Number of training epochs.",
    )
    grad_clip: Optional[float] = Field(
        default=1.0,
        description="Maximum gradient norm (None disables clipping).",
    )
    lr_schedule: Literal["flat", "cosine"] = Field(
        default="flat",
        description=("'flat' = constant LR; 'cosine' = anneal to eta_min over epochs."),
    )
    eta_min: float = Field(
        default=1e-5,
        description="Minimum LR of the cosine schedule.",
    )
    noise: OffManifoldNoiseConfig = Field(
        default_factory=OffManifoldNoiseConfig,
        description="Off-manifold perturbation distribution.",
    )
    patience: int = Field(
        default=50,
        ge=0,
        description=(
            "Epochs without significant improvement before early stopping; 0 disables."
        ),
    )
    min_delta: float = Field(
        default=1e-4,
        ge=0.0,
        lt=1.0,
        description=("Relative improvement threshold to reset the patience counter."),
    )
    val_every: int = Field(
        default=1,
        ge=1,
        description="Run validation every N epochs.",
    )


class ProjectionTrainer:
    """Projection distillation trainer; the score network must be frozen."""

    _config: ProjectionTrainerConfig
    _projector: nn.Module
    _score_network: ScoreNetwork
    _pseudo_projector: FlowPseudoProjector
    _loss_config: ProjectionLossConfig

    def __init__(
        self,
        config: ProjectionTrainerConfig,
        projector: nn.Module,
        score_network: ScoreNetwork,
        pseudo_projector: FlowPseudoProjector,
        loss_config: ProjectionLossConfig,
    ):
        self._config = config
        self._projector = projector
        self._score_network = score_network
        self._pseudo_projector = pseudo_projector
        self._loss_config = loss_config

    @property
    def config(self) -> ProjectionTrainerConfig:
        return self._config

    @property
    def projector(self) -> nn.Module:
        return self._projector

    def _loss_keys(self) -> List[str]:
        loss_keys: List[str] = ["proj"]
        if self._loss_config.lambda_fix > 0:
            loss_keys.append("fix")
        if self._loss_config.lambda_idem > 0:
            loss_keys.append("idem")
        if self._loss_config.lambda_fiber > 0:
            loss_keys.append("fiber")
        if self._loss_config.lambda_overlap > 0:
            loss_keys.append("overlap")
        if self._loss_config.lambda_orth > 0:
            loss_keys.append("orth")
        if self._loss_config.lambda_iso > 0:
            loss_keys.append("iso")
        return loss_keys

    @staticmethod
    def _unwrap_batch(batch) -> torch.Tensor:
        """Unwrap a ``(data,)`` batch tuple from a DataLoader into ``x``."""
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        loss_config: Optional[ProjectionLossConfig] = None,
    ) -> Dict[str, float]:
        """Evaluate projection losses on a validation loader."""
        was_training = self._projector.training
        self._projector.eval()
        device = next(self._projector.parameters()).device
        eval_config = loss_config or self._loss_config
        loss_keys = self._loss_keys()
        sums: Dict[str, float] = {key: 0.0 for key in loss_keys}
        sums["total"] = 0.0
        n_total = 0

        for batch in loader:
            x_on = self._unwrap_batch(batch).to(device)
            batch_size = int(x_on.shape[0])
            x_off = sample_off_manifold(x_on, self._config.noise)
            pseudo_target = self._pseudo_projector.project(x_off)
            losses = compute_losses(
                x_off,
                pseudo_target,
                x_on,
                self._projector,
                eval_config,
            )
            for key in loss_keys + ["total"]:
                if key in losses:
                    sums[key] += float(losses[key].item()) * batch_size
            n_total += batch_size

        denominator = max(1, n_total)
        metrics = {key: value / denominator for key, value in sums.items()}
        if was_training:
            self._projector.train()
        return metrics

    def fit(
        self,
        loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        x_on_proportion: float = 0.0,
    ) -> Dict[str, List[float]]:
        """Run the projection distillation training loop.

        Early stopping monitors the validation total when *val_loader* is
        given, otherwise the training projection loss; the best-monitored
        weights are restored before returning.

        Returns:
            Per-epoch histories keyed by loss name (plus ``"val_*"`` keys
            when a validation loader is given).
        """
        if not (0.0 <= x_on_proportion < 1.0):
            raise ValueError(
                f"x_on_proportion must be in [0, 1), got {x_on_proportion}"
            )

        self._projector.train()
        device = next(self._projector.parameters()).device

        opt = torch.optim.AdamW(
            self._projector.parameters(),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )

        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if self._config.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=int(self._config.epochs),
                eta_min=float(self._config.eta_min),
            )

        loss_keys = self._loss_keys()

        history: Dict[str, List[float]] = {k: [] for k in loss_keys}
        if val_loader is not None:
            history.update({f"val_{k}": [] for k in loss_keys})
            history["val_total"] = []

        best_metric = float("inf")
        best_state_dict: dict | None = None
        best_significant = float("inf")
        epochs_without_improvement = 0
        patience = int(self._config.patience)
        min_delta = float(self._config.min_delta)
        val_every = int(self._config.val_every)
        epochs = int(self._config.epochs)
        last_monitor = float("inf")

        for epoch in range(epochs):
            epoch_sums: Dict[str, float] = {k: 0.0 for k in loss_keys}
            n_total = 0

            for batch in loader:
                x_on = self._unwrap_batch(batch).to(device)
                batch_size = int(x_on.shape[0])

                if x_on_proportion > 0.0 and batch_size > 1:
                    n_on = int(x_on_proportion * batch_size)
                    n_on = max(1, min(n_on, batch_size - 1))
                    x_on_part = x_on[:n_on]
                    x_off_part = x_on[n_on:]
                else:
                    x_on_part = x_on
                    x_off_part = x_on

                x_off = sample_off_manifold(x_off_part, self._config.noise)

                pseudo_target = self._pseudo_projector.project(x_off)

                losses = compute_losses(
                    x_off,
                    pseudo_target,
                    x_on_part,
                    self._projector,
                    self._loss_config,
                )

                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                if self._config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self._projector.parameters(),
                        float(self._config.grad_clip),
                    )
                opt.step()

                for k in loss_keys:
                    if k in losses:
                        epoch_sums[k] += float(losses[k].item()) * batch_size
                n_total += batch_size

            if scheduler is not None:
                scheduler.step()

            for k in loss_keys:
                history[k].append(epoch_sums[k] / max(1, n_total))

            if val_loader is not None:
                if epoch % val_every == 0 or epoch == epochs - 1:
                    val_metrics = self.evaluate(
                        val_loader,
                        loss_config=self._loss_config,
                    )
                    for key in loss_keys:
                        history[f"val_{key}"].append(val_metrics.get(key, 0.0))
                    history["val_total"].append(val_metrics["total"])
                    last_monitor = val_metrics["total"]
                    if "iso" in val_metrics and self._loss_config.lambda_iso > 0:
                        last_monitor = (
                            last_monitor
                            - self._loss_config.lambda_iso * val_metrics["iso"]
                        )
                else:
                    for key in loss_keys:
                        history[f"val_{key}"].append(history[f"val_{key}"][-1])
                    history["val_total"].append(history["val_total"][-1])
                monitor = last_monitor
            else:
                monitor = epoch_sums["proj"] / max(1, n_total)

            if monitor < best_metric:
                best_metric = monitor
                best_state_dict = copy.deepcopy(self._projector.state_dict())

            if monitor < best_significant * (1.0 - min_delta):
                best_significant = monitor
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                break

        if best_state_dict is not None:
            self._projector.load_state_dict(best_state_dict)

        return history

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the learned manifold in inference mode."""
        self._projector.eval()
        device = next(self._projector.parameters()).device
        x = x.to(device)
        return self._projector(x)
