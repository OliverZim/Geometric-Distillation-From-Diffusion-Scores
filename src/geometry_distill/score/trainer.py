"""DSM training loop for the score network."""

from __future__ import annotations

import copy
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from pydantic import Field
from torch.utils.data import DataLoader

from ..config import ComponentConfig
from .network import ScoreNetwork
from .schedule import LogUniformSigmaSchedule


class ScoreTrainerConfig(ComponentConfig):
    """Configuration for :class:`ScoreTrainer`."""

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
    patience: int = Field(
        default=300,
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
    n_sigma_bins: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of log-spaced noise levels for per-bin validation losses."
        ),
    )


class ScoreTrainer:
    """DSM trainer tying together the score network and sigma schedule."""

    _config: ScoreTrainerConfig
    _network: ScoreNetwork
    _schedule: LogUniformSigmaSchedule

    def __init__(
        self,
        config: ScoreTrainerConfig,
        network: ScoreNetwork,
        schedule: LogUniformSigmaSchedule,
    ):
        self._config = config
        self._network = network
        self._schedule = schedule

    @property
    def config(self) -> ScoreTrainerConfig:
        return self._config

    @property
    def network(self) -> ScoreNetwork:
        return self._network

    def fit(
        self, loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> dict[str, list[float]]:
        """Run the DSM training loop.

        Returns:
            Per-epoch history (``"train_loss"`` always, plus ``"val_*"``
            keys when a validation loader is given).
        """
        self._network.train()
        device = next(self._network.parameters()).device

        opt = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )

        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if self._config.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(self._config.epochs), eta_min=float(self._config.eta_min)
            )

        history: dict[str, list[float]] = {"train_loss": []}
        sigma_bins: Optional[torch.Tensor] = None
        if val_loader is not None:
            history["val_loss"] = []
            # Build on CPU then move — aten::logspace is not implemented on MPS.
            sigma_bins = torch.logspace(
                math.log10(self._schedule.config.sigma_min),
                math.log10(self._schedule.config.sigma_max),
                int(self._config.n_sigma_bins),
            ).to(device)
            for s in sigma_bins:
                history[f"val_loss_sigma_{float(s):.3f}"] = []
            history["val_bin_mean"] = []

        best_monitor = float("inf")
        best_state_dict: dict | None = None
        best_significant = float("inf")
        epochs_without_improvement = 0
        patience = int(self._config.patience)
        min_delta = float(self._config.min_delta)

        for _epoch in range(int(self._config.epochs)):
            epoch_loss = 0.0
            n = 0

            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                batch_size = int(x.shape[0])

                sigma = self._schedule.sample(batch_size, device=device)
                eps = torch.randn(x.shape, device=device, dtype=x.dtype)

                x_tilde = x + sigma[:, None] * eps
                target = -eps / sigma[:, None]

                pred = self._network(x_tilde, sigma)
                loss = torch.mean((pred - target) ** 2 * sigma[:, None] ** 2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self._config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self._network.parameters(), float(self._config.grad_clip)
                    )
                opt.step()

                epoch_loss += float(loss.item()) * batch_size
                n += batch_size

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(1, n)
            history["train_loss"].append(avg_loss)

            if val_loader is not None:
                self._network.eval()
                val_metrics = self._evaluate_validation_losses(
                    val_loader, sigma_bins, device
                )
                self._network.train()
                for key, value in val_metrics.items():
                    history[key].append(value)
                monitor = sum(
                    float(s) ** 2 * val_metrics[f"val_loss_sigma_{float(s):.3f}"]
                    for s in sigma_bins
                ) / len(sigma_bins)
                history["val_bin_mean"].append(monitor)
            else:
                monitor = avg_loss

            if monitor < best_monitor:
                best_monitor = monitor
                best_state_dict = copy.deepcopy(self._network.state_dict())

            if monitor < best_significant * (1.0 - min_delta):
                best_significant = monitor
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                break

        if best_state_dict is not None:
            self._network.load_state_dict(best_state_dict)

        return history

    @torch.no_grad()
    def _batch_dsm_loss(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """DSM loss of a clean batch at given noise levels (fresh noise draws)."""
        eps = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        x_tilde = x + sigma[:, None] * eps
        target = -eps / sigma[:, None]
        pred = self._network(x_tilde, sigma)
        return torch.mean((pred - target) ** 2)

    @torch.no_grad()
    def _evaluate_validation_losses(
        self,
        val_loader: DataLoader,
        sigma_bins: Optional[torch.Tensor],
        device: torch.device,
    ) -> dict[str, float]:
        """Validation DSM loss, aggregated and per fixed σ bin."""
        agg_sum = 0.0
        agg_n = 0
        bin_sums: dict[str, float] = {}
        bin_n: dict[str, int] = {}
        if sigma_bins is not None:
            for s in sigma_bins:
                bin_sums[f"val_loss_sigma_{float(s):.3f}"] = 0.0
                bin_n[f"val_loss_sigma_{float(s):.3f}"] = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            batch_size = int(x.shape[0])

            sigma = self._schedule.sample(batch_size, device=device)
            agg_sum += float(self._batch_dsm_loss(x, sigma).item()) * batch_size
            agg_n += batch_size

            for key, s in zip(bin_sums, sigma_bins if sigma_bins is not None else ()):
                sigma_t = torch.full((batch_size,), float(s), device=device)
                bin_sums[key] += (
                    float(self._batch_dsm_loss(x, sigma_t).item()) * batch_size
                )
                bin_n[key] += batch_size

        metrics = {"val_loss": agg_sum / max(1, agg_n)}
        for key in bin_sums:
            metrics[key] = bin_sums[key] / max(1, bin_n[key])
        return metrics

    @torch.no_grad()
    def score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate the score network in inference mode, ``(batch, x_dim)``."""
        self._network.eval()
        device = next(self._network.parameters()).device
        x = x.to(device)
        sigma = sigma.to(device)
        return self._network(x, sigma)
