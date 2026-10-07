"""Log-space sigma schedule for denoising score matching."""

from __future__ import annotations

from typing import Self

import torch
from pydantic import Field, model_validator

from ..config import ComponentConfig


class LogUniformSigmaScheduleConfig(ComponentConfig):
    """Configuration for :class:`LogUniformSigmaSchedule`."""

    sigma_min: float = Field(
        default=0.01,
        gt=0.0,
        description="Smallest training noise level.",
    )
    sigma_max: float = Field(
        default=0.5,
        description="Largest training noise level.",
    )

    @model_validator(mode="after")
    def _check_sigma_range(self) -> Self:
        if self.sigma_min >= self.sigma_max:
            raise ValueError(
                f"sigma_min ({self.sigma_min}) must be smaller than "
                f"sigma_max ({self.sigma_max})"
            )
        return self


class LogUniformSigmaSchedule:
    """Sigma values drawn log-uniformly over ``[sigma_min, sigma_max]``."""

    _config: LogUniformSigmaScheduleConfig

    def __init__(self, config: LogUniformSigmaScheduleConfig):
        self._config = config

    @property
    def config(self) -> LogUniformSigmaScheduleConfig:
        return self._config

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        device = torch.device(device)
        u = torch.rand(batch_size, device=device)
        log_sigma = u * (
            torch.log(torch.tensor(self._config.sigma_max))
            - torch.log(torch.tensor(self._config.sigma_min))
        )
        log_sigma = log_sigma + torch.log(torch.tensor(self._config.sigma_min))
        return torch.exp(log_sigma)
