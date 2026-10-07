"""Standard checkpoint payloads for trained models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .config import ComponentConfig

FORMAT_VERSION = 1


def model_checkpoint(model: nn.Module, config: ComponentConfig) -> dict[str, Any]:
    """Return the standard checkpoint payload for a trained model."""
    return {
        "format_version": FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
