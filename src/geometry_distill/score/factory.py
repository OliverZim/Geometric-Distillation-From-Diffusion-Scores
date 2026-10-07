"""Build score networks from stored checkpoints."""

from __future__ import annotations

from typing import Any

import torch

from .network import ScoreNetwork, ScoreNetworkConfig


def score_config_from_checkpoint(ckpt_cfg: Any) -> ScoreNetworkConfig:
    """Return a score-network config from a checkpoint's stored config."""
    if isinstance(ckpt_cfg, ScoreNetworkConfig):
        return ckpt_cfg
    if not isinstance(ckpt_cfg, dict):
        raise TypeError(f"unsupported checkpoint config type {type(ckpt_cfg).__name__}")
    valid = ScoreNetworkConfig.model_fields
    return ScoreNetworkConfig(**{k: v for k, v in ckpt_cfg.items() if k in valid})


def score_from_checkpoint(
    ckpt: dict, device: torch.device | str | None = None
) -> ScoreNetwork:
    """Return the score network stored in a training checkpoint, in eval mode."""
    score = ScoreNetwork(score_config_from_checkpoint(ckpt["config"]))
    score.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        score = score.to(device)
    return score.eval()
