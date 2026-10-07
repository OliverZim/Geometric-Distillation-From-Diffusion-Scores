"""Build projector instances and configs from stored checkpoints."""

from __future__ import annotations

from typing import Any, Union

import torch.nn as nn

from .atlas import SoftAtlasConfig, SoftAtlasProjector
from .network import SingleChartConfig, SingleChartProjector

ProjectorConfig = Union[SingleChartConfig, SoftAtlasConfig]


def projector_config_from_checkpoint(ckpt_cfg: Any) -> ProjectorConfig:
    """Return a projector config from a checkpoint's stored config."""
    if isinstance(ckpt_cfg, (SingleChartConfig, SoftAtlasConfig)):
        return ckpt_cfg
    if not isinstance(ckpt_cfg, dict):
        raise TypeError(f"unsupported checkpoint config type {type(ckpt_cfg).__name__}")
    cls = SoftAtlasConfig if int(ckpt_cfg.get("n_charts", 1)) > 1 else SingleChartConfig
    valid = cls.model_fields
    return cls(**{k: v for k, v in ckpt_cfg.items() if k in valid})


def projector_from_config(cfg: ProjectorConfig) -> nn.Module:
    """Instantiate the projector matching *cfg* (untrained weights)."""
    if isinstance(cfg, SoftAtlasConfig):
        return SoftAtlasProjector(cfg)
    if isinstance(cfg, SingleChartConfig):
        return SingleChartProjector(cfg)
    raise TypeError(f"unsupported projector config {type(cfg).__name__}")


def projector_from_checkpoint(ckpt: dict, device=None) -> nn.Module:
    """Rebuild and load the projector stored in a training checkpoint dict
    (``config`` + ``model_state_dict``), in eval mode."""
    proj = projector_from_config(projector_config_from_checkpoint(ckpt["config"]))
    proj.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        proj = proj.to(device)
    return proj.eval()
