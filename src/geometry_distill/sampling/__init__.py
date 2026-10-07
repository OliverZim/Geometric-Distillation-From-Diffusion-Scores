"""Sampling utilities: off-manifold noise, Langevin dynamics, volume elements."""

from .langevin import LangevinConfig, sample_langevin
from .noise import OffManifoldNoiseConfig, sample_off_manifold
from .volume import (
    encoder_jacobian,
    log_volume_element,
    log_volume_from_jacobian,
    projection_residual_squared,
)

__all__ = [
    "LangevinConfig",
    "OffManifoldNoiseConfig",
    "encoder_jacobian",
    "log_volume_element",
    "log_volume_from_jacobian",
    "projection_residual_squared",
    "sample_langevin",
    "sample_off_manifold",
]
