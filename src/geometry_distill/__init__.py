"""Geometry Distill: distill manifold geometry from a pretrained score model."""

from geometry_distill.device import resolve_device
from geometry_distill.geodesics import (
    ManifoldPathEvaluator,
    ManifoldPathEvaluatorConfig,
    PathRefinementConfig,
    WarpedPullbackMetricConfig,
    compute_f_plus,
    pullback_velocity,
)
from geometry_distill.projection import (
    FlowConfig,
    FlowPseudoProjector,
    ProjectionLossConfig,
    ProjectionTrainer,
    ProjectionTrainerConfig,
    SingleChartConfig,
    SingleChartProjector,
    SoftAtlasConfig,
    SoftAtlasProjector,
    compute_losses,
)
from geometry_distill.sampling import (
    LangevinConfig,
    OffManifoldNoiseConfig,
    encoder_jacobian,
    log_volume_element,
    log_volume_from_jacobian,
    projection_residual_squared,
    sample_langevin,
    sample_off_manifold,
)
from geometry_distill.score import (
    LogUniformSigmaSchedule,
    LogUniformSigmaScheduleConfig,
    ScoreNetwork,
    ScoreNetworkConfig,
    ScoreTrainer,
    ScoreTrainerConfig,
)
from geometry_distill.seeding import set_seed

__all__ = [
    "LogUniformSigmaSchedule",
    "LogUniformSigmaScheduleConfig",
    "ScoreNetwork",
    "ScoreNetworkConfig",
    "ScoreTrainer",
    "ScoreTrainerConfig",
    "FlowConfig",
    "FlowPseudoProjector",
    "ProjectionLossConfig",
    "ProjectionTrainer",
    "ProjectionTrainerConfig",
    "SingleChartConfig",
    "SingleChartProjector",
    "SoftAtlasConfig",
    "SoftAtlasProjector",
    "compute_losses",
    "ManifoldPathEvaluator",
    "ManifoldPathEvaluatorConfig",
    "PathRefinementConfig",
    "WarpedPullbackMetricConfig",
    "compute_f_plus",
    "pullback_velocity",
    "LangevinConfig",
    "OffManifoldNoiseConfig",
    "encoder_jacobian",
    "log_volume_element",
    "log_volume_from_jacobian",
    "projection_residual_squared",
    "sample_langevin",
    "sample_off_manifold",
    "resolve_device",
    "set_seed",
]
