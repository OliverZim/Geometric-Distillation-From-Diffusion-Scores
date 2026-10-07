"""Common utilities for toy-manifold experiments."""

from .evaluation import (
    evaluate_manifold_distance,
    evaluate_uniformity,
    manifold_distance,
    summarize_manifold_distances,
)
from .kde import evaluate_kde, fit_kde
from .manifolds import (
    MANIFOLDS,
    Hemisphere,
    Manifold,
    ManifoldName,
    Paraboloid,
    Saddle,
)
from .sampling import sample_imbalanced, sample_uniform

__all__ = [
    "MANIFOLDS",
    "Manifold",
    "ManifoldName",
    "Hemisphere",
    "Saddle",
    "Paraboloid",
    "fit_kde",
    "evaluate_kde",
    "sample_imbalanced",
    "sample_uniform",
    "evaluate_uniformity",
    "evaluate_manifold_distance",
    "manifold_distance",
    "summarize_manifold_distances",
]
