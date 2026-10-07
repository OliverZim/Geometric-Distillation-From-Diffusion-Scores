"""Scenario metadata shared by the synthetic notebooks."""

from typing import Any

SCENARIO_META: dict[str, dict[str, Any]] = {
    "linear": {
        "label": "Linear trajectory",
        "level": 1,
        "path_from": (0, 1),
        "group_prob": (0.4, 0.6),
        "seed": 1010,
        "description": "progenitor to a single terminal state",
    },
    "bifurcation": {
        "label": "Single bifurcation",
        "level": 2,
        "path_from": (0, 1, 1),
        "group_prob": (0.2, 0.4, 0.4),
        "seed": 1020,
        "description": "one branch point, two terminal fates",
    },
    "tree": {
        "label": "Multi-bifurcation tree",
        "level": 3,
        "path_from": (0, 1, 1, 2, 2, 3, 3),
        "group_prob": (1 / 7,) * 7,
        "seed": 1030,
        "description": "three branch points, four terminal fates",
    },
}
