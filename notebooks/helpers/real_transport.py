"""Aggregated comparison of the real-data transport runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

DATASET_LABELS: Mapping[str, str] = {
    "eb": "EB",
    "cite50": "Cite 50",
    "cite100": "Cite 100",
    "multi50": "Multi 50",
    "multi100": "Multi 100",
}

GAGA_PUBLISHED: Mapping[str, float] = {
    "cite50": 23.29,
    "cite100": 26.72,
    "multi50": 19.68,
    "multi100": 27.04,
}

RUN_PATTERN: str = "*/holdout_*-s*/transport/evaluation.json"

_T975: Mapping[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
}

_RUN_DIR = re.compile(r"holdout_(?P<holdout>.+)-s(?P<seed>\d+)$")


def find_runs(results_root: Path | str) -> list[tuple[str, str, int, Path]]:
    """(dataset, holdout, seed, evaluation path) for every finished transport run."""
    runs: list[tuple[str, str, int, Path]] = []
    for path in sorted(Path(results_root).glob(RUN_PATTERN)):
        match = _RUN_DIR.match(path.parent.parent.name)
        if match:
            runs.append((path.parts[-4], match["holdout"], int(match["seed"]), path))
    return runs


def load_values(path: Path | str) -> dict[str, float]:
    """Sample-size-matched W1 of every arm scored in one run."""
    protocol = json.loads(Path(path).read_text())["gaga_protocol"]
    return {
        arm: float(entry["W1_traj_matched"])
        for arm, entry in protocol.items()
        if isinstance(entry, dict) and "W1_traj_matched" in entry
    }


def collect(results_root: Path | str) -> dict[str, dict[str, list[float]]]:
    """Per-seed day-averaged W1 per arm: dataset -> arm -> one value per seed."""
    by_cell: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for dataset, holdout, seed, path in find_runs(results_root):
        cells = by_cell.setdefault(dataset, {}).setdefault(holdout, {})
        cells[seed] = load_values(path)

    collected: dict[str, dict[str, list[float]]] = {}
    for dataset, holdouts in by_cell.items():
        seeds = sorted({seed for runs in holdouts.values() for seed in runs})
        arms: set[str] = set()
        for runs in holdouts.values():
            for values in runs.values():
                arms.update(values)
        for arm in sorted(arms):
            per_seed = []
            for seed in seeds:
                per_day = [
                    runs[seed][arm]
                    for runs in holdouts.values()
                    if arm in runs.get(seed, {})
                ]
                if per_day:
                    per_seed.append(float(np.mean(per_day)))
            collected.setdefault(dataset, {})[arm] = per_seed
    return collected


def summarize(collected: dict[str, dict[str, list[float]]]) -> dict[str, dict]:
    """Mean, population sd, seed count and 95% CI half-width per arm."""
    summary: dict[str, dict] = {}
    for dataset, arms in collected.items():
        entry: dict = {}
        for arm, values in arms.items():
            entry[arm] = [float(np.mean(values)), float(np.std(values))]
            entry.setdefault("_n", {})[arm] = len(values)
            half = 0.0
            if len(values) > 1:
                half = (
                    _T975.get(len(values) - 1, 1.959964)
                    * float(np.std(values, ddof=1))
                    / np.sqrt(len(values))
                )
            entry.setdefault("_ci95", {})[arm] = float(half)
        summary[dataset] = entry
    return summary
