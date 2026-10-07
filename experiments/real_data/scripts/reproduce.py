#!/usr/bin/env python
"""Dispatch the real-data transport campaign across datasets, holdouts and seeds."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from geometry_distill.config import ExperimentConfig, deep_update

logger = logging.getLogger("reproduce")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "real_data" / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "real_data"
DEFAULT_SEEDS = (42, 43, 44)

Dataset = Literal["eb", "cite50", "cite100", "multi50", "multi100"]
Stage = Literal["score", "projection", "transport"]

DATASETS: tuple[Dataset, ...] = ("eb", "cite50", "cite100", "multi50", "multi100")
STAGES: tuple[Stage, ...] = ("score", "projection", "transport")

DATA_ROOTS: dict[Dataset, Path] = {
    "eb": Path("experiments/real_data/eb/data"),
    "cite50": Path("experiments/real_data/cite_multi/data"),
    "cite100": Path("experiments/real_data/cite_multi/data"),
    "multi50": Path("experiments/real_data/cite_multi/data"),
    "multi100": Path("experiments/real_data/cite_multi/data"),
}
HOLDOUTS: dict[Dataset, tuple[str, ...]] = {
    "eb": ("T2_3B", "T4_5C", "T6_7D"),
    "cite50": ("day3", "day4"),
    "cite100": ("day3", "day4"),
    "multi50": ("day3", "day4"),
    "multi100": ("day3", "day4"),
}
STAGE_SCRIPT: dict[Stage, str] = {
    "score": "train_score",
    "projection": "train_projection",
    "transport": "eb_transport",
}
STAGE_CONFIG: dict[Stage, str] = {
    "score": "score.json",
    "projection": "projector.json",
    "transport": "transport.json",
}


class ReproduceConfig(ExperimentConfig):
    """Configuration for the full real-data reproduction campaign."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    config_dir: Path = Field(
        default=DEFAULT_CONFIG_DIR,
        description="Directory containing the per-stage JSON configs.",
    )
    results_root: Path = Field(
        default=DEFAULT_RESULTS_ROOT,
        description="Root for all stage outputs and per-job logs.",
    )
    log_dir: Path | None = Field(
        default=None,
        description="Per-job stdout/stderr logs (default: <results_root>/_logs).",
    )
    datasets: Annotated[tuple[Dataset, ...], NoDecode] = Field(
        default=DATASETS,
        description="Datasets to run.",
    )
    holdouts: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(),
        description="Held-out windows to run (default: every window of each dataset).",
    )
    seeds: Annotated[tuple[int, ...], NoDecode] = Field(
        default=DEFAULT_SEEDS,
        description="Seeds run for every holdout.",
    )
    stages: Annotated[tuple[Stage, ...], NoDecode] = Field(
        default=STAGES,
        description="Stages to run, in the given order.",
    )
    workers: int | None = Field(
        default=None,
        ge=1,
        description="Parallel jobs per stage (default: number of GPUs, else 1).",
    )
    gpus: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(),
        description="GPU ids assigned to jobs round-robin; empty leaves all visible.",
    )
    override: bool = Field(
        default=False,
        description="Pass --run.override to every job.",
    )
    skip_existing: bool = Field(
        default=False,
        description="Skip jobs whose complete.json already exists.",
    )
    keep_going: bool = Field(
        default=False,
        description="Run the rest of a stage after a failure.",
    )
    dry_run: bool = Field(
        default=False,
        description="Print commands without executing them.",
    )
    stage_overrides: dict[Stage, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-stage overrides deep-merged into the stage config file.",
    )
    dataset_overrides: dict[Dataset, dict[Stage, dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Per-dataset stage overrides ``{dataset: {stage: {field: value}}}``; "
            "take precedence over ``stage_overrides``."
        ),
    )

    @field_validator("datasets", "holdouts", "seeds", "stages", "gpus", mode="before")
    @classmethod
    def _split_comma_separated(cls, value):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value


@dataclass(frozen=True)
class Job:
    """One stage-script invocation and the marker that proves it finished."""

    stage: Stage
    dataset: Dataset
    holdout: str
    seed: int
    argv: tuple[str, ...]
    marker: Path
    log: Path

    @property
    def label(self) -> str:
        return f"{self.stage}/{self.dataset}/{self.holdout}/s{self.seed}"


@dataclass(frozen=True)
class Failure:
    """A failed job with the tail of its log for reporting."""

    job: Job
    returncode: int
    log_tail: str


def _run_id(holdout: str, seed: int) -> str:
    """Return the run id shared by every stage of one cell."""
    return f"holdout_{holdout}-s{seed}"


def _stage_out_dir(
    results_root: Path, dataset: Dataset, holdout: str, seed: int, stage: Stage
) -> Path:
    """Return one stage's run directory under the cell's shared run id."""
    return results_root / dataset / _run_id(holdout, seed) / stage


def _portable(path: Path) -> str:
    """Return ``path`` relative to the repo root for child processes."""
    return os.path.relpath(path, REPO_ROOT)


def _calibration(data_root: Path, dataset: Dataset) -> dict[str, Any]:
    """Return the dataset's derived calibration values."""
    path = REPO_ROOT / data_root / dataset / "geometry" / "calibration.json"
    if not path.is_file():
        raise SystemExit(
            f"missing calibration: {path}\n"
            f"Run prepare_eb_geometry.py / calibrate_scales.py for {dataset} first."
        )
    return json.loads(path.read_text())["derived"]


def _effective_config_path(
    stage: Stage,
    dataset: Dataset,
    config_dir: Path,
    log_dir: Path,
    overrides: Mapping[str, Any],
    dataset_overrides: Mapping[str, Any],
) -> Path:
    """Return the stage config for one dataset, materialising merged overrides."""
    base_path = config_dir / STAGE_CONFIG[stage]
    if not overrides and not dataset_overrides:
        return base_path
    merged = deep_update(
        json.loads(base_path.read_text()),
        deep_update(dict(overrides), dict(dataset_overrides)),
    )
    dest = log_dir / "_configs" / f"{base_path.stem}_{dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, indent=2) + "\n")
    return dest


def _make_job(
    stage: Stage,
    dataset: Dataset,
    holdout: str,
    seed: int,
    *,
    config_path: Path,
    results_root: Path,
    log_dir: Path,
    override: bool,
) -> Job:
    data_root = DATA_ROOTS[dataset]
    out_dir = _stage_out_dir(results_root, dataset, holdout, seed, stage)
    argv = [
        sys.executable,
        "-m",
        f"experiments.real_data.scripts.{STAGE_SCRIPT[stage]}",
        "--config",
        _portable(config_path),
        "--scenario",
        dataset,
        "--data-root",
        str(data_root),
        "--seed",
        str(seed),
        "--run.out-dir",
        _portable(out_dir),
    ]
    if stage == "score":
        calibration = _calibration(data_root, dataset)
        argv += [
            "--variant",
            f"holdout_{holdout}",
            "--schedule.sigma-min",
            str(calibration["sigma_min"]),
            "--schedule.sigma-max",
            str(calibration["sigma_max"]),
        ]
    elif stage == "projection":
        upstream = _stage_out_dir(results_root, dataset, holdout, seed, "score")
        argv += [
            "--variant",
            f"holdout_{holdout}",
            "--score-checkpoint",
            _portable(upstream / "score.pt"),
        ]
    else:
        upstream = _stage_out_dir(results_root, dataset, holdout, seed, "projection")
        argv += [
            "--held-out",
            holdout,
            "--projector-checkpoint",
            _portable(upstream / "projector.pt"),
        ]
    if override:
        argv.append("--run.override")
    return Job(
        stage=stage,
        dataset=dataset,
        holdout=holdout,
        seed=seed,
        argv=tuple(argv),
        marker=out_dir / "complete.json",
        log=log_dir / stage / f"{dataset}_{holdout}_s{seed}.log",
    )


def _dataset_holdouts(dataset: Dataset, selected: Sequence[str]) -> tuple[str, ...]:
    if not selected:
        return HOLDOUTS[dataset]
    return tuple(h for h in HOLDOUTS[dataset] if h in set(selected))


def build_jobs(
    stage: Stage,
    datasets: Sequence[Dataset],
    holdouts: Sequence[str],
    seeds: Sequence[int],
    *,
    config_dir: Path,
    results_root: Path,
    log_dir: Path,
    override: bool,
    overrides: Mapping[str, Any],
    dataset_overrides: Mapping[str, Mapping[str, Any]],
) -> list[Job]:
    """Build the stage's jobs, one per dataset, holdout and seed."""
    jobs: list[Job] = []
    for dataset in datasets:
        dataset_holdouts = _dataset_holdouts(dataset, holdouts)
        if not dataset_holdouts:
            raise SystemExit(f"No holdouts selected for {dataset}.")
        config_path = _effective_config_path(
            stage,
            dataset,
            config_dir,
            log_dir,
            overrides,
            dataset_overrides.get(dataset, {}).get(stage, {}),
        )
        for holdout in dataset_holdouts:
            for seed in seeds:
                jobs.append(
                    _make_job(
                        stage,
                        dataset,
                        holdout,
                        seed,
                        config_path=config_path,
                        results_root=results_root,
                        log_dir=log_dir,
                        override=override,
                    )
                )
    return jobs


def _log_tail(path: Path, lines: int = 25) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _run_one(job: Job, env: dict[str, str], stop: threading.Event) -> int | None:
    if stop.is_set():
        return None
    job.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    display_argv = (_portable(Path(job.argv[0])), *job.argv[1:])
    with job.log.open("w") as log_file:
        log_file.write(f"$ {' '.join(display_argv)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            job.argv,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    logger.info(
        "%-44s %4s %7.1fs",
        job.label,
        "ok" if completed.returncode == 0 else "FAIL",
        time.monotonic() - started,
    )
    return completed.returncode


def run_stage(
    stage: Stage,
    jobs: Sequence[Job],
    *,
    workers: int,
    gpus: Sequence[int],
    skip_existing: bool,
    keep_going: bool,
    dry_run: bool,
) -> list[Failure]:
    """Run a stage's jobs in parallel and return the failures."""
    pending = []
    skipped = 0
    for job in jobs:
        if skip_existing and job.marker.exists():
            skipped += 1
            logger.info("skip %s (already complete)", job.label)
        else:
            pending.append(job)
    logger.info("Stage %s: %d to run, %d skipped", stage, len(pending), skipped)

    if dry_run:
        for job in pending:
            logger.info("$ %s", " ".join(job.argv))
        return []

    stop = threading.Event()
    failures: list[Failure] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for index, job in enumerate(pending):
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
            futures[pool.submit(_run_one, job, env, stop)] = job
        for future in as_completed(futures):
            job = futures[future]
            returncode = future.result()
            if returncode is None or returncode == 0:
                continue
            failures.append(Failure(job, returncode, _log_tail(job.log)))
            if not keep_going:
                stop.set()

    for failure in failures:
        logger.error(
            "Job %s exited with %d; last lines of %s:\n%s",
            failure.job.label,
            failure.returncode,
            failure.job.log,
            failure.log_tail,
        )
    return failures


def _selected_stages(config: ReproduceConfig) -> tuple[Stage, ...]:
    ordered = tuple(stage for stage in STAGES if stage in set(config.stages))
    if ordered != tuple(config.stages):
        raise SystemExit(f"--stages must follow the order {STAGES}")
    return ordered


def _check_configs(config_dir: Path, stages: Sequence[Stage]) -> None:
    missing = [
        config_dir / STAGE_CONFIG[stage]
        for stage in stages
        if not (config_dir / STAGE_CONFIG[stage]).is_file()
    ]
    if missing:
        raise SystemExit("Missing config file(s): " + ", ".join(map(str, missing)))


def main() -> int:
    config = ReproduceConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_dir = config.config_dir.expanduser().resolve()
    results_root = config.results_root.expanduser().resolve()
    log_dir = (config.log_dir or results_root / "_logs").expanduser().resolve()
    workers = config.workers or (len(config.gpus) if config.gpus else 1)
    stages = _selected_stages(config)
    _check_configs(config_dir, stages)
    if not config.datasets:
        raise SystemExit("No datasets selected.")
    if not config.seeds:
        raise SystemExit("No seeds selected.")
    known_holdouts = {h for dataset in config.datasets for h in HOLDOUTS[dataset]}
    unknown = [h for h in config.holdouts if h not in known_holdouts]
    if unknown:
        raise SystemExit(f"Unknown holdouts: {unknown}")

    n_cells = sum(
        len(_dataset_holdouts(dataset, config.holdouts)) for dataset in config.datasets
    )
    logger.info("Datasets: %s", ", ".join(config.datasets))
    logger.info(
        "Holdouts: %s",
        ", ".join(config.holdouts) if config.holdouts else "all per dataset",
    )
    logger.info("Seeds: %s", ", ".join(str(seed) for seed in config.seeds))
    logger.info("Stages: %s", ", ".join(stages))
    logger.info("Results root: %s", results_root)
    logger.info(
        "Workers: %d%s", workers, f" (GPUs {config.gpus})" if config.gpus else ""
    )
    logger.info(
        "Plan: %d cells, %d jobs per stage", n_cells, n_cells * len(config.seeds)
    )
    if config.dry_run:
        logger.info("Dry run: no jobs will be executed.")

    for stage in stages:
        logger.info("=" * 72)
        jobs = build_jobs(
            stage,
            config.datasets,
            config.holdouts,
            config.seeds,
            config_dir=config_dir,
            results_root=results_root,
            log_dir=log_dir,
            override=config.override,
            overrides=config.stage_overrides.get(stage, {}),
            dataset_overrides=config.dataset_overrides,
        )
        failures = run_stage(
            stage,
            jobs,
            workers=workers,
            gpus=config.gpus,
            skip_existing=config.skip_existing,
            keep_going=config.keep_going,
            dry_run=config.dry_run,
        )
        if failures:
            logger.error("Stage %s aborted with %d failure(s)", stage, len(failures))
            return 1

    logger.info("Campaign complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
