#!/usr/bin/env python
"""Dispatch all toy-manifold stages over the configured manifolds and seeds."""

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

from experiments.toy_manifolds.common import MANIFOLDS, ManifoldName
from geometry_distill.config import ExperimentConfig, deep_update

logger = logging.getLogger("reproduce")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "toy_manifolds" / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "toy_manifolds"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

Stage = Literal["score", "projection", "uniform", "geodesics"]

STAGES: tuple[Stage, ...] = ("score", "projection", "uniform", "geodesics")

STAGE_SCRIPT: dict[Stage, str] = {
    "score": "train_score",
    "projection": "train_projection",
    "uniform": "generate_uniform",
    "geodesics": "generate_geodesics",
}

STAGE_CONFIG: dict[Stage, str] = {
    "score": "score.json",
    "projection": "projector.json",
    "uniform": "uniform_generation.json",
    "geodesics": "geodesics.json",
}

STAGE_DIR: dict[Stage, str] = {
    "score": "score_model",
    "projection": "projector",
    "uniform": "uniform_generation",
    "geodesics": "geodesics",
}


class ReproduceConfig(ExperimentConfig):
    """Configuration for the full toy-manifold reproduction campaign."""

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
    checkpoint_root: Path | None = Field(
        default=None,
        description="Root holding upstream stage checkpoints (default: results_root).",
    )
    log_dir: Path | None = Field(
        default=None,
        description="Per-job stdout/stderr logs (default: <results_root>/_logs).",
    )
    manifolds: Annotated[tuple[ManifoldName, ...], NoDecode] = Field(
        default=tuple(MANIFOLDS),
        description="Manifolds to include.",
    )
    seeds: Annotated[tuple[int, ...], NoDecode] = Field(
        default=DEFAULT_SEEDS,
        description="Seeds to run for every manifold.",
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
    manifold_overrides: dict[ManifoldName, dict[Stage, dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Per-manifold stage overrides ``{manifold: {stage: {field: value}}}``; "
            "take precedence over ``stage_overrides``."
        ),
    )

    @field_validator("manifolds", "seeds", "stages", "gpus", mode="before")
    @classmethod
    def _split_comma_separated(cls, value):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value


@dataclass(frozen=True)
class Job:
    """One stage-script invocation and the marker that proves it finished."""

    stage: Stage
    manifold: str
    seed: int
    argv: tuple[str, ...]
    marker: Path
    log: Path

    @property
    def label(self) -> str:
        return f"{self.stage}/{self.manifold}/seed_{self.seed}"


@dataclass(frozen=True)
class Failure:
    """A failed job with the tail of its log for reporting."""

    job: Job
    returncode: int
    log_tail: str


def _stage_out_dir(results_root: Path, stage: Stage, manifold: str, seed: int) -> Path:
    return results_root / STAGE_DIR[stage] / manifold / f"seed_{seed}"


def _portable(path: Path) -> str:
    """Return ``path`` relative to the repo root for child processes."""
    return os.path.relpath(path, REPO_ROOT)


def _effective_config_path(
    stage: Stage,
    manifold: str,
    config_dir: Path,
    log_dir: Path,
    overrides: Mapping[str, Any],
    manifold_overrides: Mapping[str, Any],
) -> Path:
    """Return the stage config for one manifold, materialising merged overrides."""
    base_path = config_dir / STAGE_CONFIG[stage]
    if not overrides and not manifold_overrides:
        return base_path
    merged = deep_update(
        json.loads(base_path.read_text()),
        deep_update(dict(overrides), dict(manifold_overrides)),
    )
    dest = log_dir / "_configs" / f"{base_path.stem}_{manifold}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, indent=2) + "\n")
    return dest


def _make_job(
    stage: Stage,
    manifold: str,
    seed: int,
    *,
    config_path: Path,
    results_root: Path,
    checkpoint_root: Path,
    log_dir: Path,
    override: bool,
) -> Job:
    out_dir = _stage_out_dir(results_root, stage, manifold, seed)
    argv = [
        sys.executable,
        "-m",
        f"experiments.toy_manifolds.scripts.{STAGE_SCRIPT[stage]}",
        "--config",
        _portable(config_path),
        "--manifold",
        manifold,
        "--seed",
        str(seed),
        "--run.out-dir",
        _portable(out_dir),
    ]
    if stage == "projection":
        score_ckpt = (
            _stage_out_dir(checkpoint_root, "score", manifold, seed) / "score.pt"
        )
        argv += ["--score-checkpoint", _portable(score_ckpt)]
    elif stage in ("uniform", "geodesics"):
        projector_ckpt = (
            _stage_out_dir(checkpoint_root, "projection", manifold, seed)
            / "projector.pt"
        )
        argv += ["--projector-checkpoint", _portable(projector_ckpt)]
    if override:
        argv.append("--run.override")
    return Job(
        stage=stage,
        manifold=manifold,
        seed=seed,
        argv=tuple(argv),
        marker=out_dir / "complete.json",
        log=log_dir / stage / f"{manifold}_seed{seed}.log",
    )


def build_jobs(
    stage: Stage,
    manifolds: Sequence[str],
    seeds: Sequence[int],
    *,
    config_dir: Path,
    results_root: Path,
    checkpoint_root: Path | None = None,
    log_dir: Path,
    override: bool,
    overrides: Mapping[str, Any],
    manifold_overrides: Mapping[str, Mapping[str, Any]],
) -> list[Job]:
    """Build the stage's jobs, one per manifold and seed."""
    checkpoint_root = checkpoint_root or results_root
    config_paths = {
        manifold: _effective_config_path(
            stage,
            manifold,
            config_dir,
            log_dir,
            overrides,
            manifold_overrides.get(manifold, {}).get(stage, {}),
        )
        for manifold in manifolds
    }
    return [
        _make_job(
            stage,
            manifold,
            seed,
            config_path=config_paths[manifold],
            results_root=results_root,
            checkpoint_root=checkpoint_root,
            log_dir=log_dir,
            override=override,
        )
        for manifold in manifolds
        for seed in seeds
    ]


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
        "%-40s %4s %7.1fs",
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
    checkpoint_root = (
        (config.checkpoint_root or config.results_root).expanduser().resolve()
    )
    log_dir = (config.log_dir or results_root / "_logs").expanduser().resolve()
    workers = config.workers or (len(config.gpus) if config.gpus else 1)
    _check_configs(config_dir, config.stages)

    n_jobs = len(config.manifolds) * len(config.seeds)
    logger.info("Manifolds: %s", ", ".join(config.manifolds))
    logger.info("Seeds: %s", ", ".join(str(seed) for seed in config.seeds))
    logger.info("Stages: %s", ", ".join(config.stages))
    logger.info("Results root: %s", results_root)
    logger.info(
        "Workers: %d%s", workers, f" (GPUs {config.gpus})" if config.gpus else ""
    )
    logger.info("Plan: %d jobs, %d per stage", n_jobs * len(config.stages), n_jobs)
    if config.dry_run:
        logger.info("Dry run: no jobs will be executed.")

    for stage in config.stages:
        logger.info("=" * 72)
        jobs = build_jobs(
            stage,
            config.manifolds,
            config.seeds,
            config_dir=config_dir,
            results_root=results_root,
            checkpoint_root=checkpoint_root,
            log_dir=log_dir,
            override=config.override,
            overrides=config.stage_overrides.get(stage, {}),
            manifold_overrides=config.manifold_overrides,
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
