#!/usr/bin/env python
"""Dispatch the synthetic scRNA-seq campaign across scenarios, variants and seeds."""

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
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "synthetic_data" / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthetic_data"
DEFAULT_SEEDS = (42, 43, 44, 45)

Scenario = Literal["linear", "bifurcation", "tree"]
Stage = Literal["score", "projection", "geodesics", "transport"]
Pipeline = Literal["geodesics", "transport"]

SCENARIOS: tuple[Scenario, ...] = ("linear", "bifurcation", "tree")
STAGES: tuple[Stage, ...] = ("score", "projection", "geodesics", "transport")
PIPELINES: tuple[Pipeline, ...] = ("geodesics", "transport")

PIPELINE_VARIANT: dict[Pipeline, str] = {
    "geodesics": "full",
    "transport": "holdout_mid",
}
PIPELINE_STAGES: dict[Pipeline, tuple[Stage, ...]] = {
    "geodesics": ("score", "projection", "geodesics"),
    "transport": ("score", "projection", "transport"),
}
PIPELINE_RUN_PREFIX: dict[Pipeline, str] = {
    "geodesics": "geodesics-full",
    "transport": "transport-holdout",
}

STAGE_SCRIPT: dict[Stage, str] = {
    "score": "train_score",
    "projection": "train_projection",
    "geodesics": "train_geodesics",
    "transport": "sc_transport",
}
STAGE_CONFIG: dict[Stage, str] = {
    "score": "score.json",
    "projection": "projector.json",
    "geodesics": "geodesics.json",
    "transport": "transport.json",
}


class ReproduceConfig(ExperimentConfig):
    """Configuration for the full synthetic scRNA-seq reproduction campaign."""

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
    scenarios: Annotated[tuple[Scenario, ...], NoDecode] = Field(
        default=SCENARIOS,
        description="Scenarios to run.",
    )
    seeds: Annotated[tuple[int, ...], NoDecode] = Field(
        default=DEFAULT_SEEDS,
        description="Seeds run for every scenario and pipeline.",
    )
    pipelines: Annotated[tuple[Pipeline, ...], NoDecode] = Field(
        default=PIPELINES,
        description=(
            "'geodesics' (full variant: score, projection, geodesics) and "
            "'transport' (holdout_mid variant: score, projection, transport)."
        ),
    )
    stages: Annotated[tuple[Stage, ...], NoDecode] | None = Field(
        default=None,
        description="Stages to run (default: every stage of the selected pipelines).",
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

    @field_validator("scenarios", "seeds", "pipelines", "stages", "gpus", mode="before")
    @classmethod
    def _split_comma_separated(cls, value):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value


@dataclass(frozen=True)
class Job:
    """One stage-script invocation and the marker that proves it finished."""

    pipeline: Pipeline
    stage: Stage
    scenario: Scenario
    seed: int
    argv: tuple[str, ...]
    marker: Path
    log: Path

    @property
    def label(self) -> str:
        return f"{self.pipeline}/{self.stage}/{self.scenario}/s{self.seed}"


@dataclass(frozen=True)
class Failure:
    """A failed job with the tail of its log for reporting."""

    job: Job
    returncode: int
    log_tail: str


def _run_id(pipeline: Pipeline, seed: int) -> str:
    """Return the run id shared by every stage of one pipeline and seed."""
    return f"{PIPELINE_RUN_PREFIX[pipeline]}-s{seed}"


def _stage_out_dir(
    results_root: Path, pipeline: Pipeline, scenario: str, seed: int, stage: Stage
) -> Path:
    """Return one stage's run directory under the scenario's shared run id."""
    return results_root / scenario / _run_id(pipeline, seed) / stage


def _portable(path: Path) -> str:
    """Return ``path`` relative to the repo root for child processes."""
    return os.path.relpath(path, REPO_ROOT)


def _effective_config_path(
    stage: Stage,
    config_dir: Path,
    log_dir: Path,
    overrides: Mapping[str, Any],
) -> Path:
    """Return the stage config for one stage, materialising merged overrides."""
    base_path = config_dir / STAGE_CONFIG[stage]
    if not overrides:
        return base_path
    merged = deep_update(json.loads(base_path.read_text()), dict(overrides))
    dest = log_dir / "_configs" / f"{base_path.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, indent=2) + "\n")
    return dest


def _make_job(
    stage: Stage,
    pipeline: Pipeline,
    scenario: Scenario,
    seed: int,
    *,
    config_path: Path,
    results_root: Path,
    log_dir: Path,
    override: bool,
) -> Job:
    out_dir = _stage_out_dir(results_root, pipeline, scenario, seed, stage)
    argv = [
        sys.executable,
        "-m",
        f"experiments.synthetic_data.scripts.{STAGE_SCRIPT[stage]}",
        "--config",
        _portable(config_path),
        "--scenario",
        scenario,
        "--variant",
        PIPELINE_VARIANT[pipeline],
        "--seed",
        str(seed),
        "--run.out-dir",
        _portable(out_dir),
    ]
    if stage == "projection":
        upstream = _stage_out_dir(results_root, pipeline, scenario, seed, "score")
        argv += ["--score-checkpoint", _portable(upstream / "score.pt")]
    elif stage in ("geodesics", "transport"):
        upstream = _stage_out_dir(results_root, pipeline, scenario, seed, "projection")
        argv += ["--projector-checkpoint", _portable(upstream / "projector.pt")]
    if override:
        argv.append("--run.override")
    return Job(
        pipeline=pipeline,
        stage=stage,
        scenario=scenario,
        seed=seed,
        argv=tuple(argv),
        marker=out_dir / "complete.json",
        log=log_dir / stage / f"{pipeline}_{scenario}_s{seed}.log",
    )


def build_jobs(
    stage: Stage,
    pipelines: Sequence[Pipeline],
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    *,
    config_dir: Path,
    results_root: Path,
    log_dir: Path,
    override: bool,
    overrides: Mapping[str, Any],
) -> list[Job]:
    """Build the stage's jobs for every selected pipeline, scenario and seed."""
    config_path = _effective_config_path(stage, config_dir, log_dir, overrides)
    return [
        _make_job(
            stage,
            pipeline,
            scenario,
            seed,
            config_path=config_path,
            results_root=results_root,
            log_dir=log_dir,
            override=override,
        )
        for pipeline in pipelines
        if stage in PIPELINE_STAGES[pipeline]
        for scenario in scenarios
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


def _selected_pipelines(config: ReproduceConfig) -> tuple[Pipeline, ...]:
    """Return the selected pipelines in pipeline order."""
    ordered = tuple(
        pipeline for pipeline in PIPELINES if pipeline in set(config.pipelines)
    )
    if ordered != tuple(config.pipelines):
        raise SystemExit(f"--pipelines must follow the order {PIPELINES}")
    return ordered


def _selected_stages(
    config: ReproduceConfig, pipelines: Sequence[Pipeline]
) -> tuple[Stage, ...]:
    """Return the selected stages in stage order for the selected pipelines."""
    allowed = {stage for pipeline in pipelines for stage in PIPELINE_STAGES[pipeline]}
    if config.stages is None:
        return tuple(stage for stage in STAGES if stage in allowed)
    unknown = [stage for stage in config.stages if stage not in allowed]
    if unknown:
        raise SystemExit(f"stages not in the selected pipelines: {unknown}")
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
    pipelines = _selected_pipelines(config)
    stages = _selected_stages(config, pipelines)
    _check_configs(config_dir, stages)
    if not config.scenarios:
        raise SystemExit("No scenarios selected.")
    if not config.seeds:
        raise SystemExit("No seeds selected.")

    n_runs = len(pipelines) * len(config.scenarios) * len(config.seeds)
    logger.info("Scenarios: %s", ", ".join(config.scenarios))
    logger.info("Seeds: %s", ", ".join(str(seed) for seed in config.seeds))
    logger.info("Pipelines: %s", ", ".join(pipelines))
    logger.info("Stages: %s", ", ".join(stages))
    logger.info("Results root: %s", results_root)
    logger.info(
        "Workers: %d%s", workers, f" (GPUs {config.gpus})" if config.gpus else ""
    )
    logger.info("Plan: %d runs per stage group", n_runs)
    if config.dry_run:
        logger.info("Dry run: no jobs will be executed.")

    for stage in stages:
        logger.info("=" * 72)
        jobs = build_jobs(
            stage,
            pipelines,
            config.scenarios,
            config.seeds,
            config_dir=config_dir,
            results_root=results_root,
            log_dir=log_dir,
            override=config.override,
            overrides=config.stage_overrides.get(stage, {}),
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
