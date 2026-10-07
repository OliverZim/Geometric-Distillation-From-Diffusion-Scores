"""Write one run's config, metadata, metrics, logs, and artifacts."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from .config import ENV_PREFIX, ComponentConfig, ExperimentConfig


class RunConfig(ComponentConfig):
    """Where a run writes its outputs and how existing outputs are handled."""

    out_dir: Path = Field(
        description="Run output directory (must be set explicitly).",
    )
    override: bool = Field(
        default=False,
        description="Replace an existing non-empty out_dir instead of failing.",
    )


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _repo_root() -> Path | None:
    toplevel = _git("rev-parse", "--show-toplevel")
    return Path(toplevel) if toplevel else None


def _relativize(token: str, repo_root: Path | None) -> str:
    """Shorten absolute paths to repo- or home-relative form."""
    if not Path(token).is_absolute():
        return token
    normalized = Path(os.path.abspath(token))
    for root, prefix in ((repo_root, None), (Path.home(), "~")):
        if root is None:
            continue
        try:
            rel = normalized.relative_to(Path(os.path.abspath(root)))
        except ValueError:
            continue
        return str(rel) if prefix is None else str(Path(prefix) / rel)
    return token


def _relativize_data(value: Any, repo_root: Path | None) -> Any:
    """Recursively anonymize absolute path strings in a config dump."""
    if isinstance(value, str):
        return _relativize(value, repo_root)
    if isinstance(value, dict):
        return {key: _relativize_data(item, repo_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativize_data(item, repo_root) for item in value]
    return value


class _PathAnonymizer(logging.Filter):
    """Rewrite repo- and home-absolute paths in log messages to relative form."""

    def __init__(self, repo_root: Path | None):
        super().__init__()
        prefixes = [(str(Path.home()) + os.sep, "~/")]
        if repo_root is not None:
            prefixes.insert(0, (str(repo_root) + os.sep, ""))
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        anonymized = message
        for prefix, replacement in self._prefixes:
            anonymized = anonymized.replace(prefix, replacement)
        if anonymized != message:
            record.msg = anonymized
            record.args = ()
        return True


def _prefixed_env(repo_root: Path | None) -> dict[str, str]:
    """Snapshot environment variables with the experiment prefix."""
    return {
        key: _relativize(value, repo_root)
        for key, value in sorted(os.environ.items())
        if key.upper().startswith(ENV_PREFIX)
    }


def _setup_logging(
    out_dir: Path, level: int, repo_root: Path | None
) -> list[logging.Handler]:
    """Log to stdout and ``console.log``."""
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    )
    log_file = logging.FileHandler(out_dir / "console.log", mode="w")
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    anonymizer = _PathAnonymizer(repo_root)
    console.addFilter(anonymizer)
    log_file.addFilter(anonymizer)
    root.addHandler(console)
    root.addHandler(log_file)
    logging.captureWarnings(True)
    return [console, log_file]


def _guard_override(resolved: Path) -> None:
    cwd = Path.cwd().resolve()
    if (
        resolved == Path(resolved.anchor)
        or resolved == Path.home()
        or cwd.is_relative_to(resolved)
    ):
        raise ValueError(
            f"Refusing to override {resolved}: it is the current working "
            "directory, one of its parents, the home directory, or a "
            "filesystem root."
        )


class RunWriter:
    """Write one run's config, metadata, metrics, logs, and artifacts."""

    def __init__(self, config: RunConfig, *, level: int = logging.INFO):
        self._out_dir = self._prepare_dir(config.out_dir, config.override)
        self._metrics_path = self._out_dir / "metrics.jsonl"
        self._closed = False
        self._repo_root = _repo_root()
        self._metadata: dict[str, Any] = {
            "created_at": _timestamp(),
            "argv": [_relativize(arg, self._repo_root) for arg in sys.argv],
            "env": _prefixed_env(self._repo_root),
            "git_commit": _git("rev-parse", "HEAD"),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        self._write_json("metadata.json", self._metadata)
        self._handlers = _setup_logging(self._out_dir, level, self._repo_root)

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(complete=exc_type is None)

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    def write_config(self, config: ExperimentConfig) -> None:
        """Write the resolved experiment config to ``config.json``."""
        data = _relativize_data(config.to_dict(), self._repo_root)
        self._write_json("config.json", data)
        config_file = config.config
        self._metadata["config_file"] = (
            _relativize(str(config_file), self._repo_root) if config_file else None
        )
        self._write_json("metadata.json", self._metadata)

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Append one metrics record for ``step`` to ``metrics.jsonl``."""
        record: dict[str, Any] = {"step": int(step)}
        record.update({str(key): float(value) for key, value in metrics.items()})
        with open(self._metrics_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def save_json(self, name: str, data: Mapping[str, Any]) -> Path:
        """Atomically write a JSON artifact to the run directory."""
        path = self._artifact_path(name)
        self._write_json_path(path, dict(data))
        return path

    def save_checkpoint(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Atomically save a checkpoint to the run directory."""
        path = self._artifact_path(name)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(dict(payload), tmp)
        tmp.replace(path)
        return path

    def close(self, complete: bool = True) -> None:
        """Mark the run complete and detach the logging handlers."""
        if self._closed:
            return
        if complete:
            self._write_json("complete.json", {"finished_at": _timestamp()})
        for handler in self._handlers:
            logging.getLogger().removeHandler(handler)
            handler.close()
        self._closed = True

    @staticmethod
    def _prepare_dir(out_dir: Path, override: bool) -> Path:
        expanded = out_dir.expanduser()
        resolved = expanded.resolve()
        if resolved.is_dir() and any(resolved.iterdir()):
            if not override:
                raise FileExistsError(
                    f"Output directory {resolved} already exists and is not "
                    "empty; choose another out_dir or set override to replace it."
                )
            _guard_override(resolved)
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        return expanded

    def _artifact_path(self, name: str) -> Path:
        parts = Path(name).parts
        if len(parts) != 1 or parts[0] in (".", ".."):
            raise ValueError(f"artifact name must be a plain file name, got {name!r}")
        return self._out_dir / name

    def _write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        path = self._artifact_path(name)
        self._write_json_path(path, data)
        return path

    @staticmethod
    def _write_json_path(path: Path, data: Mapping[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(dict(data), indent=2, default=str))
        tmp.replace(path)
