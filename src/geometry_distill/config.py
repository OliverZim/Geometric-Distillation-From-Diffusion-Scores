"""Base classes for experiment and component configurations."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prefix for experiment environment variables.
ENV_PREFIX = "GD_"


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _nested_model_defaults(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """Return the nested model defaults of ``settings_cls`` as plain dicts."""
    defaults: dict[str, Any] = {}
    for name, field in settings_cls.model_fields.items():
        if field.default_factory is not None:
            default = field.default_factory()
        elif field.is_required():
            continue
        else:
            default = field.default
        if isinstance(default, BaseModel):
            defaults[name] = default.model_dump()
    return defaults


class ComponentConfig(BaseModel):
    """Base class for immutable component configurations."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExperimentConfig(BaseSettings):
    """Base class for experiment settings."""

    config: Path | None = None

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        cli_implicit_flags="toggle",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_config_file(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values

        file_values = cls._read_config_file(values.get("config"))
        merged = deep_update(_nested_model_defaults(cls), file_values)
        return deep_update(merged, values)

    @staticmethod
    def _read_config_file(path: str | Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            values = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(values, dict):
            raise ValueError(f"Config file {path} must contain a JSON object")
        return values

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved config as a JSON-compatible dict without ``config``."""
        return self.model_dump(mode="json", exclude={"config"})

    def to_json(self, indent: int = 2) -> str:
        """Return the resolved config as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_json(self, indent: int = 2) -> None:
        """Print the resolved config as JSON."""
        print(self.to_json(indent=indent))
