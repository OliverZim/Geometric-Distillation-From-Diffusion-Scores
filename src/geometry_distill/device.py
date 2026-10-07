"""Torch device selection."""

from __future__ import annotations

import torch

_AUTO_ORDER = ("cuda", "mps", "cpu")


def _is_available(device: str) -> bool:
    """Whether ``device`` (``cpu``, ``cuda`` or ``mps``) is usable."""
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def resolve_device(preference: str | None = "auto") -> torch.device:
    """Resolve ``preference`` to an available torch device."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        pref = next(d for d in _AUTO_ORDER if _is_available(d))
    elif pref not in _AUTO_ORDER:
        raise ValueError(
            f"Unknown device preference {preference!r}; use auto|cpu|cuda|mps"
        )
    elif not _is_available(pref):
        raise RuntimeError(f"Requested device {pref!r} is not available")
    return torch.device(pref)
