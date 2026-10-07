"""Publication figure style shared by the figure notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

DPI: int = 300

FORMATS: tuple[str, ...] = ("svg", "png")
FORMAT: str = FORMATS[0]

_EXTENSIONS = (".png", ".pdf", ".svg")

_RC = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Labels are drawn as glyph outlines: a substituted font would space
    # ``$...$`` spans visibly wrong, and outlines render identically everywhere.
    "svg.fonttype": "path",
}


def apply_paper_style() -> None:
    """Apply the matplotlib rcParams used by the paper figures."""
    mpl.rcParams.update(_RC)


def _sibling_paths(path: Path) -> list[Path]:
    """Primary path first, then the other entries in :data:`FORMATS`."""
    stem = path.with_suffix("")
    primary_ext = path.suffix.lstrip(".").lower() or FORMAT
    ordered = [primary_ext] + [f for f in FORMATS if f != primary_ext]
    seen: set[str] = set()
    out: list[Path] = []
    for fmt in ordered:
        if fmt in seen:
            continue
        seen.add(fmt)
        out.append(stem.with_suffix(f".{fmt}"))
    return out


def savefig(
    fig: plt.Figure,
    path: Path | str,
    *,
    dpi: int = DPI,
    close: bool = True,
    bbox_inches: Union[str, Bbox, None] = "tight",
    pad_inches: float = 0.05,
) -> Path:
    """Save a figure as SVG and PNG at publication DPI, returning the SVG path.

    Pass a ``Bbox`` with ``pad_inches=0`` to freeze the canvas instead of
    letting ``tight`` crop 3D panes and panel titles.
    """
    if dpi < 300:
        raise ValueError(f"Paper figures must be >= 300 DPI, got {dpi}")
    requested = Path(path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for out in _sibling_paths(requested):
        fig.savefig(
            out,
            dpi=dpi,
            facecolor="white",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        written[out.suffix.lstrip(".").lower()] = out.resolve()
    if close:
        plt.close(fig)
    return written.get(FORMAT, next(iter(written.values())))


def figure_path(
    out_root: Path | str,
    *parts: str,
    ext: Optional[str] = None,
) -> Path:
    """Build ``{out_root}/.../name.{ext}`` without creating directories."""
    root = Path(out_root)
    name = parts[-1]
    if not name.endswith(_EXTENSIONS):
        name = f"{name}.{ext or FORMAT}"
    return root.joinpath(*parts[:-1], name)
