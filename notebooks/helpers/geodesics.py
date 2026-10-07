"""Chord-refine geodesic figures for the toy-manifold notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import to_rgb
from matplotlib.transforms import Bbox
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from experiments.toy_manifolds.common import (
    MANIFOLDS,
    Manifold,
    manifold_distance,
    sample_uniform,
)
from geometry_distill.geodesics import (
    PathRefinementConfig,
    WarpedPullbackMetricConfig,
    refine_paths,
)
from geometry_distill.projection import projector_from_checkpoint
from geometry_distill.seeding import set_seed

from .style import figure_path, savefig


def geodesics_dir(results_root: Path | str, manifold: str, run_id: str) -> Path:
    """Directory of one geodesics run."""
    return Path(results_root) / "geodesics" / manifold / run_id


def projector_checkpoint(results_root: Path | str, manifold: str, run_id: str) -> Path:
    """Checkpoint path of the projector a geodesics run was built from."""
    return Path(results_root) / "projector" / manifold / run_id / "projector.pt"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


_INK = "#2b2b2b"


_DISPLAY = {
    "hemisphere": "Hemisphere",
    "saddle": "Saddle",
    "paraboloid": "Paraboloid",
}


_COL_W = 7.20


_GEO_PREDICTED_SCALE = 1.5


_GEO_GALLERY_PANEL_BOOST: dict[str, float] = {
    "hemisphere": 1.7,  # 0.85 × previous 2.0
    "saddle": 1.25,
    "paraboloid": 1.0,
}


_DIR_GEODESICS = "geodesics"


def _topic_path(out_dir: Path, topic: str, name: str) -> Path:
    """``{out_dir}/{topic}/{name}.{ext}`` for paper figure layout."""
    return figure_path(out_dir, topic, name)


def _domain_disk_radius(
    manifold: Manifold,
    n_angles: int = 48,
    n_bisect: int = 40,
) -> Optional[float]:
    """Radius of the evaluation domain if it is a disk, else ``None``.

    Found by bisecting along rays from the chart origin. A rotationally
    symmetric domain needs a polar mesh, since masking a Cartesian grid
    against a circle leaves a stair-stepped rim.
    """
    th = torch.linspace(0.0, 2.0 * np.pi, n_angles + 1)[:-1]
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    reach = float(
        min(
            manifold.u_bounds[1] - manifold.u_bounds[0],
            manifold.v_bounds[1] - manifold.v_bounds[0],
        )
    )
    lo = torch.zeros(n_angles)
    hi = torch.full((n_angles,), reach)
    if not bool(manifold.boundary_mask(lo, lo).all()):
        return None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        inside = manifold.boundary_mask(mid * cos_t, mid * sin_t)
        lo = torch.where(inside, mid, lo)
        hi = torch.where(inside, hi, mid)
    r = 0.5 * (lo + hi)
    r_mean = float(r.mean())
    if r_mean <= 0 or float(r.max() - r.min()) / r_mean > 1e-3:
        return None
    return r_mean


_SURFACE_RGB = np.array(to_rgb("#c5d4ce"))


_LIGHT_DIR = np.array([-0.35, -0.55, 0.76])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)


_SHADE_FLOOR = 0.72


def _facet_shading(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Per-quad RGB for a surface grid, lit by a fixed directional source."""
    p = np.stack([X, Y, Z], axis=-1)
    # Cross product of the quad diagonals gives the facet normal.
    d1 = p[1:, 1:] - p[:-1, :-1]
    d2 = p[:-1, 1:] - p[1:, :-1]
    n = np.cross(d1, d2)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
    # Translucent surfaces show both faces, so orientation is ignored.
    intensity = np.abs(n @ _LIGHT_DIR)
    intensity = np.nan_to_num(intensity, nan=1.0)
    shade = _SHADE_FLOOR + (1.0 - _SHADE_FLOOR) * intensity
    return np.clip(shade[..., None] * _SURFACE_RGB, 0.0, 1.0)


def _plot_surface(
    ax,
    manifold: Manifold,
    n_per_axis: int = 80,
    alpha: float = 0.45,
    *,
    zorder: float = 0.5,
    rasterized: bool = True,
    display_chart: bool = False,
) -> None:
    radius = _domain_disk_radius(manifold)
    if manifold.name == "saddle":
        # Elongated valley chart + figure-only XY/Z display scales.
        u = torch.linspace(-_SADDLE_U_HALF, _SADDLE_U_HALF, n_per_axis)
        v = torch.linspace(-_SADDLE_V_HALF, _SADDLE_V_HALF, n_per_axis)
        U, V = torch.meshgrid(u, v, indexing="xy")
        pts = manifold.parametrization(U.ravel(), V.ravel()).numpy()
        X = (_SADDLE_XY_SCALE * pts[:, 0]).reshape(U.shape)
        Y = (_SADDLE_XY_SCALE * pts[:, 1]).reshape(U.shape)
        Z = (_SADDLE_Z_SCALE * pts[:, 2]).reshape(U.shape)
    elif manifold.name == "paraboloid" and display_chart:
        u = torch.linspace(-_PARABOLOID_R_MAX, _PARABOLOID_R_MAX, n_per_axis)
        v = torch.linspace(-_PARABOLOID_R_MAX, _PARABOLOID_R_MAX, n_per_axis)
        U, V = torch.meshgrid(u, v, indexing="xy")
        pts = manifold.parametrization(U.ravel(), V.ravel()).numpy()
        X, Y, Z = (pts[:, i].reshape(U.shape) for i in range(3))
    elif radius is not None:
        # Elliptical grid mapping of the square onto the disk. The square's
        # edge lands exactly on the domain circle, so the silhouette is smooth
        # instead of following grid cells, and unlike a polar mesh it stays
        # regular at the centre rather than collapsing into a fan of slivers.
        a = torch.linspace(-1.0, 1.0, n_per_axis)
        A, B = torch.meshgrid(a, a, indexing="xy")
        U = radius * A * torch.sqrt(torch.clamp(1.0 - 0.5 * B**2, min=0.0))
        V = radius * B * torch.sqrt(torch.clamp(1.0 - 0.5 * A**2, min=0.0))
        pts = manifold.parametrization(U.ravel(), V.ravel()).numpy()
        X, Y, Z = (pts[:, i].reshape(U.shape) for i in range(3))
    else:
        u = torch.linspace(*manifold.u_bounds, n_per_axis)
        v = torch.linspace(*manifold.v_bounds, n_per_axis)
        U, V = torch.meshgrid(u, v, indexing="xy")
        mask = manifold.boundary_mask(U.ravel(), V.ravel()).numpy().reshape(U.shape)
        pts = manifold.parametrization(U.ravel(), V.ravel()).numpy()
        X = np.where(mask, pts[:, 0].reshape(U.shape), np.nan)
        Y = np.where(mask, pts[:, 1].reshape(U.shape), np.nan)
        Z = np.where(mask, pts[:, 2].reshape(U.shape), np.nan)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        alpha=alpha,
        facecolors=_facet_shading(X, Y, Z),
        shade=False,
        edgecolor="none",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        # Default: bake the shell to pixels so SVG does not emit one
        # translucent polygon per quad (antialiased seams → banding).
        # Geodesic figures pass rasterized=False so zorder keeps strokes
        # on top of the shell in the SVG compositor.
        rasterized=rasterized,
    )
    # With ``computed_zorder=False`` the surface must sit below curves, or
    # the mesh paints over every geodesic and they read as interior.
    try:
        surf.set_zorder(zorder)
    except Exception:
        pass


_VIEW_ELEV = 32.0


_VIEW_AZIM = -42.0


_VIEW_BY_NAME: dict[str, tuple[float, float]] = {
    "hemisphere": (32.0, -42.0),
    "saddle": (34.0, -75.0),
    "paraboloid": (32.0, -42.0),
}


_SADDLE_XY_SCALE = 1.15


_SADDLE_Z_SCALE = 0.80


_SADDLE_U_HALF = 1.20


_SADDLE_V_HALF = 1.35


_PARABOLOID_R_MAX = 1.20


def _view_for(manifold: Manifold | str) -> tuple[float, float]:
    name = manifold if isinstance(manifold, str) else manifold.name
    return _VIEW_BY_NAME.get(name, (_VIEW_ELEV, _VIEW_AZIM))


def _display_xyz(manifold: Manifold, pts: np.ndarray) -> np.ndarray:
    """Ambient coordinates used for paper 3-D panels."""
    out = np.asarray(pts, dtype=np.float64)
    if out.size == 0:
        return out
    out = out.copy()
    if manifold.name == "saddle":
        out[..., 0] *= _SADDLE_XY_SCALE
        out[..., 1] *= _SADDLE_XY_SCALE
        out[..., 2] *= _SADDLE_Z_SCALE
    return out


def _clip_to_display_chart(manifold: Manifold, pts: np.ndarray) -> np.ndarray:
    """Drop points outside the chart window used for paper surfaces."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    if manifold.name == "saddle":
        uv = manifold.inverse(
            torch.from_numpy(pts[:, 0].astype(np.float32)),
            torch.from_numpy(pts[:, 1].astype(np.float32)),
            torch.from_numpy(pts[:, 2].astype(np.float32)),
        ).numpy()
        keep = (np.abs(uv[:, 0]) <= _SADDLE_U_HALF) & (
            np.abs(uv[:, 1]) <= _SADDLE_V_HALF
        )
        return pts[keep]
    if manifold.name == "paraboloid":
        uv = manifold.inverse(
            torch.from_numpy(pts[:, 0].astype(np.float32)),
            torch.from_numpy(pts[:, 1].astype(np.float32)),
            torch.from_numpy(pts[:, 2].astype(np.float32)),
        ).numpy()
        keep = np.linalg.norm(uv, axis=-1) <= _PARABOLOID_R_MAX
        return pts[keep]
    if manifold.name == "hemisphere":
        return pts[pts[:, 2] >= -0.01]
    return pts


def _frame_xyz(manifold: Manifold, *clouds: np.ndarray) -> np.ndarray:
    """Concatenated display-space points for equal-aspect framing."""
    parts = []
    for cloud in clouds:
        clipped = _clip_to_display_chart(manifold, np.asarray(cloud))
        if clipped.size:
            parts.append(_display_xyz(manifold, clipped))
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _style_3d(
    ax,
    *,
    elev: float = _VIEW_ELEV,
    azim: float = _VIEW_AZIM,
    tick_labels: bool = True,
    axis_labels: bool = True,
    frame: bool = True,
) -> None:
    # Diagonally above: enough elevation to see the outer cap, enough
    # azimuth to read the side wall.
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d0d0d0")
        axis.line.set_linewidth(0.6)
    ax.tick_params(labelsize=7, pad=-2)
    if axis_labels:
        ax.set_xlabel(r"$x$", labelpad=-4, fontsize=9)
        ax.set_ylabel(r"$y$", labelpad=-4, fontsize=9)
        ax.set_zlabel(r"$z$", labelpad=-4, fontsize=9)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    if not tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    if not frame:
        ax.set_axis_off()


def _inch_axes(
    fig, x: float, y: float, w: float, h: float, fig_w: float, fig_h: float, **kwargs
):
    """Add axes from a rectangle in inches."""
    return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h], **kwargs)


_TITLE_LETTER_W = 0.30  # inches, width of "$(a)$  " at 9.5 pt


def _fig_panel_title(
    fig,
    x: float,
    y: float,
    letter: str,
    text: str,
    fig_w: float,
    fig_h: float,
    *,
    fontsize: float = 9.5,
    letter_w: Optional[float] = None,
) -> None:
    """Left-aligned panel title on a shared figure-inch baseline.

    Letter and body are separate texts so math subscripts cannot shift
    the ``(a)`` / ``(b)`` / ``(c)`` markers off the common baseline.
    """
    gap = _TITLE_LETTER_W if letter_w is None else letter_w
    kw = dict(
        ha="left",
        va="baseline",
        fontsize=fontsize,
        color=_INK,
        transform=fig.transFigure,
        clip_on=False,
        zorder=10,
    )
    fig.text(x / fig_w, y / fig_h, rf"$\mathbf{{({letter})}}$", **kw)
    fig.text((x + gap) / fig_w, y / fig_h, text, **kw)


def _save_canvas(fig: plt.Figure, path: Path, w: float, h: float) -> Path:
    """Save at a frozen inch canvas so tight-crop cannot move 3D panes."""
    return savefig(
        fig,
        path,
        bbox_inches=Bbox.from_extents(0.0, 0.0, w, h),
        pad_inches=0.0,
    )


def _save_balanced(
    fig: plt.Figure,
    path: Path,
    fig_w: float,
    fig_h: float,
    *,
    margin: float = 0.07,
    halign: str = "center",
    vmargin: Optional[float] = None,
) -> Path:
    """Centre the content across the canvas, then trim to a uniform margin.

    The full width is kept, so every two-column figure keeps one text scale
    once placed at a fixed width in the paper; only the vertical slack, which
    does not affect that scale, is cropped away.
    """
    v_margin = margin if vmargin is None else vmargin
    ink = _ink_mask(fig)
    h_px, w_px = ink.shape
    cols = np.where(ink.any(axis=0))[0]
    if cols.size:
        x0 = cols[0] / w_px * fig_w
        x1 = (cols[-1] + 1) / w_px * fig_w
        if halign == "left":
            dx = margin - x0
        else:
            dx = 0.5 * (fig_w - (x1 - x0)) - x0
        if abs(dx) > 5e-3:
            for ax in fig.axes:
                pos = ax.get_position()
                ax.set_position([pos.x0 + dx / fig_w, pos.y0, pos.width, pos.height])
            for text in fig.texts:
                if text.get_transform() is fig.transFigure:
                    text.set_x(text.get_position()[0] + dx / fig_w)
            ink = _ink_mask(fig)
            h_px = ink.shape[0]

    rows = np.where(ink.any(axis=1))[0]
    if rows.size == 0:
        return _save_canvas(fig, path, fig_w, fig_h)
    return savefig(
        fig,
        path,
        bbox_inches=Bbox.from_extents(
            0.0,
            (1.0 - (rows[-1] + 1) / h_px) * fig_h - v_margin,
            fig_w,
            (1.0 - rows[0] / h_px) * fig_h + v_margin,
        ),
        pad_inches=0.0,
    )


def _set_equal_3d(
    ax,
    pts: np.ndarray,
    pad: float = 0.05,
    *,
    zoom: float = 1.0,
) -> None:
    """Tight, undistorted 3D framing.

    Limits hug the data on each axis and the box aspect is proportional to
    those spans, so the surface fills the panel at unit scale instead of
    floating inside a cube sized by the longest axis. ``zoom`` removes the
    padding matplotlib leaves around the 3D box.
    """
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    half = np.maximum(0.5 * (maxs - mins), 1e-3) * (1.0 + pad)
    ax.set_xlim(centers[0] - half[0], centers[0] + half[0])
    ax.set_ylim(centers[1] - half[1], centers[1] + half[1])
    ax.set_zlim(centers[2] - half[2], centers[2] + half[2])
    aspect = tuple(half / half.max())
    ax.set_box_aspect(aspect, zoom=zoom)
    # Remembered so _autofit_3d can rescale without recomputing the framing.
    ax._gd_aspect = aspect
    ax._gd_zoom = zoom


def _ink_mask(fig) -> np.ndarray:
    """Boolean mask of pixels carrying drawn content."""
    fig.canvas.draw()
    return (np.asarray(fig.canvas.buffer_rgba())[..., :3] < 245).any(axis=-1)


def _apply_zoom(ax, zoom: float) -> None:
    ax.set_box_aspect(getattr(ax, "_gd_aspect", (1.0, 1.0, 1.0)), zoom=zoom)
    ax._gd_zoom = zoom


def _silhouette_center(fig, ax) -> Optional[tuple[float, float, float]]:
    """Ink-weighted centroid and bbox half-span, as fractions of the axes rect.

    Returns ``(cx, cy, reach)`` with ``c`` in ``[0, 1]²`` (x right, y up).
    The centroid is the centre of mass of drawn pixels, not the bounding-box
    centre: a bowl-shaped silhouette has most of its ink in the rim, so the
    box centre sits too low and the shape still reads as high in the panel.
    ``None`` if the axes is empty.
    """
    ink = _ink_mask(fig)
    h_px, w_px = ink.shape
    pos = ax.get_position()
    x0, x1 = int(round(pos.x0 * w_px)), int(round(pos.x1 * w_px))
    y0, y1 = int(round((1.0 - pos.y1) * h_px)), int(round((1.0 - pos.y0) * h_px))
    sub = ink[max(y0, 0) : y1, max(x0, 0) : x1]
    if sub.size == 0 or not sub.any():
        return None
    rows, cols = np.where(sub.any(axis=1))[0], np.where(sub.any(axis=0))[0]
    n_rows, n_cols = sub.shape
    ys, xs = np.where(sub)
    cx = float(xs.mean()) / n_cols
    cy = 1.0 - float(ys.mean()) / n_rows  # convert down-from-top to up-from-bottom
    reach = max(
        abs(cx - 0.5),
        abs(cy - 0.5),
        0.5 * (cols[-1] - cols[0] + 1) / n_cols,
        0.5 * (rows[-1] - rows[0] + 1) / n_rows,
    )
    return cx, cy, reach


def _pan_limits(ax, dx: float, dy: float, dz: float) -> None:
    """Translate the 3D box without changing its spans or aspect."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    ax.set_xlim(x0 + dx, x1 + dx)
    ax.set_ylim(y0 + dy, y1 + dy)
    ax.set_zlim(z0 + dz, z1 + dz)


def _center_3d(fig, ax, *, steps: int = 3) -> None:
    """Pan the 3D box so the projected silhouette sits at the panel centre.

    Matplotlib zooms about the box centre, not the silhouette, so an off-centre
    shape (the paraboloid under this view) cannot be recentred by moving the
    axes. Translating the data limits is a true camera pan.
    """
    loc = _silhouette_center(fig, ax)
    if loc is None:
        return
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
    frac = 0.04
    jacobian = np.zeros((2, 3))
    for j, (dx, dy, dz) in enumerate(
        (
            (frac * x_span, 0.0, 0.0),
            (0.0, frac * y_span, 0.0),
            (0.0, 0.0, frac * z_span),
        )
    ):
        _pan_limits(ax, dx, dy, dz)
        moved = _silhouette_center(fig, ax)
        _pan_limits(ax, -dx, -dy, -dz)
        if moved is None:
            continue
        jacobian[0, j] = (moved[0] - loc[0]) / frac
        jacobian[1, j] = (moved[1] - loc[1]) / frac

    # Screen offset we still need, in axes-fraction. Least-squares in the
    # three data axes because a 3D pan is underdetermined in 2D.
    for _ in range(steps):
        loc = _silhouette_center(fig, ax)
        if loc is None:
            return
        delta = np.array([0.5 - loc[0], 0.5 - loc[1]])
        if float(np.linalg.norm(delta)) < 5e-3:
            return
        try:
            shift, *_ = np.linalg.lstsq(jacobian, delta, rcond=None)
        except np.linalg.LinAlgError:
            return
        _pan_limits(ax, shift[0] * x_span, shift[1] * y_span, shift[2] * z_span)


def _autofit_3d(
    fig,
    groups: list[list],
    *,
    target: float = 0.90,
    probe: float = 0.50,
) -> None:
    """Centre each silhouette in its panel, then scale it to fill the panel.

    Content is measured at a small zoom first so a shape that currently
    overflows cannot report itself as already filling the rect. Groups share
    a zoom factor so compared panels stay at a common scale; each panel is
    panned independently.
    """
    flat = [ax for group in groups for ax in group]
    if not flat:
        return
    bases = {ax: getattr(ax, "_gd_zoom", 1.0) for ax in flat}
    for ax in flat:
        _apply_zoom(ax, bases[ax] * probe)
        _center_3d(fig, ax)

    locs = {ax: _silhouette_center(fig, ax) for ax in flat}
    for group in groups:
        reaches = [locs[ax][2] for ax in group if locs[ax] is not None]
        if not reaches:
            continue
        scale = (0.5 * target) / max(reaches)
        for ax in group:
            _apply_zoom(ax, bases[ax] * probe * scale)
            _center_3d(fig, ax, steps=2)
        # Panning after the zoom can push a rim past the rect; shrink the
        # group together if any panel now overflows.
        overflow = []
        for ax in group:
            loc = _silhouette_center(fig, ax)
            if loc is not None and loc[2] > 0.49:
                overflow.append(0.49 / loc[2])
        if overflow:
            shrink = min(overflow)
            for ax in group:
                _apply_zoom(ax, getattr(ax, "_gd_zoom", 1.0) * shrink)


def _boost_gallery_panel(fig, ax, boost: float) -> None:
    """Apply an extra per-manifold zoom after the shared gallery autofit."""
    if boost == 1.0:
        return
    _apply_zoom(ax, getattr(ax, "_gd_zoom", 1.0) * boost)
    _center_3d(fig, ax, steps=2)
    loc = _silhouette_center(fig, ax)
    if loc is not None and loc[2] > 0.49:
        _apply_zoom(ax, getattr(ax, "_gd_zoom", 1.0) * (0.49 / loc[2]))
        _center_3d(fig, ax, steps=2)


def _content_bounds_inches(
    fig,
    axes: list,
    fig_w: float,
    fig_h: float,
) -> tuple[float, float]:
    """Vertical extent of drawn 3D content, in figure inches from the bottom.

    Neighbouring 3D axes and titles are hidden for the measurement: a 3D
    projection routinely draws a few pixels past its rectangle, and using
    the clipped rect would place labels on top of that overflow.
    """
    hide_ax = [a for a in fig.axes if a not in axes]
    hide_tx = list(fig.texts)
    for a in hide_ax:
        a.set_visible(False)
    for t in hide_tx:
        t.set_visible(False)
    ink = _ink_mask(fig)
    for a in hide_ax:
        a.set_visible(True)
    for t in hide_tx:
        t.set_visible(True)
    if not ink.any():
        return 0.0, fig_h
    rows = np.where(ink.any(axis=1))[0]
    h_px = ink.shape[0]
    bottom = (1.0 - (rows[-1] + 1) / h_px) * fig_h
    top = (1.0 - rows[0] / h_px) * fig_h
    return bottom, top


def _content_boxes_inches(
    fig,
    groups: list[list],
    fig_w: float,
    fig_h: float,
) -> list[Optional[tuple[float, float, float, float]]]:
    """Per-group content boxes ``(x0, x1, y0, y1)`` in figure inches.

    Groups are measured one at a time with the others hidden, because a 3D
    axes draws outside its own rectangle and neighbouring panels otherwise
    contaminate the measurement.
    """
    flat = [ax for group in groups for ax in group]
    texts = [t for t in fig.texts if t.get_visible()]
    for t in texts:
        t.set_visible(False)

    boxes: list[Optional[tuple[float, float, float, float]]] = []
    for group in groups:
        for ax in flat:
            ax.set_visible(ax in group)
        ink = _ink_mask(fig)
        h_px, w_px = ink.shape
        rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
        if rows.size == 0:
            boxes.append(None)
            continue
        boxes.append(
            (
                cols[0] / w_px * fig_w,
                (cols[-1] + 1) / w_px * fig_w,
                (1.0 - (rows[-1] + 1) / h_px) * fig_h,
                (1.0 - rows[0] / h_px) * fig_h,
            )
        )

    for ax in flat:
        ax.set_visible(True)
    for t in texts:
        t.set_visible(True)
    return boxes


def _distribute_columns(
    fig,
    columns: list[list],
    fig_w: float,
    fig_h: float,
    *,
    left: float = 0.10,
    right: float = 0.10,
) -> list[float]:
    """Space columns by their drawn content, and return the new left edges.

    Panels sit on a grid of equal rectangles, but a 3D shape rarely fills its
    slot, so equal rectangles leave visibly unequal gaps and a lopsided outer
    margin. Distributing the measured content gives even optical spacing, and
    the returned edges let titles sit against the shape rather than the slot.
    """
    boxes = _content_boxes_inches(fig, columns, fig_w, fig_h)
    if any(b is None for b in boxes):
        return [b[0] if b else 0.0 for b in boxes]

    widths = [b[1] - b[0] for b in boxes]
    free = fig_w - left - right - sum(widths)
    gap = free / (len(columns) - 1) if len(columns) > 1 else 0.0
    if gap < 0.0:  # content wider than the canvas: centre it instead
        gap, left = 0.0, 0.5 * (fig_w - sum(widths))

    lefts, x = [], left
    for group, box, w in zip(columns, boxes, widths):
        dx = (x - box[0]) / fig_w
        for ax in group:
            pos = ax.get_position()
            ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])
        lefts.append(x)
        x += w + gap
    return lefts


def _great_circle(x0: np.ndarray, x1: np.ndarray, n_t: int = 80) -> np.ndarray:
    """Spherical linear interpolation on the unit sphere."""
    t = np.linspace(0.0, 1.0, n_t)
    x0 = x0 / np.linalg.norm(x0)
    x1 = x1 / np.linalg.norm(x1)
    dot = float(np.clip(np.dot(x0, x1), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-6:
        return np.outer(np.ones_like(t), x0)
    return (
        np.sin((1 - t)[:, None] * omega) * x0 + np.sin(t[:, None] * omega) * x1
    ) / np.sin(omega)


def _chart_path(
    m: Manifold,
    uv0: np.ndarray,
    uv1: np.ndarray,
    n_t: int = 80,
) -> np.ndarray:
    """Straight chart segment mapped through the parametrisation."""
    t = np.linspace(0.0, 1.0, n_t)[:, None]
    uv = (1.0 - t) * uv0[None, :] + t * uv1[None, :]
    with torch.no_grad():
        return m.parametrization(
            torch.from_numpy(uv[:, 0].astype(np.float32)),
            torch.from_numpy(uv[:, 1].astype(np.float32)),
        ).numpy()


def _embed_uv(m: Manifold, uv: np.ndarray) -> Optional[np.ndarray]:
    """Map a (n, 2) chart polyline through the parametrisation, or None."""
    u = torch.from_numpy(uv[:, 0].astype(np.float32))
    v = torch.from_numpy(uv[:, 1].astype(np.float32))
    if not bool(m.boundary_mask(u, v).all()):
        return None
    with torch.no_grad():
        pts = m.parametrization(u, v).numpy()
    if m.name == "hemisphere":
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        if np.any(pts[:, 2] < 0.2):
            return None
    return pts


def _polar_chart_path(
    m: Manifold,
    r0: float,
    phi0: float,
    r1: float,
    phi1: float,
    n_t: int,
) -> Optional[np.ndarray]:
    """Interpolate in polar chart coords so the path follows the graph.

    A straight (u, v) chord on the paraboloid dives toward the axis and
    reads as if it has left the bowl. Level sets (const r) and meridians
    (const φ) stay on the surface the viewer sees.
    """
    t = np.linspace(0.0, 1.0, n_t)
    r = (1.0 - t) * r0 + t * r1
    phi = (1.0 - t) * phi0 + t * phi1
    uv = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    pts = _embed_uv(m, uv)
    return pts if pts is not None and len(pts) >= 8 else None


def _clip_curve_to_patch(m: Manifold, curve: np.ndarray) -> Optional[np.ndarray]:
    """Keep the longest in-domain run so a great circle can sit on the drawn patch."""
    uv = m.inverse(
        torch.from_numpy(curve[:, 0].astype(np.float32)),
        torch.from_numpy(curve[:, 1].astype(np.float32)),
        torch.from_numpy(curve[:, 2].astype(np.float32)),
    )
    keep = np.asarray(m.boundary_mask(uv[:, 0], uv[:, 1]))
    if m.name == "hemisphere":
        keep = keep & (curve[:, 2] >= 0.18)
    best: Optional[np.ndarray] = None
    i = 0
    n = len(curve)
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i + 1
        while j < n and keep[j]:
            j += 1
        if best is None or (j - i) > len(best):
            best = curve[i:j]
        i = j
    return best if best is not None and len(best) >= 8 else None


def _center_chart_path(
    m: Manifold,
    uv_c: np.ndarray,
    theta: float,
    half_len: float,
    n_t: int,
) -> Optional[np.ndarray]:
    d = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    uv = np.linspace(-half_len, half_len, n_t)[:, None] * d + np.asarray(
        uv_c, dtype=np.float64
    )
    if m.name == "hemisphere":
        p0 = _embed_uv(m, uv[:1])
        p1 = _embed_uv(m, uv[-1:])
        if p0 is None or p1 is None:
            return None
        return _clip_curve_to_patch(m, _great_circle(p0[0], p1[0], n_t))
    pts = _embed_uv(m, uv)
    return pts if pts is not None and len(pts) >= 8 else None


def _designed_geodesics(m: Manifold, n_t: int) -> list[np.ndarray]:
    """Locked chart-space arcs for stable paper figures (front face, spread)."""
    out: list[np.ndarray] = []
    if m.name == "hemisphere":
        specs = [
            ((0.28, -0.22), 1.10, 0.28),
            ((0.52, -0.42), 0.30, 0.32),
            ((-0.32, -0.20), 0.90, 0.28),
            ((0.08, -0.52), 1.80, 0.34),
            ((0.22, 0.38), 1.45, 0.24),
        ]
        for uv_c, theta, hl in specs:
            c = _center_chart_path(m, uv_c, theta, hl, n_t)
            if c is not None:
                out.append(c)
    elif m.name == "saddle":
        # Near valley + side ridges for the saddle camera (azim ≈ −75°).
        specs = [
            ((0.06, -0.88), 1.55, 0.46),
            ((0.72, -0.50), 0.40, 0.46),
            ((-0.72, -0.42), 1.00, 0.46),
            ((0.78, 0.20), 0.25, 0.44),
            ((-0.18, -1.08), 2.15, 0.38),
        ]
        for uv_c, theta, hl in specs:
            if (
                abs(uv_c[0]) > 0.92 * _SADDLE_U_HALF
                or abs(uv_c[1]) > 0.92 * _SADDLE_V_HALF
            ):
                continue
            c = _center_chart_path(m, uv_c, theta, hl, n_t)
            if c is not None:
                out.append(c)
    elif m.name == "paraboloid":
        # Intentionally irregular: mixed lengths and shapes. Specs 0–1 sit on
        # the near wall toward the camera (φ ≈ −0.75), separated left/right.
        specs = [
            (0.80, -1.60, 0.80, -1.05),  # short front-left level
            (0.50, -0.85, 1.45, -0.10),  # long front-right spiral climbing high
            (1.25, -2.60, 1.25, -1.90),  # left flank mid wrap
            (1.05, 0.35, 1.05, 1.25),  # right flank longer wrap
            (0.70, 2.05, 1.40, 2.55),  # far spiral, different character
        ]
        for r0, phi0, r1, phi1 in specs:
            c = _polar_chart_path(m, r0, phi0, r1, phi1, n_t)
            if c is not None:
                out.append(c)
    return out


def _plot_geodesic(
    ax,
    m: Manifold,
    curve: np.ndarray,
    *,
    color,
    lw: float,
    endpoints: bool = True,
    s: float = 16.0,
    ls: str = "-",
    alpha: float = 1.0,
    zorder: float = 10.0,
) -> None:
    """Draw a geodesic polyline exactly as given (no lift, cull, or chart clip)."""
    if curve.size == 0:
        return
    seg_d = _display_xyz(m, curve)
    ax.plot(
        seg_d[:, 0],
        seg_d[:, 1],
        seg_d[:, 2],
        color=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )
    if endpoints:
        ends_d = _display_xyz(m, np.vstack([curve[0], curve[-1]]))
        ax.scatter(
            ends_d[:, 0],
            ends_d[:, 1],
            ends_d[:, 2],
            c="k",
            s=s,
            depthshade=False,
            zorder=20,
        )


def _analytic_geodesic_curve(
    m: Manifold,
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    n_t: int = 80,
) -> np.ndarray:
    """Exact (hemisphere) or chart-analytic geodesic between on-manifold points."""
    if m.name == "hemisphere":
        curve = _great_circle(x0, x1, n_t=n_t)
        clipped = _clip_curve_to_patch(m, curve)
        return clipped if clipped is not None else curve
    uv0 = m.inverse(
        torch.tensor([x0[0]], dtype=torch.float32),
        torch.tensor([x0[1]], dtype=torch.float32),
        torch.tensor([x0[2]], dtype=torch.float32),
    ).numpy()[0]
    uv1 = m.inverse(
        torch.tensor([x1[0]], dtype=torch.float32),
        torch.tensor([x1[1]], dtype=torch.float32),
        torch.tensor([x1[2]], dtype=torch.float32),
    ).numpy()[0]
    return _chart_path(m, uv0, uv1, n_t=n_t)


def _dijkstra_geodesic_paths(
    x_background: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    k_neighbors: int,
) -> list[np.ndarray]:
    """Shortest k-NN graph path per pair; returns waypoint coordinates."""
    x = x_background.numpy().astype(np.float64)
    n_points = x.shape[0]
    tree = cKDTree(x)
    dists, idx = tree.query(x, k=k_neighbors + 1)
    rows = np.repeat(np.arange(n_points), k_neighbors)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel().astype(np.float64)

    paths: list[np.ndarray] = []
    for i in range(x0.shape[0]):
        a0 = x0[i].numpy().reshape(1, -1).astype(np.float32)
        a1 = x1[i].numpy().reshape(1, -1).astype(np.float32)
        d0, i0 = tree.query(a0, k=k_neighbors)
        d1, i1 = tree.query(a1, k=k_neighbors)
        d0 = np.asarray(d0).ravel().astype(np.float64)
        d1 = np.asarray(d1).ravel().astype(np.float64)
        r = np.concatenate(
            [
                rows,
                np.full(k_neighbors, n_points, dtype=np.int64),
                np.full(k_neighbors, n_points + 1, dtype=np.int64),
            ]
        )
        c = np.concatenate([cols, i0.ravel(), i1.ravel()])
        d = np.concatenate([data, d0, d1])
        graph = coo_matrix((d, (r, c)), shape=(n_points + 2, n_points + 2)).tocsr()
        _, pred = dijkstra(
            graph,
            directed=False,
            indices=n_points,
            return_predecessors=True,
        )
        node = n_points + 1
        seq: list[int] = []
        while node != n_points and node != -9999 and len(seq) < n_points + 2:
            seq.append(node)
            node = int(pred[node])
        seq.append(n_points)
        seq = seq[::-1]
        waypoints = np.empty((len(seq), x.shape[1]))
        for j, nid in enumerate(seq):
            if nid == n_points:
                waypoints[j] = a0
            elif nid == n_points + 1:
                waypoints[j] = a1
            else:
                waypoints[j] = x[nid]
        paths.append(waypoints)
    return paths


def _geodesic_gt_curves(
    m: Manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    n_t: int = 80,
    n_background: int = 100_000,
    k_neighbors: int = 15,
    seed: int = 0,
) -> list[np.ndarray]:
    """Ground-truth geodesic polylines for overlay (analytic or k-NN Dijkstra)."""
    n = x0.shape[0]
    if m.name == "hemisphere":
        return [
            _analytic_geodesic_curve(m, x0[i].numpy(), x1[i].numpy(), n_t=n_t)
            for i in range(n)
        ]
    set_seed(seed)
    x_bg, _ = sample_uniform(m, n_points=n_background)
    return _dijkstra_geodesic_paths(x_bg, x0, x1, k_neighbors)


def _plot_geodesic_endpoints(
    ax,
    m: Manifold,
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    s: float = 16.0,
) -> None:
    """Endpoint markers at the stored pair coordinates."""
    ends_d = _display_xyz(m, np.vstack([x0, x1]))
    ax.scatter(
        ends_d[:, 0],
        ends_d[:, 1],
        ends_d[:, 2],
        c="k",
        s=s,
        depthshade=False,
        zorder=20,
    )


def _plot_predicted_geodesics_panel(
    ax,
    m: Manifold,
    paths: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    gt_paths: list[np.ndarray],
    *,
    gt_label: str,
    legend_fontsize: float = 6.5,
) -> None:
    """One 3D panel: predicted (solid) with ground truth (dashed) overlay."""
    ax.computed_zorder = False
    _plot_surface(
        ax,
        m,
        n_per_axis=80,
        alpha=0.92,
        zorder=0.5,
        rasterized=False,
        display_chart=True,
    )
    all_pts = [paths.reshape(-1, paths.shape[-1]), x0, x1]
    for i in range(paths.shape[0]):
        color = cm.tab10(i)
        _plot_geodesic(
            ax,
            m,
            gt_paths[i],
            color=color,
            lw=1.8,
            ls=(0, (4, 2)),
            alpha=0.85,
            zorder=10.0,
            endpoints=False,
        )
        _plot_geodesic(
            ax,
            m,
            paths[i],
            color=color,
            lw=2.2,
            s=16,
            ls="-",
            zorder=11.0,
            endpoints=False,
        )
        _plot_geodesic_endpoints(ax, m, x0[i], x1[i], s=16)
        all_pts.append(gt_paths[i])
    frame_ref, _ = sample_uniform(m, n_points=350)
    frame_ref = _clip_to_display_chart(m, frame_ref.numpy())
    frame_d = _frame_xyz(m, frame_ref, *all_pts)
    _set_equal_3d(ax, frame_d, pad=0.02)
    elev, azim = _view_for(m)
    _style_3d(
        ax, elev=elev, azim=azim, tick_labels=False, axis_labels=False, frame=False
    )
    ax.plot([], [], [], color=_INK, lw=2.2, ls="-", label="predicted")
    ax.plot([], [], [], color=_INK, lw=1.8, ls=(0, (4, 2)), label=gt_label)
    ax.legend(
        loc="upper left",
        fontsize=legend_fontsize,
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.2,
    )


def fig_chord_refine_geodesics_gallery(
    out_dir: Path,
    results_root: Path,
    run_id: str = "seed_42",
    *,
    n_show: int = 5,
    n_t_gt: int = 80,
    seed: int = 0,
    beta: float = 10.0,
    refine_steps: int = 1000,
    refine_lr: float = 3e-3,
    text_scale: float = 1.5,
    manifolds: Optional[list[str]] = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Three-panel chord + ambient refinement geodesics with GT overlay.

    Uses the designed paper endpoint set. Curves are the straight chord
    refined under the ambient projector metric (no curve network).
    """
    names = manifolds or list(MANIFOLDS)
    s = _GEO_PREDICTED_SCALE
    fig_w, fig_h = _COL_W * s, 2.82 * s
    fig = plt.figure(figsize=(fig_w, fig_h))
    col_w = 2.30 * s
    gap = 0.04 * s
    left = 0.05 * s
    ax_y, ax_h = 0.26 * s, col_w
    title_fs = 9.5 * text_scale
    legend_fs = 6.5 * text_scale
    title_letter_w = _TITLE_LETTER_W * text_scale

    groups: list[list] = []
    metrics_rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        proj_ckpt = projector_checkpoint(results_root, name, run_id)
        geod_dir = geodesics_dir(results_root, name, run_id)
        eval_path = geod_dir / "evaluation.json"
        if not proj_ckpt.is_file() or not eval_path.is_file():
            continue
        m = MANIFOLDS[name]
        projector = projector_from_checkpoint(
            torch.load(proj_ckpt, map_location="cpu", weights_only=False)
        )
        metrics = load_json(eval_path)
        refinement = metrics.get("refinement") or {}
        n_wp = int(refinement.get("n_waypoints", metrics.get("eval_M", 30) + 1))
        n_background = int(metrics.get("n_background", 100_000))
        k_neighbors = int(metrics.get("k_neighbors", 15))

        designed = _designed_geodesics(m, n_t_gt)[:n_show]
        if len(designed) < n_show:
            raise ValueError(
                f"Need {n_show} designed geodesics for {name}, got {len(designed)}."
            )
        x0 = torch.tensor(np.stack([c[0] for c in designed]), dtype=torch.float32)
        x1 = torch.tensor(np.stack([c[-1] for c in designed]), dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, n_wp)[None, :, None]
        chord = (1.0 - t) * x0[:, None, :] + t * x1[:, None, :]
        refined, _ = refine_paths(
            chord,
            projector,
            PathRefinementConfig(
                metric=WarpedPullbackMetricConfig(beta=beta, space="ambient"),
                steps=refine_steps,
                lr=refine_lr,
            ),
        )
        with torch.no_grad():
            interior = refined[:, 1:-1]
            refined[:, 1:-1] = projector(
                interior.reshape(-1, refined.shape[-1])
            ).reshape(refined.shape[0], -1, refined.shape[-1])
        paths = refined.detach().cpu().numpy()

        gt_label = "analytic GT" if m.name == "hemisphere" else r"$k$-NN GT"
        gt_paths = _geodesic_gt_curves(
            m,
            x0,
            x1,
            n_t=n_t_gt,
            n_background=n_background,
            k_neighbors=k_neighbors,
            seed=seed + 3 * i,
        )

        metrics_rows.append({"manifold": name, **metrics})
        x = left + len(groups) * (col_w + gap)
        ax = _inch_axes(fig, x, ax_y, col_w, ax_h, fig_w, fig_h, projection="3d")
        _plot_predicted_geodesics_panel(
            ax,
            m,
            paths,
            x0.numpy(),
            x1.numpy(),
            gt_paths,
            gt_label=gt_label,
            legend_fontsize=legend_fs,
        )
        groups.append([ax])

    if not groups:
        plt.close(fig)
        raise FileNotFoundError(
            f"No projection/geodesics artifacts for run_id={run_id!r} under {results_root}"
        )

    for group, metrics in zip(groups, metrics_rows):
        _autofit_3d(fig, [group], target=0.97)
        _boost_gallery_panel(
            fig,
            group[0],
            _GEO_GALLERY_PANEL_BOOST.get(metrics["manifold"], 1.0),
        )

    side_left, side_right = 0.05 * s, 0.18 * s
    lefts = _distribute_columns(
        fig,
        groups,
        fig_w,
        fig_h,
        left=side_left,
        right=side_right,
    )
    _, top = _content_bounds_inches(fig, [g[0] for g in groups], fig_w, fig_h)
    for j, (_, metrics) in enumerate(zip(groups, metrics_rows)):
        _fig_panel_title(
            fig,
            lefts[j],
            top + 0.10 * s,
            chr(ord("a") + j),
            _DISPLAY[metrics["manifold"]],
            fig_w,
            fig_h,
            fontsize=title_fs,
            letter_w=title_letter_w,
        )

    path = _save_balanced(
        fig,
        _topic_path(out_dir, _DIR_GEODESICS, "chord_refine_gallery"),
        fig_w,
        fig_h,
        margin=0.05,
        halign="left",
        vmargin=0.015,
    )
    return path, metrics_rows


_CHORD_REFINE_METHODS: tuple[tuple[str, str, str], ...] = (
    ("chord", "Chord", "Chord"),
    ("refine", "Chord+Refine", "Chord+Refine"),
)

_CHORD_REFINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("", "Length MSE", "Length MSE"),
    ("_dist", "Distance to true manifold", "Distance to True Manifold"),
)


def _path_distances_to_manifold(paths: torch.Tensor, manifold: Manifold) -> np.ndarray:
    """Per-path mean distance of the interior waypoints to the analytic manifold."""
    interior = paths[:, 1:-1].reshape(-1, paths.shape[-1])
    distances = manifold_distance(interior, manifold)
    return distances.reshape(paths.shape[0], -1).mean(axis=1)


def collect_chord_refine_table(
    results_root: Path,
    manifolds: list[str],
    run_ids: list[str],
) -> list[dict[str, Any]]:
    """Per-manifold mean ± sd of length MSE and waypoint distance to the manifold.

    ``run_ids`` is the seed sweep, one run id per seed shared by every manifold.
    """
    keys = ("chord", "refine", "chord_dist", "refine_dist")
    per_seed: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in keys} for name in manifolds
    }
    for name in manifolds:
        manifold = MANIFOLDS[name]
        for run_id in run_ids:
            run_dir = geodesics_dir(results_root, name, run_id)
            ev_path = run_dir / "evaluation.json"
            paths_path = run_dir / "paths.pt"
            if not ev_path.is_file():
                for key in keys:
                    per_seed[name][key].append(np.nan)
                continue
            ev = load_json(ev_path)
            refinement = ev.get("refinement") or {}
            per_seed[name]["chord"].append(float(ev.get("LengthMSE_straight", np.nan)))
            per_seed[name]["refine"].append(float(refinement.get("LengthMSE", np.nan)))
            if paths_path.is_file():
                payload = torch.load(paths_path, map_location="cpu", weights_only=False)
                per_seed[name]["chord_dist"].append(
                    float(
                        _path_distances_to_manifold(
                            payload["refined_chord"], manifold
                        ).mean()
                    )
                )
                per_seed[name]["refine_dist"].append(
                    float(
                        _path_distances_to_manifold(
                            payload["refined_proj"], manifold
                        ).mean()
                    )
                )
            else:
                per_seed[name]["chord_dist"].append(np.nan)
                per_seed[name]["refine_dist"].append(np.nan)

    def _summary(values: list[float]) -> tuple[float, float, int]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan"), 0
        return float(arr.mean()), float(arr.std()), int(arr.size)

    def _row(label: str, values: dict[str, list[float]]) -> dict[str, Any]:
        row: dict[str, Any] = {"manifold": label}
        for key in keys:
            mean, std, _ = _summary(values[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        row["n_seeds"] = _summary(values["chord"])[2]
        return row

    rows = [_row(name, per_seed[name]) for name in manifolds]
    averaged = {
        key: list(
            np.nanmean(
                np.array([per_seed[name][key] for name in manifolds], dtype=float),
                axis=0,
            )
        )
        for key in keys
    }
    rows.append(_row("average", averaged))
    return rows


def _chord_refine_label(row: dict[str, Any]) -> str:
    if row["manifold"] == "average":
        return "Average"
    return _DISPLAY.get(row["manifold"], row["manifold"].capitalize())


def _chord_refine_best(row: dict[str, Any], suffix: str) -> str:
    """Method key with the lower mean for one metric in this row."""
    means = {key: row[f"{key}{suffix}_mean"] for key, _, _ in _CHORD_REFINE_METHODS}
    finite = {k: v for k, v in means.items() if np.isfinite(v)}
    return min(finite, key=finite.get) if finite else ""


def _md_chord_refine_value(mean: float, std: float, *, distance: bool = False) -> str:
    if not np.isfinite(mean):
        return "—"
    if distance:
        return f"{mean:.4f} ± {std:.4f}"
    if abs(mean) < 1e-2:
        return f"{mean:.1e} ± {std:.1e}"
    return f"{mean:.3f} ± {std:.3f}"


def _tex_sci(value: float, digits: int = 1) -> str:
    if not np.isfinite(value) or value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits}f}{{\times}}10^{{{exponent}}}"


def _tex_chord_refine_body(mean: float, std: float, *, distance: bool = False) -> str:
    if not np.isfinite(mean):
        return "\u2014"
    if distance:
        return rf"{mean:.4f} \pm {std:.4f}"
    if abs(mean) < 1e-2:
        return rf"{_tex_sci(mean)} \pm {_tex_sci(std)}"
    return rf"{mean:.3f} \pm {std:.3f}"


def format_chord_refine_table_markdown(rows: list[dict[str, Any]]) -> str:
    """GitHub-flavored markdown table with the best method per metric bold."""
    header = ["Manifold"]
    for _, label, _ in _CHORD_REFINE_METRICS:
        header += [f"{label} — {method}" for _, method, _ in _CHORD_REFINE_METHODS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [_chord_refine_label(row)]
        for suffix, _, _ in _CHORD_REFINE_METRICS:
            best = _chord_refine_best(row, suffix)
            for key, _, _ in _CHORD_REFINE_METHODS:
                text = _md_chord_refine_value(
                    row[f"{key}{suffix}_mean"],
                    row[f"{key}{suffix}_std"],
                    distance=bool(suffix),
                )
                cells.append(f"**{text}**" if key == best else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_chord_refine_table_latex(
    rows: list[dict[str, Any]],
    *,
    caption: Optional[str] = None,
    label: str = "tab:chord-refine",
) -> str:
    """Booktabs table with one column block per metric."""
    if caption is None:
        n_seeds = next((r["n_seeds"] for r in rows if r["manifold"] != "average"), 0)
        caption = (
            r"\textbf{Geodesic length MSE and mean distance of the interior "
            "waypoints to the true manifold} (mean $\pm$ sd over "
            f"{n_seeds} seeds) for the straight chord and chord-started "
            "refinement under the ambient projector metric."
        )
    n_methods = len(_CHORD_REFINE_METHODS)
    colspec = "l" + "c" * n_methods
    for _ in range(len(_CHORD_REFINE_METRICS) - 1):
        colspec += "c" + "c" * n_methods
    group_headers, cmidrules = [], []
    for i, (_, _, tex_label) in enumerate(_CHORD_REFINE_METRICS):
        start = 2 + i * (n_methods + 1)
        end = start + n_methods - 1
        group_headers.append(rf"\multicolumn{{{n_methods}}}{{c}}{{{tex_label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")

    def join_blocks(blocks: list[list[str]], first: str) -> str:
        cells = [first]
        for i, block in enumerate(blocks):
            if i > 0:
                cells.append("")
            cells.extend(block)
        return " & ".join(cells) + r" \\"

    method_headers = [tex for _, _, tex in _CHORD_REFINE_METHODS]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        join_blocks([[header] for header in group_headers], "").lstrip(),
        " ".join(cmidrules),
        join_blocks(
            [list(method_headers) for _ in _CHORD_REFINE_METRICS],
            r"\textbf{Manifold}",
        ),
        r"\midrule",
    ]
    for row in rows:
        blocks: list[list[str]] = []
        for suffix, _, _ in _CHORD_REFINE_METRICS:
            best = _chord_refine_best(row, suffix)
            cells = []
            for key, _, _ in _CHORD_REFINE_METHODS:
                body = _tex_chord_refine_body(
                    row[f"{key}{suffix}_mean"],
                    row[f"{key}{suffix}_std"],
                    distance=bool(suffix),
                )
                if key == best:
                    body = rf"\mathbf{{{body}}}"
                cells.append(f"${body}$")
            blocks.append(cells)
        lines.append(join_blocks(blocks, _chord_refine_label(row)))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
