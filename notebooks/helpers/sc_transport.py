"""Population-transport results on the synthetic scRNA-seq benchmark."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .scenarios import SCENARIO_META
from .style import figure_path, savefig

DEFAULT_PREFIX: str = "transport-holdout"


DEFAULT_PATTERN: str = f"{DEFAULT_PREFIX}-s*"


DEFAULT_PROTOCOL: str = "pooled"


PROTOCOLS: tuple[str, ...] = ("pooled",)


PROTOCOL_LABELS: Mapping[str, str] = {"pooled": "pooled"}


DEFAULT_METHODS: tuple[str, ...] = (
    "bin_prev",
    "ot_midpoint",
    "geodesic_midpoint_projected",
)


METHOD_LABELS: Mapping[str, str] = {
    "bin_prev": "copy previous",
    "bin_next": "copy next",
    "ot_midpoint": "OT midpoint",
    "ot_midpoint_projected": "OT mid + proj",
    "geodesic_midpoint": "geodesic mid",
    "geodesic_midpoint_projected": "geodesic mid + proj (ours)",
}


METHOD_HEADERS: Mapping[str, str] = {
    "bin_prev": "Copy prev",
    "bin_next": "Copy next",
    "ot_midpoint": "OT mid",
    "ot_midpoint_projected": "OT mid+proj",
    "geodesic_midpoint": "Geo mid",
    "geodesic_midpoint_projected": "Geo mid+proj",
}


METHOD_COLORS: tuple[str, ...] = ("0.6", "C1", "C0", "C2", "C3", "C4")


METHOD_COLOR_MAP: Mapping[str, str] = {
    "bin_prev": "0.6",
    "bin_next": "0.45",
    "ot_midpoint": "C1",
    "ot_midpoint_projected": "C0",
    "geodesic_midpoint": "C2",
    "geodesic_midpoint_projected": "C2",
}


SEED_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


OVERVIEW_METHODS: tuple[str, ...] = (
    "bin_prev",
    "bin_next",
    "ot_midpoint",
    "geodesic_midpoint_projected",
)


OVERVIEW_PROTOCOL: str = DEFAULT_PROTOCOL


_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def transport_dir(results_root: Path | str, scenario: str, run_id: str) -> Path:
    return Path(results_root) / scenario / run_id / "transport"


def has_transport_run(
    results_root: Path | str, scenario: str, run_id: str
) -> dict[str, bool]:
    """Which transport artifacts exist for ``scenario`` / ``run_id``."""
    run_dir = transport_dir(results_root, scenario, run_id)
    flags = {
        "evaluation": (run_dir / "evaluation.json").is_file(),
        "paths": (run_dir / "paths.pt").is_file(),
    }
    flags["complete"] = flags["evaluation"]
    return flags


def seed_label(run_id: str) -> str:
    """``s<seed>`` component of a transport run id, else the run id."""
    match = re.search(r"-s(\d+)(?:-|$)", run_id)
    return match.group(1) if match else run_id


def find_transport_runs(
    results_root: Path | str,
    scenario: str,
    *,
    pattern: str = DEFAULT_PATTERN,
) -> list[tuple[str, str]]:
    """``(run_id, seed_label)`` for every complete run matching ``pattern``."""
    root = Path(results_root) / scenario
    runs: list[tuple[str, str]] = []
    for run_dir in sorted(root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        if not has_transport_run(results_root, scenario, run_dir.name)["complete"]:
            continue
        runs.append((run_dir.name, seed_label(run_dir.name)))
    return sorted(
        runs, key=lambda pair: (int(pair[1]) if pair[1].isdigit() else 0, pair[1])
    )


def find_protocol_runs(
    results_root: Path | str,
    scenario: str,
    protocol: str,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> list[tuple[str, str]]:
    """Runs whose stored evaluation reports ``protocol``."""
    runs = find_transport_runs(results_root, scenario, pattern=f"{prefix}-s*")
    return [
        (run_id, label)
        for run_id, label in runs
        if load_evaluation(results_root, scenario, run_id).get("protocol") == protocol
    ]


def _resolve_runs(
    results_root: Path | str,
    scenario: str,
    run_ids: Sequence[str] | None,
    pattern: str,
) -> list[tuple[str, str]]:
    if run_ids is None:
        runs = find_transport_runs(results_root, scenario, pattern=pattern)
    else:
        runs = [(run_id, seed_label(run_id)) for run_id in run_ids]
    if not runs:
        raise FileNotFoundError(
            f"no transport runs for {scenario!r} (pattern {pattern!r}); expected "
            f"evaluation.json under {results_root}/"
            f"{scenario}/<run_id>/transport/"
        )
    return runs


def load_evaluation(results_root: Path | str, scenario: str, run_id: str) -> dict:
    import json

    with open(transport_dir(results_root, scenario, run_id) / "evaluation.json") as f:
        return json.load(f)


def _metric_values(
    results_root: Path | str,
    scenario: str,
    runs: Sequence[tuple[str, str]],
    method: str,
    metric: str,
) -> list[float]:
    """Per-seed values of one metric, skipping runs without the method."""
    values: list[float] = []
    for run_id, _ in runs:
        evaluation = load_evaluation(results_root, scenario, run_id)
        row = evaluation.get("metrics", {}).get(method)
        if row is None or row.get(metric) is None:
            continue
        values.append(float(row[metric]))
    return values


def available_methods(
    results_root: Path | str,
    scenario: str,
    runs: Sequence[tuple[str, str]],
    methods: Sequence[str],
) -> tuple[str, ...]:
    """Methods present in *all* runs of the group (keeps bars comparable)."""
    if not runs:
        return ()
    present: list[str] = []
    evaluations = [
        load_evaluation(results_root, scenario, run_id) for run_id, _ in runs
    ]
    for method in methods:
        if all(method in evaluation.get("metrics", {}) for evaluation in evaluations):
            present.append(method)
    return tuple(present)


def _seed_handles(runs: Sequence[tuple[str, str]]) -> list[Line2D]:
    """Legend handles: one marker shape per seed."""
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=SEED_MARKERS[k % len(SEED_MARKERS)],
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
        )
        for k in range(len(runs))
    ]


def _ci95(values: Sequence[float]) -> float:
    """Half-width of the two-sided 95 % t-interval of the seed mean."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 2:
        return 0.0
    return float(_T975.get(n - 1, 1.959964) * vals.std(ddof=1) / np.sqrt(n))


def _seed_bars(
    ax: plt.Axes,
    values: Mapping[str, Sequence[float]],
    *,
    methods: Sequence[str],
    runs: Sequence[tuple[str, str]],
    ylabel: str | None,
    floor: float | None = None,
    floor_label: str = "sampling floor",
    log: bool = False,
    error: str = "std",
    show_seeds: bool = True,
    method_labels: Mapping[str, str] = METHOD_LABELS,
    method_colors: Sequence[str] = METHOD_COLORS,
    legend: bool = True,
    xticklabels: bool = True,
) -> dict[str, Any]:
    """Bar per method (mean over seeds) with an error bar per method.

    ``error="std"`` uses the population std over seeds; ``error="ci95"``
    the half-width of the two-sided 95 % t-interval.  ``show_seeds``
    overlays one white marker per seed (the run-level values).
    """
    if error not in ("std", "ci95"):
        raise ValueError(f"error must be 'std' or 'ci95', got {error!r}")
    positions = np.arange(1, len(methods) + 1, dtype=float)
    means = [float(np.mean(values[method])) for method in methods]
    stds = [float(np.std(values[method])) for method in methods]
    ci95 = [_ci95(values[method]) for method in methods]
    ax.bar(
        positions,
        means,
        yerr=stds if error == "std" else ci95,
        width=0.56,
        capsize=2.5,
        color=[
            METHOD_COLOR_MAP.get(m, method_colors[i % len(method_colors)])
            for i, m in enumerate(methods)
        ],
        alpha=0.35,
        edgecolor="black",
        linewidth=0.8,
        error_kw=dict(ecolor="black", lw=0.8, capthick=0.8),
        zorder=3,
    )
    if show_seeds:
        offsets = np.linspace(-0.14, 0.14, len(runs))
        for k in range(len(runs)):
            ax.plot(
                positions + offsets[k],
                [values[method][k] for method in methods],
                linestyle="none",
                marker=SEED_MARKERS[k % len(SEED_MARKERS)],
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=5,
            )
    if floor is not None:
        ax.axhline(
            floor,
            color="black",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
            label=floor_label,
        )
    ax.set_xticks(
        positions,
        [method_labels.get(m, m) for m in methods],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", labelsize=8.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    if not xticklabels:
        ax.tick_params(labelbottom=False)
    if legend and (show_seeds or floor is not None):
        handles = _seed_handles(runs) if show_seeds else []
        labels = [f"seed {label}" for _, label in runs] if show_seeds else []
        if floor is not None:
            handles.append(Line2D([], [], color="black", linestyle="--", lw=1.0))
            labels.append(floor_label)
        ax.legend(
            handles,
            labels,
            frameon=False,
            ncol=len(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            handletextpad=0.3,
            columnspacing=1.1,
            borderaxespad=0.0,
        )
    ax.margins(x=0.24)
    return {
        "n_seeds": len(runs),
        "mean": {m: means[i] for i, m in enumerate(methods)},
        "std": {m: stds[i] for i, m in enumerate(methods)},
        "ci95": {m: ci95[i] for i, m in enumerate(methods)},
        "per_seed": {
            m: [float(values[m][k]) for k in range(len(runs))] for m in methods
        },
    }


def fig_w1_by_method(
    out_dir: Path,
    results_root: Path | str,
    scenario: str,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    run_ids: Sequence[str] | None = None,
    pattern: str | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    log: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """W1 against the withheld clean population, per method.

    Bar = mean over seeds (std error bar); white markers = per-seed values;
    dashed line = within-population sampling floor.
    """
    runs = _resolve_runs(
        results_root,
        scenario,
        run_ids,
        pattern or DEFAULT_PATTERN,
    )
    methods = available_methods(results_root, scenario, runs, methods)
    values = {
        method: _metric_values(results_root, scenario, runs, method, "W1")
        for method in methods
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    info = _seed_bars(
        ax,
        values,
        methods=methods,
        runs=runs,
        ylabel="W1 vs. withheld population",
        log=log,
    )
    ax.set_title(
        f"{SCENARIO_META[scenario]['label']} — {PROTOCOL_LABELS.get(protocol, protocol)}\n"
        f"{len(runs)} seeds $\\times$ withheld cells",
        fontsize=10,
    )
    fig.tight_layout()
    path = savefig(fig, figure_path(out_dir, scenario, "01_w1_by_method"))
    return path, {
        "scenario": scenario,
        "protocol": protocol,
        "runs": runs,
        "methods": list(methods),
        **info,
    }


def fig_onmanifold_by_method(
    out_dir: Path,
    results_root: Path | str,
    scenario: str,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    run_ids: Sequence[str] | None = None,
    pattern: str | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> tuple[Path, dict[str, Any]]:
    """Two panels: off-manifold residual and NN distance to the clean backbone.

    The on-manifoldness claim, using the same metric as the geodesics
    off-GT analysis: the left panel is the exact distance of the predicted
    cells to the ground-truth clean-tree polyline, the right panel the
    projector residual ``||x - P(x)||`` (distance to the *learned*
    manifold) as the model-level companion.
    """
    runs = _resolve_runs(
        results_root,
        scenario,
        run_ids,
        pattern or DEFAULT_PATTERN,
    )
    methods = available_methods(results_root, scenario, runs, methods)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True)
    panels: dict[str, dict[str, Any]] = {}
    for ax, (metric, ylabel) in zip(
        axes,
        (
            ("gt_tree_dist_mean", "Distance to the true manifold"),
            ("offmanifold_resid_mean", "off-manifold $\\|x-P(x)\\|$ (learned)"),
        ),
    ):
        values = {
            method: _metric_values(results_root, scenario, runs, method, metric)
            for method in methods
        }
        panels[metric] = _seed_bars(
            ax,
            values,
            methods=methods,
            runs=runs,
            ylabel=ylabel,
            legend=metric == "gt_tree_dist_mean",
            xticklabels=True,
        )
    axes[0].set_title(
        f"{SCENARIO_META[scenario]['label']} — "
        f"{PROTOCOL_LABELS.get(protocol, protocol)}",
        fontsize=10,
    )
    fig.tight_layout()
    path = savefig(fig, figure_path(out_dir, scenario, "02_onmanifold_by_method"))
    return path, {
        "scenario": scenario,
        "protocol": protocol,
        "runs": runs,
        "methods": list(methods),
        "panels": panels,
    }


def fig_metrics_overview(
    out_dir: Path,
    results_root: Path | str,
    scenarios: Sequence[str] = ("linear", "bifurcation", "tree"),
    *,
    protocol: str = OVERVIEW_PROTOCOL,
    methods: Sequence[str] = OVERVIEW_METHODS,
    prefix: str = DEFAULT_PREFIX,
) -> tuple[Path, dict[str, Any]]:
    """Combined paper figure: one row per metric, one column per scenario.

    Row 1 W1, row 2 the exact distance to the GT clean-tree polyline; both
    against the withheld clean population.  Bars are seed means with 95 %
    t-interval error bars (no per-seed markers, no sampling-floor line);
    the y-axis is shared per row across scenarios.  The default protocol is
    :data:`OVERVIEW_PROTOCOL` (the primary pooled run set).
    """
    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=(3.35 * len(scenarios), 6.6),
        sharey="row",
        squeeze=False,
    )
    panels: dict[str, dict[str, Any]] = {}
    runs_per_scenario: dict[str, list[tuple[str, str]]] = {}
    for col, scenario in enumerate(scenarios):
        runs = find_protocol_runs(results_root, scenario, protocol, prefix=prefix)
        runs_per_scenario[scenario] = runs
        available = available_methods(results_root, scenario, runs, methods)
        for row, (metric, ylabel) in enumerate(
            (
                ("W1", "1-Wasserstein distance"),
                ("gt_tree_dist_mean", "Distance to the true manifold"),
            )
        ):
            ax = axes[row][col]
            values = {
                m: _metric_values(results_root, scenario, runs, m, metric)
                for m in available
            }
            panels[f"{metric}:{scenario}"] = _seed_bars(
                ax,
                values,
                methods=available,
                runs=runs,
                ylabel=ylabel if col == 0 else None,
                error="ci95",
                show_seeds=False,
                legend=False,
                xticklabels=True,
            )
            if row == 0:
                ax.set_title(SCENARIO_META[scenario]["label"], fontsize=10)

    # Shared y axes hide the numeric labels on the non-first columns; force
    # them back on so every panel carries its tick labels (shared values).
    for ax in axes.ravel():
        ax.tick_params(axis="y", labelleft=True)

    fig.tight_layout()
    path = savefig(fig, figure_path(out_dir, "00_metrics_overview"))
    return path, {
        "scenarios": list(scenarios),
        "protocol": protocol,
        "runs": runs_per_scenario,
        "panels": panels,
    }


def collect_seed_metrics(
    results_root: Path | str,
    scenarios: Sequence[str],
    *,
    protocol: str = DEFAULT_PROTOCOL,
    run_ids: Sequence[str] | None = None,
    pattern: str | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> list[dict[str, Any]]:
    """One row per scenario x method: mean ± std over seeds.

    The per-seed aggregate is the run-level metric (W1, exact distance to
    the GT clean tree, off-manifold residual, composition TV, mean per-path
    arc W1), reported as mean and population std over the seeds of the
    protocol.
    """
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        runs = _resolve_runs(
            results_root,
            scenario,
            run_ids,
            pattern or DEFAULT_PATTERN,
        )
        available = available_methods(results_root, scenario, runs, methods)
        for method in available:
            w1 = np.array(_metric_values(results_root, scenario, runs, method, "W1"))
            gt = np.array(
                _metric_values(
                    results_root, scenario, runs, method, "gt_tree_dist_mean"
                )
            )
            off = np.array(
                _metric_values(
                    results_root, scenario, runs, method, "offmanifold_resid_mean"
                )
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_META[scenario]["label"],
                    "protocol": protocol,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method).replace(
                        "\n", " "
                    ),
                    "n_seeds": len(runs),
                    "w1_mean": float(w1.mean()),
                    "w1_std": float(w1.std()),
                    "gt_tree_dist_mean": float(gt.mean()),
                    "gt_tree_dist_std": float(gt.std()),
                    "offmanifold_mean": float(off.mean()),
                    "offmanifold_std": float(off.std()),
                }
            )
    return rows


_TABLE_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("w1", "W1 vs. withheld", r"W1 vs.\ withheld", False),
    (
        "gt_tree_dist",
        "Distance to GT manifold",
        "Distance to GT manifold",
        False,
    ),
)


_TEX_BLOCK_LABELS: Mapping[str, str] = {
    "w1": r"\textit{(a) $W_1$ vs. withheld ($\downarrow$)}",
    "gt_tree_dist": (r"\textit{(b) Distance to GT manifold ($\downarrow$)}"),
}


def _scenario_blocks(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, str, dict[str, Mapping[str, Any]]]], list[str]]:
    order: list[str] = []
    labels: dict[str, str] = {}
    table: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        scenario = row["scenario"]
        if scenario not in table:
            order.append(scenario)
            labels[scenario] = row["scenario_label"]
            table[scenario] = {}
        table[scenario][row["method"]] = row
    methods = list(dict.fromkeys(row["method"] for row in rows))
    return [(s, labels[s], table[s]) for s in order], methods


def _fmt(mean: float, std: float, *, scientific: bool) -> str:
    if scientific:
        return f"{mean:.2e} ± {std:.1e}"
    return f"{mean:.4f} ± {std:.4f}"


def _tex(mean: float, std: float, *, scientific: bool, bold: bool) -> str:
    if scientific:
        exponent = int(np.floor(np.log10(abs(mean))))
        scale = 10.0**exponent
        body = rf"({mean / scale:.2f} \pm {std / scale:.2f})\times 10^{{{exponent}}}"
    else:
        body = rf"{mean:.4f} \pm {std:.4f}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def _default_caption(rows: Sequence[Mapping[str, Any]]) -> str:
    first = rows[0] if rows else {}
    n_seeds = first.get("n_seeds", 0)
    return (
        "Population transport on simulated scRNA-seq trajectories "
        f"(mean $\\pm$ std over {n_seeds} seeds against withheld clean "
        "CellMeans population, strict \\texttt{holdout\\_mid} target). "
        "$W_1$ is exact Euclidean earth-mover distance; manifold distance is "
        "exact Euclidean distance of predicted cells to ground-truth "
        "clean-tree polyline. Lower values are better ($\\downarrow$); best "
        "result per scenario is in \\textbf{bold}."
    )


def format_seed_table_html(
    rows: Sequence[Mapping[str, Any]], *, caption: str | None = None
) -> str:
    """Two side-by-side HTML tables (one per metric) with a spanning header.

    Each table has the metric name as a single header cell spanning the
    method columns, the method names below it, and one row per scenario.
    """
    blocks, methods = _scenario_blocks(rows)
    table_style = "border-collapse: collapse; margin-right: 2.5em;"
    cell_style = "text-align: right; padding: 0.1em 0.7em;"
    left_cell_style = "text-align: left; padding: 0.1em 0.7em;"
    head_style = "padding: 0.15em 0.7em; border-bottom: 1px solid #888;"
    parts = ['<div style="display: flex; align-items: flex-start; flex-wrap: wrap;">']
    for key, metric_label, _, scientific in _TABLE_METRICS:
        parts.append(f'<table style="{table_style}">')
        parts.append(
            "<thead>"
            "<tr><th></th>"
            f'<th colspan="{len(methods)}" style="{head_style} text-align: center;">'
            f"{metric_label}</th></tr>"
            "<tr>"
            f'<th style="{head_style} text-align: left;">Scenario</th>'
            + "".join(
                f'<th style="{head_style}">{METHOD_HEADERS.get(m, m)}</th>'
                for m in methods
            )
            + "</tr></thead><tbody>"
        )
        for _, scenario_label, table in blocks:
            means = {m: float(table[m][f"{key}_mean"]) for m in methods}
            best = min(means.values())
            cells = [f'<td style="{left_cell_style}">{scenario_label}</td>']
            for method in methods:
                mean = float(table[method][f"{key}_mean"])
                std = float(table[method][f"{key}_std"])
                text = _fmt(mean, std, scientific=scientific)
                if means[method] == best:
                    text = f"<b>{text}</b>"
                cells.append(f'<td style="{cell_style}">{text}</td>')
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</div>")
    if caption is not None:
        parts.append(f'<p style="font-size: 0.9em;">{caption}</p>')
    return "\n".join(parts)


def format_seed_table_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    caption: str | None = None,
    label: str = "tab:sc-transport",
) -> str:
    """Single booktabs tabular: method header + metric blocks as sections.

    One method-header row (``Method`` spanning the method columns), then one
    block per metric with a ``(a)`` / ``(b)`` section row and one row per
    scenario; best value per scenario and metric is bold.
    """
    blocks, methods = _scenario_blocks(rows)
    n_methods = len(methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{l *{{{n_methods}}}{{c}}}}",
        r"\toprule",
        rf"& \multicolumn{{{n_methods}}}{{c}}{{\textbf{{Method}}}} \\",
        rf"\cmidrule(lr){{2-{n_methods + 1}}}",
        r"\textbf{Scenario} & "
        + " & ".join(METHOD_HEADERS.get(m, m) for m in methods)
        + r" \\",
        r"\midrule",
    ]
    for i, (key, _, _, scientific) in enumerate(_TABLE_METRICS):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_methods + 1}}}{{l}}{{{_TEX_BLOCK_LABELS[key]}}} \\[2pt]"
        )
        for _, scenario_label, table in blocks:
            means = {m: float(table[m][f"{key}_mean"]) for m in methods}
            best = min(means.values())
            cells = [scenario_label]
            for method in methods:
                mean = float(table[method][f"{key}_mean"])
                std = float(table[method][f"{key}_std"])
                cells.append(
                    _tex(mean, std, scientific=scientific, bold=means[method] == best)
                )
            lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption or _default_caption(rows)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
