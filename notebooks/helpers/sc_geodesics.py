"""Stage III geodesic results on the synthetic scRNA-seq benchmark."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .scenarios import SCENARIO_META
from .style import figure_path, savefig

DEFAULT_PATTERN: str = "geodesics-full-s*"


METHOD_LABELS: Mapping[str, str] = {
    "refine_proj": "refined + projected\n(ours)",
    "graph_noisy_k15": "$k$NN-15 Dijkstra\n(noisy data)",
}


SEED_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


METHOD_COLORS: tuple[str, ...] = ("C0", "C1", "C2", "C3")


DEFAULT_METHODS: tuple[str, ...] = ("refine_proj", "graph_noisy_k15")


def geodesics_dir(results_root: Path | str, scenario: str, run_id: str) -> Path:
    return Path(results_root) / scenario / run_id / "geodesics"


def has_geodesics_run(
    results_root: Path | str, scenario: str, run_id: str
) -> dict[str, bool]:
    """Which Stage III artifacts exist for ``scenario`` / ``run_id``."""
    run_dir = geodesics_dir(results_root, scenario, run_id)
    flags = {
        "evaluation": (run_dir / "evaluation.json").is_file(),
        "paths": (run_dir / "paths.pt").is_file(),
        "off_gt": (run_dir / "off_gt.npz").is_file(),
    }
    flags["complete"] = flags["evaluation"] and flags["paths"] and flags["off_gt"]
    return flags


def seed_label(run_id: str) -> str:
    """Trailing ``-s<seed>`` of a run id, else the run id itself."""
    match = re.search(r"-s(\d+)$", run_id)
    return match.group(1) if match else run_id


def find_geodesics_runs(
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
        if not has_geodesics_run(results_root, scenario, run_dir.name)["complete"]:
            continue
        runs.append((run_dir.name, seed_label(run_dir.name)))
    return sorted(
        runs, key=lambda pair: (int(pair[1]) if pair[1].isdigit() else 0, pair[1])
    )


def _resolve_runs(
    results_root: Path | str,
    scenario: str,
    run_ids: Sequence[str] | None,
    pattern: str,
) -> list[tuple[str, str]]:
    if run_ids is None:
        runs = find_geodesics_runs(results_root, scenario, pattern=pattern)
    else:
        runs = [(run_id, seed_label(run_id)) for run_id in run_ids]
    if not runs:
        raise FileNotFoundError(
            f"no Stage III runs for {scenario!r} (pattern {pattern!r}); "
            "expected evaluation.json + paths.pt + off_gt.npz under "
            f"{results_root}/{scenario}/<run_id>/geodesics/"
        )
    return runs


#: Estimator name -> ``paths.pt`` length key written by Stage III.
_LENGTH_KEYS: Mapping[str, str] = {
    "chord_raw": "L_chord_raw",
    "chord_proj": "L_chord_proj",
    "refine_raw": "L_ref_raw",
    "refine_proj": "L_ref_proj",
    "graph_noisy_k15": "L_graph_noisy_k15",
}


def load_length_errors(
    results_root: Path | str,
    scenario: str,
    runs: Sequence[tuple[str, str]],
    methods: Sequence[str],
) -> dict[str, list[np.ndarray]]:
    """Per-pair squared length error ``(L_method - L_tree)**2`` per method and run."""
    out: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    for run_id, _ in runs:
        payload = torch.load(
            geodesics_dir(results_root, scenario, run_id) / "paths.pt",
            map_location="cpu",
            weights_only=False,
        )
        gt = np.asarray(payload["L_tree"], dtype=float)
        for method in methods:
            key = _LENGTH_KEYS.get(method, f"L_{method}")
            if key not in payload:
                continue
            out[method].append((np.asarray(payload[key], dtype=float) - gt) ** 2)
    return out


def load_offgt_distances(
    results_root: Path | str,
    scenario: str,
    runs: Sequence[tuple[str, str]],
    methods: Sequence[str],
) -> dict[str, list[np.ndarray]]:
    """Per-pair distance of interior waypoints to the GT clean tree."""
    out: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    for run_id, _ in runs:
        with np.load(geodesics_dir(results_root, scenario, run_id) / "off_gt.npz") as z:
            for method in methods:
                out[method].append(np.asarray(z[f"mean__{method}"], dtype=float))
    return out


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


def _seed_boxplot(
    ax: plt.Axes,
    values: Mapping[str, Sequence[np.ndarray]],
    *,
    methods: Sequence[str],
    runs: Sequence[tuple[str, str]],
    ylabel: str | None,
    yscale: str = "linear",
    method_labels: Mapping[str, str] = METHOD_LABELS,
    legend: bool = True,
    xticklabels: bool = True,
) -> dict[str, Any]:
    """Pooled per-pair box per method + one marker shape per seed.

    Each box shows the per-pair distribution pooled over all seeds
    (whiskers 1.5 IQR, outliers hidden); the marker of a seed sits at that
    seed's mean, which is the aggregate reported in the comparison tables.
    """
    positions = np.arange(1, len(methods) + 1, dtype=float)
    pooled = [np.concatenate(values[method]) for method in methods]
    box = ax.boxplot(
        pooled,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", lw=1.0),
        whiskerprops=dict(color="black", lw=0.8),
        capprops=dict(color="black", lw=0.8),
    )
    for patch, method, color in zip(box["boxes"], methods, METHOD_COLORS):
        patch.set(facecolor=color, alpha=0.28, edgecolor="black", linewidth=0.8)

    offsets = np.linspace(-0.15, 0.15, len(runs))
    for k in range(len(runs)):
        xs = positions + offsets[k]
        ys = [float(np.mean(values[method][k])) for method in methods]
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=SEED_MARKERS[k % len(SEED_MARKERS)],
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            zorder=5,
        )

    ax.set_xticks(positions, [method_labels.get(m, m) for m in methods])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    if not xticklabels:
        ax.tick_params(labelbottom=False)
    if legend:
        ax.legend(
            _seed_handles(runs),
            [f"seed {label}" for _, label in runs],
            frameon=False,
            ncol=len(runs),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            handletextpad=0.3,
            columnspacing=1.1,
            borderaxespad=0.0,
        )
    ax.margins(x=0.22)

    return {
        "n_pairs_per_seed": int(np.concatenate(values[methods[0]]).size // len(runs)),
        "pooled_mean": {m: float(np.mean(pooled[i])) for i, m in enumerate(methods)},
        "pooled_median": {
            m: float(np.median(pooled[i])) for i, m in enumerate(methods)
        },
        "per_seed_mean": {
            m: [float(np.mean(values[m][k])) for k in range(len(runs))] for m in methods
        },
    }


def fig_length_mse_by_seed(
    out_dir: Path,
    results_root: Path | str,
    scenario: str,
    *,
    run_ids: Sequence[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
    methods: Sequence[str] = DEFAULT_METHODS,
    yscale: str = "linear",
) -> tuple[Path, dict[str, Any]]:
    """MSE of the estimated path length against the clean-tree GT.

    Box: per-pair squared error pooled over seeds; markers: per-seed mean
    squared error (the aggregate of the comparison tables).
    """
    runs = _resolve_runs(results_root, scenario, run_ids, pattern)
    values = load_length_errors(results_root, scenario, runs, methods)

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    info = _seed_boxplot(
        ax,
        values,
        methods=methods,
        runs=runs,
        ylabel="MSE vs. tree ground truth",
        yscale=yscale,
    )
    ax.set_title(
        f"{SCENARIO_META[scenario]['label']}\n"
        f"{len(runs)} seeds $\\times$ {info['n_pairs_per_seed']} pairs",
        fontsize=10,
    )
    fig.tight_layout()
    path = savefig(fig, figure_path(out_dir, scenario, "01_length_mse_by_seed"))
    return path, {"scenario": scenario, "runs": runs, "methods": list(methods), **info}


def fig_offgt_distance_by_seed(
    out_dir: Path,
    results_root: Path | str,
    scenario: str,
    *,
    run_ids: Sequence[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
    methods: Sequence[str] = DEFAULT_METHODS,
    yscale: str = "linear",
) -> tuple[Path, dict[str, Any]]:
    """Distance of the interior waypoints to the GT manifold (clean tree).

    Box: per-pair mean distance to the clean-tree polyline pooled over seeds;
    markers: per-seed mean (the aggregate of the off-GT comparison).
    """
    runs = _resolve_runs(results_root, scenario, run_ids, pattern)
    values = load_offgt_distances(results_root, scenario, runs, methods)

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    info = _seed_boxplot(
        ax,
        values,
        methods=methods,
        runs=runs,
        ylabel="distance to GT manifold (mean over interior)",
        yscale=yscale,
    )
    ax.set_title(
        f"{SCENARIO_META[scenario]['label']}\n"
        f"{len(runs)} seeds $\\times$ {info['n_pairs_per_seed']} pairs",
        fontsize=10,
    )
    fig.tight_layout()
    path = savefig(fig, figure_path(out_dir, scenario, "02_offgt_distance_by_seed"))
    return path, {"scenario": scenario, "runs": runs, "methods": list(methods), **info}


def fig_metrics_overview(
    out_dir: Path,
    results_root: Path | str,
    scenarios: Sequence[str] = ("linear", "bifurcation", "tree"),
    *,
    run_ids: Sequence[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> tuple[Path, dict[str, Any]]:
    """Combined paper figure: one row per metric, one column per scenario.

    Row 1 is the MSE of the estimated path length against the clean-tree GT
    (log scale); row 2 the distance of the interior waypoints to the GT
    manifold. Both rows share their y-axis across scenarios; the seed legend
    appears once below the grid and the method labels only on the bottom row.
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
        runs = _resolve_runs(results_root, scenario, run_ids, pattern)
        runs_per_scenario[scenario] = runs
        values = {
            "mse": load_length_errors(results_root, scenario, runs, methods),
            "offgt": load_offgt_distances(results_root, scenario, runs, methods),
        }
        for row, (key, ylabel, yscale) in enumerate(
            (
                ("mse", "Absolute error of the geodesic length", "log"),
                ("offgt", "Distance to the true manifold", "linear"),
            )
        ):
            ax = axes[row][col]
            panels[f"{key}:{scenario}"] = _seed_boxplot(
                ax,
                values[key],
                methods=methods,
                runs=runs,
                ylabel=ylabel if col == 0 else None,
                yscale=yscale,
                method_labels=METHOD_HEADERS,
                legend=False,
                xticklabels=row == 1,
            )
            if row == 0:
                ax.set_title(SCENARIO_META[scenario]["label"], fontsize=10)

    first_runs = runs_per_scenario[scenarios[0]]
    fig.legend(
        _seed_handles(first_runs),
        [f"seed {label}" for _, label in first_runs],
        frameon=False,
        ncol=len(first_runs),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        handletextpad=0.3,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    path = savefig(fig, figure_path(out_dir, "00_metrics_overview"))
    return path, {
        "scenarios": list(scenarios),
        "runs": runs_per_scenario,
        "methods": list(methods),
        "panels": panels,
    }


def collect_seed_metrics(
    results_root: Path | str,
    scenarios: Sequence[str],
    *,
    run_ids: Sequence[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> list[dict[str, Any]]:
    """One row per scenario x method: mean ± std over seeds.

    The per-seed aggregate is the quantity marked in the figures: the mean
    per-pair squared length error against the clean-tree GT, and the mean
    per-pair distance of the interior waypoints to the GT manifold. The
    reported std is the population std over seeds (``np.std``), matching
    ``compare_geodesic_runs.py``.
    """
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        runs = _resolve_runs(results_root, scenario, run_ids, pattern)
        errors = load_length_errors(results_root, scenario, runs, methods)
        distances = load_offgt_distances(results_root, scenario, runs, methods)
        for method in methods:
            mse = np.array([np.mean(values) for values in errors[method]])
            dist = np.array([np.mean(values) for values in distances[method]])
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_META[scenario]["label"],
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method).replace(
                        "\n", " "
                    ),
                    "n_seeds": len(runs),
                    "n_pairs_per_seed": int(errors[method][0].size),
                    "mse_mean": float(mse.mean()),
                    "mse_std": float(mse.std()),
                    "offgt_mean": float(dist.mean()),
                    "offgt_std": float(dist.std()),
                }
            )
    return rows


def _md_metric(mean: float, std: float, *, scale: float | None = None) -> str:
    if scale:
        return f"{mean / scale:.2f} ± {std / scale:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def _tex_metric(
    mean: float, std: float, *, scale: float | None = None, bold: bool = False
) -> str:
    """Scaled plain-decimal math; the unit lives in the block header."""
    body = (
        rf"{mean / scale:.2f} \pm {std / scale:.2f}"
        if scale
        else rf"{mean:.4f} \pm {std:.4f}"
    )
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


_TABLE_METRICS: tuple[tuple[str, str, str, float | None], ...] = (
    (
        "mse",
        "MSE geodesic lengths (×1e-3)",
        r"\textbf{MSE Geodesic Lengths ($\times 10^{-3}, \downarrow$)}",
        1e-3,
    ),
    (
        "offgt",
        "Distance to manifold",
        r"\textbf{Distance to Manifold ($\downarrow$)}",
        None,
    ),
)


METHOD_HEADERS: Mapping[str, str] = {
    "refine_proj": "Refine+project (ours)",
    "graph_noisy_k15": r"$k$NN-15",
}


def _scenario_blocks(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, str, dict[str, Mapping[str, Any]]]], list[str]]:
    """Group rows into ``(scenario, scenario_label, {method: row})`` blocks.

    Returns the blocks in row order together with the method order.
    """
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


def _default_caption(rows: Sequence[Mapping[str, Any]]) -> str:
    return (
        "Trajectory evaluation comparing our refine+project method against "
        "$k$NN-15. Lower values are better ($\\downarrow$); best performance "
        "per metric is highlighted in \\textbf{bold}."
    )


def _metric_value(row: Mapping[str, Any], key: str) -> tuple[float, float]:
    return float(row[f"{key}_mean"]), float(row[f"{key}_std"])


def format_seed_table_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Grid table: one row per scenario, one column per method and metric."""
    blocks, methods = _scenario_blocks(rows)
    header = ["Scenario"]
    for _, label, _, _ in _TABLE_METRICS:
        header += [f"{label} — {METHOD_HEADERS.get(m, m)}" for m in methods]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for _, scenario_label, table in blocks:
        cells = [scenario_label]
        for key, _, _, scale in _TABLE_METRICS:
            means = {m: _metric_value(table[m], key)[0] for m in methods}
            best = min(means.values())
            for method in methods:
                mean, std = _metric_value(table[method], key)
                text = _md_metric(mean, std, scale=scale)
                cells.append(f"**{text}**" if means[method] == best else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_seed_table_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    caption: str | None = None,
    label: str = "tab:geodesic-results",
) -> str:
    """Single booktabs tabular with one method block per metric.

    Blocks are separated by an empty spacer column (``l cc c cc`` for two
    methods); each block has a bold metric header spanning its method
    columns, the method names below it and one row per scenario.  The MSE
    block is displayed in units of :math:`10^{-3}` (the unit lives in the
    header) and the best value per scenario and metric is bold.
    """
    blocks, methods = _scenario_blocks(rows)
    n_methods = len(methods)
    colspec = "l" + "c" * n_methods + "c" + "c" * n_methods
    group_headers, cmidrules = [], []
    for i, (_, _, tex_label, _) in enumerate(_TABLE_METRICS):
        start = 2 + i * (n_methods + 1)
        end = start + n_methods - 1
        group_headers.append(rf"\multicolumn{{{n_methods}}}{{c}}{{{tex_label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")

    def join_blocks(block_cells: list[list[str]], first: str) -> str:
        cells = [first]
        for i, block in enumerate(block_cells):
            if i > 0:
                cells.append("")
            cells.extend(block)
        return " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        join_blocks([[header] for header in group_headers], "").lstrip(),
        " ".join(cmidrules),
        join_blocks(
            [[METHOD_HEADERS.get(m, m) for m in methods] for _ in _TABLE_METRICS],
            r"\textbf{Scenario}",
        ),
        r"\midrule",
    ]
    for _, scenario_label, table in blocks:
        block_cells: list[list[str]] = []
        for key, _, _, scale in _TABLE_METRICS:
            means = {m: _metric_value(table[m], key)[0] for m in methods}
            best = min(means.values())
            block_cells.append(
                [
                    _tex_metric(
                        *_metric_value(table[method], key),
                        scale=scale,
                        bold=means[method] == best,
                    )
                    for method in methods
                ]
            )
        lines.append(join_blocks(block_cells, scenario_label))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption or _default_caption(rows)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
