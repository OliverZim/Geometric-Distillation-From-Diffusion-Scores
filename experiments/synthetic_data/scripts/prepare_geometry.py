#!/usr/bin/env python
"""Build the geometry artifacts for the synthetic scRNA-seq data."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from experiments.synthetic_data.common.clean import (
    build_clean_tree,
    load_clean_working_spaces,
    save_clean_tree,
)
from experiments.synthetic_data.common.space import (
    build_ground_truth,
    load_geometry,
    make_bins_and_roots,
    make_splits,
    make_variants,
    prepare_working_space,
)
from geometry_distill.config import ExperimentConfig
from geometry_distill.seeding import set_seed

SCENARIOS = ("linear", "bifurcation", "tree")


class GeometryConfig(ExperimentConfig):
    """Configuration for the geometry preparation script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    task: Literal[
        "space",
        "splits",
        "variants",
        "clean",
        "ground_truth",
        "all",
    ] = Field(
        default="all",
        description="Which artifact group to build.",
    )
    scenario: Literal["all", "linear", "bifurcation", "tree"] = Field(
        default="all",
        description="Which scenario to build ('all' expands to every simulation).",
    )
    n_pcs: int = Field(
        default=50,
        ge=1,
        description="Number of principal components.",
    )
    k_values: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(10, 15, 30),
        description="kNN k values for the ground-truth graphs.",
    )
    max_cells: int = Field(
        default=4000,
        ge=0,
        description="Subsample for the bulk all-pairs Dijkstra (0 = all cells).",
    )
    clean_chunk: int = Field(
        default=1000,
        ge=1,
        description="Gene chunk size for streaming the clean matrices.",
    )
    seed: int = Field(
        default=42,
        description="Random seed.",
    )
    data_root: Path = Field(
        default=Path("experiments/synthetic_data/synthetic/data"),
        description="Directory with the simulation outputs.",
    )
    out_root: Path | None = Field(
        default=None,
        description="Output directory (default: data_root).",
    )

    @field_validator("k_values", mode="before")
    @classmethod
    def _parse_k_values(cls, value):
        if isinstance(value, str):
            return tuple(int(part) for part in value.split(","))
        if isinstance(value, list):
            return tuple(sorted({int(v) for v in value}))
        return value

    @property
    def scenarios(self) -> tuple[str, ...]:
        return SCENARIOS if self.scenario == "all" else (self.scenario,)

    @property
    def resolved_out_root(self) -> Path:
        return self.out_root or self.data_root


def _geometry_dir(config: GeometryConfig, scenario: str) -> Path:
    return config.resolved_out_root / scenario / "geometry"


def run_space(config: GeometryConfig, scenario: str) -> None:
    """Build the PCA working space, labels, splits, bins and roots."""
    out_dir = _geometry_dir(config, scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {scenario}: space (PCA-{config.n_pcs}) ===", flush=True)
    art = prepare_working_space(scenario, config.data_root, config.n_pcs, config.seed)
    X_pca, labels, cell_index, pca = (
        art["X_pca"],
        art["labels"],
        art["cell_index"],
        art["pca"],
    )
    print(
        f"  {X_pca.shape[0]} cells x {X_pca.shape[1]} PCs "
        f"(kept {len(cell_index)} of 20000, var {pca.explained_variance_ratio_.sum():.3f})",
        flush=True,
    )

    np.savez(
        out_dir / "pca_fit.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio_=pca.explained_variance_ratio_,
        explained_variance_=pca.explained_variance_,
        n_pcs=pca.n_components_,
        seed=config.seed,
    )
    np.save(out_dir / "X_pca.npy", X_pca)
    labels_out = labels.copy()
    labels_out["cell_index"] = cell_index
    labels_out.to_csv(out_dir / "labels_aligned.csv", index=False)

    train_idx, val_idx = make_splits(labels, config.seed)
    bin_id, root_idx = make_bins_and_roots(labels)
    np.savez(out_dir / "split.npz", train_idx=train_idx, val_idx=val_idx)
    np.save(out_dir / "bins.npy", bin_id)
    np.save(out_dir / "roots.npy", root_idx)
    print(f"  train {len(train_idx)} / val {len(val_idx)} cells, {len(root_idx)} roots")
    print(
        f"  wrote {out_dir}/{{pca_fit.npz, X_pca.npy, labels_aligned.csv, "
        "split.npz, bins.npy, roots.npy}",
        flush=True,
    )


def run_splits(config: GeometryConfig, scenario: str) -> None:
    """Rebuild the splits, bins and roots from the stored labels."""
    out_dir = _geometry_dir(config, scenario)
    print(f"\n=== {scenario}: splits / bins / roots ===", flush=True)
    labels = pd.read_csv(out_dir / "labels_aligned.csv")
    train_idx, val_idx = make_splits(labels, config.seed)
    bin_id, root_idx = make_bins_and_roots(labels)
    np.savez(out_dir / "split.npz", train_idx=train_idx, val_idx=val_idx)
    np.save(out_dir / "bins.npy", bin_id)
    np.save(out_dir / "roots.npy", root_idx)
    print(f"  train {len(train_idx)} / val {len(val_idx)} cells, {len(root_idx)} roots")
    print(f"  wrote {out_dir}/{{split.npz, bins.npy, roots.npy}}", flush=True)


def run_variants(config: GeometryConfig, scenario: str) -> None:
    """Build the data variant masks."""
    out_dir = _geometry_dir(config, scenario)
    print(f"\n=== {scenario}: variants ===", flush=True)
    split = np.load(out_dir / "split.npz")
    bin_id = np.load(out_dir / "bins.npy")
    variants = make_variants(split["train_idx"], split["val_idx"], bin_id)
    np.savez(
        out_dir / "variants.npz",
        **{
            f"{name}/{part}": np.asarray(values)
            for name, v in variants.items()
            for part, values in v.items()
        },
    )
    for name, v in variants.items():
        print(
            f"  {name:<14} train {len(v['train']):>6}  val {len(v['val']):>6}  "
            f"held_out {len(v['held_out']):>6}",
            flush=True,
        )
    print(f"  wrote {out_dir}/variants.npz", flush=True)


def run_clean(config: GeometryConfig, scenario: str) -> None:
    """Embed the clean targets and build the exact tree ground truth."""
    out_dir = _geometry_dir(config, scenario)
    print(
        f"\n=== {scenario}: clean targets (PCA-{config.n_pcs} embeddings + tree) ===",
        flush=True,
    )
    geom = load_geometry(scenario, config.out_root or config.data_root)
    spaces = load_clean_working_spaces(
        scenario, config.data_root, geom, chunk=config.clean_chunk
    )
    for matrix, X in spaces.items():
        if X.shape != geom["X_pca"].shape:
            raise ValueError(
                f"{matrix} embedding shape {X.shape} does not match X_pca "
                f"{geom['X_pca'].shape}"
            )
    X_clean = spaces["base_cell_means"]
    tree = build_clean_tree(X_clean, geom["labels"])

    np.save(out_dir / "X_clean_pca.npy", X_clean)
    np.save(out_dir / "X_cell_means_pca.npy", spaces["cell_means"])
    save_clean_tree(tree, out_dir / "clean_tree.npz")

    attachments = ", ".join(
        f"{tree.path_names[p]}<-{tree.path_names[int(tree.parents[p])]}"
        for p in range(len(tree.path_names))
        if tree.parents[p] != -1
    )
    lengths = ", ".join(
        f"{name} {length:.4f}"
        for name, length in zip(tree.path_names, tree.lengths, strict=True)
    )
    print(
        f"  {X_clean.shape[0]} clean points x {X_clean.shape[1]} PCs "
        "(row-aligned with X_pca.npy) for both embeddings",
        flush=True,
    )
    print(f"  tree: {len(tree.path_names)} paths; {attachments}", flush=True)
    print(f"  path lengths: {lengths}", flush=True)
    print(
        f"  junction max dist {tree.junction_max_dist:.2e}, "
        f"step spread max {tree.step_spread_max:.2e}",
        flush=True,
    )
    print(
        f"  wrote {out_dir}/{{X_clean_pca.npy, X_cell_means_pca.npy, clean_tree.npz}}",
        flush=True,
    )


def run_ground_truth(config: GeometryConfig, scenario: str) -> None:
    """Build the kNN graphs and the bulk all-pairs Dijkstra ground truth."""
    out_dir = _geometry_dir(config, scenario)
    print(f"\n=== {scenario}: ground truth ===", flush=True)
    X_pca = np.load(out_dir / "X_pca.npy")
    info = build_ground_truth(
        X_pca, out_dir, config.k_values, config.max_cells, config.seed
    )
    print(f"  {info['n_cells']} cells; dijkstra subsample {info['n_subset']}")
    for k, ncomp in info["connectivity"].items():
        print(f"  k={k}: {ncomp} connected component(s)", flush=True)
    print(
        f"  wrote {out_dir}/{{knn_k*.npz, dijkstra_subset_{config.max_cells}_k*.npz}}",
        flush=True,
    )


def run_scenario(config: GeometryConfig, scenario: str) -> None:
    """Run the configured tasks for one scenario and write its config file."""
    task = config.task
    if task in ("space", "all"):
        run_space(config, scenario)
    if task in ("splits", "all"):
        run_splits(config, scenario)
    if task in ("variants", "all"):
        run_variants(config, scenario)
    if task in ("clean", "all"):
        run_clean(config, scenario)
    if task in ("ground_truth", "all"):
        run_ground_truth(config, scenario)
    out_dir = _geometry_dir(config, scenario)
    (out_dir / "config.json").write_text(config.to_json() + "\n")
    print(f"  wrote {out_dir}/config.json", flush=True)


def main() -> None:
    config = GeometryConfig()
    config = config.model_copy(update={"out_root": config.resolved_out_root})
    set_seed(config.seed)
    for scenario in config.scenarios:
        run_scenario(config, scenario)


if __name__ == "__main__":
    main()
