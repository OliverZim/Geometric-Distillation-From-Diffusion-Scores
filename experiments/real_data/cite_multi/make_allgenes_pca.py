#!/usr/bin/env python
"""Recompute the Cite/Multi PCA on all genes to match the SF2M/GAGA representation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from sklearn.decomposition import PCA

from geometry_distill.config import ExperimentConfig

HERE = Path(__file__).resolve().parent
SCALE_SUBSET = 800


class AllGenesPCAConfig(ExperimentConfig):
    """Configuration for the all-genes Cite/Multi PCA script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    src: Path = Field(
        default_factory=lambda: HERE / "data" / "processed",
        description="Directory with the donor-13176 h5ad files.",
    )
    out: Path = Field(
        default_factory=lambda: HERE / "data_allgenes" / "processed",
        description="Output directory for the all-genes PCA npz files.",
    )
    n_components: int = Field(
        default=100,
        ge=1,
        description="Number of principal components.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for the PCA solver.",
    )


def _obs_array(obs, key):
    """Read an obs column, handling both plain arrays and categorical groups."""
    node = obs[key]
    if isinstance(node, h5py.Group):
        cats = np.asarray(node["categories"][:])
        codes = np.asarray(node["codes"][:])
        return cats[codes]
    return np.asarray(node[:])


def main() -> None:
    config = AllGenesPCAConfig()
    config.out.mkdir(parents=True, exist_ok=True)
    for name in ("cite", "multi"):
        path = config.src / f"{name}_donor-13176.h5ad"
        with h5py.File(path, "r") as f:
            X = f["X"][:].astype(np.float32)
            day = _obs_array(f["obs"], "day")
            cell_id = _obs_array(f["obs"], "_index")
        print(
            f"{name}: {X.shape[0]} cells x {X.shape[1]} genes (ALL genes into PCA)",
            flush=True,
        )

        Z = PCA(
            n_components=config.n_components,
            svd_solver="randomized",
            random_state=config.seed,
        ).fit_transform(X.astype(np.float64))
        Z = Z.astype(np.float32)
        del X

        rng = np.random.default_rng(0)
        sub = Z[rng.choice(len(Z), min(SCALE_SUBSET, len(Z)), replace=False)][:, :50]
        d = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
        iu = np.triu_indices(len(sub), 1)
        print(
            f"  50-PC scale: |x| mean {np.linalg.norm(Z[:, :50], axis=1).mean():.2f}  "
            f"median pairwise {np.median(d[iu]):.2f}   "
            "(1000-HVG reference: 28.47 / 39.57)",
            flush=True,
        )

        days_sorted = sorted(set(np.asarray(day).tolist()))
        day_class = np.array(
            [days_sorted.index(v) for v in np.asarray(day)], dtype=np.int64
        )
        for d in (50, 100):
            np.savez(
                config.out / f"{name}_D-{d}_pca.npz",
                data=Z[:, :d],
                colors=day_class,
                day=np.asarray(day),
                cell_id=np.asarray(cell_id),
            )
            print(f"  wrote {name}_D-{d}_pca.npz", flush=True)


if __name__ == "__main__":
    main()
