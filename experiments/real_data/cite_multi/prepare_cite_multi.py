#!/usr/bin/env python
"""Build the donor-13176 Cite/Multi datasets used by GAGA from the raw Kaggle files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import anndata as ad
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import scanpy as sc
from pydantic import Field, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from geometry_distill.config import ExperimentConfig

HERE = Path(__file__).resolve().parent
DONOR, DAYS = 13176, [2, 3, 4, 7]
N_HVG = 1000


class PrepareConfig(ExperimentConfig):
    """Configuration for the Cite/Multi dataset preparation script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    raw: Path = Field(
        default_factory=lambda: HERE / "data" / "raw",
        description="Directory with the Kaggle .h5 files and metadata.csv.",
    )
    out: Path = Field(
        default_factory=lambda: HERE / "data" / "processed",
        description="Output directory for the h5ad and PCA npz files.",
    )
    which: Annotated[tuple[Literal["cite", "multi"], ...], NoDecode] = Field(
        default=("cite", "multi"),
        description="Datasets to build.",
    )

    @field_validator("which", mode="before")
    @classmethod
    def _parse_which(cls, value):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value


def read_fixed_hdf(path: Path, keep_ids=None, chunk: int = 5000) -> pd.DataFrame:
    """Read a pandas fixed-format HDF5 frame, keeping only rows in ``keep_ids``."""
    with h5py.File(path, "r") as f:
        g = f[list(f.keys())[0]]
        genes = g["axis0"][:].astype(str)
        cells = g["axis1"][:].astype(str)
        vals = g["block0_values"]
        rows = (
            np.arange(len(cells))
            if keep_ids is None
            else np.flatnonzero(np.isin(cells, list(keep_ids)))
        )
        out = np.empty((len(rows), vals.shape[1]), dtype=np.float32)
        for s in range(0, len(rows), chunk):
            r = rows[s : s + chunk]
            out[s : s + len(r)] = vals[r[0] : r[-1] + 1][r - r[0]]
    return pd.DataFrame(out, index=cells[rows], columns=genes)


def build(name: str, frames: list[pd.DataFrame], meta: pd.DataFrame, out: Path) -> None:
    """Write the filtered h5ad and first-50/100-PC npz files for one dataset."""
    df = pd.concat(frames)
    adata = ad.AnnData(df)
    adata.obs["donor"] = DONOR
    by_cell = meta.set_index("cell_id")
    adata.obs["day"] = by_cell.loc[adata.obs_names, "day"].values
    if "cell_type" in meta:
        adata.obs["cell_type"] = by_cell.loc[adata.obs_names, "cell_type"].values
    day_class = {d: i for i, d in enumerate(sorted(adata.obs["day"].unique()))}
    adata.obs["day_class"] = adata.obs["day"].map(day_class).astype(int)
    print(name, adata.shape, "days", dict(adata.obs["day"].value_counts().sort_index()))
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    sc.tl.pca(adata, n_comps=100, use_highly_variable=True)
    adata.write_h5ad(out / f"{name}_donor-{DONOR}.h5ad")
    for d in (50, 100):
        np.savez(
            out / f"{name}_D-{d}_pca.npz",
            data=adata.obsm["X_pca"][:, :d].astype(np.float32),
            colors=adata.obs["day_class"].values,
            day=adata.obs["day"].values,
            cell_id=adata.obs_names.values,
        )


def main() -> None:
    config = PrepareConfig()
    config.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(config.raw / "metadata.csv")
    meta = meta[(meta.donor == DONOR) & meta.day.isin(DAYS)]
    if "cite" in config.which:
        m = meta[(meta.technology == "citeseq") & (meta.cell_type != "unknown")]
        ids = set(m.cell_id)
        build(
            "cite",
            [
                read_fixed_hdf(config.raw / f, ids)
                for f in ("train_cite_inputs.h5", "test_cite_inputs.h5")
            ],
            m,
            config.out,
        )
    if "multi" in config.which:
        m = meta[meta.technology == "multiome"]
        build(
            "multi",
            [read_fixed_hdf(config.raw / "train_multi_targets.h5", set(m.cell_id))],
            m,
            config.out,
        )


if __name__ == "__main__":
    main()
