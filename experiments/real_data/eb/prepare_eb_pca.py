#!/usr/bin/env python
"""Build the EB PCA working space from the raw 10x timepoints."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scprep
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from scipy.io import mmread
from sklearn.decomposition import PCA

from geometry_distill.config import ExperimentConfig

HERE = Path(__file__).resolve().parent


class PrepareConfig(ExperimentConfig):
    """Configuration for the EB dataset preparation script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    raw: Path = Field(
        default_factory=lambda: HERE / "data" / "scRNAseq",
        description="Directory with the raw 10x timepoint folders.",
    )
    out: Path = Field(
        default_factory=lambda: HERE / "data" / "eb_after_pca_log1p.csv",
        description="Output CSV with barcode, timepoint and PC columns.",
    )
    n_pcs: int = Field(default=100, ge=1, description="Number of principal components.")
    min_cells: int = Field(
        default=10, ge=1, description="Minimum cells expressing a retained gene."
    )
    percentile: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Library-size percentile below which cells are dropped.",
    )
    seed: int = Field(default=42, description="Random seed for the PCA fit.")


def load_timepoints(raw: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the 10x timepoints and concatenate them into a cells-by-genes frame."""
    frames = []
    labels = []
    for tp_dir in sorted(glob.glob(str(raw / "T*"))):
        name = Path(tp_dir).name
        matrix = mmread(Path(tp_dir) / "matrix.mtx").T.tocsr()
        barcodes = pd.read_csv(Path(tp_dir) / "barcodes.tsv", header=None)[0].values
        genes = pd.read_csv(Path(tp_dir) / "genes.tsv", header=None, sep="\t")
        gene_names = genes[1].values if genes.shape[1] > 1 else genes[0].values
        frames.append(
            pd.DataFrame.sparse.from_spmatrix(
                matrix, index=barcodes, columns=gene_names
            )
        )
        labels.extend([name] * len(barcodes))
        print(
            f"  {name}: {matrix.shape[0]} cells x {matrix.shape[1]} genes", flush=True
        )

    data = pd.concat(frames, axis=0)
    # genes.tsv maps several Ensembl IDs to one symbol; PCA rejects duplicate columns.
    data = data.loc[:, ~data.columns.duplicated()]
    return data, pd.Series(labels, index=data.index, name="timepoint")


def main() -> None:
    config = PrepareConfig()
    data, timepoint = load_timepoints(config.raw)
    print(f"raw: {data.shape[0]} cells x {data.shape[1]} genes", flush=True)

    data = scprep.filter.filter_rare_genes(data, min_cells=config.min_cells)
    # Barcodes repeat across timepoints and scprep selects rows by label, so a
    # kept barcode also keeps its namesakes in the other timepoints.
    data, timepoint = scprep.filter.filter_library_size(
        data, timepoint, percentile=config.percentile, keep_cells="above"
    )
    data = data.sparse.to_dense()
    print(f"filtered: {data.shape[0]} cells x {data.shape[1]} genes", flush=True)

    data_norm = scprep.normalize.library_size_normalize(data)
    pca = PCA(n_components=config.n_pcs, random_state=config.seed)
    components = pca.fit_transform(np.log1p(data_norm))
    variance = pca.explained_variance_ratio_.sum()
    print(f"PCA: {variance:.1%} variance in {config.n_pcs} PCs", flush=True)

    table = pd.DataFrame(
        components, columns=[f"PC{i}" for i in range(1, config.n_pcs + 1)]
    )
    table.insert(0, "timepoint", timepoint.to_numpy())
    table.insert(0, "barcode", data.index.to_numpy())
    config.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.out, index=False)
    print(f"wrote {config.out} ({table.shape[0]} cells)", flush=True)


if __name__ == "__main__":
    main()
