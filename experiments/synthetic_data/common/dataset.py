"""Dataset loading for the Splatter-simulated scRNA-seq data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


@dataclass(frozen=True)
class SimulatedDataset:
    """A loaded simulation: sparse cells×genes counts and aligned ground-truth labels."""

    counts: sparse.csr_matrix
    gene_names: np.ndarray
    cell_ids: np.ndarray
    labels: pd.DataFrame
    base_cell_means: np.ndarray | None = None
    cell_means: np.ndarray | None = None


def load_simulation(
    scenario: str, root: str | Path, load_clean: bool = False
) -> SimulatedDataset:
    """Load ``{root}/{scenario}/`` (10x mtx, barcodes and labels)."""
    data_dir = Path(root) / scenario
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Simulation output not found: {data_dir} "
            "(run experiments/synthetic_data/synthetic/data_generation/simulate_splatter.R first)"
        )

    counts = mmread(data_dir / "matrix.mtx").T.tocsr()
    genes = pd.read_csv(data_dir / "genes.tsv", header=None, sep="\t")
    gene_names = genes[1].values if genes.shape[1] > 1 else genes[0].values
    cell_ids = pd.read_csv(data_dir / "barcodes.tsv", header=None)[0].values
    labels = pd.read_csv(data_dir / "labels.csv")

    if len(cell_ids) != counts.shape[0]:
        raise ValueError(
            f"{data_dir}: {counts.shape[0]} cells but {len(cell_ids)} barcodes"
        )
    if len(labels) != counts.shape[0]:
        raise ValueError(
            f"{data_dir}: {counts.shape[0]} cells but {len(labels)} label rows"
        )
    if not (labels["Cell"].values == cell_ids).all():
        raise ValueError(
            f"{data_dir}: labels.csv 'Cell' column does not match barcodes.tsv "
            "(label/count misalignment would corrupt all downstream analyses)"
        )

    base_cell_means = cell_means = None
    if load_clean:
        # The .npy files are genes×cells; expose cells×genes like ``counts``.
        clean_paths = {
            "base_cell_means": data_dir / "base_cell_means.npy",
            "cell_means": data_dir / "cell_means.npy",
        }
        missing = [p for p in clean_paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Clean targets not found: "
                + ", ".join(str(p) for p in missing)
                + " (regenerate with "
                "experiments/synthetic_data/synthetic/data_generation/simulate_splatter.R)"
            )
        base_cell_means = np.load(clean_paths["base_cell_means"]).T
        cell_means = np.load(clean_paths["cell_means"]).T
        for name, arr in (
            ("base_cell_means", base_cell_means),
            ("cell_means", cell_means),
        ):
            if arr.shape != (counts.shape[0], len(gene_names)):
                raise ValueError(
                    f"{data_dir}: {name} shape {arr.shape} does not match "
                    f"counts ({counts.shape[0]} cells, {len(gene_names)} genes)"
                )

    return SimulatedDataset(
        counts=counts,
        gene_names=gene_names,
        cell_ids=cell_ids,
        labels=labels,
        base_cell_means=base_cell_means,
        cell_means=cell_means,
    )
