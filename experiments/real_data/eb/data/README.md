# EB data

Not redistributed with this code (gitignored).

Source: Moon et al. (2019), the PHATE embryoid-body scRNA-seq time course (Mendeley
Data, doi.org/10.17632/v6n743h5ng.1 ). Extract the `scRNAseq/` archive with the five 10x
timepoint folders (`T0_1A` … `T8_9E`) here.

Then run:

```bash
python experiments/real_data/eb/prepare_eb_pca.py
python -m experiments.real_data.scripts.prepare_eb_geometry
python -m experiments.real_data.scripts.calibrate_scales --scenario eb
```

The first command writes `eb_after_pca_log1p.csv` (25.7k cells × 100 PCs); the other
two write `eb/geometry/` here.
