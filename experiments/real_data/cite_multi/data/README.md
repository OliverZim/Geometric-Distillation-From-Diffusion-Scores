# CITE-seq and multiome data go here

Empty by design — the datasets are not redistributed with this code.

Source: NeurIPS 2022 Open Problems in Single-Cell Analysis, *Multimodal Single-Cell
Integration* (GEO GSE194122). CITE-seq arm: 29,394 cells. Multiome arm: 35,396 cells.
Donor 13176, days 2/3/4/7 in both.

Then run from the repository root:

```bash
# raw Kaggle .h5 files -> processed/{cite,multi}_D-{50,100}_pca.npz
python experiments/real_data/cite_multi/prepare_cite_multi.py

# build the four geometry working spaces (also writes calibration.json)
for spec in cite50:cite_D-50 cite100:cite_D-100 multi50:multi_D-50 multi100:multi_D-100; do
    scenario=${spec%%:*}; npz=${spec#*:}
    python -m experiments.real_data.scripts.prepare_eb_geometry \
        --scenario "$scenario" \
        --data-root experiments/real_data/cite_multi/data \
        --npz "experiments/real_data/cite_multi/data/processed/${npz}_pca.npz"
done
```

which write `{cite50,cite100,multi50,multi100}/geometry/` here. To recompute
only the scale table afterwards, run
`python -m experiments.real_data.scripts.calibrate_scales` (included above).
See the top-level `README.md` for the full local reproduction workflow.
