# Study data

| Dataset | Included in the repository | Source |
| --- | --- | --- |
| PDAC | Native paired traces and interval references for all 3,030 test groups. The split index covers all 30,294 groups. | Park et al. · [MassIVE MSV000089914, CC0](https://doi.org/10.25345/C5JW86R8P) · [Signal archive](https://doi.org/10.5281/zenodo.7964403) |
| EOC | All 78 peptide targets in acquisition `ruthhu_H120701_044` and the extraction index for Dataset 2. | Hüttenhain et al. · [MassIVE MSV000084048, CC0](https://doi.org/10.25345/C5PW7B) |
| HILIC2 and HILIC1a | Assay indices and study splits. The preparer retrieves chromatograms and integration workbooks from the authors. | [automRm study](https://doi.org/10.1021/acs.analchem.1c05224) · [Pinned data release](https://gitlab.gwdg.de/joerg.buescher/demodata/-/tree/c57f03d248097a70a5795195c7380d256a920381) |

From the repository root, run the complete PDAC test set with the bundled checkpoint.

```bash
chromboundary predict --model pdac --signals data/pdac_test --out predictions.csv
chromboundary evaluate --predictions predictions.csv --references data/pdac_test/references.csv
```

Strict top-ranked recall at IoU 0.90 is **1,480/2,747 (53.88%)**. IoU is the overlap duration divided by the union duration of the predicted and reference intervals.

To prepare the study datasets locally, use the commands below. HILIC2 and HILIC1a share the study's matrix–solvent split. For pooled training, pass `--selection-panel HILIC2` to `chromboundary train`.

```bash
python -m pip install ".[data]"
python data/prepare_pdac.py --out datasets/pdac
python data/prepare_hilic.py --panel both --out datasets/hilic
python data/prepare_eoc.py --out datasets/eoc
```

Each command writes `signals.npz` and `references.csv`. References use `1` for a peak, `0` for absence and `-1` for unknown status. Positive intervals are in seconds. Training uses the `train` and `val` splits. EOC is an external test set for the PDAC checkpoint.

Figure 1 includes a public-domain instrument photograph courtesy of the U.S. Geological Survey.
