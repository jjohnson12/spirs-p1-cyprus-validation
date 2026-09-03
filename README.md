# SPIRS-P1 Cyprus longitudinal validation

Audited analysis code for the manuscript **“Longitudinal External Testing of Deep Learning Brain Metastasis Segmentation on a Public MRI Dataset.”** The repository evaluates SPIRS-P1 Brain-Mets-Seg predictions on the longitudinal Cyprus/PROTEAS brain-metastasis MRI dataset.

This is the focused manuscript-analysis repository. Model architecture, inference code, and weights are maintained separately in [QTIM-Lab/Brain-Mets-Seg](https://github.com/QTIM-Lab/Brain-Mets-Seg).

## What is included

- Source-archive and denominator auditing
- Reference-mask relinking and normalized manifest creation
- Cross-sectional lesion detection and segmentation rescoring
- Patient-clustered bootstrap confidence intervals
- Whole-burden longitudinal volume-change analysis
- Reference-anchored modified RANO-BM trajectory analysis
- Separate audits of reference-new and model-only new measurable lesions
- Machine-readable manuscript targets and code checksums

The final manuscript values are locked to the audited version-3 Cyprus archive: [Zenodo DOI 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793). Dataset files and model outputs are not redistributed here.

## Reproducibility status

All scripts are included and syntax-checked. Full execution requires the public dataset, a local `dataset.csv` manifest, and SPIRS-P1 tracked prediction maps. The submission bundle did not preserve exact historical Python package builds or the exact inference checkpoint hash; these limitations are documented in [`provenance/README.md`](provenance/README.md).

The principal locked settings are:

- Cyprus tumor core: labels 1 and 2; edema label 3 excluded
- 26-connected components
- One-to-one Hungarian lesion matching at Jaccard ≥ 0.10
- Patient-level nonparametric bootstrap: 10,000 resamples, seed `20260809`
- Modified RANO-BM measurability floor: 10 mm

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The bounds in `requirements.txt` describe compatible modern environments; they are not represented as an exact reconstruction of the historical environment.

## Running the analysis

Copy the example configuration and set paths to your local data and prediction outputs:

```bash
cp config/paths.env.example config/paths.env
```

Then run:

```bash
bash scripts/run_audited_analysis.sh config/paths.env
```

The scripts can also be run individually. See [`docs/analysis_workflow.md`](docs/analysis_workflow.md) for the analysis order, inputs, and outputs. The pipeline never modifies the source patient ZIP archives. The relinking script refuses to replace prior outputs unless `--force` is supplied.

## Repository map

```text
config/       Example local path configuration
docs/         Analysis workflow and interpretation
provenance/   Scope, limitations, and SHA-256 code checksums
results/      Manuscript-locked numerical targets
scripts/      Reproduction entry point
src/          Audited analysis programs
```

`src/ranobm_endpoint.py` is retained for traceability and shared utilities. The manuscript-locked primary trajectory endpoint is `src/ranobm_endpoint_v4_reference_anchored.py`, finalized by `src/finalize_ranobm_v4_primary_v2.py`.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until a journal citation is available, cite this software repository and the public dataset DOI above.

## License

MIT License. See [`LICENSE`](LICENSE).
