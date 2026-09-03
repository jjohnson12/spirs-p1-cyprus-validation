# Provenance and reproducibility boundaries

## Dataset

The manuscript is locked to the audited version-3 Cyprus/PROTEAS archive at [Zenodo DOI 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793). The public archive is not copied into this repository.

## Model outputs

The analysis starts from SPIRS-P1 segmentation and tracked lesion-ID outputs. Model architecture, inference code, and weights are maintained in [QTIM-Lab/Brain-Mets-Seg](https://github.com/QTIM-Lab/Brain-Mets-Seg), not duplicated here.

The manuscript submission bundle used to assemble this repository did not preserve a machine-verifiable inference commit, checkpoint SHA-256, or container digest. Those identifiers must not be inferred from filenames. They should be added here if recovered from the original inference environment before an archival software release is minted.

## Software environment

The exact historical package lockfile was not preserved. `requirements.txt` therefore supplies compatible version bounds, not a claim of bit-for-bit environment reconstruction. Python 3.11 is recommended.

## Code identity

`code_sha256.txt` records SHA-256 hashes for every released analysis script. Hashes identify this code snapshot; they do not identify the external dataset or prediction volumes.

## Scope

These programs reproduce analysis from locally supplied imaging, manifests, and predictions. They do not download data, perform model inference, or redistribute protected or restricted content.
