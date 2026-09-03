# Analysis workflow

## Inputs

1. Cyprus/PROTEAS version-3 patient ZIP archives from [Zenodo DOI 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793).
2. The local preprocessing manifest, `dataset.csv`.
3. SPIRS-P1 tracked prediction-ID maps, normally organized beneath `04_tracked/<case unit>/<study>/anat/`.

The repository does not contain imaging data, reference masks, prediction volumes, or protected health information.

## Ordered stages

1. `audit_cyprus_denominators_zips.py` reads the patient archives without modifying them and reconciles T1-post images, reference masks, positive tumor cores, case units, and longitudinal pairs.
2. `relink_cyprus_ground_truth.py` extracts only reference masks needed by the manifest and corrects case-sensitive timepoint linkage. It writes `dataset_relinked.csv` while preserving the input manifest.
3. `rescore_cyprus_relinked_v2.py` recomputes cross-sectional lesion-level and study-level metrics. Its old 166-study calculation is retained as a historical control; the corrected positive-reference cohort contains 169 studies.
4. `patient_cluster_bootstrap_cross_sectional_v3_direct.py` reconstructs the corrected point estimates and computes patient-clustered percentile confidence intervals using 10,000 resamples and seed `20260809`.
5. `rescore_longitudinal_volume_change_relinked_v2.py` evaluates total tumor-core volume-change categories for consecutive available-reference pairs and the strict original-adjacency sensitivity subset.
6. `ranobm_endpoint_v4_reference_anchored.py` computes reference-anchored lesion trajectories. `finalize_ranobm_v4_primary_v2.py` applies the primary index-target eligibility flow and produces the final 3 × 3 contingency table.
7. `audit_ranobm_new_lesions_v4.py` reports reference-new and model-only new measurable lesions separately from index-target agreement.

## Interpretation safeguards

- `ranobm_endpoint.py` is historical and supplies shared image, component, tracking, diameter, and threshold utilities. Do not substitute its historical endpoint output for the v4 primary analysis.
- A missing or temporally unmatched reference component is not automatically interpreted as true lesion disappearance.
- Model diameter is set to zero only when the matched model lesion is absent at a known reference-positive follow-up observation.
- The modified RANO-BM analysis is an image-derived trajectory endpoint and is not a clinical response assessment incorporating all clinical and imaging criteria.
- Synthetic study folder dates are ordering identifiers; they must not be used to derive rates or time-to-progression.

## Verification

Compare generated summaries with `results/manuscript_targets.json`. A mismatch should be investigated rather than overwritten. Exact numerical reproduction also depends on using the same public archive version and the same SPIRS-P1 prediction maps.
