#!/usr/bin/env python3
"""
patient_cluster_bootstrap_cross_sectional_v3_direct.py
======================================================

Standalone patient-cluster bootstrap for the corrected Cyprus cross-sectional
external-validation cohort.

This version DOES NOT depend on study_metrics.csv / lesion_metrics.csv.
It rebuilds the corrected cross-sectional metrics directly from:
  * dataset_relinked.csv
  * expert reference masks (GTMaskPath)
  * surviving tracked SPIRS-P1 prediction ID maps under 04_tracked

Definitions match the audited corrected analysis:
  * Cyprus tumor core = GT labels 1 + 2
  * Cross-sectional cohort = nonempty reference tumor-core studies only
  * 26-connected components
  * one-to-one Hungarian lesion matching
  * match threshold Jaccard >= 0.10
  * lesion volume tiers:
      <0.05 mL
      0.05-0.5 mL
      0.5-4 mL
      >=4 mL
  * bootstrap cluster = underlying patient (P04a/P04b -> P04)
  * default = 10,000 patient-level nonparametric percentile bootstrap replicates

The script validates the reconstructed point estimates before bootstrapping:
  169 studies
  413 reference lesions
  285 matched
  128 missed
  267 false positives
  sensitivities 9/54, 128/193, 94/107, 54/59

Place beside ranobm_endpoint.py.

Example:
BASE=/path/to/cyprus_validation
SCRIPTS=/path/to/this/repository/src

python "$SCRIPTS/patient_cluster_bootstrap_cross_sectional_v3_direct.py" \
  --output-dir "$BASE" \
  --dataset-csv "$BASE/dataset_relinked.csv" \
  --out-dir "$BASE/bootstrap_clustered_cross_sectional" \
  --n-bootstrap 10000 \
  --seed 20260809
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

import ranobm_endpoint as base


GT_CORE_LABELS = (1, 2)
MATCH_JACCARD = 0.10

TARGETS = {
    "n_studies": 169,
    "n_gt": 413,
    "n_matched": 285,
    "n_missed": 128,
    "n_fp": 267,
    "tiers": {
        "micro": (9, 54),
        "small": (128, 193),
        "medium": (94, 107),
        "large": (54, 59),
    },
}

TIER_ORDER = ["micro", "small", "medium", "large"]
TIER_LABEL = {
    "micro": "<0.05 mL",
    "small": "0.05-0.5 mL",
    "medium": "0.5-4 mL",
    "large": ">=4 mL",
}


def parent_patient_id(case_unit: str) -> str:
    s = str(case_unit).strip()
    m = re.fullmatch(r"(P\d{2})[ab]", s, flags=re.I)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"P\d{2}", s, flags=re.I):
        return s.upper()
    return re.sub(r"(?<=\d)[ab]$", "", s, flags=re.I)


def tier_from_volume(v: float) -> str:
    if v < 0.05:
        return "micro"
    if v < 0.5:
        return "small"
    if v < 4.0:
        return "medium"
    return "large"


def label_components(mask: np.ndarray) -> list[np.ndarray]:
    """26-connected 3D components, returned as boolean masks."""
    lab, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    return [(lab == i) for i in range(1, n + 1)]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return float(inter / union) if union else 0.0


def match_components(gt_components, pred_components):
    """One-to-one Hungarian matching maximizing Jaccard; retain J >= threshold."""
    if not gt_components or not pred_components:
        return [], list(range(len(gt_components))), list(range(len(pred_components)))

    J = np.zeros((len(gt_components), len(pred_components)), dtype=float)
    for i, g in enumerate(gt_components):
        for j, p in enumerate(pred_components):
            J[i, j] = jaccard(g, p)

    rr, cc = linear_sum_assignment(-J)
    matches = []
    used_g, used_p = set(), set()

    for i, j in zip(rr, cc):
        if J[i, j] >= MATCH_JACCARD:
            matches.append((int(i), int(j), float(J[i, j])))
            used_g.add(int(i))
            used_p.add(int(j))

    unmatched_gt = [i for i in range(len(gt_components)) if i not in used_g]
    unmatched_pred = [j for j in range(len(pred_components)) if j not in used_p]
    return matches, unmatched_gt, unmatched_pred


def dsc(gt: np.ndarray, pred: np.ndarray) -> float:
    ng = int(np.count_nonzero(gt))
    npred = int(np.count_nonzero(pred))
    inter = int(np.count_nonzero(gt & pred))
    den = ng + npred
    return float(2 * inter / den) if den else 1.0


def load_mask(path: str):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
    vv_ml = float(np.prod(zooms) / 1000.0)
    return arr, zooms, vv_ml


def reconstruct(dataset_csv: Path, tracked_root: str):
    df = pd.read_csv(dataset_csv, dtype=str)
    if "NormalizedSeriesDescription" in df.columns:
        df = df[df["NormalizedSeriesDescription"] == "T1Post"].copy()

    study_rows = []
    lesion_rows = []
    audit = {
        "manifest_t1post_rows": int(len(df)),
        "blank_gt_path": 0,
        "missing_gt_file": 0,
        "missing_prediction": 0,
        "reference_mask_studies": 0,
        "zero_core_studies_excluded": 0,
        "positive_core_studies": 0,
    }

    for _, row in df.iterrows():
        case_unit = str(row["AnonPatientID"])
        patient = parent_patient_id(case_unit)
        study_id = str(row["AnonStudyID"])
        tp = str(row.get("Timepoint", ""))
        tp_order = int(float(row.get("TimepointOrder", 0)))

        gt_path = str(row.get("GTMaskPath") or "").strip()
        if not gt_path or gt_path.lower() == "nan":
            audit["blank_gt_path"] += 1
            continue
        if not os.path.exists(gt_path):
            audit["missing_gt_file"] += 1
            continue

        audit["reference_mask_studies"] += 1

        pred_path = base.find_pred_ids(tracked_root, case_unit, study_id, row)
        if not pred_path or not os.path.exists(pred_path):
            audit["missing_prediction"] += 1
            continue

        gt_arr, gt_zooms, vv_ml = load_mask(gt_path)
        pred_arr, pred_zooms, _ = load_mask(pred_path)

        if gt_arr.shape != pred_arr.shape:
            raise ValueError(
                f"Shape mismatch {case_unit}/{study_id}: "
                f"GT {gt_arr.shape}, pred {pred_arr.shape}"
            )
        if any(abs(a - b) > 1e-5 for a, b in zip(gt_zooms, pred_zooms)):
            raise ValueError(
                f"Spacing mismatch {case_unit}/{study_id}: "
                f"GT {gt_zooms}, pred {pred_zooms}"
            )

        gt_core = np.isin(gt_arr, GT_CORE_LABELS)
        if not np.any(gt_core):
            audit["zero_core_studies_excluded"] += 1
            continue

        audit["positive_core_studies"] += 1

        pred_core = pred_arr > 0
        gt_components = label_components(gt_core)
        pred_components = label_components(pred_core)
        matches, unmatched_gt, unmatched_pred = match_components(
            gt_components, pred_components
        )

        matched_gt_to_pred = {gi: (pj, jac) for gi, pj, jac in matches}

        study_rows.append({
            "patient_id": patient,
            "case_unit": case_unit,
            "study_id": study_id,
            "timepoint": tp,
            "timepoint_order": tp_order,
            "dsc": dsc(gt_core, pred_core),
            "n_gt_lesions": len(gt_components),
            "n_pred_lesions": len(pred_components),
            "n_matched": len(matches),
            "n_missed": len(unmatched_gt),
            "n_false_positive": len(unmatched_pred),
        })

        for gi, gm in enumerate(gt_components):
            vol = float(np.count_nonzero(gm) * vv_ml)
            tier = tier_from_volume(vol)
            if gi in matched_gt_to_pred:
                pj, jac = matched_gt_to_pred[gi]
                status = "matched"
                pred_id = pj + 1
                jj = jac
            else:
                status = "missed"
                pred_id = np.nan
                jj = 0.0

            lesion_rows.append({
                "patient_id": patient,
                "case_unit": case_unit,
                "study_id": study_id,
                "timepoint": tp,
                "gt_component_id": gi + 1,
                "pred_component_id": pred_id,
                "match_status": status,
                "jaccard": jj,
                "gt_volume_mL": vol,
                "size_tier": tier,
            })

        for pj in unmatched_pred:
            lesion_rows.append({
                "patient_id": patient,
                "case_unit": case_unit,
                "study_id": study_id,
                "timepoint": tp,
                "gt_component_id": np.nan,
                "pred_component_id": pj + 1,
                "match_status": "false_positive",
                "jaccard": 0.0,
                "gt_volume_mL": np.nan,
                "size_tier": "",
            })

    studies = pd.DataFrame(study_rows)
    lesions = pd.DataFrame(lesion_rows)

    if not studies.empty:
        studies = studies.sort_values(
            ["patient_id", "case_unit", "timepoint_order", "study_id"],
            kind="stable",
        ).reset_index(drop=True)
    if not lesions.empty:
        lesions = lesions.sort_values(
            ["patient_id", "case_unit", "study_id", "match_status"],
            kind="stable",
        ).reset_index(drop=True)

    return studies, lesions, audit


def point_estimates(studies, lesions):
    gt = lesions[lesions["match_status"].isin(["matched", "missed"])].copy()
    fp = lesions[lesions["match_status"] == "false_positive"].copy()

    out = {
        "n_patients": int(studies["patient_id"].nunique()),
        "n_studies": int(len(studies)),
        "n_gt": int(len(gt)),
        "n_matched": int((gt["match_status"] == "matched").sum()),
        "n_missed": int((gt["match_status"] == "missed").sum()),
        "n_fp": int(len(fp)),
        "fp_per_study": float(len(fp) / len(studies)),
        "median_dsc": float(np.median(studies["dsc"])),
    }

    for tier in TIER_ORDER:
        x = gt[gt["size_tier"] == tier]
        n = int(len(x))
        m = int((x["match_status"] == "matched").sum())
        out[f"{tier}_n"] = n
        out[f"{tier}_matched"] = m
        out[f"{tier}_sensitivity"] = float(m / n) if n else np.nan

    return out


def validate(pt):
    problems = []
    for key in ["n_studies", "n_gt", "n_matched", "n_missed", "n_fp"]:
        if pt[key] != TARGETS[key]:
            problems.append(f"{key}: got {pt[key]}, expected {TARGETS[key]}")

    for tier, (mexp, nexp) in TARGETS["tiers"].items():
        mgot = pt[f"{tier}_matched"]
        ngot = pt[f"{tier}_n"]
        if (mgot, ngot) != (mexp, nexp):
            problems.append(
                f"{tier}: got {mgot}/{ngot}, expected {mexp}/{nexp}"
            )
    return problems


def percentile_ci(values, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return (
        float(np.percentile(x, 100 * alpha / 2)),
        float(np.percentile(x, 100 * (1 - alpha / 2))),
    )


def bootstrap(studies, lesions, n_boot, seed):
    patients = sorted(studies["patient_id"].unique())
    n_patients = len(patients)
    rng = np.random.default_rng(seed)

    gt = lesions[lesions["match_status"].isin(["matched", "missed"])].copy()
    fp = lesions[lesions["match_status"] == "false_positive"].copy()

    # Patient-level sufficient statistics.
    tier_counts = {}
    fp_count = {}
    study_count = {}
    dsc_values = {}

    for p in patients:
        gp = gt[gt["patient_id"] == p]
        tier_counts[p] = {}
        for tier in TIER_ORDER:
            z = gp[gp["size_tier"] == tier]
            tier_counts[p][tier] = (
                int((z["match_status"] == "matched").sum()),
                int(len(z)),
            )
        fp_count[p] = int((fp["patient_id"] == p).sum())
        sp = studies[studies["patient_id"] == p]
        study_count[p] = int(len(sp))
        dsc_values[p] = sp["dsc"].to_numpy(dtype=float)

    recs = []
    for b in range(n_boot):
        sample = rng.choice(patients, size=n_patients, replace=True)
        unique, counts = np.unique(sample, return_counts=True)
        mult = dict(zip(unique, counts))

        r = {"replicate": b + 1}

        for tier in TIER_ORDER:
            matched = sum(
                int(k) * tier_counts[p][tier][0] for p, k in mult.items()
            )
            total = sum(
                int(k) * tier_counts[p][tier][1] for p, k in mult.items()
            )
            r[f"sens_{tier}"] = matched / total if total else np.nan

        nfp = sum(int(k) * fp_count[p] for p, k in mult.items())
        nstud = sum(int(k) * study_count[p] for p, k in mult.items())
        r["fp_per_study"] = nfp / nstud if nstud else np.nan

        arrs = []
        for p, k in mult.items():
            vals = dsc_values[p]
            for _ in range(int(k)):
                arrs.append(vals)
        r["median_dsc"] = float(np.median(np.concatenate(arrs))) if arrs else np.nan

        recs.append(r)

    return pd.DataFrame(recs)


def summary_table(pt, boot):
    rows = []
    for tier in TIER_ORDER:
        lo, hi = percentile_ci(boot[f"sens_{tier}"])
        rows.append({
            "endpoint": "lesion_sensitivity",
            "stratum": TIER_LABEL[tier],
            "numerator": pt[f"{tier}_matched"],
            "denominator": pt[f"{tier}_n"],
            "estimate": pt[f"{tier}_sensitivity"],
            "ci_low": lo,
            "ci_high": hi,
        })

    lo, hi = percentile_ci(boot["fp_per_study"])
    rows.append({
        "endpoint": "false_positives_per_study",
        "stratum": "all",
        "numerator": np.nan,
        "denominator": np.nan,
        "estimate": pt["fp_per_study"],
        "ci_low": lo,
        "ci_high": hi,
    })

    lo, hi = percentile_ci(boot["median_dsc"])
    rows.append({
        "endpoint": "median_per_study_DSC",
        "stratum": "all",
        "numerator": np.nan,
        "denominator": np.nan,
        "estimate": pt["median_dsc"],
        "ci_low": lo,
        "ci_high": hi,
    })

    return pd.DataFrame(rows)


def print_point_validation(pt, audit):
    print("=== RECONSTRUCTED CORRECTED CROSS-SECTIONAL COHORT ===")
    print(f"Manifest T1-post rows: {audit['manifest_t1post_rows']}")
    print(f"Blank GT path: {audit['blank_gt_path']}")
    print(f"Reference-mask studies: {audit['reference_mask_studies']}")
    print(f"Annotated zero-core studies excluded: {audit['zero_core_studies_excluded']}")
    print(f"Positive-core studies analyzed: {pt['n_studies']}")
    print(f"Patients: {pt['n_patients']}")
    print(f"Reference lesions: {pt['n_gt']} "
          f"(matched {pt['n_matched']}, missed {pt['n_missed']})")
    print(f"False positives: {pt['n_fp']} = {pt['fp_per_study']:.3f}/study")
    print(f"Median DSC: {pt['median_dsc']:.4f}")
    for tier in TIER_ORDER:
        print(
            f"{TIER_LABEL[tier]:>12s}: "
            f"{pt[f'{tier}_matched']}/{pt[f'{tier}_n']} = "
            f"{100*pt[f'{tier}_sensitivity']:.1f}%"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--tracked-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument(
        "--allow-target-mismatch",
        action="store_true",
        help="continue even if reconstructed point estimates do not match audited targets",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    tracked_root = str(Path(args.tracked_root or output_dir / "04_tracked"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reconstructing cross-sectional cohort directly from masks...")
    studies, lesions, audit = reconstruct(Path(args.dataset_csv), tracked_root)
    pt = point_estimates(studies, lesions)
    print_point_validation(pt, audit)

    problems = validate(pt)
    if problems:
        print("\nAUDITED TARGET VALIDATION: FAILED")
        for x in problems:
            print("  - " + x)
        if not args.allow_target_mismatch:
            raise SystemExit(
                "\nStopping before bootstrap because reconstructed counts do not "
                "match the previously validated corrected cohort."
            )
    else:
        print("\nAUDITED TARGET VALIDATION: PASS")

    print(
        f"\nRunning {args.n_bootstrap:,} patient-cluster bootstrap replicates "
        f"(seed={args.seed})..."
    )
    boot = bootstrap(studies, lesions, args.n_bootstrap, args.seed)
    summ = summary_table(pt, boot)

    print("\n=== PATIENT-CLUSTERED PERCENTILE 95% CIs ===")
    for _, r in summ.iterrows():
        if r["endpoint"] == "lesion_sensitivity":
            print(
                f"{r['stratum']:>12s}: "
                f"{int(r['numerator'])}/{int(r['denominator'])} = "
                f"{100*r['estimate']:.1f}% "
                f"(95% CI {100*r['ci_low']:.1f}-{100*r['ci_high']:.1f}%)"
            )
        elif r["endpoint"] == "false_positives_per_study":
            print(
                f"FP/study: {r['estimate']:.3f} "
                f"(95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f})"
            )
        else:
            print(
                f"Median DSC: {r['estimate']:.4f} "
                f"(95% CI {r['ci_low']:.4f}-{r['ci_high']:.4f})"
            )

    studies.to_csv(out_dir / "reconstructed_study_metrics.csv", index=False)
    lesions.to_csv(out_dir / "reconstructed_lesion_metrics.csv", index=False)
    boot.to_csv(out_dir / "bootstrap_replicates.csv", index=False)
    summ.to_csv(out_dir / "clustered_bootstrap_summary.csv", index=False)

    metadata = {
        "dataset_csv": str(Path(args.dataset_csv).resolve()),
        "tracked_root": tracked_root,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "cluster_unit": "underlying patient",
        "patient_mapping": "case units with trailing a/b collapsed to parent patient",
        "ci_method": "nonparametric patient-cluster percentile bootstrap",
        "matching": "26-connected components; Hungarian one-to-one assignment; Jaccard >= 0.10",
        "gt_core_labels": list(GT_CORE_LABELS),
        "audit": audit,
        "point_estimates": pt,
        "target_validation_problems": problems,
    }
    (out_dir / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2))

    lines = [
        "=== PATIENT-CLUSTERED BOOTSTRAP CROSS-SECTIONAL CIs ===",
        f"Patients: {pt['n_patients']}",
        f"Bootstrap replicates: {args.n_bootstrap}",
        f"Seed: {args.seed}",
        "CI method: nonparametric patient-cluster percentile bootstrap",
        "",
    ]
    for _, r in summ.iterrows():
        if r["endpoint"] == "lesion_sensitivity":
            lines.append(
                f"{r['stratum']}: {int(r['numerator'])}/{int(r['denominator'])} "
                f"= {100*r['estimate']:.1f}% "
                f"(95% CI {100*r['ci_low']:.1f}-{100*r['ci_high']:.1f}%)"
            )
        elif r["endpoint"] == "false_positives_per_study":
            lines.append(
                f"FP/study: {r['estimate']:.3f} "
                f"(95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f})"
            )
        else:
            lines.append(
                f"Median DSC: {r['estimate']:.4f} "
                f"(95% CI {r['ci_low']:.4f}-{r['ci_high']:.4f})"
            )
    (out_dir / "bootstrap_summary.txt").write_text("\n".join(lines) + "\n")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
