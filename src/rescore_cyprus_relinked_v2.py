#!/usr/bin/env python3
"""
Re-score Cyprus/PROTEAS Brain-Mets-Seg predictions against the corrected
relinked reference masks, without re-running preprocessing or inference.

If original model_ensemble-label.nii.gz files were purged, the script falls
back to the corrected Preprocessed_label1_ids tracked volumes and binarizes
them as >0. Those tracked volumes were generated to preserve exactly the
source prediction voxels.

Designed to answer two questions after repairing the P31 GT linkage:
  1) Can we reproduce the manuscript's original 166-study cross-sectional
     metrics using the original GT-linked subset?
  2) How do those metrics change when the 3 tumor-core-positive P31 studies
     are restored (169-study corrected positive-reference cohort)?

Cyprus labels:
  1 = necrotic core
  2 = enhancing tumor
  3 = edema
Tumor core = labels 1 + 2.

Detection definition follows the manuscript:
  - 26-connected components in 3D
  - GT/pred lesion pair eligible when Jaccard >= 0.10
  - one-to-one matching maximizing number of eligible matches, then Jaccard
  - sensitivity = matched GT lesions / GT lesions
  - false positives = unmatched predicted components

Outputs:
  study_metrics.csv
  lesion_metrics.csv
  rescore_summary.txt

Example:
  python rescore_cyprus_relinked.py \
    --dataset /path/to/cyprus_validation/dataset_relinked.csv \
    --out-dir /path/to/cyprus_validation/rescore_relinked
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment


def make_loader():
    try:
        import SimpleITK as sitk
        def load(path: Path):
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img)
            spacing = tuple(float(x) for x in img.GetSpacing())
            vox_mm3 = spacing[0] * spacing[1] * spacing[2]
            size = tuple(int(x) for x in img.GetSize())  # x,y,z
            return np.asarray(arr), vox_mm3, spacing, size, "SimpleITK"
        return load
    except Exception:
        pass
    try:
        import nibabel as nib
        def load(path: Path):
            img = nib.load(str(path))
            arr = np.asanyarray(img.dataobj)
            zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
            vox_mm3 = zooms[0] * zooms[1] * zooms[2]
            size = tuple(int(x) for x in arr.shape[:3])
            return np.asarray(arr), vox_mm3, zooms, size, "nibabel"
        return load
    except Exception as e:
        raise RuntimeError("Need SimpleITK or nibabel") from e


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool, copy=False)
    pred = pred.astype(bool, copy=False)
    a = int(gt.sum())
    b = int(pred.sum())
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    inter = int(np.logical_and(gt, pred).sum())
    return 2.0 * inter / (a + b)


def component_stats(binary: np.ndarray):
    lab, n = cc_label(binary.astype(bool), structure=CONNECTIVITY_26)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, n, sizes


def jaccard_matrix(gt_lab, n_gt, gt_sizes, pred_lab, n_pred, pred_sizes):
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=float), np.zeros((n_gt, n_pred), dtype=int)

    # Count pairwise intersections using a single bincount over combined labels.
    mask = (gt_lab > 0) & (pred_lab > 0)
    if not mask.any():
        return np.zeros((n_gt, n_pred), dtype=float), np.zeros((n_gt, n_pred), dtype=int)

    g = gt_lab[mask].astype(np.int64) - 1
    p = pred_lab[mask].astype(np.int64) - 1
    code = g * n_pred + p
    inter_flat = np.bincount(code, minlength=n_gt * n_pred)
    inter = inter_flat.reshape(n_gt, n_pred)
    unions = gt_sizes[1:, None] + pred_sizes[None, 1:] - inter
    jac = np.divide(inter, unions, out=np.zeros_like(inter, dtype=float), where=unions > 0)
    return jac, inter


def match_components(jac: np.ndarray, threshold: float = 0.10):
    n_gt, n_pred = jac.shape
    if n_gt == 0 or n_pred == 0:
        return []
    n = max(n_gt, n_pred)
    score = np.zeros((n, n), dtype=float)
    valid = jac >= threshold
    # Large base reward ensures maximum number of threshold-qualified matches;
    # Jaccard breaks ties among matchings with equal cardinality.
    score[:n_gt, :n_pred] = np.where(valid, 1000.0 + jac, 0.0)
    rows, cols = linear_sum_assignment(-score)
    matches = []
    for r, c in zip(rows, cols):
        if r < n_gt and c < n_pred and valid[r, c]:
            matches.append((r + 1, c + 1, float(jac[r, c])))
    return matches


def clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def iqr(series: pd.Series):
    q = series.quantile([0.25, 0.75])
    return float(q.iloc[0]), float(q.iloc[1])


def summarize(name: str, studies: pd.DataFrame, lesions: pd.DataFrame, lines: list[str]):
    lines.append("")
    lines.append(name)
    lines.append("-" * len(name))
    lines.append(f"studies: {len(studies)}")
    if len(studies):
        q1, q3 = iqr(studies["dsc"])
        lines.append(f"median DSC: {studies['dsc'].median():.4f} (IQR {q1:.4f}-{q3:.4f})")
        lines.append(f"mean DSC: {studies['dsc'].mean():.4f}")
        lines.append(f"total GT lesions: {int(studies['n_gt_lesions'].sum())}")
        lines.append(f"total predicted lesions: {int(studies['n_pred_lesions'].sum())}")
        lines.append(f"total matched GT lesions: {int(studies['n_matched'].sum())}")
        lines.append(f"total false-positive lesions: {int(studies['n_false_positive'].sum())}")
        lines.append(f"false-positive lesions/study: {studies['n_false_positive'].sum()/len(studies):.4f}")

    tiers = [
        ("<0.05 mL", lambda x: x < 0.05),
        ("0.05-0.5 mL", lambda x: (x >= 0.05) & (x < 0.5)),
        ("0.5-4 mL", lambda x: (x >= 0.5) & (x < 4.0)),
        (">=4 mL", lambda x: x >= 4.0),
    ]
    lines.append("size-stratified detection sensitivity (GT volume):")
    for label, fn in tiers:
        sub = lesions[fn(lesions["gt_volume_mL"])]
        n = len(sub)
        det = int(sub["detected"].sum()) if n else 0
        sens = det / n if n else float("nan")
        lines.append(f"  {label}: {det}/{n} = {sens:.4f}" if n else f"  {label}: n=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--jaccard-threshold", type=float, default=0.10)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    load = make_loader()

    required = ["PreprocessedSeg", "GTMaskPath", "CaseUnit", "Timepoint", "AnonStudyID"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Corrected manifest missing columns: {missing_cols}")

    # The corrected positive-reference cohort.
    eligible = clean_bool(df["GTTumorCorePositive"]) if "GTTumorCorePositive" in df.columns else df["GTMaskPath"].notna()

    # Reconstruct original 166-study subset from preserved pre-relink paths.
    if "GTMaskPathOriginal" in df.columns:
        orig_linked = df["GTMaskPathOriginal"].notna() & (df["GTMaskPathOriginal"].astype(str).str.strip() != "")
    else:
        orig_linked = pd.Series(False, index=df.index)

    study_rows = []
    lesion_rows = []

    for idx, row in df.loc[df["GTMaskPath"].notna() & (df["GTMaskPath"].astype(str).str.strip() != "")].iterrows():
        gt_path = Path(str(row["GTMaskPath"]))
        if not gt_path.exists():
            print(f"WARN missing GT: {gt_path}")
            continue

        # Prefer the original Brain-Mets-Seg binary prediction when it still exists.
        # On Roberts scratch, many/all model_ensemble-label.nii.gz files may have
        # been purged while the corrected tracked-ID volumes survived. The
        # corrected tracking products are voxel-identical to the source model
        # segmentation by construction/QA, so >0 is a valid binary fallback.
        pred_path = None
        pred_source = None
        raw_pred = row.get("PreprocessedSeg", "")
        if isinstance(raw_pred, str) and raw_pred.strip():
            p = Path(raw_pred)
            if p.exists():
                pred_path = p
                pred_source = "PreprocessedSeg"

        if pred_path is None:
            raw_tracked = row.get("Preprocessed_label1_ids", "")
            if isinstance(raw_tracked, str) and raw_tracked.strip():
                p = Path(raw_tracked)
                if p.exists():
                    pred_path = p
                    pred_source = "Preprocessed_label1_ids"

        if pred_path is None:
            print(
                f"WARN no surviving prediction or tracked-ID mask: "
                f"{row['CaseUnit']}:{row['Timepoint']} "
                f"PreprocessedSeg={row.get('PreprocessedSeg','')} "
                f"Tracked={row.get('Preprocessed_label1_ids','')}"
            )
            continue

        gt_raw, gt_vox_mm3, gt_spacing, gt_size, gt_loader = load(gt_path)
        pred_raw, pred_vox_mm3, pred_spacing, pred_size, pred_loader = load(pred_path)
        if gt_raw.shape != pred_raw.shape:
            raise RuntimeError(
                f"Geometry mismatch {row['CaseUnit']}:{row['Timepoint']}: "
                f"GT shape={gt_raw.shape}, pred shape={pred_raw.shape}"
            )
        # Allow tiny float spacing differences, but not meaningful voxel-volume mismatch.
        if not math.isclose(gt_vox_mm3, pred_vox_mm3, rel_tol=1e-4, abs_tol=1e-6):
            raise RuntimeError(
                f"Voxel-volume mismatch {row['CaseUnit']}:{row['Timepoint']}: "
                f"GT={gt_vox_mm3}, pred={pred_vox_mm3} mm3"
            )

        gt_core = (gt_raw == 1) | (gt_raw == 2)
        pred = pred_raw > 0
        gt_lab, n_gt, gt_sizes = component_stats(gt_core)
        pred_lab, n_pred, pred_sizes = component_stats(pred)
        jac, inter = jaccard_matrix(gt_lab, n_gt, gt_sizes, pred_lab, n_pred, pred_sizes)
        matches = match_components(jac, args.jaccard_threshold)
        gt_to_match = {g: (p, j) for g, p, j in matches}
        matched_pred = {p for g, p, j in matches}

        study_rows.append({
            "manifest_index": idx,
            "case_unit": row["CaseUnit"],
            "flouri_patient_id": row.get("FlouriPatientId", ""),
            "study_id": row["AnonStudyID"],
            "timepoint": row["Timepoint"],
            "timepoint_order": row.get("TimepointOrder", np.nan),
            "original_gt_linked": bool(orig_linked.loc[idx]),
            "tumor_core_positive": bool(gt_core.any()),
            "dsc": dice(gt_core, pred),
            "gt_volume_mL": float(gt_core.sum() * gt_vox_mm3 / 1000.0),
            "pred_volume_mL": float(pred.sum() * pred_vox_mm3 / 1000.0),
            "n_gt_lesions": int(n_gt),
            "n_pred_lesions": int(n_pred),
            "n_matched": int(len(matches)),
            "n_false_positive": int(n_pred - len(matched_pred)),
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "prediction_source": pred_source,
        })

        for gid in range(1, n_gt + 1):
            gt_vox = int(gt_sizes[gid])
            vol_ml = gt_vox * gt_vox_mm3 / 1000.0
            if gid in gt_to_match:
                pid, j = gt_to_match[gid]
                overlap = int(inter[gid - 1, pid - 1])
                detected = True
            else:
                pid, j, overlap, detected = None, 0.0, 0, False
            lesion_rows.append({
                "manifest_index": idx,
                "case_unit": row["CaseUnit"],
                "study_id": row["AnonStudyID"],
                "timepoint": row["Timepoint"],
                "original_gt_linked": bool(orig_linked.loc[idx]),
                "gt_lesion_id": gid,
                "gt_voxels": gt_vox,
                "gt_volume_mL": float(vol_ml),
                "detected": bool(detected),
                "matched_pred_id": pid,
                "jaccard": float(j),
                "overlap_voxels": overlap,
            })

    studies = pd.DataFrame(study_rows)
    lesions = pd.DataFrame(lesion_rows)
    studies.to_csv(args.out_dir / "study_metrics.csv", index=False)
    lesions.to_csv(args.out_dir / "lesion_metrics.csv", index=False)

    if studies.empty:
        raise SystemExit(
            "No studies could be scored. Neither PreprocessedSeg nor "
            "Preprocessed_label1_ids resolved to surviving files. "
            "See WARN lines above."
        )

    source_counts = studies["prediction_source"].value_counts().to_dict()
    lines = [
        "Cyprus corrected-reference re-score",
        "==================================",
        f"manifest rows: {len(df)}",
        f"reference masks scored (including zero-core): {len(studies)}",
        f"tumor-core-positive reference studies scored: {int(studies['tumor_core_positive'].sum())}",
        f"prediction sources used: {source_counts}",
        f"Jaccard detection threshold: {args.jaccard_threshold:.2f}",
        "connected components: 26-connected",
    ]

    old_studies = studies[studies["original_gt_linked"] & studies["tumor_core_positive"]].copy()
    old_lesions = lesions[lesions["original_gt_linked"]].copy()
    corrected_studies = studies[studies["tumor_core_positive"]].copy()
    corrected_lesions = lesions.copy()  # zero-core study contributes no lesion rows

    summarize("ORIGINAL LINKED POSITIVE-REFERENCE COHORT", old_studies, old_lesions, lines)
    summarize("CORRECTED 169-STUDY POSITIVE-REFERENCE COHORT", corrected_studies, corrected_lesions, lines)

    # P31 only
    p31s = studies[studies["case_unit"].astype(str).str.upper().eq("P31")].copy()
    lines += ["", "P31 STUDY-LEVEL IMPACT", "----------------------"]
    if len(p31s):
        for _, r in p31s.sort_values("timepoint_order").iterrows():
            lines.append(
                f"  P31:{r['timepoint']} core_positive={bool(r['tumor_core_positive'])} "
                f"DSC={r['dsc']:.4f} GT={r['gt_volume_mL']:.4f}mL "
                f"Pred={r['pred_volume_mL']:.4f}mL GTlesions={int(r['n_gt_lesions'])} "
                f"PredLesions={int(r['n_pred_lesions'])} matched={int(r['n_matched'])} "
                f"FP={int(r['n_false_positive'])}"
            )
    else:
        lines.append("  no P31 rows scored")

    # Manuscript-reproduction checks, intentionally rounded to manuscript precision.
    lines += ["", "MANUSCRIPT REPRODUCTION CHECK (original linked cohort)", "-------------------------------------------------------"]
    if len(old_studies):
        q1, q3 = iqr(old_studies["dsc"])
        fp_rate = old_studies["n_false_positive"].sum() / len(old_studies)
        lines.append(f"  median DSC rounds to 0.78: {round(old_studies['dsc'].median(), 2) == 0.78} ({old_studies['dsc'].median():.4f})")
        lines.append(f"  IQR observed: {q1:.4f}-{q3:.4f} (manuscript 0.54-0.89)")
        lines.append(f"  FP/study observed: {fp_rate:.4f} (manuscript 1.51)")

        tier_targets = [(0.0,0.05,0.17),(0.05,0.5,0.66),(0.5,4.0,0.88),(4.0,float('inf'),0.91)]
        for lo, hi, target in tier_targets:
            if math.isinf(hi): sub=old_lesions[old_lesions.gt_volume_mL>=lo]
            else: sub=old_lesions[(old_lesions.gt_volume_mL>=lo)&(old_lesions.gt_volume_mL<hi)]
            sens=float(sub.detected.mean()) if len(sub) else float('nan')
            lines.append(f"  sensitivity [{lo},{'inf' if math.isinf(hi) else hi}) = {sens:.4f}; manuscript {target:.2f}; rounded match={round(sens,2)==target if not math.isnan(sens) else False}")

    out_summary = args.out_dir / "rescore_summary.txt"
    out_summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote:\n  {args.out_dir/'study_metrics.csv'}\n  {args.out_dir/'lesion_metrics.csv'}\n  {out_summary}")


if __name__ == "__main__":
    main()
