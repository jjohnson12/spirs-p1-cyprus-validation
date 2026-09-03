#!/usr/bin/env python3
"""
audit_ranobm_new_lesions_v4.py
==============================

Audit NEW measurable lesions separately from the v4 reference-anchored
index-target trajectory endpoint.

For each reference-new measurable lesion:
  * identifies its first positive reference timepoint;
  * reports whether a union-Jaccard matched prediction track exists;
  * tests whether that matched prediction is present at the SAME timepoint;
  * computes same-timepoint Jaccard and prediction longest diameter;
  * reports:
      - same_timepoint_detected_jaccard10
      - same_timepoint_model_measurable_10mm

Also reproduces the legacy model-only new measurable lesion set and can compare
it against the original reader-study per_lesion_ranobm.csv.

Place beside ranobm_endpoint.py.

Example:
BASE=/path/to/cyprus_validation

python audit_ranobm_new_lesions_v4.py \
  --output-dir "$BASE" \
  --dataset-csv "$BASE/dataset_relinked.csv" \
  --out-dir "$BASE/ranobm_new_lesion_audit_v4" \
  --old-per-lesion "$BASE/ranobm_out/per_lesion_ranobm.csv"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import ranobm_endpoint as base


def _score_case(case_unit, g, tracked_root):
    g = g.sort_values("TimepointOrder").copy()

    pred_tp, ref_tp, evaluable = [], [], []
    row_by_ti = {}
    spacing = None

    for ti, (_, row) in enumerate(g.iterrows()):
        row_by_ti[ti] = row
        study_id = str(row["AnonStudyID"])
        pred_path = base.find_pred_ids(tracked_root, case_unit, study_id, row)
        gt_path = str(row.get("GTMaskPath") or "").strip()

        pred_ok = bool(pred_path and os.path.exists(pred_path))
        gt_ok = bool(gt_path and gt_path != "nan" and os.path.exists(gt_path))

        if not pred_ok or not gt_ok:
            pred_tp.append({})
            ref_tp.append({})
            evaluable.append(False)
            continue

        parr, sp = base.load(pred_path)
        garr, _ = base.load(gt_path)
        spacing = sp

        pred_tp.append(
            base.pred_lesions(parr)
            if base.TRACK_PRED == "label"
            else base.cc_lesions(parr > 0)
        )
        ref_tp.append(base.cc_lesions(np.isin(garr, base.GT_CORE_LABELS)))
        evaluable.append(True)

    eval_tps = [i for i, ok in enumerate(evaluable) if ok]
    if spacing is None or not eval_tps:
        return [], []

    first_eval_tp = min(eval_tps)

    # Prediction tracks
    if base.TRACK_PRED == "label":
        ptracks = {}
        for ti, lesions in enumerate(pred_tp):
            for lid, m in lesions.items():
                ptracks.setdefault(int(lid), {})[ti] = m
    else:
        ptracks = base.greedy_track(pred_tp)

    # Reference tracks
    rtracks = base.greedy_track(ref_tp)

    def positive_diameters(track):
        return {
            int(t): float(base.in_plane_longest_diameter(m, spacing))
            for t, m in track.items()
            if m is not None and m.sum() > 0
        }

    rdiam = {rid: positive_diameters(tr) for rid, tr in rtracks.items()}
    pdiam = {pid: positive_diameters(tr) for pid, tr in ptracks.items()}

    measurable_rids = []
    for rid, d in rdiam.items():
        if not d:
            continue
        t0 = min(d)
        if d[t0] >= base.MEASURABLE_MM:
            measurable_rids.append(rid)

    pids = list(ptracks)
    r_union = {rid: base.union_idx(rtracks[rid].values()) for rid in measurable_rids}
    p_union = {pid: base.union_idx(ptracks[pid].values()) for pid in pids}

    assign = {}
    union_j = {}
    if measurable_rids and pids:
        J = np.zeros((len(measurable_rids), len(pids)), float)
        for i, rid in enumerate(measurable_rids):
            for j, pid in enumerate(pids):
                J[i, j] = base.jaccard_idx(r_union[rid], p_union[pid])
        rr, cc = linear_sum_assignment(-J)
        for i, j in zip(rr, cc):
            if J[i, j] >= base.MATCH_JACCARD:
                rid, pid = measurable_rids[i], pids[j]
                assign[rid] = pid
                union_j[rid] = float(J[i, j])

    matched_pids = set(assign.values())

    ref_new_rows = []
    for rid in measurable_rids:
        rd = rdiam[rid]
        entry_tp = min(rd)
        if entry_tp <= first_eval_tp:
            continue

        pid = assign.get(rid)
        pred_present_same = False
        pred_d_same = np.nan
        j_same = 0.0

        if pid is not None and entry_tp in ptracks[pid]:
            pm = ptracks[pid][entry_tp]
            pred_present_same = bool(pm.sum() > 0)
            if pred_present_same:
                pred_d_same = float(base.in_plane_longest_diameter(pm, spacing))
                j_same = float(
                    base.jaccard_idx(
                        base.vox_index_set(rtracks[rid][entry_tp]),
                        base.vox_index_set(pm),
                    )
                )

        ref_row = row_by_ti[entry_tp]
        ref_new_rows.append({
            "case_unit": case_unit,
            "ref_id": int(rid),
            "matched_pred_id": int(pid) if pid is not None else np.nan,
            "first_eval_tp_index": int(first_eval_tp),
            "ref_first_positive_tp_index": int(entry_tp),
            "timepoint": str(ref_row.get("Timepoint", "")),
            "study_id": str(ref_row.get("AnonStudyID", "")),
            "reference_diameter_mm": float(rd[entry_tp]),
            "union_track_jaccard": union_j.get(rid, np.nan),
            "prediction_present_same_timepoint": pred_present_same,
            "prediction_diameter_mm_same_timepoint": pred_d_same,
            "same_timepoint_jaccard": j_same,
            "same_timepoint_detected_jaccard10": bool(
                pred_present_same and j_same >= base.MATCH_JACCARD
            ),
            "same_timepoint_model_measurable_10mm": bool(
                pred_present_same
                and j_same >= base.MATCH_JACCARD
                and pred_d_same >= base.MEASURABLE_MM
            ),
            "n_reference_positive_timepoints": len(rd),
        })

    # Legacy model-only NEW measurable lesions: not matched to a measurable
    # reference track, first positive after first evaluable TP, >=10 mm at entry.
    pred_only_rows = []
    for pid, pd_ in pdiam.items():
        if pid in matched_pids or not pd_:
            continue
        entry_tp = min(pd_)
        if entry_tp <= first_eval_tp:
            continue
        entry_d = pd_[entry_tp]
        if entry_d < base.MEASURABLE_MM:
            continue
        row = row_by_ti[entry_tp]
        pred_only_rows.append({
            "case_unit": case_unit,
            "pred_id": int(pid),
            "first_positive_tp_index": int(entry_tp),
            "timepoint": str(row.get("Timepoint", "")),
            "study_id": str(row.get("AnonStudyID", "")),
            "prediction_diameter_mm": float(entry_d),
        })

    return ref_new_rows, pred_only_rows


def _norm_id(x):
    if pd.isna(x):
        return ""
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(x).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-csv", default=None)
    ap.add_argument("--tracked-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--old-per-lesion", default=None,
                    help="optional original reader-study per_lesion_ranobm.csv")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    ds_csv = Path(args.dataset_csv or output_dir / "dataset.csv")
    tracked_root = str(Path(args.tracked_root or output_dir / "04_tracked"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ds_csv, dtype=str)
    if "NormalizedSeriesDescription" in df.columns:
        df = df[df["NormalizedSeriesDescription"] == "T1Post"].copy()
    df["TimepointOrder"] = pd.to_numeric(df["TimepointOrder"], errors="raise").astype(int)

    ref_rows, po_rows = [], []
    for case_unit, g in df.groupby("AnonPatientID"):
        r, p = _score_case(case_unit, g, tracked_root)
        ref_rows.extend(r)
        po_rows.extend(p)

    ref = pd.DataFrame(ref_rows)
    po = pd.DataFrame(po_rows)

    ref.to_csv(out_dir / "reference_new_same_timepoint_audit.csv", index=False)
    po.to_csv(out_dir / "model_only_new_legacy_audit.csv", index=False)

    n = len(ref)
    n_track = int(ref["matched_pred_id"].notna().sum()) if n else 0
    n_same = int(ref["same_timepoint_detected_jaccard10"].sum()) if n else 0
    n_meas = int(ref["same_timepoint_model_measurable_10mm"].sum()) if n else 0

    lines = []
    lines.append("=== V4 NEW-LESION AUDIT ===")
    lines.append(f"Reference-new measurable lesions: {n}")
    lines.append(f"Any union-track match: {n_track}/{n}")
    lines.append(f"Detected at reference first-appearance timepoint (Jaccard >=0.10): {n_same}/{n}")
    lines.append(f"Detected AND model-measurable >=10 mm at first appearance: {n_meas}/{n}")
    lines.append(f"Legacy model-only new measurable lesions: {len(po)}")

    if n:
        lines.append("")
        lines.append("Reference-new lesion detail:")
        for _, r in ref.sort_values(["case_unit", "ref_id"]).iterrows():
            pid = "none" if pd.isna(r["matched_pred_id"]) else str(int(r["matched_pred_id"]))
            pdmm = (
                "NA"
                if pd.isna(r["prediction_diameter_mm_same_timepoint"])
                else f"{r['prediction_diameter_mm_same_timepoint']:.1f}"
            )
            lines.append(
                f"  {r['case_unit']} ref{int(r['ref_id'])}: "
                f"GT {r['reference_diameter_mm']:.1f} mm at {r['timepoint']}; "
                f"pred={pid}; same-TP J={r['same_timepoint_jaccard']:.3f}; "
                f"pred_diam={pdmm} mm; "
                f"detected={bool(r['same_timepoint_detected_jaccard10'])}; "
                f"model_measurable={bool(r['same_timepoint_model_measurable_10mm'])}"
            )

    # Compare model-only set with the original reader-study set, if available.
    if args.old_per_lesion and Path(args.old_per_lesion).exists():
        old = pd.read_csv(args.old_per_lesion)
        old = old[old["status"] == "pred_only_new"].copy()
        old_keys = {
            (str(r["case_unit"]), _norm_id(r["pred_id"]))
            for _, r in old.iterrows()
        }
        new_keys = {
            (str(r["case_unit"]), _norm_id(r["pred_id"]))
            for _, r in po.iterrows()
        }

        lines.append("")
        lines.append("MODEL-ONLY NEW SET VS ORIGINAL READER STUDY")
        lines.append(f"Original reader-study model-only set: {len(old_keys)}")
        lines.append(f"Current legacy v4 model-only set: {len(new_keys)}")
        lines.append(f"Exact set match: {old_keys == new_keys}")

        only_old = sorted(old_keys - new_keys)
        only_new = sorted(new_keys - old_keys)
        if only_old:
            lines.append("Only in original:")
            lines.extend([f"  {c} pred{p}" for c, p in only_old])
        if only_new:
            lines.append("Only in current:")
            lines.extend([f"  {c} pred{p}" for c, p in only_new])

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "new_lesion_audit_summary.txt").write_text(summary + "\n")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
