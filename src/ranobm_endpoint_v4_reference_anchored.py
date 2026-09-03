#!/usr/bin/env python3
"""
ranobm_endpoint_v4_reference_anchored.py
=======================================

Reference-anchored primary modified RANO-BM endpoint for the Cyprus/Flouri
longitudinal brain-metastasis validation.

WHY V4
------
The public reference masks are expert-reviewed but NOT lesion-instance-tracked.
QC showed that a reference connected component can disappear from the temporal
track at an annotated timepoint even though the lesion is visibly still present
(e.g. component merge/split/tracking ambiguity). Therefore:

  * Missing/temporally-unmatched REFERENCE components are NOT interpreted as
    true disappearance and are NOT assigned diameter 0.
  * The REFERENCE trajectory is defined only by timepoints where that reference
    lesion is positively represented by a component.
  * The matched MODEL lesion is measured at those SAME reference-positive
    timepoints.
  * If the model lesion is absent at a reference-positive FOLLOW-UP timepoint,
    the model diameter is explicitly 0 mm (a real model miss at a known-positive
    reference observation).
  * If the matched model is absent at the reference-defined ENTRY timepoint,
    the lesion is reported separately as a baseline/entry detection failure and
    is NOT included in the 3x3 trajectory-agreement contingency.
  * A measurable reference lesion must have >=2 positive reference observations
    to enter trajectory agreement.

This deliberately preserves the original component construction, temporal
tracking, union-Jaccard matching, diameter measurement, and RANO thresholds from
ranobm_endpoint.py. Only the longitudinal sampling/eligibility rule changes.

Outputs:
  contingency_ranobm.csv
  per_lesion_ranobm.csv
  per_timepoint_ranobm.csv
  lesion_locators.json
  ranobm_v4_summary.txt

Place this script beside ranobm_endpoint.py.

Example
-------
BASE=/path/to/cyprus_validation

python ranobm_endpoint_v4_reference_anchored.py \
  --output-dir "$BASE" \
  --dataset-csv "$BASE/dataset_relinked.csv" \
  --out-dir "$BASE/ranobm_out_relinked_v4"
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Reuse the already-validated implementation for image loading, reference
# connected components, temporal linkage, union-Jaccard matching, longest axial
# diameter, and RANO thresholds.
import ranobm_endpoint as base


CATS = ["improved", "stable", "progressed"]
MEASURABLE_MM = base.MEASURABLE_MM
MATCH_JACCARD = base.MATCH_JACCARD


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p * 100, (c - h) * 100, (c + h) * 100


def positive_diameters(track, spacing):
    """Diameter only where this track has an actual positive component."""
    return {
        int(ti): float(base.in_plane_longest_diameter(mask, spacing))
        for ti, mask in track.items()
        if mask is not None and mask.sum() > 0
    }


def _allocated_cpus():
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def _score_case(task):
    case_unit, g, tracked_root = task
    g = g.sort_values("TimepointOrder").copy()

    pred_tp = []
    ref_tp = []
    evaluable = []
    spacing = None

    # Keep a per-index record so the audit CSV can name the original timepoint.
    row_by_ti = {}

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
        gcore = np.isin(garr, base.GT_CORE_LABELS)
        ref_tp.append(base.cc_lesions(gcore))
        evaluable.append(True)

    eval_tps = [i for i, ok in enumerate(evaluable) if ok]
    result = {
        "case_unit": case_unit,
        "contingency": np.zeros((3, 3), dtype=int),
        "per_lesion": [],
        "per_tp": [],
        "locators": {},
        "n_evaluable_tps": len(eval_tps),
        "n_ref_zero_core_tps": sum(1 for i in eval_tps if len(ref_tp[i]) == 0),
        "n_meas_ref": 0,
        "n_insufficient_ref_followup": 0,
        "n_undetected": 0,
        "n_entry_missed": 0,
        "n_scored": 0,
        "n_model_followup_zeros": 0,
        "n_ambiguous_ref_gap_tps": 0,
        "n_pred_only_new": 0,
    }

    if spacing is None or len(eval_tps) < 2:
        return result

    first_eval_tp = min(eval_tps)

    # Same temporal tracking logic as the original endpoint.
    if base.TRACK_PRED == "label":
        ptracks = {}
        for ti, lesions in enumerate(pred_tp):
            for lid, m in lesions.items():
                ptracks.setdefault(int(lid), {})[ti] = m
    else:
        ptracks = base.greedy_track(pred_tp)

    rtracks = base.greedy_track(ref_tp)

    # Reference eligibility is determined by the first positive reference
    # observation. Reference gaps are left missing, never zero-filled.
    rdiam = {rid: positive_diameters(tr, spacing) for rid, tr in rtracks.items()}
    pdiam_positive = {pid: positive_diameters(tr, spacing) for pid, tr in ptracks.items()}

    measurable_rids = []
    for rid, d in rdiam.items():
        if not d:
            continue
        entry_tp = min(d)
        if d[entry_tp] >= MEASURABLE_MM:
            measurable_rids.append(rid)

    result["n_meas_ref"] = len(measurable_rids)

    # Match measurable reference tracks to prediction tracks exactly as before:
    # union-of-positive-voxels Jaccard + one-to-one Hungarian assignment.
    pids = list(ptracks.keys())
    p_union_idx = {pid: base.union_idx(ptracks[pid].values()) for pid in pids}
    r_union_idx = {rid: base.union_idx(rtracks[rid].values()) for rid in measurable_rids}

    assign = {}
    if measurable_rids and pids:
        J = np.zeros((len(measurable_rids), len(pids)), dtype=float)
        for i, rid in enumerate(measurable_rids):
            for j, pid in enumerate(pids):
                J[i, j] = base.jaccard_idx(r_union_idx[rid], p_union_idx[pid])
        rr, cc = linear_sum_assignment(-J)
        for i, j in zip(rr, cc):
            if J[i, j] >= MATCH_JACCARD:
                assign[measurable_rids[i]] = pids[j]

    matched_pids = set(assign.values())

    for rid in measurable_rids:
        rd = rdiam[rid]
        ref_positive_tps = sorted(rd)
        entry_tp = ref_positive_tps[0]
        last_ref_tp = ref_positive_tps[-1]
        ref_new = entry_tp > first_eval_tp

        # Count annotated/evaluable gaps INSIDE the observed reference trajectory.
        ref_gap_tps = [
            ti for ti in eval_tps
            if entry_tp < ti < last_ref_tp and ti not in rd
        ]
        result["n_ambiguous_ref_gap_tps"] += len(ref_gap_tps)

        # Need at least two positive reference observations for a longitudinal
        # response trajectory. A single positive scan is not "stable"; it is NE.
        if len(ref_positive_tps) < 2:
            result["n_insufficient_ref_followup"] += 1
            result["per_lesion"].append({
                "case_unit": case_unit,
                "ref_id": rid,
                "pred_id": assign.get(rid, "NA"),
                "ref_cat": "NE",
                "pred_cat": "NE",
                "ref_new": ref_new,
                "status": "insufficient_reference_followup",
                "ref_n_positive_tps": len(ref_positive_tps),
                "ref_n_gap_tps": len(ref_gap_tps),
                "model_followup_zero_count": 0,
                **_summary_fields("ref", rd),
                **_summary_fields("pred", {}),
            })
            continue

        ref_cat = base.classify_track(rd)
        pid = assign.get(rid)

        if pid is None:
            result["n_undetected"] += 1
            result["per_lesion"].append({
                "case_unit": case_unit,
                "ref_id": rid,
                "pred_id": "NA",
                "ref_cat": ref_cat,
                "pred_cat": "NE",
                "ref_new": ref_new,
                "status": "undetected",
                "ref_n_positive_tps": len(ref_positive_tps),
                "ref_n_gap_tps": len(ref_gap_tps),
                "model_followup_zero_count": 0,
                **_summary_fields("ref", rd),
                **_summary_fields("pred", {}),
            })
            # Audit reference-positive timepoints even with no matched prediction.
            for ti in ref_positive_tps:
                _append_tp_row(
                    result["per_tp"], case_unit, rid, "NA", ti, row_by_ti,
                    ref_present=True, ref_diam=rd[ti],
                    pred_present=False, pred_diam=np.nan,
                    included=False, note="undetected_no_matched_prediction"
                )
            continue

        pt = ptracks[pid]

        # Primary trajectory requires a model delineation at the reference-defined
        # entry/baseline timepoint. If it is absent there, report as an entry
        # detection failure rather than inventing a model baseline later.
        if entry_tp not in pt or pt[entry_tp].sum() == 0:
            result["n_entry_missed"] += 1
            result["per_lesion"].append({
                "case_unit": case_unit,
                "ref_id": rid,
                "pred_id": pid,
                "ref_cat": ref_cat,
                "pred_cat": "NE",
                "ref_new": ref_new,
                "status": "entry_detection_failure",
                "ref_n_positive_tps": len(ref_positive_tps),
                "ref_n_gap_tps": len(ref_gap_tps),
                "model_followup_zero_count": 0,
                **_summary_fields("ref", rd),
                **_summary_fields("pred", {}),
            })
            for ti in ref_positive_tps:
                pred_present = ti in pt and pt[ti].sum() > 0
                pd = (
                    base.in_plane_longest_diameter(pt[ti], spacing)
                    if pred_present else 0.0
                )
                _append_tp_row(
                    result["per_tp"], case_unit, rid, pid, ti, row_by_ti,
                    ref_present=True, ref_diam=rd[ti],
                    pred_present=pred_present, pred_diam=pd,
                    included=False,
                    note="entry_detection_failure"
                    if ti == entry_tp else "not_scored_due_entry_failure"
                )
            continue

        # REFERENCE-ANCHORED MODEL TRAJECTORY:
        # sample model only at reference-positive observations. Subsequent model
        # absence at a known-positive reference timepoint is a real miss => 0 mm.
        pd = {}
        followup_zero_count = 0
        for ti in ref_positive_tps:
            pred_present = ti in pt and pt[ti].sum() > 0
            if pred_present:
                d = float(base.in_plane_longest_diameter(pt[ti], spacing))
            else:
                d = 0.0
                if ti != entry_tp:
                    followup_zero_count += 1
                    result["n_model_followup_zeros"] += 1
            pd[ti] = d

            _append_tp_row(
                result["per_tp"], case_unit, rid, pid, ti, row_by_ti,
                ref_present=True, ref_diam=rd[ti],
                pred_present=pred_present, pred_diam=d,
                included=True,
                note="model_followup_miss_zero"
                if (ti != entry_tp and not pred_present) else "scored"
            )

        # Also emit ambiguous reference-gap observations for audit, but do not
        # use them in either trajectory.
        for ti in ref_gap_tps:
            pred_present = ti in pt and pt[ti].sum() > 0
            pred_d = (
                float(base.in_plane_longest_diameter(pt[ti], spacing))
                if pred_present else np.nan
            )
            _append_tp_row(
                result["per_tp"], case_unit, rid, pid, ti, row_by_ti,
                ref_present=False, ref_diam=np.nan,
                pred_present=pred_present, pred_diam=pred_d,
                included=False,
                note="reference_component_absent_uninterpretable"
            )

        pred_cat = base.classify_track(pd)
        result["contingency"][CATS.index(ref_cat), CATS.index(pred_cat)] += 1
        result["n_scored"] += 1

        result["per_lesion"].append({
            "case_unit": case_unit,
            "ref_id": rid,
            "pred_id": pid,
            "ref_cat": ref_cat,
            "pred_cat": pred_cat,
            "ref_new": ref_new,
            "status": "matched",
            "ref_n_positive_tps": len(ref_positive_tps),
            "ref_n_gap_tps": len(ref_gap_tps),
            "model_followup_zero_count": followup_zero_count,
            **_summary_fields("ref", rd),
            **_summary_fields("pred", pd),
        })

        result["locators"][f"{case_unit}|ref{rid}_pred{pid}"] = {
            "ref": base.lesion_locator(rtracks[rid]),
            "pred": base.lesion_locator(ptracks[pid]),
        }

    # Preserve the original exploratory model-only-new-lesion definition so the
    # downstream reader-study inventory can be compared with prior runs.
    for pid, track in ptracks.items():
        if pid in matched_pids:
            continue
        pd_pos = pdiam_positive.get(pid, {})
        if not pd_pos:
            continue
        entry_tp = min(pd_pos)
        entry_d = pd_pos[entry_tp]
        pnew = entry_tp > first_eval_tp
        pmeas = entry_d >= MEASURABLE_MM
        if pnew and pmeas:
            result["n_pred_only_new"] += 1
            result["per_lesion"].append({
                "case_unit": case_unit,
                "ref_id": "NA",
                "pred_id": pid,
                "ref_cat": "none",
                "pred_cat": "progressed",
                "ref_new": True,
                "status": "pred_only_new",
                "ref_n_positive_tps": 0,
                "ref_n_gap_tps": 0,
                "model_followup_zero_count": 0,
                **_summary_fields("ref", {}),
                **_summary_fields("pred", pd_pos),
            })

    return result


def _summary_fields(prefix, diam):
    """Compatible summary fields used by existing montage/reader scripts."""
    if not diam:
        return {
            f"{prefix}_base_mm": np.nan,
            f"{prefix}_last_mm": np.nan,
            f"{prefix}_maxfu_mm": np.nan,
            f"{prefix}_minfu_mm": np.nan,
            f"{prefix}_pct_base": np.nan,
            f"{prefix}_pct_nadir": np.nan,
        }

    vals = base.diam_summary(diam)
    names = ["base_mm", "last_mm", "maxfu_mm", "minfu_mm", "pct_base", "pct_nadir"]
    return {f"{prefix}_{name}": val for name, val in zip(names, vals)}


def _append_tp_row(
    out, case_unit, rid, pid, ti, row_by_ti,
    ref_present, ref_diam, pred_present, pred_diam,
    included, note
):
    row = row_by_ti[ti]
    out.append({
        "case_unit": case_unit,
        "ref_id": rid,
        "pred_id": pid,
        "tp_index": int(ti),
        "timepoint": str(row.get("Timepoint", "")),
        "timepoint_order": str(row.get("TimepointOrder", "")),
        "study_id": str(row.get("AnonStudyID", "")),
        "reference_component_present": bool(ref_present),
        "reference_diameter_mm": ref_diam,
        "prediction_component_present": bool(pred_present),
        "prediction_diameter_mm": pred_diam,
        "included_in_trajectory": bool(included),
        "note": note,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="pipeline output directory")
    ap.add_argument("--dataset-csv", default=None)
    ap.add_argument("--tracked-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    ds_csv = Path(args.dataset_csv or output_dir / "dataset.csv")
    tracked_root = str(Path(args.tracked_root or output_dir / "04_tracked"))
    out_dir = Path(args.out_dir or output_dir / "ranobm_out_v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ds_csv, dtype=str)
    if "NormalizedSeriesDescription" in df.columns:
        df = df[df["NormalizedSeriesDescription"] == "T1Post"].copy()
    df["TimepointOrder"] = pd.to_numeric(df["TimepointOrder"], errors="raise").astype(int)

    groups = list(df.groupby("AnonPatientID"))
    tasks = [(cu, g, tracked_root) for cu, g in groups]
    workers = args.workers or _allocated_cpus()
    workers = max(1, min(int(workers), len(tasks)))

    print(f"Scoring {len(tasks)} case units with {workers} worker(s)...", flush=True)

    if workers == 1:
        partials = [_score_case(t) for t in tasks]
    else:
        with mp.Pool(workers) as pool:
            partials = list(pool.imap_unordered(_score_case, tasks))

    partials.sort(key=lambda r: r["case_unit"])

    contingency = np.zeros((3, 3), dtype=int)
    per_lesion = []
    per_tp = []
    locators = {}

    counters = {
        "n_evaluable_tps": 0,
        "n_ref_zero_core_tps": 0,
        "n_meas_ref": 0,
        "n_insufficient_ref_followup": 0,
        "n_undetected": 0,
        "n_entry_missed": 0,
        "n_scored": 0,
        "n_model_followup_zeros": 0,
        "n_ambiguous_ref_gap_tps": 0,
        "n_pred_only_new": 0,
    }

    for r in partials:
        contingency += r["contingency"]
        per_lesion.extend(r["per_lesion"])
        per_tp.extend(r["per_tp"])
        locators.update(r["locators"])
        for k in counters:
            counters[k] += int(r[k])

    per_lesion_df = pd.DataFrame(per_lesion)
    if not per_lesion_df.empty:
        per_lesion_df = per_lesion_df.sort_values(
            ["case_unit", "status", "ref_id", "pred_id"],
            kind="stable"
        )
    per_tp_df = pd.DataFrame(per_tp)
    if not per_tp_df.empty:
        per_tp_df = per_tp_df.sort_values(
            ["case_unit", "ref_id", "pred_id", "tp_index"],
            kind="stable"
        )

    total = int(contingency.sum())
    diag = int(np.trace(contingency))
    p, lo, hi = wilson(diag, total)

    lines = []
    lines.append("=== V4 PRIMARY modified RANO-BM: reference-anchored ===")
    lines.append(f"manifest T1-post rows: {len(df)}")
    lines.append(f"evaluable prediction+reference timepoints: {counters['n_evaluable_tps']}")
    lines.append(f"annotated zero-core studies: {counters['n_ref_zero_core_tps']}")
    lines.append(f"measurable reference lesions at entry: {counters['n_meas_ref']}")
    lines.append(f"insufficient reference-positive follow-up: {counters['n_insufficient_ref_followup']}")
    lines.append(f"undetected / no matched model track: {counters['n_undetected']}")
    lines.append(f"matched but model absent at reference entry: {counters['n_entry_missed']}")
    lines.append(f"scored matched trajectories: {counters['n_scored']}")
    lines.append(f"model follow-up misses assigned 0 mm: {counters['n_model_followup_zeros']}")
    lines.append(f"ambiguous reference-component gap timepoints skipped: {counters['n_ambiguous_ref_gap_tps']}")
    lines.append(f"model-only new measurable lesions (legacy definition): {counters['n_pred_only_new']}")
    lines.append("")
    lines.append("Contingency rows=reference cols=prediction; order = " + str(CATS))
    lines.append(str(contingency))
    lines.append("")
    lines.append(
        f"Overall agreement: {diag}/{total} = {p:.1f}% "
        f"(95% CI {lo:.0f}-{hi:.0f})"
    )

    for i, cat in enumerate(CATS):
        n = int(contingency[i].sum())
        x = int(contingency[i, i])
        if n:
            pp, ll, hh = wilson(x, n)
            lines.append(
                f"  {cat:11s}: {x}/{n} = {pp:.1f}% "
                f"(95% CI {ll:.0f}-{hh:.0f})"
            )
        else:
            lines.append(f"  {cat:11s}: n=0")

    direct_reversals = int(contingency[0, 2] + contingency[2, 0])
    lines.append(f"Direct improved<->progressed reversals: {direct_reversals}")

    summary = "\n".join(lines)
    print(summary)

    pd.DataFrame(
        contingency,
        index=[f"ref_{c}" for c in CATS],
        columns=[f"pred_{c}" for c in CATS],
    ).to_csv(out_dir / "contingency_ranobm.csv")

    per_lesion_df.to_csv(out_dir / "per_lesion_ranobm.csv", index=False)
    per_tp_df.to_csv(out_dir / "per_timepoint_ranobm.csv", index=False)

    with open(out_dir / "lesion_locators.json", "w") as f:
        json.dump(locators, f)

    (out_dir / "ranobm_v4_summary.txt").write_text(summary + "\n")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
