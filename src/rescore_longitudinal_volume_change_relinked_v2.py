#!/usr/bin/env python3
"""
rescore_longitudinal_volume_change_relinked_v2.py

Recompute the historical CASE-UNIT TOTAL TUMOR-CORE volume-change endpoint
using relinked Cyprus GT masks and surviving tracked SPIRS-P1 prediction maps.

Historical endpoint:
  CR  : v0 > 0 and v1 == 0
  PR  : v1 <= 0.35 * v0
  PD  : v1 >= 1.40 * v0 and (v1 - v0) > 0.10 mL
  SD  : otherwise
  NEW : v0 == 0 and v1 > 0

For a faithful old-166 control, use dataset_relinked.csv and exclude P31.
The original dataset.csv may point to GT files that were later purged.

GT tumor core = Cyprus labels 1 + 2.
Prediction tumor core = tracked prediction labels > 0.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import ranobm_endpoint as base

GT_CORE_LABELS = (1, 2)

STUDY_COLUMNS = [
    "case_unit_id", "study_id", "timepoint", "timepoint_order",
    "gt_volume_mL", "pred_volume_mL", "gt_core_positive", "pred_positive",
    "gt_mask_path", "pred_ids_path",
]


def volume_change_category(v0: float, v1: float) -> str:
    if v0 == 0 and v1 == 0:
        return "SD"
    if v0 == 0 and v1 > 0:
        return "NEW"
    if v1 == 0 and v0 > 0:
        return "CR"
    if v1 <= 0.35 * v0:
        return "PR"
    if v1 >= 1.40 * v0 and (v1 - v0) > 0.10:
        return "PD"
    return "SD"


def voxel_volume_ml(img):
    z = img.header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2] / 1000.0)


def load_volume_ml(path, positive_rule):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    mask = positive_rule(arr)
    vv = voxel_volume_ml(img)
    return (
        float(np.count_nonzero(mask) * vv),
        tuple(arr.shape),
        tuple(float(x) for x in img.header.get_zooms()[:3]),
    )


def collect_per_study(df, tracked_root):
    rows, warnings = [], []
    audit = {
        "input_rows": int(len(df)),
        "blank_gt_path": 0,
        "missing_gt_file": 0,
        "missing_prediction": 0,
        "included": 0,
    }

    for _, row in df.iterrows():
        case_unit = str(row["AnonPatientID"])
        study_id = str(row["AnonStudyID"])
        gt_path = str(row.get("GTMaskPath") or "").strip()

        if not gt_path or gt_path.lower() == "nan":
            audit["blank_gt_path"] += 1
            continue
        if not os.path.exists(gt_path):
            audit["missing_gt_file"] += 1
            warnings.append(f"missing GT: {case_unit} {study_id} {gt_path}")
            continue

        pred_path = base.find_pred_ids(tracked_root, case_unit, study_id, row)
        if not pred_path or not os.path.exists(pred_path):
            audit["missing_prediction"] += 1
            warnings.append(f"missing prediction: {case_unit} {study_id} {pred_path}")
            continue

        gt_v, gt_shape, gt_zooms = load_volume_ml(
            gt_path, lambda a: np.isin(a, GT_CORE_LABELS)
        )
        pred_v, pred_shape, pred_zooms = load_volume_ml(
            pred_path, lambda a: a > 0
        )

        if gt_shape != pred_shape:
            warnings.append(
                f"shape mismatch {case_unit} {study_id}: GT {gt_shape} vs pred {pred_shape}"
            )
        if any(abs(a - b) > 1e-5 for a, b in zip(gt_zooms, pred_zooms)):
            warnings.append(
                f"spacing mismatch {case_unit} {study_id}: GT {gt_zooms} vs pred {pred_zooms}"
            )

        rows.append({
            "case_unit_id": case_unit,
            "study_id": study_id,
            "timepoint": str(row.get("Timepoint", "")),
            "timepoint_order": int(float(row.get("TimepointOrder", 0))),
            "gt_volume_mL": gt_v,
            "pred_volume_mL": pred_v,
            "gt_core_positive": bool(gt_v > 0),
            "pred_positive": bool(pred_v > 0),
            "gt_mask_path": gt_path,
            "pred_ids_path": str(pred_path),
        })
        audit["included"] += 1

    out = pd.DataFrame(rows, columns=STUDY_COLUMNS)
    if not out.empty:
        out = out.sort_values(
            ["case_unit_id", "timepoint_order", "study_id"], kind="stable"
        ).reset_index(drop=True)

    return out, warnings, audit


def compute_longitudinal(studies):
    pair_columns = [
        "case_unit_id", "study_a", "study_b", "timepoint_a", "timepoint_b",
        "timepoint_order_a", "timepoint_order_b", "originally_adjacent_order",
        "pred_volume_a_mL", "pred_volume_b_mL", "gt_volume_a_mL", "gt_volume_b_mL",
        "pred_delta_mL", "gt_delta_mL", "delta_abs_error_mL",
        "pred_pct_change", "gt_pct_change", "pred_category", "gt_category",
        "category_agree", "gt_a_zero_core", "gt_b_zero_core",
    ]
    rows = []
    if studies.empty:
        return pd.DataFrame(columns=pair_columns)

    for case_unit, g in studies.groupby("case_unit_id", sort=True):
        recs = g.sort_values("timepoint_order").to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            pv0, pv1 = float(a["pred_volume_mL"]), float(b["pred_volume_mL"])
            gv0, gv1 = float(a["gt_volume_mL"]), float(b["gt_volume_mL"])
            pdlt, gdlt = pv1 - pv0, gv1 - gv0
            pcat = volume_change_category(pv0, pv1)
            gcat = volume_change_category(gv0, gv1)

            rows.append({
                "case_unit_id": case_unit,
                "study_a": a["study_id"],
                "study_b": b["study_id"],
                "timepoint_a": a["timepoint"],
                "timepoint_b": b["timepoint"],
                "timepoint_order_a": a["timepoint_order"],
                "timepoint_order_b": b["timepoint_order"],
                "originally_adjacent_order":
                    int(b["timepoint_order"]) - int(a["timepoint_order"]) == 1,
                "pred_volume_a_mL": pv0,
                "pred_volume_b_mL": pv1,
                "gt_volume_a_mL": gv0,
                "gt_volume_b_mL": gv1,
                "pred_delta_mL": pdlt,
                "gt_delta_mL": gdlt,
                "delta_abs_error_mL": abs(pdlt - gdlt),
                "pred_pct_change": 100.0 * pdlt / pv0 if pv0 > 0 else np.nan,
                "gt_pct_change": 100.0 * gdlt / gv0 if gv0 > 0 else np.nan,
                "pred_category": pcat,
                "gt_category": gcat,
                "category_agree": bool(pcat == gcat),
                "gt_a_zero_core": bool(gv0 == 0),
                "gt_b_zero_core": bool(gv1 == 0),
            })

    return pd.DataFrame(rows, columns=pair_columns)


def iqr(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (float("nan"),) * 3
    return (
        float(np.median(a)),
        float(np.percentile(a, 25)),
        float(np.percentile(a, 75)),
    )


def summarize(studies, pairs, audit, excluded_cases):
    n = len(pairs)
    agree = int(pairs["category_agree"].sum()) if n else 0
    med, q1, q3 = iqr(pairs["delta_abs_error_mL"]) if n else (np.nan, np.nan, np.nan)
    cats = ["CR", "PR", "SD", "PD", "NEW"]

    contingency = {}
    for gt in cats:
        contingency[gt] = {}
        for pr in cats:
            contingency[gt][pr] = int(
                ((pairs["gt_category"] == gt) & (pairs["pred_category"] == pr)).sum()
            ) if n else 0

    return {
        "input_audit": audit,
        "excluded_case_units": excluded_cases,
        "n_studies_with_reference_mask": int(len(studies)),
        "n_case_units_with_reference_mask":
            int(studies["case_unit_id"].nunique()) if len(studies) else 0,
        "n_reference_zero_core_studies":
            int((studies["gt_volume_mL"] == 0).sum()) if len(studies) else 0,
        "n_pairs": int(n),
        "n_pairs_category_agree": agree,
        "category_concordance": float(agree / n) if n else None,
        "delta_abs_error_mL": {"median": med, "q1": q1, "q3": q3},
        "n_pairs_originally_adjacent_order":
            int(pairs["originally_adjacent_order"].sum()) if n else 0,
        "n_pairs_compressed_over_missing_order":
            int((~pairs["originally_adjacent_order"]).sum()) if n else 0,
        "n_pairs_involving_reference_zero_core":
            int((pairs["gt_a_zero_core"] | pairs["gt_b_zero_core"]).sum()) if n else 0,
        "contingency_gt_rows_pred_cols": contingency,
    }


def print_summary(s):
    a = s["input_audit"]
    print("=== WHOLE-BURDEN LONGITUDINAL VOLUME-CHANGE RESCORE ===")
    print(f"Input manifest rows after explicit case exclusions: {a['input_rows']}")
    print(f"  blank GT path: {a['blank_gt_path']}")
    print(f"  missing GT file: {a['missing_gt_file']}")
    print(f"  missing prediction: {a['missing_prediction']}")
    print(f"  included studies: {a['included']}")
    if s["excluded_case_units"]:
        print("Explicitly excluded case units: " + ", ".join(s["excluded_case_units"]))
    print("")
    print(f"Studies with reference mask: {s['n_studies_with_reference_mask']}")
    print(f"Case units represented: {s['n_case_units_with_reference_mask']}")
    print(f"Annotated zero-core studies: {s['n_reference_zero_core_studies']}")
    print(f"Consecutive available-reference pairs: {s['n_pairs']}")
    print(f"  originally adjacent in TimepointOrder: {s['n_pairs_originally_adjacent_order']}")
    print(f"  compressed across missing reference timepoint(s): {s['n_pairs_compressed_over_missing_order']}")
    print(f"  involving annotated zero-core reference: {s['n_pairs_involving_reference_zero_core']}")

    if s["n_pairs"]:
        print(
            f"Category concordance: {s['n_pairs_category_agree']}/{s['n_pairs']} = "
            f"{100*s['category_concordance']:.1f}%"
        )
        e = s["delta_abs_error_mL"]
        print(
            f"Absolute total-volume-change error: median {e['median']:.3f} mL "
            f"(IQR {e['q1']:.3f}-{e['q3']:.3f})"
        )
        print("")
        print("Contingency rows=GT cols=prediction; CR, PR, SD, PD, NEW")
        cats = ["CR", "PR", "SD", "PD", "NEW"]
        print("          " + " ".join(f"{c:>5s}" for c in cats))
        for gt in cats:
            row = s["contingency_gt_rows_pred_cols"][gt]
            print(f"{gt:>5s}     " + " ".join(f"{row[p]:5d}" for p in cats))
    else:
        print("")
        print("NO PAIRS SCORED. Inspect GT/prediction path counts above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--tracked-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--exclude-case-unit", action="append", default=[],
        help="case unit to exclude; repeat option if needed"
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    tracked_root = str(Path(args.tracked_root or output_dir / "04_tracked"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_csv, dtype=str)
    if "NormalizedSeriesDescription" in df.columns:
        df = df[df["NormalizedSeriesDescription"] == "T1Post"].copy()

    excluded = [str(x) for x in args.exclude_case_unit]
    if excluded:
        df = df[~df["AnonPatientID"].astype(str).isin(excluded)].copy()

    studies, warnings, audit = collect_per_study(df, tracked_root)
    pairs = compute_longitudinal(studies)
    summary = summarize(studies, pairs, audit, excluded)

    studies.to_csv(out_dir / "per_study_total_volume.csv", index=False)
    pairs.to_csv(out_dir / "longitudinal_change.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(summary)
    (out_dir / "summary.txt").write_text(buf.getvalue())

    print_summary(summary)

    if warnings:
        print("")
        print(f"WARNINGS: {len(warnings)}")
        for w in warnings[:20]:
            print("  " + w)
        if len(warnings) > 20:
            print(f"  ... {len(warnings)-20} more")
        (out_dir / "warnings.txt").write_text("\n".join(warnings) + "\n")
    else:
        print("\nWarnings: 0")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
