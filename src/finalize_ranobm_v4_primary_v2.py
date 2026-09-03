#!/usr/bin/env python3
"""
finalize_ranobm_v4_primary_v2.py

Corrected post-processing for v4 reference-anchored RANO-BM output.

Key distinctions:
  * PRIMARY INDEX-TARGET cohort:
      reference lesions present at the first evaluable reference timepoint
      (ref_new == False), status == matched.
  * REFERENCE-NEW lesions:
      reference lesions first appearing after the first evaluable timepoint
      (ref_new == True), excluding model-only rows.
  * MODEL-ONLY new lesions:
      status == pred_only_new.

Also prints the full index-target eligibility flow so the manuscript denominator
can be reported transparently.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["improved", "stable", "progressed"]
REF_STATUSES = {
    "matched",
    "insufficient_reference_followup",
    "undetected",
    "entry_detection_failure",
}


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def id_is_present(s):
    x = s.fillna("").astype(str).str.strip().str.lower()
    return ~x.isin(["", "na", "nan", "none"])


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p * 100, (c - h) * 100, (c + h) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    v4 = Path(args.v4_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(v4 / "per_lesion_ranobm.csv")
    df["_ref_new"] = as_bool(df["ref_new"])
    df["_has_pred_id"] = id_is_present(df["pred_id"])

    ref_rows = df[df["status"].isin(REF_STATUSES)].copy()
    index_ref = ref_rows[~ref_rows["_ref_new"]].copy()
    reference_new = ref_rows[ref_rows["_ref_new"]].copy()
    model_only_new = df[df["status"] == "pred_only_new"].copy()

    primary = index_ref[index_ref["status"] == "matched"].copy()

    cont = np.zeros((3, 3), dtype=int)
    for _, r in primary.iterrows():
        if r["ref_cat"] in CATS and r["pred_cat"] in CATS:
            cont[CATS.index(r["ref_cat"]), CATS.index(r["pred_cat"])] += 1

    total = int(cont.sum())
    diag = int(np.trace(cont))
    p, lo, hi = wilson(diag, total)

    lines = []
    lines.append("=== FINAL PRIMARY: reference-anchored index target lesions ===")
    lines.append(
        "Index target = measurable reference lesion present at the first "
        "evaluable reference timepoint."
    )
    lines.append(
        "Reference-new lesions and model-only new lesions are tabulated separately."
    )
    lines.append("")

    lines.append("INDEX-TARGET ELIGIBILITY FLOW")
    lines.append(f"Measurable index reference lesions: {len(index_ref)}")
    for status in [
        "insufficient_reference_followup",
        "undetected",
        "entry_detection_failure",
        "matched",
    ]:
        lines.append(f"  {status}: {(index_ref['status'] == status).sum()}")
    lines.append("")

    lines.append(f"Primary matched index-target trajectories: {total}")
    lines.append(
        "Contingency rows=reference cols=prediction; order = " + str(CATS)
    )
    lines.append(str(cont))
    lines.append("")
    lines.append(
        f"Overall agreement: {diag}/{total} = {p:.1f}% "
        f"(95% CI {lo:.1f}-{hi:.1f})"
    )
    for i, cat in enumerate(CATS):
        n = int(cont[i].sum())
        x = int(cont[i, i])
        if n:
            pp, ll, hh = wilson(x, n)
            lines.append(
                f"  {cat:11s}: {x}/{n} = {pp:.1f}% "
                f"(95% CI {ll:.1f}-{hh:.1f})"
            )
        else:
            lines.append(f"  {cat:11s}: n=0")
    reversals = int(cont[0, 2] + cont[2, 0])
    lines.append(f"Direct improved<->progressed reversals: {reversals}")
    lines.append("")

    lines.append("REFERENCE-NEW LESION AUDIT")
    lines.append(f"Reference-new measurable lesions: {len(reference_new)}")
    for status, n in reference_new["status"].value_counts(dropna=False).items():
        lines.append(f"  {status}: {n}")
    lines.append(
        "  with any union-Jaccard matched prediction track: "
        f"{int(reference_new['_has_pred_id'].sum())}/{len(reference_new)}"
    )
    lines.append(
        "  NOTE: this is track-level matching, not yet same-timepoint new-lesion "
        "detection concordance."
    )
    lines.append("")

    lines.append("MODEL-ONLY NEW LESION AUDIT")
    lines.append(
        f"Model-only new measurable lesions (legacy v4 definition): {len(model_only_new)}"
    )

    summary = "\n".join(lines)
    print(summary)
    (out / "ranobm_primary_final_summary.txt").write_text(summary + "\n")

    pd.DataFrame(
        cont,
        index=[f"ref_{c}" for c in CATS],
        columns=[f"pred_{c}" for c in CATS],
    ).to_csv(out / "contingency_ranobm_primary.csv")

    dropcols = ["_ref_new", "_has_pred_id"]
    primary.drop(columns=dropcols).to_csv(
        out / "per_lesion_ranobm_primary.csv", index=False
    )
    index_ref.drop(columns=dropcols).to_csv(
        out / "index_reference_lesions_all.csv", index=False
    )
    reference_new.drop(columns=dropcols).to_csv(
        out / "reference_new_lesions_audit.csv", index=False
    )
    model_only_new.drop(columns=dropcols).to_csv(
        out / "model_only_new_lesions_audit.csv", index=False
    )

    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
