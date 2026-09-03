#!/usr/bin/env python3
"""
Audit denominators directly from the Cyprus PROTEAS patient ZIP archives.

Read-only. The ZIPs are never modified and do not need to be fully extracted.
For every case/timepoint, the script reports:
  - presence of BraTS-space post-contrast T1 (t1c.nii.gz)
  - presence of a tumor segmentation mask
  - voxel counts for Cyprus labels:
      1 = necrotic core
      2 = enhancing tumor
      3 = edema
  - tumor-core positivity (label 1 + label 2 > 0)
  - enhancing positivity (label 2 > 0)
  - necrotic-only/non-enhancing tumor core (label 1 > 0, label 2 = 0)
  - edema-only masks (label 1 = 0, label 2 = 0, label 3 > 0)
  - empty tumor-label masks

It also summarizes study, case-unit, and consecutive-pair denominators and
compares them with the final manuscript targets: 186 T1-post timepoints, 170
available reference masks, 169 tumor-core-positive studies, 44 longitudinal
case units, 125 consecutive available-reference pairs, and a 116-pair strict
original-adjacency sensitivity subset.

Example:
  python audit_cyprus_denominators_zips.py \
    --zip-root /path/to/cyprus_patient_zips \
    --out-dir /path/to/denominator_audit

Dependencies: numpy plus either SimpleITK or nibabel.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

TIMEPOINT_RE = re.compile(r"(?:^|/)BraTS/(baseline|fu\d+)/", re.I)
MASK_RE = re.compile(r"_tumor_mask_(baseline|fu\d+)\.nii(?:\.gz)?$", re.I)
T1C_RE = re.compile(r"(?:^|/)BraTS/(baseline|fu\d+)/t1c\.nii(?:\.gz)?$", re.I)


def tp_key(name: str) -> tuple[int, int]:
    s = name.lower()
    if s == "baseline":
        return (0, 0)
    m = re.fullmatch(r"fu(\d+)", s)
    return (1, int(m.group(1))) if m else (9, 999999)


def make_loader():
    try:
        import SimpleITK as sitk  # type: ignore
        import numpy as np

        def load_mask(path: Path):
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])
            return np.asarray(arr), voxel_mm3, "SimpleITK"

        return load_mask
    except Exception:
        pass

    try:
        import nibabel as nib  # type: ignore
        import numpy as np

        def load_mask(path: Path):
            img = nib.load(str(path))
            arr = np.asanyarray(img.dataobj)
            voxel_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
            return arr, voxel_mm3, "nibabel"

        return load_mask
    except Exception as e:
        raise RuntimeError(
            "Neither SimpleITK nor nibabel is available. Activate the imaging "
            "environment used for this project, or install nibabel."
        ) from e


def mask_label_counts(arr) -> dict[str, int]:
    import numpy as np

    vals, counts = np.unique(arr.astype(int), return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, counts)}
    return {
        "label0_vox": d.get(0, 0),
        "label1_necrotic_vox": d.get(1, 0),
        "label2_enhancing_vox": d.get(2, 0),
        "label3_edema_vox": d.get(3, 0),
        "other_label_vox": sum(c for v, c in d.items() if v not in (0, 1, 2, 3, 10, 30, 40, 50)),
    }


def load_mask_from_zip(zf: zipfile.ZipFile, member: str, load_mask):
    # Write only this one compressed NIfTI member to node-local temporary storage.
    # This avoids extracting the complete patient archive.
    suffix = ".nii.gz" if member.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        with zf.open(member, "r") as src:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.flush()
        return load_mask(Path(tmp.name))


def scan(zip_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_mask = make_loader()
    zips = sorted(zip_root.glob("P*.zip"), key=lambda p: p.name.lower())
    if not zips:
        raise SystemExit(f"No P*.zip archives found under {zip_root}")

    rows: list[dict[str, Any]] = []
    archive_errors: list[str] = []
    per_case_archive_members: dict[str, int] = {}

    for zpath in zips:
        case_id = zpath.stem
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                per_case_archive_members[case_id] = len(names)

                t1c_by_tp: defaultdict[str, list[str]] = defaultdict(list)
                masks_by_tp: defaultdict[str, list[str]] = defaultdict(list)
                all_tp_names: set[str] = set()

                for n in names:
                    m = T1C_RE.search(n)
                    if m:
                        tp = m.group(1).lower()
                        t1c_by_tp[tp].append(n)
                        all_tp_names.add(tp)
                    m = MASK_RE.search(Path(n).name)
                    if m and "tumor" in n.lower() and "segment" in n.lower():
                        tp = m.group(1).lower()
                        masks_by_tp[tp].append(n)
                        all_tp_names.add(tp)
                    m = TIMEPOINT_RE.search(n)
                    if m:
                        all_tp_names.add(m.group(1).lower())

                for tp in sorted(all_tp_names, key=tp_key):
                    t1cs = t1c_by_tp.get(tp, [])
                    masks = masks_by_tp.get(tp, [])
                    row: dict[str, Any] = {
                        "case_id": case_id,
                        "zip_path": str(zpath),
                        "timepoint": tp,
                        "timepoint_order": 0 if tp == "baseline" else tp_key(tp)[1],
                        "t1c_present": int(len(t1cs) > 0),
                        "t1c_member": t1cs[0] if t1cs else "",
                        "t1c_duplicates_for_key": len(t1cs),
                        "mask_present": int(len(masks) > 0),
                        "mask_member": masks[0] if masks else "",
                        "mask_duplicates_for_key": len(masks),
                        "paired_t1c_mask": int(bool(t1cs and masks)),
                        "label1_necrotic_vox": "",
                        "label2_enhancing_vox": "",
                        "label3_edema_vox": "",
                        "tumor_core_vox": "",
                        "tumor_core_mL": "",
                        "enhancing_mL": "",
                        "tumor_core_positive": 0,
                        "enhancing_positive": 0,
                        "necrotic_only_nonenhancing_core": 0,
                        "edema_only": 0,
                        "empty_tumor_labels": 0,
                        "loader": "",
                        "error": "",
                    }

                    if masks:
                        try:
                            arr, voxel_mm3, loader_name = load_mask_from_zip(zf, masks[0], load_mask)
                            cnt = mask_label_counts(arr)
                            l1 = cnt["label1_necrotic_vox"]
                            l2 = cnt["label2_enhancing_vox"]
                            l3 = cnt["label3_edema_vox"]
                            core = l1 + l2
                            row.update({
                                "label1_necrotic_vox": l1,
                                "label2_enhancing_vox": l2,
                                "label3_edema_vox": l3,
                                "tumor_core_vox": core,
                                "tumor_core_mL": round(core * voxel_mm3 / 1000.0, 6),
                                "enhancing_mL": round(l2 * voxel_mm3 / 1000.0, 6),
                                "tumor_core_positive": int(core > 0),
                                "enhancing_positive": int(l2 > 0),
                                "necrotic_only_nonenhancing_core": int(l1 > 0 and l2 == 0),
                                "edema_only": int(l1 == 0 and l2 == 0 and l3 > 0),
                                "empty_tumor_labels": int(l1 == 0 and l2 == 0 and l3 == 0),
                                "loader": loader_name,
                            })
                        except Exception as e:
                            row["error"] = f"{type(e).__name__}: {e}"

                    rows.append(row)
        except Exception as e:
            archive_errors.append(f"{zpath.name}: {type(e).__name__}: {e}")

    by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    for cr in by_case.values():
        cr.sort(key=lambda r: tp_key(r["timepoint"]))

    def eligible(r, kind: str) -> bool:
        if not r["paired_t1c_mask"]:
            return False
        if kind == "reference":
            return True
        if kind == "core":
            return bool(r["tumor_core_positive"])
        if kind == "enh":
            return bool(r["enhancing_positive"])
        raise ValueError(kind)

    def pair_stats(kind: str):
        case_units = 0
        original_adjacent = 0
        compressed = 0
        per_case = {}
        for cid, cr in sorted(by_case.items()):
            flags = [eligible(r, kind) for r in cr]
            n = sum(flags)
            if n >= 2:
                case_units += 1
            original_adjacent += sum(a and b for a, b in zip(flags, flags[1:]))
            compressed += max(n - 1, 0)
            per_case[cid] = n
        return case_units, original_adjacent, compressed, per_case

    ref_case_units, ref_pairs_adj, ref_pairs_comp, ref_per_case = pair_stats("reference")
    core_case_units, core_pairs_adj, core_pairs_comp, core_per_case = pair_stats("core")
    enh_case_units, enh_pairs_adj, enh_pairs_comp, enh_per_case = pair_stats("enh")

    summary: dict[str, Any] = {
        "zip_root": str(zip_root),
        "zip_archives_discovered": len(zips),
        "case_units_discovered": len(by_case),
        "timepoint_keys_union": len(rows),
        "t1c_timepoints": sum(int(r["t1c_present"]) for r in rows),
        "mask_timepoints": sum(int(r["mask_present"]) for r in rows),
        "paired_t1c_mask_timepoints": sum(int(r["paired_t1c_mask"]) for r in rows),
        "tumor_core_positive_timepoints": sum(int(r["paired_t1c_mask"] and r["tumor_core_positive"]) for r in rows),
        "enhancing_positive_timepoints": sum(int(r["paired_t1c_mask"] and r["enhancing_positive"]) for r in rows),
        "necrotic_only_nonenhancing_core_timepoints": sum(int(r["necrotic_only_nonenhancing_core"]) for r in rows),
        "edema_only_timepoints": sum(int(r["edema_only"]) for r in rows),
        "empty_tumor_label_timepoints": sum(int(r["empty_tumor_labels"]) for r in rows),
        "missing_t1c_with_mask": sum(int(r["mask_present"] and not r["t1c_present"]) for r in rows),
        "missing_mask_with_t1c": sum(int(r["t1c_present"] and not r["mask_present"]) for r in rows),
        "duplicate_t1c_case_timepoint_keys": sum(int(r["t1c_duplicates_for_key"] > 1) for r in rows),
        "duplicate_mask_case_timepoint_keys": sum(int(r["mask_duplicates_for_key"] > 1) for r in rows),
        "case_units_with_2plus_reference_masks": ref_case_units,
        "reference_pairs_original_adjacent": ref_pairs_adj,
        "reference_pairs_compressed_available_sequence": ref_pairs_comp,
        "case_units_with_2plus_core_positive": core_case_units,
        "core_pairs_original_adjacent": core_pairs_adj,
        "core_pairs_compressed_eligible_sequence": core_pairs_comp,
        "case_units_with_2plus_enhancing_positive": enh_case_units,
        "enhancing_pairs_original_adjacent": enh_pairs_adj,
        "enhancing_pairs_compressed_eligible_sequence": enh_pairs_comp,
        "archive_errors": archive_errors,
        "per_case_reference_masks": ref_per_case,
        "per_case_core_positive": core_per_case,
        "per_case_enhancing_positive": enh_per_case,
        "per_case_archive_members": per_case_archive_members,
        "manuscript_targets": {
            "t1c_timepoints": 186,
            "reference_masks": 170,
            "tumor_core_positive_studies": 169,
            "longitudinal_case_units": 44,
            "consecutive_available_reference_pairs": 125,
            "strict_original_adjacent_pairs": 116,
        },
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cyprus_timepoint_audit.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "cyprus_denominator_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    txt_path = out_dir / "cyprus_denominator_summary.txt"
    keys = [
        "zip_archives_discovered",
        "case_units_discovered",
        "timepoint_keys_union",
        "t1c_timepoints",
        "mask_timepoints",
        "paired_t1c_mask_timepoints",
        "tumor_core_positive_timepoints",
        "enhancing_positive_timepoints",
        "necrotic_only_nonenhancing_core_timepoints",
        "edema_only_timepoints",
        "empty_tumor_label_timepoints",
        "missing_t1c_with_mask",
        "missing_mask_with_t1c",
        "duplicate_t1c_case_timepoint_keys",
        "duplicate_mask_case_timepoint_keys",
        "case_units_with_2plus_reference_masks",
        "reference_pairs_original_adjacent",
        "reference_pairs_compressed_available_sequence",
        "case_units_with_2plus_core_positive",
        "core_pairs_original_adjacent",
        "core_pairs_compressed_eligible_sequence",
        "case_units_with_2plus_enhancing_positive",
        "enhancing_pairs_original_adjacent",
        "enhancing_pairs_compressed_eligible_sequence",
    ]

    lines = ["Cyprus ZIP denominator audit", "=" * 30]
    for k in keys:
        lines.append(f"{k}: {summary[k]}")

    lines += ["", "Manuscript targets:"]
    for k, v in summary["manuscript_targets"].items():
        lines.append(f"  {k}: {v}")

    lines += ["", "Fast reconciliation checks:"]
    lines.append(f"  T1-post timepoints == 186: {summary['t1c_timepoints'] == 186}")
    lines.append(f"  available reference masks == 170: {summary['paired_t1c_mask_timepoints'] == 170}")
    lines.append(f"  tumor-core-positive studies == 169: {summary['tumor_core_positive_timepoints'] == 169}")
    lines.append(f"  longitudinal case units == 44: {summary['case_units_with_2plus_reference_masks'] == 44}")
    lines.append(f"  consecutive available-reference pairs == 125: {summary['reference_pairs_compressed_available_sequence'] == 125}")
    lines.append(f"  strict original-adjacent pairs == 116: {summary['reference_pairs_original_adjacent'] == 116}")

    nec = [
        f"{r['case_id']}:{r['timepoint']} (L1={r['label1_necrotic_vox']}, L2={r['label2_enhancing_vox']}, L3={r['label3_edema_vox']})"
        for r in rows if r["necrotic_only_nonenhancing_core"]
    ]
    edema = [
        f"{r['case_id']}:{r['timepoint']} (L3={r['label3_edema_vox']})"
        for r in rows if r["edema_only"]
    ]
    empty = [f"{r['case_id']}:{r['timepoint']}" for r in rows if r["empty_tumor_labels"]]
    missing_mask = [f"{r['case_id']}:{r['timepoint']}" for r in rows if r["t1c_present"] and not r["mask_present"]]
    missing_t1c = [f"{r['case_id']}:{r['timepoint']}" for r in rows if r["mask_present"] and not r["t1c_present"]]

    lines += ["", "Necrotic-only / non-enhancing tumor-core timepoints (L1 > 0, L2 = 0):"]
    lines.append("  " + (", ".join(nec) if nec else "none"))
    lines += ["", "Edema-only timepoints (L1 = L2 = 0, L3 > 0):"]
    lines.append("  " + (", ".join(edema) if edema else "none"))
    lines += ["", "Empty tumor-label masks (L1 = L2 = L3 = 0):"]
    lines.append("  " + (", ".join(empty) if empty else "none"))
    lines += ["", "T1c present but mask missing:"]
    lines.append("  " + (", ".join(missing_mask) if missing_mask else "none"))
    lines += ["", "Mask present but T1c missing:"]
    lines.append("  " + (", ".join(missing_t1c) if missing_t1c else "none"))

    if summary["archive_errors"]:
        lines += ["", "Archive errors:"]
        lines.extend(f"  {e}" for e in summary["archive_errors"])

    txt_path.write_text("\n".join(lines) + "\n")
    return csv_path, json_path, txt_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-root", required=True, type=Path, help="Directory containing P*.zip patient archives")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for audit CSV/JSON/TXT")
    args = ap.parse_args()

    zip_root = args.zip_root.resolve()
    out_dir = args.out_dir.resolve()
    rows, summary = scan(zip_root)
    paths = write_outputs(rows, summary, out_dir)
    print(Path(paths[2]).read_text())
    print("Wrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
