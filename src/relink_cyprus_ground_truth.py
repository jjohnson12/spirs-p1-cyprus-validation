#!/usr/bin/env python3
"""
Relink Cyprus PROTEAS ground-truth tumor masks from the original patient ZIPs
into a stable extracted directory, repair case-sensitive timepoint matching
(e.g. P31 Baseline/Fu1/Fu2/Fu3), and write a corrected dataset manifest.

This script is deliberately conservative:
  * NEVER overwrites the input dataset.csv.
  * NEVER modifies the source ZIP archives.
  * Extracts only tumor segmentation masks needed by the manifest.
  * Preserves the original Timepoint in TimepointOriginal and writes a
    normalized lowercase Timepoint (baseline, fu1, fu2, ...).
  * Fills ZipStem from the processed-path case-unit directory.
  * Adds explicit GT linkage / label-content / eligibility columns.

Cyprus label convention (Scientific Data descriptor):
  1 = necrotic core
  2 = enhancing tumor
  3 = edema
Tumor core for this project = labels 1 + 2.

Expected reconciliation for the current v3-style project inventory, based on
prior audit:
  186 manifest rows / T1-post studies
  170 source tumor masks paired to those studies
  169 tumor-core-positive reference studies
    1 edema-only reference study (P31:fu2)
   16 T1-post studies with no source tumor mask

Example:
  python relink_cyprus_ground_truth.py \
    --dataset /path/to/cyprus_validation/dataset.csv \
    --zip-root /path/to/cyprus_patient_zips \
    --output-root /path/to/cyprus_validation/03_ground_truth_relinked \
    --out-dataset /path/to/cyprus_validation/dataset_relinked.csv \
    --summary /path/to/cyprus_validation/relink_summary.txt

Dependencies: pandas, numpy, and either SimpleITK or nibabel.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MASK_BASENAME_RE = re.compile(r"^(?P<case>P\d{2}[ab]?)_tumor_mask_(?P<tp>baseline|fu\d+)\.nii(?:\.gz)?$", re.I)
PROCESSED_CASE_RE = re.compile(r"/02_preprocessed/([^/]+)/")


def normalize_timepoint(value: Any, order: Any = None) -> str:
    if pd.notna(value):
        s = str(value).strip().lower()
        if s == "baseline" or re.fullmatch(r"fu\d+", s):
            return s
    if pd.notna(order):
        n = int(order)
        return "baseline" if n == 0 else f"fu{n}"
    raise ValueError(f"Cannot normalize timepoint value={value!r}, order={order!r}")


def infer_case_id(row: pd.Series) -> str:
    # Best source: case unit embedded in the stable preprocessing path. This
    # preserves split units such as P20a/P20b whereas FlouriPatientId does not.
    p = str(row.get("Preprocessed", ""))
    m = PROCESSED_CASE_RE.search(p)
    if m:
        return m.group(1)
    for col in ("AnonPatientID", "ZipStem"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    raise ValueError(f"Could not infer case unit for row {row.name}")


def make_loader():
    try:
        import SimpleITK as sitk  # type: ignore

        def load(path: Path):
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])
            return np.asarray(arr), voxel_mm3, "SimpleITK"

        return load
    except Exception:
        pass

    try:
        import nibabel as nib  # type: ignore

        def load(path: Path):
            img = nib.load(str(path))
            arr = np.asanyarray(img.dataobj)
            voxel_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
            return arr, voxel_mm3, "nibabel"

        return load
    except Exception as e:
        raise RuntimeError("Need SimpleITK or nibabel in the active environment") from e


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def label_counts(arr: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(arr.astype(np.int64, copy=False), return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, counts)}
    return {
        "GTLabel1NecroticVox": d.get(1, 0),
        "GTLabel2EnhancingVox": d.get(2, 0),
        "GTLabel3EdemaVox": d.get(3, 0),
    }


def index_zip_masks(zip_root: Path) -> tuple[dict[tuple[str, str], tuple[Path, str]], list[str]]:
    index: dict[tuple[str, str], tuple[Path, str]] = {}
    errors: list[str] = []
    zips = sorted(zip_root.glob("P*.zip"), key=lambda p: p.name.lower())
    if not zips:
        raise SystemExit(f"No P*.zip files found in {zip_root}")

    for zp in zips:
        archive_case = zp.stem
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    base = Path(member).name
                    m = MASK_BASENAME_RE.match(base)
                    if not m:
                        continue
                    # Prefer archive stem as the case-unit key; verify basename
                    # case agrees when possible.
                    member_case = m.group("case")
                    tp = m.group("tp").lower()
                    if member_case.lower() != archive_case.lower():
                        errors.append(
                            f"case-name mismatch: archive={archive_case}, member={member}"
                        )
                    key = (archive_case.lower(), tp)
                    if key in index:
                        errors.append(
                            f"duplicate mask key {archive_case}:{tp}: {index[key][1]} AND {member}"
                        )
                    else:
                        index[key] = (zp, member)
        except Exception as e:
            errors.append(f"{zp.name}: {type(e).__name__}: {e}")
    return index, errors


def extract_member(zp: Path, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically to avoid partially extracted masks if interrupted.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(zp, "r") as zf, zf.open(member, "r") as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--zip-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Directory where only reference tumor masks will be extracted")
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="Allow replacing prior output files/masks")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Input dataset not found: {args.dataset}")
    if not args.zip_root.exists():
        raise SystemExit(f"ZIP root not found: {args.zip_root}")
    if args.out_dataset.resolve() == args.dataset.resolve():
        raise SystemExit("Refusing to overwrite the input dataset. Use a new --out-dataset path.")
    if args.out_dataset.exists() and not args.force:
        raise SystemExit(f"Output dataset already exists: {args.out_dataset} (use --force to replace)")

    load_mask = make_loader()
    mask_index, index_errors = index_zip_masks(args.zip_root)

    df = pd.read_csv(args.dataset)
    if len(df) == 0:
        raise SystemExit("Input dataset is empty")

    # Preserve original values before normalizing.
    if "TimepointOriginal" not in df.columns:
        df["TimepointOriginal"] = df["Timepoint"]
    if "GTMaskPathOriginal" not in df.columns:
        df["GTMaskPathOriginal"] = df["GTMaskPath"]

    # Initialize/replace derived columns deterministically.
    new_cols = {
        "CaseUnit": "",
        "TimepointNormalized": "",
        "GTZipPath": "",
        "GTZipMember": "",
        "GTLinkStatus": "",
        "GTMaskPresent": False,
        "GTMaskSHA256": "",
        "GTLoader": "",
        "GTVoxelVolumeMM3": np.nan,
        "GTLabel1NecroticVox": 0,
        "GTLabel2EnhancingVox": 0,
        "GTLabel3EdemaVox": 0,
        "GTTumorCoreVox": 0,
        "GTTumorCoreML": 0.0,
        "GTEnhancingML": 0.0,
        "GTTumorCorePositive": False,
        "GTEnhancingPositive": False,
        "GTEdemaOnly": False,
        "GTEmptyTumorLabels": False,
        "EligiblePerStudyTumorCore": False,
    }
    for c, default in new_cols.items():
        df[c] = default

    # Normalize Timepoint itself for downstream case-sensitive consumers, but
    # retain TimepointOriginal for provenance.
    statuses = Counter()
    case_counts = defaultdict(lambda: Counter())
    extraction_errors: list[str] = []

    for i, row in df.iterrows():
        try:
            case = infer_case_id(row)
            tp = normalize_timepoint(row.get("Timepoint"), row.get("TimepointOrder"))
        except Exception as e:
            df.at[i, "GTLinkStatus"] = f"manifest_key_error:{type(e).__name__}"
            extraction_errors.append(f"row {i}: {e}")
            statuses[df.at[i, "GTLinkStatus"]] += 1
            continue

        df.at[i, "CaseUnit"] = case
        df.at[i, "ZipStem"] = case
        df.at[i, "TimepointNormalized"] = tp
        df.at[i, "Timepoint"] = tp

        key = (case.lower(), tp)
        if key not in mask_index:
            df.at[i, "GTMaskPath"] = ""
            df.at[i, "GTLinkStatus"] = "source_mask_missing"
            statuses["source_mask_missing"] += 1
            case_counts[case]["source_mask_missing"] += 1
            continue

        zp, member = mask_index[key]
        dest = args.output_root / case / f"{case}_tumor_mask_{tp}.nii.gz"
        try:
            if not dest.exists() or args.force:
                extract_member(zp, member, dest)
            arr, vox_mm3, loader_name = load_mask(dest)
            cnt = label_counts(arr)
            l1 = cnt["GTLabel1NecroticVox"]
            l2 = cnt["GTLabel2EnhancingVox"]
            l3 = cnt["GTLabel3EdemaVox"]
            core = l1 + l2

            df.at[i, "GTMaskPath"] = str(dest)
            df.at[i, "GTZipPath"] = str(zp)
            df.at[i, "GTZipMember"] = member
            df.at[i, "GTMaskPresent"] = True
            df.at[i, "GTMaskSHA256"] = sha256_file(dest)
            df.at[i, "GTLoader"] = loader_name
            df.at[i, "GTVoxelVolumeMM3"] = vox_mm3
            df.at[i, "GTLabel1NecroticVox"] = l1
            df.at[i, "GTLabel2EnhancingVox"] = l2
            df.at[i, "GTLabel3EdemaVox"] = l3
            df.at[i, "GTTumorCoreVox"] = core
            df.at[i, "GTTumorCoreML"] = core * vox_mm3 / 1000.0
            df.at[i, "GTEnhancingML"] = l2 * vox_mm3 / 1000.0
            df.at[i, "GTTumorCorePositive"] = bool(core > 0)
            df.at[i, "GTEnhancingPositive"] = bool(l2 > 0)
            df.at[i, "GTEdemaOnly"] = bool(l1 == 0 and l2 == 0 and l3 > 0)
            df.at[i, "GTEmptyTumorLabels"] = bool(l1 == 0 and l2 == 0 and l3 == 0)
            df.at[i, "EligiblePerStudyTumorCore"] = bool(core > 0)

            if core > 0:
                status = "linked_tumor_core_positive"
            elif l3 > 0:
                status = "linked_edema_only_reference_core_empty"
            else:
                status = "linked_empty_tumor_reference"
            df.at[i, "GTLinkStatus"] = status
            statuses[status] += 1
            case_counts[case][status] += 1
        except Exception as e:
            df.at[i, "GTMaskPath"] = ""
            status = f"extract_or_read_error:{type(e).__name__}"
            df.at[i, "GTLinkStatus"] = status
            statuses[status] += 1
            case_counts[case][status] += 1
            extraction_errors.append(f"{case}:{tp}: {e}")

    # Explicit longitudinal counts under two transparent definitions.
    # A) tumor-core-positive sequence (appropriate for DSC/detection analyses)
    # B) any reference-mask sequence, including an edema-only/zero-core timepoint
    def longitudinal_counts(eligible_col: str) -> tuple[int, int, int, dict[str, int]]:
        represented = 0
        with_pair = 0
        compressed_pairs = 0
        per_case: dict[str, int] = {}
        for case, sub in df.groupby("CaseUnit", dropna=False):
            if not case:
                continue
            n = int(sub[eligible_col].fillna(False).astype(bool).sum())
            per_case[str(case)] = n
            if n >= 1:
                represented += 1
            if n >= 2:
                with_pair += 1
                compressed_pairs += n - 1
        return represented, with_pair, compressed_pairs, per_case

    df["EligibleAnyReferenceMask"] = df["GTMaskPresent"].astype(bool)
    tc_repr, tc_pair_cases, tc_pairs, tc_per_case = longitudinal_counts("EligiblePerStudyTumorCore")
    ref_repr, ref_pair_cases, ref_pairs, ref_per_case = longitudinal_counts("EligibleAnyReferenceMask")

    # Case-sensitive linkage diagnostic: rows whose original Timepoint differs
    # from normalized spelling and whose source mask exists.
    normalized_changed = (df["TimepointOriginal"].astype(str) != df["TimepointNormalized"].astype(str))
    repaired_case_rows = df.loc[normalized_changed & df["GTMaskPresent"], [
        "CaseUnit", "AnonStudyID", "TimepointOriginal", "TimepointNormalized", "GTLinkStatus", "GTMaskPath"
    ]]

    args.out_dataset.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dataset, index=False)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += ["Cyprus GT relink summary", "=" * 25]
    lines.append(f"input_dataset_rows: {len(df)}")
    lines.append(f"zip_archives_indexed: {len(list(args.zip_root.glob('P*.zip')))}")
    lines.append(f"unique_source_mask_keys_indexed: {len(mask_index)}")
    lines.append(f"GTMaskPresent: {int(df['GTMaskPresent'].sum())}")
    lines.append(f"GTTumorCorePositive: {int(df['GTTumorCorePositive'].sum())}")
    lines.append(f"GTEnhancingPositive: {int(df['GTEnhancingPositive'].sum())}")
    lines.append(f"GTEdemaOnly: {int(df['GTEdemaOnly'].sum())}")
    lines.append(f"source_mask_missing: {int((df['GTLinkStatus'] == 'source_mask_missing').sum())}")
    lines.append("")
    lines.append("GTLinkStatus counts:")
    for k, v in sorted(statuses.items()):
        lines.append(f"  {k}: {v}")

    lines += ["", "Longitudinal denominators:"]
    lines.append(f"  tumor-core-positive represented case units: {tc_repr}")
    lines.append(f"  tumor-core-positive case units with >=2 studies: {tc_pair_cases}")
    lines.append(f"  tumor-core-positive compressed pairs: {tc_pairs}")
    lines.append(f"  any-reference-mask represented case units: {ref_repr}")
    lines.append(f"  any-reference-mask case units with >=2 studies: {ref_pair_cases}")
    lines.append(f"  any-reference-mask compressed pairs: {ref_pairs}")

    lines += ["", "Rows whose Timepoint spelling was normalized and source mask was successfully linked:"]
    if len(repaired_case_rows):
        for _, r in repaired_case_rows.iterrows():
            lines.append(
                f"  {r['CaseUnit']}:{r['TimepointOriginal']} -> {r['TimepointNormalized']} "
                f"({r['GTLinkStatus']})"
            )
    else:
        lines.append("  none")

    edema_rows = df[df["GTEdemaOnly"]]
    lines += ["", "Edema-only / zero tumor-core reference rows:"]
    if len(edema_rows):
        for _, r in edema_rows.iterrows():
            lines.append(
                f"  {r['CaseUnit']}:{r['TimepointNormalized']} "
                f"L1={int(r['GTLabel1NecroticVox'])} L2={int(r['GTLabel2EnhancingVox'])} "
                f"L3={int(r['GTLabel3EdemaVox'])}"
            )
    else:
        lines.append("  none")

    missing_rows = df[df["GTLinkStatus"] == "source_mask_missing"]
    lines += ["", "T1-post manifest rows with no source tumor mask:"]
    if len(missing_rows):
        for _, r in missing_rows.iterrows():
            lines.append(f"  {r['CaseUnit']}:{r['TimepointNormalized']} (AnonStudyID={r['AnonStudyID']})")
    else:
        lines.append("  none")

    if index_errors:
        lines += ["", "ZIP indexing warnings:"] + [f"  {x}" for x in index_errors]
    if extraction_errors:
        lines += ["", "Extraction/read errors:"] + [f"  {x}" for x in extraction_errors]

    lines += ["", "Expected key checks:"]
    lines.append(f"  186 manifest rows: {len(df) == 186}")
    lines.append(f"  170 reference masks linked: {int(df['GTMaskPresent'].sum()) == 170}")
    lines.append(f"  169 tumor-core-positive studies: {int(df['GTTumorCorePositive'].sum()) == 169}")
    lines.append(f"  1 edema-only reference row: {int(df['GTEdemaOnly'].sum()) == 1}")
    lines.append(f"  16 source masks missing: {int((df['GTLinkStatus'] == 'source_mask_missing').sum()) == 16}")

    args.summary.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote corrected manifest: {args.out_dataset}")
    print(f"Extracted/relinked GT masks: {args.output_root}")
    print(f"Summary: {args.summary}")

    # Nonzero exit if structural expectations fail, so batch jobs can catch it.
    if index_errors or extraction_errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
