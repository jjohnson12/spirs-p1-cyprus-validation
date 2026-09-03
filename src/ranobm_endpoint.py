#!/usr/bin/env python3
"""
ranobm_endpoint.py  (v2 - dataset.csv driven; historical/shared utilities)
===========================================================================
Historical modified RANO-BM trajectory implementation retained for
traceability and for utilities imported by the manuscript-locked v4 analysis.
The primary endpoint is produced by ranobm_endpoint_v4_reference_anchored.py.

Wired to the existing Flouri validation pipeline:
  * reads <output_dir>/dataset.csv (from run_flouri_validation_pipeline.py)
      AnonPatientID  = case unit (P01, P04a, P20b, ...)
      AnonStudyID    = synthetic study id / date folder (e.g. 20200331)
      TimepointOrder = 0 baseline, 1 fu1, ...
      GTMaskPath     = Flouri expert mask for that study (labels 1/2/3)
  * predicted lesions from the TRACKED ids map:
      <TRACKED_ROOT>/<case_unit>/<study_id>/anat/<...ids*label1*>.nii.gz
      (integer voxel value = tracked lesion id, consistent across timepoints)
  * reference tumor core = GT labels 1 (necrotic) + 2 (enhancing); edema (3) excluded

Emits in one run:
  - 3x3 reference-vs-predicted RANO-BM contingency (historical implementation)
  - overall + per-category agreement with Wilson 95% CIs
  - new-measurable-lesion tally (and PD calls it drives)
  - per-volume-tier reference lesion denominators  (-> sensitivity CIs in Table 1)
  - false-positive distribution (mean, median, IQR, % studies with >=1 FP)

ASSUMPTIONS to sanity-check:
  * BraTS space, 1 mm iso (verified on a sample); axial = axis 2.
  * Predicted ids consistent across timepoints (Stage G track-tumors). If a
    sample shows per-timepoint re-assignment, set TRACK_PRED="overlap".
  * GT is not instance-tracked: reference lesions are connected components on
    the core, tracked here by greedy overlap.
  * GT_CORE_LABELS must match what compute_external_validation_metrics.py used
    for the DSC/sensitivity numbers already in the manuscript - confirm.
  * Synthetic 90-day study dates carry NO real interval info; RANO categories
    depend only on magnitude vs baseline/nadir, so this is fine, but do NOT
    derive any rate / time-to-progression quantity from these folders.

Run:
    conda activate qtim_preprocessing
    python ranobm_endpoint.py --output-dir /path/to/cyprus_validation

Deps: numpy, nibabel, scipy, pandas
"""

import os, glob, math, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ============================ CONFIG ============================
PRED_IDS_GLOB = "*ids*label1*.nii.gz"   # tracked lesion-id map in .../<study>/anat/
GT_CORE_LABELS = (1, 2)                  # necrotic + enhancing; edema (3) excluded
AXIAL_AXIS     = 2

MEASURABLE_MM  = 10.0     # RANO-BM measurability floor
PR_FRAC        = 0.30     # improved: >=30% decrease vs baseline (CR = disappearance)
PD_FRAC        = 0.20     # progressed: >=20% increase vs nadir ...
PD_ABS_MM      = 5.0      # ... AND >=5 mm absolute increase (or new measurable lesion)
MATCH_JACCARD  = 0.10
TRACK_OVERLAP_MIN = 0.20   # temporal linkage: min (intersection / smaller-mask area)
                           # to link a lesion to itself across timepoints. Size-
                           # invariant, so growth/shrinkage does not break tracks.
TRACK_CENTROID_MM = 12.0   # fallback: link by 3D centroid distance (voxels) when
                           # overlap fails (component merge/split/shift)

TRACK_PRED = "label"      # "label" (use tracked ids) or "overlap"
FP_NEW_LESION_IS_PD  = True
SCORE_UNDETECTED_REF = False

VOL_TIERS = [(0.0, 0.05), (0.05, 0.5), (0.5, 4.0), (4.0, np.inf)]
CATS = ["improved", "stable", "progressed"]
# ================================================================


def wilson(x, n, z=1.96):
    if n == 0: return (float("nan"),)*3
    p = x/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/den
    return p*100, (c-h)*100, (c+h)*100

def load(path):
    img = nib.load(str(path))
    return np.rint(np.asanyarray(img.dataobj)).astype(np.int32), img.header.get_zooms()[:3]

def _max_pairwise(coords):
    """Longest pairwise distance, hardened against large point sets.
    Always reduce to convex-hull vertices first (few points), so the O(n^2)
    cdist runs on the hull, not the full slice. Falls back safely for tiny or
    degenerate (collinear) point sets."""
    n = coords.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(coords[0] - coords[1]))
    try:
        v = coords[ConvexHull(coords).vertices]
    except Exception:
        # collinear / degenerate: project onto principal axis, take extent
        c = coords - coords.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        t = c @ u
        return float(t.max() - t.min())
    return float(cdist(v, v).max())

def _gated_new_pd_tps(new_lesions, all_tps, min_tps, min_mm):
    """Determine at which timepoints a NEW measurable lesion qualifies to trigger
    PD, under a confirmation/size gate.
      new_lesions : list of {tp: longest_diameter_mm} for each new lesion
      min_tps     : new lesion must be measurable for >=min_tps CONSECUTIVE
                    annotated timepoints to count (1 = strict, no confirmation)
      min_mm      : minimum longest diameter to count (>=MEASURABLE_MM always)
    Returns (pd_tps:set, n_unconfirmable:int). A new lesion appearing only at the
    final timepoint (or only once) cannot be confirmed and does NOT trigger PD
    under min_tps>=2; such events are counted in n_unconfirmable.
    """
    thr = max(MEASURABLE_MM, min_mm)
    idx = {t: i for i, t in enumerate(all_tps)}
    pd_tps, unconfirmable = set(), 0
    for traj in new_lesions:
        mtps = sorted(t for t, d in traj.items() if d >= thr)
        if not mtps:
            continue
        if min_tps <= 1:
            pd_tps.add(mtps[0]); continue
        # need min_tps consecutive annotated timepoints measurable
        run, confirmed_at = 1, None
        for j in range(1, len(mtps)):
            run = run + 1 if idx[mtps[j]] == idx[mtps[j-1]] + 1 else 1
            if run >= min_tps:
                confirmed_at = mtps[j]; break
        if confirmed_at is not None:
            pd_tps.add(confirmed_at)
        else:
            unconfirmable += 1   # appeared too late / too briefly to confirm
    return pd_tps, unconfirmable

def _sum_response_trajectory(sum_by_tp, new_lesions, min_tps=1, min_mm=0.0):
    """Classify a patient-level target-sum trajectory into per-timepoint RANO-BM
    responses, with confirmation/size-gated new-lesion PD calls.
    Returns ({tp: 'CR'|'PR'|'SD'|'PD'}, n_unconfirmable). Once a qualifying new
    lesion or sum-based PD occurs, PD is absorbing for subsequent timepoints."""
    tps = sorted(sum_by_tp)
    if not tps:
        return {}, 0
    base = sum_by_tp[tps[0]]
    nadir = base
    pd_tps, unconf = _gated_new_pd_tps(new_lesions, tps, min_tps, min_mm)
    out, progressed = {}, False
    for t in tps[1:]:
        s = sum_by_tp[t]
        if progressed or t in pd_tps:
            out[t] = "PD"; progressed = True
        elif s == 0:
            out[t] = "CR"
        elif base > 0 and s <= base * (1 - PR_FRAC):
            out[t] = "PR"
        elif s >= nadir * (1 + PD_FRAC) and (s - nadir) >= PD_ABS_MM:
            out[t] = "PD"; progressed = True
        else:
            out[t] = "SD"
        nadir = min(nadir, s)
    return out, unconf

_RESP_RANK = {"CR": 0, "PR": 1, "SD": 2, "PD": 3}  # best (lowest) -> worst
def _best_overall(resp_by_tp):
    """Unconfirmed best overall response = best (lowest-rank) per-timepoint response.
    If no post-baseline timepoint, returns 'NE' (not evaluable)."""
    if not resp_by_tp:
        return "NE"
    return min(resp_by_tp.values(), key=lambda r: _RESP_RANK[r])

PL_CATS = ["CR", "PR", "SD", "PD"]
# Patient-level sensitivity sweep over the new-lesion -> PD rule:
#   new-lesion size floor (mm) x confirmation persistence (consecutive TPs)
# Per-lesion absolute-growth guard (RECIST-style >=PD_ABS_MM AND >=PD_FRAC) is
# applied in classify_track already; the abs-guard toggle here governs whether the
# patient-SUM PD also requires the absolute floor (it does by default).
NEW_MM_SWEEP   = [5.0, 8.0, 10.0, 12.0]   # new-lesion measurable floor
PERSIST_SWEEP  = [1, 2]                    # 1 = strict, 2 = confirmed (>=2 consecutive TPs)
# Primary patient-level result for the headline:
PRIMARY_GATE   = (10.0, 2)                 # 10 mm floor, confirmed (>=2 TP)
def _gate_name(mm, ntp):
    return f"new{int(mm)}mm_{'strict' if ntp==1 else 'confirmed'}"
GATES = {_gate_name(mm, ntp): (ntp, mm) for mm in NEW_MM_SWEEP for ntp in PERSIST_SWEEP}
PRIMARY_GATE_NAME = _gate_name(PRIMARY_GATE[0], PRIMARY_GATE[1])
def patient_level_responses(rinfo, pinfo, assign, n_target=5):
    """Compute GT and model patient-level RANO-BM responses for one case unit.
    Targets selected from GT baseline (n_target largest measurable; None = all).
    The model arm sums the SAME anatomical target lesions via `assign` (GT rid ->
    model pid). New measurable lesions: GT-new from unmatched GT lesions appearing
    after baseline; model-new from unmatched model lesions appearing after baseline.
    Returns (gt_resp_by_tp, model_resp_by_tp) as {tp: category}."""
    # GT measurable lesions with their per-tp diameters
    gt_meas = {rid: rd for rid, (rd, rmeas, _, _) in rinfo.items() if rmeas}
    if not gt_meas:
        return {gate: ({}, {}, 0, 0) for gate in GATES}
    # baseline = earliest tp present across GT lesions
    all_tps = sorted({t for rd in gt_meas.values() for t in rd})
    base_tp = all_tps[0]
    # select targets: largest baseline diameter (present at baseline)
    base_diam = {rid: rd.get(base_tp, 0.0) for rid, rd in gt_meas.items()}
    ranked = sorted(base_diam, key=lambda r: base_diam[r], reverse=True)
    ranked = [r for r in ranked if base_diam[r] >= MEASURABLE_MM]
    targets = ranked if n_target is None else ranked[:n_target]
    if not targets:
        return {gate: ({}, {}, 0, 0) for gate in GATES}

    # GT sum trajectory over target lesions
    gt_sum = {}
    for t in all_tps:
        gt_sum[t] = sum(gt_meas[rid].get(t, 0.0) for rid in targets)
    # GT new lesions: measurable GT lesions absent at baseline, present later.
    # Collect each new lesion's per-tp diameter trajectory (for confirmation gating).
    gt_new = []
    for rid, rd in gt_meas.items():
        if rid in targets:
            continue
        if rd.get(base_tp, 0.0) < MEASURABLE_MM and any(
                t != base_tp and d >= MEASURABLE_MM for t, d in rd.items()):
            gt_new.append({t: d for t, d in rd.items() if t != base_tp})

    # MODEL sum over the SAME targets (matched lesions); missing match -> 0 at all tp
    model_meas = {pid: pd_ for pid, (pd_, pmeas, _, _) in pinfo.items() if pmeas}
    model_sum = {t: 0.0 for t in all_tps}
    matched_pids = set()
    for rid in targets:
        pid = assign.get(rid)
        if pid is None or pid not in model_meas:
            continue  # model failed to segment this target -> contributes 0
        matched_pids.add(pid)
        for t in all_tps:
            model_sum[t] += model_meas[pid].get(t, 0.0)
    # model new lesions: measurable model lesions not matched to a target, appearing later
    model_new = []
    for pid, pd_ in model_meas.items():
        if pid in matched_pids:
            continue
        if pd_.get(base_tp, 0.0) < MEASURABLE_MM and any(
                t != base_tp and d >= MEASURABLE_MM for t, d in pd_.items()):
            model_new.append({t: d for t, d in pd_.items() if t != base_tp})

    # responses under each gating level
    out = {}
    for gate, (mtps, mmm) in GATES.items():
        gr, gu = _sum_response_trajectory(gt_sum, gt_new, min_tps=mtps, min_mm=mmm)
        mr, mu = _sum_response_trajectory(model_sum, model_new, min_tps=mtps, min_mm=mmm)
        out[gate] = (gr, mr, gu, mu)
    return out


def in_plane_longest_diameter(mask, spacing, axis=AXIAL_AXIS):
    ax = [a for a in range(3) if a != axis]
    sx, sy = spacing[ax[0]], spacing[ax[1]]
    best = 0.0
    for k in range(mask.shape[axis]):
        sl = np.take(mask, k, axis=axis)
        pts = np.column_stack(np.nonzero(sl))
        if pts.shape[0] < 2:
            continue
        coords = pts.astype(float) * np.array([sx, sy])
        best = max(best, _max_pairwise(coords))
    return best

def pred_lesions(ids_arr):
    return {int(v): (ids_arr == v) for v in np.unique(ids_arr) if v != 0}

def lesion_locator(track):
    """For one tracked lesion ({tp: bool_mask}), return per-timepoint
    {tp: {"slice": z, "cx": x, "cy": y, "vox": n}} giving the largest-area axial
    slice and in-plane centroid. Lets the figure tool overlay THIS specific
    lesion's component (GT is otherwise not instance-tracked)."""
    out = {}
    for ti, m in track.items():
        if m is None or m.sum() == 0:
            continue
        areas = m.reshape(-1, m.shape[2]).sum(axis=0)
        z = int(np.argmax(areas))
        ys, xs = np.where(m[:, :, z])
        if len(xs) == 0:
            continue
        out[int(ti)] = {"slice": z, "cx": float(xs.mean()), "cy": float(ys.mean()),
                        "vox": int(m.sum())}
    return out

def cc_lesions(core_bool):
    lab, n = ndimage.label(core_bool)
    return {i: (lab == i) for i in range(1, n+1)}

def lesion_sizes(labelmap, ids):
    """voxel count per lesion id, from a label map (numpy, no python sets)."""
    return {int(i): int((labelmap == i).sum()) for i in ids}

def overlap_counts(lab_a, ids_a, lab_b, ids_b):
    """
    Pairwise overlap voxel counts between lesion ids in label map A and label map B.
    Returns dict {(ia, ib): inter_voxels} for voxels where both are nonzero.
    Pure numpy via bincount over a*K+b on the co-nonzero voxels.
    """
    both = (lab_a > 0) & (lab_b > 0)
    if not both.any():
        return {}
    a = lab_a[both].astype(np.int64)
    b = lab_b[both].astype(np.int64)
    K = int(max(ids_b)) + 1 if len(ids_b) else 1
    keys = a * K + b
    bc = np.bincount(keys)
    out = {}
    for k in np.nonzero(bc)[0]:
        ia, ib = divmod(int(k), K)
        out[(ia, ib)] = int(bc[k])
    return out

def vox_index_set(mask):
    """Flat frozenset of linear voxel indices for a boolean mask (cheap to intersect)."""
    return set(np.flatnonzero(mask.ravel()).tolist())

def union_idx(track_masks):
    """Union of voxel-index sets across a lesion's per-timepoint masks."""
    out = set()
    for m in track_masks:
        out |= vox_index_set(m)
    return out

def jaccard_idx(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)

def greedy_track(per_tp):
    """Track lesions across timepoints. Linkage uses overlap-over-minimum
    (intersection / smaller-mask area), NOT Jaccard, so a lesion that grows or
    shrinks substantially between timepoints still links to itself — Jaccard
    (intersection/union) breaks these links precisely for the high-change lesions
    whose response category matters most. Assignment between the previous frame's
    active tracks and the current frame's lesions is done optimally (Hungarian),
    preventing mis-links in multi-lesion frames.
    Name kept as greedy_track for call-site compatibility."""
    tracks, last, nid = {}, {}, 1
    for t, lesions in enumerate(per_tp):
        if not lesions:
            # Empty frame = unannotated timepoint (blank GT). Do NOT break tracks:
            # leave each track's `last` state intact so the next annotated frame can
            # still link across the gap. The timepoint simply contributes no mask.
            continue
        lids = list(lesions.keys())
        tids = list(last.keys())
        used = set()
        if tids and lids:
            # primary: overlap-over-min cost matrix between active tracks and lesions
            M = np.zeros((len(tids), len(lids)))
            for i, tid in enumerate(tids):
                lt, lm = last[tid]; a_lm = lm.sum()
                for k, lid in enumerate(lids):
                    m = lesions[lid]
                    inter = np.logical_and(lm, m).sum()
                    if inter == 0:
                        continue
                    M[i, k] = inter / max(min(a_lm, m.sum()), 1)
            ri, ci = linear_sum_assignment(-M)
            assigned_t = set()
            for i, k in zip(ri, ci):
                if M[i, k] >= TRACK_OVERLAP_MIN:
                    tid = tids[i]; lid = lids[k]
                    tracks[tid][t] = lesions[lid]; last[tid] = (t, lesions[lid])
                    used.add(lid); assigned_t.add(tid)
            # fallback: link still-unassigned tracks to unused lesions by 3D centroid
            # proximity. Handles GT connected-component merge/split/shift where the
            # lesion is physically the same but overlap dropped below threshold.
            un_t = [tid for tid in tids if tid not in assigned_t]
            un_l = [lid for lid in lids if lid not in used]
            if un_t and un_l:
                def cen(mask):
                    pts = np.argwhere(mask)
                    return pts.mean(0) if len(pts) else None
                tc = {tid: cen(last[tid][1]) for tid in un_t}
                lc = {lid: cen(lesions[lid]) for lid in un_l}
                D = np.full((len(un_t), len(un_l)), 1e9)
                for i, tid in enumerate(un_t):
                    if tc[tid] is None: continue
                    for k, lid in enumerate(un_l):
                        if lc[lid] is None: continue
                        D[i, k] = np.linalg.norm(tc[tid] - lc[lid])
                ri2, ci2 = linear_sum_assignment(D)
                for i, k in zip(ri2, ci2):
                    if D[i, k] <= TRACK_CENTROID_MM:
                        tid = un_t[i]; lid = un_l[k]
                        if lid in used: continue
                        tracks[tid][t] = lesions[lid]; last[tid] = (t, lesions[lid]); used.add(lid)
        for lid, m in lesions.items():
            if lid in used:
                continue
            tracks[nid] = {t: m}; last[nid] = (t, m); nid += 1
    return tracks

def diam_summary(diam_by_tp):
    """Return (baseline_mm, last_mm, max_fu_mm, min_fu_mm, pct_vs_base, pct_vs_nadir)
    for a lesion's per-timepoint longest-diameter dict. Drives the per-lesion CSV so
    category disagreements can be traced to the actual millimetre change."""
    tps = sorted(diam_by_tp)
    if not tps:
        return (float("nan"),)*6
    base = diam_by_tp[tps[0]]
    fu = [diam_by_tp[t] for t in tps[1:]] or [base]
    last = diam_by_tp[tps[-1]]
    max_fu = max(fu); min_fu = min(fu)
    nadir = min([base] + fu[:-1]) if len(fu) > 1 else base
    pct_base = ((last - base) / base * 100.0) if base > 0 else float("nan")
    pct_nadir = ((last - nadir) / nadir * 100.0) if nadir > 0 else float("nan")
    return (round(base,1), round(last,1), round(max_fu,1), round(min_fu,1),
            round(pct_base,1), round(pct_nadir,1))

def classify_track(diam_by_tp):
    tps = sorted(diam_by_tp); base = diam_by_tp[tps[0]]; nadir = base
    saw_pd = saw_pr = False
    for t in tps[1:]:
        d = diam_by_tp[t]
        if d >= nadir*(1+PD_FRAC) and (d - nadir) >= PD_ABS_MM: saw_pd = True
        if base > 0 and d <= base*(1-PR_FRAC): saw_pr = True
        nadir = min(nadir, d)
    if saw_pd: return "progressed"
    if saw_pr: return "improved"
    return "stable"

def vox_ml(spacing): return (spacing[0]*spacing[1]*spacing[2])/1000.0

def find_pred_ids(tracked_root, case_unit, study_id, row=None):
    # Prefer the path the pipeline recorded in dataset.csv.
    if row is not None:
        col = str(row.get("Preprocessed_label1_ids") or "").strip()
        if col and col != "nan" and os.path.exists(col):
            return col
    base = os.path.join(tracked_root, case_unit, study_id, "anat")
    hits = glob.glob(os.path.join(base, PRED_IDS_GLOB)) or glob.glob(os.path.join(base, "*.nii.gz"))
    return hits[0] if hits else None


def _score_one(task):
    """Score a single case unit; return partial results for merging. Top-level so
    it is picklable by multiprocessing. Pure function of its inputs + module config."""
    case_unit, g, tracked = task
    g = g.sort_values("TimepointOrder")
    res = {
        "contingency": np.zeros((3, 3), int),
        "per_lesion_rows": [], "fp_list": [],
        "tier_counts": {i: 0 for i in range(len(VOL_TIERS))},
        "undetected_ref": 0, "new_ref": 0, "new_pred": 0, "new_both": 0,
        "locators": {},
        "PL": {(sch, gate): {"pertp": np.zeros((4, 4), int), "best": np.zeros((4, 4), int),
                             "rows": [], "unconf": 0}
               for sch in ("5target", "allmeas") for gate in GATES},
    }
    pred_tp, ref_tp, spacing = [], [], None
    for _, row in g.iterrows():
        study_id = str(row["AnonStudyID"])
        pred_path = find_pred_ids(tracked, case_unit, study_id, row)
        gt_path = (str(row.get("GTMaskPath")) or "").strip()
        if not pred_path or not gt_path or gt_path == "nan" or not os.path.exists(gt_path):
            pred_tp.append({}); ref_tp.append({}); continue
        parr, sp = load(pred_path); spacing = sp
        garr, _ = load(gt_path)
        gcore = np.isin(garr, GT_CORE_LABELS)
        pred_tp.append(pred_lesions(parr) if TRACK_PRED == "label" else cc_lesions(parr > 0))
        ref_tp.append(cc_lesions(gcore))
    if spacing is None or len(pred_tp) < 2:
        return case_unit, res
    vml = vox_ml(spacing)

    for ti in range(len(ref_tp)):
        r_idx_ti = union_idx(ref_tp[ti].values()) if ref_tp[ti] else set()
        res["fp_list"].append(sum(1 for m in pred_tp[ti].values()
                                  if not (vox_index_set(m) & r_idx_ti)))
    for m in ref_tp[0].values():
        vol = m.sum()*vml
        for i, (lo, hi) in enumerate(VOL_TIERS):
            if lo <= vol < hi: res["tier_counts"][i] += 1; break

    if TRACK_PRED == "label":
        ptracks = {}
        for ti, lesions in enumerate(pred_tp):
            for lid, m in lesions.items(): ptracks.setdefault(lid, {})[ti] = m
    else:
        ptracks = greedy_track(pred_tp)
    rtracks = greedy_track(ref_tp)

    def info(tr):
        diam = {ti: in_plane_longest_diameter(m, spacing) for ti, m in tr.items()}
        b = min(tr); return diam, diam[b] >= MEASURABLE_MM, (b > 0), classify_track(diam)
    rinfo = {k: info(t) for k, t in rtracks.items()}
    pinfo = {k: info(t) for k, t in ptracks.items()}

    p_union_idx = {pid_: union_idx(pt.values()) for pid_, pt in ptracks.items()}
    meas_rids = [rid for rid, (_, rmeas, _, _) in rinfo.items() if rmeas]
    pid_list = list(p_union_idx.keys())
    r_idx_map = {rid: union_idx(rtracks[rid].values()) for rid in meas_rids}

    assign = {}
    if meas_rids and pid_list:
        J = np.zeros((len(meas_rids), len(pid_list)))
        for i, rid in enumerate(meas_rids):
            for k, pid_ in enumerate(pid_list):
                J[i, k] = jaccard_idx(r_idx_map[rid], p_union_idx[pid_])
        rows_i, cols_i = linear_sum_assignment(-J)
        for i, k in zip(rows_i, cols_i):
            if J[i, k] >= MATCH_JACCARD:
                assign[meas_rids[i]] = pid_list[k]

    matched_pids = set(assign.values())
    NA6 = ["NA"]*6
    for rid in meas_rids:
        rd, rmeas, rnew, rcat = rinfo[rid]
        if rnew: res["new_ref"] += 1
        rdiam = list(diam_summary(rd))
        best = assign.get(rid)
        if best is None:
            res["undetected_ref"] += 1
            if SCORE_UNDETECTED_REF:
                pcat = "stable" if rcat != "stable" else "improved"
                res["contingency"][CATS.index(rcat), CATS.index(pcat)] += 1
                res["per_lesion_rows"].append([case_unit, rid, "NA", rcat, pcat, rnew, "undetected"]
                                              + rdiam + NA6)
            continue
        pd_b, pmeas, pnew, pcat = pinfo[best]
        if pnew: res["new_pred"] += 1
        if rnew and pnew: res["new_both"] += 1
        res["contingency"][CATS.index(rcat), CATS.index(pcat)] += 1
        res["per_lesion_rows"].append([case_unit, rid, best, rcat, pcat, rnew, "matched"]
                                      + rdiam + list(diam_summary(pd_b)))
        res["locators"][f"{case_unit}|ref{rid}_pred{best}"] = {
            "ref": lesion_locator(rtracks[rid]), "pred": lesion_locator(ptracks[best])}

    if FP_NEW_LESION_IS_PD:
        for pid_, (pd_, pmeas, pnew, pcat) in pinfo.items():
            if pmeas and pnew and pid_ not in matched_pids:
                res["new_pred"] += 1
                res["per_lesion_rows"].append([case_unit, "NA", pid_, "none", "progressed",
                                               True, "pred_only_new"] + NA6 + list(diam_summary(pd_)))

    for sch, ntgt in (("5target", 5), ("allmeas", None)):
        resp = patient_level_responses(rinfo, pinfo, assign, n_target=ntgt)
        for gate, (gt_r, mod_r, gu, mu) in resp.items():
            key = (sch, gate)
            res["PL"][key]["unconf"] += gu + mu
            for t in sorted(set(gt_r) & set(mod_r)):
                gi, pi = PL_CATS.index(gt_r[t]), PL_CATS.index(mod_r[t])
                res["PL"][key]["pertp"][gi, pi] += 1
                res["PL"][key]["rows"].append([case_unit, t, sch, gate, gt_r[t], mod_r[t], "pertp"])
            gb, mb = _best_overall(gt_r), _best_overall(mod_r)
            if gb != "NE" and mb != "NE":
                res["PL"][key]["best"][PL_CATS.index(gb), PL_CATS.index(mb)] += 1
                res["PL"][key]["rows"].append([case_unit, "best", sch, gate, gb, mb, "best"])
    return case_unit, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="pipeline output dir with dataset.csv and 04_tracked/")
    ap.add_argument("--dataset-csv", default=None)
    ap.add_argument("--tracked-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", default=None,
                    help="parallel worker processes over case units (default: all CPUs)")
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    ds_csv = args.dataset_csv or os.path.join(out, "dataset.csv")
    tracked = args.tracked_root or os.path.join(out, "04_tracked")
    out_dir = args.out_dir or os.path.join(out, "ranobm_out")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(ds_csv, dtype=str)
    if "NormalizedSeriesDescription" in df.columns:
        df = df[df["NormalizedSeriesDescription"] == "T1Post"].copy()
    df["TimepointOrder"] = df["TimepointOrder"].astype(int)

    contingency = np.zeros((3, 3), int)   # rows=reference, cols=predicted
    per_lesion_rows, fp_per_study = [], []
    tier_counts = {i: 0 for i in range(len(VOL_TIERS))}
    undetected_ref = new_ref = new_pred = new_both = 0
    locators = {}   # "case_unit|ref<rid>_pred<pid>" -> {"ref": {...}, "pred": {...}}
    # patient-level RANO-BM: {scheme: {"pertp": 4x4 contingency, "best": 4x4, "rows": [...]}}
    PL = {(sch, gate): {"pertp": np.zeros((4, 4), int), "best": np.zeros((4, 4), int),
                        "rows": [], "unconf": 0}
          for sch in ("5target", "allmeas") for gate in GATES}

    import time as _time, multiprocessing as _mp
    _t0 = _time.time()
    _cu_list = list(df.groupby("AnonPatientID"))
    tasks = [(cu, g, tracked) for cu, g in _cu_list]
    def _allocated_cpus():
        n = os.environ.get("SLURM_CPUS_PER_TASK")
        if n:
            return int(n)
        try:
            return len(os.sched_getaffinity(0))  # respects the Slurm cgroup cpuset
        except AttributeError:
            return os.cpu_count() or 1
    workers = max(1, int(args.workers) if args.workers else _allocated_cpus())
    workers = min(workers, len(tasks))
    print(f"Scoring {len(tasks)} case units with {workers} worker(s)...", flush=True)

    partials = []
    if workers == 1:
        for i, t in enumerate(tasks, 1):
            cu, r = _score_one(t)
            partials.append((cu, r))
            print(f"[{i}/{len(tasks)}] {cu}  (+{_time.time()-_t0:.1f}s)", flush=True)
    else:
        with _mp.Pool(workers) as pool:
            for i, (cu, r) in enumerate(pool.imap_unordered(_score_one, tasks), 1):
                partials.append((cu, r))
                print(f"[{i}/{len(tasks)}] {cu}  (+{_time.time()-_t0:.1f}s)", flush=True)

    # ---- deterministic merge (sort by case unit so output is run-order independent) ----
    partials.sort(key=lambda x: x[0])
    for _cu, r in partials:
        contingency += r["contingency"]
        per_lesion_rows += r["per_lesion_rows"]
        fp_per_study += r["fp_list"]
        for i in r["tier_counts"]: tier_counts[i] += r["tier_counts"][i]
        undetected_ref += r["undetected_ref"]; new_ref += r["new_ref"]
        new_pred += r["new_pred"]; new_both += r["new_both"]
        locators.update(r["locators"])
        for key, v in r["PL"].items():
            PL[key]["pertp"] += v["pertp"]; PL[key]["best"] += v["best"]
            PL[key]["rows"] += v["rows"]; PL[key]["unconf"] += v["unconf"]
    # sort row lists for fully deterministic CSV output
    per_lesion_rows.sort(key=lambda row: (str(row[0]), str(row[1]), str(row[2])))
    for key in PL: PL[key]["rows"].sort(key=lambda row: (str(row[0]), str(row[1])))

    total = int(contingency.sum()); diag = int(np.trace(contingency))
    print("\n=== PRIMARY modified RANO-BM (>=10 mm measurable) ===")
    print(f"Measurable reference target lesions scored: {total}")
    print(f"Undetected reference targets: {undetected_ref}  (SCORE_UNDETECTED_REF={SCORE_UNDETECTED_REF})")
    print("\nContingency rows=reference cols=predicted, order =", CATS)
    print(contingency)
    p, lo, hi = wilson(diag, total)
    print(f"\nOverall agreement: {diag}/{total} = {p:.1f}% (95% CI {lo:.0f}-{hi:.0f})")
    for i, c in enumerate(CATS):
        n = int(contingency[i].sum()); x = int(contingency[i, i])
        if n:
            pp, ll, hh = wilson(x, n); print(f"  {c:11s}: {x}/{n} = {pp:.1f}% (95% CI {ll:.0f}-{hh:.0f})")
        else:
            print(f"  {c:11s}: n=0")
    print(f"\nNew measurable lesions  ref={new_ref}  pred={new_pred}  concordant={new_both}")
    fp = np.array(fp_per_study) if fp_per_study else np.array([0])
    print(f"\nFalse positives/study: mean={fp.mean():.2f} median={np.median(fp):.0f} "
          f"IQR={np.percentile(fp,25):.0f}-{np.percentile(fp,75):.0f} "
          f">=1 FP in {(fp>=1).mean()*100:.0f}% of studies")
    print("\nReference lesion denominators by volume tier (baseline):")
    for i,(a,b) in enumerate(VOL_TIERS): print(f"  {a}-{b} mL: n={tier_counts[i]}")

    # ---- patient-level RANO-BM reporting ----
    def _kappa(C):
        C = C.astype(float); n = C.sum()
        if n == 0: return float("nan")
        po = np.trace(C) / n
        pe = (C.sum(0) * C.sum(1)).sum() / (n * n)
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")

    pl_rows_all = []
    print("\n=== PATIENT-LEVEL RANO-BM (GT-derived vs model-derived response) ===")
    print("Categories order:", PL_CATS)
    print(f"Sweep: new-lesion floor {NEW_MM_SWEEP} mm x persistence {PERSIST_SWEEP} TP "
          f"(1=strict, 2=confirmed). PRIMARY = {PRIMARY_GATE_NAME}")
    # compact sweep summary (5-target scheme; per-timepoint and best-overall)
    for unit, ulabel in (("pertp", "per-timepoint"), ("best", "best-overall (unconfirmed)")):
        print(f"\n-- sweep summary | 5 target lesions | {ulabel} (concordance%, kappa) --")
        print(f"  {'new-floor':>9} | " + " | ".join(f"{'strict' if n==1 else 'confirmed':>10}" for n in PERSIST_SWEEP))
        for mm in NEW_MM_SWEEP:
            cells = []
            for ntp in PERSIST_SWEEP:
                C = PL[("5target", _gate_name(mm, ntp))][unit]
                tot = int(C.sum()); dg = int(np.trace(C))
                star = "*" if _gate_name(mm, ntp) == PRIMARY_GATE_NAME else " "
                cells.append(f"{(100*dg/tot if tot else 0):4.0f}% k={_kappa(C):.2f}{star}" if tot else "   n/a    ")
            print(f"  {int(mm):>7}mm | " + " | ".join(f"{c:>10}" for c in cells))
    # full detail for the PRIMARY gate (both units), with the contingency matrices
    print(f"\n-- PRIMARY: {PRIMARY_GATE_NAME} (5 target lesions) --")
    for unit, ulabel in (("pertp", "per-timepoint"), ("best", "best-overall (unconfirmed)")):
        C = PL[("5target", PRIMARY_GATE_NAME)][unit]; tot = int(C.sum()); dg = int(np.trace(C))
        print(f"  [{ulabel}]  rows=GT cols=model  {PL_CATS}")
        print(C)
        if tot:
            p, lo, hi = wilson(dg, tot)
            print(f"  Concordance: {dg}/{tot} = {p:.1f}% (95% CI {lo:.0f}-{hi:.0f}); Cohen kappa = {_kappa(C):.2f}")
    # also print STRICT-10mm as the autonomous-baseline comparator
    strict10 = _gate_name(10.0, 1)
    Cs = PL[("5target", strict10)]["pertp"]; ts = int(Cs.sum()); ds = int(np.trace(Cs))
    if ts:
        ps, los, his = wilson(ds, ts)
        print(f"\n-- comparator: strict 10mm (autonomous, no confirmation) | per-timepoint: "
              f"{ds}/{ts} = {ps:.1f}% (95% CI {los:.0f}-{his:.0f}); kappa={_kappa(Cs):.2f}")
    for (sch, gate) in PL:
        pl_rows_all += PL[(sch, gate)]["rows"]

    pd.DataFrame(pl_rows_all,
                 columns=["case_unit", "timepoint", "scheme", "gate", "gt_response", "model_response", "unit"]
                 ).to_csv(os.path.join(out_dir, "patient_level_ranobm.csv"), index=False)
    for (sch, gate), v in PL.items():
        for unit in ("pertp", "best"):
            pd.DataFrame(v[unit], index=[f"gt_{c}" for c in PL_CATS],
                         columns=[f"mod_{c}" for c in PL_CATS]).to_csv(
                os.path.join(out_dir, f"patient_contingency_{sch}_{gate}_{unit}.csv"))

    pd.DataFrame(contingency, index=[f"ref_{c}" for c in CATS],
                 columns=[f"pred_{c}" for c in CATS]).to_csv(os.path.join(out_dir, "contingency_ranobm.csv"))
    pd.DataFrame(per_lesion_rows,
                 columns=["case_unit","ref_id","pred_id","ref_cat","pred_cat","ref_new","status",
                          "ref_base_mm","ref_last_mm","ref_maxfu_mm","ref_minfu_mm","ref_pct_base","ref_pct_nadir",
                          "pred_base_mm","pred_last_mm","pred_maxfu_mm","pred_minfu_mm","pred_pct_base","pred_pct_nadir"]
                 ).to_csv(os.path.join(out_dir, "per_lesion_ranobm.csv"), index=False)
    import json
    with open(os.path.join(out_dir, "lesion_locators.json"), "w") as _lf:
        json.dump(locators, _lf)
    print(f"Wrote lesion locators for {len(locators)} matched lesions to lesion_locators.json")
    print(f"\nWrote CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
