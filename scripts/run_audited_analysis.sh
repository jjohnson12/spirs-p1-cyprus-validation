#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 config/paths.env" >&2
  exit 2
fi

CONFIG_FILE=$1
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

REQUIRED_VARS=(CYPRUS_ZIP_ROOT CYPRUS_OUTPUT_ROOT CYPRUS_DATASET_CSV CYPRUS_TRACKED_ROOT CYPRUS_RESULTS_ROOT)
for REQUIRED_VAR in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!REQUIRED_VAR:-}" ]]; then
    echo "Missing required setting: $REQUIRED_VAR" >&2
    exit 2
  fi
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIR="$REPO_ROOT/src"
N_WORKERS=${N_WORKERS:-4}

mkdir -p "$CYPRUS_RESULTS_ROOT"

python "$SRC_DIR/audit_cyprus_denominators_zips.py" \
  --zip-root "$CYPRUS_ZIP_ROOT" \
  --out-dir "$CYPRUS_RESULTS_ROOT/denominator_audit"

python "$SRC_DIR/relink_cyprus_ground_truth.py" \
  --dataset "$CYPRUS_DATASET_CSV" \
  --zip-root "$CYPRUS_ZIP_ROOT" \
  --output-root "$CYPRUS_OUTPUT_ROOT/03_ground_truth_relinked" \
  --out-dataset "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --summary "$CYPRUS_RESULTS_ROOT/relink_summary.txt"

python "$SRC_DIR/rescore_cyprus_relinked_v2.py" \
  --dataset "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --out-dir "$CYPRUS_RESULTS_ROOT/cross_sectional_rescore"

PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$SRC_DIR/patient_cluster_bootstrap_cross_sectional_v3_direct.py" \
  --output-dir "$CYPRUS_OUTPUT_ROOT" \
  --dataset-csv "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --tracked-root "$CYPRUS_TRACKED_ROOT" \
  --out-dir "$CYPRUS_RESULTS_ROOT/bootstrap_clustered_cross_sectional" \
  --n-bootstrap 10000 \
  --seed 20260809

PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$SRC_DIR/rescore_longitudinal_volume_change_relinked_v2.py" \
  --output-dir "$CYPRUS_OUTPUT_ROOT" \
  --dataset-csv "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --tracked-root "$CYPRUS_TRACKED_ROOT" \
  --out-dir "$CYPRUS_RESULTS_ROOT/longitudinal_volume_change"

PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$SRC_DIR/ranobm_endpoint_v4_reference_anchored.py" \
  --output-dir "$CYPRUS_OUTPUT_ROOT" \
  --dataset-csv "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --tracked-root "$CYPRUS_TRACKED_ROOT" \
  --out-dir "$CYPRUS_RESULTS_ROOT/ranobm_v4" \
  --workers "$N_WORKERS"

python "$SRC_DIR/finalize_ranobm_v4_primary_v2.py" \
  --v4-dir "$CYPRUS_RESULTS_ROOT/ranobm_v4" \
  --out-dir "$CYPRUS_RESULTS_ROOT/ranobm_v4_final"

PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$SRC_DIR/audit_ranobm_new_lesions_v4.py" \
  --output-dir "$CYPRUS_OUTPUT_ROOT" \
  --dataset-csv "$CYPRUS_RESULTS_ROOT/dataset_relinked.csv" \
  --tracked-root "$CYPRUS_TRACKED_ROOT" \
  --out-dir "$CYPRUS_RESULTS_ROOT/ranobm_new_lesion_audit_v4"

echo "Audited outputs written under: $CYPRUS_RESULTS_ROOT"
