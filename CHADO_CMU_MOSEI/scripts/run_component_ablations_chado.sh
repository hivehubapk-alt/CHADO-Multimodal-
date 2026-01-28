#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CHADO Component Ablations (NO screen)
# GPUs: 5,6,7,8,9  (5 GPUs)
#
# Runs:
#   - ALL_CHADO (optional but recommended)
#   - WO_CAUSAL
#   - WO_HYPERBOLIC
#   - WO_TRANSPORT
#   - WO_CAUSAL_HYPERBOLIC
#   - WO_CAUSAL_TRANSPORT
#   - WO_HYPERBOLIC_TRANSPORT
#
# Saves:
#   - logs per run: outputs/ablations_components/<TAG>/run.log
#   - summary CSV : outputs/ablations_components/component_ablation_summary.csv
# ============================================================

PROJECT_ROOT="/home/tahirahmad/CHADO_CMU_MOSEI"
CFG="src/configs/chado_mosei_emo6.yaml"

# fixed manifests (edit if you want LOEO fold manifests later)
TRAIN="data/manifests/mosei_train.jsonl"
VAL="data/manifests/mosei_val.jsonl"
TEST="data/manifests/mosei_test.jsonl"

# modality is fixed for component ablation table
ABL="TAV"

OUT_BASE="outputs/ablations_components"
mkdir -p "${OUT_BASE}"

# environment
export CUDA_VISIBLE_DEVICES=5,6,7,8,9
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF

NPROC=5
MASTER_PORT=29780

SUMMARY_CSV="${OUT_BASE}/component_ablation_summary.csv"
echo "tag,test_acc,test_wf1" > "${SUMMARY_CSV}"

extract_last_metrics () {
  # Parse LAST occurrence of: [TEST] Acc=XX.XX%  WF1=YY.YY
  # from a log file.
  local LOGFILE="$1"
  local line
  line="$(grep -E "\[TEST\] Acc=" "${LOGFILE}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ","
    return
  fi

  # Example:
  # [TEST] Acc=68.04%  WF1=57.22  T=1.300
  local acc wf1
  acc="$(echo "${line}" | sed -n 's/.*Acc=\([0-9.]\+\)%.*/\1/p')"
  wf1="$(echo "${line}" | sed -n 's/.*WF1=\([0-9.]\+\).*/\1/p')"
  echo "${acc},${wf1}"
}

run_one () {
  local TAG="$1"
  shift
  local OVERRIDES=("$@")

  local OUT_DIR="${OUT_BASE}/${TAG}"
  mkdir -p "${OUT_DIR}"
  local LOG="${OUT_DIR}/run.log"

  echo "============================================================" | tee "${LOG}"
  echo "[RUN] ${TAG}  ABL=${ABL}" | tee -a "${LOG}"
  echo "[OUT] ${OUT_DIR}" | tee -a "${LOG}"
  echo "[OVERRIDES] ${OVERRIDES[*]}" | tee -a "${LOG}"
  echo "============================================================" | tee -a "${LOG}"

  # IMPORTANT:
  # This assumes your src/train/train.py supports:
  # --train_manifest --val_manifest --test_manifest --out_root --override
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    src/train/train.py \
      --cfg "${CFG}" \
      --ablation "${ABL}" \
      --train_manifest "${TRAIN}" \
      --val_manifest "${VAL}" \
      --test_manifest "${TEST}" \
      --out_root "${OUT_DIR}" \
      $(printf -- "--override %q " "${OVERRIDES[@]}") \
    2>&1 | tee -a "${LOG}"

  # parse metrics and append to CSV
  local metrics
  metrics="$(extract_last_metrics "${LOG}")"
  echo "${TAG},${metrics}" >> "${SUMMARY_CSV}"

  echo "[SAVED] ${SUMMARY_CSV}" | tee -a "${LOG}"

  MASTER_PORT=$((MASTER_PORT+1))
}

cd "${PROJECT_ROOT}"

# -------------------------
# Optional baseline: All components enabled
# -------------------------
run_one "ALL_CHADO" \
  "chado.causal.enable=true" \
  "chado.hyperbolic.enable=true" \
  "chado.ot.enable=true"

# -------------------------
# Required ablations
# -------------------------
run_one "WO_CAUSAL" \
  "chado.causal.enable=false" \
  "chado.hyperbolic.enable=true" \
  "chado.ot.enable=true"

run_one "WO_HYPERBOLIC" \
  "chado.causal.enable=true" \
  "chado.hyperbolic.enable=false" \
  "chado.ot.enable=true"

run_one "WO_TRANSPORT" \
  "chado.causal.enable=true" \
  "chado.hyperbolic.enable=true" \
  "chado.ot.enable=false"

run_one "WO_CAUSAL_HYPERBOLIC" \
  "chado.causal.enable=false" \
  "chado.hyperbolic.enable=false" \
  "chado.ot.enable=true"

run_one "WO_CAUSAL_TRANSPORT" \
  "chado.causal.enable=false" \
  "chado.hyperbolic.enable=true" \
  "chado.ot.enable=false"

run_one "WO_HYPERBOLIC_TRANSPORT" \
  "chado.causal.enable=true" \
  "chado.hyperbolic.enable=false" \
  "chado.ot.enable=false"

echo "============================================================"
echo "[DONE] All component ablations finished."
echo "[CSV ] ${SUMMARY_CSV}"
echo "============================================================"
