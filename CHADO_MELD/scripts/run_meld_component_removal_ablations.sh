#!/usr/bin/env bash
set -euo pipefail

# -------- Conda (REQUIRED for scripts) --------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chado_meld

cd /home/tahirahmad/CHADO_MELD

export CUDA_VISIBLE_DEVICES=5,6,7,8,9
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# If your node is offline for HF downloads
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${OUT_ROOT}" "${CFG_DIR}"
echo "ablation,use_causal,use_hyperbolic,use_transport,accuracy,precision_w,recall_w,f1_w" > "${SUMMARY_CSV}"

run_one () {
  local name="$1"
  local use_causal="$2"
  local use_hyp="$3"
  local use_ot="$4"
  local port="$5"

  local run_dir="${OUT_ROOT}/${name}"
  local cfg_out="${CFG_DIR}/${name}.yaml"
  mkdir -p "${run_dir}"

  echo "------------------------------------------------------------"
  echo "[RUN] ${name}"
  echo " out_dir=${run_dir}"
  echo " cfg=${cfg_out}"
  echo " use_causal=${use_causal} use_hyperbolic=${use_hyp} use_transport=${use_ot}"
  echo " MASTER_PORT=${port}"
  echo "------------------------------------------------------------"

  python scripts/make_meld_ablation_cfg.py \
    --base_config "${BASE_CONFIG}" \
    --out_config "${cfg_out}" \
    --run_name "${name}" \
    --out_dir "${run_dir}" \
    --use_causal "${use_causal}" \
    --use_hyperbolic "${use_hyp}" \
    --use_transport "${use_ot}" \
    --use_refinement 1 \
    --baseline_ckpt "${BASELINE_CKPT}"

  # Train (5 GPUs)
  torchrun --nproc_per_node=5 --master_port="${port}" \
    src/train/train_chado.py --config "${cfg_out}"

  # Evaluate best checkpoint from the run directory using your metrics script
  METRICS=$(python - << 'PY'
import os, re, sys, subprocess

run_dir = os.environ["RUN_DIR"]
cfg_out = os.environ["CFG_OUT"]
split = os.environ["SPLIT"]
batch = os.environ["BATCH_SIZE"]
workers = os.environ["NUM_WORKERS"]

cmd = [
    sys.executable, "scripts/eval_meld_run_dir.py",
    "--run_dir", run_dir,
    "--config", cfg_out,
    "--split", split,
    "--batch_size", str(batch),
    "--num_workers", str(workers),
]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout)
if p.returncode != 0:
    print(p.stderr)
    raise SystemExit(p.returncode)

txt = p.stdout
def grab(key):
    m = re.search(rf"^{re.escape(key)}\\s*:\\s*([0-9.]+)", txt, flags=re.MULTILINE)
    return m.group(1) if m else ""

acc = grab("Accuracy")
pw  = grab("Precision (weighted)")
rw  = grab("Recall (weighted)")
f1w = grab("F1 (weighted)")
print("::PARSED::", acc, pw, rw, f1w)
PY
)

  PARSED=$(echo "${METRICS}" | grep "::PARSED::" | tail -n 1 | awk '{print $2","$3","$4","$5}')
  echo "[PARSED] ${PARSED}"
  echo "${name},${use_causal},${use_hyp},${use_ot},${PARSED}" >> "${SUMMARY_CSV}"
  echo "[OK] appended to ${SUMMARY_CSV}"
}

export SPLIT BATCH_SIZE NUM_WORKERS

# 6 ablations
export RUN_DIR CFG_OUT

RUN_DIR="${OUT_ROOT}/wo_causal" CFG_OUT="${CFG_DIR}/wo_causal.yaml" run_one "wo_causal" 0 1 1 29501
RUN_DIR="${OUT_ROOT}/wo_hyperbolic" CFG_OUT="${CFG_DIR}/wo_hyperbolic.yaml" run_one "wo_hyperbolic" 1 0 1 29502
RUN_DIR="${OUT_ROOT}/wo_transport" CFG_OUT="${CFG_DIR}/wo_transport.yaml" run_one "wo_transport" 1 1 0 29503
RUN_DIR="${OUT_ROOT}/wo_causal_hyperbolic" CFG_OUT="${CFG_DIR}/wo_causal_hyperbolic.yaml" run_one "wo_causal_hyperbolic" 0 0 1 29504
RUN_DIR="${OUT_ROOT}/wo_causal_transport" CFG_OUT="${CFG_DIR}/wo_causal_transport.yaml" run_one "wo_causal_transport" 0 1 0 29505
RUN_DIR="${OUT_ROOT}/wo_hyperbolic_transport" CFG_OUT="${CFG_DIR}/wo_hyperbolic_transport.yaml" run_one "wo_hyperbolic_transport" 1 0 0 29506

echo "============================================================"
echo "[DONE] Ablations complete."
echo "[CSV ] ${SUMMARY_CSV}"
echo "============================================================"
