#!/usr/bin/env bash
set -euo pipefail

# EXPECTATION:
#   You already ran:
#     conda activate chado_meld
#     cd /home/tahirahmad/CHADO_MELD

BASE_CONFIG="configs/chado_meld.yaml"
BASELINE_CKPT="/home/tahirahmad/CHADO_MELD/runs/baseline_trimodal_meld_best.pt"

OUT_ROOT="runs/chado_component_wo_ablations"
CFG_DIR="configs/chado_component_wo_ablations"
SUMMARY_CSV="${OUT_ROOT}/component_removal_summary.csv"
LATEX_OUT="${OUT_ROOT}/component_removal_table.tex"

SPLIT="test"
BATCH_SIZE="8"
NUM_WORKERS="6"

mkdir -p "${OUT_ROOT}" "${CFG_DIR}"
echo "ablation,use_causal,use_hyperbolic,use_transport,accuracy,precision_w,recall_w,f1_w,ckpt_path" > "${SUMMARY_CSV}"

run_one () {
  local name="$1"
  local use_causal="$2"
  local use_hyp="$3"
  local use_ot="$4"
  local port="$5"

  local run_dir="${OUT_ROOT}/${name}"
  local cfg_out="${CFG_DIR}/${name}.yaml"
  mkdir -p "${run_dir}"

  echo "============================================================"
  echo "[RUN] ${name}"
  echo " cfg=${cfg_out}"
  echo " out_dir=${run_dir}"
  echo " toggles: causal=${use_causal} hyp=${use_hyp} ot=${use_ot}"
  echo " master_port=${port}"
  echo "============================================================"

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

  # Train on 5 GPUs (must set CUDA_VISIBLE_DEVICES outside or here)
  torchrun --nproc_per_node=5 --master_port="${port}" \
    src/train/train_chado.py --config "${cfg_out}"

  # Evaluate best checkpoint produced by train_chado.py
  local best_ckpt="${run_dir}/best.pt"
  if [[ ! -f "${best_ckpt}" ]]; then
    echo "[FATAL] best.ckpt not found: ${best_ckpt}"
    exit 1
  fi

  MET=$(python scripts/eval_meld_ckpt_metrics.py \
    --config "${cfg_out}" \
    --ckpt "${best_ckpt}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    | python - << 'PY'
import sys, re
txt=sys.stdin.read()
def grab(k):
  m=re.search(rf"^{re.escape(k)}\s*:\s*([0-9.]+)", txt, flags=re.M)
  return m.group(1) if m else ""
acc=grab("Accuracy")
pw=grab("Precision (weighted)")
rw=grab("Recall (weighted)")
f1=grab("F1 (weighted)")
print(f"{acc},{pw},{rw},{f1}")
PY
)

  echo "${name},${use_causal},${use_hyp},${use_ot},${MET},${best_ckpt}" >> "${SUMMARY_CSV}"
  echo "[OK] wrote row -> ${SUMMARY_CSV}"
}

# 6 professor-required "WITHOUT" ablations:
# w/o Causal
# w/o Hyperbolic
# w/o Transport
# w/o (Causal + Hyperbolic)
# w/o (Causal + Transport)
# w/o (Hyperbolic + Transport)

run_one "wo_causal"                0 1 1 29501
run_one "wo_hyperbolic"            1 0 1 29502
run_one "wo_transport"             1 1 0 29503
run_one "wo_causal_hyperbolic"     0 0 1 29504
run_one "wo_causal_transport"      0 1 0 29505
run_one "wo_hyperbolic_transport"  1 0 0 29506

echo "============================================================"
echo "[DONE] Ablations complete."
echo "[CSV] ${SUMMARY_CSV}"
echo "============================================================"




# #!/usr/bin/env bash
# set -euo pipefail

# ###############################################################################
# # Robust conda activation (works in non-interactive scripts; no conda init needed)
# ###############################################################################
# set +u
# if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
#   _CONDA="${CONDA_EXE}"
# elif command -v conda >/dev/null 2>&1; then
#   _CONDA="$(command -v conda)"
# elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
#   _CONDA="$HOME/miniconda3/bin/conda"
# elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
#   _CONDA="$HOME/anaconda3/bin/conda"
# else
#   echo "[FATAL] conda not found. Check conda installation path." >&2
#   exit 1
# fi
# eval "$("${_CONDA}" shell.bash hook)"
# conda activate chado_meld
# set -u

# ###############################################################################
# # Project + environment
# ###############################################################################
# cd /home/tahirahmad/CHADO_MELD

# export CUDA_VISIBLE_DEVICES=5,6,7,8,9
# export OMP_NUM_THREADS=1
# export TOKENIZERS_PARALLELISM=false
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# # Offline HF (avoid 401 / downloads)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# # DDP stability knobs (helps "marked ready twice" in many checkpointing cases)
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_DDP_STATIC_GRAPH=1
# # Disable HF checkpointing behaviors if they are enabled indirectly
# export HF_DISABLE_GRADIENT_CHECKPOINTING=1

# ###############################################################################
# # Paths / settings
# ###############################################################################
# BASE_CONFIG="configs/chado_meld.yaml"
# BASELINE_CKPT="/CHADO_MELD/runs/baseline_trimodal_meld_best.pt"

# OUT_ROOT="runs/ablations_component_removal_meld"
# CFG_DIR="configs/ablations_component_removal_meld"
# SUMMARY_CSV="${OUT_ROOT}/component_removal_summary.csv"

# SPLIT="test"
# BATCH_SIZE="8"
# NUM_WORKERS="6"

# mkdir -p "${OUT_ROOT}" "${CFG_DIR}"
# echo "ablation,use_causal,use_hyperbolic,use_transport,accuracy,precision_w,recall_w,f1_w" > "${SUMMARY_CSV}"

# ###############################################################################
# # Helper: run one ablation
# ###############################################################################
# run_one () {
#   local name="$1"
#   local use_causal="$2"
#   local use_hyp="$3"
#   local use_ot="$4"
#   local port="$5"

#   local run_dir="${OUT_ROOT}/${name}"
#   local cfg_out="${CFG_DIR}/${name}.yaml"
#   mkdir -p "${run_dir}"

#   echo "------------------------------------------------------------"
#   echo "[RUN] ${name}"
#   echo " out_dir=${run_dir}"
#   echo " cfg=${cfg_out}"
#   echo " use_causal=${use_causal} use_hyperbolic=${use_hyp} use_transport=${use_ot}"
#   echo " MASTER_PORT=${port}"
#   echo "------------------------------------------------------------"

#   # 1) Write config for this ablation
#   python scripts/make_meld_ablation_cfg.py \
#     --base_config "${BASE_CONFIG}" \
#     --out_config "${cfg_out}" \
#     --run_name "${name}" \
#     --out_dir "${run_dir}" \
#     --use_causal "${use_causal}" \
#     --use_hyperbolic "${use_hyp}" \
#     --use_transport "${use_ot}" \
#     --use_refinement 1 \
#     --baseline_ckpt "${BASELINE_CKPT}"

#   # 2) Train CHADO (DDP: 5 GPUs)
#   # Note: if your train_chado.py currently only evaluates and exits,
#   # this will still finish quickly; the eval step below will still run.
#   torchrun --nproc_per_node=5 --master_port="${port}" \
#     src/train/train_chado.py --config "${cfg_out}"

#   # 3) Evaluate best ckpt in run_dir and parse metrics
#   # Expect eval script to print lines like:
#   # Accuracy: 0.6372
#   # Precision (weighted): 0.638x
#   # Recall (weighted): 0.637x
#   # F1 (weighted): 0.6387
#   METRICS=$(
#     RUN_DIR="${run_dir}" CFG_OUT="${cfg_out}" SPLIT="${SPLIT}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
#     python - << 'PY'
# import os, re, sys, subprocess

# run_dir = os.environ["RUN_DIR"]
# cfg_out = os.environ["CFG_OUT"]
# split = os.environ["SPLIT"]
# batch = os.environ["BATCH_SIZE"]
# workers = os.environ["NUM_WORKERS"]

# cmd = [
#     sys.executable, "scripts/eval_meld_run_dir.py",
#     "--run_dir", run_dir,
#     "--config", cfg_out,
#     "--split", split,
#     "--batch_size", str(batch),
#     "--num_workers", str(workers),
# ]
# p = subprocess.run(cmd, capture_output=True, text=True)
# print(p.stdout)
# if p.returncode != 0:
#     print(p.stderr)
#     raise SystemExit(p.returncode)

# txt = p.stdout

# def grab(key):
#     m = re.search(rf"^{re.escape(key)}\s*:\s*([0-9.]+)", txt, flags=re.MULTILINE)
#     return m.group(1) if m else ""

# acc = grab("Accuracy")
# pw  = grab("Precision (weighted)")
# rw  = grab("Recall (weighted)")
# f1w = grab("F1 (weighted)")

# print("::PARSED::", acc, pw, rw, f1w)
# PY
#   )

#   PARSED=$(echo "${METRICS}" | grep "::PARSED::" | tail -n 1 | awk '{print $2","$3","$4","$5}')
#   if [[ -z "${PARSED}" ]]; then
#     echo "[FATAL] Could not parse metrics from eval output for ${name}." >&2
#     echo "-------- RAW EVAL OUTPUT START --------"
#     echo "${METRICS}"
#     echo "-------- RAW EVAL OUTPUT END ----------"
#     exit 1
#   fi

#   echo "[PARSED] ${PARSED}"
#   echo "${name},${use_causal},${use_hyp},${use_ot},${PARSED}" >> "${SUMMARY_CSV}"
#   echo "[OK] appended to ${SUMMARY_CSV}"
# }

# ###############################################################################
# # Run the 6 component-removal ablations (ports must be unique)
# ###############################################################################
# run_one "wo_causal"                0 1 1 29501
# run_one "wo_hyperbolic"            1 0 1 29502
# run_one "wo_transport"             1 1 0 29503
# run_one "wo_causal_hyperbolic"     0 0 1 29504
# run_one "wo_causal_transport"      0 1 0 29505
# run_one "wo_hyperbolic_transport"  1 0 0 29506

# echo "============================================================"
# echo "[DONE] Ablations complete."
# echo "[CSV ] ${SUMMARY_CSV}"
# echo "============================================================"
