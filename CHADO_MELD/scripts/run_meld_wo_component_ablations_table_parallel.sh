#!/usr/bin/env bash
set -euo pipefail

# Ensure conda works inside non-interactive bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chado_meld

cd /home/tahirahmad/CHADO_MELD

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CFG_BASE="configs/chado_component_wo_ablations.yaml"
OUTDIR="runs/chado_component_wo_ablations"
mkdir -p "${OUTDIR}"

# We will run one ablation per process, pinned to a GPU.
# 5 GPUs available -> run 5 at once, the 6th will start when one finishes.

ABLATIONS=(
  "wo_causal"
  "wo_hyperbolic"
  "wo_transport"
  "wo_causal_hyperbolic"
  "wo_causal_transport"
  "wo_hyperbolic_transport"
  "all_chado"
)

GPUS=(5 6 7 8 9)

# Function to run a single ablation by creating a temporary YAML that contains ONLY that ablation
run_one () {
  local ablation_name="$1"
  local gpu_id="$2"

  local tmp_cfg="${OUTDIR}/${ablation_name}.yaml"

  # Create a one-ablation config by copying the base and filtering at runtime inside python:
  # If your python script does not support selecting an ablation, we create a YAML containing only one.
  python - <<PY
import yaml, copy
base_path="${CFG_BASE}"
ab="${ablation_name}"
out_path="${tmp_cfg}"

with open(base_path, "r") as f:
    cfg = yaml.safe_load(f)

abl = cfg.get("ablations", [])
sel = [a for a in abl if a.get("name") == ab]
if not sel:
    raise SystemExit(f"[FATAL] ablation '{ab}' not found in {base_path}")

cfg2 = copy.deepcopy(cfg)
cfg2["ablations"] = sel

# per-ablation output files (so processes don't overwrite each other)
cfg2.setdefault("output", {})
cfg2["output"]["out_dir"] = cfg.get("output", {}).get("out_dir", "${OUTDIR}")
cfg2["output"]["csv_name"] = f"{ab}_results.csv"
cfg2["output"]["latex_name"] = f"{ab}_table.tex"

with open(out_path, "w") as f:
    yaml.safe_dump(cfg2, f, sort_keys=False)

print(f"[OK] wrote {out_path}")
PY

  echo "------------------------------------------------------------"
  echo "[RUN] ${ablation_name} on GPU ${gpu_id}"
  echo " cfg=${tmp_cfg}"
  echo "------------------------------------------------------------"

  python scripts/run_chado_component_ablations_table.py \
    --config "${tmp_cfg}" \
    --gpu "${gpu_id}"

  echo "[DONE] ${ablation_name} on GPU ${gpu_id}"
}

# Simple job scheduler: keep at most 5 jobs running (one per GPU)
pids=()
idx=0

for ab in "${ABLATIONS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${ab}" "${gpu}" &
  pids+=($!)
  idx=$((idx+1))

  # If we've launched 5 jobs, wait for all before launching more
  if (( idx % ${#GPUS[@]} == 0 )); then
    for pid in "${pids[@]}"; do wait "${pid}"; done
    pids=()
  fi
done

# wait remaining
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "============================================================"
echo "[OK] All ablations completed."
echo "Outputs in: ${OUTDIR}"
echo "============================================================"
