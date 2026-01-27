#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7,8,9}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR

NPROC=$(python - <<'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","")
print(len([x for x in v.split(",") if x.strip()!=""]) if v else 1)
PY
)

python scripts/collect_results.py --mode run_sensitivity \
  --nproc "${NPROC}" \
  --master_port_base 29911 \
  --paths configs/paths.yaml \
  --tri_config configs/iemocap_tri.yaml \
  --sensitivity_config configs/iemocap_sensitivity.yaml
