#!/usr/bin/env bash
set -euo pipefail

# You can edit these:
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

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] nproc_per_node=${NPROC}"

python - <<'PY'
import yaml, re
from pathlib import Path
import os

def subst_env(s: str):
    for k,v in os.environ.items():
        s = s.replace("${%s}"%k, v)
    return s

paths = yaml.safe_load(open("configs/paths.yaml","r"))
cfg   = yaml.safe_load(open("configs/iemocap_tri.yaml","r"))

root = paths["project_root"]
train_csv = subst_env(paths["splits"]["train_csv"])
val_csv   = subst_env(paths["splits"]["val_csv"])
test_csv  = subst_env(paths["splits"]["test_csv"])
out_root  = subst_env(paths["outputs_root"])

out_dir = f"{out_root}/iemocap/tri_seed{cfg['seed']}"
Path(out_dir).mkdir(parents=True, exist_ok=True)

print(train_csv)
print(val_csv)
print(test_csv)
print(out_dir)
PY > /tmp/chado_paths.txt

TRAIN_CSV=$(sed -n '1p' /tmp/chado_paths.txt)
VAL_CSV=$(sed -n '2p' /tmp/chado_paths.txt)
TEST_CSV=$(sed -n '3p' /tmp/chado_paths.txt)
OUT_DIR=$(sed -n '4p' /tmp/chado_paths.txt)

PORT=29711

torchrun --nproc_per_node="${NPROC}" --master_port="${PORT}" scripts/train_chado_ddp.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv   "${VAL_CSV}" \
  --test_csv  "${TEST_CSV}" \
  --out_dir   "${OUT_DIR}" \
  --config    configs/iemocap_tri.yaml \
  --save_reports
