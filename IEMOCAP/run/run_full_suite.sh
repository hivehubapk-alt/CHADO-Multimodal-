#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import yaml, os
from pathlib import Path

paths = yaml.safe_load(open("configs/paths.yaml"))
tri   = yaml.safe_load(open("configs/iemocap_tri.yaml"))
out_root = os.path.expandvars(paths["outputs_root"])
out_dir  = f"{out_root}/iemocap/tri_seed{tri['seed']}"

Path(out_dir).mkdir(parents=True, exist_ok=True)
print(out_dir)
PY > /tmp/chado_outdir.txt

OUT_DIR=$(cat /tmp/chado_outdir.txt)

python scripts/eval_confusion_and_metrics.py \
  --split_csv "$(python - <<'PY'
import yaml, os
paths=yaml.safe_load(open("configs/paths.yaml"))
print(os.path.expandvars(paths["splits"]["test_csv"]))
PY
)" \
  --ckpt "${OUT_DIR}/best.pt" \
  --ablation tri \
  --batch_size 6 --amp \
  --device cuda:0 \
  --out_dir "${OUT_DIR}/suite"

python scripts/eval_ambiguity.py \
  --test_csv "$(python - <<'PY'
import yaml, os
paths=yaml.safe_load(open("configs/paths.yaml"))
print(os.path.expandvars(paths["splits"]["test_csv"]))
PY
)" \
  --ckpt "${OUT_DIR}/best.pt" \
  --ablation tri \
  --batch_size 6 --amp \
  --device cuda:0 \
  --low_thr 0.3 --high_thr 0.7 \
  --out_dir "${OUT_DIR}/suite"

python scripts/eval_cost.py \
  --ckpt "${OUT_DIR}/best.pt" \
  --device cuda:0 \
  --batch_size 6 --amp

python scripts/eval_qualitative_samples.py \
  --test_csv "$(python - <<'PY'
import yaml, os
paths=yaml.safe_load(open("configs/paths.yaml"))
print(os.path.expandvars(paths["splits"]["test_csv"]))
PY
)" \
  --ckpt "${OUT_DIR}/best.pt" \
  --out_dir "${OUT_DIR}/suite/qualitative_samples" \
  --device cuda:0 --amp \
  --k_true 5 --k_wrong 5 \
  --seed 42
