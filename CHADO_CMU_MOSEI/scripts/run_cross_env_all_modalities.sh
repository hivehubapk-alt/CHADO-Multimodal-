# #!/bin/bash
# set -e

# # ---- Conda init for non-interactive shells ----
# source /home/tahirahmad/miniconda3/etc/profile.d/conda.sh
# conda activate chado_mm


# export OMP_NUM_THREADS=1
# export PYTHONWARNINGS=ignore
# export TORCH_CPP_LOG_LEVEL=ERROR
# export TORCH_DISTRIBUTED_DEBUG=OFF

# CFG=src/configs/chado_mosei_emo6.yaml
# GPU=5

# cd /home/tahirahmad/CHADO_CMU_MOSEI

# for FOLD in 0 1 2 3 4; do
#   TRAIN=data/cross_env/fold_${FOLD}/data/manifests/mosei_train.jsonl
#   VAL=data/cross_env/fold_${FOLD}/data/manifests/mosei_val.jsonl
#   TEST=data/cross_env/fold_${FOLD}/data/manifests/mosei_test.jsonl

#   for ABL in T TA TV TAV; do
#     OUT=outputs/cross_env/fold_${FOLD}/${ABL}
#     mkdir -p $OUT

#     echo "============================================================"
#     echo "LOEO | Fold=${FOLD} | Modality=${ABL}"

#     CUDA_VISIBLE_DEVICES=$GPU \
#     python src/train/train.py \
#       --cfg $CFG \
#       --ablation $ABL \
#       --override experiment.ddp=false \
#       --override experiment.project_root=/home/tahirahmad/CHADO_CMU_MOSEI \
#       --override experiment.out_root=$OUT \
#       --override data.train_manifest=$TRAIN \
#       --override data.val_manifest=$VAL \
#       --override data.test_manifest=$TEST
#   done
# done
#!/usr/bin/env bash
set -euo pipefail

cd /home/tahirahmad/CHADO_CMU_MOSEI

export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF

CFG=src/configs/chado_mosei_emo6.yaml

# Single-GPU sequential training (stable). You can parallelize later.
GPU=${1:-5}
export CUDA_VISIBLE_DEVICES=${GPU}

for FOLD in 0 1 2 3 4; do
  TRAIN=data/cross_env/fold_${FOLD}/data/manifests/mosei_train.jsonl
  VAL=data/cross_env/fold_${FOLD}/data/manifests/mosei_val.jsonl
  TEST=data/cross_env/fold_${FOLD}/data/manifests/mosei_test.jsonl

  for ABL in T TA TV TAV; do
    OUT=outputs/cross_env/fold_${FOLD}/${ABL}
    mkdir -p "${OUT}"

    echo "============================================================"
    echo "LOEO fold=${FOLD} ablation=${ABL} GPU=${GPU} -> OUT=${OUT}"

    python src/train/train.py \
      --cfg "${CFG}" \
      --ablation "${ABL}" \
      --train_manifest "${TRAIN}" \
      --val_manifest "${VAL}" \
      --test_manifest "${TEST}" \
      --out_root "${OUT}" \
      --ddp false
  done
done
