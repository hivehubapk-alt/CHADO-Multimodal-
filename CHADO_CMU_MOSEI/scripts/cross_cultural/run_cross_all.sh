#!/usr/bin/env bash
set -e

cd /home/tahirahmad/CHADO_CMU_MOSEI

# 1) Build cross manifests from MELD + IEMOCAP splits
python scripts/cross_cultural/build_cross_manifests.py \
  --meld_csv /home/tahirahmad/CHADO_MELD/data/processed/meld_test.csv \
  --iemocap_csv /home/tahirahmad/Full_Final_IEMOCAP/processed/iemocap_full_6class.csv \
  --out_dir data/manifests/cross

# 2) Evaluate MOSEI checkpoints (T/TA/TV/TAV) cross-culturally on MELD/IEMOCAP (emo6-mapped)
CFG=src/configs/chado_mosei_emo6.yaml

for ABL in T TA TV TAV; do
  CKPT=outputs/checkpoints/${ABL}_best.pt

  echo "============================================================"
  echo "MOSEI(${ABL}) -> MELD (emo6 mapped)"
  python scripts/cross_cultural/eval_cross_cultural.py \
    --cfg $CFG --ckpt $CKPT --ablation $ABL \
    --manifest data/manifests/cross/meld_test_emo6.jsonl

  echo "------------------------------------------------------------"
  echo "MOSEI(${ABL}) -> IEMOCAP (emo6 mapped)"
  python scripts/cross_cultural/eval_cross_cultural.py \
    --cfg $CFG --ckpt $CKPT --ablation $ABL \
    --manifest data/manifests/cross/iemocap_test_emo6.jsonl
done
