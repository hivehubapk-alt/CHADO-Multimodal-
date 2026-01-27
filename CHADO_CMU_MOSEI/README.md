conda create -n chado_mm python=3.10 -y
conda activate chado_mm
pip install -r requirements.txt
export CHADO_DATA_ROOT=/path/to/your/features_and_media
cd /home/tahirahmad/CHADO_CMU_MOSEI

# 1) Make sure outputs + data are not tracked accidentally
git init

# 2) Create docs folder
mkdir -p docs data/manifests

# 3) Move only manifests into data/manifests if they are elsewhere
# (skip if already in place)
# mkdir -p data/manifests
# mv data/manifests/*.jsonl data/manifests/ 2>/dev/null || true

# 4) Ensure scripts are executable where needed (optional)
find scripts -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# 5) Sanity: show largest files to avoid committing them
du -ah . | sort -hr | head -n 30

# CHADO (CMU-MOSEI) — Cross-modal Ambiguity + Causal + Hyperbolic + OT

This repository contains the CHADO pipeline for multimodal emotion recognition on CMU-MOSEI (emo6 setting).
It supports:
- Modalities: **T**, **TA**, **TV**, **TAV**
- CHADO components: **Causal**, **Hyperbolic**, **Optimal Transport (OT)**, **MAD ambiguity**
- Experiments: baseline training, component ablations, LOEO cross-environment evaluation, MOSEI→MELD transfer
- Metrics: Acc, macro/weighted F1, Precision, Recall, ECE, Brier, MAD, OT distances

## 1. Environment

### Option A: pip
  ```bash
  conda create -n chado_mm python=3.10 -y
  conda activate chado_mm
  pip install -r requirements.txt
###Option B: conda
  conda env create -f environment.yml
  conda activate chado_mm


python scripts/train.py \
  --cfg configs/mosei/chado_mosei_emo6.yaml \
  --tag TAV_full \
  --seed 42 \
  --gpus 5
python scripts/run_ablations.py \
  --cfg configs/mosei/chado_mosei_emo6.yaml \
  --ablations configs/mosei/ablations.yaml \
  --out_csv outputs/tables/mosei_ablations.csv \
  --gpus 5
python scripts/cross_cultural/run_loeo_modalities.py \
  --cfg configs/mosei/chado_mosei_emo6.yaml \
  --out_csv outputs/cross_env/loeo_summary_modalities.csv \
  --gpus 5
CUDA_VISIBLE_DEVICES=5,6,7,8,9 torchrun --nproc_per_node=5 scripts/train.py --cfg ...
If you want single GPU reproducibility:
CUDA_VISIBLE_DEVICES=5 python scripts/train.py --cfg ...
Run these inside your server:
    cd /home/CHADO_CMU_MOSEI

    # 1) Make sure outputs + data are not tracked accidentally
    git init

    # 2) Create docs folder
    mkdir -p docs data/manifests

    # 3) Move only manifests into data/manifests if they are elsewhere
    # (skip if already in place)
    # mkdir -p data/manifests
    # mv data/manifests/*.jsonl data/manifests/ 2>/dev/null || true

    # 4) Ensure scripts are executable where needed (optional)
    find scripts -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

    # 5) Sanity: show largest files to avoid committing them
    du -ah . | sort -hr | head -n 30
##Reproduce Main Result (TAV)
    python scripts/train.py \
      --cfg configs/chado_mosei_emo6.yaml \
      --ablation TAV
4. Component Ablations (CHADO)
    python scripts/run_component_ablations.py \
      --cfg configs/chado_mosei_emo6.yaml \
      --out_csv outputs/tables/component_ablations.csv
5. Hyperparameter Sensitivity (λ1, λ2, λ3)
  python scripts/run_hparam_sensitivity.py \
  --cfg configs/chado_mosei_emo6.yaml \
  --out_csv outputs/hparam_sensitivity/lambda_sensitivity_summary.csv
Plot
  python scripts/analysis/plot_lambda_sensitivity.py \
    --csv outputs/hparam_sensitivity/lambda_sensitivity_summary.csv \
    --out_png outputs/hparam_sensitivity/lambda_sensitivity.png
6. Cross-cultural / Cross-environment (LOEO)
  python scripts/cross_env/run_loeo_all_modalities.py \
  --cfg configs/chado_mosei_emo6.yaml \
  --k 5 \
  --out_csv outputs/cross_env/loeo_summary_modalities.csv
7. Cross-dataset Culture Shift (MOSEI→MELD)
  python scripts/cross_cultural/eval_mosei_to_meld.py \
  --cfg configs/chado_mosei_emo6.yaml \
  --manifest data/manifests/cross/meld_test_emo6.jsonl \
  --ablation TAV \
  --ckpt outputs/checkpoints/TAV_best.pt
