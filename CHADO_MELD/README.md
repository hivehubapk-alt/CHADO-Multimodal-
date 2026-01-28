
##Repository Skeleton

CHADO/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── requirements.txt
│
├── configs/
│   ├── base/
│   │   ├── meld.yaml
│   │   ├── training.yaml
│   │   └── model.yaml
│   │
│   ├── baseline/
│   │   └── baseline_trimodal_meld.yaml
│   │
│   ├── chado/
│   │   ├── chado_full.yaml
│   │   └── chado_eval_only.yaml
│   │
│   ├── ablations/
│   │   ├── wo_causal.yaml
│   │   ├── wo_hyperbolic.yaml
│   │   ├── wo_transport.yaml
│   │   ├── wo_causal_hyperbolic.yaml
│   │   ├── wo_causal_transport.yaml
│   │   └── wo_hyperbolic_transport.yaml
│   │
│   └── sensitivity/
│       ├── mad.yaml
│       ├── ot.yaml
│       ├── hyperbolic.yaml
│       └── curvature.yaml
│
├── src/
│   ├── data/
│   │   ├── meld_dataset.py
│   │   ├── collate.py
│   │   └── label_maps.py
│   │
│   ├── models/
│   │   ├── baseline_trimodal.py
│   │   ├── chado_trimodal.py
│   │   └── components/
│   │       ├── causal.py
│   │       ├── hyperbolic.py
│   │       ├── transport.py
│   │       ├── mad.py
│   │       └── refinement.py
│   │
│   ├── losses/
│   │   ├── ot_loss.py
│   │   └── mad_loss.py
│   │
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_chado.py
│   │   └── ddp_utils.py
│   │
│   └── eval/
│       ├── metrics.py
│       ├── evaluate_ckpt.py
│       └── correlation.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_chado.py
│   ├── run_component_ablations.py
│   ├── run_sensitivity.py
│   ├── run_correlation.py
│   │
│   ├── plot/
│   │   ├── plot_violin.py
│   │   ├── plot_ambiguity_curves.py
│   │   ├── plot_mad_vs_error.py
│   │   ├── plot_poincare.py
│   │   └── plot_confusion.py
│   │
│   └── utils/
│       ├── io.py
│       └── reproducibility.py
│
├── runs/               # ignored by git
├── figures/            # ignored by git
└── data/               # ignored by git
# CHADO: Causal–Hyperbolic Ambiguity Disentanglement with Optimal Transport

This repository contains the official implementation of **CHADO**, a
multimodal emotion recognition framework that explicitly models **causality,
geometric structure, and ambiguity** to improve robustness under domain shift
and human disagreement.

CHADO integrates:
- **Causal representation learning**
- **Hyperbolic geometry for hierarchical emotion structure**
- **Optimal Transport alignment across modalities**
- **MAD-based ambiguity modeling**
- **Refinement via causal interventions**

The codebase is designed for **reproducible ICML-grade experimentation** and
supports **component ablations**, **hyperparameter sensitivity analysis**, and
**rich qualitative/quantitative visualizations**.

---

## 📌 Supported Datasets
- **MELD** (primary)
- CMU-MOSEI (supported by configuration)
- IEMOCAP (supported by configuration)

---

## 🧠 Model Variants

- **Baseline Tri-Modal** (Text + Audio + Video)
- **CHADO (Full)**
- **Component Ablations**
  - w/o Causal
  - w/o Hyperbolic
  - w/o Transport (OT)
  - w/o Causal + Hyperbolic
  - w/o Causal + Transport
  - w/o Hyperbolic + Transport

---

## 📊 Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 (weighted)
- Calibration (ECE, Brier)
- Ambiguity (MAD, entropy)
- Correlation with human disagreement (Pearson / Spearman)

---

## 🚀 Quick Start

### 1. Create Environment
```bash
conda env create -f environment.yml
conda activate chado

#Train Baseline
python scripts/run_baseline.py --config configs/baseline/baseline_trimodal_meld.yaml
#Train CHADO
python scripts/run_chado.py --config configs/chado/chado_full.yaml
#Run Ablations
python scripts/run_component_ablations.py --config configs/chado/chado_full.yaml
#Plot Results
python scripts/plot/plot_violin.py
Run commands
Run one training (CHADO full) on GPUs 5–9
    conda activate chado_meld
    cd /home//CHADO_MELD

    export CUDA_VISIBLE_DEVICES=5,6,7,8,9
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    torchrun --nproc_per_node=5 --master_port=29500 \
      src/train/train_chado.py --config configs/chado_meld.yaml
Evaluate baseline (adds Precision/Recall)
  python scripts/eval_meld_ckpt_metrics.py \
    --config configs/chado_meld.yaml \
    --ckpt runs/baseline_trimodal_meld_best.pt \
    --split test \
    --batch_size 8 \
    --num_workers 6
Run all 6 “WITHOUT component” ablations (professor list)
  conda activate chado_meld
  cd /home//CHADO_MELD

  export CUDA_VISIBLE_DEVICES=5,6,7,8,9
  export OMP_NUM_THREADS=1
  export TOKENIZERS_PARALLELISM=false
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  ./scripts/run_meld_component_removal_ablations_ddp.sh
