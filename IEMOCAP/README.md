# CHADO: Causal–Hyperbolic–OT Ambiguity Decomposition for Multimodal Emotion Recognition

This repository provides a research-grade training + evaluation pipeline for CHADO on IEMOCAP (4-class: neu/hap/ang/sad),
including DDP training, component-removal ablations, hyperparameter sensitivity, ambiguity-stratified analysis, andready
plots/tables.

## 1. Environment

### Option A: Conda
```bash
conda env create -f environment.yml
conda activate chado_iemocap
pip install -r requirements.txt
2. Data
    This repo does NOT include IEMOCAP. You must prepare CSV splits containing:
    transcript (text)
    wav_path, start, end
    avi_path, start, end
    label_4 in {neu,hap,ang,sad}
    Expected columns in split CSV:
    transcript,wav_path,avi_path,start,end,label_4
3. Configure paths (portable)
    Copy:
    cp configs/paths.template.yaml configs/paths.yaml
    Edit configs/paths.yaml to point to your local project root and split CSVs.
    configs/paths.yaml is ignored by git.
4. Reproduce the tri-modal run (DDP)
    bash run/run_train_iemocap_tri.sh
5. Component-removal ablations (DDP on 5 GPUs)
Runs:
    w/o Causal
    w/o Hyperbolic
    w/o OT
    w/o (Causal + Hyperbolic)
    w/o (Causal + OT)
    w/o (Hyperbolic + OT)
    bash run/run_component_removal_ablations.sh
6. Hyperparameter sensitivity (λ₁, λ₂, λ₃, curvature)
    bash run/run_sensitivity_iemocap.sh
7. Evaluation suite (plots + CSV)
    bash run/run_full_suite.sh