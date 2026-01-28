import os
import re
import shutil
import argparse
from pathlib import Path
from datetime import date

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".cache", "wandb", "mlruns",
    "outputs", "runs", "checkpoints", "data/media", "data/processed", "data/raw",
    "CMU_MOSEI_KAGGLE", "third_party/.cache"
}

EXCLUDE_EXTS = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".h5", ".hdf5",
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".wav", ".flac", ".mp3", ".m4a",
    ".png", ".jpg", ".jpeg", ".gif",
    ".tar.gz", ".zip", ".7z"
}

INCLUDE_TOPLEVEL = {
    "src", "scripts", "configs", "data/manifests", "docs"
}

ALWAYS_INCLUDE_FILES = {
    "README.md", "LICENSE", "CITATION.cff", ".gitignore",
    "requirements.txt", "environment.yml", "pyproject.toml", "Makefile"
}

DEFAULT_GITIGNORE = r"""# Artifacts
outputs/
runs/
checkpoints/
results/
logs/
*.log
*.pt
*.pth
*.ckpt
*.npy
*.npz

# Data
data/media/
data/processed/
data/raw/
*.tar.gz
*.zip
*.hdf5
*.h5
*.mp4
*.mkv
*.avi
*.mov
*.webm
*.wav
*.flac
*.mp3
*.m4a
*.jpg
*.jpeg
*.png

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/
.env

# IDE
.vscode/
.idea/
"""

README_TEMPLATE = """# CHADO — CMU-MOSEI (emo6) + Cross-Environment + MOSEI→MELD

This repository contains a **reproducible CHADO pipeline** for multimodal emotion recognition on **CMU-MOSEI (emo6)**.
It includes:
- Modalities: **T**, **TA**, **TV**, **TAV**
- CHADO components: **MAD ambiguity**, **Optimal Transport (OT)**, **Hyperbolic regularization**, **Causal disentanglement**, **Self-correction**
- Experiments: standard training, **component ablations**, **LOEO cross-environment**, **MOSEI→MELD cross-dataset transfer**, **hyperparameter sensitivity (λ₁,λ₂,λ₃)**
- Metrics: Acc, WF1, Precision/Recall, ECE/Brier, MAD, OT distances

## Environment

### Conda (recommended)
```bash
conda create -n chado_mm python=3.10 -y
conda activate chado_mm
pip install -r requirements.txt