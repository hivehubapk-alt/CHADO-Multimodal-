# Each folder have seperate readme file and all run commands are present in each folders. 
# CHADO-Multimodal-

Environment and Versions (README section)
Operating system

Linux (tested on Ubuntu server-class setup; GPU node environment)

Python + Conda

Python: 3.10.19

Conda/Miniconda: Miniconda3 (env path example: /home/tahirahmad/miniconda3/)

pip: 25.3

You can print and record your exact versions with:

which python
python -V
python -c "import sys; print(sys.executable)"
conda -V
pip -V

GPU / CUDA stack

GPU: NVIDIA RTX A6000

Driver: 535.216.01

CUDA runtime reported by nvidia-smi: 12.2

Verify:

nvidia-smi

Core ML libraries (recommended pinned versions)

Use these as your “official” reproducible setup:

PyTorch: 2.2.x (CUDA wheel cu121 recommended; works well even if driver shows CUDA 12.2)

TorchVision: matching PyTorch

Torchaudio: matching PyTorch

Transformers: 4.4x.x (stable for RoBERTa)

Tokenizers: auto-installed by Transformers

NumPy / Pandas / SciPy / scikit-learn / Matplotlib for metrics + plots

After environment setup, record exact installed versions via:

python - <<'PY'
import torch, numpy, pandas, sklearn, matplotlib, transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("transformers:", transformers.__version__)
PY

Create Conda Environment (clean, reproducible)
Option A: Create environment + install pip requirements
# 1) Create env
conda create -n chado_mm python=3.10.19 -y
conda activate chado_mm

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install PyTorch (CUDA wheel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4) Install remaining deps
pip install -r requirements.txt


Datasets links:
IEMOCAP: https://sail.usc.edu/iemocap/
CMU-MOSEI: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
MELD: wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz 
