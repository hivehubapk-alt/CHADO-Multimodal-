import os
import kagglehub
from pathlib import Path

PROJECT_ROOT = Path("/home/tahirahmad/CHADO_CMU_MOSEI")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version (KaggleHub decides its cache location)
path = kagglehub.dataset_download("samarwarsi/cmu-mosei")
print("KaggleHub cache path:", path)

# Link it into your project so everything is consistent/reproducible
target = RAW_DIR / "cmu-mosei"
if target.exists() or target.is_symlink():
    target.unlink()

os.symlink(path, target)
print("Linked dataset into project at:", target)
