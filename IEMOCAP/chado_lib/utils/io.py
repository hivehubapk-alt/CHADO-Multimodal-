from pathlib import Path
import yaml
import os

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def expand_env(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(x) for x in obj]
    return obj

def mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
