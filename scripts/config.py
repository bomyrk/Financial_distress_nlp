import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def load_config(path: str = None) -> dict:
    if path is None:
        path = ROOT_DIR / "configs" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
