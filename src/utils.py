import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

METRICS_FILE = MODELS_DIR / "metrics.json"


def load_metrics():

    with open(METRICS_FILE) as f:
        return json.load(f)