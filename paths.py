"""Project folders. Keep every module pointing at the same data files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ZONES_CSV = DATA / "zones.csv"
KABADIWALAS_CSV = DATA / "kabadiwalas.csv"
AUDIO_DIR = DATA / "audio_samples"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "segregation_rf.joblib"
METRICS_PATH = MODELS_DIR / "segregation_metrics.json"
