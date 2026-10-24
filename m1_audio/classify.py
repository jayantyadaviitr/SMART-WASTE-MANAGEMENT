"""Load the trained forest and apply the 70% confidence gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from m1_audio.features import extract_features
from paths import METRICS_PATH, MODEL_PATH

CONFIDENCE_GATE = 0.70

# Map training folder names to demo labels / colours.
DISPLAY = {
    "wet": ("Wet / food", "#1a7f37", "Biodegradable"),
    "plastic": ("Dry recyclable (plastic)", "#1d4ed8", "Dry"),
    "paper": ("Dry recyclable (paper)", "#2563eb", "Dry"),
    "metal": ("Metal / can", "#a16207", "High-value recyclable"),
    "glass": ("Glass", "#b91c1c", "Recyclable / handle with care"),
}


def load_model(path: Path = MODEL_PATH):
    import joblib

    if not path.exists():
        return None
    return joblib.load(path)


def classify_wave(y: np.ndarray, model=None, gate: float = CONFIDENCE_GATE) -> dict:
    if model is None:
        model = load_model()
    if model is None:
        return {
            "label": "manual check",
            "raw_class": None,
            "confidence": 0.0,
            "manual": True,
            "reason": "No trained model yet. Run: python -m m1_audio.train",
            "color": "#d97706",
            "bin": "Manual-check slot",
        }
    vec = extract_features(y).reshape(1, -1)
    proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    idx = int(np.argmax(proba))
    raw = str(classes[idx])
    conf = float(proba[idx])
    if conf < gate:
        return {
            "label": "manual check",
            "raw_class": raw,
            "confidence": conf,
            "manual": True,
            "reason": f"Confidence {conf:.0%} is below {gate:.0%} (mixed or unclear drop).",
            "color": "#d97706",
            "bin": "Manual-check slot",
            "proba": {str(c): float(p) for c, p in zip(classes, proba)},
        }
    pretty, color, bin_name = DISPLAY.get(raw, (raw, "#334155", raw))
    return {
        "label": pretty,
        "raw_class": raw,
        "confidence": conf,
        "manual": False,
        "reason": "",
        "color": color,
        "bin": bin_name,
        "proba": {str(c): float(p) for c, p in zip(classes, proba)},
    }


def load_metrics() -> dict:
    import json

    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
