"""Train RandomForest on WAV folders. Hold out 20% and save a confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from m1_audio.capture import load_wav
from m1_audio.features import extract_features
from m1_audio.synth_samples import generate_samples
from paths import AUDIO_DIR, METRICS_PATH, MODEL_PATH, MODELS_DIR


def _collect() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for folder in sorted(AUDIO_DIR.iterdir()):
        if not folder.is_dir():
            continue
        wavs = list(folder.glob("*.wav"))
        for wav in wavs:
            y = load_wav(wav)
            xs.append(extract_features(y))
            ys.append(folder.name)
    if not xs:
        raise FileNotFoundError(f"No WAV files under {AUDIO_DIR}")
    return np.vstack(xs), np.array(ys)


def train_and_save(per_class: int = 32, seed: int = 42) -> dict:
    generate_samples(per_class=per_class)
    X, y = _collect()
    labels = sorted(set(y.tolist()))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    cm = confusion_matrix(y_test, pred, labels=labels).tolist()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(clf, MODEL_PATH)
    metrics = {
        "accuracy": acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "labels": labels,
        "confusion_matrix": cm,
        "note": "Trained on whatever WAVs are in data/audio_samples. sim_*.wav files are SYNTHETIC.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print("Saved", MODEL_PATH)
    print("Accuracy (held-out):", round(m["accuracy"], 3))
    print("Labels:", m["labels"])
