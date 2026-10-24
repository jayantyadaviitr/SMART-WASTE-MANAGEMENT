"""Make labelled demo WAV files when the team has not recorded real drops yet.

These clips follow the frequency *hypothesis* in the brief (wet thud, plastic
crinkle, ringing metal, brittle glass). They are labelled SIMULATED in the UI.
Replace them with real chute recordings when you can (30+ per class).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from m1_audio.features import CLIP_SECONDS, SR
from paths import AUDIO_DIR

CLASSES = {
    "wet": dict(f=(80, 450), decay=9.0, noise=0.04),
    "plastic": dict(f=(800, 2500), decay=6.0, noise=0.08, crinkle=True),
    "paper": dict(f=(900, 2200), decay=7.0, noise=0.07, crinkle=True),
    "metal": dict(f=(2500, 6000), decay=2.2, noise=0.02),
    "glass": dict(f=(3500, 8000), decay=4.5, noise=0.03),
}


def _one_clip(spec: dict, rng: np.random.Generator) -> np.ndarray:
    n = int(CLIP_SECONDS * SR)
    t = np.linspace(0, CLIP_SECONDS, n, endpoint=False)
    f0 = float(rng.uniform(*spec["f"]))
    env = np.exp(-spec["decay"] * t)
    y = env * np.sin(2 * np.pi * f0 * t)
    # A second partial so spectra are not a single spike.
    y += 0.35 * env * np.sin(2 * np.pi * (f0 * rng.uniform(1.4, 2.2)) * t)
    if spec.get("crinkle"):
        clicks = rng.normal(0, 1.0, n) * (rng.random(n) > 0.97)
        y += clicks * env
    y += rng.normal(0, spec["noise"], n)
    peak = np.max(np.abs(y)) + 1e-9
    return (0.9 * y / peak).astype(np.float32)


def generate_samples(per_class: int = 32, seed: int = 7) -> int:
    rng = np.random.default_rng(seed)
    written = 0
    for name, spec in CLASSES.items():
        folder = AUDIO_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*.wav"))
        need = max(0, per_class - len(existing))
        for i in range(need):
            path = folder / f"sim_{i:03d}.wav"
            sf.write(path, _one_clip(spec, rng), SR)
            written += 1
    return written
