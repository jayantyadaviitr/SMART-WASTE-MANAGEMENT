"""Turn a 1.5 s impact clip into numbers a Random Forest can use.

Features (from the sociothon brief):
- peak frequency (Hz)
- spectral centroid
- zero-crossing rate (ZCR)
- 13 MFCCs
"""

from __future__ import annotations

import numpy as np

SR = 22050
CLIP_SECONDS = 1.5
N_MFCC = 13
FEATURE_NAMES = (
    ["peak_hz", "spectral_centroid", "zcr"]
    + [f"mfcc_{i}" for i in range(N_MFCC)]
)


def _safe_librosa():
    import librosa

    return librosa


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a 1-D vector of length 16. Never raises on short/silent clips."""
    librosa = _safe_librosa()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 16:
        y = np.pad(y, (0, 16 - y.size))
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = y / peak

    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
    peak_hz = float(freqs[int(np.argmax(spec))]) if spec.size else 0.0

    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=min(1024, y.size), hop_length=256).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1).astype(np.float32)
    vec = np.concatenate([[peak_hz, centroid, zcr], mfcc_mean])
    return vec.astype(np.float32)


def fft_spectrum(y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
    return freqs, spec
