"""Microphone capture with an impact threshold, plus sample-clip fallback."""

from __future__ import annotations

import numpy as np

from m1_audio.features import CLIP_SECONDS, SR


def record_on_impact(
    seconds: float = CLIP_SECONDS,
    sr: int = SR,
    threshold: float = 0.08,
    listen_seconds: float = 4.0,
) -> tuple[np.ndarray | None, str]:
    """Listen for a knock, then record `seconds`.

    Returns (audio, status). status is 'ok', 'timeout', or 'mic_unavailable'.
    """
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - machine without PortAudio
        return None, f"mic_unavailable: {exc}"

    try:
        listen_n = int(listen_seconds * sr)
        buf = sd.rec(listen_n, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        wave = buf.reshape(-1)
        hits = np.where(np.abs(wave) >= threshold)[0]
        if hits.size == 0:
            return None, "timeout"
        start = int(hits[0])
        end = start + int(seconds * sr)
        clip = wave[start:end]
        if clip.size < int(seconds * sr):
            clip = np.pad(clip, (0, int(seconds * sr) - clip.size))
        return clip.astype(np.float32), "ok"
    except Exception as exc:
        return None, f"mic_unavailable: {exc}"


def load_wav(path, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    from pathlib import Path

    y, file_sr = sf.read(Path(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa

        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    n = int(CLIP_SECONDS * sr)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y[:n]
