"""Audio loading: decode MP3/FLAC/WAV/AAC to mono float32 at target sample rate."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tempfile

import librosa
import numpy as np

SUPPORTED = {"mp3", "flac", "wav", "aac", "m4a", "ogg", "opus", "webm"}


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_audio_file(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decode an audio file on disk -> (mono float32 array, sample_rate).

    Path-based counterpart to ``load_audio`` (which takes raw bytes). Format is
    inferred from the file extension; anything librosa/audioread can't open
    directly is routed through ffmpeg. Used by the on-disk batch pipeline
    (run_pipeline.py / analyze_track), where audio comes from yt-dlp downloads.
    """
    fmt = os.path.splitext(path)[1].lower().lstrip(".")
    with open(path, "rb") as f:
        data = f.read()
    # ffmpeg-decode the container formats librosa can't read from BytesIO well
    if fmt in {"aac", "m4a", "opus", "webm"}:
        data = _ffmpeg_to_wav(data, fmt)
    y, sr = librosa.load(io.BytesIO(data), sr=target_sr, mono=True)
    return y.astype(np.float32), sr


def load_audio(data: bytes, fmt: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decode audio bytes -> (mono float32 array, sample_rate).

    librosa/soundfile handles wav/flac/ogg/mp3 directly.
    aac/m4a goes through ffmpeg.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt not in SUPPORTED:
        raise ValueError(f"unsupported format: {fmt}")

    if fmt in {"aac", "m4a"}:
        data = _ffmpeg_to_wav(data, fmt)

    y, sr = librosa.load(io.BytesIO(data), sr=target_sr, mono=True)
    if len(y) < target_sr:  # < 1 second
        raise ValueError("audio too short for analysis (< 1s)")
    return y.astype(np.float32), sr


def _ffmpeg_to_wav(data: bytes, fmt: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as src:
        src.write(data)
        src.flush()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src.name, "-f", "wav", "-ac", "1", "pipe:1"],
            capture_output=True,
            check=True,
            timeout=120,
        )
        return result.stdout


def probe_duration_ms(y: np.ndarray, sr: int) -> int:
    return int(len(y) / sr * 1000)
