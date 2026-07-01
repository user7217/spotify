"""Per-track analysis pipeline: composes every extraction stage into the
unified track document (spec Output shape).

Each stage is independent, cached via ProcessingState, and failure-isolated:
one stage failing doesn't lose the others. Deterministic given the same
audio bytes + extractor versions (seeds fixed; versions recorded).
"""

import hashlib
import logging
import time
from collections.abc import Callable

import numpy as np

from .analysis.harmony import extract_harmony
from .analysis.rhythm import extract_rhythm
from .analysis.structure import extract_structure
from .audio.loader import load_audio_file
from .features.lowlevel import extract_low_level

log = logging.getLogger("audiolens.extractor.pipeline")

PIPELINE_VERSION = "pipeline-v2"


def _hash_audio(y: np.ndarray) -> str:
    return hashlib.sha256(y.tobytes()).hexdigest()


def analyze_track(
    path: str,
    sr: int = 22050,
    embed_fn: Callable[[np.ndarray, int], dict] | None = None,
    classify_fn: Callable[[np.ndarray, int, dict], dict] | None = None,
) -> dict:
    """Run all DSP stages on one audio file.

    embed_fn / classify_fn are injected by the embedder service (keeps DSP
    workers free of torch/tensorflow deps). Returns the unified document
    minus metadata (assembler joins catalog metadata in).
    """
    np.random.seed(0)  # reproducibility for any stochastic steps
    t0 = time.time()
    y, sr = load_audio_file(path, target_sr=sr)
    if y is None or len(y) < sr:  # < 1s or unreadable -> corrupted
        raise CorruptedAudioError(path)

    doc: dict = {
        "_meta": {
            "audio_hash": _hash_audio(y),
            "sample_rate": sr,
            "duration_s": round(len(y) / sr, 3),
            "pipeline_version": PIPELINE_VERSION,
        }
    }
    stages: dict[str, Callable[[], dict]] = {
        "audio_features": lambda: extract_low_level(y, sr),
        "rhythm": lambda: extract_rhythm(y, sr),
        "structure": lambda: extract_structure(y, sr),
    }
    for name, fn in stages.items():
        t = time.time()
        try:
            doc[name] = fn()
            log.info("stage=%s path=%s ok %.1fs", name, path, time.time() - t)
        except Exception as e:  # noqa: BLE001
            log.exception("stage=%s path=%s FAILED", name, path)
            doc[name] = {"error": str(e)}

    # harmony wants beat times from rhythm
    beats = np.array([b["t"] for b in doc.get("rhythm", {}).get("detail", {}).get("beats", [])])
    try:
        doc["harmony"] = extract_harmony(y, sr, beat_times=beats if len(beats) else None)
    except Exception as e:  # noqa: BLE001
        log.exception("stage=harmony path=%s FAILED", path)
        doc["harmony"] = {"error": str(e)}

    if embed_fn is not None:
        try:
            doc["embeddings"] = embed_fn(y, sr)
        except Exception as e:  # noqa: BLE001
            log.exception("stage=embeddings path=%s FAILED", path)
            doc["embeddings"] = {"error": str(e)}
    if classify_fn is not None:
        try:
            heads = classify_fn(y, sr, doc.get("embeddings", {}))
            doc["genre"] = heads.get("genre", {})
            doc["mood"] = heads.get("mood", {})
            doc["instruments"] = heads.get("instruments", {})
            doc["vocals"] = heads.get("vocals", {})
            doc["production"] = heads.get("production", {})
        except Exception as e:  # noqa: BLE001
            log.exception("stage=classify path=%s FAILED", path)
            doc["genre"] = {"error": str(e)}

    doc["_meta"]["elapsed_s"] = round(time.time() - t0, 2)
    return doc


class CorruptedAudioError(Exception):
    pass
