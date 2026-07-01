"""Full harmonic analysis (spec): key/mode + confidence, modulations,
chord progression, chord transition matrix, harmonic complexity.

Krumhansl-Schmuckler on CQT chroma for key; template matching over 24
maj/min triads for per-beat chords; windowed K-S for modulation detection.
A deep key model (e.g. Essentia keyCNN) overrides K-S when available —
hook in `deep_key_fn`.
"""

import logging
from collections.abc import Callable

import numpy as np

import librosa

log = logging.getLogger("audiolens.extractor.harmony")

EXTRACTOR_VERSION = "harmony-v2"

# Krumhansl-Kessler profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 24 triad templates (12 major + 12 minor)
def _chord_templates() -> tuple[np.ndarray, list[str]]:
    templates, names = [], []
    base = np.zeros(12)
    for root in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            t = base.copy()
            for iv in intervals:
                t[(root + iv) % 12] = 1.0
            templates.append(t / np.linalg.norm(t))
            names.append(f"{PITCH_NAMES[root]}{'' if quality == 'maj' else 'm'}")
    return np.array(templates), names


_TEMPLATES, _CHORD_NAMES = _chord_templates()


def _ks_key(chroma_mean: np.ndarray) -> tuple[int, int, float]:
    """Returns (key 0-11, mode 1=major/0=minor, confidence)."""
    scores = []
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        scores.append((np.corrcoef(rolled, _MAJOR)[0, 1], shift, 1))
        scores.append((np.corrcoef(rolled, _MINOR)[0, 1], shift, 0))
    scores.sort(reverse=True)
    (s1, key, mode), (s2, *_ ) = scores[0], scores[1]
    conf = float(np.clip((s1 - s2) / (abs(s1) + 1e-12) + s1, 0, 1))
    return key, mode, conf


def extract_harmony(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray | None = None,
    deep_key_fn: Callable[[np.ndarray, int], tuple[int, int, float]] | None = None,
) -> dict:
    y_h = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)

    # ---- global key ---------------------------------------------------------
    key, mode, conf = _ks_key(chroma.mean(axis=1))
    method = "krumhansl-schmuckler"
    if deep_key_fn is not None:
        try:
            key, mode, conf = deep_key_fn(y, sr)
            method = "deep"
        except Exception as e:  # noqa: BLE001
            log.warning("deep key model failed, K-S fallback: %s", e)

    # ---- modulations: windowed K-S (10s windows, 5s hop) ---------------------
    fps = chroma.shape[1] / (len(y) / sr)
    win, hop_w = int(10 * fps), int(5 * fps)
    modulations, prev = [], (key, mode)
    for i in range(0, max(chroma.shape[1] - win, 1), max(hop_w, 1)):
        k, m, c = _ks_key(chroma[:, i : i + win].mean(axis=1))
        if (k, m) != prev and c > 0.5:
            modulations.append({
                "t": round(i / fps, 2),
                "from": f"{PITCH_NAMES[prev[0]]}{'maj' if prev[1] else 'min'}",
                "to": f"{PITCH_NAMES[k]}{'maj' if m else 'min'}",
                "confidence": round(c, 3),
            })
            prev = (k, m)

    # ---- per-beat chords ------------------------------------------------------
    if beat_times is None or len(beat_times) < 2:
        beat_frames = np.arange(0, chroma.shape[1], max(int(fps * 0.5), 1))
    else:
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)
    sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-12)

    sims = _TEMPLATES @ sync  # (24, n_beats)
    chord_idx = sims.argmax(axis=0)
    chord_conf = sims.max(axis=0)

    progression = []
    for bi, (ci, cc) in enumerate(zip(chord_idx, chord_conf)):
        t = float(beat_times[bi]) if beat_times is not None and bi < len(beat_times) else bi * 0.5
        if not progression or progression[-1]["chord"] != _CHORD_NAMES[ci]:
            progression.append({
                "t": round(t, 3),
                "chord": _CHORD_NAMES[ci],
                "confidence": round(float(cc), 3),
            })

    # ---- transition matrix + complexity ---------------------------------------
    trans = np.zeros((24, 24))
    for a, b in zip(chord_idx[:-1], chord_idx[1:]):
        trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_p = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums > 0)

    used = np.unique(chord_idx)
    # complexity: chord vocabulary size + transition entropy, normalized
    probs = trans[trans > 0] / trans.sum() if trans.sum() else np.array([1.0])
    trans_entropy = float(-(probs * np.log2(probs)).sum() / np.log2(max(len(probs), 2)))
    complexity = float(np.clip(0.5 * len(used) / 24 + 0.5 * trans_entropy, 0, 1))

    return {
        "key": int(key),
        "mode": int(mode),
        "key_name": f"{PITCH_NAMES[key]} {'major' if mode else 'minor'}",
        "key_confidence": float(conf),
        "harmonic_complexity": complexity,
        "detail": {
            "method": method,
            "modulations": modulations,
            "chord_progression": progression,
            "chord_names": _CHORD_NAMES,
            "chord_transition_matrix": np.round(trans_p, 4).tolist(),
            "chord_vocabulary_size": int(len(used)),
        },
        "extractor_version": EXTRACTOR_VERSION,
    }
