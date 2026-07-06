"""Full rhythm analysis (spec): BPM, tempo curve, beats, downbeats, meter,
groove consistency, swing — with beat-by-beat detail.

librosa baseline; madmom DBN beat/downbeat trackers used when importable
(better on syncopated material), provenance recorded either way.
"""

import logging

import numpy as np

import librosa

log = logging.getLogger("audiolens.extractor.rhythm")

EXTRACTOR_VERSION = "rhythm-v2"

# madmom's native DBN trackers can SEGFAULT (not a catchable Python exception)
# and are ~5x slower than the librosa baseline. Set DISABLE_MADMOM=1 to force
# the librosa path — safer + much faster for large batch runs.
import os as _os

if _os.environ.get("DISABLE_MADMOM", "").lower() in {"1", "true", "yes"}:
    _MADMOM = False
else:
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )
        _MADMOM = True
    except ImportError:  # pragma: no cover
        _MADMOM = False


def _madmom_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (beat_times, downbeat_times) via madmom DBN trackers."""
    y16 = librosa.resample(y, orig_sr=sr, target_sr=44100)
    act = RNNBeatProcessor()(y16)
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    db_act = RNNDownBeatProcessor()(y16)
    db = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(db_act)
    downbeats = db[db[:, 1] == 1][:, 0] if len(db) else np.array([])
    return beats, downbeats


def _swing(beat_times: np.ndarray, onset_env: np.ndarray, sr: int, hop: int) -> float | None:
    """Swing ratio from energy at the 8th-note subdivision: 0.5=straight, ~0.66=triplet swing."""
    if len(beat_times) < 8:
        return None
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    ratios = []
    for a, b in zip(beat_times[:-1], beat_times[1:]):
        mask = (times >= a) & (times < b)
        seg = onset_env[mask]
        if len(seg) < 4:
            continue
        # offbeat position = argmax of onset energy in middle 60% of the beat
        lo, hi = int(len(seg) * 0.2), int(len(seg) * 0.8)
        if hi <= lo:
            continue
        pos = (lo + int(np.argmax(seg[lo:hi]))) / len(seg)
        ratios.append(pos)
    return float(np.median(ratios)) if ratios else None


def extract_rhythm(y: np.ndarray, sr: int, hop: int = 512) -> dict:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # global + dynamic tempo
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop)[0])
    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop, aggregate=None
    )
    curve_times = librosa.times_like(tempo_curve, sr=sr, hop_length=hop)

    # beats
    provenance = "librosa"
    downbeats = np.array([])
    if _MADMOM:
        try:
            beat_times, downbeats = _madmom_beats(y, sr)
            provenance = "madmom-dbn"
        except Exception as e:  # noqa: BLE001
            log.warning("madmom failed, librosa fallback: %s", e)
            _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    else:
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # per-beat strength: onset energy at each beat
    env_times = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    strengths = np.interp(beat_times, env_times, onset_env)
    strengths = (strengths / (strengths.max() + 1e-12)).tolist() if len(strengths) else []

    # confidence: pulse clarity (autocorr peak contrast of onset envelope)
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    tempo_confidence = float(np.clip(np.max(ac[1:]) / (ac[0] + 1e-12), 0, 1)) if len(ac) > 1 else 0.0

    # groove consistency: 1 - normalized IBI variance
    ibi = np.diff(beat_times)
    groove = float(np.clip(1 - np.std(ibi) / (np.mean(ibi) + 1e-12), 0, 1)) if len(ibi) > 1 else None

    # meter / time signature
    if len(downbeats) >= 2 and len(beat_times) > 0:
        per_bar = []
        for a, b in zip(downbeats[:-1], downbeats[1:]):
            per_bar.append(int(((beat_times >= a) & (beat_times < b)).sum()))
        ts = int(np.bincount(per_bar).argmax()) if per_bar else 4
        ts = ts if ts in (2, 3, 4, 5, 6, 7) else 4
    else:
        ts = _ts_from_autocorr(onset_env, beat_times, sr, hop)
        # synthesize downbeats every `ts` beats
        downbeats = beat_times[::ts] if len(beat_times) else np.array([])

    return {
        "bpm": tempo,
        "tempo_confidence": tempo_confidence,
        "meter": f"{ts}/4",
        "time_signature": ts,
        "groove_consistency": groove,
        "swing": _swing(beat_times, onset_env, sr, hop),
        "detail": {
            "provenance": provenance,
            "tempo_curve": [
                {"t": round(float(t), 3), "bpm": round(float(b), 2)}
                for t, b in zip(curve_times[::8], tempo_curve[::8])
            ],
            "beats": [
                {"t": round(float(t), 4), "strength": round(float(s), 4)}
                for t, s in zip(beat_times, strengths)
            ],
            "downbeats": [round(float(t), 4) for t in downbeats],
        },
        "extractor_version": EXTRACTOR_VERSION,
    }


def _ts_from_autocorr(onset_env: np.ndarray, beat_times: np.ndarray, sr: int, hop: int) -> int:
    """Meter via beat-synchronous onset autocorrelation over {3,4} candidates (existing approach)."""
    if len(beat_times) < 8:
        return 4
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    frames = frames[frames < len(onset_env)]
    beat_env = onset_env[frames]
    best, best_score = 4, -np.inf
    for cand in (3, 4):
        if len(beat_env) <= cand:
            continue
        score = np.corrcoef(beat_env[:-cand], beat_env[cand:])[0, 1]
        if np.isfinite(score) and score > best_score:
            best, best_score = cand, score
    return best
