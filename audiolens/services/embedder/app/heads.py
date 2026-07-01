"""Prediction heads: genre, mood, instruments, vocals, production.

Primary path: Essentia-TensorFlow pretrained heads on discogs-effnet /
musicnn embeddings (downloaded by `make models`):
  genre        genre_discogs400 (400 subgenre labels, multi-label sigmoid)
  mood         mood_happy/sad/aggressive/relaxed + emomusic (valence/arousal)
  instruments  mtg_jamendo_instrument (40 classes)
  vocals       gender model + voice/instrumental
  danceability danceability model

Fallback path: calibrated DSP heuristics (always available) so every head
returns a result with `provenance` marking which path produced it.
Production characteristics are DSP-only (no good public model exists).
"""

import logging
import os

import numpy as np

import librosa

log = logging.getLogger("audiolens.embedder.heads")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
HEADS_VERSION = "heads-v1"

_GENRE_GRAPH = "genre_discogs400-discogs-effnet-1.pb"
_INSTR_GRAPH = "mtg_jamendo_instrument-discogs-effnet-1.pb"


def _essentia_available() -> bool:
    try:
        import essentia.standard  # noqa: F401
        return True
    except ImportError:
        return False


def _load_labels(graph: str) -> list[str]:
    import json
    meta = os.path.join(MODEL_DIR, graph.replace(".pb", ".json"))
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get("classes", [])
    return []


# --------------------------------------------------------------------------- genre

def predict_genre(y: np.ndarray, sr: int, effnet_emb: np.ndarray | None = None) -> dict:
    if _essentia_available() and os.path.exists(os.path.join(MODEL_DIR, _GENRE_GRAPH)):
        from essentia.standard import TensorflowPredict2D
        labels = _load_labels(_GENRE_GRAPH)
        head = TensorflowPredict2D(
            graphFilename=os.path.join(MODEL_DIR, _GENRE_GRAPH),
            input="serving_default_model_Placeholder", output="PartitionedCall:0",
        )
        probs = np.array(head(effnet_emb[None, :].astype(np.float32))).mean(axis=0)
        top = np.argsort(probs)[::-1][:10]
        genres = [
            {"name": labels[i] if i < len(labels) else f"class_{i}",
             "confidence": round(float(probs[i]), 4)}
            for i in top if probs[i] > 0.05
        ]
        return {
            "genres": genres,
            "primary": genres[0]["name"] if genres else None,
            "secondary": genres[1]["name"] if len(genres) > 1 else None,
            "multi_label": True,
            "provenance": "discogs-effnet-genre400",
        }
    # fallback: coarse DSP heuristic buckets
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    guesses = []
    if flat > 0.3 and cent > 3500:
        guesses.append(("Electronic", 0.4))
    if tempo > 160 and zcr > 0.1:
        guesses.append(("Metal/Punk", 0.3))
    if tempo < 90 and cent < 2000:
        guesses.append(("Ballad/Acoustic", 0.3))
    if not guesses:
        guesses.append(("Pop/Rock", 0.25))
    return {
        "genres": [{"name": g, "confidence": c} for g, c in guesses],
        "primary": guesses[0][0], "secondary": None,
        "multi_label": True, "provenance": "dsp-heuristic",
    }


# --------------------------------------------------------------------------- mood

def predict_mood(y: np.ndarray, sr: int, musicnn_emb: np.ndarray | None = None,
                 key_mode: int | None = None) -> dict:
    scores: dict[str, float] = {}
    provenance = "dsp-heuristic"

    if _essentia_available():
        from essentia.standard import TensorflowPredict2D
        for mood, graph in (
            ("happy", "mood_happy-musicnn-msd-2.pb"),
            ("sad", "mood_sad-musicnn-msd-2.pb"),
            ("aggressive", "mood_aggressive-musicnn-msd-2.pb"),
            ("relaxed", "mood_relaxed-musicnn-msd-2.pb"),
        ):
            p = os.path.join(MODEL_DIR, graph)
            if os.path.exists(p) and musicnn_emb is not None:
                head = TensorflowPredict2D(graphFilename=p, output="model/Softmax")
                scores[mood] = float(np.array(head(musicnn_emb[None, :].astype(np.float32))).mean(axis=0)[0])
                provenance = "essentia-musicnn-mood"

    # DSP base signals (always computed; fill anything models didn't)
    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(librosa.feature.tempo(onset_envelope=onset, sr=sr)[0])
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    energy = float(np.clip(np.mean(rms) * 12, 0, 1))
    arousal = float(np.clip(0.5 * energy + 0.3 * min(tempo / 180, 1) + 0.2 * min(cent / 5000, 1), 0, 1))
    brightness = min(cent / 4500, 1)
    mode_bonus = 0.15 if key_mode == 1 else (-0.15 if key_mode == 0 else 0)
    valence = float(np.clip(0.45 + 0.3 * (brightness - 0.5) + mode_bonus
                            + (scores.get("happy", 0.5) - scores.get("sad", 0.5)) * 0.3, 0, 1))
    beat_reg = 0.0
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
    if len(beats) > 4:
        ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
        beat_reg = float(np.clip(1 - np.std(ibi) / (np.mean(ibi) + 1e-9), 0, 1))
    dance = float(np.clip(0.4 * beat_reg + 0.3 * energy
                          + 0.3 * np.exp(-((tempo - 120) ** 2) / 2400), 0, 1))

    out = {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "dominance": round(float(np.clip(0.5 * arousal + 0.5 * scores.get("aggressive", energy), 0, 1)), 3),
        "danceability": round(dance, 3),
        "energy": round(energy, 3),
        "positivity": round(valence, 3),
        "aggression": round(scores.get("aggressive", float(np.clip(energy * (1 - valence) * 1.5, 0, 1))), 3),
        "melancholy": round(scores.get("sad", float(np.clip((1 - valence) * (1 - arousal) * 1.8, 0, 1))), 3),
        "tension": round(float(np.clip(arousal * (1 - valence) * 1.6, 0, 1)), 3),
        "relaxation": round(scores.get("relaxed", float(np.clip((1 - arousal) * valence * 1.8, 0, 1))), 3),
        "provenance": provenance,
    }
    return out


# --------------------------------------------------------------------------- instruments

_INSTRUMENT_KEYS = ["vocals", "guitar", "bass", "piano", "synth", "strings",
                    "brass", "drums", "percussion", "orchestra"]

def predict_instruments(y: np.ndarray, sr: int, effnet_emb: np.ndarray | None = None) -> dict:
    if (_essentia_available() and effnet_emb is not None
            and os.path.exists(os.path.join(MODEL_DIR, _INSTR_GRAPH))):
        from essentia.standard import TensorflowPredict2D
        labels = _load_labels(_INSTR_GRAPH)
        head = TensorflowPredict2D(
            graphFilename=os.path.join(MODEL_DIR, _INSTR_GRAPH), output="model/Sigmoid"
        )
        probs = np.array(head(effnet_emb[None, :].astype(np.float32))).mean(axis=0)
        raw = {labels[i] if i < len(labels) else f"c{i}": float(p) for i, p in enumerate(probs)}
        mapping = {  # jamendo label -> spec label
            "voice": "vocals", "singer": "vocals", "guitar": "guitar",
            "electricguitar": "guitar", "acousticguitar": "guitar", "bass": "bass",
            "piano": "piano", "synthesizer": "synth", "strings": "strings",
            "violin": "strings", "cello": "strings", "brass": "brass",
            "trumpet": "brass", "drums": "drums", "drummachine": "drums",
            "percussion": "percussion", "orchestra": "orchestra",
        }
        out = {k: 0.0 for k in _INSTRUMENT_KEYS}
        for lab, p in raw.items():
            tgt = mapping.get(lab)
            if tgt:
                out[tgt] = max(out[tgt], round(p, 4))
        out["provenance"] = "mtg-jamendo-instrument"
        return out

    # DSP fallback: band-energy + harmonicity heuristics (coarse)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(S[m].mean() / (S.mean() + 1e-12))
    y_h, y_p = librosa.effects.hpss(y)
    perc_ratio = float(np.mean(y_p**2) / (np.mean(y**2) + 1e-12))
    vocal = _vocal_band_score(y, sr)
    return {
        "vocals": round(vocal, 3),
        "guitar": round(np.clip(band(200, 2000) * 0.5, 0, 1), 3),
        "bass": round(np.clip(band(40, 250) * 0.8, 0, 1), 3),
        "piano": round(np.clip(band(250, 4000) * 0.3, 0, 1), 3),
        "synth": round(float(np.clip(np.mean(librosa.feature.spectral_flatness(y=y)) * 2.5, 0, 1)), 3),
        "strings": 0.1, "brass": 0.05,
        "drums": round(np.clip(perc_ratio * 1.6, 0, 1), 3),
        "percussion": round(np.clip(perc_ratio * 1.2, 0, 1), 3),
        "orchestra": 0.05,
        "provenance": "dsp-heuristic",
    }


def _vocal_band_score(y: np.ndarray, sr: int) -> float:
    """Energy + modulation in 200-4000 Hz vocal band of the harmonic part."""
    y_h = librosa.effects.harmonic(y)
    S = np.abs(librosa.stft(y_h))
    freqs = librosa.fft_frequencies(sr=sr)
    m = (freqs >= 200) & (freqs <= 4000)
    ratio = float(S[m].sum() / (S.sum() + 1e-12))
    band_env = S[m].mean(axis=0)
    mod = float(np.std(band_env) / (np.mean(band_env) + 1e-12))
    return float(np.clip(ratio * 0.8 + min(mod, 1) * 0.4, 0, 1))


# --------------------------------------------------------------------------- vocals

def predict_vocals(y: np.ndarray, sr: int, musicnn_emb: np.ndarray | None = None) -> dict:
    density = _vocal_band_score(y, sr)
    gender = None
    provenance = "dsp-heuristic"
    if _essentia_available() and musicnn_emb is not None:
        gpath = os.path.join(MODEL_DIR, "gender-musicnn-msd-2.pb")
        if os.path.exists(gpath):
            from essentia.standard import TensorflowPredict2D
            head = TensorflowPredict2D(graphFilename=gpath, output="model/Softmax")
            p_female = float(np.array(head(musicnn_emb[None, :].astype(np.float32))).mean(axis=0)[0])
            gender = "female" if p_female > 0.6 else "male" if p_female < 0.4 else "mixed"
            provenance = "essentia-musicnn"

    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)
    voiced = f0[np.isfinite(f0) & (f0 > 80) & (f0 < 1000)]
    if gender is None and len(voiced) > 50:
        med = float(np.median(voiced))
        gender = "female" if med > 220 else "male" if med < 165 else "mixed"
    vrange = (
        {"low_hz": round(float(np.percentile(voiced, 5)), 1),
         "high_hz": round(float(np.percentile(voiced, 95)), 1)}
        if len(voiced) > 50 else None
    )
    # singing vs spoken: pitch stability within voiced runs (singing sustains)
    if len(voiced) > 50:
        stab = float(np.mean(np.abs(np.diff(librosa.hz_to_midi(voiced))) < 0.5))
    else:
        stab = 0.0
    return {
        "gender": gender or "unknown",
        "vocal_density": round(density, 3),
        "vocal_range": vrange,
        "singing_vs_spoken": round(stab, 3),
        "vocal_energy": round(float(np.clip(density * 1.2, 0, 1)), 3),
        "provenance": provenance,
    }


# --------------------------------------------------------------------------- production

def predict_production(y: np.ndarray, sr: int, release_year: int | None = None) -> dict:
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    peak = float(np.max(np.abs(y)) + 1e-12)
    crest_db = float(20 * np.log10(peak / (np.sqrt(np.mean(y**2)) + 1e-12)))

    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    acoustic = float(np.clip(1 - flat * 2.2 - min(np.mean(freqs[np.argmax(S, axis=0)]) / 8000, 1) * 0.3, 0, 1))

    # compression: low crest factor + low RMS variance => heavily compressed
    compression = float(np.clip(1 - (crest_db - 6) / 14, 0, 1))
    # distortion: spectral irregularity + clipping density
    clip_frac = float(np.mean(np.abs(y) > 0.985))
    odd_even = _odd_even_harmonic_ratio(y, sr)
    distortion = float(np.clip(clip_frac * 40 + max(odd_even - 1, 0) * 0.3, 0, 1))
    # reverb: decay tail after onsets (energy decay slope)
    reverb = _reverb_score(y, sr)
    # hf rolloff as analog/digital + era proxy: pre-90s masters roll off >16k
    hf = float(S[freqs > 16000].mean() / (S.mean() + 1e-12)) if (freqs > 16000).any() else 0.0
    analog = float(np.clip(1 - hf * 4 - flat, 0, 1))

    era = None
    if release_year:
        era = f"{(release_year // 10) * 10}s"
    elif hf < 0.02 and compression < 0.4:
        era = "pre-1990s (inferred)"
    elif compression > 0.75:
        era = "loudness-war 2000s+ (inferred)"

    return {
        "acoustic_vs_electronic": round(acoustic, 3),  # 1=acoustic
        "analog_vs_digital": round(analog, 3),        # 1=analog
        "compression_level": round(compression, 3),
        "distortion_amount": round(distortion, 3),
        "reverb_intensity": round(reverb, 3),
        "stereo_width": None,  # requires stereo source; loader is mono — set by worker if stereo
        "production_era": era,
        "crest_factor_db": round(crest_db, 2),
        "provenance": "dsp",
    }


def _odd_even_harmonic_ratio(y: np.ndarray, sr: int) -> float:
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) < 10:
        return 1.0
    f = float(np.median(f0))
    S = np.abs(librosa.stft(y)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    def h(n):
        i = np.argmin(np.abs(freqs - n * f))
        return float(S[i])
    odd = sum(h(n) for n in (3, 5, 7)) + 1e-12
    even = sum(h(n) for n in (2, 4, 6)) + 1e-12
    return odd / even


def _reverb_score(y: np.ndarray, sr: int) -> float:
    onset = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    if len(onset) < 4:
        return 0.3
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    hop = 256
    decays = []
    for o in onset[:200]:
        i = o // hop
        seg = rms[i : i + int(0.4 * sr / hop)]
        if len(seg) > 8 and seg[0] > 0:
            decays.append(float(seg[-1] / (seg[0] + 1e-12)))
    return float(np.clip(np.median(decays) * 1.8, 0, 1)) if decays else 0.3


# --------------------------------------------------------------------------- entry

def classify_all(y: np.ndarray, sr: int, embeddings: dict,
                 key_mode: int | None = None, release_year: int | None = None) -> dict:
    effnet = np.array(embeddings.get("discogs-effnet", {}).get("vector", []))
    musicnn = np.array(embeddings.get("musicnn", {}).get("vector", []))
    effnet = effnet if effnet.size else None
    musicnn = musicnn if musicnn.size else None
    return {
        "genre": predict_genre(y, sr, effnet),
        "mood": predict_mood(y, sr, musicnn, key_mode),
        "instruments": predict_instruments(y, sr, effnet),
        "vocals": predict_vocals(y, sr, musicnn),
        "production": predict_production(y, sr, release_year),
        "heads_version": HEADS_VERSION,
    }
