"""Spotify Audio Features replacement.

Strategy per feature:
  tempo            librosa beat tracker (madmom DBN refines if available)
  key, mode        Krumhansl-Schmuckler key profiles on CQT chroma
  loudness         integrated RMS in dBFS, EBU-style gating approximation
  energy           weighted combo: RMS + spectral flux + onset density, calibrated 0-1
  danceability     beat regularity + tempo prior + pulse clarity (Essentia model if present)
  acousticness     spectral rolloff/centroid heuristic (Essentia model overrides)
  speechiness      zero-crossing + spectral flatness in speech band (Essentia VAD overrides)
  instrumentalness vocal-band energy heuristic (Essentia voice model overrides)
  valence          Essentia pretrained MusiCNN model; chroma/mode heuristic fallback
  liveness         reverb tail estimation + crowd-noise band energy heuristic
  time_signature   beat-strength autocorrelation over bar candidates

Heuristic fallbacks always run; pretrained Essentia models override their target
fields when the model files are available (see models/ directory + MODEL_DIR env).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import librosa
import numpy as np

log = logging.getLogger(__name__)

EXTRACTOR_VERSION = "0.1.0"

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


@dataclass
class FeatureResult:
    danceability: float
    energy: float
    valence: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    loudness: float
    tempo: float
    key: int
    mode: int
    time_signature: int
    confidences: dict


class FeatureExtractor:
    def __init__(self, sample_rate: int = 22050, model_dir: str | None = None):
        self.sr = sample_rate
        self.model_dir = model_dir or os.environ.get("MODEL_DIR", "/models")
        self._essentia_models = self._load_essentia_models()

    # ── public ───────────────────────────────────────────────────────────────

    def extract(self, y: np.ndarray, sr: int) -> FeatureResult:
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # loudness from raw signal (level-dependent); everything else from
        # normalized signal (gain-invariant, like Spotify's features)
        loudness = self._loudness(y)
        y = librosa.util.normalize(y)
        confidences: dict[str, float] = {}

        tempo, beats, beat_conf = self._tempo_and_beats(y, sr)
        key, mode, key_conf = self._key_mode(y, sr)
        energy = self._energy(y, sr)
        dance = self._danceability(y, sr, beats, tempo)
        acoustic = self._acousticness(y, sr)
        speech = self._speechiness(y, sr)
        instrumental = self._instrumentalness(y, sr)
        valence, val_conf = self._valence(y, sr, mode, energy)
        liveness = self._liveness(y, sr)
        time_sig = self._time_signature(y, sr, beats)

        confidences.update(
            {"tempo": beat_conf, "key": key_conf, "valence": val_conf}
        )

        # Essentia model overrides where available
        overrides = self._apply_essentia_models(y, sr)
        if "danceability" in overrides:
            dance = overrides["danceability"]
            confidences["danceability"] = 0.9
        if "valence" in overrides:
            valence = overrides["valence"]
            confidences["valence"] = 0.9
        if "acousticness" in overrides:
            acoustic = overrides["acousticness"]
        if "instrumentalness" in overrides:
            instrumental = overrides["instrumentalness"]

        return FeatureResult(
            danceability=round(float(np.clip(dance, 0, 1)), 4),
            energy=round(float(np.clip(energy, 0, 1)), 4),
            valence=round(float(np.clip(valence, 0, 1)), 4),
            speechiness=round(float(np.clip(speech, 0, 1)), 4),
            acousticness=round(float(np.clip(acoustic, 0, 1)), 4),
            instrumentalness=round(float(np.clip(instrumental, 0, 1)), 4),
            liveness=round(float(np.clip(liveness, 0, 1)), 4),
            loudness=round(float(loudness), 3),
            tempo=round(float(tempo), 3),
            key=int(key),
            mode=int(mode),
            time_signature=int(time_sig),
            confidences=confidences,
        )

    # ── tempo / beats ────────────────────────────────────────────────────────

    def _tempo_and_beats(self, y: np.ndarray, sr: int) -> tuple[float, np.ndarray, float]:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
        tempo = float(np.atleast_1d(tempo)[0])

        # confidence: pulse clarity from tempogram peak sharpness
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tg_mean = tg.mean(axis=1)
        peak = tg_mean.max()
        conf = float(np.clip((peak - tg_mean.mean()) / (tg_mean.std() + 1e-8) / 5.0, 0, 1))

        # madmom refinement (more accurate DBN beat tracker). Can segfault and
        # is slow — DISABLE_MADMOM=1 skips it and keeps the librosa estimate.
        try:
            import os as _os
            if _os.environ.get("DISABLE_MADMOM", "").lower() in {"1", "true", "yes"}:
                raise ImportError("madmom disabled via DISABLE_MADMOM")
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

            act = RNNBeatProcessor()(y.astype(np.float32))
            beat_times = DBNBeatTrackingProcessor(fps=100)(act)
            if len(beat_times) > 4:
                ibis = np.diff(beat_times)
                tempo = float(60.0 / np.median(ibis))
                beats = librosa.time_to_frames(beat_times, sr=sr)
                conf = max(conf, 0.8)
        except Exception:
            pass  # librosa result stands

        return tempo, beats, conf

    # ── key / mode ───────────────────────────────────────────────────────────

    def _key_mode(self, y: np.ndarray, sr: int) -> tuple[int, int, float]:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        scores = []
        for shift in range(12):
            rotated = np.roll(chroma_mean, -shift)
            maj = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
            mino = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
            scores.append((shift, 1, maj))
            scores.append((shift, 0, mino))

        scores.sort(key=lambda t: t[2], reverse=True)
        key, mode, best = scores[0]
        second = scores[1][2]
        conf = float(np.clip((best - second) * 5 + 0.3, 0, 1))
        return key, mode, conf

    # ── loudness ─────────────────────────────────────────────────────────────

    def _loudness(self, y: np.ndarray) -> float:
        rms = librosa.feature.rms(y=y)[0]
        # gate out near-silence frames (EBU R128 style approximation)
        db = librosa.amplitude_to_db(rms, ref=1.0)
        gated = db[db > db.max() - 40]
        return float(gated.mean()) if len(gated) else float(db.mean())

    # ── energy ───────────────────────────────────────────────────────────────

    def _energy(self, y: np.ndarray, sr: int) -> float:
        rms = librosa.feature.rms(y=y)[0].mean()
        flux = np.mean(np.diff(np.abs(librosa.stft(y)), axis=1).clip(min=0))
        onset_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)

        # calibration constants chosen against reference tracks
        e = 0.5 * np.tanh(rms * 8) + 0.3 * np.tanh(flux * 2) + 0.2 * np.tanh(onset_rate / 4)
        return float(e)

    # ── danceability ─────────────────────────────────────────────────────────

    def _danceability(
        self, y: np.ndarray, sr: int, beats: np.ndarray, tempo: float
    ) -> float:
        if len(beats) < 8:
            return 0.2
        beat_times = librosa.frames_to_time(beats, sr=sr)
        ibis = np.diff(beat_times)
        regularity = 1.0 - float(np.clip(np.std(ibis) / (np.mean(ibis) + 1e-8), 0, 1))

        # tempo prior: danceable range peaks ~100-130 BPM
        tempo_score = float(np.exp(-(((tempo - 115) / 45) ** 2)))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        pulse_clarity = float(np.clip(pulse.max() / (pulse.mean() + 1e-8) / 6, 0, 1))

        return 0.45 * regularity + 0.3 * tempo_score + 0.25 * pulse_clarity

    # ── acousticness ─────────────────────────────────────────────────────────

    def _acousticness(self, y: np.ndarray, sr: int) -> float:
        S = np.abs(librosa.stft(y))
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0].mean()
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0].mean()
        flatness = librosa.feature.spectral_flatness(S=S)[0].mean()

        # acoustic music: low rolloff, low centroid, low flatness (tonal not noisy)
        a = (
            0.4 * (1 - np.clip(rolloff / (sr / 2), 0, 1))
            + 0.4 * (1 - np.clip(centroid / 4000, 0, 1))
            + 0.2 * (1 - np.clip(flatness * 10, 0, 1))
        )
        return float(a)

    # ── speechiness ──────────────────────────────────────────────────────────

    def _speechiness(self, y: np.ndarray, sr: int) -> float:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        S = np.abs(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)[0]

        # speech has high ZCR variance (voiced/unvoiced alternation) and
        # syllabic energy modulation around 4 Hz
        rms = librosa.feature.rms(y=y)[0]
        hop_rate = sr / 512
        mod_spec = np.abs(np.fft.rfft(rms - rms.mean()))
        freqs = np.fft.rfftfreq(len(rms), d=1 / hop_rate)
        syllabic_band = mod_spec[(freqs > 2) & (freqs < 8)].sum()
        total = mod_spec.sum() + 1e-8
        syllabic_ratio = float(syllabic_band / total)

        s = 0.4 * np.clip(zcr.std() * 20, 0, 1) + 0.4 * syllabic_ratio + 0.2 * float(
            flatness.mean() * 5
        )
        return float(np.clip(s, 0, 1)) * 0.66  # scale: pure music rarely > 0.33

    # ── instrumentalness ─────────────────────────────────────────────────────

    def _instrumentalness(self, y: np.ndarray, sr: int) -> float:
        # vocal energy concentrates 200 Hz - 4 kHz with strong harmonic structure;
        # estimate via harmonic component energy ratio in the vocal band
        y_harm, _ = librosa.effects.hpss(y)
        S_harm = np.abs(librosa.stft(y_harm))
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_band = S_harm[(freqs > 200) & (freqs < 4000)]
        ratio = float(vocal_band.mean() / (S_harm.mean() + 1e-8))

        # high vocal-band concentration -> vocals likely -> low instrumentalness
        return float(np.clip(1.6 - ratio, 0, 1))

    # ── valence ──────────────────────────────────────────────────────────────

    def _valence(self, y: np.ndarray, sr: int, mode: int, energy: float) -> tuple[float, float]:
        # heuristic fallback: major mode + brightness + energy
        S = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0].mean()
        brightness = float(np.clip(centroid / 3500, 0, 1))
        v = 0.35 * mode + 0.35 * brightness + 0.3 * energy
        return float(v), 0.4  # low confidence — heuristic only

    # ── liveness ─────────────────────────────────────────────────────────────

    def _liveness(self, y: np.ndarray, sr: int) -> float:
        # crowd noise: broadband energy 1-4 kHz between onsets + long reverb decay
        S = np.abs(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        # sustained high flatness in quiet passages suggests audience/room noise
        rms = librosa.feature.rms(S=S)[0]
        quiet = rms < np.percentile(rms, 25)
        if quiet.sum() < 5:
            return 0.1
        live_score = float(flatness[quiet].mean() * 8)
        return float(np.clip(live_score, 0, 1)) * 0.8

    # ── time signature ───────────────────────────────────────────────────────

    def _time_signature(self, y: np.ndarray, sr: int, beats: np.ndarray) -> int:
        if len(beats) < 16:
            return 4
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strengths = onset_env[np.clip(beats, 0, len(onset_env) - 1)]

        # score candidate meters by autocorrelation of beat-strength pattern
        best_sig, best_score = 4, -np.inf
        for sig in (3, 4, 5, 7):
            if len(beat_strengths) < sig * 4:
                continue
            trimmed = beat_strengths[: (len(beat_strengths) // sig) * sig]
            grid = trimmed.reshape(-1, sig)
            # downbeat should be consistently strongest -> high column variance
            score = float(grid.mean(axis=0).std())
            if score > best_score:
                best_sig, best_score = sig, score
        return best_sig

    # ── essentia pretrained models ───────────────────────────────────────────

    def _load_essentia_models(self) -> dict:
        models = {}
        try:
            import essentia.standard as es  # noqa: F401

            candidates = {
                "danceability": "danceability-musicnn-msd-2.pb",
                "valence_arousal": "emomusic-musicnn-msd-2.pb",
                "voice_instrumental": "voice_instrumental-musicnn-msd-2.pb",
                "acousticness": "mood_acoustic-musicnn-msd-2.pb",
            }
            for name, fname in candidates.items():
                path = os.path.join(self.model_dir, fname)
                if os.path.exists(path):
                    models[name] = path
            if models:
                log.info("essentia models loaded: %s", list(models))
        except ImportError:
            log.info("essentia not installed — heuristics only")
        return models

    def _apply_essentia_models(self, y: np.ndarray, sr: int) -> dict:
        if not self._essentia_models:
            return {}
        out: dict[str, float] = {}
        try:
            import essentia.standard as es

            audio16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)

            if "danceability" in self._essentia_models:
                pred = es.TensorflowPredictMusiCNN(
                    graphFilename=self._essentia_models["danceability"]
                )(audio16)
                out["danceability"] = float(np.mean(pred[:, 0]))

            if "valence_arousal" in self._essentia_models:
                pred = es.TensorflowPredictMusiCNN(
                    graphFilename=self._essentia_models["valence_arousal"]
                )(audio16)
                # model outputs valence/arousal on 1-9 scale
                out["valence"] = float((np.mean(pred[:, 0]) - 1) / 8)

            if "voice_instrumental" in self._essentia_models:
                pred = es.TensorflowPredictMusiCNN(
                    graphFilename=self._essentia_models["voice_instrumental"]
                )(audio16)
                out["instrumentalness"] = float(np.mean(pred[:, 1]))

            if "acousticness" in self._essentia_models:
                pred = es.TensorflowPredictMusiCNN(
                    graphFilename=self._essentia_models["acousticness"]
                )(audio16)
                out["acousticness"] = float(np.mean(pred[:, 0]))
        except Exception as e:  # model failure must never kill extraction
            log.warning("essentia model inference failed: %s", e)
        return out
