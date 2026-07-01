"""Extractor tests using synthetic audio (no fixtures needed).

Synthetic signals with known properties let us assert directional
correctness: a 120 BPM click track must yield tempo ~120, a pure
sine must read more 'acoustic' than white noise, etc.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "extractor"))

from app.analysis.extractor import AnalysisExtractor  # noqa: E402
from app.features.extractor import FeatureExtractor  # noqa: E402

SR = 22050


# ── synthetic audio generators ────────────────────────────────────────────────

def click_track(bpm: float, seconds: float = 15.0) -> np.ndarray:
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    interval = int(60 / bpm * SR)
    click = np.exp(-np.linspace(0, 30, 800)).astype(np.float32) * np.sin(
        2 * np.pi * 1000 * np.linspace(0, 800 / SR, 800)
    ).astype(np.float32)
    for start in range(0, len(y) - 800, interval):
        y[start : start + 800] += click
    return y


def sine(freq: float = 440.0, seconds: float = 15.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def white_noise(seconds: float = 15.0) -> np.ndarray:
    return (np.random.RandomState(0).randn(int(seconds * SR)) * 0.3).astype(np.float32)


def c_major_chord(seconds: float = 15.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), dtype=np.float32)
    y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0))
    return (0.3 * y / 3).astype(np.float32)


# ── feature extractor ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fx():
    return FeatureExtractor(sample_rate=SR)


class TestFeatures:
    def test_tempo_click_track(self, fx):
        r = fx.extract(click_track(120), SR)
        # accept octave errors (60/240) but expect 120 family
        assert any(abs(r.tempo - t) < 6 for t in (60, 120, 240)), r.tempo

    def test_ranges(self, fx):
        r = fx.extract(white_noise(), SR)
        for name in ("danceability", "energy", "valence", "speechiness",
                      "acousticness", "instrumentalness", "liveness"):
            v = getattr(r, name)
            assert 0 <= v <= 1, f"{name}={v}"
        assert 0 <= r.key <= 11 or r.key == -1
        assert r.mode in (0, 1)
        assert r.time_signature in (3, 4, 5, 7)

    def test_key_detection_c_major(self, fx):
        r = fx.extract(c_major_chord(), SR)
        assert r.key == 0  # C
        assert r.mode == 1  # major

    def test_acousticness_ordering(self, fx):
        pure = fx.extract(sine(), SR)
        noise = fx.extract(white_noise(), SR)
        assert pure.acousticness > noise.acousticness

    def test_loudness_ordering(self, fx):
        # energy is gain-invariant by design (Spotify behavior);
        # loudness must reflect the actual signal level
        loud = fx.extract(white_noise(), SR)
        quiet = fx.extract(white_noise() * 0.05, SR)
        assert loud.loudness > quiet.loudness

    def test_loudness_is_negative_db(self, fx):
        r = fx.extract(sine() * 0.1, SR)
        assert r.loudness < 0

    def test_rejects_nothing_but_returns_silence_features(self, fx):
        # near-silence shouldn't crash
        r = fx.extract(np.zeros(SR * 5, dtype=np.float32) + 1e-6, SR)
        assert r is not None


# ── analysis extractor ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ax():
    return AnalysisExtractor(sample_rate=SR)


class TestAnalysis:
    def test_beats_count_click_track(self, ax):
        seconds, bpm = 15, 120
        r = ax.extract(click_track(bpm, seconds), SR)
        expected = seconds * bpm / 60
        assert abs(len(r.beats) - expected) <= expected * 0.35

    def test_segment_vector_shapes(self, ax):
        r = ax.extract(click_track(100), SR)
        assert len(r.segments) > 0
        for seg in r.segments[:10]:
            assert len(seg["pitches"]) == 12
            assert len(seg["timbre"]) == 12
            assert 0 <= max(seg["pitches"]) <= 1

    def test_intervals_are_contiguous(self, ax):
        r = ax.extract(click_track(120), SR)
        for i in range(len(r.beats) - 1):
            end = r.beats[i]["start"] + r.beats[i]["duration"]
            assert abs(end - r.beats[i + 1]["start"]) < 0.01

    def test_track_block(self, ax):
        r = ax.extract(sine(seconds=12), SR)
        assert r.track["duration"] == pytest.approx(12, abs=0.5)
        assert r.track["sample_rate"] == SR

    def test_rhythm_block(self, ax):
        r = ax.extract(click_track(120), SR)
        assert r.rhythm["beat_regularity"] > 0.8  # clicks are perfectly regular
        assert 0 <= r.rhythm["syncopation"] <= 1

    def test_harmony_block(self, ax):
        r = ax.extract(c_major_chord(), SR)
        assert "chords" in r.harmony
        assert r.harmony["global_key"] == 0  # C
