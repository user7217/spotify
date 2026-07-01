"""End-to-end flow test. Requires docker-compose stack running.

pytest tests/integration -m integration --api http://localhost:8000
"""
import io
import time

import httpx
import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.integration
API = "http://localhost:8000"


def make_wav_bytes(seconds=8, bpm=120) -> bytes:
    sr = 22050
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    interval = int(60 / bpm * sr)
    click = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.03, int(0.03 * sr))).astype(np.float32)
    for s in range(0, len(y) - len(click), interval):
        y[s:s + len(click)] += click
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def test_full_pipeline():
    wav = make_wav_bytes()
    r = httpx.post(f"{API}/v1/tracks", files={"file": ("test_click.wav", wav)}, timeout=60)
    assert r.status_code == 201
    body = r.json()
    track_id = body["track"]["id"]
    job_id = body["job"]["id"]

    # poll job to completion
    for _ in range(60):
        j = httpx.get(f"{API}/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(2)
    assert j["status"] == "done", j

    feats = httpx.get(f"{API}/v1/audio-features/{track_id}").json()
    assert 0 <= feats["danceability"] <= 1
    assert any(abs(feats["tempo"] - t) < 8 for t in (60, 120, 240))

    analysis = httpx.get(f"{API}/v1/audio-analysis/{track_id}").json()
    assert len(analysis["beats"]) > 5
    assert len(analysis["segments"][0]["pitches"]) == 12
