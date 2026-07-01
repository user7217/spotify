"""Analyze every catalog track using iTunes 30-second preview clips.

This is the no-local-library path: for each unique track in catalog.db it
  1. searches the iTunes Search API (public, no key) for artist + title
  2. picks the best match (normalized title/artist + duration tolerance)
  3. downloads the 30s preview (m4a), decodes via ffmpeg
  4. runs the full DSP analysis stack (low-level, rhythm, harmony, structure)
     + classifier heads (genre/mood/instruments/vocals/production)
  5. writes results into the analysis tables of the same sqlite DB

Usage (run on your machine, not in a sandbox):
  pip install librosa soundfile pyloudnorm httpx
  # ffmpeg must be on PATH (brew install ffmpeg)
  python scripts/analyze_previews.py --sqlite data/catalog.db
  python scripts/analyze_previews.py --sqlite data/catalog.db --limit 500   # partial
  python scripts/analyze_previews.py --sqlite data/catalog.db --workers 6

Properties:
  resumable   — finished tracks are skipped (processing_state stage=preview_analysis)
  cached      — previews kept in --cache-dir, re-runs don't re-download
  rate-safe   — iTunes search throttled (~20 req/min burst-safe with backoff)
  parallel    — downloads on threads, DSP on a process pool
  honest      — every row records source=itunes_preview_30s + match score;
                structure/sections describe the CLIP, not the full song

Caveat: 30s previews are excerpts. BPM/key/spectral/mood are usually
representative; full-song structure and duration-dependent metrics are not.
"""

import argparse
import concurrent.futures as cf
import json
import logging
import pathlib
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

log = logging.getLogger("audiolens.previews")

ITUNES_URL = "https://itunes.apple.com/search"
SOURCE_TAG = "itunes_preview_30s"
STAGE = "preview_analysis"
DURATION_TOL_MS = 5000
MIN_INTERVAL_S = 3.05  # ~20 searches/min

_ANALYSIS_SCHEMA = """
CREATE TABLE IF NOT EXISTS low_level_features (
    track_id TEXT PRIMARY KEY,
    spectral_centroid REAL, spectral_bandwidth REAL, spectral_rolloff REAL,
    spectral_flatness REAL, spectral_flux REAL, spectral_entropy REAL,
    rms_energy REAL, dynamic_range_db REAL, loudness_lufs REAL,
    peak_amplitude REAL, crest_factor REAL,
    detail TEXT, extractor_version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS rhythm_analysis (
    track_id TEXT PRIMARY KEY,
    bpm REAL, tempo_confidence REAL, meter TEXT, time_signature INTEGER,
    groove_consistency REAL, swing REAL,
    detail TEXT, extractor_version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS harmony_analysis (
    track_id TEXT PRIMARY KEY,
    key INTEGER, mode INTEGER, key_name TEXT, key_confidence REAL,
    harmonic_complexity REAL,
    detail TEXT, extractor_version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS structure_analysis (
    track_id TEXT PRIMARY KEY,
    sections TEXT, section_repetition REAL, structural_similarity REAL,
    boundaries TEXT, extractor_version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS semantic_predictions (
    track_id TEXT NOT NULL, head TEXT NOT NULL,
    model_version TEXT NOT NULL, payload TEXT NOT NULL, source TEXT,
    UNIQUE(track_id, head, model_version));
CREATE TABLE IF NOT EXISTS preview_matches (
    track_id TEXT PRIMARY KEY,
    itunes_track_id INTEGER, matched_title TEXT, matched_artist TEXT,
    match_score REAL, preview_url TEXT, status TEXT);
"""

_PUNCT = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()


def _similarity(a: str, b: str) -> float:
    """Token Jaccard — cheap, good enough for title/artist agreement."""
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ITunesSearcher:
    def __init__(self):
        self._last = 0.0

    def search(self, artist: str, title: str) -> list[dict]:
        wait = MIN_INTERVAL_S - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        q = urllib.parse.urlencode({
            "term": f"{artist} {title}", "media": "music", "entity": "song", "limit": 8,
        })
        req = urllib.request.Request(
            f"{ITUNES_URL}?{q}", headers={"User-Agent": "AudioLens/0.1"}
        )
        for attempt in range(5):
            try:
                self._last = time.time()
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read()).get("results", [])
            except Exception as e:  # noqa: BLE001 — 403 rate-limit or transient
                back = 30 * (attempt + 1)
                log.warning("itunes search failed (%s), backoff %ds", e, back)
                time.sleep(back)
        return []


def best_match(results: list[dict], artist: str, title: str,
               duration_ms: int | None) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for r in results:
        if not r.get("previewUrl"):
            continue
        s = 0.6 * _similarity(title, r.get("trackName", "")) \
            + 0.4 * _similarity(artist, r.get("artistName", ""))
        if duration_ms and r.get("trackTimeMillis"):
            if abs(r["trackTimeMillis"] - duration_ms) <= DURATION_TOL_MS:
                s += 0.15
        if s > best_score:
            best, best_score = r, s
    return (best, best_score) if best_score >= 0.5 else (None, best_score)


def download_preview(url: str, dest: pathlib.Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AudioLens/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest.stat().st_size > 10_000
    except Exception as e:  # noqa: BLE001
        log.warning("download failed %s: %s", url, e)
        return False


def decode_to_wav(src: pathlib.Path, sr: int = 22050) -> pathlib.Path | None:
    out = src.with_suffix(".wav")
    if out.exists():
        return out
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr), str(out)],
        capture_output=True,
    )
    return out if r.returncode == 0 and out.exists() else None


def analyze_clip(args: tuple) -> tuple[str, dict | None, str | None]:
    """Process-pool entry: full DSP + heads on one decoded clip."""
    track_id, wav_path, key_mode_hint, release_year = args
    try:
        import numpy as np
        import librosa
        np.random.seed(0)
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        if len(y) < sr * 5:
            return track_id, None, "clip too short"
        from services.extractor.app.features.lowlevel import extract_low_level
        from services.extractor.app.analysis.rhythm import extract_rhythm
        from services.extractor.app.analysis.harmony import extract_harmony
        from services.extractor.app.analysis.structure import extract_structure
        from services.embedder.app.heads import classify_all

        doc = {"audio_features": extract_low_level(y, sr),
               "rhythm": extract_rhythm(y, sr)}
        beats = np.array([b["t"] for b in doc["rhythm"]["detail"]["beats"]])
        doc["harmony"] = extract_harmony(y, sr, beats if len(beats) else None)
        doc["structure"] = extract_structure(y, sr)
        doc["heads"] = classify_all(
            y, sr, {}, key_mode=doc["harmony"]["mode"], release_year=release_year
        )
        return track_id, doc, None
    except Exception as e:  # noqa: BLE001
        return track_id, None, f"{type(e).__name__}: {e}"


def write_results(db: sqlite3.Connection, track_id: str, doc: dict) -> None:
    ll = doc["audio_features"]
    db.execute(
        """INSERT OR REPLACE INTO low_level_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (track_id, ll["spectral_centroid"], ll["spectral_bandwidth"],
         ll["spectral_rolloff"], ll["spectral_flatness"], ll["spectral_flux"],
         ll["spectral_entropy"], ll["rms_energy"], ll["dynamic_range_db"],
         ll["loudness_lufs"], ll["peak_amplitude"], ll["crest_factor"],
         json.dumps(ll["detail"]), ll["extractor_version"], SOURCE_TAG),
    )
    r = doc["rhythm"]
    db.execute(
        "INSERT OR REPLACE INTO rhythm_analysis VALUES (?,?,?,?,?,?,?,?,?,?)",
        (track_id, r["bpm"], r["tempo_confidence"], r["meter"], r["time_signature"],
         r["groove_consistency"], r["swing"], json.dumps(r["detail"]),
         r["extractor_version"], SOURCE_TAG),
    )
    h = doc["harmony"]
    db.execute(
        "INSERT OR REPLACE INTO harmony_analysis VALUES (?,?,?,?,?,?,?,?,?)",
        (track_id, h["key"], h["mode"], h["key_name"], h["key_confidence"],
         h["harmonic_complexity"], json.dumps(h["detail"]),
         h["extractor_version"], SOURCE_TAG),
    )
    s = doc["structure"]
    db.execute(
        "INSERT OR REPLACE INTO structure_analysis VALUES (?,?,?,?,?,?,?)",
        (track_id, json.dumps(s["sections"]), s["section_repetition"],
         s["structural_similarity"], json.dumps(s["boundaries"]),
         s["extractor_version"], SOURCE_TAG),
    )
    heads = doc["heads"]
    for head in ("genre", "mood", "instruments", "vocals", "production"):
        db.execute(
            "INSERT OR REPLACE INTO semantic_predictions VALUES (?,?,?,?,?)",
            (track_id, head, heads.get("heads_version", "heads-v1"),
             json.dumps(heads[head]), SOURCE_TAG),
        )
    db.execute(
        """INSERT OR REPLACE INTO processing_state
           (catalog_track_id, stage, status) VALUES (?,?, 'done')""",
        (track_id, STAGE),
    )
    db.commit()


def mark_failed(db, track_id, err):
    db.execute(
        """INSERT OR REPLACE INTO processing_state
           (catalog_track_id, stage, status, error) VALUES (?,?,'failed',?)""",
        (track_id, STAGE, str(err)[:500]),
    )
    db.commit()


def pending_tracks(db, limit=None, order="play_count DESC"):
    q = f"""SELECT id, title, artists, duration_ms, release_year FROM catalog_tracks c
            WHERE NOT EXISTS (SELECT 1 FROM processing_state p
                              WHERE p.catalog_track_id = c.id AND p.stage = '{STAGE}'
                                AND p.status IN ('done','failed'))
            ORDER BY {order}"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return db.execute(q).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--cache-dir", default=None, help="preview cache (default: tmp)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4, help="DSP processes")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db = sqlite3.connect(a.sqlite)
    db.executescript(_ANALYSIS_SCHEMA)
    if a.retry_failed:
        db.execute(f"DELETE FROM processing_state WHERE stage='{STAGE}' AND status='failed'")
        db.commit()

    cache = pathlib.Path(a.cache_dir or tempfile.gettempdir()) / "audiolens_previews"
    cache.mkdir(parents=True, exist_ok=True)
    rows = pending_tracks(db, a.limit)
    log.info("%d tracks pending analysis", len(rows))
    if not rows:
        return 0

    searcher = ITunesSearcher()
    done = failed = no_match = 0
    t0 = time.time()

    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {}
        for cid, title, artists, dur, year in rows:
            artist = json.loads(artists)[0]
            # 1. match (cached)
            cached = db.execute(
                "SELECT preview_url, status FROM preview_matches WHERE track_id=?", (cid,)
            ).fetchone()
            if cached and cached[1] == "no_match":
                no_match += 1
                mark_failed(db, cid, "no itunes match")
                continue
            if cached and cached[0]:
                url = cached[0]
            else:
                m, score = best_match(searcher.search(artist, title), artist, title, dur)
                if m is None:
                    db.execute(
                        "INSERT OR REPLACE INTO preview_matches VALUES (?,?,?,?,?,?,?)",
                        (cid, None, None, None, score, None, "no_match"),
                    )
                    db.commit()
                    no_match += 1
                    mark_failed(db, cid, "no itunes match")
                    continue
                url = m["previewUrl"]
                db.execute(
                    "INSERT OR REPLACE INTO preview_matches VALUES (?,?,?,?,?,?,?)",
                    (cid, m.get("trackId"), m.get("trackName"), m.get("artistName"),
                     round(score, 3), url, "matched"),
                )
                db.commit()
            # 2. download + decode
            m4a = cache / f"{cid}.m4a"
            if not download_preview(url, m4a):
                mark_failed(db, cid, "preview download failed")
                failed += 1
                continue
            wav = decode_to_wav(m4a)
            if wav is None:
                mark_failed(db, cid, "ffmpeg decode failed (corrupted preview)")
                failed += 1
                continue
            # 3. submit DSP
            futures[pool.submit(analyze_clip, (cid, str(wav), None, year))] = cid

            # drain finished futures opportunistically
            done_now = [f for f in futures if f.done()]
            for f in done_now:
                tid, doc, err = f.result()
                del futures[f]
                if err:
                    mark_failed(db, tid, err)
                    failed += 1
                else:
                    write_results(db, tid, doc)
                    done += 1
                if (done + failed) % 50 == 0:
                    rate = (done + failed) / max(time.time() - t0, 1)
                    log.info("done=%d failed=%d no_match=%d (%.1f tracks/min)",
                             done, failed, no_match, rate * 60)

        for f in cf.as_completed(futures):
            tid, doc, err = f.result()
            if err:
                mark_failed(db, tid, err)
                failed += 1
            else:
                write_results(db, tid, doc)
                done += 1

    log.info("complete: %d analyzed, %d failed, %d no match, %.1f min",
             done, failed, no_match, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
