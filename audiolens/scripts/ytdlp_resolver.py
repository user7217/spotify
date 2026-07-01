"""yt-dlp audio resolver — acquire local audio for the heavy DSP analysis path.

ReccoBeats covers *features* without any download. But segment-level *analysis*
(beats, timbre arc, harmonic progression) requires the actual audio + the local
extractor. This script downloads audio for the top-N most-played catalog tracks
that don't yet have a local file, validates each download, and writes the path
back into catalog_tracks.audio_track_id — which is exactly what run_pipeline.py
reads to produce track documents.

Duration validation (the single most important correctness check):
    Compare downloaded duration against the track's known duration (ReccoBeats
    durationMs, else catalog.duration_ms). If they differ by more than
    DURATION_TOL_S, the candidate is the wrong thing — a live take, a loop, an
    hour-long mix, a cover — so we reject it and try the next search hit. This
    one check catches the large majority of yt-dlp mismatches.

Resumable: tracks that already have audio_track_id pointing at an existing file
are skipped. Source is recorded in audio_match_method ('yt-dlp-full').

Usage:
    python scripts/ytdlp_resolver.py --sqlite catalog.db --audio-dir ./audio \
        [--limit 1000] [--candidates 4] [--codec opus]
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sqlite3
import json

log = logging.getLogger("audiolens.ytdlp")

DURATION_TOL_S = 15        # >15s mismatch -> wrong video
DEFAULT_CANDIDATES = 2     # search hits to try (fewer = fewer requests = less rate-limiting)


def _known_duration_s(db: sqlite3.Connection, sid: str, catalog_dur_ms: int | None) -> float | None:
    row = db.execute(
        "SELECT duration_ms FROM reccobeats_features WHERE spotify_track_id=? AND duration_ms IS NOT NULL",
        (sid,),
    ).fetchone()
    ms = (row[0] if row else None) or catalog_dur_ms
    return ms / 1000.0 if ms else None


def _pending(db: sqlite3.Connection, limit: int | None):
    """Top tracks by play_count lacking a usable local audio file."""
    q = """SELECT id, spotify_track_id, title, artists, duration_ms, audio_track_id
           FROM catalog_tracks
           WHERE play_count > 0
           ORDER BY play_count DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = []
    for cid, sid, title, artists, dur, audio in db.execute(q).fetchall():
        if audio and pathlib.Path(audio).exists():
            continue  # already resolved
        rows.append((cid, sid, title, artists, dur))
    return rows


def _primary_artist(artists_json: str) -> str:
    try:
        a = json.loads(artists_json)
        if isinstance(a, list) and a:
            return a[0] if isinstance(a[0], str) else a[0].get("name", "")
        if isinstance(a, str):
            return a
    except Exception:
        pass
    return artists_json or ""


def _ydl_opts(audio_dir: str, codec: str, candidates: int) -> dict:
    # Let yt-dlp pick the player client by DEFAULT — with a JS runtime (Deno)
    # present it auto-selects a client that returns downloadable audio
    # (e.g. android_vr). Forcing a client list tends to *exclude* the one that
    # works (android→SABR, ios→PO-token, tv→DRM), so only override when the
    # user explicitly sets YTDLP_CLIENTS, e.g. YTDLP_CLIENTS="tv,web".
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(audio_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "default_search": f"ytsearch{candidates}",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": codec, "preferredquality": "0"},
        ],
        # don't fail the whole run on one bad video
        "ignoreerrors": True,
        "retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 30,
        # THROTTLING — YouTube rate-limits an IP that fires requests with no
        # gaps ("Video unavailable... rate-limited for up to an hour"). Random
        # sleeps between downloads + between metadata requests keep us under the
        # radar. Tune via YTDLP_SLEEP_MIN / YTDLP_SLEEP_MAX (seconds).
        "sleep_interval": float(os.environ.get("YTDLP_SLEEP_MIN", "3")),
        "max_sleep_interval": float(os.environ.get("YTDLP_SLEEP_MAX", "9")),
        "sleep_interval_requests": float(os.environ.get("YTDLP_SLEEP_REQ", "1")),
    }
    clients = os.environ.get("YTDLP_CLIENTS", "").strip()
    if clients:
        opts["extractor_args"] = {
            "youtube": {"player_client": [c.strip() for c in clients.split(",") if c.strip()]}
        }
    # optional: a cookies.txt (exported from a logged-in browser) bypasses most
    # remaining bot walls. Drop it in the work dir and set YTDLP_COOKIES=/path.
    cookies = os.environ.get("YTDLP_COOKIES")
    if cookies and os.path.exists(cookies):
        opts["cookiefile"] = cookies
    return opts


def _resolve_one(ydl, query: str, want_s: float | None) -> tuple[str, float] | None:
    """Search, evaluate candidates by duration, download the first good match.

    Returns (filepath, duration_s) or None. Uses extract_info(download=False)
    to vet candidates *before* downloading, so a wrong 1-hour mix never gets
    pulled.
    """
    info = ydl.extract_info(query, download=False)
    entries = info.get("entries") if info else None
    if not entries:
        return None
    for entry in entries:
        if not entry:
            continue
        dur = entry.get("duration")
        if want_s is not None and dur and abs(dur - want_s) > DURATION_TOL_S:
            log.debug("skip candidate %s (%.0fs vs want %.0fs)", entry.get("id"), dur, want_s)
            continue
        # accept: download this specific candidate
        dl = ydl.extract_info(entry["webpage_url"], download=True)
        if not dl:
            continue
        path = ydl.prepare_filename(dl)
        # postprocessor changed the extension to the target codec
        base, _ = os.path.splitext(path)
        for ext in (dl.get("ext"), "opus", "mp3", "m4a", "webm"):
            cand = f"{base}.{ext}"
            if os.path.exists(cand):
                return cand, (dl.get("duration") or dur or 0.0)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="yt-dlp audio resolver")
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--limit", type=int, help="cap top-N tracks this run")
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    ap.add_argument("--codec", default="opus", help="opus (small) | mp3 | wav")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        log.error("yt-dlp not installed. pip install yt-dlp")
        return 2

    pathlib.Path(a.audio_dir).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(a.sqlite)
    # progress/state columns already exist in catalog schema (audio_track_id,
    # audio_match_method); add a failures log table for visibility.
    db.execute("""CREATE TABLE IF NOT EXISTS download_failures (
        catalog_track_id TEXT PRIMARY KEY, spotify_track_id TEXT,
        reason TEXT, attempts INTEGER DEFAULT 1,
        last_attempt TEXT DEFAULT (datetime('now')))""")
    db.commit()

    rows = _pending(db, a.limit)
    log.info("%d tracks need audio (resume-aware, top-by-plays first)", len(rows))
    if not rows:
        return 0

    opts = _ydl_opts(a.audio_dir, a.codec, a.candidates)
    got = miss = 0
    with YoutubeDL(opts) as ydl:
        for cid, sid, title, artists, dur in rows:
            artist = _primary_artist(artists)
            want_s = _known_duration_s(db, sid, dur)
            query = f"ytsearch{a.candidates}:{title} {artist} audio"
            try:
                result = _resolve_one(ydl, query, want_s)
            except Exception as e:  # noqa: BLE001
                result = None
                log.warning("error resolving %s - %s: %s", title, artist, e)
            if result:
                path, got_s = result
                db.execute(
                    "UPDATE catalog_tracks SET audio_track_id=?, audio_match_method=?, "
                    "audio_match_score=? WHERE id=?",
                    (os.path.abspath(path), "yt-dlp-full",
                     1.0 if want_s is None else max(0.0, 1 - abs(got_s - want_s) / (want_s or 1)),
                     cid),
                )
                db.execute("DELETE FROM download_failures WHERE catalog_track_id=?", (cid,))
                got += 1
            else:
                db.execute(
                    """INSERT INTO download_failures (catalog_track_id, spotify_track_id, reason)
                       VALUES (?,?,?)
                       ON CONFLICT(catalog_track_id) DO UPDATE SET
                         attempts=attempts+1, last_attempt=datetime('now')""",
                    (cid, sid, "no_valid_candidate"),
                )
                miss += 1
            db.commit()
            if (got + miss) % 25 == 0:
                log.info("progress: %d downloaded, %d missed", got, miss)

    log.info("done: %d downloaded, %d missed", got, miss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
