"""ReccoBeats feature backfill — cheap audio features by Spotify track ID.

Two-step flow (verified against api.reccobeats.com, June 2026):

  1. GET /v1/track?ids=<spotifyId,...>   (batch, up to 40 ids)
        -> {"content": [{ "id": <reccobeats-uuid>, "href": ".../track/<spotifyId>",
                          "durationMs", "isrc", "popularity", ... }, ...]}
     ReccoBeats keys features by its OWN uuid, not the Spotify id, so this
     resolve step is mandatory. The response also hands us durationMs / isrc /
     popularity for free — durationMs is reused by the yt-dlp resolver to
     validate downloads.

  2. GET /v1/track/<reccobeats-uuid>/audio-features
        -> {acousticness, danceability, energy, instrumentalness, key, liveness,
            loudness, mode, speechiness, tempo, valence}   (11 features; the only
            Spotify field missing is time_signature)

Writes everything into the same SQLite catalog the ingest CLI produced, in a
`reccobeats_features` table. Idempotent + resumable: tracks already present are
skipped, so re-running after an interruption only does the remainder.

Usage:
    python scripts/reccobeats_client.py --sqlite catalog.db [--limit N] [--qps 8]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
import urllib.parse

import httpx

log = logging.getLogger("audiolens.reccobeats")

BASE = "https://api.reccobeats.com/v1"
RESOLVE_BATCH = 40           # max ids per /v1/track call
FEATURE_COLS = (
    "acousticness", "danceability", "energy", "instrumentalness", "key",
    "liveness", "loudness", "mode", "speechiness", "tempo", "valence",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS reccobeats_features (
    spotify_track_id TEXT PRIMARY KEY,
    reccobeats_id    TEXT,
    isrc             TEXT,
    duration_ms      INTEGER,
    popularity       INTEGER,
    acousticness     REAL,
    danceability     REAL,
    energy           REAL,
    instrumentalness REAL,
    key              INTEGER,
    liveness         REAL,
    loudness         REAL,
    mode             INTEGER,
    speechiness      REAL,
    tempo            REAL,
    valence          REAL,
    status           TEXT NOT NULL DEFAULT 'ok',   -- ok | not_found | error
    fetched_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _spotify_id_from_href(href: str | None) -> str | None:
    """ '.../track/2jdAk8ATWIL3dwT47XpRfu' -> '2jdAk8ATWIL3dwT47XpRfu' """
    if not href:
        return None
    return href.rstrip("/").rsplit("/", 1)[-1]


def _request(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response | None:
    """GET with bounded retries; honours 429 Retry-After; None on hard failure."""
    for attempt in range(6):
        try:
            r = client.get(url, params=params, timeout=30.0)
        except httpx.HTTPError as e:
            log.warning("network error %s (attempt %d): %s", url, attempt, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            log.info("429 rate-limited, sleeping %.1fs", wait)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return r  # caller treats as not_found
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return r
    return None


def resolve_batch(client: httpx.Client, spotify_ids: list[str]) -> dict[str, dict]:
    """Spotify ids -> {spotify_id: {reccobeats_id, duration_ms, isrc, popularity}}."""
    params = {"ids": ",".join(spotify_ids)}
    r = _request(client, f"{BASE}/track", params=params)
    out: dict[str, dict] = {}
    if r is None or r.status_code != 200:
        return out
    for t in (r.json() or {}).get("content", []):
        sid = _spotify_id_from_href(t.get("href"))
        if not sid:
            continue
        out[sid] = {
            "reccobeats_id": t.get("id"),
            "duration_ms": t.get("durationMs"),
            "isrc": t.get("isrc"),
            "popularity": t.get("popularity"),
        }
    return out


def fetch_features(client: httpx.Client, reccobeats_id: str) -> dict | None:
    r = _request(client, f"{BASE}/track/{reccobeats_id}/audio-features")
    if r is None or r.status_code != 200:
        return None
    return r.json()


def _pending_ids(db: sqlite3.Connection, limit: int | None) -> list[str]:
    q = """SELECT c.spotify_track_id
           FROM catalog_tracks c
           WHERE c.spotify_track_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM reccobeats_features f
                             WHERE f.spotify_track_id = c.spotify_track_id)
           ORDER BY c.play_count DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return [row[0] for row in db.execute(q).fetchall()]


def _store(db, sid, resolved, feats, status):
    row = {"spotify_track_id": sid, "status": status,
           "reccobeats_id": None, "isrc": None, "duration_ms": None, "popularity": None}
    row.update({c: None for c in FEATURE_COLS})
    if resolved:
        row.update({k: resolved.get(k) for k in ("reccobeats_id", "isrc", "duration_ms", "popularity")})
    if feats:
        row["isrc"] = row["isrc"] or feats.get("isrc")
        for c in FEATURE_COLS:
            row[c] = feats.get(c)
    cols = ", ".join(row)
    ph = ", ".join("?" for _ in row)
    db.execute(
        f"INSERT OR REPLACE INTO reccobeats_features ({cols}, fetched_at) "
        f"VALUES ({ph}, datetime('now'))",
        list(row.values()),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ReccoBeats feature backfill")
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--limit", type=int, help="cap tracks this run (resume-friendly)")
    ap.add_argument("--qps", type=float, default=8.0, help="max feature requests/sec")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    db = sqlite3.connect(a.sqlite)
    db.executescript(SCHEMA)
    db.commit()

    pending = _pending_ids(db, a.limit)
    log.info("%d tracks need ReccoBeats features", len(pending))
    if not pending:
        return 0

    min_interval = 1.0 / a.qps if a.qps > 0 else 0.0
    ok = not_found = error = 0
    headers = {
        "Accept": "application/json",
        "User-Agent": "audiolens-personal/0.1 (+https://reccobeats.com)",
    }
    with httpx.Client(headers=headers) as client:
        for i in range(0, len(pending), RESOLVE_BATCH):
            batch = pending[i:i + RESOLVE_BATCH]
            resolved = resolve_batch(client, batch)
            for sid in batch:
                info = resolved.get(sid)
                if not info or not info.get("reccobeats_id"):
                    _store(db, sid, None, None, "not_found")
                    not_found += 1
                    continue
                t0 = time.time()
                feats = fetch_features(client, info["reccobeats_id"])
                if feats is None:
                    _store(db, sid, info, None, "error")
                    error += 1
                else:
                    _store(db, sid, info, feats, "ok")
                    ok += 1
                dt = time.time() - t0
                if dt < min_interval:
                    time.sleep(min_interval - dt)
            db.commit()
            log.info("progress: %d ok, %d not_found, %d error (of %d)",
                     ok, not_found, error, len(pending))

    log.info("done: %d ok, %d not_found, %d error", ok, not_found, error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
