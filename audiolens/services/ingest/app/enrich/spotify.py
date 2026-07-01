"""Spotify Web API enrichment.

Fills isrc, duration_ms, release_year, full artist list, popularity for every
catalog track that has a spotify_track_id. Uses client-credentials flow
(no user scope needed for /v1/tracks).

  export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
  python -m services.ingest.app.enrich.spotify --sqlite catalog.db

Resumable: tracks with enrichment already stored are skipped; 429s honor
Retry-After; progress is committed per batch.
"""

import argparse
import json
import logging
import os
import sqlite3
import time

import httpx

log = logging.getLogger("audiolens.enrich.spotify")

TOKEN_URL = "https://accounts.spotify.com/api/token"
TRACKS_URL = "https://api.spotify.com/v1/tracks"
BATCH = 50  # API max for /v1/tracks


class SpotifyClient:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.environ["SPOTIFY_CLIENT_ID"]
        self.client_secret = client_secret or os.environ["SPOTIFY_CLIENT_SECRET"]
        self._token: str | None = None
        self._token_exp = 0.0
        self.http = httpx.Client(timeout=30)

    def _auth(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        r = self.http.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        r.raise_for_status()
        d = r.json()
        self._token = d["access_token"]
        self._token_exp = time.time() + d["expires_in"]
        return self._token

    def tracks(self, ids: list[str]) -> list[dict | None]:
        assert len(ids) <= BATCH
        while True:
            r = self.http.get(
                TRACKS_URL,
                params={"ids": ",".join(ids)},
                headers={"Authorization": f"Bearer {self._auth()}"},
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "5"))
                log.warning("rate limited, sleeping %ds", wait)
                time.sleep(wait + 1)
                continue
            if r.status_code == 401:
                self._token = None
                continue
            r.raise_for_status()
            return r.json()["tracks"]


def _payload(t: dict) -> dict:
    release = t.get("album", {}).get("release_date") or ""
    return {
        "isrc": (t.get("external_ids") or {}).get("isrc"),
        "duration_ms": t.get("duration_ms"),
        "release_year": int(release[:4]) if release[:4].isdigit() else None,
        "artists": [a["name"] for a in t.get("artists", [])],
        "album": (t.get("album") or {}).get("name"),
        "popularity": t.get("popularity"),
        "explicit": t.get("explicit"),
    }


def enrich_sqlite(db_path: str, client: SpotifyClient | None = None) -> dict:
    """Enrich all pending tracks in a local sqlite catalog. Returns stats."""
    client = client or SpotifyClient()
    db = sqlite3.connect(db_path)
    rows = db.execute(
        """SELECT id, spotify_track_id FROM catalog_tracks
           WHERE spotify_track_id IS NOT NULL AND enrichment IS NULL"""
    ).fetchall()
    log.info("enriching %d tracks", len(rows))
    stats = {"done": 0, "missing": 0}
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        results = client.tracks([sid for _, sid in chunk])
        for (cid, _sid), t in zip(chunk, results):
            if t is None:
                stats["missing"] += 1
                db.execute(
                    "UPDATE catalog_tracks SET enrichment = ? WHERE id = ?",
                    (json.dumps({"missing": True}), cid),
                )
                continue
            p = _payload(t)
            db.execute(
                """UPDATE catalog_tracks SET enrichment = ?, isrc = ?,
                   duration_ms = ?, release_year = ?, artists = ?
                   WHERE id = ?""",
                (json.dumps(p), p["isrc"], p["duration_ms"], p["release_year"],
                 json.dumps(p["artists"]), cid),
            )
            stats["done"] += 1
        db.execute(
            """UPDATE processing_state SET status='done'
               WHERE stage='enrich' AND catalog_track_id IN (%s)"""
            % ",".join("?" * len(chunk)),
            [cid for cid, _ in chunk],
        )
        db.commit()
        if (i // BATCH) % 20 == 0:
            log.info("progress %d/%d", i + len(chunk), len(rows))
    # ISRC merge pass: same ISRC -> point at canonical (most played)
    merged = db.execute(
        """UPDATE catalog_tracks SET canonical_id = (
             SELECT k.id FROM catalog_tracks k
             WHERE k.isrc = catalog_tracks.isrc AND k.id != catalog_tracks.id
               AND k.canonical_id IS NULL
             ORDER BY k.play_count DESC LIMIT 1)
           WHERE isrc IS NOT NULL AND canonical_id IS NULL
             AND EXISTS (SELECT 1 FROM catalog_tracks k2
                         WHERE k2.isrc = catalog_tracks.isrc
                           AND k2.id != catalog_tracks.id
                           AND k2.play_count > catalog_tracks.play_count)"""
    ).rowcount
    db.commit()
    stats["isrc_merged"] = merged
    log.info("enrichment stats: %s", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO)
    enrich_sqlite(a.sqlite)
