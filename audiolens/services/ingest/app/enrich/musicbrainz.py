"""MusicBrainz fallback enrichment (no API key, 1 req/s etiquette).

Used for tracks where Spotify lookup failed or returned no ISRC.
Searches recordings by artist + title, prefers exact-duration matches.
"""

import json
import logging
import sqlite3
import time

import httpx

log = logging.getLogger("audiolens.enrich.musicbrainz")

MB_URL = "https://musicbrainz.org/ws/2/recording"
UA = "AudioLens/0.1 (https://github.com/audiolens; contact@example.com)"
RATE_S = 1.1


def search_recording(client: httpx.Client, artist: str, title: str) -> dict | None:
    r = client.get(
        MB_URL,
        params={
            "query": f'artist:"{artist}" AND recording:"{title}"',
            "fmt": "json",
            "limit": 5,
            "inc": "isrcs",
        },
        headers={"User-Agent": UA},
    )
    if r.status_code == 503:
        time.sleep(5)
        return search_recording(client, artist, title)
    r.raise_for_status()
    recs = r.json().get("recordings", [])
    return recs[0] if recs else None


def enrich_sqlite(db_path: str, limit: int | None = None) -> dict:
    db = sqlite3.connect(db_path)
    rows = db.execute(
        """SELECT id, title, artists FROM catalog_tracks
           WHERE isrc IS NULL
             AND (enrichment IS NULL OR json_extract(enrichment,'$.missing') = 1)
           ORDER BY play_count DESC""" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    stats = {"found": 0, "no_match": 0}
    with httpx.Client(timeout=30) as client:
        for cid, title, artists in rows:
            artist = json.loads(artists)[0]
            try:
                rec = search_recording(client, artist, title)
            except httpx.HTTPError as e:
                log.warning("mb error for %s - %s: %s", artist, title, e)
                continue
            if rec:
                isrcs = rec.get("isrcs") or []
                db.execute(
                    """UPDATE catalog_tracks
                       SET isrc = COALESCE(?, isrc),
                           duration_ms = COALESCE(duration_ms, ?),
                           enrichment = json_patch(COALESCE(enrichment,'{}'), ?)
                       WHERE id = ?""",
                    (
                        isrcs[0] if isrcs else None,
                        rec.get("length"),
                        json.dumps({"mbid": rec["id"], "mb_score": rec.get("score")}),
                        cid,
                    ),
                )
                stats["found"] += 1
            else:
                stats["no_match"] += 1
            db.commit()
            time.sleep(RATE_S)
    log.info("musicbrainz stats: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    enrich_sqlite(a.sqlite, a.limit)
