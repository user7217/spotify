"""Streaming-history JSON parser.

Handles Spotify extended streaming history exports
(Streaming_History_Audio_*.json). Skips podcast episodes, audiobooks and
video rows. Generator-based: constant memory regardless of dataset size.
"""

import json
import logging
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger("audiolens.ingest.parser")


@dataclass(slots=True)
class PlayRecord:
    ts: datetime
    ms_played: int
    track_name: str
    artist_name: str
    album_name: str | None
    spotify_track_id: str | None  # bare id, not URI
    platform: str | None
    conn_country: str | None
    reason_start: str | None
    reason_end: str | None
    shuffle: bool | None
    skipped: bool | None
    offline: bool | None
    incognito: bool | None
    source_file: str


def _track_id(uri: str | None) -> str | None:
    if uri and uri.startswith("spotify:track:"):
        return uri.rsplit(":", 1)[1]
    return None


def iter_history_files(root: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(root.glob("**/Streaming_History_Audio_*.json"))
    if not files:  # fall back to any json that looks like history
        files = sorted(root.glob("**/*.json"))
    return files


def iter_plays(root: pathlib.Path) -> Iterator[PlayRecord]:
    stats = {"files": 0, "rows": 0, "tracks": 0, "skipped_non_music": 0, "bad_rows": 0}
    for path in iter_history_files(root):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("unreadable history file %s: %s", path.name, e)
            continue
        if not isinstance(rows, list):
            continue
        stats["files"] += 1
        for row in rows:
            stats["rows"] += 1
            name = row.get("master_metadata_track_name")
            artist = row.get("master_metadata_album_artist_name")
            if not name or not artist:
                # episodes / audiobooks / metadata-less rows
                stats["skipped_non_music"] += 1
                continue
            try:
                ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                yield PlayRecord(
                    ts=ts,
                    ms_played=int(row.get("ms_played") or 0),
                    track_name=name,
                    artist_name=artist,
                    album_name=row.get("master_metadata_album_album_name"),
                    spotify_track_id=_track_id(row.get("spotify_track_uri")),
                    platform=row.get("platform"),
                    conn_country=row.get("conn_country"),
                    reason_start=row.get("reason_start"),
                    reason_end=row.get("reason_end"),
                    shuffle=row.get("shuffle"),
                    skipped=row.get("skipped"),
                    offline=row.get("offline"),
                    incognito=row.get("incognito_mode"),
                    source_file=path.name,
                )
                stats["tracks"] += 1
            except (KeyError, ValueError, TypeError) as e:
                stats["bad_rows"] += 1
                log.debug("bad row in %s: %s", path.name, e)
    log.info("parse done: %s", stats)
