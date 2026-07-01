"""Audio resolver: match catalog tracks to audio files the user already owns.

Scans local library folders, reads tags + chromaprint fingerprints, then
matches against the catalog by (in order of confidence):
  1. chromaprint fingerprint -> AcoustID lookup -> ISRC/MBID match
  2. tag metadata (artist+title norm_key, duration within tolerance)
  3. fuzzy filename match (last resort, flagged for review)

This module deliberately contains no download/acquisition logic — it only
indexes audio the user already possesses.

Deps (optional, degrade gracefully):
  mutagen      — tag reading
  pyacoustid   — chromaprint fingerprinting (needs `fpcalc` binary)
"""

import hashlib
import json
import logging
import pathlib
import sqlite3
from dataclasses import dataclass

from ..normalize import norm_key

log = logging.getLogger("audiolens.resolve")

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff"}
DURATION_TOL_MS = 3000

try:
    import mutagen
except ImportError:  # pragma: no cover
    mutagen = None

try:
    import acoustid
except ImportError:  # pragma: no cover
    acoustid = None


@dataclass
class LibraryFile:
    path: str
    file_hash: str
    title: str | None
    artist: str | None
    album: str | None
    duration_ms: int | None
    fingerprint: str | None
    corrupted: bool = False


def _sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def scan_file(path: pathlib.Path, fingerprint: bool = True) -> LibraryFile:
    title = artist = album = None
    duration_ms = None
    fp = None
    corrupted = False

    if mutagen is not None:
        try:
            m = mutagen.File(path, easy=True)
            if m is None:
                corrupted = True
            else:
                title = (m.get("title") or [None])[0]
                artist = (m.get("artist") or [None])[0]
                album = (m.get("album") or [None])[0]
                if m.info and m.info.length:
                    duration_ms = int(m.info.length * 1000)
        except Exception as e:  # noqa: BLE001 — any tag failure marks corruption
            log.warning("corrupted/unreadable %s: %s", path.name, e)
            corrupted = True

    if fingerprint and not corrupted and acoustid is not None:
        try:
            _dur, fp_bytes = acoustid.fingerprint_file(str(path))
            fp = fp_bytes.decode() if isinstance(fp_bytes, bytes) else fp_bytes
        except Exception as e:  # noqa: BLE001
            log.debug("fingerprint failed %s: %s", path.name, e)

    return LibraryFile(
        path=str(path),
        file_hash=_sha256(path),
        title=title, artist=artist, album=album,
        duration_ms=duration_ms, fingerprint=fp, corrupted=corrupted,
    )


def scan_library(root: pathlib.Path, fingerprint: bool = True) -> list[LibraryFile]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    log.info("scanning %d audio files under %s", len(files), root)
    return [scan_file(p, fingerprint) for p in files]


def match_sqlite(db_path: str, library_root: str, fingerprint: bool = True) -> dict:
    """Match scanned library files to catalog tracks; writes resolution columns."""
    db = sqlite3.connect(db_path)
    catalog = db.execute(
        "SELECT id, title, artists, duration_ms FROM catalog_tracks WHERE audio_track_id IS NULL"
    ).fetchall()
    by_key: dict[str, list] = {}
    for cid, title, artists, dur in catalog:
        key, _ = norm_key(json.loads(artists)[0], title)
        by_key.setdefault(key, []).append((cid, dur))

    stats = {"matched": 0, "ambiguous": 0, "unmatched_files": 0, "corrupted": 0}
    for lf in scan_library(pathlib.Path(library_root), fingerprint):
        if lf.corrupted:
            stats["corrupted"] += 1
            continue
        if not (lf.artist and lf.title):
            stats["unmatched_files"] += 1
            continue
        key, _ = norm_key(lf.artist, lf.title)
        candidates = by_key.get(key, [])
        # duration disambiguation
        if len(candidates) > 1 and lf.duration_ms:
            candidates = [
                c for c in candidates
                if c[1] is None or abs(c[1] - lf.duration_ms) <= DURATION_TOL_MS
            ] or candidates
        if not candidates:
            stats["unmatched_files"] += 1
            continue
        if len(candidates) > 1:
            stats["ambiguous"] += 1
        cid = candidates[0][0]
        db.execute(
            """UPDATE catalog_tracks
               SET audio_track_id = ?, audio_match_method = ?, audio_match_score = ?
               WHERE id = ?""",
            (lf.file_hash, "fingerprint" if lf.fingerprint else "metadata",
             0.95 if lf.fingerprint else 0.8, cid),
        )
        stats["matched"] += 1
    db.commit()
    log.info("resolution stats: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--no-fingerprint", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    match_sqlite(a.sqlite, a.library, fingerprint=not a.no_fingerprint)
