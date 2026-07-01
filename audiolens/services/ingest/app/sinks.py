"""Persistence sinks for the ingest pipeline.

PostgresSink  — production path, SQLAlchemy models (pgvector stack).
SQLiteSink    — stdlib-only fallback for local runs / CI; same logical schema.
Both are idempotent: re-running ingest upserts rather than duplicates.
"""

import json
import logging
import sqlite3
import uuid
from pathlib import Path

from .dedup import DedupResult

log = logging.getLogger("audiolens.ingest.sinks")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_tracks (
    id TEXT PRIMARY KEY,
    spotify_track_id TEXT UNIQUE,
    isrc TEXT,
    title TEXT NOT NULL,
    artists TEXT NOT NULL,
    album TEXT,
    release_year INTEGER,
    duration_ms INTEGER,
    norm_key TEXT NOT NULL,
    variant_type TEXT NOT NULL DEFAULT 'original',
    variant_tags TEXT NOT NULL DEFAULT '[]',
    canonical_id TEXT REFERENCES catalog_tracks(id),
    audio_track_id TEXT,
    audio_match_method TEXT,
    audio_match_score REAL,
    enrichment TEXT,
    play_count INTEGER NOT NULL DEFAULT 0,
    total_ms_played INTEGER NOT NULL DEFAULT 0,
    first_played_at TEXT,
    last_played_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_catalog_norm ON catalog_tracks(norm_key);
CREATE INDEX IF NOT EXISTS ix_catalog_isrc ON catalog_tracks(isrc);

CREATE TABLE IF NOT EXISTS variant_links (
    track_a TEXT NOT NULL,
    track_b TEXT NOT NULL,
    method TEXT NOT NULL,
    relation TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,
    UNIQUE(track_a, track_b, method)
);

CREATE TABLE IF NOT EXISTS plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_track_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    ms_played INTEGER NOT NULL,
    platform TEXT, conn_country TEXT,
    reason_start TEXT, reason_end TEXT,
    shuffle INTEGER, skipped INTEGER, offline INTEGER, incognito INTEGER,
    source_file TEXT
);
CREATE INDEX IF NOT EXISTS ix_plays_track ON plays(catalog_track_id);

CREATE TABLE IF NOT EXISTS processing_state (
    catalog_track_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    input_hash TEXT, versions TEXT, error TEXT,
    UNIQUE(catalog_track_id, stage)
);
"""


class SQLiteSink:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SQLITE_SCHEMA)

    def write(self, result: DedupResult, with_plays: bool = True) -> None:
        c = self.conn
        c.execute("DELETE FROM plays")  # plays are fully re-derived each run
        c.executemany(
            """INSERT INTO catalog_tracks
               (id, spotify_track_id, isrc, title, artists, album, norm_key,
                variant_type, variant_tags, canonical_id, play_count,
                total_ms_played, first_played_at, last_played_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 play_count=excluded.play_count,
                 total_ms_played=excluded.total_ms_played,
                 isrc=excluded.isrc,
                 canonical_id=excluded.canonical_id,
                 first_played_at=excluded.first_played_at,
                 last_played_at=excluded.last_played_at""",
            [
                (
                    str(e.id), e.spotify_track_id, e.isrc, e.title,
                    json.dumps(e.artists), e.album, e.norm_key, e.variant_type,
                    json.dumps(e.variant_tags),
                    str(e.canonical_id) if e.canonical_id else None,
                    e.play_count, e.total_ms_played,
                    e.first_played_at.isoformat() if e.first_played_at else None,
                    e.last_played_at.isoformat() if e.last_played_at else None,
                )
                for e in result.entries
            ],
        )
        c.executemany(
            """INSERT OR IGNORE INTO variant_links
               (track_a, track_b, method, relation, score) VALUES (?,?,?,?,?)""",
            [(str(l.track_a), str(l.track_b), l.method, l.relation, l.score)
             for l in result.links],
        )
        if with_plays:
            c.executemany(
                """INSERT INTO plays
                   (catalog_track_id, ts, ms_played, platform, conn_country,
                    reason_start, reason_end, shuffle, skipped, offline,
                    incognito, source_file)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        str(tid), p.ts.isoformat(), p.ms_played, p.platform,
                        p.conn_country, p.reason_start, p.reason_end,
                        p.shuffle, p.skipped, p.offline, p.incognito, p.source_file,
                    )
                    for p, tid in result.play_assignments
                ],
            )
        # seed processing checkpoints
        c.executemany(
            """INSERT OR IGNORE INTO processing_state
               (catalog_track_id, stage, status) VALUES (?, 'enrich', 'pending')""",
            [(str(e.id),) for e in result.entries],
        )
        c.commit()
        log.info("sqlite sink: %d tracks, %d links, %d plays committed",
                 len(result.entries), len(result.links), len(result.play_assignments))

    def close(self):
        self.conn.close()


class PostgresSink:
    """Writes via SQLAlchemy into the alembic-managed schema."""

    def __init__(self, database_url: str):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(self.engine)

    def write(self, result: DedupResult, with_plays: bool = True) -> None:
        from sqlalchemy.dialects.postgresql import insert

        from services.api.app.db.models import (
            CatalogTrack,
            Play,
            ProcessingState,
            VariantLink,
        )

        with self.Session.begin() as s:
            rows = [
                dict(
                    id=e.id, spotify_track_id=e.spotify_track_id, isrc=e.isrc,
                    title=e.title, artists=e.artists, album=e.album,
                    norm_key=e.norm_key, variant_type=e.variant_type,
                    canonical_id=e.canonical_id, play_count=e.play_count,
                    total_ms_played=e.total_ms_played,
                    first_played_at=e.first_played_at, last_played_at=e.last_played_at,
                )
                for e in result.entries
            ]
            for chunk in _chunks(rows, 1000):
                stmt = insert(CatalogTrack).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[CatalogTrack.id],
                    set_={
                        "play_count": stmt.excluded.play_count,
                        "total_ms_played": stmt.excluded.total_ms_played,
                        "isrc": stmt.excluded.isrc,
                        "canonical_id": stmt.excluded.canonical_id,
                        "first_played_at": stmt.excluded.first_played_at,
                        "last_played_at": stmt.excluded.last_played_at,
                    },
                )
                s.execute(stmt)

            link_rows = [
                dict(id=uuid.uuid4(), track_a=l.track_a, track_b=l.track_b,
                     method=l.method, relation=l.relation, score=l.score)
                for l in result.links
            ]
            for chunk in _chunks(link_rows, 1000):
                s.execute(insert(VariantLink).values(chunk).on_conflict_do_nothing())

            if with_plays:
                s.query(Play).delete()
                play_rows = [
                    dict(
                        catalog_track_id=tid, ts=p.ts, ms_played=p.ms_played,
                        platform=p.platform, conn_country=p.conn_country,
                        reason_start=p.reason_start, reason_end=p.reason_end,
                        shuffle=p.shuffle, skipped=p.skipped, offline=p.offline,
                        incognito=p.incognito, source_file=p.source_file,
                    )
                    for p, tid in result.play_assignments
                ]
                for chunk in _chunks(play_rows, 5000):
                    s.execute(insert(Play).values(chunk))

            state_rows = [
                dict(id=uuid.uuid4(), catalog_track_id=e.id, stage="enrich")
                for e in result.entries
            ]
            for chunk in _chunks(state_rows, 1000):
                s.execute(insert(ProcessingState).values(chunk).on_conflict_do_nothing())

        log.info("postgres sink: %d tracks committed", len(result.entries))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
