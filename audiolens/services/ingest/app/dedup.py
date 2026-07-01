"""Dedup engine.

Identity resolution, strongest evidence first:
  1. spotify_track_id   (exact)
  2. isrc               (exact, available after enrichment)
  3. norm_key           (normalized primary-artist|base-title)
  4. audio fingerprint  (after audio resolution; see resolve/fingerprint.py)

Tracks sharing a norm_key but differing variant tags (remaster, live,
radio edit, deluxe, explicit/clean, ...) are kept as separate catalog rows
and linked: each group elects a canonical track, others point at it via
`canonical_id`, with pairwise `variant_links` evidence rows.
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .normalize import norm_key
from .parser import PlayRecord

log = logging.getLogger("audiolens.ingest.dedup")

_VARIANT_RELATION = {
    "remaster": "remaster_of",
    "live": "live_of",
    "radio_edit": "edit_of",
    "extended": "edit_of",
    "clean": "clean_of",
    "explicit": "clean_of",
    "remix": "remix_of",
    "acoustic": "remix_of",
    "instrumental": "remix_of",
    "demo": "remix_of",
    "sped_up": "remix_of",
}


@dataclass
class CatalogEntry:
    id: uuid.UUID
    spotify_track_id: str | None
    title: str
    artists: list[str]
    album: str | None
    norm_key: str
    variant_type: str
    variant_tags: list[str]
    isrc: str | None = None
    canonical_id: uuid.UUID | None = None
    play_count: int = 0
    total_ms_played: int = 0
    first_played_at: datetime | None = None
    last_played_at: datetime | None = None


@dataclass
class VariantLinkRow:
    track_a: uuid.UUID
    track_b: uuid.UUID
    method: str
    relation: str
    score: float = 1.0


@dataclass
class DedupResult:
    entries: list[CatalogEntry]
    links: list[VariantLinkRow]
    play_assignments: list[tuple[PlayRecord, uuid.UUID]] = field(default_factory=list)


class DedupEngine:
    def __init__(self, keep_plays: bool = True):
        self.keep_plays = keep_plays
        self._by_spotify_id: dict[str, CatalogEntry] = {}
        self._by_norm_variant: dict[tuple[str, tuple[str, ...]], CatalogEntry] = {}
        self._by_norm: dict[str, list[CatalogEntry]] = defaultdict(list)
        self._plays: list[tuple[PlayRecord, uuid.UUID]] = []

    # -- ingestion ---------------------------------------------------------

    def add(self, rec: PlayRecord) -> CatalogEntry:
        key, tp = norm_key(rec.artist_name, rec.track_name, rec.album_name)
        entry = None

        if rec.spotify_track_id:
            entry = self._by_spotify_id.get(rec.spotify_track_id)
        if entry is None:
            # same normalized identity + same variant tags == same track,
            # even across differing spotify ids (re-released singles etc.)
            vkey = (key, tuple(tp.variant_tags))
            entry = self._by_norm_variant.get(vkey)
            if entry is not None and rec.spotify_track_id and entry.spotify_track_id is None:
                entry.spotify_track_id = rec.spotify_track_id
                self._by_spotify_id[rec.spotify_track_id] = entry

        if entry is None:
            entry = CatalogEntry(
                id=uuid.uuid4(),
                spotify_track_id=rec.spotify_track_id,
                title=rec.track_name,
                artists=[rec.artist_name],
                album=rec.album_name,
                norm_key=key,
                variant_type=tp.variant_type,
                variant_tags=tp.variant_tags,
            )
            if rec.spotify_track_id:
                self._by_spotify_id[rec.spotify_track_id] = entry
            self._by_norm_variant[(key, tuple(tp.variant_tags))] = entry
            self._by_norm[key].append(entry)

        entry.play_count += 1
        entry.total_ms_played += rec.ms_played
        if entry.first_played_at is None or rec.ts < entry.first_played_at:
            entry.first_played_at = rec.ts
        if entry.last_played_at is None or rec.ts > entry.last_played_at:
            entry.last_played_at = rec.ts
        if self.keep_plays:
            self._plays.append((rec, entry.id))
        return entry

    # -- ISRC merge pass (post-enrichment) ----------------------------------

    def merge_by_isrc(self, isrc_map: dict[str, str]) -> list[VariantLinkRow]:
        """isrc_map: spotify_track_id -> isrc. Same ISRC == same recording."""
        groups: dict[str, list[CatalogEntry]] = defaultdict(list)
        for sid, isrc in isrc_map.items():
            e = self._by_spotify_id.get(sid)
            if e is not None:
                e.isrc = isrc
                groups[isrc].append(e)
        links = []
        for isrc, members in groups.items():
            if len(members) < 2:
                continue
            canon = max(members, key=lambda e: e.play_count)
            for m in members:
                if m is not canon:
                    m.canonical_id = canon.id
                    links.append(VariantLinkRow(m.id, canon.id, "isrc", "duplicate"))
        return links

    # -- variant grouping ----------------------------------------------------

    def finalize(self) -> DedupResult:
        links: list[VariantLinkRow] = []
        n_groups = 0
        for _key, members in self._by_norm.items():
            if len(members) < 2:
                continue
            n_groups += 1
            canon = self._elect_canonical(members)
            for m in members:
                if m is canon or m.canonical_id is not None:
                    continue
                m.canonical_id = canon.id
                relation = next(
                    (_VARIANT_RELATION[t] for t in m.variant_tags if t in _VARIANT_RELATION),
                    "duplicate",
                )
                links.append(VariantLinkRow(m.id, canon.id, "name", relation, score=0.9))

        entries = list(self._iter_entries())
        log.info(
            "dedup: %d catalog tracks, %d variant groups, %d links, %d plays",
            len(entries), n_groups, len(links), len(self._plays),
        )
        return DedupResult(entries=entries, links=links, play_assignments=self._plays)

    @staticmethod
    def _elect_canonical(members: list[CatalogEntry]) -> CatalogEntry:
        originals = [m for m in members if m.variant_type == "original"]
        pool = originals or members
        return max(pool, key=lambda e: e.play_count)

    def _iter_entries(self):
        seen = set()
        for e in self._by_norm_variant.values():
            if e.id not in seen:
                seen.add(e.id)
                yield e
        for e in self._by_spotify_id.values():
            if e.id not in seen:
                seen.add(e.id)
                yield e
