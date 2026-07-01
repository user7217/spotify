"""Ingest unit tests: normalization, variant detection, dedup. Stdlib + pytest only."""

import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.ingest.app.dedup import DedupEngine
from services.ingest.app.normalize import norm_key, normalize, parse_title
from services.ingest.app.parser import PlayRecord


def _rec(title, artist, sid=None, album=None, ts=None):
    return PlayRecord(
        ts=ts or datetime(2024, 1, 1, tzinfo=timezone.utc), ms_played=30000,
        track_name=title, artist_name=artist, album_name=album,
        spotify_track_id=sid, platform=None, conn_country=None,
        reason_start=None, reason_end=None, shuffle=None, skipped=None,
        offline=None, incognito=None, source_file="t.json",
    )


class TestNormalize:
    def test_accents_case_punct(self):
        assert normalize("Beyoncé — HALO!") == "beyonce halo"

    def test_remaster(self):
        tp = parse_title("I Want to Know What Love Is - 1999 Remaster")
        assert tp.variant_type == "remaster"
        assert tp.norm_title == "i want to know what love is"

    def test_live(self):
        tp = parse_title("Echoes (Live At Pompeii)")
        assert tp.variant_type == "live"

    def test_radio_edit(self):
        assert parse_title("Levels - Radio Edit").variant_type == "radio_edit"

    def test_clean(self):
        assert parse_title("HUMBLE. (Clean Version)").variant_type == "clean"

    def test_deluxe_from_album(self):
        tp = parse_title("Another Love", album="Long Way Down (Deluxe)")
        assert "deluxe" in tp.variant_tags

    def test_feat_stripped_from_key(self):
        k1, _ = norm_key("Artist feat. Guest", "Song")
        k2, _ = norm_key("Artist", "Song")
        assert k1 == k2

    def test_original_untouched(self):
        tp = parse_title("Bohemian Rhapsody")
        assert tp.variant_type == "original"
        assert tp.variant_tags == []


class TestDedup:
    def test_same_spotify_id_merges(self):
        e = DedupEngine()
        a = e.add(_rec("Song", "Artist", sid="abc123"))
        b = e.add(_rec("Song (different metadata)", "Artist", sid="abc123"))
        assert a.id == b.id
        assert a.play_count == 2

    def test_same_norm_no_id_merges(self):
        e = DedupEngine()
        a = e.add(_rec("Hello!", "Adele"))
        b = e.add(_rec("hello", "ADELE"))
        assert a.id == b.id

    def test_variant_kept_separate_and_linked(self):
        e = DedupEngine()
        orig = e.add(_rec("One", "Metallica", sid="id1"))
        e.add(_rec("One", "Metallica", sid="id1"))  # boost canonical play count
        rem = e.add(_rec("One - 2011 Remaster", "Metallica", sid="id2"))
        assert orig.id != rem.id
        res = e.finalize()
        assert rem.canonical_id == orig.id
        link = next(l for l in res.links if l.track_a == rem.id)
        assert link.relation == "remaster_of"
        assert link.method == "name"

    def test_isrc_merge(self):
        e = DedupEngine()
        a = e.add(_rec("Song A", "X", sid="s1"))
        e.add(_rec("Song A", "X", sid="s1"))
        b = e.add(_rec("Song A (Re-issue 2020)", "X", sid="s2"))
        links = e.merge_by_isrc({"s1": "USXX12345678", "s2": "USXX12345678"})
        assert b.canonical_id == a.id
        assert links[0].method == "isrc"

    def test_play_stats(self):
        e = DedupEngine()
        e.add(_rec("S", "A", ts=datetime(2020, 1, 1, tzinfo=timezone.utc)))
        x = e.add(_rec("S", "A", ts=datetime(2023, 1, 1, tzinfo=timezone.utc)))
        assert x.first_played_at.year == 2020
        assert x.last_played_at.year == 2023
        assert x.total_ms_played == 60000
