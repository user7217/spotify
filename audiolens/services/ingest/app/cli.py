"""Ingest CLI.

  python -m services.ingest.app.cli SP_DATA_DIR --sqlite catalog.db
  python -m services.ingest.app.cli SP_DATA_DIR --pg postgresql+psycopg://...
  python -m services.ingest.app.cli SP_DATA_DIR --export catalog.jsonl

Parses streaming history, dedups, detects variants, persists, and prints a
summary. Idempotent — safe to re-run after new exports are dropped in.
"""

import argparse
import json
import logging
import pathlib
import sys
import time

from .dedup import DedupEngine
from .parser import iter_plays
from .sinks import PostgresSink, SQLiteSink


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AudioLens streaming-history ingest")
    p.add_argument("data_dir", type=pathlib.Path)
    p.add_argument("--sqlite", type=pathlib.Path, help="write to a local sqlite file")
    p.add_argument("--pg", help="postgres DATABASE_URL")
    p.add_argument("--export", type=pathlib.Path, help="also dump catalog as JSONL")
    p.add_argument("--no-plays", action="store_true", help="skip storing raw play events")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("audiolens.ingest")

    t0 = time.time()
    engine = DedupEngine(keep_plays=not args.no_plays)
    n = 0
    for rec in iter_plays(args.data_dir):
        engine.add(rec)
        n += 1
        if n % 50_000 == 0:
            log.info("processed %d plays...", n)
    result = engine.finalize()

    if args.sqlite:
        sink = SQLiteSink(args.sqlite)
        sink.write(result, with_plays=not args.no_plays)
        sink.close()
    if args.pg:
        PostgresSink(args.pg).write(result, with_plays=not args.no_plays)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            for e in result.entries:
                f.write(json.dumps({
                    "track_id": str(e.id),
                    "spotify_track_id": e.spotify_track_id,
                    "isrc": e.isrc,
                    "track_name": e.title,
                    "artists": e.artists,
                    "album": e.album,
                    "variant_type": e.variant_type,
                    "variant_tags": e.variant_tags,
                    "canonical_id": str(e.canonical_id) if e.canonical_id else None,
                    "play_count": e.play_count,
                    "total_ms_played": e.total_ms_played,
                }, ensure_ascii=False) + "\n")

    variants = sum(1 for e in result.entries if e.canonical_id)
    print(
        f"\nplays parsed        {n:>10,}"
        f"\ncatalog tracks      {len(result.entries):>10,}"
        f"\n  variant tracks    {variants:>10,}"
        f"\n  variant links     {len(result.links):>10,}"
        f"\nelapsed             {time.time() - t0:>9.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
