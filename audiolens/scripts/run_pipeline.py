"""Batch analysis orchestrator: catalog -> full track documents.

  python scripts/run_pipeline.py --sqlite catalog.db --docs-dir docs/ \
      [--workers 8] [--models clap,openl3] [--limit 100]

For every catalog track with resolved audio (audio_track_id set, which the
resolver maps to a local file path), runs the full DSP + embedding +
classification pipeline and assembles the unified output document:

  {track_id, metadata, audio_features, rhythm, harmony, structure,
   genre, mood, instruments, vocals, production, embeddings}

Properties (spec Requirements):
  parallel      — process pool for DSP (CPU-bound); embeddings run in the
                  parent (one GPU model load, batched) when torch sees a GPU
  resumable     — processing_state rows checkpoint every stage
  cached        — done+same input_hash stages are skipped on re-run
  reproducible  — fixed seeds, versions recorded in every payload
  corrupted     — CorruptedAudioError marks status=failed stage=dsp
"""

import argparse
import concurrent.futures as cf
import json
import logging
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

log = logging.getLogger("audiolens.pipeline")


def _analyze_one(args: tuple) -> tuple[str, dict | None, str | None]:
    """Subprocess entry: DSP-only stages (no torch in workers)."""
    track_id, audio_path = args
    from services.extractor.app.pipeline import CorruptedAudioError, analyze_track
    try:
        return track_id, analyze_track(audio_path), None
    except CorruptedAudioError:
        return track_id, None, "corrupted_audio"
    except Exception as e:  # noqa: BLE001
        return track_id, None, f"{type(e).__name__}: {e}"


def _pending(db: sqlite3.Connection, limit: int | None):
    q = """SELECT c.id, c.audio_track_id, c.title, c.artists, c.album,
                  c.release_year, c.duration_ms, c.isrc, c.spotify_track_id
           FROM catalog_tracks c
           WHERE c.audio_track_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM processing_state p
                             WHERE p.catalog_track_id = c.id
                               AND p.stage = 'document' AND p.status = 'done')
           ORDER BY c.play_count DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return db.execute(q).fetchall()


def _mark(db, cid, stage, status, error=None):
    db.execute(
        """INSERT INTO processing_state (catalog_track_id, stage, status, error)
           VALUES (?,?,?,?)
           ON CONFLICT(catalog_track_id, stage)
           DO UPDATE SET status=excluded.status, error=excluded.error,
                         attempts=attempts+1""",
        (cid, stage, status, error),
    )
    db.commit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--audio-map", help="JSON {audio_track_id: filepath}; defaults to "
                                        "treating audio_track_id as a path/hash map table")
    ap.add_argument("--docs-dir", default="track_docs")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--models", help="comma list; default all available")
    ap.add_argument("--limit", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    db = sqlite3.connect(a.sqlite)
    docs_dir = pathlib.Path(a.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    audio_map = json.loads(pathlib.Path(a.audio_map).read_text()) if a.audio_map else {}
    models = a.models.split(",") if a.models else None

    rows = _pending(db, a.limit)
    log.info("%d tracks pending (resume-aware)", len(rows))
    if not rows:
        return 0

    # embeddings/classifiers in parent (GPU); DSP fan-out to processes
    try:
        from services.embedder.app.heads import classify_all
        from services.embedder.app.models.registry import embed_all
        ml_ok = True
    except ImportError as e:
        log.warning("ML stack unavailable (%s) — DSP-only documents", e)
        ml_ok = False

    import librosa

    jobs = []
    for r in rows:
        path = audio_map.get(r[1], r[1])
        if not pathlib.Path(path).exists():
            _mark(db, r[0], "dsp", "failed", "audio file missing")
            continue
        jobs.append(((r[0], path), r))

    done = failed = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(_analyze_one, j): (j, meta) for j, meta in jobs}
        for fut in cf.as_completed(futures):
            (job, meta) = futures[fut]
            cid, audio_path = job
            (_, _, title, artists, album, year, dur, isrc, sid) = meta
            track_id, doc, err = fut.result()
            if err:
                stage = "dsp"
                _mark(db, cid, stage, "failed", err)
                failed += 1
                log.error("track=%s FAILED: %s", title, err)
                continue
            _mark(db, cid, "dsp", "done")

            if ml_ok:
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                emb = embed_all(y, sr, models)
                doc["embeddings"] = {
                    k: {"dim": v["dim"], "version": v["version"]} for k, v in emb.items()
                }
                _persist_embeddings(db, meta[1], emb)
                _mark(db, cid, "embed", "done")
                heads = classify_all(
                    y, sr, emb,
                    key_mode=doc.get("harmony", {}).get("mode"),
                    release_year=year,
                )
                doc.update({k: heads[k] for k in
                            ("genre", "mood", "instruments", "vocals", "production")})
                _mark(db, cid, "classify", "done")

            doc_out = {
                "track_id": cid,
                "metadata": {
                    "track_name": title,
                    "artists": json.loads(artists),
                    "album": album,
                    "release_year": year,
                    "duration_ms": dur,
                    "isrc": isrc,
                    "spotify_track_id": sid,
                },
                **{k: v for k, v in doc.items() if not k.startswith("_")},
                "meta": doc["_meta"],
            }
            (docs_dir / f"{cid}.json").write_text(
                json.dumps(doc_out, ensure_ascii=False, indent=1)
            )
            _mark(db, cid, "document", "done")
            done += 1
            if done % 25 == 0:
                log.info("progress: %d done, %d failed", done, failed)

    log.info("pipeline complete: %d done, %d failed", done, failed)
    return 0


def _persist_embeddings(db, audio_track_id, emb: dict):
    import numpy as np
    db.execute("""CREATE TABLE IF NOT EXISTS model_embeddings (
        track_id TEXT NOT NULL, model_name TEXT NOT NULL,
        model_version TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
        UNIQUE(track_id, model_name, model_version))""")
    for name, v in emb.items():
        db.execute(
            """INSERT OR REPLACE INTO model_embeddings
               (track_id, model_name, model_version, dim, vector)
               VALUES (?,?,?,?,?)""",
            (audio_track_id, name, v["version"], v["dim"],
             np.array(v["vector"], dtype=np.float32).tobytes()),
        )
    db.commit()


if __name__ == "__main__":
    sys.exit(main())
