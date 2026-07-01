"""Nearest-neighbor index builders (spec: Similarity System).

Per embedding model:
  FAISS  — IndexFlatIP for <100k vectors, IVF-PQ above (millions of tracks)
  HNSW   — hnswlib graph index (fast online queries, incremental adds)
plus artist-level indexes built from play-count-weighted track centroids.

pgvector's HNSW (migration 0001) covers the unified 128-d vector in-DB;
these file indexes cover the per-model high-dim vectors.
"""

import json
import logging
import pathlib

import numpy as np

log = logging.getLogger("audiolens.indexes")

IVF_THRESHOLD = 100_000


def _l2n(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def build_faiss(vectors: np.ndarray, out_path: pathlib.Path) -> None:
    import faiss
    X = _l2n(np.ascontiguousarray(vectors.astype(np.float32)))
    n, d = X.shape
    if n >= IVF_THRESHOLD:
        nlist = int(4 * np.sqrt(n))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quant, d, nlist, max(d // 16, 8), 8, faiss.METRIC_INNER_PRODUCT)
        index.train(X)
        index.add(X)
        index.nprobe = max(nlist // 32, 8)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(X)
    if faiss.get_num_gpus() > 0:
        log.info("faiss: %d GPUs visible (searches can use them via index_cpu_to_all_gpus)", faiss.get_num_gpus())
    faiss.write_index(index, str(out_path))
    log.info("faiss index: %d x %d -> %s", n, d, out_path)


def build_hnsw(vectors: np.ndarray, out_path: pathlib.Path,
               m: int = 16, ef_construction: int = 200) -> None:
    import hnswlib
    X = _l2n(vectors.astype(np.float32))
    n, d = X.shape
    index = hnswlib.Index(space="cosine", dim=d)
    index.init_index(max_elements=max(n * 2, 1024), ef_construction=ef_construction, M=m)
    index.add_items(X, np.arange(n))
    index.set_ef(64)
    index.save_index(str(out_path))
    log.info("hnsw index: %d x %d -> %s", n, d, out_path)


def build_all(db_path: str, out_dir: str) -> None:
    """Build FAISS+HNSW per model from sqlite/postgres model_embeddings rows."""
    import sqlite3
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    models = [r[0] for r in db.execute(
        "SELECT DISTINCT model_name FROM model_embeddings")]
    for model in models:
        rows = db.execute(
            "SELECT track_id, dim, vector FROM model_embeddings WHERE model_name=?",
            (model,),
        ).fetchall()
        ids = [r[0] for r in rows]
        X = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        build_faiss(X, out / f"{model}.faiss")
        build_hnsw(X, out / f"{model}.hnsw")
        (out / f"{model}.ids.json").write_text(json.dumps(ids))

        # artist-level: weighted centroid per primary artist
        artist_rows = db.execute(
            """SELECT m.track_id, c.artists, c.play_count
               FROM model_embeddings m
               JOIN catalog_tracks c ON c.audio_track_id = m.track_id
               WHERE m.model_name=?""", (model,),
        ).fetchall()
        if artist_rows:
            by_artist: dict[str, list[tuple[int, float]]] = {}
            idx_of = {tid: i for i, tid in enumerate(ids)}
            for tid, artists, plays in artist_rows:
                if tid in idx_of:
                    a = json.loads(artists)[0]
                    by_artist.setdefault(a, []).append((idx_of[tid], float(plays or 1)))
            names, centroids = [], []
            for a, members in by_artist.items():
                idxs, w = zip(*members)
                w = np.array(w)[:, None]
                centroids.append((X[list(idxs)] * w).sum(axis=0) / w.sum())
                names.append(a)
            A = np.stack(centroids)
            build_faiss(A, out / f"{model}.artists.faiss")
            (out / f"{model}.artists.json").write_text(json.dumps(names))
    log.info("indexes built for models: %s", models)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", default="indexes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_all(a.sqlite, a.out)
