"""Embedding inference worker.

Listens to extraction-complete events, computes embeddings for new tracks,
writes to pgvector, and periodically rebuilds a FAISS index for bulk
similarity jobs (pgvector serves online KNN; FAISS serves offline/batch).

Without a trained checkpoint it falls back to a deterministic feature-vector
embedding (audio_features + analysis stats -> 128-dim), so the similarity
API works from day one and upgrades transparently once a model is trained.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import numpy as np
import structlog
from aiokafka import AIOKafkaConsumer
from minio import Minio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import sys
sys.path.insert(0, "/srv/api")

from app.core.config import get_settings  # noqa: E402
from app.db.models import AudioAnalysis, AudioFeatures, Embedding, Track  # noqa: E402

log = structlog.get_logger()
settings = get_settings()

engine = create_async_engine(settings.pg_dsn, pool_size=5)
Session = async_sessionmaker(engine, expire_on_commit=False)

CHECKPOINT = os.environ.get("ENCODER_CHECKPOINT", "/models/encoder.ckpt")
MODEL_VERSION_NEURAL = "encoder-v1"
MODEL_VERSION_FALLBACK = "featvec-v1"

minio = Minio(
    settings.s3_endpoint,
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(CHECKPOINT):
        import torch

        from app.models.encoder import SongEncoder

        model = SongEncoder()
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(
            {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
        )
        model.eval()
        _model = model
        log.info("neural encoder loaded", checkpoint=CHECKPOINT)
    else:
        _model = "fallback"
        log.info("no checkpoint — using feature-vector fallback embedding")
    return _model


def neural_embedding(audio_bytes: bytes, fmt: str) -> np.ndarray:
    import librosa
    import torch

    sys.path.insert(0, "/srv/embedder")
    from app.training.train import to_mel

    import io
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
    # average embedding over 10s windows
    win = 22050 * 10
    chunks = [y[i : i + win] for i in range(0, max(len(y) - win, 1), win)] or [y]
    mels = torch.stack([to_mel(c) for c in chunks[:6]])
    with torch.no_grad():
        out = load_model()(mels)
    z = out["embedding"].mean(dim=0)
    z = z / (z.norm() + 1e-8)
    return z.numpy()


def fallback_embedding(feats: AudioFeatures, analysis: AudioAnalysis | None) -> np.ndarray:
    """Deterministic 128-dim vector from features + analysis aggregates."""
    base = np.array(
        [
            feats.danceability, feats.energy, feats.valence, feats.speechiness,
            feats.acousticness, feats.instrumentalness, feats.liveness,
            (feats.loudness + 60) / 60, feats.tempo / 250, feats.mode,
        ],
        dtype=np.float32,
    )
    key_onehot = np.zeros(12, dtype=np.float32)
    if 0 <= feats.key <= 11:
        key_onehot[feats.key] = 1

    extra = np.zeros(20, dtype=np.float32)
    if analysis:
        seg = analysis.segments[:200]
        if seg:
            pitches = np.array([s["pitches"] for s in seg])
            timbre = np.array([s["timbre"] for s in seg])
            extra[:12] = timbre.mean(axis=0) / 100
            extra[12:14] = [pitches.std(), timbre.std() / 100]
        if analysis.rhythm:
            extra[14] = analysis.rhythm.get("beat_regularity") or 0
            extra[15] = analysis.rhythm.get("syncopation") or 0
            extra[16] = analysis.rhythm.get("rhythmic_entropy") or 0
        if analysis.harmony:
            extra[17] = analysis.harmony.get("harmonic_change_rate") or 0

    v = np.concatenate([base, key_onehot, extra])
    # project to 128 dims with a fixed random matrix (seeded -> reproducible)
    rng = np.random.RandomState(42)
    proj = rng.randn(len(v), 128).astype(np.float32) / np.sqrt(len(v))
    z = v @ proj
    return z / (np.linalg.norm(z) + 1e-8)


async def embed_track(track_id: uuid.UUID):
    async with Session() as session:
        track = (await session.execute(select(Track).where(Track.id == track_id))).scalar_one()
        feats = (
            await session.execute(select(AudioFeatures).where(AudioFeatures.track_id == track_id))
        ).scalar_one_or_none()
        analysis = (
            await session.execute(select(AudioAnalysis).where(AudioAnalysis.track_id == track_id))
        ).scalar_one_or_none()

        model = load_model()
        if model == "fallback":
            if not feats:
                log.warning("no features yet, skipping", track_id=str(track_id))
                return
            vec = fallback_embedding(feats, analysis)
            version = MODEL_VERSION_FALLBACK
        else:
            resp = minio.get_object(settings.s3_bucket_audio, track.s3_key)
            try:
                audio = resp.read()
            finally:
                resp.close()
                resp.release_conn()
            vec = await asyncio.to_thread(neural_embedding, audio, track.format)
            version = MODEL_VERSION_NEURAL

        existing = (
            await session.execute(
                select(Embedding).where(
                    Embedding.track_id == track_id, Embedding.model_version == version
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.vector = vec.tolist()
        else:
            session.add(Embedding(track_id=track_id, model_version=version, vector=vec.tolist()))
        await session.commit()
        log.info("embedded", track_id=str(track_id), version=version)


async def rebuild_faiss_index():
    """Periodic: dump all embeddings into a FAISS index for batch jobs."""
    try:
        import faiss
    except ImportError:
        return
    async with Session() as session:
        rows = (
            await session.execute(
                select(Embedding.track_id, Embedding.vector).where(
                    Embedding.model_version.in_([MODEL_VERSION_NEURAL, MODEL_VERSION_FALLBACK])
                )
            )
        ).all()
    if not rows:
        return
    vecs = np.array([r.vector for r in rows], dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
    faiss.write_index(index, settings.faiss_index_path)
    ids = [str(r.track_id) for r in rows]
    with open(settings.faiss_index_path + ".ids.json", "w") as f:
        json.dump(ids, f)
    log.info("faiss index rebuilt", n=len(ids))


async def main():
    consumer = AIOKafkaConsumer(
        settings.kafka_topic_results,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="embedder-workers",
    )
    await consumer.start()
    log.info("embedder worker started")

    async def periodic_faiss():
        while True:
            await asyncio.sleep(600)
            await rebuild_faiss_index()

    asyncio.create_task(periodic_faiss())
    try:
        async for msg in consumer:
            event = json.loads(msg.value)
            if event.get("status") == "done" and event.get("track_id"):
                await embed_track(uuid.UUID(event["track_id"]))
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
