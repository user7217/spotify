"""REST API routes.

POST /v1/tracks                       upload audio file -> track + analysis job
POST /v1/tracks/batch                 upload multiple files (album / catalog)
GET  /v1/tracks/{id}                  track metadata
GET  /v1/audio-features/{track_id}    Spotify Audio Features replacement
GET  /v1/audio-analysis/{track_id}    Spotify Audio Analysis replacement
GET  /v1/tracks/{id}/similar          embedding similarity search (pgvector)
GET  /v1/jobs/{id}                    job status
"""

from __future__ import annotations

import json
import uuid

from aiokafka import AIOKafkaProducer
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from minio import Minio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import AnalysisJob, AudioAnalysis, AudioFeatures, Embedding, Track
from app.db.session import get_db
from app.schemas.api import (
    AudioAnalysisOut,
    AudioFeaturesOut,
    BatchUploadResponse,
    JobOut,
    SimilarityResponse,
    SimilarTrack,
    TrackOut,
    UploadResponse,
)

settings = get_settings()
router = APIRouter(prefix="/v1")

_minio = Minio(
    settings.s3_endpoint,
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap)
        await _producer.start()
    return _producer


# ── upload helpers ────────────────────────────────────────────────────────────

import hashlib


async def _ingest_one(
    file: UploadFile,
    db: AsyncSession,
    producer: AIOKafkaProducer,
    batch_id: uuid.UUID | None = None,
    spotify_track_id: str | None = None,
) -> UploadResponse:
    fmt = (file.filename or "").rsplit(".", 1)[-1].lower()
    if fmt not in settings.supported_formats:
        raise HTTPException(415, f"unsupported format '{fmt}'")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "file too large")

    fhash = hashlib.sha256(data).hexdigest()

    existing = (
        await db.execute(select(Track).where(Track.file_hash == fhash))
    ).scalar_one_or_none()
    if existing:
        job = (
            await db.execute(
                select(AnalysisJob)
                .where(AnalysisJob.track_id == existing.id)
                .order_by(AnalysisJob.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return UploadResponse(
            track=TrackOut.model_validate(existing),
            job=JobOut.model_validate(job) if job else None,
            deduplicated=True,
        )

    s3_key = f"{fhash[:2]}/{fhash}.{fmt}"
    import io

    _minio.put_object(
        settings.s3_bucket_audio, s3_key, io.BytesIO(data), length=len(data)
    )

    track = Track(
        file_hash=fhash,
        spotify_track_id=spotify_track_id,
        title=(file.filename or "").rsplit(".", 1)[0],
        format=fmt,
        s3_key=s3_key,
    )
    db.add(track)
    await db.flush()

    job = AnalysisJob(track_id=track.id, batch_id=batch_id)
    db.add(job)
    await db.commit()
    await db.refresh(track)
    await db.refresh(job)

    await producer.send_and_wait(
        settings.kafka_topic_jobs,
        json.dumps(
            {
                "job_id": str(job.id),
                "track_id": str(track.id),
                "s3_key": s3_key,
                "format": fmt,
                "outputs": ["features", "analysis", "embedding"],
            }
        ).encode(),
    )

    return UploadResponse(
        track=TrackOut.model_validate(track), job=JobOut.model_validate(job)
    )


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/tracks", response_model=UploadResponse, status_code=201)
async def upload_track(
    file: UploadFile = File(...),
    spotify_track_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    producer = await get_producer()
    return await _ingest_one(file, db, producer, spotify_track_id=spotify_track_id)


@router.post("/tracks/batch", response_model=BatchUploadResponse, status_code=201)
async def upload_batch(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    producer = await get_producer()
    batch_id = uuid.uuid4()
    jobs = []
    for f in files:
        resp = await _ingest_one(f, db, producer, batch_id=batch_id)
        if resp.job:
            jobs.append(resp.job)
    return BatchUploadResponse(batch_id=batch_id, jobs=jobs)


@router.get("/tracks/{track_id}", response_model=TrackOut)
async def get_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = (await db.execute(select(Track).where(Track.id == track_id))).scalar_one_or_none()
    if not track:
        raise HTTPException(404, "track not found")
    return track


@router.get("/audio-features/{track_id}", response_model=AudioFeaturesOut)
async def get_audio_features(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    feats = (
        await db.execute(select(AudioFeatures).where(AudioFeatures.track_id == track_id))
    ).scalar_one_or_none()
    if not feats:
        raise HTTPException(404, "features not found (job may still be running)")
    return feats


@router.get("/audio-analysis/{track_id}", response_model=AudioAnalysisOut)
async def get_audio_analysis(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    a = (
        await db.execute(select(AudioAnalysis).where(AudioAnalysis.track_id == track_id))
    ).scalar_one_or_none()
    if not a:
        raise HTTPException(404, "analysis not found (job may still be running)")
    return {
        "track_id": a.track_id,
        "meta": a.meta,
        "track": a.track_summary,
        "bars": a.bars,
        "beats": a.beats,
        "tatums": a.tatums,
        "sections": a.sections,
        "segments": a.segments,
        "harmony": a.harmony,
        "rhythm": a.rhythm,
    }


@router.get("/tracks/{track_id}/similar", response_model=SimilarityResponse)
async def similar_tracks(
    track_id: uuid.UUID,
    k: int = Query(10, le=100),
    model_version: str = Query("encoder-v1"),
    db: AsyncSession = Depends(get_db),
):
    emb = (
        await db.execute(
            select(Embedding).where(
                Embedding.track_id == track_id, Embedding.model_version == model_version
            )
        )
    ).scalar_one_or_none()
    if not emb:
        raise HTTPException(404, "embedding not found for this track")

    # pgvector cosine distance KNN
    stmt = (
        select(Embedding, Embedding.vector.cosine_distance(emb.vector).label("dist"))
        .where(Embedding.model_version == model_version, Embedding.track_id != track_id)
        .order_by("dist")
        .limit(k)
    )
    rows = (await db.execute(stmt)).all()

    results = []
    for e, dist in rows:
        track = (await db.execute(select(Track).where(Track.id == e.track_id))).scalar_one()
        results.append(SimilarTrack(track=TrackOut.model_validate(track), distance=float(dist)))

    return SimilarityResponse(
        query_track_id=track_id, model_version=model_version, results=results
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = (await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(404, "job not found")
    return job
