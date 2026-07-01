"""Extraction worker.

Consumes analysis jobs from Kafka, pulls audio from MinIO, runs feature +
analysis extraction, writes results to Postgres, publishes completion events.

Horizontally scalable: run N replicas, Kafka consumer group balances partitions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from minio import Minio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import sys
sys.path.insert(0, "/srv/api")  # shared models from api service

from app.core.config import get_settings  # noqa: E402
from app.db.models import AnalysisJob, AudioAnalysis, AudioFeatures  # noqa: E402

from .analysis.extractor import ANALYSIS_VERSION, AnalysisExtractor  # noqa: E402
from .audio.loader import load_audio  # noqa: E402
from .features.extractor import EXTRACTOR_VERSION, FeatureExtractor  # noqa: E402

log = structlog.get_logger()
settings = get_settings()

engine = create_async_engine(settings.pg_dsn, pool_size=5)
Session = async_sessionmaker(engine, expire_on_commit=False)

feature_extractor = FeatureExtractor(sample_rate=settings.sample_rate)
analysis_extractor = AnalysisExtractor(sample_rate=settings.sample_rate)

minio = Minio(
    settings.s3_endpoint,
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)


async def set_job(session, job_id: uuid.UUID, **kwargs):
    await session.execute(update(AnalysisJob).where(AnalysisJob.id == job_id).values(**kwargs))
    await session.commit()


def fetch_audio(s3_key: str) -> bytes:
    resp = minio.get_object(settings.s3_bucket_audio, s3_key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


async def process_job(msg: dict, producer: AIOKafkaProducer):
    job_id = uuid.UUID(msg["job_id"])
    track_id = uuid.UUID(msg["track_id"])
    s3_key = msg["s3_key"]
    fmt = msg["format"]
    outputs = set(msg.get("outputs", ["features", "analysis"]))

    async with Session() as session:
        try:
            await set_job(session, job_id, status="running", stage="loading")
            data = await asyncio.to_thread(fetch_audio, s3_key)
            y, sr = await asyncio.to_thread(load_audio, data, fmt, settings.sample_rate)

            if "features" in outputs:
                await set_job(session, job_id, stage="features")
                feats = await asyncio.to_thread(feature_extractor.extract, y, sr)
                session.add(
                    AudioFeatures(
                        track_id=track_id,
                        danceability=feats.danceability,
                        energy=feats.energy,
                        valence=feats.valence,
                        speechiness=feats.speechiness,
                        acousticness=feats.acousticness,
                        instrumentalness=feats.instrumentalness,
                        liveness=feats.liveness,
                        loudness=feats.loudness,
                        tempo=feats.tempo,
                        key=feats.key,
                        mode=feats.mode,
                        time_signature=feats.time_signature,
                        extractor_version=EXTRACTOR_VERSION,
                        model_confidences=feats.confidences,
                    )
                )
                await session.commit()

            if "analysis" in outputs:
                await set_job(session, job_id, stage="analysis")
                analysis = await asyncio.to_thread(analysis_extractor.extract, y, sr)
                session.add(
                    AudioAnalysis(
                        track_id=track_id,
                        meta=analysis.meta,
                        track_summary=analysis.track,
                        bars=analysis.bars,
                        beats=analysis.beats,
                        tatums=analysis.tatums,
                        sections=analysis.sections,
                        segments=analysis.segments,
                        harmony=analysis.harmony,
                        rhythm=analysis.rhythm,
                        extractor_version=ANALYSIS_VERSION,
                    )
                )
                await session.commit()

            await set_job(session, job_id, status="done", stage="complete")
            await producer.send_and_wait(
                settings.kafka_topic_results,
                json.dumps({"job_id": str(job_id), "track_id": str(track_id), "status": "done"}).encode(),
            )
            log.info("job done", job_id=str(job_id), track_id=str(track_id))

        except Exception as e:
            log.exception("job failed", job_id=str(job_id))
            await set_job(session, job_id, status="failed", error=str(e)[:2000])
            await producer.send_and_wait(
                settings.kafka_topic_results,
                json.dumps({"job_id": str(job_id), "status": "failed", "error": str(e)[:500]}).encode(),
            )


async def main():
    logging.basicConfig(level=settings.log_level)
    consumer = AIOKafkaConsumer(
        settings.kafka_topic_jobs,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id=settings.kafka_group_extractor,
        enable_auto_commit=False,
        max_poll_records=1,  # extraction is heavy — one job at a time per worker
    )
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap)
    await consumer.start()
    await producer.start()
    log.info("extractor worker started")
    try:
        async for msg in consumer:
            payload = json.loads(msg.value)
            await process_job(payload, producer)
            await consumer.commit()
    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
