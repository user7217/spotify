"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-10
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tracks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("file_hash", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("spotify_track_id", sa.String(32), nullable=True, index=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("artist", sa.Text, nullable=True),
        sa.Column("album", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.BigInteger, nullable=True),
        sa.Column("format", sa.String(8), nullable=True),
        sa.Column("s3_key", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "audio_features",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("danceability", sa.Float, nullable=False),
        sa.Column("energy", sa.Float, nullable=False),
        sa.Column("valence", sa.Float, nullable=False),
        sa.Column("speechiness", sa.Float, nullable=False),
        sa.Column("acousticness", sa.Float, nullable=False),
        sa.Column("instrumentalness", sa.Float, nullable=False),
        sa.Column("liveness", sa.Float, nullable=False),
        sa.Column("loudness", sa.Float, nullable=False),
        sa.Column("tempo", sa.Float, nullable=False),
        sa.Column("key", sa.Integer, nullable=False),
        sa.Column("mode", sa.Integer, nullable=False),
        sa.Column("time_signature", sa.Integer, nullable=False),
        sa.Column("extractor_version", sa.String(32), nullable=False),
        sa.Column("model_confidences", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "audio_analysis",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("meta", JSONB, nullable=False),
        sa.Column("track_summary", JSONB, nullable=False),
        sa.Column("bars", JSONB, nullable=False),
        sa.Column("beats", JSONB, nullable=False),
        sa.Column("tatums", JSONB, nullable=False),
        sa.Column("sections", JSONB, nullable=False),
        sa.Column("segments", JSONB, nullable=False),
        sa.Column("harmony", JSONB, nullable=True),
        sa.Column("rhythm", JSONB, nullable=True),
        sa.Column("extractor_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("vector", Vector(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "model_version"),
    )
    # HNSW index for fast cosine KNN
    op.execute(
        "CREATE INDEX embeddings_vector_hnsw ON embeddings "
        "USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "analysis_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued", index=True),
        sa.Column("stage", sa.String(32), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("requested_outputs", JSONB, nullable=False,
                  server_default='["features","analysis","embedding"]'),
        sa.Column("batch_id", UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("analysis_jobs")
    op.drop_table("embeddings")
    op.drop_table("audio_analysis")
    op.drop_table("audio_features")
    op.drop_table("tracks")
