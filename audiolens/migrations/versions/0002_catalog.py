"""catalog layer: history ingestion, dedup, feature store

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-11
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalog_tracks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("spotify_track_id", sa.String(32), unique=True, index=True, nullable=True),
        sa.Column("isrc", sa.String(15), index=True, nullable=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("artists", JSONB, nullable=False, server_default="[]"),
        sa.Column("album", sa.Text, nullable=True),
        sa.Column("release_year", sa.Integer, nullable=True),
        sa.Column("duration_ms", sa.BigInteger, nullable=True),
        sa.Column("norm_key", sa.String(512), index=True, nullable=False),
        sa.Column("variant_type", sa.String(24), nullable=False, server_default="original"),
        sa.Column("canonical_id", UUID(as_uuid=True),
                  sa.ForeignKey("catalog_tracks.id", ondelete="SET NULL"),
                  nullable=True, index=True),
        sa.Column("audio_track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("audio_match_method", sa.String(24), nullable=True),
        sa.Column("audio_match_score", sa.Float, nullable=True),
        sa.Column("enrichment", JSONB, nullable=True),
        sa.Column("play_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_ms_played", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("first_played_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_played_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "variant_links",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_a", UUID(as_uuid=True),
                  sa.ForeignKey("catalog_tracks.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("track_b", UUID(as_uuid=True),
                  sa.ForeignKey("catalog_tracks.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("method", sa.String(24), nullable=False),
        sa.Column("score", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("relation", sa.String(24), nullable=False, server_default="duplicate"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_a", "track_b", "method"),
    )

    op.create_table(
        "plays",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("catalog_track_id", UUID(as_uuid=True),
                  sa.ForeignKey("catalog_tracks.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ms_played", sa.Integer, nullable=False),
        sa.Column("platform", sa.Text, nullable=True),
        sa.Column("conn_country", sa.String(8), nullable=True),
        sa.Column("reason_start", sa.String(32), nullable=True),
        sa.Column("reason_end", sa.String(32), nullable=True),
        sa.Column("shuffle", sa.Boolean, nullable=True),
        sa.Column("skipped", sa.Boolean, nullable=True),
        sa.Column("offline", sa.Boolean, nullable=True),
        sa.Column("incognito", sa.Boolean, nullable=True),
        sa.Column("source_file", sa.Text, nullable=True),
    )

    op.create_table(
        "processing_state",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_track_id", UUID(as_uuid=True),
                  sa.ForeignKey("catalog_tracks.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("stage", sa.String(48), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("input_hash", sa.String(64), nullable=True),
        sa.Column("versions", JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("catalog_track_id", "stage"),
    )

    op.create_table(
        "low_level_features",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("spectral_centroid", sa.Float, nullable=False),
        sa.Column("spectral_bandwidth", sa.Float, nullable=False),
        sa.Column("spectral_rolloff", sa.Float, nullable=False),
        sa.Column("spectral_flatness", sa.Float, nullable=False),
        sa.Column("spectral_flux", sa.Float, nullable=False),
        sa.Column("spectral_entropy", sa.Float, nullable=False),
        sa.Column("rms_energy", sa.Float, nullable=False),
        sa.Column("dynamic_range_db", sa.Float, nullable=False),
        sa.Column("loudness_lufs", sa.Float, nullable=False),
        sa.Column("peak_amplitude", sa.Float, nullable=False),
        sa.Column("crest_factor", sa.Float, nullable=False),
        sa.Column("detail", JSONB, nullable=False),
        sa.Column("extractor_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "rhythm_analysis",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("bpm", sa.Float, nullable=False),
        sa.Column("tempo_confidence", sa.Float, nullable=False),
        sa.Column("meter", sa.String(8), nullable=True),
        sa.Column("time_signature", sa.Integer, nullable=False),
        sa.Column("groove_consistency", sa.Float, nullable=True),
        sa.Column("swing", sa.Float, nullable=True),
        sa.Column("detail", JSONB, nullable=False),
        sa.Column("extractor_version", sa.String(32), nullable=False),
    )

    op.create_table(
        "harmony_analysis",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("key", sa.Integer, nullable=False),
        sa.Column("mode", sa.Integer, nullable=False),
        sa.Column("key_confidence", sa.Float, nullable=False),
        sa.Column("harmonic_complexity", sa.Float, nullable=True),
        sa.Column("detail", JSONB, nullable=False),
        sa.Column("extractor_version", sa.String(32), nullable=False),
    )

    op.create_table(
        "structure_analysis",
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("sections", JSONB, nullable=False),
        sa.Column("section_repetition", sa.Float, nullable=True),
        sa.Column("structural_similarity", sa.Float, nullable=True),
        sa.Column("boundaries", JSONB, nullable=False),
        sa.Column("extractor_version", sa.String(32), nullable=False),
    )

    op.create_table(
        "semantic_predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("head", sa.String(24), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "head", "model_version"),
    )

    op.create_table(
        "model_embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("model_name", sa.String(48), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("vector", sa.LargeBinary, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "model_name", "model_version"),
    )


def downgrade() -> None:
    for t in (
        "model_embeddings", "semantic_predictions", "structure_analysis",
        "harmony_analysis", "rhythm_analysis", "low_level_features",
        "processing_state", "plays", "variant_links", "catalog_tracks",
    ):
        op.drop_table(t)
