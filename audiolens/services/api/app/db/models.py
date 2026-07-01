"""Database models.

Schema design:
- tracks: canonical track registry (file hash dedupes uploads; spotify_track_id optional)
- audio_features: 1:1 with track, Spotify Audio Features replacement
- audio_analysis: 1:1 with track, large JSONB blobs for segments/beats/etc.
- embeddings: pgvector column, one row per (track, model_version)
- analysis_jobs: async job tracking for the Kafka pipeline
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSONB, list: JSONB}


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # sha256 of audio bytes — dedupe key
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    spotify_track_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    artist: Mapped[str | None] = mapped_column(Text, nullable=True)
    album: Mapped[str | None] = mapped_column(Text, nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    format: Mapped[str | None] = mapped_column(String(8), nullable=True)
    s3_key: Mapped[str] = mapped_column(Text)  # object storage location

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    features: Mapped["AudioFeatures | None"] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan"
    )
    analysis: Mapped["AudioAnalysis | None"] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["Embedding"]] = relationship(
        back_populates="track", cascade="all, delete-orphan"
    )


class AudioFeatures(Base):
    """Spotify Audio Features replacement — one row per track."""

    __tablename__ = "audio_features"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )

    danceability: Mapped[float] = mapped_column(Float)
    energy: Mapped[float] = mapped_column(Float)
    valence: Mapped[float] = mapped_column(Float)
    speechiness: Mapped[float] = mapped_column(Float)
    acousticness: Mapped[float] = mapped_column(Float)
    instrumentalness: Mapped[float] = mapped_column(Float)
    liveness: Mapped[float] = mapped_column(Float)
    loudness: Mapped[float] = mapped_column(Float)  # dB
    tempo: Mapped[float] = mapped_column(Float)  # BPM
    key: Mapped[int] = mapped_column(Integer)  # 0-11 pitch class, -1 unknown
    mode: Mapped[int] = mapped_column(Integer)  # 1 major, 0 minor
    time_signature: Mapped[int] = mapped_column(Integer)

    # provenance: which extractor/model versions produced these values
    extractor_version: Mapped[str] = mapped_column(String(32))
    model_confidences: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    track: Mapped[Track] = relationship(back_populates="features")


class AudioAnalysis(Base):
    """Spotify Audio Analysis replacement.

    Large time-series blobs stored as JSONB. Each follows the Spotify shape:
      beats/bars/tatums: [{start, duration, confidence}]
      sections: [{start, duration, loudness, tempo, key, mode, time_signature, confidence}]
      segments: [{start, duration, loudness_start, loudness_max, loudness_max_time,
                  pitches[12], timbre[12], confidence}]
    """

    __tablename__ = "audio_analysis"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )

    meta: Mapped[dict] = mapped_column(JSONB)  # sample_rate, analysis duration, versions
    track_summary: Mapped[dict] = mapped_column(JSONB)  # global tempo/key/loudness block

    bars: Mapped[list] = mapped_column(JSONB)
    beats: Mapped[list] = mapped_column(JSONB)
    tatums: Mapped[list] = mapped_column(JSONB)
    sections: Mapped[list] = mapped_column(JSONB)
    segments: Mapped[list] = mapped_column(JSONB)

    # extended (beyond Spotify): harmonic + rhythm analysis
    harmony: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    rhythm: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    extractor_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    track: Mapped[Track] = relationship(back_populates="analysis")


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("track_id", "model_version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64))
    vector: Mapped[list[float]] = mapped_column(Vector(128))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    track: Mapped[Track] = relationship(back_populates="embeddings")


# --------------------------------------------------------------------------
# Catalog layer (streaming-history ingestion, dedup, enrichment, resolution)
# --------------------------------------------------------------------------


class CatalogTrack(Base):
    """One logical track from listening history (deduplicated).

    Identity resolution order: spotify_track_id -> isrc -> normalized
    artist+title -> audio fingerprint (once audio is resolved).
    `canonical_id` groups variants: remasters, live, radio edits, etc.
    point at the canonical recording; canonical rows have canonical_id NULL.
    """

    __tablename__ = "catalog_tracks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    spotify_track_id: Mapped[str | None] = mapped_column(
        String(32), unique=True, index=True, nullable=True
    )
    isrc: Mapped[str | None] = mapped_column(String(15), index=True, nullable=True)

    title: Mapped[str] = mapped_column(Text)
    artists: Mapped[list] = mapped_column(JSONB, default=list)
    album: Mapped[str | None] = mapped_column(Text, nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # normalization keys used for dedup
    norm_key: Mapped[str] = mapped_column(String(512), index=True)  # artist|title normalized
    variant_type: Mapped[str] = mapped_column(String(24), default="original")
    # original|remaster|live|radio_edit|deluxe|explicit|clean|acoustic|remix|extended|demo
    canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalog_tracks.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # resolved audio asset (NULL until audio located in user library)
    audio_track_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    audio_match_method: Mapped[str | None] = mapped_column(String(24), nullable=True)
    # fingerprint | metadata | manual
    audio_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    enrichment: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # raw API payloads
    play_count: Mapped[int] = mapped_column(Integer, default=0)
    total_ms_played: Mapped[int] = mapped_column(BigInteger, default=0)
    first_played_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_played_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    variants: Mapped[list["CatalogTrack"]] = relationship(remote_side=[id])


class VariantLink(Base):
    """Pairwise evidence that two catalog tracks are the same recording/work."""

    __tablename__ = "variant_links"
    __table_args__ = (UniqueConstraint("track_a", "track_b", "method"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_a: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_tracks.id", ondelete="CASCADE"), index=True
    )
    track_b: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_tracks.id", ondelete="CASCADE"), index=True
    )
    method: Mapped[str] = mapped_column(String(24))  # isrc|spotify_id|name|fingerprint
    score: Mapped[float] = mapped_column(Float, default=1.0)
    relation: Mapped[str] = mapped_column(String(24), default="duplicate")
    # duplicate | remaster_of | live_of | edit_of | clean_of | remix_of
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Play(Base):
    """One playback event from streaming history."""

    __tablename__ = "plays"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    catalog_track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_tracks.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ms_played: Mapped[int] = mapped_column(Integer)
    platform: Mapped[str | None] = mapped_column(Text, nullable=True)
    conn_country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    reason_start: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason_end: Mapped[str | None] = mapped_column(String(32), nullable=True)
    shuffle: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    skipped: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    offline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    incognito: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    source_file: Mapped[str | None] = mapped_column(Text, nullable=True)


class ProcessingState(Base):
    """Per-(catalog track, stage) checkpoint — resume + cache layer.

    Stages: enrich | resolve | dsp | rhythm | harmony | structure |
            embed:<model> | classify:<head> | index | document
    """

    __tablename__ = "processing_state"
    __table_args__ = (UniqueConstraint("catalog_track_id", "stage"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    catalog_track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_tracks.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # pending | running | done | failed | skipped
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # reproducibility
    versions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # lib/model versions
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------
# Feature store (keyed on the audio asset; catalog joins via audio_track_id)
# --------------------------------------------------------------------------


class LowLevelFeatures(Base):
    """Scalar spectral/energy/frequency/pitch summary stats; full matrices in `detail`."""

    __tablename__ = "low_level_features"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    # spectral (mean/std stored in detail; means here for fast filtering)
    spectral_centroid: Mapped[float] = mapped_column(Float)
    spectral_bandwidth: Mapped[float] = mapped_column(Float)
    spectral_rolloff: Mapped[float] = mapped_column(Float)
    spectral_flatness: Mapped[float] = mapped_column(Float)
    spectral_flux: Mapped[float] = mapped_column(Float)
    spectral_entropy: Mapped[float] = mapped_column(Float)
    # energy
    rms_energy: Mapped[float] = mapped_column(Float)
    dynamic_range_db: Mapped[float] = mapped_column(Float)
    loudness_lufs: Mapped[float] = mapped_column(Float)
    peak_amplitude: Mapped[float] = mapped_column(Float)
    crest_factor: Mapped[float] = mapped_column(Float)
    # detail: spectral_contrast bands, mfcc/delta/delta2 stats, mel stats,
    #         chroma/hpcp/tonnetz stats, pitch histogram
    detail: Mapped[dict] = mapped_column(JSONB)
    extractor_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RhythmAnalysis(Base):
    __tablename__ = "rhythm_analysis"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    bpm: Mapped[float] = mapped_column(Float)
    tempo_confidence: Mapped[float] = mapped_column(Float)
    meter: Mapped[str | None] = mapped_column(String(8), nullable=True)
    time_signature: Mapped[int] = mapped_column(Integer)
    groove_consistency: Mapped[float | None] = mapped_column(Float, nullable=True)
    swing: Mapped[float | None] = mapped_column(Float, nullable=True)
    # detail: tempo_curve, beats[{t, strength}], downbeats[], per-beat info
    detail: Mapped[dict] = mapped_column(JSONB)
    extractor_version: Mapped[str] = mapped_column(String(32))


class HarmonyAnalysis(Base):
    __tablename__ = "harmony_analysis"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[int] = mapped_column(Integer)  # 0-11, -1 unknown
    mode: Mapped[int] = mapped_column(Integer)  # 1 major 0 minor
    key_confidence: Mapped[float] = mapped_column(Float)
    harmonic_complexity: Mapped[float | None] = mapped_column(Float, nullable=True)
    # detail: modulations[], chord_progression[], chord_transition_matrix,
    #         method provenance (ks|deep), per-beat chroma/hpcp summary
    detail: Mapped[dict] = mapped_column(JSONB)
    extractor_version: Mapped[str] = mapped_column(String(32))


class StructureAnalysis(Base):
    __tablename__ = "structure_analysis"

    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    sections: Mapped[list] = mapped_column(JSONB)  # [{start,end,label,confidence}]
    section_repetition: Mapped[float | None] = mapped_column(Float, nullable=True)
    structural_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    boundaries: Mapped[list] = mapped_column(JSONB)  # segment boundary times
    extractor_version: Mapped[str] = mapped_column(String(32))


class SemanticPredictions(Base):
    """Model-head outputs: genre, mood, instruments, vocals, production."""

    __tablename__ = "semantic_predictions"
    __table_args__ = (UniqueConstraint("track_id", "head", "model_version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), index=True
    )
    head: Mapped[str] = mapped_column(String(24), index=True)
    # genre | mood | instruments | vocals | production
    model_version: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelEmbedding(Base):
    """Variable-dimension embeddings, one row per (track, model).

    Stored as float32 bytes (pgvector needs a fixed dim per column).
    The unified 128-d similarity vector stays in `embeddings` (pgvector+HNSW);
    FAISS/HNSW file indexes are built from these rows per model.
    """

    __tablename__ = "model_embeddings"
    __table_args__ = (UniqueConstraint("track_id", "model_name", "model_version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(48), index=True)
    # openl3 | clap | music2vec | musicfm | discogs-effnet | musicnn | byola | wav2vec2
    model_version: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32 little-endian
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), default="queued", index=True
    )  # queued | running | done | failed
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_outputs: Mapped[list] = mapped_column(
        JSON, default=lambda: ["features", "analysis", "embedding"]
    )
    batch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
