"""API schemas. Feature/analysis shapes mirror Spotify's deprecated endpoints
so existing client code can be pointed at AudioLens with minimal changes."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── audio features ────────────────────────────────────────────────────────────

class AudioFeaturesOut(BaseModel):
    track_id: uuid.UUID
    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    speechiness: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    liveness: float = Field(ge=0, le=1)
    loudness: float
    tempo: float
    key: int = Field(ge=-1, le=11)
    mode: int = Field(ge=0, le=1)
    time_signature: int
    extractor_version: str

    model_config = {"from_attributes": True}


# ── audio analysis ────────────────────────────────────────────────────────────

class TimeInterval(BaseModel):
    start: float
    duration: float
    confidence: float


class Section(TimeInterval):
    loudness: float
    tempo: float
    tempo_confidence: float
    key: int
    key_confidence: float
    mode: int
    mode_confidence: float
    time_signature: int
    time_signature_confidence: float


class Segment(TimeInterval):
    loudness_start: float
    loudness_max: float
    loudness_max_time: float
    loudness_end: float
    pitches: list[float] = Field(min_length=12, max_length=12)
    timbre: list[float] = Field(min_length=12, max_length=12)


class AudioAnalysisOut(BaseModel):
    track_id: uuid.UUID
    meta: dict
    track: dict  # global summary block (Spotify naming)
    bars: list[TimeInterval]
    beats: list[TimeInterval]
    tatums: list[TimeInterval]
    sections: list[Section]
    segments: list[Segment]
    harmony: dict | None = None
    rhythm: dict | None = None

    model_config = {"from_attributes": True}


# ── tracks / jobs ─────────────────────────────────────────────────────────────

class TrackOut(BaseModel):
    id: uuid.UUID
    file_hash: str
    spotify_track_id: str | None
    title: str | None
    artist: str | None
    album: str | None
    duration_ms: int | None
    format: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class JobOut(BaseModel):
    id: uuid.UUID
    track_id: uuid.UUID | None
    status: str
    stage: str | None
    error: str | None
    batch_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    track: TrackOut
    job: JobOut
    deduplicated: bool = False


class BatchUploadResponse(BaseModel):
    batch_id: uuid.UUID
    jobs: list[JobOut]


# ── similarity ────────────────────────────────────────────────────────────────

class SimilarTrack(BaseModel):
    track: TrackOut
    distance: float


class SimilarityResponse(BaseModel):
    query_track_id: uuid.UUID
    model_version: str
    results: list[SimilarTrack]
