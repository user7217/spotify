"""Central configuration. All services read from env vars via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # service
    env: str = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # postgres
    pg_host: str = "postgres"
    pg_port: int = 5432
    pg_user: str = "audiolens"
    pg_password: str = "audiolens"
    pg_db: str = "audiolens"

    # redis
    redis_url: str = "redis://redis:6379/0"

    # kafka
    kafka_bootstrap: str = "kafka:9092"
    kafka_topic_jobs: str = "audiolens.analysis.jobs"
    kafka_topic_results: str = "audiolens.analysis.results"
    kafka_group_extractor: str = "extractor-workers"

    # object storage
    s3_endpoint: str = "minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_audio: str = "audio"
    s3_bucket_artifacts: str = "artifacts"
    s3_secure: bool = False

    # audio processing
    sample_rate: int = 22050
    max_upload_mb: int = 100
    supported_formats: tuple[str, ...] = ("mp3", "flac", "wav", "aac", "m4a", "ogg")

    # embedding
    embedding_dim: int = 128
    faiss_index_path: str = "/data/faiss/songs.index"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    @property
    def pg_dsn_sync(self) -> str:
        return self.pg_dsn


@lru_cache
def get_settings() -> Settings:
    return Settings()
