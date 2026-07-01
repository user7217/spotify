FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
WORKDIR /srv/embedder
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libpq-dev gcc && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2" librosa soundfile pandas pytorch-lightning \
    faiss-cpu scikit-learn sqlalchemy "psycopg[binary]" pgvector \
    aiokafka minio structlog pydantic pydantic-settings wandb

COPY services/api/app /srv/api/app
COPY services/embedder/app /srv/embedder/app

CMD ["python", "-m", "app.inference.worker"]
