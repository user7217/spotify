FROM python:3.11-slim AS base
WORKDIR /srv/extractor
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libpq-dev gcc g++ git && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2" scipy librosa soundfile \
    sqlalchemy "psycopg[binary]" pgvector aiokafka minio structlog \
    pydantic pydantic-settings cython
# madmom needs cython at build time; essentia optional (linux x86 wheels)
RUN pip install "madmom @ git+https://github.com/CPJKU/madmom.git" || true
RUN pip install essentia-tensorflow || true

COPY services/api/app /srv/api/app
COPY services/extractor/app /srv/extractor/app

CMD ["python", "-m", "app.worker"]
