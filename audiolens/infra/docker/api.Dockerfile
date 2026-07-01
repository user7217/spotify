FROM python:3.11-slim AS base
WORKDIR /srv/api
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /srv/
RUN pip install fastapi "uvicorn[standard]" pydantic pydantic-settings \
    sqlalchemy alembic "psycopg[binary]" pgvector redis aiokafka minio \
    httpx "numpy<2" structlog python-multipart

COPY services/api/app /srv/api/app
COPY migrations /srv/api/migrations
COPY alembic.ini /srv/api/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
