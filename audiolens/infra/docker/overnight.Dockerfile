# Single self-contained image for the overnight personal build (SQLite, no
# Postgres/Kafka/MinIO). Bundles the DSP extractor deps + yt-dlp + ffmpeg and
# runs scripts/run_overnight.sh end to end.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 gcc g++ git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# JavaScript runtime for yt-dlp. YouTube scrambles stream URLs with a JS
# "signature / n" challenge; without a JS engine yt-dlp drops every format
# ("Requested format is not available"). Deno is yt-dlp's recommended runtime.
ENV DENO_INSTALL=/usr/local
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -y \
    && deno --version

# DSP + ingest + clients. (No torch/essentia: run_pipeline degrades to DSP-only
# documents when the ML stack is absent — exactly what we want for this batch.)
# Pin a numpy/numba/llvmlite trio that is KNOWN ABI-compatible. Leaving numba
# unpinned lets pip pull a build compiled against a different numpy than the
# pinned numpy<2 — its guvectorize ufuncs then SEGFAULT (librosa chroma_cqt ->
# piptrack). This exact trio matches librosa 0.10.x.
RUN pip install \
        "numpy==1.26.4" "numba==0.60.0" "llvmlite==0.43.0" \
        "librosa==0.10.2.post1" scipy soundfile pyloudnorm \
        httpx yt-dlp mutagen \
        sqlalchemy pydantic pydantic-settings structlog cython
# madmom sharpens tempo but needs cython at build time; optional.
RUN pip install "madmom @ git+https://github.com/CPJKU/madmom.git" || true

# yt-dlp breaks whenever YouTube changes; keep it in its own late layer and
# always pull the newest release. Bump YTDLP_REFRESH (or build --no-cache) to
# force a fresh yt-dlp when downloads start 403ing again.
ARG YTDLP_REFRESH=2026-07-01
RUN pip install --upgrade --force-reinstall yt-dlp

# Repo code (services/ + scripts/). Mounts at runtime supply data + work dir.
COPY services /app/services
COPY scripts  /app/scripts
RUN chmod +x /app/scripts/run_overnight.sh

# Long-running batch; no healthcheck needed. Logs to stdout (docker logs -f).
ENTRYPOINT ["/app/scripts/run_overnight.sh"]
