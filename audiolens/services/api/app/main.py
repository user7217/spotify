"""AudioLens API entrypoint."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import v1

settings = get_settings()
log = structlog.get_logger()

app = FastAPI(
    title="AudioLens",
    description=(
        "Open-source replacement for Spotify's deprecated Audio Features "
        "and Audio Analysis APIs. Upload audio, get features, time-series "
        "analysis, and learned embeddings."
    ),
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.on_event("shutdown")
async def shutdown():
    from app.routers.v1 import _producer

    if _producer:
        await _producer.stop()
