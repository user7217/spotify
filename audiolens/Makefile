.PHONY: up down test lint migrate models ingest history enrich resolve pipeline indexes

up:
	docker compose up -d --build

down:
	docker compose down -v

test:
	python -m pytest tests/unit -q

test-integration:
	python -m pytest tests/integration -q -m integration

lint:
	ruff check services tests

migrate:
	docker compose run --rm migrate

models:
	./scripts/download_models.sh ./models

ingest:
	python scripts/bulk_ingest.py $(DIR)

# --- catalog pipeline (streaming-history -> analyzed documents) -----------
DB ?= data/catalog.db

history:                       # parse + dedup streaming history JSON
	python -m services.ingest.app.cli $(DIR) --sqlite $(DB) --export data/catalog.jsonl

enrich:                        # Spotify API: ISRC/duration/year (needs SPOTIFY_CLIENT_ID/SECRET)
	python -m services.ingest.app.enrich.spotify --sqlite $(DB)
	python -m services.ingest.app.enrich.musicbrainz --sqlite $(DB)

resolve:                       # match catalog tracks to local audio library
	python -m services.ingest.app.resolve.library --sqlite $(DB) --library $(LIBRARY)

pipeline:                      # full DSP + embeddings + classification -> track docs
	python scripts/run_pipeline.py --sqlite $(DB) --docs-dir data/track_docs

indexes:                       # FAISS + HNSW per embedding model
	python -m services.embedder.app.inference.indexes --sqlite $(DB) --out data/indexes
