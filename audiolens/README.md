# AudioLens

Open-source replacement for Spotify's deprecated **Audio Features** and **Audio Analysis** APIs, plus a learned song embedding model for similarity search, mood clustering, and recommendation.

## Architecture

```
                       ┌──────────────┐
 audio upload ───────► │   API (FastAPI)│──► Postgres + pgvector
                       └──────┬───────┘         ▲
                              │ job              │ results
                              ▼                  │
                       ┌──────────────┐          │
                       │    Kafka     │          │
                       └──────┬───────┘          │
              ┌───────────────┴──────────────┐   │
              ▼                              ▼   │
     ┌─────────────────┐            ┌─────────────────┐
     │ Extractor worker│            │ Embedder worker │
     │ librosa/madmom/ │            │ PyTorch encoder │
     │ essentia        │            │ + FAISS index   │
     └────────┬────────┘            └────────┬────────┘
              │      audio from MinIO/S3     │
              └───────────────┬──────────────┘
                              ▼
                       ┌──────────────┐
                       │  MinIO / S3  │
                       └──────────────┘
```

## Components

| Service | Role |
|---|---|
| `services/api` | REST API, upload, dedupe, job dispatch, similarity search (pgvector HNSW) |
| `services/extractor` | DSP pipeline: 12 audio features + full time-series analysis |
| `services/embedder` | Contrastive encoder training + inference; FAISS batch index |

## Feature replication strategy

| Feature | Method |
|---|---|
| tempo | librosa beat tracker, madmom DBN refinement |
| key / mode | Krumhansl-Schmuckler profiles on CQT chroma |
| loudness | gated RMS dBFS (pre-normalization) |
| energy | RMS + spectral flux + onset density (gain-invariant) |
| danceability | beat regularity + tempo prior + pulse clarity; Essentia model override |
| valence | Essentia MusiCNN emomusic model; mode+brightness fallback |
| acousticness | rolloff/centroid/flatness; Essentia mood_acoustic override |
| speechiness | ZCR variance + 4 Hz syllabic modulation + flatness |
| instrumentalness | harmonic vocal-band ratio; Essentia voice model override |
| liveness | spectral flatness in quiet passages (crowd/room noise) |
| time signature | beat-strength autocorrelation over meter candidates |

Analysis output mirrors Spotify's shape: `bars`, `beats`, `tatums`, `sections`, `segments` (with 12-dim `pitches` + 12-dim `timbre`), plus extended `harmony` (per-beat chords, harmonic change rate) and `rhythm` (syncopation, rhythmic entropy, beat regularity) blocks.

## Embedding model

CNN frontend → Transformer → attention pooling → 128-dim L2-normalized vector.

Training (PyTorch Lightning, DDP-ready):
- **Contrastive (NT-Xent)** — augmented crops of the same track as positives
- **Multi-task regression** — predict the 7 perceptual features from the embedding
- Augmentations: random crop, gain, noise, time-stretch, pitch-shift, SpecAugment

Before a checkpoint exists, the embedder produces a deterministic feature-vector embedding (`featvec-v1`), so `/similar` works from day one.

## Catalog pipeline (streaming history → analyzed library)

Turns a Spotify extended-streaming-history export into a fully analyzed,
deduplicated catalog:

```
sp_data/*.json ──► ingest (parse, dedup, variant linking)      services/ingest
               ──► enrich (Spotify API ISRC/year, MusicBrainz) services/ingest/enrich
               ──► resolve (match to YOUR local audio files)   services/ingest/resolve
               ──► analyze (DSP + embeddings + classifiers)    scripts/run_pipeline.py
               ──► index   (FAISS + HNSW per model)            embedder/inference/indexes
```

```bash
make history DIR=../sp_data          # -> data/catalog.db + catalog.jsonl
export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
make enrich                          # ISRC, duration, release year + ISRC dedup pass
make resolve LIBRARY=~/Music         # fingerprint/metadata match to owned audio
make pipeline                        # parallel, resumable, cached; GPU auto
make indexes                         # FAISS + HNSW similarity indexes
```

Dedup identity order: Spotify track ID → ISRC → normalized artist+title →
audio fingerprint. Remasters / live / radio edits / deluxe / explicit-clean
versions stay as separate rows linked to a canonical track (`canonical_id` +
`variant_links` with typed relations).

Per-track output document (`data/track_docs/<id>.json`):
`{track_id, metadata, audio_features, rhythm, harmony, structure, genre,
mood, instruments, vocals, production, embeddings}` — see `docs/` and the
`low_level_features`/`rhythm_analysis`/`harmony_analysis`/`structure_analysis`/
`semantic_predictions`/`model_embeddings` tables (migration 0002).

Note: the resolver only matches audio files you already own — the pipeline
contains no track downloading.

### No local library? Preview-clip analysis

`scripts/analyze_previews.py` analyzes every catalog track from 30-second
iTunes preview clips (public API, no key):

```bash
pip install librosa soundfile pyloudnorm
# ffmpeg required on PATH
python scripts/analyze_previews.py --sqlite data/catalog.db            # all ~20k
python scripts/analyze_previews.py --sqlite data/catalog.db --limit 1000  # top played first
```

Resumable (re-run any time), previews cached, parallel DSP. Expect ~17-20 h
for a full 20k-track run — the iTunes search rate limit (~20/min) is the
bottleneck, not the DSP. Results carry `source=itunes_preview_30s`; BPM,
key, spectral and mood metrics are representative, but structure describes
the 30s excerpt, not the full song.

## Quickstart

```bash
make models          # download Essentia pretrained models (optional, improves accuracy)
make up              # full stack: postgres, kafka, minio, api, 2x extractor
make test            # unit tests (13 tests, synthetic audio)

# upload a track
curl -F "file=@song.mp3" http://localhost:8000/v1/tracks

# poll job, then:
curl http://localhost:8000/v1/audio-features/{track_id}
curl http://localhost:8000/v1/audio-analysis/{track_id}
curl http://localhost:8000/v1/tracks/{track_id}/similar?k=10

# bulk catalog ingestion
python scripts/bulk_ingest.py ~/Music
```

OpenAPI docs: `http://localhost:8000/docs`

## Training the encoder

```bash
# 1. ingest your catalog (extractor produces features per track)
# 2. export features to CSV: path,danceability,energy,...
# 3. train
python -m app.training.train \
  --data-dir /data/audio --features-csv features.csv \
  --epochs 100 --batch-size 64 --devices 4   # DDP across 4 GPUs

# 4. drop checkpoint at /models/encoder.ckpt — embedder picks it up,
#    new tracks get encoder-v1 embeddings
```

## Deploy

- `infra/k8s/` — Deployments, HPA for the API, KEDA Kafka-lag autoscaling for extractors
- `infra/terraform/` — S3, RDS Postgres 16, MSK Serverless, EKS with dedicated extraction node group
- `.github/workflows/ci.yml` — lint, test, multi-service image build to GHCR

## Repo layout

```
services/
  api/app/            FastAPI app, routers, schemas, SQLAlchemy models
  extractor/app/      features/ analysis/ audio/ + Kafka worker
  embedder/app/       models/ training/ inference/
migrations/           Alembic (hand-written, pgvector + HNSW index)
infra/                docker/ k8s/ terraform/
scripts/              model download, bulk ingestion
tests/                unit (synthetic audio) + integration (live stack)
```
