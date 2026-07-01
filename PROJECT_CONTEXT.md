# Project Context — Personal Music–Mind Correlation System

> Living context document for the Cowork project. Captures vision, decisions, architecture,
> the audio-analytics problem and its resolution, infrastructure, and open questions.
> Read this first before working on any part of the system.

---

## 1. What this project actually is

A continuously running personal service (stats.fm-style) that watches what I listen to on
Spotify and builds a model linking **music** to **my mental/behavioral state** — when I listen,
what I'm doing, how I feel, and what I'm reaching for.

The goal is **not** a general music model. It is a **personal correlation model**:

> "When I'm in state X, what properties in music am I reaching for — and why does that specific
> thing work for my brain?"

Two songs are "similar" in this system if they **work for me in the same mental context** — not
if they sound alike. A death-metal track and a lo-fi track might both work at 2am during
hyperfocus: acoustically opposite, contextually identical. The model must learn that.

### ADHD-specific framing
Standard emotion models use valence (happy↔sad) + energy (calm↔intense). For my brain, the more
useful axes are:
- **Arousal regulation** — calming ↔ stimulating
- **Cognitive load** — background ↔ distracting

Focus is on **what music *does to me*, not what it "means."**

---

## 2. The deeper question driving the model

Not just "what song" but **"which part of a song, and why, given my state."**
Candidate axes that may carry the signal:
- **Rhythmic entrainment** — beat regularity / IBI consistency → external pacemaker for attention
- **Arousal trajectory** — energy arc across a song → regulates internal arousal
- **Cognitive load** — speechiness, timbre complexity, harmonic change rate → competes for attention
- **Timbral texture** — brightness/warmth (MFCCs, spectral centroid) → mood without cognitive demand
- **Harmonic tension** — dissonance, key stability, unresolved progressions → subconscious anxiety
- **Predictability** — rhythmic entropy, structural repetition → mental energy spent anticipating

These live in the **time-series analysis** of a track (how segments evolve), not in single scalar
features. This is the core reason the project needs per-track audio *analysis*, not just features.

---

## 3. Data sources

### 3a. Spotify Extended Streaming History (HAVE IT — 2019→present)
Files: `Streaming_History_Audio_YYYY[_N].json`. Per-event fields include:
`ts, ms_played, master_metadata_track_name, master_metadata_album_artist_name,
master_metadata_album_album_name, spotify_track_uri, reason_start, reason_end,
shuffle, skipped, offline, conn_country`.

**Key insight: this data is already implicitly labeled.** Behavioral signals carry mental state
without any self-reporting:
- `ms_played / duration` → completion rate (did it work?)
- `skipped = true` / `reason_end = "fwdbtn"` → actively rejected for that moment
- replay in same session → strong positive signal
- session length → depth of state
- time of day + completion → contextual pattern

7 years of history = a large, pre-labeled behavioral dataset.

### 3b. Live currently-playing (TO BUILD)
Poll Spotify `/me/player/currently-playing` (~every 20s) on the always-on server. Log each new
track immediately with context (timestamp, etc.), then fill in analysis asynchronously.

### 3c. Audio analysis per track (THE HARD PROBLEM — see §5)

---

## 4. What's already built — the "AudioLens" repo

A production-grade, open-source replacement for Spotify's deprecated Audio Features + Audio
Analysis APIs. **13/13 unit tests pass against real librosa.** Three independent components:

### 4a. Feature Extractor (`services/extractor/app/features/`)
Pure DSP, no ML, deterministic. Audio → 12 numbers.
| Feature | Method |
|---|---|
| tempo | librosa beat tracker + madmom DBN refinement |
| key / mode | Krumhansl-Schmuckler profiles on CQT chroma |
| loudness | gated RMS dBFS (computed pre-normalization) |
| energy | RMS + spectral flux + onset density (gain-invariant) |
| danceability | beat regularity + tempo prior + pulse clarity |
| acousticness | spectral rolloff/centroid/flatness |
| speechiness | ZCR variance + 4Hz syllabic modulation |
| instrumentalness | harmonic vocal-band (200Hz–4kHz) ratio |
| valence | mode + brightness + energy (weakest; Essentia model improves it) |
| liveness | spectral flatness in quiet passages |
| time_signature | beat-strength autocorrelation over meter candidates |

Essentia pretrained models (danceability, valence, voice/instrumental, acousticness) override
heuristics when present; heuristics always run as fallback.

### 4b. Analysis Extractor (`services/extractor/app/analysis/`)
Pure DSP. Audio → time-series structure, Spotify-shaped:
- `beats`, `bars`, `tatums` — `[{start, duration, confidence}]`
- `sections` — per-section tempo/key/mode/loudness
- `segments` — `[{start, duration, pitches[12], timbre[12], loudness_*}]`
  - `pitches[12]` = chroma (which notes active) ; `timbre[12]` = MFCC texture/brightness/attack
- Extended beyond Spotify: `harmony` (per-beat chords, harmonic change rate),
  `rhythm` (beat regularity, syncopation, rhythmic entropy)

**This is the component that answers "how does the song progress" — the data the deep model needs.**

### 4c. Embedding Model (`services/embedder/app/models/encoder.py`)
The only ML piece. CNN frontend → Transformer (4 layers) → attention pooling → 128-dim
L2-normalized vector. Trained with NT-Xent contrastive (augmented crops of same track = positives)
+ multi-task feature regression. **No trained weights yet** — until a checkpoint exists, a
deterministic fallback projects (features + analysis stats) → 128-dim so similarity search works
from day one.

### 4d. Surrounding system
- **API** (FastAPI): upload (single/batch), sha256 dedupe, `/audio-features/{id}`,
  `/audio-analysis/{id}`, `/similar` (pgvector HNSW cosine KNN), job tracking
- **Pipeline**: Kafka job queue, MinIO/S3 audio storage, extractor + embedder workers, horizontal
  scaling via consumer groups
- **DB**: Postgres 16 + pgvector; Alembic migration with HNSW index
- **Infra**: docker-compose (full local stack), K8s (HPA + KEDA Kafka-lag autoscale), Terraform
  (S3/RDS/MSK/EKS), GitHub Actions CI
- **Scripts**: Essentia model downloader, bulk catalog ingestion

---

## 5. THE CORE PROBLEM — getting audio analytics

Spotify deprecated both the Audio Features and Audio Analysis APIs. You cannot get analysis from a
track ID anymore. Resolution requires matching a track to an audio source. This is the project's
central engineering problem.

### 5a. Decision: hybrid strategy (ReccoBeats + own extractor)

**ReccoBeats** (https://reccobeats.com) — free hosted API, verified June 2026:
- `GET /v1/track/:id/audio-features` — **lookup by Spotify track ID, no audio download.** Instant,
  free. The big win — kills the yt-dlp problem for tracks in their DB (millions of tracks).
- `POST /v1/analysis/audio-features` — upload audio file → features. 30s limit per file; for longer
  tracks, split into chunks and average.
- Returns **9 features**: acousticness, danceability, energy, instrumentalness, liveness, loudness,
  speechiness, tempo, valence. **Missing vs Spotify: key, mode, time_signature** (own extractor
  computes these trivially if needed).
- Free; rate-limited internally (429 + `Retry-After` header on exceed; exact limit unpublished).
- **CRITICAL GAP: ReccoBeats provides Audio Features ONLY. It does NOT provide Audio Analysis**
  (no segments, beats, bars, timbre/pitch arc). No hosted API provides analysis — that's why the
  extractor exists.

### 5b. The resulting hybrid flow
```
Currently-playing track detected
        │
        ▼
  ReccoBeats lookup by track_id  ──► 9 features (instant, free, no download)
        │
        ▼
  need segment-level analysis too?
        │
   yes ─┴─ no
    │        └──► store features, done
    ▼
  yt-dlp + own Analysis Extractor  ──► segments / beats / timbre arc / progression
```

- **Features** → ReccoBeats by track ID (primary). yt-dlp + own extractor only when a track isn't
  in their DB.
- **Analysis (segments / timbre / progression)** → always own extractor + yt-dlp. No alternative.
- Run the heavy extractor only on **top / most-played tracks** where deep analysis actually matters;
  let ReccoBeats cover the long tail of features.

### 5c. Consistency warning
ReccoBeats features come from their own models, not Spotify's — values won't be identical to old
Spotify numbers. **Pick ONE source per feature and stay consistent** across the dataset, or the
correlation model will mix incompatible scales. Sanity-check a few known tracks before committing.

### 5d. yt-dlp resolution details (for the analysis path)
- Search `"{track} {artist} audio"`, download best audio, convert to mp3.
- **Validate downloaded duration against Spotify's known `duration_ms`** (>15s mismatch → wrong
  video: live/cover/remix/loop → flag or retry). This single check catches most mismatches.
- Store the audio `source` (`reccobeats` / `yt-dlp-full` / `spotify-preview-30s`) per track so it's
  known whether analysis is full or partial.
- **Caching is the whole point:** new tracks need a download once; after that it's all cache hits.
  Heavy at first, near-zero in steady state.
- Legal note: yt-dlp downloads are a personal-use grey area; keep it personal/local.

---

## 6. Embedding-model design philosophy (how to actually proceed)

The architecture is the LAST decision, not the first. Order:
1. **Define "similar"** = listened-to-in-the-same-mental-context (already decided, §1).
2. **EDA first — prove signal exists before modeling.** Run on streaming history by hand:
   - `df.groupby('hour')['ms_played'].sum()` — when do I listen longest?
   - `df.groupby('hour')['skipped'].mean()` — when am I most selective?
   - completion rate by hour: `(ms_played > 120_000).groupby(hour).mean()`
   - session "modes" — do different artists appear in long vs short sessions?
   - **If simple stats show no pattern, no neural net will help.**
3. **Dumbest model first** — logistic regression: predict `completed` from
   `[hour, day_of_week, prev_track_skipped, session_position]`. Tells you if signal is learnable
   and which context features matter.
4. **Add music features** — see which improve prediction. Now you're measuring what the model needs
   to encode, not guessing.
5. **Embedding emerges from the predictor** — train a small MLP
   `[music features + context] → completion probability`; the **middle layer IS the embedding**.
   You don't design the embedding; you define the task and the network learns the representation.
6. **Only then** scale up to segment-level Transformer encoding / dual-encoder.

### Eventual target architecture — dual encoder
```
Music segment  ──► Music Encoder   ──► 128-dim ──┐
                                                  ├──► trained so paired (segment, context)
Context        ──► Context Encoder ──► 128-dim ──┘     are close, skipped pairs are far
```
- **Music encoder input** (per segment): `timbre[12], pitches[12], loudness, tempo,
  beat_regularity, rhythmic_entropy, harmonic_change_rate, segment_position, energy_slope,
  brightness`
- **Context encoder input**: `hour_sin, hour_cos, day_of_week, session_length_so_far,
  skips_this_session, completion_rate_last_5, prev_song_energy, prev_song_valence`
- **Training signal** (from behavior, no self-report needed): completion = positive, skip = negative,
  replay = strong positive.

After training: query "at 2am, 90min into a session, last 3 tracks completed" → nearest music
segments → inspect what they share (high regularity? low harmonic change? warm timbre?) = what my
brain is actually seeking.

### Open design decision — training signal source
- **A: behavioral only** (skips/completions/replays) — no labels, starts now, surfaces unconscious
  patterns. **Likely most honest for ADHD** (behavior > self-report).
- **B: self-reported context** (log "focused/anxious/unwinding" in real time) — richer, needs a
  capture UI.
- **C: both** — start A, layer B later. (Current lean: start with A.)

---

## 7. Infrastructure — the always-on server

- **Host OS:** Ubuntu 24.04 LTS (Noble Numbat)
- **Network:** TP-Link Deco mesh, subnet `192.168.68.0/24`, Fast Ethernet 100 Mbps (hardware
  constraint). Overlay + NAT traversal via **Tailscale**. Local DNS sinkholed/resolved via
  **Pi-hole**.
- **Containers:** Docker engine; lifecycle/volumes via **CasaOS** (user UI) + **Portainer** (admin,
  port 9000, `/var/run/docker.sock`). Active: Pi-hole, Jellyfin, Portainer.
- **Remote desktop:** XRDP + XFCE4; env init via custom `/etc/xrdp/startwm.sh` (bypasses Gnome);
  clipboard via `autocutsel -fork` at session login.
- **Robotics:** ROS 2 Jazzy Jalisco on bare metal (`/opt/ros/jazzy`), native GUI nodes over XRDP.
- **Telemetry:** `btop`.

Implication: the service runs as Docker container(s) alongside existing stack; reachable over
Tailscale; modest bandwidth so prefer ReccoBeats lookups (no download) over yt-dlp where possible.

---

## 8. The always-on service shape (to build)

```
┌─────────────────┐
│  Poller (loop)  │  every ~20s: Spotify currently-playing, dedupe consecutive same-track
└────────┬────────┘
         │ new track
         ▼
┌─────────────────┐
│  Log play event │  → DB: track_id, ts, ms_played, context   (INSTANT — never blocks)
└────────┬────────┘
         │ analysis/features missing for this track?
         ▼
┌─────────────────┐
│ Resolution queue│  → enqueue "get features/analysis for track_id"
└────────┬────────┘
         ▼
┌─────────────────┐
│  Audio worker   │  ReccoBeats (features) ── and/or ── yt-dlp + extractor (analysis)
│                 │  → embedder → store in DB
└─────────────────┘
```
The play event is captured instantly; analysis fills in asynchronously. Decoupled — yt-dlp latency
never blocks logging.

New code needed: **poller**, **ReccoBeats client**, **yt-dlp resolver w/ duration validation**, the
**queue** gluing them. Extractor + embedder already exist.

---

## 9. Open decisions / things to settle (not a sequence — pick as needed)

- Training signal: behavioral-only (A) vs +self-report (B) vs both (C). Current lean: A.
- Feature source policy: which features come from ReccoBeats vs own extractor (consistency, §5c).
- Which tracks get full analysis (top-N by play count?) vs features-only.
- Context-capture interface if going with B (CLI prompt / phone / Telegram bot).
- Embedding granularity: whole-track vs segment-level (segment-level is the eventual aim).
- Where the vector store lives (pgvector on the server vs FAISS file).
- Confirm ReccoBeats base URL + exact rate limit empirically before relying on it.

---

## 10. Quick reference — repo + commands

```
audiolens/
  services/api/app/         FastAPI, routers, schemas, SQLAlchemy models
  services/extractor/app/   features/ analysis/ audio/ + Kafka worker   ← analysis lives here
  services/embedder/app/    models/ training/ inference/                 ← embedding model
  migrations/               Alembic (pgvector + HNSW)
  infra/                    docker/ k8s/ terraform/
  scripts/                  model download, bulk ingest
  tests/                    unit (synthetic audio) + integration
```

```bash
make up        # full local stack
make test      # 13 unit tests
# standalone extractor (no Docker): load_audio → FeatureExtractor().extract / AnalysisExtractor().extract
# ReccoBeats feature lookup:
curl "https://api.reccobeats.com/v1/track/<spotify_track_id>/audio-features"
```

---

### One-line summary
Build an always-on personal logger that captures what I play + my behavioral context, gets features
cheaply from ReccoBeats (by track ID) and deep segment-level analysis from my own extractor
(yt-dlp + librosa) for top tracks, and trains a behavior-driven embedding that links *what's in the
music* to *what's going on in my head* — starting from simple EDA and a logistic-regression baseline
before any neural model.
