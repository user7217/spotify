# Overnight Build — Runbook

One Docker container that turns your Spotify streaming history into a database of
**audio features** (every track) + **audio analysis** (downloaded tracks). SQLite
output, fully resumable. Built to run on the Ubuntu home server overnight.

## What it does (4 resumable stages)

1. **ingest** — filters history to 2024–2026, builds `catalog.db` (~14,300 unique
   tracks after variant grouping).
2. **reccobeats** — fetches 11 cheap features per track by Spotify ID (no
   download). Two-step: resolve IDs via `/v1/track?ids=`, then
   `/v1/track/{id}/audio-features`. Also captures real `durationMs` (used by
   stage 3) + popularity.
3. **download** — yt-dlp pulls audio for the heavy path, **most-played first**.
   Each download is duration-validated against the known `durationMs`
   (>15s mismatch = wrong video → tries next search hit).
4. **analyze** — librosa DSP on each downloaded file → one JSON doc per track in
   `track_docs/` (rhythm, harmony, structure, low-level features).

Kill it any time (`Ctrl-C`, reboot, crash) and re-run — each stage skips finished
work via the DB, so it continues, it doesn't restart.

## Deploy on the server

```bash
# 1. get the repo onto the server (git clone / rsync), then:
cd audiolens

# 2. put your streaming-history JSON folder where compose expects it
#    (the Streaming_History_Audio_*.json files), e.g.:
ln -s /path/to/your/spotify_export ./sp_data
mkdir -p ./work          # outputs land here (catalog.db, audio/, track_docs/)

# 3. build + run, detached, with logs
docker compose -f docker-compose.overnight.yml up --build -d
docker compose -f docker-compose.overnight.yml logs -f
```

Outputs on the host:

```
work/catalog.db          SQLite: catalog_tracks, reccobeats_features, plays, ...
work/audio/              downloaded audio (.opus by default)
work/track_docs/<id>.json  full DSP analysis per analyzed track
```

## Config (env in `docker-compose.overnight.yml`)

| Var | Default | Meaning |
|---|---|---|
| `YEARS` | `2024,2025,2026` | history years to include |
| `TOP_N` | `all` | heavy path scope; set a number (e.g. `1500`) to cap |
| `CODEC` | `opus` | downloaded audio codec (`opus` smallest; `mp3`/`wav` ok) |
| `QPS` | `8` | ReccoBeats feature requests/sec (back off if you see 429s) |
| `WORKERS` | CPU count | DSP process-pool size |
| `STAGES` | all four | comma list — run/redo a subset, e.g. `reccobeats` only |

## Reality check on timing

Features (stage 2) for all ~14k tracks is fast — no downloads, just API calls.
The heavy path (stages 3–4, yt-dlp + librosa on ~14k tracks) **won't finish in one
night** — realistically several. That's expected and fine: it's ordered by play
count and resumable, so your most-listened tracks are done first and each night
continues the tail. To get a complete run in one night instead, set `TOP_N` to a
few thousand.

## Useful queries / re-runs

```bash
# progress
sqlite3 work/catalog.db "SELECT status,COUNT(*) FROM reccobeats_features GROUP BY status;"
sqlite3 work/catalog.db "SELECT COUNT(*) FROM catalog_tracks WHERE audio_track_id IS NOT NULL;"
ls work/track_docs | wc -l

# re-run only one stage (container picks up where it left off)
STAGES=reccobeats docker compose -f docker-compose.overnight.yml up
```

## Notes / gotchas

- **SQLite over network mounts** can throw `disk I/O error` (file locking). Keep
  `work/` on the server's local disk, not an NFS/SMB share.
- **yt-dlp is a personal-use grey area** — keep this local/personal (per project
  notes). yt-dlp self-updates; rebuild the image periodically if YouTube changes.
- **Feature-source consistency**: every feature here comes from ReccoBeats (one
  source) — don't mix with old Spotify numbers downstream (scales differ).
- `time_signature` is the only Spotify feature ReccoBeats omits; the local
  extractor computes it from audio if you ever need it.
- No GPU / ML stack in this image by design → DSP-only documents (what the EDA +
  baseline phase needs). Add `torch`/`essentia` later for embeddings.
```
