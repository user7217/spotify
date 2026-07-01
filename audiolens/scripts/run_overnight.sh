#!/usr/bin/env bash
# Overnight build orchestrator — runs the full catalog -> features -> audio ->
# analysis pipeline as one resumable job. Designed to run inside the
# overnight Docker container on the home server. Every stage is idempotent:
# kill it any time, re-run, it picks up where it left off.
#
# Config via env (see docker-compose.overnight.yml):
#   SP_DATA_DIR   streaming-history JSON folder              (default /data/sp_data)
#   WORK_DIR      where catalog.db / audio / docs live       (default /data/work)
#   YEARS         comma list to include                      (default 2024,2025,2026)
#   TOP_N         tracks to download+analyze (heavy path)    (default 1500)
#   CODEC         downloaded audio codec                     (default opus)
#   QPS           ReccoBeats feature requests/sec            (default 8)
#   WORKERS       DSP process pool size                      (default = CPU count)
#   STAGES        which stages to run                        (default all)
set -euo pipefail

SP_DATA_DIR="${SP_DATA_DIR:-/data/sp_data}"
WORK_DIR="${WORK_DIR:-/data/work}"
YEARS="${YEARS:-2024,2025,2026}"
TOP_N="${TOP_N:-1500}"
CODEC="${CODEC:-opus}"
QPS="${QPS:-8}"
WORKERS="${WORKERS:-}"
STAGES="${STAGES:-ingest,reccobeats,download,analyze}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

CATALOG="${WORK_DIR}/catalog.db"
AUDIO_DIR="${WORK_DIR}/audio"
DOCS_DIR="${WORK_DIR}/track_docs"
FILTERED_DIR="${WORK_DIR}/_history_filtered"
mkdir -p "${WORK_DIR}" "${AUDIO_DIR}" "${DOCS_DIR}"

log() { echo "[$(date '+%F %T')] $*"; }
has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ── stage 1: scope streaming history to the chosen years, build catalog ──────
if has_stage ingest; then
  # The ingest sink uses plain INSERTs (not upserts), so re-running it against
  # an already-built catalog fails on the UNIQUE(spotify_track_id) constraint.
  # The catalog only needs building once; skip if it already has tracks so a
  # fresh container (down/up) resumes straight into features/downloads without
  # touching the existing DB (which also holds the ReccoBeats features).
  existing=$(python - "${CATALOG}" <<'PY'
import sqlite3, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
try:
    n = sqlite3.connect(p).execute("SELECT COUNT(*) FROM catalog_tracks").fetchone()[0]
    print(n)
except Exception:
    print(0)
PY
)
  if [ "${existing:-0}" -gt 0 ]; then
    log "STAGE ingest — catalog already has ${existing} tracks, skipping (delete ${CATALOG} to rebuild from scratch)"
  else
    log "STAGE ingest — filtering history to years: ${YEARS}"
    rm -rf "${FILTERED_DIR}"; mkdir -p "${FILTERED_DIR}"
    IFS=',' read -ra YRS <<< "${YEARS}"
    shopt -s nullglob
    for f in "${SP_DATA_DIR}"/Streaming_History_Audio_*.json; do
      base="$(basename "$f")"
      for y in "${YRS[@]}"; do
        if [[ "$base" == *"_${y}"*.json ]]; then ln -sf "$f" "${FILTERED_DIR}/$base"; break; fi
      done
    done
    n=$(ls -1 "${FILTERED_DIR}" | wc -l)
    log "  linked ${n} history files"
    python -m services.ingest.app.cli "${FILTERED_DIR}" --sqlite "${CATALOG}"
    log "STAGE ingest — done"
  fi
fi

# ── stage 2: cheap features for EVERY track (no download) ────────────────────
if has_stage reccobeats; then
  log "STAGE reccobeats — backfilling features for all tracks"
  python "${REPO}/scripts/reccobeats_client.py" --sqlite "${CATALOG}" --qps "${QPS}"
  log "STAGE reccobeats — done"
fi

# ── stage 3: download audio for the heavy path, most-played first ────────────
# TOP_N=all (or 0/empty) -> every track, ordered by play_count so the tracks
# that matter get done first even if the run spans several nights.
if has_stage download; then
  LARG=()
  if [[ -n "${TOP_N}" && "${TOP_N}" != "all" && "${TOP_N}" != "0" ]]; then
    LARG=(--limit "${TOP_N}"); log "STAGE download — top ${TOP_N} tracks via yt-dlp (codec=${CODEC})"
  else
    log "STAGE download — ALL tracks via yt-dlp, play_count order (codec=${CODEC})"
  fi
  python "${REPO}/scripts/ytdlp_resolver.py" --sqlite "${CATALOG}" \
      --audio-dir "${AUDIO_DIR}" --codec "${CODEC}" "${LARG[@]}"
  log "STAGE download — done"
fi

# ── stage 4: full DSP analysis on everything that has audio ──────────────────
if has_stage analyze; then
  log "STAGE analyze — DSP documents for downloaded tracks"
  WARG=(); [[ -n "${WORKERS}" ]] && WARG=(--workers "${WORKERS}")
  python "${REPO}/scripts/run_pipeline.py" --sqlite "${CATALOG}" \
      --docs-dir "${DOCS_DIR}" "${WARG[@]}"
  log "STAGE analyze — done"
fi

log "ALL DONE. catalog=${CATALOG}  audio=${AUDIO_DIR}  docs=${DOCS_DIR}"
