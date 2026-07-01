#!/usr/bin/env bash
# Download Essentia pretrained models for feature overrides.
set -euo pipefail
MODEL_DIR="${1:-./models}"
mkdir -p "$MODEL_DIR"
BASE="https://essentia.upf.edu/models/classification-heads"

curl -fsSL -o "$MODEL_DIR/danceability-musicnn-msd-2.pb" \
  "$BASE/danceability/danceability-musicnn-msd-2.pb"
curl -fsSL -o "$MODEL_DIR/mood_acoustic-musicnn-msd-2.pb" \
  "$BASE/mood_acoustic/mood_acoustic-musicnn-msd-2.pb"
curl -fsSL -o "$MODEL_DIR/voice_instrumental-musicnn-msd-2.pb" \
  "$BASE/voice_instrumental/voice_instrumental-musicnn-msd-2.pb"
curl -fsSL -o "$MODEL_DIR/emomusic-musicnn-msd-2.pb" \
  "https://essentia.upf.edu/models/regression-heads/emomusic/emomusic-musicnn-msd-2.pb"

# --- embedding backbones --------------------------------------------------
FE="https://essentia.upf.edu/models/feature-extractors"
curl -fsSL -o "$MODEL_DIR/discogs-effnet-bs64-1.pb" \
  "$FE/discogs-effnet/discogs-effnet-bs64-1.pb"
curl -fsSL -o "$MODEL_DIR/msd-musicnn-1.pb" \
  "$FE/musicnn/msd-musicnn-1.pb"

# --- classification heads on those backbones -------------------------------
CH="https://essentia.upf.edu/models/classification-heads"
for f in genre_discogs400-discogs-effnet-1 mtg_jamendo_instrument-discogs-effnet-1; do
  head_dir="${f%%-discogs*}"
  curl -fsSL -o "$MODEL_DIR/$f.pb"   "$CH/$head_dir/$f.pb"   || echo "warn: $f.pb"
  curl -fsSL -o "$MODEL_DIR/$f.json" "$CH/$head_dir/$f.json" || echo "warn: $f.json"
done
for m in mood_happy mood_sad mood_aggressive mood_relaxed gender; do
  curl -fsSL -o "$MODEL_DIR/$m-musicnn-msd-2.pb" \
    "$CH/$m/$m-musicnn-msd-2.pb" || echo "warn: $m"
done

echo "models downloaded to $MODEL_DIR"
echo "torch-side models (CLAP/Music2Vec/Wav2Vec2) download on first use via HF;"
echo "MusicFM + BYOL-A weights: see README (github releases)."
