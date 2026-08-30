#!/usr/bin/env bash
set -euo pipefail

# This script should live inside the website/ folder.
# It creates ../website-skeleton.tar.gz by default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
PARENT_DIR="$(dirname "$PROJECT_DIR")"

OUT="${1:-$PARENT_DIR/${PROJECT_NAME}-skeleton.tar.gz}"

cd "$PARENT_DIR"

tar cv \
  --exclude-vcs \
  --exclude="${PROJECT_NAME}/_site" \
  --exclude="${PROJECT_NAME}/.jekyll-cache" \
  --exclude="${PROJECT_NAME}/.sass-cache" \
  --exclude="${PROJECT_NAME}/.bundle" \
  --exclude="${PROJECT_NAME}/vendor" \
  --exclude="${PROJECT_NAME}/node_modules" \
  --exclude="${PROJECT_NAME}/CLT_plots" \
  --exclude="${PROJECT_NAME}/BNG/pics" \
  --exclude="${PROJECT_NAME}/pics" \
  --exclude="${PROJECT_NAME}/fractal/textures" \
  --exclude="${PROJECT_NAME}/fractal/mandelbrot" \
  --exclude="${PROJECT_NAME}/fractal/wasm" \
  --exclude="${PROJECT_NAME}/fractal/paths" \
  --exclude="${PROJECT_NAME}/fractal/3D" \
  --exclude="*.png" \
  --exclude="*.jpg" \
  --exclude="*.jpeg" \
  --exclude="*.webp" \
  --exclude="*.gif" \
  --exclude="*.svg" \
  --exclude="*.fits" \
  --exclude="*.nb" \
  --exclude="*.npy" \
  --exclude="*.npz" \
  --exclude="*.h5" \
  --exclude="*.hdf5" \
  --exclude="*.wasm" \
  --exclude="*.json" \
  --exclude="*.bin" \
  -zf "$OUT" \
  "$PROJECT_NAME/"

echo
echo "Created: $OUT"
ls -lh "$OUT"