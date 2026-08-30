#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
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
  --exclude="${PROJECT_NAME}/work" \
  --exclude="${PROJECT_NAME}/website/CLT_plots" \
  --exclude="${PROJECT_NAME}/website/BNG/pics" \
  --exclude="${PROJECT_NAME}/website/pics" \
  --exclude="${PROJECT_NAME}/website/fractal/textures" \
  --exclude="${PROJECT_NAME}/website/fractal/mandelbrot" \
  --exclude="${PROJECT_NAME}/website/fractal/wasm" \
  --exclude="${PROJECT_NAME}/website/fractal/paths" \
  --exclude="${PROJECT_NAME}/website/fractal/3D" \
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