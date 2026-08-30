#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BIN_DIR="$ROOT_DIR/bin"
OBJ_DIR="$BIN_DIR/.objects"
META_DIR="$BIN_DIR/.buildmeta"
mkdir -p "$BIN_DIR" "$OBJ_DIR" "$META_DIR"

CXX=${CXX:-g++}
COMMON_FLAGS=(-std=c++20 -O3 -march=native -pthread -Wall -Wextra -Wno-maybe-uninitialized -pedantic -I"$ROOT_DIR")
COMPONENT_FLAGS=(-std=c++20 -O2 -march=native -pthread -Wall -Wextra -Wno-maybe-uninitialized -pedantic -I"$ROOT_DIR")
SQLITE_LIBS=(-lsqlite3)
RUN_TESTS=false

if (($#)); then
  case "$1" in
    --clean)
      rm -rf "$BIN_DIR"
      echo "Removed $BIN_DIR"
      exit 0
      ;;
    --test)
      RUN_TESTS=true
      ;;
    -h|--help)
      echo "Usage: ./build.sh [--clean|--test]"
      exit 0
      ;;
    *)
      echo "Usage: ./build.sh [--clean|--test]" >&2
      echo "All numerical executables use long double by default and select Boost.Multiprecision internally when required." >&2
      exit 2
      ;;
  esac
fi

needs_rebuild() {
  local output=$1 tag=$2
  shift 2
  local stamp="$META_DIR/$(basename "$output").stamp"
  [[ ! -e "$output" || ! -f "$stamp" ]] && return 0
  [[ $(<"$stamp") != "$tag" ]] && return 0
  local dep
  for dep in "$@" "$ROOT_DIR/build.sh"; do
    [[ -e "$dep" && "$dep" -nt "$output" ]] && return 0
  done
  return 1
}

mark_built() {
  printf '%s\n' "$2" > "$META_DIR/$(basename "$1").stamp"
}

build_or_skip() {
  local label=$1 output=$2 tag=$3
  shift 3
  local -a deps=()
  while [[ $# -gt 0 && $1 != -- ]]; do deps+=("$1"); shift; done
  shift
  local -a command=("$@")
  if needs_rebuild "$output" "$tag" "${deps[@]}"; then
    echo "Building $label..."
    "${command[@]}"
    mark_built "$output" "$tag"
  else
    echo "Up to date: $label"
  fi
}

CAT_HPP="$ROOT_DIR/components/catalogue/component_catalogue.hpp"
CAT_CPP="$ROOT_DIR/components/catalogue/component_catalogue.cpp"
CAT_OBJ="$OBJ_DIR/component_catalogue.o"
CFG_HPP="$ROOT_DIR/common/repo_config.hpp"

build_or_skip "catalogue module" "$CAT_OBJ" \
  "catalogue:${COMPONENT_FLAGS[*]}:$CXX" \
  "$CAT_HPP" "$CAT_CPP" "$CFG_HPP" -- \
  "$CXX" "${COMPONENT_FLAGS[@]}" -c "$CAT_CPP" -o "$CAT_OBJ"

SHAPE_HPP="$ROOT_DIR/components/shapes/component_shapes.hpp"
SHAPE_CPP="$ROOT_DIR/components/shapes/component_shapes.cpp"
SHAPE_OBJ="$OBJ_DIR/component_shapes.o"

build_or_skip "component shape-analysis module" "$SHAPE_OBJ" \
  "component_shapes:${COMPONENT_FLAGS[*]}:$CXX" \
  "$SHAPE_HPP" "$SHAPE_CPP" "$CAT_HPP" "$CAT_OBJ" -- \
  "$CXX" "${COMPONENT_FLAGS[@]}" -c "$SHAPE_CPP" -o "$SHAPE_OBJ"

build_or_skip "contour tracer" "$BIN_DIR/contours" \
  "contours:${COMMON_FLAGS[*]}:$CXX" \
  "$ROOT_DIR/contours/contours.cpp" "$CFG_HPP" -- \
  "$CXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/contours/contours.cpp" -o "$BIN_DIR/contours"

build_or_skip "contour postprocessor" "$BIN_DIR/postprocess_contours" \
  "postprocess:${COMMON_FLAGS[*]}:$CXX" \
  "$ROOT_DIR/contours/postprocess_contours.cpp" "$CFG_HPP" -- \
  "$CXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/contours/postprocess_contours.cpp" -o "$BIN_DIR/postprocess_contours"

for spec in \
  "component_area_scan:component area scanner" \
  "component_atlas_hunter:satellite atlas hunter" \
  "component_boundary_hunter:boundary-guided component hunter" \
  "component_quadtree_hunter:adaptive quadtree atlas hunter" \
  "fit_for_demo:analytic demo fitter"; do
  target=${spec%%:*}; label=${spec#*:}
  deps=("$ROOT_DIR/components/$target.cpp" "$CAT_HPP" "$CAT_OBJ" "$CFG_HPP")
  objects=("$CAT_OBJ")
  if [[ "$target" == fit_for_demo ]]; then
    deps+=("$SHAPE_HPP" "$SHAPE_OBJ")
    objects+=("$SHAPE_OBJ")
  fi
  build_or_skip "$label" "$BIN_DIR/$target" \
    "$target:${COMPONENT_FLAGS[*]}:${SQLITE_LIBS[*]}:$CXX" "${deps[@]}" -- \
    "$CXX" "${COMPONENT_FLAGS[@]}" "$ROOT_DIR/components/$target.cpp" "${objects[@]}" "${SQLITE_LIBS[@]}" -o "$BIN_DIR/$target"
done

build_or_skip "quick component-shape classifier" "$BIN_DIR/classify_component_shapes" \
  "classify_component_shapes:${COMPONENT_FLAGS[*]}:${SQLITE_LIBS[*]}:$CXX" \
  "$ROOT_DIR/components/classify_component_shapes.cpp" "$SHAPE_HPP" "$SHAPE_OBJ" "$CAT_HPP" "$CAT_OBJ" "$CFG_HPP" -- \
  "$CXX" "${COMPONENT_FLAGS[@]}" "$ROOT_DIR/components/classify_component_shapes.cpp" "$SHAPE_OBJ" "$CAT_OBJ" "${SQLITE_LIBS[@]}" -o "$BIN_DIR/classify_component_shapes"

build_or_skip "catalogue CLI" "$BIN_DIR/catalogue_tool" \
  "catalogue_tool:${COMPONENT_FLAGS[*]}:${SQLITE_LIBS[*]}:$CXX" \
  "$ROOT_DIR/components/catalogue/catalogue_tool.cpp" "$CAT_HPP" "$CAT_OBJ" "$CFG_HPP" -- \
  "$CXX" "${COMPONENT_FLAGS[@]}" "$ROOT_DIR/components/catalogue/catalogue_tool.cpp" "$CAT_OBJ" "${SQLITE_LIBS[@]}" -o "$BIN_DIR/catalogue_tool"

if [[ "$RUN_TESTS" == true ]]; then
  build_or_skip "catalogue C++ tests" "$BIN_DIR/test_catalogue" \
    "test_catalogue:${COMPONENT_FLAGS[*]}:${SQLITE_LIBS[*]}:$CXX" \
    "$ROOT_DIR/components/catalogue/test_catalogue.cpp" "$CAT_HPP" "$CAT_OBJ" -- \
    "$CXX" "${COMPONENT_FLAGS[@]}" "$ROOT_DIR/components/catalogue/test_catalogue.cpp" "$CAT_OBJ" "${SQLITE_LIBS[@]}" -o "$BIN_DIR/test_catalogue"
  "$BIN_DIR/test_catalogue"
  python3 "$ROOT_DIR/components/catalogue/test_catalogue.py"
  python3 "$ROOT_DIR/components/catalogue/test_restart.py"
  python3 "$ROOT_DIR/components/catalogue/test_migration.py"
fi

cat <<EOF

Build complete. Binaries are in:
  $BIN_DIR

Default workflow (all tools read mandelbrot.json automatically):
  ./bin/component_area_scan
  ./bin/classify_component_shapes
  ./bin/component_boundary_hunter
  ./bin/component_atlas_hunter
  python3 demos/atlas/make_atlas_demo.py

Use an alternate config with --config PATH.
EOF
