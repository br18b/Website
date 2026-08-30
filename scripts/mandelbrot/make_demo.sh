#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)
cd "$ROOT_DIR"

BUILD=1
RESET_BOUNDARY=0
RESET_ATLAS=0
RECOMPUTE_DEMO=0
SKIP_AREA=0
SKIP_CLASSIFICATION=0
SKIP_BOUNDARY=0
SKIP_ATLAS=0
SKIP_DEMO=0
CONFIG_FILE=""
LOG_FILE=""

usage() {
  cat <<'USAGE'
Usage: ./make_demo.sh [options]

Runs the Mandelbrot component pipeline sequentially. All tools use the root
mandelbrot.json automatically unless --config is supplied.

Options:
  --config FILE          Use an alternate unified JSON configuration
  --no-build             Do not run build.sh
  --reset-boundary       Reset the boundary-hunter run before starting
  --reset-atlas          Reset the satellite-hunter run before starting
  --recompute-demo       Force regeneration of compact demo component data
  --skip-area            Skip component_area_scan
  --skip-classification  Skip classify_component_shapes
  --skip-boundary        Skip component_boundary_hunter
  --skip-atlas           Skip component_atlas_hunter
  --skip-demo            Skip make_atlas_demo.py
  --log FILE             Mirror output to FILE
  -h, --help             Show this help

Typical run:
  ./make_demo.sh

Overnight run:
  nohup ./make_demo.sh --log make_demo_overnight.log >/dev/null 2>&1 &
USAGE
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG_FILE="${2:?--config requires a path}"
      shift 2
      ;;
    --no-build) BUILD=0; shift ;;
    --reset-boundary) RESET_BOUNDARY=1; shift ;;
    --reset-atlas) RESET_ATLAS=1; shift ;;
    --recompute-demo) RECOMPUTE_DEMO=1; shift ;;
    --skip-area) SKIP_AREA=1; shift ;;
    --skip-classification) SKIP_CLASSIFICATION=1; shift ;;
    --skip-boundary) SKIP_BOUNDARY=1; shift ;;
    --skip-atlas) SKIP_ATLAS=1; shift ;;
    --skip-demo) SKIP_DEMO=1; shift ;;
    --log)
      LOG_FILE="${2:?--log requires a path}"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname -- "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

mkdir -p "$ROOT_DIR/.locks"
exec 9>"$ROOT_DIR/.locks/make_demo.lock"
if ! flock -n 9; then
  echo "Another make_demo.sh process is already running." >&2
  exit 1
fi

started_epoch="$(date +%s)"
current_stage="startup"

finish_report() {
  local status=$?
  local ended_epoch elapsed
  ended_epoch="$(date +%s)"
  elapsed=$((ended_epoch - started_epoch))
  printf '\n'
  if ((status == 0)); then
    printf 'Atlas pipeline completed successfully in %dh %dm %ds.\n' \
      $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  else
    echo "Atlas pipeline failed during: $current_stage" >&2
    echo "Exit status: $status" >&2
  fi
}
trap finish_report EXIT

run_stage() {
  current_stage="$1"
  shift
  printf '\n============================================================\n'
  echo "$current_stage"
  printf '============================================================\n'
  "$@"
}

config_args=()
if [[ -n "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$(realpath "$CONFIG_FILE")"
  config_args=(--config "$CONFIG_FILE")
fi

AREA_BIN="bin/component_area_scan"
CLASSIFIER_BIN="bin/classify_component_shapes"
BOUNDARY_BIN="bin/component_boundary_hunter"
ATLAS_BIN="bin/component_atlas_hunter"
FITTER_BIN="bin/fit_for_demo"
DEMO_PY="demos/atlas/make_atlas_demo.py"

if ((BUILD)); then
  run_stage "Building C++ tools" ./build.sh
fi

for path in "$AREA_BIN" "$CLASSIFIER_BIN" "$BOUNDARY_BIN" "$ATLAS_BIN" "$FITTER_BIN" "$DEMO_PY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required program is missing: $path" >&2
    exit 1
  fi
done

echo "Mandelbrot atlas pipeline"
echo "  root:       $ROOT_DIR"
echo "  config:     ${CONFIG_FILE:-$ROOT_DIR/mandelbrot.json}"
echo "  started:    $(date --iso-8601=seconds)"
echo "  PID:        $$"

if ((!SKIP_AREA)); then
  run_stage "Stage 1/5: exact-period component catalogue" \
    "$AREA_BIN" "${config_args[@]}"
fi

if ((!SKIP_CLASSIFICATION)); then
  run_stage "Stage 2/5: quick disk classification" \
    "$CLASSIFIER_BIN" "${config_args[@]}"
fi

if ((!SKIP_BOUNDARY)); then
  args=("$BOUNDARY_BIN" "${config_args[@]}")
  ((RESET_BOUNDARY)) && args+=(--reset)
  run_stage "Stage 3/5: boundary-guided component hierarchy" "${args[@]}"
fi

if ((!SKIP_ATLAS)); then
  args=("$ATLAS_BIN" "${config_args[@]}")
  ((RESET_ATLAS)) && args+=(--reset)
  run_stage "Stage 4/5: satellite component expansion" "${args[@]}"
fi

if ((!SKIP_DEMO)); then
  args=(python3 "$DEMO_PY" "${config_args[@]}")
  ((RECOMPUTE_DEMO)) && args+=(--recompute-components)
  run_stage "Stage 5/5: browser atlas" "${args[@]}"
fi
