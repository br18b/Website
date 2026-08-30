#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)

find_repository() {
  local current=$SCRIPT_DIR
  while [[ $current != / ]]; do
    if [[ -e "$current/.git" || -e "$current/.root" ]]; then
      printf '%s\n' "$current"
      return 0
    fi
    current=$(dirname -- "$current")
  done
  echo "Could not locate a repository marker above $SCRIPT_DIR" >&2
  return 1
}

REPOSITORY=$(find_repository)
DEFAULT_DATA_ROOT="$REPOSITORY/work/mandelbrot"
STATUS_TOOL="$SCRIPT_DIR/catalogue_status.py"

usage() {
  cat <<'EOF'
Usage: ./scripts/mandelbrot/ops.sh COMMAND [OPTIONS]

Commands:
  build                         Build the Mandelbrot tools
  test                          Build and run lightweight tests
  status [status options]       Read the existing catalogue without modifying it
  resume --config PATH [--plan] [--no-paths] [--data-root PATH] [-- scanner options]

Normal operation always uses <repository>/work/mandelbrot.  --data-root is an
explicit advanced override.  The resume command runs in the foreground, holds
a single-writer lock, and records output under work/mandelbrot/logs/.
EOF
}

run_python() {
  env -u MANDELBROT_DATA_ROOT PYTHONDONTWRITEBYTECODE=1 python3 -B "$@"
}

case ${1:-} in
  build)
    shift
    (($# == 0)) || { echo "build accepts no options" >&2; exit 2; }
    exec "$SCRIPT_DIR/build.sh"
    ;;
  test)
    shift
    (($# == 0)) || { echo "test accepts no options" >&2; exit 2; }
    exec "$SCRIPT_DIR/build.sh" --test
    ;;
  status)
    shift
    exec env -u MANDELBROT_DATA_ROOT PYTHONDONTWRITEBYTECODE=1 \
      python3 -B "$STATUS_TOOL" "$@"
    ;;
  resume)
    shift
    config_arg=
    explicit_data_root=
    plan_only=false
    show_paths=true
    scanner_args=()
    while (($#)); do
      case $1 in
        --config)
          (($# >= 2)) || { echo "--config requires a path" >&2; exit 2; }
          config_arg=$2
          shift 2
          ;;
        --data-root)
          (($# >= 2)) || { echo "--data-root requires a path" >&2; exit 2; }
          explicit_data_root=$2
          shift 2
          ;;
        --plan)
          plan_only=true
          shift
          ;;
        --no-paths)
          show_paths=false
          shift
          ;;
        --)
          shift
          scanner_args=("$@")
          break
          ;;
        *)
          echo "Unknown resume option: $1" >&2
          usage >&2
          exit 2
          ;;
      esac
    done
    [[ -n $config_arg ]] || { echo "resume requires --config PATH" >&2; exit 2; }
    if [[ $config_arg != /* ]]; then
      config_arg="$REPOSITORY/$config_arg"
    fi
    [[ -f $config_arg ]] || { echo "Resume config does not exist: $config_arg" >&2; exit 1; }
    config_path=$(realpath -- "$config_arg")

    status_args=(--config "$config_path" --paths-only)
    if [[ -n $explicit_data_root ]]; then
      status_args+=(--data-root "$explicit_data_root")
    fi
    declare -A resolved=()
    resolved_output=$(run_python "$STATUS_TOOL" "${status_args[@]}")
    while IFS=$'\t' read -r key value; do
      resolved["$key"]=$value
    done <<< "$resolved_output"
    for required in repository data_root database run_root config period_start period resume reset_root_checkpoint; do
      [[ -v "resolved[$required]" ]] || {
        echo "Path resolver did not return $required" >&2
        exit 1
      }
    done

    data_root=${resolved[data_root]}
    if [[ -z $explicit_data_root && $data_root != "$DEFAULT_DATA_ROOT" ]]; then
      echo "Default data root escaped the repository: $data_root" >&2
      exit 1
    fi
    if [[ -z $explicit_data_root && $data_root/ != "$REPOSITORY"/* ]]; then
      echo "Refusing a non-repository data root without explicit --data-root" >&2
      exit 1
    fi
    [[ ${resolved[resume]} == true ]] || { echo "Resume config must set resume=true" >&2; exit 1; }
    [[ ${resolved[reset_root_checkpoint]} == false ]] || {
      echo "Resume config must set reset_root_checkpoint=false" >&2
      exit 1
    }

    scanner="$SCRIPT_DIR/bin/component_area_scan"
    checkpoint="${resolved[run_root]}/root_checkpoints/period_${resolved[period_start]}.chk"
    if [[ $show_paths == true ]]; then
      echo "Repository:   ${resolved[repository]}"
      echo "Data root:    $data_root"
      echo "Database:     ${resolved[database]}"
      echo "Run root:     ${resolved[run_root]}"
      echo "Config:       ${resolved[config]}"
    else
      echo "Repository-local paths: validated (display suppressed)"
    fi
    echo "Period range: ${resolved[period_start]}..${resolved[period]}"
    if [[ -f $checkpoint ]]; then
      [[ $show_paths == true ]] && echo "Checkpoint:   $checkpoint (present)" \
        || echo "Checkpoint:   present for period ${resolved[period_start]}"
    else
      [[ $show_paths == true ]] && echo "Checkpoint:   $checkpoint (missing)" \
        || echo "Checkpoint:   missing for period ${resolved[period_start]}"
    fi
    [[ $show_paths == true ]] && echo "Scanner:      $scanner" \
      || echo "Scanner path: validated (display suppressed)"
    [[ -f ${resolved[database]} ]] || { echo "Catalogue database is missing" >&2; exit 1; }
    if [[ $plan_only == true ]]; then
      echo "Plan only: scanner not started"
      exit 0
    fi
    [[ -x $scanner ]] || {
      echo "Scanner is not built; run ./scripts/mandelbrot/ops.sh build first" >&2
      exit 1
    }
    lock_dir="$data_root/.locks"
    log_dir="$data_root/logs"
    mkdir -p -- "$lock_dir" "$log_dir"
    lock_file="$lock_dir/catalogue-writer.lock"
    exec 9>"$lock_file"
    if ! flock -n 9; then
      echo "Another catalogue writer holds $lock_file" >&2
      exit 1
    fi
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    log_file="$log_dir/component-area-scan-$timestamp.log"
    echo "Log:          $log_file"
    echo "Stop cleanly with Ctrl+C; rerun the same command to restart from checkpoints."

    command=("$scanner" --config "$config_path" "${scanner_args[@]}")
    set +e
    if [[ -n $explicit_data_root ]]; then
      env -u MANDELBROT_DATA_ROOT MANDELBROT_DATA_ROOT="$data_root" \
        "${command[@]}" 2>&1 | tee -a "$log_file"
    else
      env -u MANDELBROT_DATA_ROOT "${command[@]}" 2>&1 | tee -a "$log_file"
    fi
    scanner_status=${PIPESTATUS[0]}
    set -e
    exit "$scanner_status"
    ;;
  -h|--help|help|'')
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
