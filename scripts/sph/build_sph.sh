#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../.." && pwd)"
output_dir="${SPH_PROMOTE_DIR:-${repository_root}/work/promote/SPH_demo}"

mkdir -p -- "${output_dir}"

emcc "${script_dir}/sph_core.cpp" \
  -o "${output_dir}/sph.js" \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='Module' \
  -s EXPORTED_FUNCTIONS='["_malloc", "_free", "_setup_grid", "_fill_grid", "_compute_density_forces", "_integrate", "_set_simulation_params", "_init_W_lookup"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall", "HEAPU8", "HEAPF32", "HEAP32"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -O3

printf 'Staged SPH browser build in %s\n' "${output_dir}"
