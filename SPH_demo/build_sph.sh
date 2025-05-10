emcc sph_core.cpp \
  -o sph.js \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='Module' \
  -s EXPORTED_FUNCTIONS='["_malloc", "_free", "_setup_grid", "_fill_grid", "_compute_density_forces", "_integrate", "_set_simulation_params", "_init_W_lookup"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall", "HEAPU8", "HEAPF32", "HEAP32"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -O3