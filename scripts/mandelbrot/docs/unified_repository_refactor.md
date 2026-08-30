# Unified repository design

## Configuration

Primary C++ and Python programs load `mandelbrot.json` automatically. An alternate file is selected with `--config PATH`. `runtime.threads` is the shared worker count; algorithm sections contain only settings specific to that algorithm.

Private state resolves as an explicit tool path, then
`MANDELBROT_DATA_ROOT`, then `<repository>/work/mandelbrot`. Demo output is
staged below `<repository>/work/promote/mandelbrot` for review.

## Precision

C++ numerical paths start with `long double`. Delicate continuation and contour operations may retry with Boost.Multiprecision tiers. There is no build-time precision selector.

## Catalogue boundary

`components/catalogue/` is the persistence boundary for component science data. Its C++ structs and Python dataclasses represent components, periods, hierarchies, manifests, queries, snapshots, and exact-scan records.

Discovery and demo programs do not parse SQLite payloads, scanner checkpoints, or other tools' exports. The standard flow is:

```text
Catalogue -> typed snapshot -> numerical work -> typed upsert -> derived indexes/export
```

Algorithm-specific restart state remains under `runs/<algorithm>/<run-name>/` and may use a format suited to that algorithm. Accepted scientific records are committed to the catalogue before transient checkpoints are advanced, so a crash cannot mark a discovery complete while losing its component.

## Default workflow

```bash
./build.sh
./bin/component_area_scan
./bin/component_boundary_hunter
./bin/component_atlas_hunter
python3 demos/atlas/make_atlas_demo.py
```

The quadtree hunter remains available as a residual global search:

```bash
./bin/component_quadtree_hunter
```

## Exact scanner publication

`component_area_scan` processes one period at a time:

1. load or solve exact-period centers;
2. load or compute missing area measurements;
3. trace canonical geometry for every exact component, without the browser cutoff;
4. upsert canonical UUID records;
5. update that period index;
6. continue to the next period.

The area scanner's tabular restart data is accessed through `AreaScanStore`; callers receive typed records rather than CSV dictionaries.

## Discovery cutoff

The exact period scan is exhaustive through its configured maximum period. The `1e-10` cutoff belongs to boundary, satellite, quadtree, and browser-selection work, where only components useful to the interactive demo are retained.

## Browser demo

`make_atlas_demo.py` queries typed catalogue records at the configured cutoff. It expands canonical conjugate symmetry in memory, obtains compact analytic fits from `fit_for_demo`, and writes demo-only cache and browser files. It never reads `atlas_components*.json` back into the pipeline.

Those files are staged, not public. A separate explicit allow-list and checksum
review controls promotion into the Jekyll tree.
