# Mandelbrot numerical tools

This repository contains the C++ numerical tools and Python generators used for
Mandelbrot component catalogues, boundary/equipotential contours, and browser
demos.

## One configuration

All primary programs read the repository-level `mandelbrot.json` by default.
Technical and numerical settings live there, including the single global
thread count:

```json
{
  "runtime": {
    "threads": 10
  }
}
```

Use another configuration explicitly with `--config PATH`:

```bash
./bin/component_area_scan --config experiments/large-search.json
python3 demos/atlas/make_atlas_demo.py --config experiments/large-search.json
```

Legacy key-value `.cfg` files and the old demo-local component configuration
have been removed. There is one configuration contract: the repository-level
JSON file (or an explicitly supplied alternate JSON file).

Private state uses the same portable resolution in Python and C++:

```text
explicit tool path > MANDELBROT_DATA_ROOT > <repository>/work/mandelbrot
```

`$data_root` in the JSON expands to that resolved root. Catalogue, contour,
run, restart, and export state stays below it. Demo generators stage their
outputs below `<repository>/work/promote/mandelbrot`; reviewed allow-lists are
required to promote browser assets into `<repository>/website`.

The normal active root is the repository-local default. Do not set
`MANDELBROT_DATA_ROOT` for routine work; it is only an advanced explicit escape
hatch. Ignored `work/` state is not protected by Git and must be backed up.

## Build

From any working directory, use the repository-rooted facade:

```bash
/path/to/repository/scripts/mandelbrot/ops.sh build
```

There is no build-time precision selector. Native arithmetic is `long double`.
Numerically delicate component operations use Boost.Multiprecision where their
solver requests it, and the contour tracer can retry its potential/derivative
jet at 50, 80, 120, 160, or 200 decimal digits when native evaluation becomes
unreliable near very small potential levels.

Clean local build products with:

```bash
./build.sh --clean
```

Build and run the C++/Python catalogue, restart, deduplication, and migration
tests with:

```bash
/path/to/repository/scripts/mandelbrot/ops.sh test
```

The build is incremental and produces:

```text
bin/contours
bin/postprocess_contours
bin/component_area_scan
bin/classify_component_shapes
bin/component_boundary_hunter
bin/component_atlas_hunter
bin/component_quadtree_hunter
bin/fit_for_demo
bin/catalogue_tool
```

## Canonical component catalogue

`components/catalogue/` contains matching C++ and Python catalogue APIs.
Canonical component records use decimal strings and arbitrary-precision
numbers:

- C++: `boost::multiprecision::cpp_dec_float<200>`
- Python: `decimal.Decimal`

Programs use the catalogue API rather than inventing component, period, or
hierarchy formats. The SQLite catalogue is authoritative; period records,
hierarchy trees, CSV files, and atlas JSON files are typed materializations or
generated views.

The catalogue root is configured by `paths.catalogue_root`. Its layout is:

```text
component_catalogue/
├── component_catalogue.sqlite
├── runs/
└── exports/
```

- `component_catalogue.sqlite`: authoritative records and scanner state
- `runs/`: algorithm-specific restart/checkpoint state
- `exports/`: compatibility CSV/JSON and browser inputs

Catalogue CLI examples:

```bash
./bin/catalogue_tool --command init
./bin/catalogue_tool --command list
./bin/catalogue_tool --command show --component-id COMPONENT_ID
./bin/catalogue_tool --command verify
./bin/catalogue_tool --command rebuild-indexes \
  --area-cutoff 1e-10 \
  --exact-through-period 14
```

See `docs/component_catalogue.md` for the schema and API.

For a genuinely read-only report on an existing catalogue:

```bash
./scripts/mandelbrot/ops.sh status
```

This opens the database with SQLite `mode=ro` and `query_only`, sees existing
WAL state, refuses to initialize a missing database, and checks that persistent
database/WAL metadata did not change. A no-WAL restored database is opened as
immutable to prevent SQLite from creating empty sidecars. See
`docs/mandelbrot-operations.md` for guarded resume, stopping, restart, and
backup procedures.

## Component workflow

Each executable finds `mandelbrot.json` automatically:

```bash
./bin/component_area_scan
./bin/classify_component_shapes
./bin/component_boundary_hunter
./bin/component_atlas_hunter
python3 demos/atlas/make_atlas_demo.py
```

Or run the guarded wrapper:

```bash
./make_demo.sh
```

The wrapper builds first, prevents concurrent runs, executes the stages in
order, and stops if a numerical stage exits unsuccessfully. Useful options:

```bash
./make_demo.sh --reset-boundary
./make_demo.sh --reset-atlas --recompute-demo
./make_demo.sh --skip-classification
./make_demo.sh --config experiments/test.json
./make_demo.sh --log make_demo.log
```

### Exact-period area scanner

```bash
./bin/component_area_scan
```

Builds or resumes the exhaustive low-period centre/area catalogue and exports
trusted geometry. It writes canonical component records through the catalogue
module and keeps its restart state under `runs/area_scan/`.

### Quick disk classification

```bash
./bin/classify_component_shapes
```

Fits a free geometric centre and radius to each still-unknown canonical
polygon. Components passing the configured relative RMS and maximum-error
thresholds are stored as typed `disk` classifications; everything else stays
`unknown` for a later cardioid fit. The fitter updates records exclusively
through the catalogue API.

The component-area postprocessor uses this classification directly:

```bash
python3 components/postprocess_area_scan.py
```

Confident disks retain their period colour in the vertical-stack plot, while
unresolved shapes are drawn as opaque black points.

### Boundary-guided hierarchy hunter

```bash
./bin/component_boundary_hunter
```

Searches around known component boundaries, recovers attracting cycles without
a conceptual period ceiling, traces accepted polygons, records parent-child
attachments, and writes canonical catalogue records. Reset its configured run:

```bash
./bin/component_boundary_hunter --reset
```

The algorithm is described in
`docs/boundary_guided_component_search_algorithm.md`.

### Satellite atlas hunter

```bash
./bin/component_atlas_hunter
```

Expands rational attachments from the available seed catalogue. Reset its run:

```bash
./bin/component_atlas_hunter --reset
```

### Optional quadtree hunter

The older adaptive plane search is still built and available for experiments
and future residual-cardioid searches:

```bash
./bin/component_quadtree_hunter
./bin/component_quadtree_hunter --restart-frontier
./bin/component_quadtree_hunter --reset
```

It is no longer required as the primary discovery stage.

### Browser atlas

```bash
python3 demos/atlas/make_atlas_demo.py
```

The generator queries typed `ComponentRecord` objects directly from the catalogue,
invokes `fit_for_demo` when compact analytic fits are stale, and writes the
self-contained browser demo to ignored review staging. It never parses catalogue
persistence, scanner checkpoints, or `atlas_components*.json` files. Promotion
into `website/` requires a separate reviewed allow-list and
`scripts/promote_assets.py --apply`.
Force regeneration with:

```bash
python3 demos/atlas/make_atlas_demo.py --recompute-components
```

## Equipotential contours

```bash
./bin/contours
./bin/postprocess_contours
python3 contours/postprocess_contours.py
```

The contour tracer uses long double normally and can escalate its Mandelbrot
potential jet to Boost.Multiprecision according to the `contours` section of
`mandelbrot.json`.

## Shared repository modules

Repository imports are mandatory and deterministic:

```text
common/project_root.py
common/repo_config.py
common/repo_config.hpp
components/catalogue/component_catalogue.py
components/catalogue/component_catalogue.hpp
```

There is deliberately no silent “standalone fallback” import mode. A missing
repository module is a real setup or programming error and should fail loudly.
