# Mandelbrot component catalogue

The catalogue is the only module that knows how persistent Mandelbrot
component data is stored. Numerical programs use the matching typed C++ and
Python APIs:

```text
components/catalogue/component_catalogue.hpp
components/catalogue/component_catalogue.cpp
components/catalogue/component_catalogue.py
```

Canonical state lives in:

```text
<catalogue-root>/component_catalogue.sqlite
```

`runs/` still contains algorithm-specific checkpoint files and `exports/`
contains disposable CSV/JSON/browser outputs. Consumers must not query the
database directly or parse a canonical persistence payload.

## Why the schema is hybrid

`components` contains stable identity plus scalar fields used for selection:
period, approximate center and area, validation flags, provenance, hierarchy,
shape class, and polygon-point count. Exact decimal strings are retained beside
the approximate `REAL` accelerators.

`component_records` contains the complete canonical decimal-string record.
This separate payload keeps large polygons out of period/identity scans and
allows the typed record to evolve without a destructive table migration.
The payload is authoritative; `REAL` columns are never used to reconstruct a
component or make a precision-sensitive final decision.

This is preferable to the earlier draft schema in three ways:

1. A center cannot usually be classified as merely "ordinary" or "exact".
   The existing API consistently carries working and validated precision for
   every component, so exact decimal center strings are stored for every row.
2. Converting centered polygons to binary-double BLOBs would discard the
   catalogue's current arbitrary-precision representation. A future,
   versioned compressed geometry table can be added after an explicit accuracy
   study; migration does not silently reduce precision.
3. Classification is richer than the draft `circle/cardioid/slanted_cardioid`
   row. The current typed API stores optional disk and cardioid fits,
   confidence, and errors. Keeping the canonical payload avoids losing these
   values while indexed `shape_class` still supports fast selection.

The full schema is in `components/catalogue/schema.sql`.

## Important tables

- `catalogue_manifest`: the typed catalogue manifest.
- `components`: stable UUID and queryable scalar/index fields.
- `component_records`: authoritative complete component records.
- `period_records`: rebuildable typed period materializations.
- `hierarchy_records`: rebuildable typed hierarchy materializations.
- `area_scan_centers`: exact-scanner center state.
- `area_scan_measurements`: one row per run, period, component, and radius.
- `area_scan_summaries`: period/radius summaries.
- `json_migrations`: completed one-time imports.

`components.uuid` is unique. Scanner tables use composite primary keys.
Replaying the same discovery or checkpoint therefore updates the same logical
row rather than creating a duplicate.

## Durability and restart behavior

Every connection enables:

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA busy_timeout = 60000;
```

Every component batch, manifest revision, period rebuild, classification batch,
and scanner snapshot is committed in a transaction. A process or machine crash
reveals either the previous committed state or the entire new state.

The area scanner additionally makes every completed radius eligible for a small
atomic checkpoint batch and flushes pending rows every
`area_checkpoint_components` completions or `area_checkpoint_seconds` seconds
(20 seconds by default; configuration above 600 seconds is rejected). On
restart it merges those batches with committed database measurements by the stable
`(period, component_index, rho)` identity. Checkpoint batches are removed only
after measurements and summaries have both committed.

Period and hierarchy records are derived views. Deduplication never relies on
them: `upsert_components` queries authoritative `components` rows for the
target periods and confirms arbitrary-precision center distance. A stale view
after interruption therefore cannot create a duplicate or hide a component.

## Public operations

Callers continue to use typed methods such as:

```text
load_component
component_exists
query_components
load_snapshot
upsert_component / upsert_components
update_component_classifications
load_period / save_period / rebuild_period_indexes
load_hierarchy / save_hierarchy / rebuild_hierarchy_indexes
area_scan_store
write_component_export / write_skeleton_export
verify_integrity
```

Python exposes the same operations and dataclasses. The Python
`component_area_scan.py` remains a launcher for the authoritative C++ scanner.

Exports remain normal files because they are external interchange products,
not authoritative catalogue storage. Root-finder/frontier checkpoints also
remain algorithm-owned files under `runs/`.

## Optional conversion of a legacy JSON catalogue

Fresh initialization and population do not require legacy JSON. The following
tool is retained only for an owner who deliberately wants to import an old
catalogue. Back up that legacy root first. An in-place invocation is:

```bash
python3 components/catalogue/migrate_json_catalogue.py \
  /path/to/component_catalogue
```

This creates `component_catalogue.sqlite` beside the supplied JSON catalogue.
It does not delete or modify the JSON files. In-place conversion also preserves
any `runs/` root, frontier, and area checkpoint files found there.

The importer:

1. imports components in committed batches;
2. imports period and hierarchy metadata;
3. imports `centers.csv`, `components.csv`, and `period_summary.csv` into the
   `default` scanner run when present;
4. publishes the old manifest only after all data is present;
5. runs SQLite integrity and foreign-key checks;
6. compares every source UUID with the destination;
7. records the completed migration.

The operation is idempotent. If it is interrupted, run the same command again.
UUID uniqueness and scanner primary keys make replay safe.

For a separate destination:

```bash
python3 components/catalogue/migrate_json_catalogue.py \
  /old/component_catalogue \
  /new/component_catalogue \
  --copy-run-state
```

Do not resume imported state until the importer prints `verification: OK` and
the normal workflow has been exercised. Normal clean-repository use should
initialize a fresh catalogue instead.

## Verification and backups

Run:

```bash
./bin/catalogue_tool --command verify
```

For a live backup, use SQLite's backup command rather than copying only the main
file while a writer is active:

```bash
sqlite3 component_catalogue.sqlite \
  ".backup '/backup/component_catalogue.sqlite'"
```

When all catalogue processes are stopped and WAL changes are checkpointed,
copying the database file is also safe. Never discard a `-wal` file from a live
database copy.
