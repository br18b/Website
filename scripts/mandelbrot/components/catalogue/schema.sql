-- Mandelbrot component catalogue, SQLite storage schema v1.
--
-- Decimal strings remain authoritative.  REAL columns are deliberately only
-- query accelerators; they must never be used to reconstruct a canonical
-- ComponentRecord.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS catalogue_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

INSERT OR IGNORE INTO catalogue_meta(key, value)
VALUES
    ('storage_schema', 'mandelbrot-catalogue-sqlite-v1'),
    ('storage_schema_version', '1');

CREATE TABLE IF NOT EXISTS catalogue_manifest (
    singleton   INTEGER PRIMARY KEY CHECK (singleton = 1),
    record_json TEXT NOT NULL
) STRICT;

-- Frequently selected scalar fields live with the stable identity.  The full
-- canonical decimal-string record is kept in component_records so adding a
-- typed field does not require a destructive database migration.
CREATE TABLE IF NOT EXISTS components (
    id                       INTEGER PRIMARY KEY,
    uuid                     TEXT NOT NULL UNIQUE CHECK (length(uuid) >= 2),
    period                   INTEGER NOT NULL CHECK (period >= 1),
    center_re_text           TEXT NOT NULL,
    center_im_text           TEXT NOT NULL,
    center_re_real           REAL NOT NULL,
    center_im_real           REAL NOT NULL,
    area_estimate_text       TEXT NOT NULL,
    area_estimate_real       REAL NOT NULL,
    area_error_text          TEXT NOT NULL,
    characteristic_size_real REAL NOT NULL,
    polygon_points           INTEGER NOT NULL CHECK (polygon_points >= 0),
    multiplicity             INTEGER NOT NULL CHECK (multiplicity IN (1, 2)),
    shape_class              TEXT NOT NULL
        CHECK (shape_class IN ('unknown', 'disk', 'cardioid')),
    provenance_method        TEXT NOT NULL,
    hierarchy_root_uuid      TEXT,
    geometric_parent_uuid    TEXT,
    center_validated         INTEGER NOT NULL CHECK (center_validated IN (0, 1)),
    exact_period_validated   INTEGER NOT NULL CHECK (exact_period_validated IN (0, 1)),
    polygon_converged        INTEGER NOT NULL CHECK (polygon_converged IN (0, 1)),
    area_above_cutoff        INTEGER NOT NULL CHECK (area_above_cutoff IN (0, 1)),
    inserted_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at               TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE TABLE IF NOT EXISTS component_records (
    component_id INTEGER PRIMARY KEY,
    record_json  TEXT NOT NULL,
    FOREIGN KEY (component_id) REFERENCES components(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_components_period
ON components(period);

CREATE INDEX IF NOT EXISTS idx_components_period_center
ON components(period, center_re_real, center_im_real);

CREATE INDEX IF NOT EXISTS idx_components_period_area
ON components(period, area_estimate_real);

CREATE INDEX IF NOT EXISTS idx_components_provenance
ON components(provenance_method, period);

CREATE INDEX IF NOT EXISTS idx_components_hierarchy_root
ON components(hierarchy_root_uuid, period);

CREATE INDEX IF NOT EXISTS idx_components_parent
ON components(geometric_parent_uuid);

-- Period and hierarchy rows are rebuildable materialized views.  Their typed
-- JSON payloads preserve the existing C++/Python API and scientific metadata.
CREATE TABLE IF NOT EXISTS period_records (
    period      INTEGER PRIMARY KEY CHECK (period >= 1),
    record_json TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS hierarchy_records (
    root_uuid   TEXT PRIMARY KEY,
    record_json TEXT NOT NULL
) STRICT;

-- Exact area-scanner state.  Composite primary keys make replay idempotent.
-- row_csv is an internal, versioned serialization of the typed record; callers
-- never parse it and all identity columns remain directly queryable.
CREATE TABLE IF NOT EXISTS area_scan_centers (
    run_name        TEXT NOT NULL,
    period          INTEGER NOT NULL CHECK (period >= 1),
    component_index INTEGER NOT NULL CHECK (component_index >= 0),
    row_csv         TEXT NOT NULL,
    PRIMARY KEY (run_name, period, component_index)
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS area_scan_measurements (
    run_name        TEXT NOT NULL,
    period          INTEGER NOT NULL CHECK (period >= 1),
    component_index INTEGER NOT NULL CHECK (component_index >= 0),
    rho_text        TEXT NOT NULL,
    row_csv         TEXT NOT NULL,
    PRIMARY KEY (run_name, period, component_index, rho_text)
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_area_scan_measurements_period
ON area_scan_measurements(run_name, period);

CREATE INDEX IF NOT EXISTS idx_area_scan_measurements_period_rho
ON area_scan_measurements(run_name, period, rho_text, component_index);

CREATE TABLE IF NOT EXISTS area_scan_summaries (
    run_name TEXT NOT NULL,
    period   INTEGER NOT NULL CHECK (period >= 1),
    rho_text TEXT NOT NULL,
    row_csv  TEXT NOT NULL,
    PRIMARY KEY (run_name, period, rho_text)
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS json_migrations (
    source_root              TEXT PRIMARY KEY,
    source_catalogue_revision INTEGER NOT NULL,
    component_count          INTEGER NOT NULL,
    completed_at             TEXT NOT NULL
) STRICT;

PRAGMA user_version = 1;
