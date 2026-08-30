# SQLite component catalogue

`component_catalogue.cpp` and `component_catalogue.py` implement the same typed
catalogue API over `<catalogue-root>/component_catalogue.sqlite`.

The schema is defined in `schema.sql`. Its main design is:

- compact, indexed scalar metadata in `components`;
- an internal integer key for joins and a stable external UUID;
- authoritative decimal strings for every center and scientific value;
- complete typed records in the separate `component_records` payload table;
- rebuildable period and hierarchy materializations;
- scanner tables with composite primary keys for idempotent replay.

The approximate `REAL` columns are accelerators only. They are not
authoritative and are not used for final arbitrary-precision filtering.
Centered polygons remain decimal-string data; the earlier draft's binary64 BLOB
proposal would have silently reduced precision.

All mutations use transactions. Connections use WAL, full synchronous commits,
foreign-key enforcement, and a 60-second busy timeout.

See `../../docs/component_catalogue.md` for the API, restart guarantees,
migration procedure, and backup guidance.
