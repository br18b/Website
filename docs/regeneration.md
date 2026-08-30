# Regeneration and promotion

## Fresh catalogue

The retained v8 code can initialize an empty SQLite catalogue without legacy
JSON:

```bash
scripts/mandelbrot/bin/catalogue_tool --command init
scripts/mandelbrot/bin/catalogue_tool --command verify
```

Use `--catalogue-root PATH` for an explicit catalogue location or set
`MANDELBROT_DATA_ROOT` for the common local root. The clean profile starts at
period 1 with resume disabled. `component_area_scan` is the first population
tool, but it is computationally expensive and was not run during Phase 2A.

## Browser generators

The atlas and boundary generators read `scripts/mandelbrot/mandelbrot.json`
and write to `work/promote/mandelbrot/demos/...` by default. They never publish
directly into the Jekyll tree.

After human review, prepare a tab-separated allow-list containing
`source_relative`, `destination_relative`, `sha256`, and optional
`required_destinations`. Validate it first:

```bash
python3 scripts/promote_assets.py --allow-list REVIEWED.tsv
```

Promotion requires the explicit `--apply` flag. List every HTML/JS import,
worker, `data.json`, image, texture, and required WASM companion in the same
allow-list. The command rejects work-state paths and non-browser artifacts.

The atlas remains staged and unpublished. The top-level WebGL demo, alternate
WASM experiment, and all 3D outputs remain archive-only in this phase.

## Optional legacy JSON import

`scripts/mandelbrot/components/catalogue/migrate_json_catalogue.py` and
`test_migration.py` are optional migration/provenance tools for someone who
already owns the legacy JSON catalogue. Normal initialization, scanners,
hunters, classifiers, wrappers, and tests do not use them.
