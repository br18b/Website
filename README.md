# Personal website

This repository contains the source for rabatinb.com together with the
scientific generators and inputs used to produce selected site assets.

`website/` is the sole authoritative Jekyll source boundary. The repository-only
`scripts/`, `data/`, `docs/`, and ignored `work/` trees are never part of the
published site.

Key locations:

- `website/`: Jekyll source and reviewed public static assets;
- `scripts/mandelbrot/`: the retained v8 generators, schema, tests, and docs;
- `scripts/scientific/` and `scripts/notebooks/`: selected scientific sources;
- `data/source/`: irreplaceable input data tracked with hashes;
- `work/`: ignored private calculation state and promotion staging;
- `docs/`: architecture, provenance, development, and regeneration guidance.

See `docs/local-development.md` for `--source website` Jekyll commands and
`docs/mandelbrot-operations.md` before running catalogue tools. Valuable files
under `work/` are outside Git and need ordinary backups.
