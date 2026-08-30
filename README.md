# Personal website

This repository is in the first local reconstruction stage of a forensic
migration. Git history and the current uncommitted website work are preserved,
but no remote deployment change has been made.

Jekyll intentionally remains at the repository root for the first controlled
GitHub Actions transition. The later move into `website/` is documented but is
not part of this branch.

Key locations:

- root Jekyll files and `fractal/`: current website source and public assets;
- `scripts/mandelbrot/`: the retained v8 generators, schema, tests, and docs;
- `scripts/scientific/` and `scripts/notebooks/`: selected scientific sources;
- `data/source/`: irreplaceable input data tracked with hashes;
- `work/`: ignored private state and promotion staging;
- `docs/`: architecture, provenance, development, and regeneration guidance.

See `docs/local-development.md` for Jekyll commands and
`docs/mandelbrot-data-root.md` before running catalogue tools.
