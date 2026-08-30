# Personal website

This repository is the selectively reconstructed result of a forensic website
migration. Git history, the selected current website work, generator source,
and irreplaceable input data are preserved. No remote deployment change has
been made.

`website/` is the sole authoritative Jekyll source boundary. The repository-only
`scripts/`, `data/`, `docs/`, and ignored `work/` trees are never part of the
published site.

Key locations:

- `website/`: Jekyll source and reviewed public static assets;
- `scripts/mandelbrot/`: the retained v8 generators, schema, tests, and docs;
- `scripts/scientific/` and `scripts/notebooks/`: selected scientific sources;
- `data/source/`: irreplaceable input data tracked with hashes;
- `work/`: ignored private state and promotion staging;
- `docs/`: architecture, provenance, development, and regeneration guidance.

See `docs/local-development.md` for `--source website` Jekyll commands and
`docs/mandelbrot-data-root.md` before running catalogue tools.
