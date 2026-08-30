# Repository architecture

## Current transition layout

The production Jekyll source remains at the repository root. This preserves
existing URLs while the proposed root-layout Pages workflow is reviewed and
validated. A later phase will move the same source under `website/` and change
only the build source path.

Tracked website content includes Jekyll source, public static assets, and the
reviewed Mandelbrot boundary demo closure. `_drafts/fractals2.md` is source but
is excluded from ordinary production builds.

Generator source lives under `scripts/`. Irreplaceable inputs live under
`data/source/`; small checksums and provenance records live under
`data/manifests/`. Generated private state, build products, and review staging
live under ignored `work/`.

## Publication boundary

Generators write to `work/promote/` rather than the Jekyll tree. An asset is
public only after an explicit allow-list, checksum review, dependency-closure
review, and promotion with `scripts/promote_assets.py`. SQLite catalogues,
sidecars, component shards, contours, checkpoints, restarts, caches, and native
build objects are never promotable.

## Deliberately absent

The clean repository omits the revision-23 JSON catalogue, SQLite/WAL/SHM,
contour and restart state, v7 and older Mandelbrot trees, compiled experiments,
3D outputs, real-quadratic outputs, generated `_site`, caches, and archives.
They remain preserved in the immutable migration snapshot.

## Deferred stale-URL compatibility

The Phase 1 generated-site manifest contained these unlinked legacy paths:

```text
/real_quadratic_output/real_quadratic_density.png
/real_quadratic_output/real_quadratic_iterates.png
/real_quadratic_output/real_quadratic_table.md
```

Their current counterparts remain at `/fractal/2D/`. Phase 2A did not import
the stale family and did not invent redirects. Before remote deployment, choose
between explicit redirects, reviewed compatibility copies, or accepting their
retirement based on access/link evidence.
