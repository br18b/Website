# Repository architecture

## Final local source boundary

The authoritative Jekyll source is `website/`. Moving the repository source
boundary does not change a public URL: `website/fractal/...` is a repository
path, while `/fractal/...` remains the corresponding public path. The Jekyll
`baseurl` stays empty.

Tracked website content includes Jekyll source, public static assets, and the
reviewed Mandelbrot boundary demo closure. `website/_drafts/fractals2.md` is
source but is excluded from ordinary production builds.

Generator source lives under `scripts/`. Irreplaceable inputs live under
`data/source/`; small checksums and provenance records live under
`data/manifests/`. Generated private state, build products, and review staging
live under ignored `work/`.

## Publication boundary

Generators write to `work/promote/` rather than directly into `website/`. An
asset is public only after an explicit allow-list, checksum review,
dependency-closure review, and promotion with `scripts/promote_assets.py`,
whose default destination root is `website/`. SQLite catalogues, sidecars,
component shards, contours, checkpoints, restarts, caches, native build
objects, and archive bundles are never promotable.

The SPH generator source lives under `scripts/sph/` and stages rebuilt browser
artifacts under `work/promote/SPH_demo/`. The reviewed public runtime closure is
limited to `hydro_sph_WASM.html`, `sph.js`, `sph.wasm`, and
`sph_snapshot.json` under `website/SPH_demo/`.

## Generated scientific state

Git omits SQLite/WAL/SHM, contour and restart state, generated `_site`, caches,
and build products. The active Mandelbrot catalogue, checkpoints, exports, and
contours belong under ignored `work/mandelbrot/` inside the repository working
tree. This keeps source, private state, and public output within one portable
workspace without treating generated scientific state as a deployable asset.

Ignored state is not protected by Git. Back up valuable `work/` contents and
never run `git clean -fdx` in a workspace that contains the active catalogue.
The superseded JSON catalogue and old implementation variants are not required
for current operation and are not part of the active work tree.

## Deferred stale-URL compatibility

The retained generated-site inventory contained these unlinked legacy paths:

```text
/real_quadratic_output/real_quadratic_density.png
/real_quadratic_output/real_quadratic_iterates.png
/real_quadratic_output/real_quadratic_table.md
```

Their current counterparts remain at `/fractal/2D/`. The stale family is not
included and no redirects were added. Before changing compatibility, choose
between explicit redirects, reviewed compatibility copies, or accepting their
retirement based on access/link evidence.
