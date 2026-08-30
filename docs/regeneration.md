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
tool, but it is computationally expensive and is not part of a smoke test.

For an existing repository-local catalogue, use the non-modifying status and
guarded resume workflow documented in `docs/mandelbrot-operations.md`:

```bash
./scripts/mandelbrot/ops.sh status
./scripts/mandelbrot/ops.sh resume \
  --config work/mandelbrot/config/resume-period22.json \
  --plan
```

The tracked `mandelbrot.json` remains a fresh-run profile. Resume-sensitive
values belong in an ignored local configuration under `work/mandelbrot/config/`.

## Browser generators

The atlas and boundary generators read `scripts/mandelbrot/mandelbrot.json`
and write to `work/promote/mandelbrot/demos/...` by default. Other selected
scientific generators also stage review candidates below `work/promote/`.
They never publish directly into `website/`.

After human review, prepare a tab-separated allow-list containing
`source_relative`, `destination_relative`, `sha256`, and optional
`required_destinations`. Validate it first:

```bash
python3 scripts/promote_assets.py --allow-list REVIEWED.tsv
```

Paths in `destination_relative` are relative to the default destination root,
`website/`. Promotion requires the explicit `--apply` flag. List every
HTML/JS import, worker, `data.json`, image, texture, and required WASM companion
in the same allow-list. The command rejects work-state paths and non-browser
artifacts.

The five preserved Fractals II scaling images are ignored review state at
`work/promote/mandelbrot/scaling_plots/`. Their hashes and future destinations
are fixed by `data/manifests/fractals2-draft-assets.tsv`. Validate the complete
set with:

```bash
python3 scripts/promote_assets.py \
  --allow-list data/manifests/fractals2-draft-assets.tsv
```

When the article is ready, review the draft, move it from
`website/_drafts/fractals2.md` to an appropriately dated file under
`website/_posts/`, then promote the reviewed image set with the same command
plus `--apply`. Four images are literal references in the current draft;
`curvature_vs_G.png` is retained as a related fifth review candidate. Do not
promote any of them merely to satisfy an old generated-site path.

The atlas remains staged and unpublished. The top-level WebGL demo, alternate
WASM experiment, and all 3D outputs remain archive-only in this phase.

## SPH browser demo

The tracked browser closure is `website/SPH_demo/hydro_sph_WASM.html`,
`sph.js`, `sph.wasm`, and `sph_snapshot.json`. The HTML loads the JavaScript
glue and snapshot, while the JavaScript glue loads the matching WebAssembly
binary.

Rebuilding requires an existing Emscripten `emcc` installation:

```bash
bash scripts/sph/build_sph.sh
```

The script compiles `scripts/sph/sph_core.cpp` and stages `sph.js` and
`sph.wasm` under `work/promote/SPH_demo/` by default. Set `SPH_PROMOTE_DIR` to
another review directory when needed. Review both generated files as one
runtime unit before explicitly promoting them into `website/SPH_demo/`.

## Optional legacy JSON import

`scripts/mandelbrot/components/catalogue/migrate_json_catalogue.py` and
`test_migration.py` are optional migration/provenance tools for someone who
already owns the legacy JSON catalogue. Normal initialization, scanners,
hunters, classifiers, wrappers, and tests do not use them.
