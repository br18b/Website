# Mandelbrot private data root

Catalogue, contour, run, restart, and export state is local generated work—not
a public website or Git asset.

Resolution precedence is:

1. a tool's explicit CLI path, when supported;
2. `MANDELBROT_DATA_ROOT`;
3. `<repository>/work/mandelbrot`.

Relative environment or API overrides are anchored at the repository root,
not at the process working directory. The Python and C++ `RepoConfig` modules
implement the same rule and expand `$data_root` in `mandelbrot.json`.

Derived defaults are:

```text
work/mandelbrot/component_catalogue/
work/mandelbrot/component_catalogue/runs/
work/mandelbrot/component_catalogue/exports/
work/mandelbrot/G_contours/
work/promote/mandelbrot/
```

For an external writable root:

```bash
export MANDELBROT_DATA_ROOT=/path/chosen/by/the/user
```

That root may point at compatible resumed local state, but the reconstruction
does not copy any database, WAL/SHM, contour, checkpoint, or restart file.
