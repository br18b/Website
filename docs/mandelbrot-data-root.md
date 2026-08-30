# Mandelbrot private data root

Catalogue, contour, run, restart, and export state is local generated work—not
a public website or Git asset. The authoritative active root is normally:

```text
<repository>/work/mandelbrot
```

Keeping this state inside the repository working tree makes the whole
workspace movable. Tools discover the repository from their own source or
executable location and a `.git` (directory or file) or `.root` marker; they do
not derive paths from the shell's current directory.

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

`MANDELBROT_DATA_ROOT` remains an advanced escape hatch for an intentionally
external or isolated catalogue:

```bash
export MANDELBROT_DATA_ROOT=/path/chosen/by/the/user
```

Do not set it for ordinary operation. The portable operations wrapper clears
an inherited value and uses the repository-local root unless `--data-root` is
supplied explicitly.

A new clone can initialize a new empty catalogue. To restore an existing
active catalogue, restore its complete consistent state beneath
`work/mandelbrot/`, including required checkpoints and contours. A SQLite main
file copied without meaningful WAL state is not a complete backup.

`work/` is ignored, but ignore rules do not protect it from filesystem loss or
destructive cleanup. Maintain ordinary backups and never run `git clean -fdx`
in a workspace containing valuable local state.
