# Deferred generator dependency locking

The existing `Gemfile` and `Gemfile.lock` are preserved as the current Jekyll
dependency evidence. The repository reconstruction did not install dependencies
or change the lock file.

Mandelbrot source evidence currently includes `scripts/mandelbrot/to_install.txt`,
`build.sh`, imports, headers, and retained repository notes. Known requirements include
Python, NumPy, Pillow, Matplotlib, a C++ compiler, Boost headers, and SQLite
headers/library.

Selecting a Python/C++ package manager and committing a new lock are deferred
until the user approves an environment policy. No speculative `pyproject.toml`
or lock file was created.
