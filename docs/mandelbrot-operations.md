# Mandelbrot catalogue operations

## Workspace model

Tracked implementation source is under `scripts/mandelbrot/`. Active generated
catalogue state is ignored by Git but remains inside the same working tree at
`work/mandelbrot/`. Public files are separate again under `website/`; a
scientific tool cannot publish merely by writing private state.

The normal paths are:

```text
work/mandelbrot/component_catalogue/component_catalogue.sqlite
work/mandelbrot/component_catalogue/runs/
work/mandelbrot/component_catalogue/exports/
work/mandelbrot/G_contours/
work/mandelbrot/config/
work/mandelbrot/logs/
work/promote/mandelbrot/
```

The repository can move as a unit. Python, C++, and shell entry points search
upward from their own location for a `.git` directory, a `.git` file (as used
by worktrees), or a `.root` marker. Failure to find a marker is an error; the
tools do not silently choose the current directory.

`MANDELBROT_DATA_ROOT` is an optional advanced override, not part of the normal
workflow. The operations facade clears an inherited value and uses
`<repository>/work/mandelbrot`. Its explicit `--data-root` option is the visible
way to opt into another root.

## Build and lightweight tests

These commands may be invoked from any current directory by using the path to
the repository:

```bash
/path/to/repository/scripts/mandelbrot/ops.sh build
/path/to/repository/scripts/mandelbrot/ops.sh test
```

Build products remain under ignored `scripts/mandelbrot/bin/`. Tests use
temporary directories and do not populate the active scientific catalogue.
The required toolchain is a C++20 compiler, Boost headers, SQLite development
headers/library and Python 3. Python plotting and demo stages additionally use
the packages listed in `scripts/mandelbrot/to_install.txt`; dependency versions
are not yet locked.

## Read-only status

```bash
./scripts/mandelbrot/ops.sh status
./scripts/mandelbrot/ops.sh status --json
```

Status refuses to initialize a missing database. It opens the existing SQLite
database with URI `mode=ro`, enables `query_only`, reads visible WAL state, and
reports the storage/catalogue revisions, counts, complete periods, exact-through
period, run identity, and root-checkpoint headers. For a consistent restored
main database with no non-empty WAL, it also sets `immutable=1` so SQLite cannot
create empty WAL/SHM sidecars merely because the database header records WAL
mode. Once a non-empty WAL exists, it requires the corresponding pre-existing
SHM and uses ordinary read-only WAL access; SQLite may take transient shared
read locks and update its existing shared-memory read marks, but it does not
create sidecars or alter the database/WAL content. Status never checkpoints,
vacuums, optimizes, migrates, or creates schema. A non-empty WAL without its SHM
sidecar is treated as an unsafe/incomplete state and rejected.

## Resume profile and dry plan

The tracked `scripts/mandelbrot/mandelbrot.json` describes a fresh run. Never
change its start period merely to operate one local catalogue. Create an
ignored profile under `work/mandelbrot/config/` with the same numerical values
and catalogue paths, changing only resume-sensitive and log-friendly progress
values. For a period-22 root checkpoint those values are:

```json
{
  "component_area_scan": {
    "period_start": 22,
    "period": 22,
    "resume": true,
    "reset_root_checkpoint": false,
    "progress_style": "lines",
    "progress_screen": "normal"
  }
}
```

The actual profile is a complete copy of the tracked configuration, not the
partial illustration above. First resolve and validate every target without
launching the scanner:

```bash
./scripts/mandelbrot/ops.sh resume \
  --config work/mandelbrot/config/resume-period22.json \
  --plan
```

Review the printed repository, data root, database, run root, configuration,
period range, and scanner path. Then obtain a fresh read-only status report.

## Start, monitor, stop, and restart

Start only after the profile, checkpoint, free space, backup, and status have
been reviewed:

```bash
./scripts/mandelbrot/ops.sh resume \
  --config work/mandelbrot/config/resume-period22.json
```

The foreground wrapper acquires a nonblocking single-writer lock under
`work/mandelbrot/.locks/`, prints all resolved paths, and tees output to a
timestamped file under `work/mandelbrot/logs/`. A second wrapper refuses to
start while the lock is held.

Monitor the terminal and, in another terminal, follow the printed log with
`tail -f`. Progress is indicated by changing iteration/checkpoint messages,
growing log output, and updated checkpoint metadata—not by high CPU usage
alone.

Stop with Ctrl+C once and wait for the foreground process to exit. Do not kill
the terminal, power off, or start another writer while it is stopping. After
exit, run `ops.sh status`, inspect the latest log, and verify the checkpoint
header. Restart by issuing the exact same resume command; the scanner reads the
existing root checkpoint because `resume` is true and
`reset_root_checkpoint` is false.

## Backups and destructive-cleanup warning

`work/` is ignored by Git. That prevents accidental commits, but it provides no
backup and no deletion protection. Never run `git clean -fdx` in a workspace
that contains valuable work state.

Before backing up an active catalogue, stop its writer and verify that no
process holds the database, WAL, SHM, checkpoint, or contour files. Use a
WAL-aware SQLite backup operation to create a consistent destination database;
do not copy only the main SQLite file when a WAL exists. Back up current
`component_catalogue/runs/`, `component_catalogue/exports/`, and `G_contours/`
with their manifests after the database backup. Validate integrity, logical
counts, checkpoint identity, hashes, and restore instructions before treating
the backup as usable.

A new clone can build the tools and initialize an empty catalogue. To continue
an existing catalogue after moving or recloning, restore the validated state
beneath the new checkout's `work/mandelbrot/`; tracked source alone cannot
recreate checkpoints or costly contour results.
