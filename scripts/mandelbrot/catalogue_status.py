#!/usr/bin/env python3
"""Report catalogue state without initializing or modifying it."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.repo_config import RepoConfig  # noqa: E402


def file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sidecar_state(database: Path) -> dict[str, dict[str, Any]]:
    return {
        "database": file_state(database),
        "wal": file_state(Path(f"{database}-wal")),
        "shm": file_state(Path(f"{database}-shm")),
    }


def compact_periods(periods: list[int]) -> str:
    if not periods:
        return "none"
    ranges: list[str] = []
    first = previous = periods[0]
    for value in periods[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = value
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def read_checkpoint(path: Path) -> dict[str, Any]:
    state = file_state(path)
    if not state["exists"]:
        return state
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        header = [handle.readline().rstrip("\n") for _ in range(6)]
    values: dict[str, Any] = {"exists": True, "path": str(path), **state}
    for line in header:
        key, separator, value = line.partition(" ")
        if separator and key in {"version", "representation", "period", "iteration", "count"}:
            values[key] = int(value) if key in {"version", "period", "iteration", "count"} else value
    return values


def scalar(connection: sqlite3.Connection, sql: str) -> Any:
    row = connection.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no row: {sql}")
    return row[0]


def resolve(args: argparse.Namespace) -> tuple[RepoConfig, Path, Path, Path]:
    config = RepoConfig.load(
        args.config,
        start=SCRIPT_DIR,
        data_root=args.data_root,
    )
    catalogue_root = config.path("paths.catalogue_root")
    database = catalogue_root / "component_catalogue.sqlite"
    run_name = config.string("component_area_scan.run_name", "default")
    run_root = catalogue_root / "runs" / "area_scan" / run_name
    return config, catalogue_root, database, run_root


def path_report(
    config: RepoConfig,
    catalogue_root: Path,
    database: Path,
    run_root: Path,
) -> dict[str, Any]:
    return {
        "repository": str(config.paths.project_root),
        "code_root": str(config.paths.code_root),
        "data_root": str(config.paths.data_root),
        "catalogue_root": str(catalogue_root),
        "database": str(database),
        "run_root": str(run_root),
        "config": str(config.paths.config_path),
        "period_start": config.integer("component_area_scan.period_start"),
        "period": config.integer("component_area_scan.period"),
        "resume": config.boolean("component_area_scan.resume"),
        "reset_root_checkpoint": config.boolean(
            "component_area_scan.reset_root_checkpoint"
        ),
    }


def catalogue_report(
    config: RepoConfig,
    catalogue_root: Path,
    database: Path,
    run_root: Path,
) -> dict[str, Any]:
    if not database.is_file():
        raise FileNotFoundError(
            f"Catalogue database does not exist; read-only status will not initialize it: {database}"
        )
    before = sidecar_state(database)
    wal_aware = bool(before["wal"]["exists"] and before["wal"].get("size", 0) > 0)
    if wal_aware and not before["shm"]["exists"]:
        raise RuntimeError(
            "A WAL exists without its SHM sidecar; refusing a read that could require sidecar creation"
        )

    # A freshly restored logical backup may retain WAL journal-mode bits while
    # having no WAL. Opening that file with plain mode=ro makes SQLite create
    # empty sidecars. Immutable mode is safe only in this no-meaningful-WAL
    # case. When a non-empty WAL exists, plain mode=ro is required to include it.
    uri = database.resolve().as_uri() + ("?mode=ro" if wal_aware else "?mode=ro&immutable=1")
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        if scalar(connection, "PRAGMA query_only") != 1:
            raise RuntimeError("SQLite query_only could not be enabled")
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key, value FROM catalogue_meta ORDER BY key"))
        manifest_text = scalar(
            connection,
            "SELECT record_json FROM catalogue_manifest WHERE singleton=1",
        )
        manifest = json.loads(manifest_text)
        tables = (
            "components",
            "component_records",
            "period_records",
            "hierarchy_records",
            "area_scan_centers",
            "area_scan_measurements",
            "area_scan_summaries",
            "json_migrations",
        )
        counts = {
            table: int(scalar(connection, f"SELECT COUNT(*) FROM {table}"))
            for table in tables
        }
        periods = [int(row[0]) for row in connection.execute(
            "SELECT period FROM period_records ORDER BY period"
        )]
        runs = [
            {
                "name": row[0],
                "centers": row[1],
                "min_period": row[2],
                "max_period": row[3],
            }
            for row in connection.execute(
                "SELECT run_name, COUNT(*), MIN(period), MAX(period) "
                "FROM area_scan_centers GROUP BY run_name ORDER BY run_name"
            )
        ]
        user_version = int(scalar(connection, "PRAGMA user_version"))
        journal_mode = str(scalar(connection, "PRAGMA journal_mode"))
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()

    after = sidecar_state(database)
    if wal_aware:
        stable_before = {"database": before["database"], "wal": before["wal"]}
        stable_after = {"database": after["database"], "wal": after["wal"]}
        same_sidecar_set = before["shm"]["exists"] == after["shm"]["exists"]
        if stable_before != stable_after or not same_sidecar_set:
            raise RuntimeError("Database/WAL metadata or sidecar presence changed during status")
    elif before != after:
        raise RuntimeError("Database or SQLite sidecar metadata changed during immutable status")

    checkpoints = []
    root_checkpoints = run_root / "root_checkpoints"
    if root_checkpoints.is_dir():
        for path in sorted(root_checkpoints.glob("period_*.chk")):
            checkpoints.append(read_checkpoint(path))

    return {
        "paths": path_report(config, catalogue_root, database, run_root),
        "sqlite": {
            "library_version": sqlite3.sqlite_version,
            "journal_mode": journal_mode,
            "query_only": True,
            "access_mode": "wal-aware-read-only" if wal_aware else "immutable-main-no-wal",
            "user_version": user_version,
            "files": after,
        },
        "schema": metadata,
        "manifest": {
            "schema": manifest.get("schema"),
            "catalogue_revision": manifest.get("catalogueRevision"),
            "exact_through_period": manifest.get("exactThroughPeriod"),
            "updated_at": manifest.get("updatedAt"),
        },
        "counts": counts,
        "complete_periods": periods,
        "complete_period_ranges": compact_periods(periods),
        "runs": runs,
        "checkpoints": checkpoints,
    }


def print_human(report: dict[str, Any], *, show_paths: bool = True) -> None:
    paths = report["paths"]
    if show_paths:
        print(f"Repository:       {paths['repository']}")
        print(f"Data root:        {paths['data_root']}")
        print(f"Database:         {paths['database']}")
        print(f"Run root:         {paths['run_root']}")
        print(f"Config:           {paths['config']}")
    print(f"Storage schema:   {report['schema'].get('storage_schema', 'unknown')}")
    print(f"SQLite revision:  {report['sqlite']['user_version']}")
    print(f"Catalogue rev:    {report['manifest']['catalogue_revision']}")
    print(f"Components:       {report['counts']['components']}")
    print(f"Centers:          {report['counts']['area_scan_centers']}")
    print(f"Measurements:     {report['counts']['area_scan_measurements']}")
    print(f"Complete periods: {report['complete_period_ranges']}")
    print(f"Exact through:    {report['manifest']['exact_through_period']}")
    for run in report["runs"]:
        print(
            f"Run:              {run['name']} "
            f"centers={run['centers']} periods={run['min_period']}-{run['max_period']}"
        )
    if not report["checkpoints"]:
        print("Checkpoints:      none")
    for checkpoint in report["checkpoints"]:
        print(
            f"Checkpoint:       period={checkpoint.get('period', '?')} "
            f"iteration={checkpoint.get('iteration', '?')} "
            f"representation={checkpoint.get('representation', '?')}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Read a Mandelbrot SQLite catalogue without initializing or modifying it."
    )
    result.add_argument("--config", type=Path, default=None)
    result.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Advanced explicit data-root override",
    )
    result.add_argument("--json", action="store_true")
    result.add_argument(
        "--no-paths",
        action="store_true",
        help="Suppress local absolute paths in displayed status output",
    )
    result.add_argument(
        "--paths-only",
        action="store_true",
        help="Resolve and print paths without opening SQLite",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, catalogue_root, database, run_root = resolve(args)
    if args.paths_only:
        paths = path_report(config, catalogue_root, database, run_root)
        if args.json:
            print(json.dumps(paths, indent=2, sort_keys=True))
        else:
            for key, value in paths.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                print(f"{key}\t{value}")
        return 0
    report = catalogue_report(config, catalogue_root, database, run_root)
    if args.json:
        if args.no_paths:
            report = {key: value for key, value in report.items() if key != "paths"}
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report, show_paths=not args.no_paths)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, sqlite3.Error, ValueError) as error:
        print(f"catalogue status failed: {error}", file=sys.stderr)
        raise SystemExit(1)
