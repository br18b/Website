#!/usr/bin/env python3
"""Promote reviewed browser assets from work/promote using an exact allow-list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


ALLOWED_SUFFIXES = {
    ".css",
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".png",
    ".svg",
    ".wasm",
    ".webp",
}
REJECTED_SUFFIXES = {
    ".a",
    ".bin",
    ".checkpoint",
    ".db",
    ".dll",
    ".dylib",
    ".exe",
    ".lock",
    ".npy",
    ".npz",
    ".o",
    ".obj",
    ".pyc",
    ".restart",
    ".shm",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".stamp",
    ".wal",
}
REJECTED_PARTS = {
    ".buildmeta",
    ".cache",
    ".locks",
    ".objects",
    "__pycache__",
    "cache",
    "checkpoints",
    "component_catalogue",
    "contours_bin",
    "exports",
    "g_contours",
    "restart",
    "runs",
}


@dataclass(frozen=True)
class Promotion:
    source: Path
    destination: Path
    sha256: str
    required_destinations: tuple[Path, ...]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(raw: str, field: str) -> Path:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a non-empty relative path: {raw!r}")
    return path


def validate_browser_asset(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & REJECTED_PARTS:
        raise ValueError(f"work-state path is not promotable: {path}")

    lowered_name = path.name.lower()
    if lowered_name.endswith(("-wal", "-shm")):
        raise ValueError(f"SQLite sidecar is not promotable: {path}")
    if path.suffix.lower() in REJECTED_SUFFIXES:
        raise ValueError(f"compiled/calculation artifact is not promotable: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"file type is not on the browser allow-list: {path}")

    # The reviewed name data.json is permitted. Hash-sharded component records
    # and catalogue manifests remain private even though they also use JSON.
    if path.suffix.lower() == ".json":
        stem = path.stem.lower()
        if len(stem) >= 32 and all(character in "0123456789abcdef" for character in stem):
            raise ValueError(f"hash-sharded component JSON is not promotable: {path}")


def load_allow_list(path: Path) -> list[Promotion]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"source_relative", "destination_relative", "sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"allow-list must contain columns: {sorted(required)}")
        entries = []
        for row_number, row in enumerate(reader, start=2):
            source = safe_relative(row["source_relative"], f"row {row_number} source")
            destination = safe_relative(
                row["destination_relative"], f"row {row_number} destination"
            )
            expected = row["sha256"].strip().lower()
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                raise ValueError(f"row {row_number} has an invalid SHA-256")
            validate_browser_asset(source)
            validate_browser_asset(destination)
            companions = tuple(
                safe_relative(value.strip(), f"row {row_number} companion")
                for value in row.get("required_destinations", "").split(";")
                if value.strip()
            )
            for companion in companions:
                validate_browser_asset(companion)
            entries.append(Promotion(source, destination, expected, companions))
    if not entries:
        raise ValueError("allow-list is empty")
    if len({entry.destination for entry in entries}) != len(entries):
        raise ValueError("allow-list contains duplicate destination paths")
    destinations = {entry.destination for entry in entries}
    for entry in entries:
        missing = set(entry.required_destinations) - destinations
        if missing:
            raise ValueError(
                f"{entry.destination} requires unlisted companions: "
                + ", ".join(str(path) for path in sorted(missing))
            )
    return entries


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-list", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=root / "work" / "promote")
    parser.add_argument("--destination-root", type=Path, default=root)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="copy verified files; without this flag the command is validation-only",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="with --apply, permit replacement of an existing reviewed file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = load_allow_list(args.allow_list.resolve())
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()

    verified: list[tuple[Promotion, Path, Path]] = []
    for entry in entries:
        source = (source_root / entry.source).resolve()
        destination = (destination_root / entry.destination).resolve()
        if source_root not in source.parents or destination_root not in destination.parents:
            raise ValueError(f"resolved path escapes its root: {entry}")
        if not source.is_file():
            raise FileNotFoundError(f"staged file is missing: {source}")
        actual = sha256(source)
        if actual != entry.sha256:
            raise ValueError(f"checksum mismatch for {entry.source}: {actual}")
        if args.apply and destination.exists() and not args.replace:
            raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
        if destination.exists() and not destination.is_file():
            raise FileExistsError(f"destination is not a regular file: {destination}")
        verified.append((entry, source, destination))

    action = "validated"
    if args.apply:
        for _, source, destination in verified:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        action = "promoted"

    print(f"{action} {len(verified)} explicitly allowed browser assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
