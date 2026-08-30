#!/usr/bin/env python3
"""Verify binary Mandelbrot contour headers and file sizes."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument


def verify(path: Path) -> bool:
    print(f"\n{path}")
    if not path.is_file():
        print("exists: false")
        return False

    size = path.stat().st_size
    with path.open("rb") as handle:
        magic = handle.read(8)
        header = handle.read(16)
        count_raw = handle.read(8)
    if len(header) != 16 or len(count_raw) != 8:
        print(f"size: {size}")
        print("valid header: false")
        return False

    version, code, value_size, endian = struct.unpack("<IIII", header)
    (count,) = struct.unpack("<Q", count_raw)
    expected = 32 + count * 2 * value_size
    matches = size == expected

    print(f"size: {size}")
    print(f"magic: {magic!r}")
    print(f"version: {version}  code: {code}  value_size: {value_size}  endian: {endian:#x}")
    print(f"count: {count}")
    print(f"expected size: {expected}")
    print(f"size matches: {str(matches).lower()}")
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Contour .bin files. Defaults to all files in <contours.output_dir>/contours_bin.",
    )
    args = parser.parse_args()

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    if args.files:
        paths = [
            (path if path.is_absolute() else repo.paths.code_root / path).resolve()
            for path in args.files
        ]
    else:
        paths = sorted((repo.path("contours.output_dir") / "contours_bin").glob("*.bin"))
        if not paths:
            raise SystemExit("No contour binaries found. Pass file paths explicitly or run ./bin/contours first.")

    valid = sum(verify(path) for path in paths)
    print(f"\nverified: {valid}/{len(paths)}")
    return 0 if valid == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
