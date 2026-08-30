#!/usr/bin/env python3
"""Compatibility launcher for the authoritative C++ area scanner.

The repository previously carried a second Python implementation with its own
CSV/JSON persistence rules.  That duplicated the numerical algorithm and,
more importantly, bypassed the typed catalogue API.  The C++ scanner is now the
single implementation; this entry point simply forwards its arguments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
BINARY = CODE_ROOT / "bin" / "component_area_scan"


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(
            f"Missing {BINARY}. Build the repository first with {CODE_ROOT / 'build.sh'}."
        )
    os.execv(str(BINARY), [str(BINARY), *sys.argv[1:]])


if __name__ == "__main__":
    main()
