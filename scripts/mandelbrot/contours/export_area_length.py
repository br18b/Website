#!/usr/bin/env python3
"""
Export Mandelbrot contour scaling data for Mathematica/Wolfram Language.

Reads contour_scaling.csv and writes:
  - area_points.wl
  - length_points.wl
  - area_length_points.wl
  - area_length_points.json

Usage:
  python3 contours/export_area_length.py
  python3 contours/export_area_length.py --csv G_contours/contour_scaling.csv
  python3 contours/export_area_length.py --csv G_contours/contour_scaling.csv --out G_contours/fit_data
  python3 contours/export_area_length.py --config experiments/contours.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument


def pick_column(fieldnames: Iterable[str], candidates: list[str], contains: list[str] | None = None) -> str:
    fields = list(fieldnames)
    lower_map = {f.lower(): f for f in fields}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if contains:
        for f in fields:
            lf = f.lower()
            if all(token.lower() in lf for token in contains):
                return f

    raise KeyError(f"Could not find a column matching {candidates}; available columns: {fields}")


def finite_float(value: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value: {value!r}")
    return x


def wolfram_number(x: float) -> str:
    # Mathematica accepts scientific notation as 1.23*^-6, not 1.23e-6.
    s = f"{x:.17g}"
    s = re.sub(r"e([+-]?\d+)", r"*^\1", s, flags=re.IGNORECASE)
    return s


def wolfram_pairs(name: str, pairs: list[tuple[float, float]]) -> str:
    body = ",\n  ".join(
        "{" + wolfram_number(x) + ", " + wolfram_number(y) + "}"
        for x, y in pairs
    )
    return f"{name} = {{\n  {body}\n}};\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--csv", dest="csv_path", help="Path to contour_scaling.csv")
    ap.add_argument("--out", dest="out_dir", help="Output directory; default: beside contour_scaling.csv")
    ap.add_argument("--sort", choices=["asc", "desc", "none"], default="asc", help="Sort by G")
    args = ap.parse_args()

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser()
        if not csv_path.is_absolute():
            csv_path = repo.paths.code_root / csv_path
        csv_path = csv_path.resolve()
    else:
        csv_path = repo.path("contours.output_dir") / "contour_scaling.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else csv_path.parent / "fit_data"
    if not out_dir.is_absolute():
        out_dir = repo.paths.code_root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV header in {csv_path}")

        g_col = pick_column(reader.fieldnames, ["G", "target_G", "g"])
        area_col = pick_column(reader.fieldnames, ["area", "Area"])
        length_col = pick_column(reader.fieldnames, ["length", "Length"])

        area: list[tuple[float, float]] = []
        length: list[tuple[float, float]] = []

        for row in reader:
            try:
                G = finite_float(row[g_col])
                A = finite_float(row[area_col])
                L = finite_float(row[length_col])
            except Exception:
                continue
            area.append((G, A))
            length.append((G, L))

    if args.sort != "none":
        reverse = args.sort == "desc"
        area.sort(key=lambda pair: pair[0], reverse=reverse)
        length.sort(key=lambda pair: pair[0], reverse=reverse)

    (out_dir / "area_points.wl").write_text(wolfram_pairs("areaData", area), encoding="utf-8")
    (out_dir / "length_points.wl").write_text(wolfram_pairs("lengthData", length), encoding="utf-8")
    (out_dir / "area_length_points.wl").write_text(
        wolfram_pairs("areaData", area) + "\n" + wolfram_pairs("lengthData", length),
        encoding="utf-8",
    )
    (out_dir / "area_length_points.json").write_text(
        json.dumps(
            {
                "source_csv": str(csv_path),
                "columns": {"G": g_col, "area": area_col, "length": length_col},
                "area": area,
                "length": length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"read: {csv_path}")
    print(f"points: {len(area)}")
    for name in ("area_points.wl", "length_points.wl", "area_length_points.wl", "area_length_points.json"):
        print(f"wrote: {out_dir / name}")


if __name__ == "__main__":
    main()
