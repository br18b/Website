#!/usr/bin/env python3
"""Write a disposable atlas export from typed canonical component records.

No catalogue database or CSV file is opened directly here. Selection, decimal parsing,
component loading, and JSON serialization are all delegated to ``Catalogue``.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument
from components.catalogue.component_catalogue import Catalogue, ComponentQuery


def main() -> int:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-period", type=int, default=None)
    parser.add_argument("--min-area", type=Decimal, default=None)
    # Kept as a harmless compatibility flag for old commands.
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    demo = repo.section("demo.atlas")
    component_cfg = demo.get("components", {})
    catalogue = Catalogue(repo.path("paths.catalogue_root"))
    catalogue.ensure_layout()

    output = args.output or catalogue.export_path("atlas_components.json")
    if not output.is_absolute():
        output = (repo.paths.code_root / output).resolve()
    max_period = args.max_period or int(component_cfg.get("max_period", 50))
    min_area = args.min_area
    if min_area is None:
        min_area = Decimal(str(component_cfg.get("min_area", 1.0e-10)))

    query = ComponentQuery(
        min_period=1,
        max_period=max_period,
        min_area=min_area,
        require_polygon=True,
        require_polygon_converged=True,
    )
    selected = catalogue.query_components(query)
    catalogue.write_component_export(
        output,
        query=query,
        format_name="mandelbrot-atlas-geometry-v3",
        complete=False,
        coordinate_digits=int(component_cfg.get("coordinate_digits", 11)),
    )
    print(f"catalogue: {catalogue.root}")
    print(f"selected:  {len(selected)} canonical representative(s)")
    print(f"output:    {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
