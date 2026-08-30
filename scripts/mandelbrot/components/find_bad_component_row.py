#!/usr/bin/env python3
"""Inspect unconverged area measurements through the typed scan store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument
from components.catalogue.component_catalogue import Catalogue, decimal_string


def main() -> int:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    run_name = args.run_name or repo.string("component_area_scan.run_name", "default")
    catalogue = Catalogue(repo.path("paths.catalogue_root"))
    store = catalogue.area_scan_store(run_name)
    bad = [
        row for row in store.load_measurements()
        if row.period == args.period and not row.converged
    ]

    print(f"catalogue: {catalogue.root}")
    print(f"run: {run_name}")
    print(f"period: {args.period}")
    print(f"unconverged rows: {len(bad)}")
    print("affected components:", sorted({row.component_index for row in bad}))
    for row in bad:
        print("\n" + "-" * 80)
        values = {
            "component_index": row.component_index,
            "conjugate_index": row.conjugate_index,
            "center": f"[{decimal_string(row.center.re)}, {decimal_string(row.center.im)}]",
            "rho": decimal_string(row.rho),
            "theta_points": row.theta_points,
            "area_estimate": decimal_string(row.area_estimate),
            "resolution_delta": decimal_string(row.resolution_delta),
            "closure_error": decimal_string(row.closure_error),
            "marked_z_closure_error": decimal_string(row.marked_z_closure_error),
            "max_residual": decimal_string(row.max_residual),
            "failed_attempts": row.failed_attempts,
            "max_subdivision_depth": row.max_subdivision_depth,
            "rejected_branch_candidates": row.rejected_branch_candidates,
            "cyclic_seed_attempts": row.cyclic_seed_attempts,
            "cyclic_recoveries": row.cyclic_recoveries,
            "failure_reason": row.failure_reason,
        }
        for name, value in values.items():
            print(f"{name:30s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
