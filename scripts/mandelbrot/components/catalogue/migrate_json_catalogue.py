#!/usr/bin/env python3
"""Idempotently import a JSON catalogue into the SQLite catalogue.

The recommended conversion is in place: SOURCE_ROOT and DESTINATION_ROOT are
the same directory.  The legacy files are left untouched as a rollback copy,
while ``component_catalogue.sqlite`` becomes authoritative.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.catalogue.component_catalogue import (
    AreaScanCenterRecord,
    Catalogue,
    ComplexValue,
    HierarchyNode,
    HierarchyTree,
    Manifest,
    PeriodRecord,
    decimal,
    utc_timestamp,
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path: Path) -> Manifest:
    data = read_json(path)
    return Manifest(
        catalogue_revision=int(data["catalogueRevision"]),
        family=data["family"],
        canonical_half_plane=data["canonicalHalfPlane"],
        component_count_stored=int(data["componentCountStored"]),
        component_count_with_symmetry=int(
            data["componentCountWithSymmetry"]
        ),
        minimum_area=decimal(data["minimumArea"]),
        exact_through_period=int(data["exactThroughPeriod"]),
        created_at=data["createdAt"],
        updated_at=data["updatedAt"],
        software_revision=data.get("softwareRevision", ""),
    )


def read_period(path: Path) -> PeriodRecord:
    data = read_json(path)
    return PeriodRecord(
        period=int(data["period"]),
        theoretical_component_count=data.get(
            "theoreticalComponentCount", ""
        ),
        known_representative_count=int(data["knownRepresentativeCount"]),
        known_component_count_with_symmetry=int(
            data["knownComponentCountWithSymmetry"]
        ),
        catalogue_complete=bool(data["catalogueComplete"]),
        known_area=decimal(data["knownArea"]),
        known_area_error=decimal(data["knownAreaError"]),
        area_cutoff=decimal(data["areaCutoff"]),
        exact_geometry_complete=bool(
            data.get("exactGeometryComplete", False)
        ),
        polygon_rho=decimal(data.get("polygonRho", "0")),
        area_rho=decimal(data.get("areaRho", "0")),
        polygon_points=int(data.get("polygonPoints", 0)),
        component_ids=list(data.get("componentIds", [])),
        generated_from_catalogue_revision=int(
            data["generatedFromCatalogueRevision"]
        ),
    )


def read_hierarchy(path: Path) -> HierarchyTree:
    data = read_json(path)
    statistics = data["statistics"]
    return HierarchyTree(
        root=data["root"],
        nodes=[
            HierarchyNode(
                node["id"],
                node.get("parent"),
                list(node.get("children", [])),
            )
            for node in data.get("nodes", [])
        ],
        node_count=int(statistics["nodeCount"]),
        maximum_known_generation=int(
            statistics["maximumKnownGeneration"]
        ),
        known_area=decimal(statistics["knownArea"]),
        minimum_stored_area=decimal(statistics["minimumStoredArea"]),
        complete_above_cutoff=bool(
            statistics.get("completeAboveCutoff", False)
        ),
        generated_from_catalogue_revision=int(
            data["generatedFromCatalogueRevision"]
        ),
    )


def import_area_scan(
    source_root: Path,
    catalogue: Catalogue,
    run_name: str,
) -> tuple[int, int, int]:
    store = catalogue.area_scan_store(run_name)
    exports = source_root / "exports"
    centers: dict[int, list[AreaScanCenterRecord]] = {}
    centers_path = exports / "centers.csv"
    if centers_path.is_file():
        with centers_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record = AreaScanCenterRecord(
                    period=int(row["period"]),
                    component_index=int(row["component_index"]),
                    expected_period_count=int(row["expected_period_count"]),
                    center=ComplexValue(
                        decimal(row["center_re"]),
                        decimal(row["center_im"]),
                    ),
                    center_residual=decimal(row["center_residual"]),
                    detected_exact_period=int(row["detected_exact_period"]),
                    conjugate_index=int(row["conjugate_index"]),
                    center_newton_iterations=int(
                        row.get("center_newton_iterations") or 0
                    ),
                    center_refinement_method=(
                        row.get("center_refinement_method")
                        or "cpp-long-double"
                    ),
                    center_refinement_dps=int(
                        row.get("center_refinement_dps") or 0
                    ),
                )
                centers.setdefault(record.period, []).append(record)
        store.save_centers(centers)

    measurements = []
    measurements_path = exports / "components.csv"
    if measurements_path.is_file():
        with measurements_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                measurements.append(store._measurement_from_row(row))
        store.save_measurements(measurements)

    summaries = []
    summaries_path = exports / "period_summary.csv"
    if summaries_path.is_file():
        with summaries_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                summaries.append(store._summary_from_row(row))
        store.save_summaries(summaries)

    center_count = sum(len(records) for records in centers.values())
    store.close()
    return center_count, len(measurements), len(summaries)


def migrate(
    source_root: Path,
    destination_root: Path,
    *,
    run_name: str,
    batch_size: int,
    skip_area_scan: bool,
    copy_run_state: bool,
) -> None:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    component_dir = source_root / "catalogue" / "components"
    manifest_path = source_root / "manifest.json"
    if not component_dir.is_dir() or not manifest_path.is_file():
        raise SystemExit(
            f"{source_root} is not a legacy JSON component catalogue"
        )

    source_manifest = read_manifest(manifest_path)
    component_paths = sorted(component_dir.rglob("*.json"))
    catalogue = Catalogue(destination_root)
    catalogue.ensure_layout()

    seen_ids: set[str] = set()
    exact_identities: dict[tuple[int, int], str] = {}
    imported = 0
    batch = []
    for path in component_paths:
        component = Catalogue._component_from_json(read_json(path))
        if component.id in seen_ids:
            raise RuntimeError(
                f"Duplicate component UUID in JSON source: {component.id}"
            )
        seen_ids.add(component.id)
        exact_identity = Catalogue.exact_period_index(component)
        if exact_identity is not None:
            key = (
                exact_identity.period,
                exact_identity.component_index,
            )
            previous = exact_identities.get(key)
            if previous is not None and previous != component.id:
                raise RuntimeError(
                    "Duplicate exact-scanner identity in JSON source: "
                    f"{key} belongs to both {previous} and {component.id}"
                )
            exact_identities[key] = component.id
        batch.append(component)
        if len(batch) >= batch_size:
            catalogue.save_components(batch, bump_revision=False)
            imported += len(batch)
            batch.clear()
            print(f"\rcomponents: {imported}/{len(component_paths)}", end="")
    if batch:
        catalogue.save_components(batch, bump_revision=False)
        imported += len(batch)
    print(f"\rcomponents: {imported}/{len(component_paths)}")

    period_paths = sorted(
        (source_root / "catalogue" / "periods").glob("*.json")
    )
    for path in period_paths:
        catalogue.save_period(read_period(path))
    print(f"period records: {len(period_paths)}")

    hierarchy_paths = sorted(
        (source_root / "catalogue" / "hierarchies").glob("*.json")
    )
    for path in hierarchy_paths:
        catalogue.save_hierarchy(read_hierarchy(path))
    print(f"hierarchy records: {len(hierarchy_paths)}")

    if skip_area_scan:
        area_counts = (0, 0, 0)
    else:
        area_counts = import_area_scan(source_root, catalogue, run_name)
    print(
        "area scanner rows: "
        f"centers={area_counts[0]}, measurements={area_counts[1]}, "
        f"summaries={area_counts[2]}"
    )

    if (
        copy_run_state
        and source_root != destination_root
        and (source_root / "runs").is_dir()
    ):
        shutil.copytree(
            source_root / "runs",
            destination_root / "runs",
            dirs_exist_ok=True,
        )
        print("copied algorithm checkpoint files")

    # Publishing the source manifest is deliberately last.  An interrupted
    # import therefore remains obviously incomplete and can simply be rerun.
    catalogue.save_manifest(source_manifest)
    with catalogue._connection:
        catalogue._connection.execute(
            """
            INSERT INTO json_migrations(
                source_root, source_catalogue_revision,
                component_count, completed_at
            ) VALUES(?, ?, ?, ?)
            ON CONFLICT(source_root) DO UPDATE SET
                source_catalogue_revision =
                    excluded.source_catalogue_revision,
                component_count = excluded.component_count,
                completed_at = excluded.completed_at
            """,
            (
                str(source_root),
                source_manifest.catalogue_revision,
                imported,
                utc_timestamp(),
            ),
        )

    catalogue.verify_integrity()
    destination_ids = set(catalogue.list_component_ids())
    if destination_ids != seen_ids:
        missing = sorted(seen_ids - destination_ids)[:5]
        extra = sorted(destination_ids - seen_ids)[:5]
        raise RuntimeError(
            "Post-migration UUID mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if source_manifest.component_count_stored not in {0, imported}:
        print(
            "warning: source manifest component count "
            f"{source_manifest.component_count_stored} differs from "
            f"the {imported} authoritative JSON records"
        )
    catalogue.close()
    print(f"SQLite catalogue: {destination_root / 'component_catalogue.sqlite'}")
    print("verification: OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently import a legacy JSON Mandelbrot component "
            "catalogue into SQLite."
        )
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "destination_root",
        type=Path,
        nargs="?",
        help="default: convert in place beside the JSON files",
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--skip-area-scan", action="store_true")
    parser.add_argument(
        "--copy-run-state",
        action="store_true",
        help="copy runs/ checkpoints when destination differs from source",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    migrate(
        args.source_root,
        args.destination_root or args.source_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        skip_area_scan=args.skip_area_scan,
        copy_run_state=args.copy_run_state,
    )


if __name__ == "__main__":
    main()
