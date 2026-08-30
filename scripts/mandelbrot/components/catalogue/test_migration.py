from __future__ import annotations

import json
import csv
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.catalogue.component_catalogue import (
    AreaScanStore,
    Catalogue,
    ComplexValue,
    ComponentRecord,
    GeometryRecord,
    Manifest,
    NumericMetadata,
    PeriodRecord,
    ProvenanceRecord,
    QualityRecord,
)
from components.catalogue.migrate_json_catalogue import migrate


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def make_component(identity: str, period: int) -> ComponentRecord:
    value = Decimal("1e-4")
    return ComponentRecord(
        id=Catalogue.stable_id(identity),
        period=period,
        center=ComplexValue(Decimal("-0.1"), Decimal("0.7")),
        numeric=NumericMetadata(
            working_precision_digits=80,
            validated_digits=60,
        ),
        geometry=GeometryRecord(
            polygon=[
                ComplexValue(Decimal(0), Decimal(0)),
                ComplexValue(value, Decimal(0)),
                ComplexValue(Decimal(0), value),
            ],
            polygon_area=Decimal("5e-9"),
            area_estimate=Decimal("5e-9"),
            area_error=Decimal("1e-20"),
            characteristic_size=value,
            bbox_centered=[Decimal(0), value, Decimal(0), value],
        ),
        provenance=ProvenanceRecord(method="exact-period-area-scan"),
        quality=QualityRecord(
            center_validated=True,
            exact_period_validated=True,
            polygon_converged=True,
        ),
    )


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    legacy = root / "legacy-json"
    destination = root / "sqlite"
    component = Catalogue.canonicalize_symmetry(
        make_component("migration-test", 7)
    )
    component_path = (
        legacy
        / "catalogue"
        / "components"
        / component.id[:2]
        / f"{component.id}.json"
    )
    write_json(component_path, Catalogue._component_to_json(component))
    write_json(
        legacy / "manifest.json",
        {
            "schema": "mandelbrot-catalogue-v2",
            "catalogueRevision": "42",
            "family": "z^2+c",
            "canonicalHalfPlane": "imaginary>=0",
            "componentCountStored": "1",
            "componentCountWithSymmetry": "2",
            "minimumArea": "0",
            "exactThroughPeriod": "7",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "softwareRevision": "test",
        },
    )
    period = PeriodRecord(
        period=7,
        known_representative_count=1,
        known_component_count_with_symmetry=2,
        component_ids=[component.id],
        generated_from_catalogue_revision=42,
    )
    write_json(
        legacy / "catalogue" / "periods" / "000007.json",
        {
            "schema": "mandelbrot-period-v3",
            "period": "7",
            "theoreticalComponentCount": "",
            "knownRepresentativeCount": "1",
            "knownComponentCountWithSymmetry": "2",
            "catalogueComplete": False,
            "knownArea": "1e-8",
            "knownAreaError": "2e-20",
            "areaCutoff": "0",
            "exactGeometryComplete": False,
            "polygonRho": "0",
            "areaRho": "0",
            "polygonPoints": "0",
            "componentIds": period.component_ids,
            "generatedFromCatalogueRevision": "42",
        },
    )
    write_csv(
        legacy / "exports" / "centers.csv",
        AreaScanStore.CENTER_FIELDS,
        {
            "period": "7",
            "component_index": "0",
            "expected_period_count": "1",
            "center_re": "-0.1",
            "center_im": "0.7",
            "center_residual": "0",
            "detected_exact_period": "7",
            "conjugate_index": "0",
            "center_newton_iterations": "3",
            "center_refinement_method": "test",
            "center_refinement_dps": "80",
        },
    )
    measurement = {field: "0" for field in AreaScanStore.MEASUREMENT_FIELDS}
    measurement.update({
        "period": "7",
        "component_index": "0",
        "conjugate_index": "0",
        "symmetry_source_component_index": "0",
        "center_re": "-0.1",
        "center_im": "0.7",
        "rho": "0.99999",
        "area_fourier": "nan",
        "spectral_spread": "nan",
        "fourier_tail_ratio": "nan",
        "negative_mode_ratio": "nan",
        "seed_rho": "nan",
        "converged": "True",
        "exact_area_at_rho": "nan",
        "exact_relative_error": "nan",
        "failure_reason": "",
    })
    write_csv(
        legacy / "exports" / "components.csv",
        AreaScanStore.MEASUREMENT_FIELDS,
        measurement,
    )
    summary = {field: "0" for field in AreaScanStore.SUMMARY_FIELDS}
    summary.update({
        "period": "7",
        "rho": "0.99999",
        "expected_components": "1",
        "completed_components": "1",
        "converged_components": "1",
        "missing_or_unconverged_components": "0",
        "period_complete": "True",
        "min_area": "nan",
        "p10_area": "nan",
        "median_area": "nan",
        "mean_area": "nan",
        "p90_area": "nan",
        "max_area": "nan",
        "cumulative_complete_through_period": "True",
        "radial_increment_from_previous_rho": "nan",
    })
    write_csv(
        legacy / "exports" / "period_summary.csv",
        AreaScanStore.SUMMARY_FIELDS,
        summary,
    )

    for _ in range(2):
        migrate(
            legacy,
            destination,
            run_name="default",
            batch_size=1,
            skip_area_scan=False,
            copy_run_state=False,
        )

    catalogue = Catalogue(destination)
    catalogue.verify_integrity()
    assert catalogue.list_component_ids() == [component.id]
    assert catalogue.load_component(component.id) == component
    assert catalogue.load_manifest().catalogue_revision == 42
    assert catalogue.load_period(7).component_ids == [component.id]
    store = catalogue.area_scan_store("default")
    assert len(store.load_centers()[7]) == 1
    assert len(store.load_measurements()) == 1
    assert len(store.load_summaries()) == 1
    store.close()
    catalogue.close()

print("JSON-to-SQLite migration and idempotent replay: OK")
