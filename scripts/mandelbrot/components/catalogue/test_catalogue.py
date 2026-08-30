from decimal import Decimal
from pathlib import Path
import json
import sys
import tempfile

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.catalogue.component_catalogue import (
    AreaMeasurementRecord, AreaPeriodSummaryRecord,
    AreaScanCenterRecord,
    Catalogue,
    CardioidFitRecord,
    CircleFitRecord,
    ClassificationRecord,
    ClassificationUpdate,
    ComplexValue,
    ComponentKey,
    ComponentQuery,
    ComponentRecord,
    GeometryRecord,
    HierarchyRecord,
    NumericMetadata,
    ProvenanceRecord,
    QualityRecord,
    SymmetryRecord,
    UpsertOptions,
)


def triangle(scale: str = "1e-4") -> list[ComplexValue]:
    value = Decimal(scale)
    return [
        ComplexValue(Decimal(0), Decimal(0)),
        ComplexValue(value, Decimal(0)),
        ComplexValue(Decimal(0), value),
    ]


def record(
    identity: str, *, period: int, re: str, im: str, area: str,
    method: str = "exact-period-area-scan", validated_digits: int = 93,
) -> ComponentRecord:
    component_id = Catalogue.stable_id(identity)
    component = ComponentRecord(
        id=component_id,
        period=period,
        center=ComplexValue(Decimal(re), Decimal(im)),
        numeric=NumericMetadata(working_precision_digits=120, validated_digits=validated_digits),
        geometry=GeometryRecord(
            polygon_rho=Decimal("0.999999999999999999999999999999999999"),
            polygon=triangle(),
            polygon_area=Decimal(area),
            area_estimate=Decimal(area),
            area_error=Decimal("1e-100"),
            area_rho=Decimal("0.999999999999999999999999999999999999"),
            characteristic_size=Decimal("1e-4"),
            bbox_centered=[Decimal(0), Decimal("1e-4"), Decimal(0), Decimal("1e-4")],
        ),
        classification=ClassificationRecord(shape_class="unknown"),
        hierarchy=HierarchyRecord(hierarchy_root=component_id, generation=0),
        symmetry=SymmetryRecord("real-axis" if Decimal(im) == 0 else "has-conjugate",
                                1 if Decimal(im) == 0 else 2),
        provenance=ProvenanceRecord(method=method, run_id="test"),
        quality=QualityRecord(
            center_validated=True,
            exact_period_validated=True,
            polygon_converged=True,
            area_above_cutoff=True,
        ),
    )
    Catalogue.set_exact_period_index(component, period, 17)
    return component


with tempfile.TemporaryDirectory() as directory:
    catalogue = Catalogue(directory)
    catalogue.ensure_layout()

    root = record("test-root", period=1, re="0", im="0", area="1.1780972450961724644234912687298135815739385247657")
    catalogue.save_component(root)
    loaded = catalogue.load_component(root.id)
    assert loaded.geometry.area_estimate == root.geometry.area_estimate
    assert loaded.geometry.area_error == Decimal("1e-100")
    assert loaded.numeric.working_precision_digits == 120

    # Hashable immutable identity and exact-distance-confirmed batch upsert.
    key = ComponentKey.from_center(root.period, root.center)
    assert hash(key) == hash(ComponentKey.from_center(root.period, root.center))
    exact_index = Catalogue.exact_period_index(root)
    assert exact_index is not None
    assert exact_index.period == 1 and exact_index.component_index == 17
    Catalogue.set_exact_period_index(root, 1, 23)
    replaced_index = Catalogue.exact_period_index(root)
    assert replaced_index is not None and replaced_index.component_index == 23
    first = record("candidate-a", period=7, re="-0.1", im="0.7", area="5e-9")
    duplicate = record("candidate-b", period=7, re="-0.1000000001", im="0.7", area="6e-9")
    results = catalogue.upsert_components(
        [first, duplicate],
        UpsertOptions(center_tolerance=Decimal("1e-9")),
    )
    assert sum(result.inserted for result in results) == 1
    assert sum(result.updated for result in results) == 1
    unchanged = catalogue.upsert_component(
        results[-1].component,
        UpsertOptions(center_tolerance=Decimal("1e-9")),
    )
    assert not unchanged.inserted and not unchanged.updated

    catalogue.rebuild_period_indexes(Decimal("1e-10"))
    catalogue.rebuild_manifest_from_period_indexes(
        exact_through_period=1,
        minimum_area=Decimal("1e-10"),
    )
    snapshot = catalogue.load_snapshot(ComponentQuery(min_period=7, max_period=7))
    assert len(snapshot.components) == 1
    assert snapshot.find_near_center(7, first.center, Decimal("1e-9")) is not None

    # Partial index rebuilds preserve scientific completion metadata and only
    # refresh the requested period's typed aggregate fields.
    period_seven = catalogue.load_period(7)
    period_seven.theoretical_component_count = "63"
    period_seven.catalogue_complete = True
    period_seven.exact_geometry_complete = True
    period_seven.polygon_rho = Decimal("0.9995")
    period_seven.area_rho = Decimal("0.99999")
    period_seven.polygon_points = 192
    catalogue.save_period(period_seven)
    catalogue.rebuild_period_indexes([7], Decimal("1e-10"))
    rebuilt_seven = catalogue.load_period(7)
    assert rebuilt_seven.theoretical_component_count == "63"
    assert rebuilt_seven.catalogue_complete
    assert rebuilt_seven.exact_geometry_complete
    assert rebuilt_seven.polygon_points == 192
    assert rebuilt_seven.known_representative_count == 1

    # Discovery passes may enrich an exact record, but they must never replace
    # its authoritative exact-scan geometry merely because they use the same
    # floating-point type.
    exact = record(
        "merge-exact", period=9, re="-0.2", im="0.6", area="2e-12",
        method="exact-period-area-scan", validated_digits=16,
    )
    boundary = record(
        "merge-boundary", period=9, re="-0.2", im="0.6", area="9e-12",
        method="boundary-hunter", validated_digits=16,
    )
    boundary.classification = ClassificationRecord(
        shape_class="disk",
        shape_confidence=Decimal("0.95"),
        circle_fit=CircleFitRecord(
            center_centered=ComplexValue(Decimal("1e-8"), Decimal("-2e-8")),
            radius=Decimal("3e-4"),
            rms=Decimal("0.001"),
            max_error=Decimal("0.003"),
        ),
    )
    boundary.hierarchy.geometric_parent = root.id
    merged = Catalogue.merge_component_records(exact, boundary)
    assert merged.geometry.area_estimate == Decimal("2e-12")
    assert merged.provenance.method == "exact-period-area-scan"
    assert merged.classification.shape_class == "disk"
    assert merged.hierarchy.geometric_parent == root.id

    changed = catalogue.update_component_classifications([
        ClassificationUpdate(
            results[-1].component.id,
            ClassificationRecord(
                shape_class="cardioid",
                shape_confidence=Decimal("0.8"),
                cardioid_fit=CardioidFitRecord(
                    center_centered=ComplexValue(Decimal(0), Decimal(0)),
                    size=Decimal("0.5"),
                    angle=Decimal(0),
                    xi=Decimal("0.125"),
                    rms=Decimal("0.02"),
                    max_error=Decimal("0.05"),
                ),
            ),
        )
    ])
    assert changed == 1
    classified = catalogue.load_component(results[-1].component.id)
    assert classified.classification.cardioid_fit.xi == Decimal("0.125")

    # A later exact scan is allowed to refresh an older discovery record.
    upgraded = Catalogue.merge_component_records(boundary, exact)
    assert upgraded.geometry.area_estimate == Decimal("2e-12")
    assert upgraded.provenance.method == "exact-period-area-scan"
    assert "discovery-method:boundary-hunter" in upgraded.provenance.aliases

    # The compatibility CSV is private to AreaScanStore; callers see records.
    store = catalogue.area_scan_store("test")
    store.save_centers({
        7: [AreaScanCenterRecord(
            period=7,
            component_index=0,
            expected_period_count=1,
            center=first.center,
            detected_exact_period=7,
        )]
    })
    store.save_measurements([AreaMeasurementRecord(
        period=7,
        component_index=0,
        conjugate_index=0,
        symmetry_source_component_index=0,
        center=first.center,
        rho=Decimal("0.99999"),
        area_estimate=Decimal("5e-9"),
        converged=True,
    )])
    assert store.load_centers()[7][0].center == first.center
    assert store.load_measurements()[0].area_estimate == Decimal("5e-9")
    store.save_summaries([AreaPeriodSummaryRecord(
        period=7, rho=Decimal("0.99999"), expected_components=1,
        completed_components=1, converged_components=1, period_complete=True,
        period_area=Decimal("5e-9"), cumulative_area=Decimal("5e-9"),
    )])
    assert store.load_summaries()[0].period_complete

    # Replaying the same logical scanner key cannot create duplicate rows.
    duplicate_measurement = AreaMeasurementRecord(
        period=7,
        component_index=0,
        conjugate_index=0,
        symmetry_source_component_index=0,
        center=first.center,
        rho=Decimal("0.99999"),
        area_estimate=Decimal("6e-9"),
        converged=True,
    )
    store.save_measurements([
        AreaMeasurementRecord(
            period=7,
            component_index=0,
            conjugate_index=0,
            symmetry_source_component_index=0,
            center=first.center,
            rho=Decimal("0.99999"),
            area_estimate=Decimal("5e-9"),
            converged=True,
        ),
        duplicate_measurement,
    ])
    replayed = store.load_measurements()
    assert len(replayed) == 1
    assert replayed[0].area_estimate == Decimal("6e-9")

    # Period-scoped replacement leaves every other period untouched.
    period_eight_center = AreaScanCenterRecord(
        period=8,
        component_index=0,
        expected_period_count=1,
        center=first.center,
        detected_exact_period=8,
    )
    store.save_center_period(8, [period_eight_center])
    assert len(store.load_center_period(8)) == 1
    assert len(store.load_center_period(7)) == 1

    period_eight_measurement = AreaMeasurementRecord(
        period=8,
        component_index=0,
        conjugate_index=0,
        symmetry_source_component_index=0,
        center=first.center,
        rho=Decimal("0.99999"),
        area_estimate=Decimal("8e-9"),
        converged=True,
    )
    store.save_measurement_period(8, [period_eight_measurement])
    assert len(store.load_measurements(period=7, rho=Decimal("0.99999"))) == 1
    assert len(store.load_measurements(period=8)) == 1

    period_eight_summary = AreaPeriodSummaryRecord(
        period=8,
        rho=Decimal("0.99999"),
        expected_components=1,
        completed_components=1,
        converged_components=1,
        period_complete=True,
        period_area=Decimal("8e-9"),
        cumulative_area=Decimal("13e-9"),
    )
    store.save_summary_period(8, [period_eight_summary])
    assert store.has_summaries(7)
    assert store.has_summaries(8)
    assert len(store.load_summaries()) == 2

    # An interrupted transaction publishes neither its component nor revision.
    rolled_back = record(
        "rolled-back", period=11, re="-0.3", im="0.5", area="1e-12"
    )
    revision_before = catalogue.load_manifest().catalogue_revision
    catalogue._connection.execute("BEGIN IMMEDIATE")
    catalogue.save_component(rolled_back)
    catalogue._connection.rollback()
    assert not catalogue.component_exists(rolled_back.id)
    assert catalogue.load_manifest().catalogue_revision == revision_before

    export = Path(directory) / "exports" / "test.json"
    catalogue.write_component_export(
        export,
        query=ComponentQuery(min_area=Decimal("1e-10"), require_polygon=True),
    )
    payload = json.loads(export.read_text())
    assert payload["components"]
    assert isinstance(payload["components"][0]["dynamics"]["center"], list)
    assert catalogue.database_path.is_file()
    assert not catalogue.component_path(root.id).exists()
    assert catalogue._connection.execute(
        "PRAGMA journal_mode"
    ).fetchone()[0].lower() == "wal"
    catalogue.verify_integrity()
    store.close()
    catalogue.close()

print("catalogue typed API round-trip: OK")
