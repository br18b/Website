from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.catalogue.component_catalogue import (
    Catalogue,
    ComplexValue,
    ComponentRecord,
    GeometryRecord,
    NumericMetadata,
    ProvenanceRecord,
    QualityRecord,
    UpsertOptions,
)


def component() -> ComponentRecord:
    scale = Decimal("1e-4")
    return ComponentRecord(
        id=Catalogue.stable_id("crash-recovery-component"),
        period=13,
        center=ComplexValue(Decimal("-0.4"), Decimal("0.5")),
        numeric=NumericMetadata(
            working_precision_digits=80,
            validated_digits=60,
        ),
        geometry=GeometryRecord(
            polygon=[
                ComplexValue(Decimal(0), Decimal(0)),
                ComplexValue(scale, Decimal(0)),
                ComplexValue(Decimal(0), scale),
            ],
            polygon_area=Decimal("1e-10"),
            area_estimate=Decimal("1e-10"),
            area_error=Decimal("1e-20"),
            characteristic_size=scale,
            bbox_centered=[Decimal(0), scale, Decimal(0), scale],
        ),
        provenance=ProvenanceRecord(method="restart-test"),
        quality=QualityRecord(
            center_validated=True,
            exact_period_validated=True,
            polygon_converged=True,
        ),
    )


def crash_writer(root: Path) -> None:
    catalogue = Catalogue(root)
    record = Catalogue.canonicalize_symmetry(component())
    Catalogue.validate_component(record)
    catalogue._connection.execute("BEGIN IMMEDIATE")
    catalogue._save_component_row(record)
    manifest = catalogue.load_manifest()
    manifest.catalogue_revision += 1
    catalogue.save_manifest(manifest)
    # Simulate power/process loss: no Python cleanup and no COMMIT.
    os._exit(23)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--crash-writer":
        crash_writer(Path(sys.argv[2]))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        catalogue = Catalogue(root)
        catalogue.ensure_layout()
        revision = catalogue.load_manifest().catalogue_revision
        catalogue.close()

        result = subprocess.run(
            [sys.executable, __file__, "--crash-writer", str(root)],
            check=False,
        )
        assert result.returncode == 23

        catalogue = Catalogue(root)
        catalogue.verify_integrity()
        record = component()
        assert not catalogue.component_exists(record.id)
        assert catalogue.load_manifest().catalogue_revision == revision

        options = UpsertOptions(center_tolerance=Decimal("1e-15"))
        first = catalogue.upsert_component(record, options)
        second = catalogue.upsert_component(component(), options)
        assert first.inserted
        assert not second.inserted and not second.updated
        assert catalogue.list_component_ids() == [record.id]
        catalogue.verify_integrity()
        catalogue.close()

    print("crash rollback, restart, and idempotent catch-up: OK")


if __name__ == "__main__":
    main()
