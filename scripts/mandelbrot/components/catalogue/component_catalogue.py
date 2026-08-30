"""Canonical arbitrary-precision Mandelbrot component catalogue.

This mirrors component_catalogue.{hpp,cpp}. Canonical numeric values are
``decimal.Decimal`` instances in memory and decimal strings in JSON. No
canonical field is converted through binary float during I/O.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

MANIFEST_SCHEMA = "mandelbrot-catalogue-v2"
COMPONENT_SCHEMA = "mandelbrot-component-v4"
PERIOD_SCHEMA = "mandelbrot-period-v3"
HIERARCHY_SCHEMA = "mandelbrot-hierarchy-v2"
NUMERIC_ENCODING = "decimal-string"

D = Decimal


def decimal(value: Any) -> Decimal:
    """Convert without routing decimal strings or integers through float."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a catalogue decimal")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid catalogue decimal: {value!r}") from exc
        if not result.is_finite():
            raise ValueError(f"Catalogue decimal must be finite: {value!r}")
        return result
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Catalogue decimal must be finite")
        # Explicitly represents the human decimal rendering, not the exact
        # binary expansion. Callers with authoritative precision should pass str.
        return Decimal(str(value))
    raise TypeError(f"Unsupported catalogue decimal type: {type(value).__name__}")


def decimal_string(value: Decimal, digits: int = 0) -> str:
    value = decimal(value)
    if value.is_zero():
        return "0"
    if digits > 0:
        with localcontext() as context:
            context.prec = digits
            value = +value
    sign, digits_tuple, exponent = value.as_tuple()
    digits = list(digits_tuple)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1

    digit_text = "".join(str(digit) for digit in digits)
    prefix = "-" if sign else ""
    scientific_exponent = len(digits) + exponent - 1

    scientific = prefix + digit_text[0]
    if len(digit_text) > 1:
        scientific += "." + digit_text[1:]
    if scientific_exponent:
        scientific += f"e{scientific_exponent}"

    point = len(digit_text) + exponent
    if point <= 0:
        plain = prefix + "0." + "0" * (-point) + digit_text
    elif point >= len(digit_text):
        plain = prefix + digit_text + "0" * (point - len(digit_text))
    else:
        plain = prefix + digit_text[:point] + "." + digit_text[point:]
    return plain if len(plain) <= len(scientific) else scientific


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _open_database(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        root / "component_catalogue.sqlite",
        timeout=60.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")
    schema_path = Path(__file__).with_name("schema.sql")
    connection.executescript(schema_path.read_text(encoding="utf-8"))
    return connection


@contextmanager
def _transaction(connection: sqlite3.Connection):
    owner = not connection.in_transaction
    if owner:
        connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if owner and connection.in_transaction:
            connection.rollback()
        raise
    else:
        if owner and connection.in_transaction:
            connection.commit()


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass
class NumericMetadata:
    encoding: str = NUMERIC_ENCODING
    working_precision_digits: int = 0
    validated_digits: int = 0


@dataclass
class ComplexValue:
    re: Decimal = D(0)
    im: Decimal = D(0)

    @classmethod
    def from_json(cls, value: Any) -> "ComplexValue":
        if isinstance(value, dict):
            return cls(decimal(value["re"]), decimal(value["im"]))
        if isinstance(value, list) and len(value) == 2:  # v1 compatibility
            return cls(decimal(value[0]), decimal(value[1]))
        raise ValueError("Complex value must be {'re','im'} or [re, im]")

    def to_json(self, digits: int = 0) -> list[str]:
        return [decimal_string(self.re, digits), decimal_string(self.im, digits)]


@dataclass
class GeometryRecord:
    coordinate_frame: str = "centered"
    polygon_rho: Decimal = D(0)
    polygon: list[ComplexValue] = field(default_factory=list)
    polygon_area: Decimal = D(0)
    area_estimate: Decimal = D(0)
    area_error: Decimal = D(0)
    area_rho: Decimal = D(0)
    characteristic_size: Decimal = D(0)
    bbox_centered: list[Decimal] = field(default_factory=lambda: [D(0)] * 4)


@dataclass
class CircleFitRecord:
    """Geometric disk fit in component-centred coordinates."""

    center_centered: Optional[ComplexValue] = None
    radius: Optional[Decimal] = None
    rms: Decimal = D(0)
    max_error: Optional[Decimal] = None


@dataclass
class CardioidFitRecord:
    """Rotated, translated, optionally slanted cardioid fit.

    Before rotation and translation the model is::

        x = size * (cos(phi) - 0.5*cos(2*phi)) * (1 - xi*sin(phi))
        y = size * (sin(phi) - 0.5*sin(2*phi)) * (1 - xi*sin(phi))

    ``xi=0`` is the ordinary symmetric cardioid.
    """

    center_centered: Optional[ComplexValue] = None
    size: Optional[Decimal] = None
    angle: Decimal = D(0)
    xi: Decimal = D(0)
    rms: Decimal = D(0)
    max_error: Optional[Decimal] = None


@dataclass
class ClassificationRecord:
    # Canonical classes are unknown, disk, cardioid.  The v3 value "circle"
    # is normalized to "disk" while loading.
    shape_class: str = "unknown"
    shape_confidence: Decimal = D(0)
    circle_fit: Optional[CircleFitRecord] = None
    cardioid_fit: Optional[CardioidFitRecord] = None


@dataclass(frozen=True)
class ClassificationUpdate:
    component_id: str
    classification: ClassificationRecord


@dataclass(frozen=True)
class ExactPeriodIndex:
    period: int
    component_index: int


@dataclass
class SymmetryRecord:
    relation: str = "has-conjugate"
    multiplicity: int = 2


@dataclass
class AttachmentRecord:
    parent_point: Optional[ComplexValue] = None
    child_point_centered: Optional[ComplexValue] = None
    gap: Optional[Decimal] = None
    gap_relative_to_child_size: Optional[Decimal] = None
    verified: bool = False


@dataclass
class HierarchyRecord:
    geometric_parent: Optional[str] = None
    renormalization_parent: Optional[str] = None
    hierarchy_root: Optional[str] = None
    generation: Optional[int] = None
    attachment: Optional[AttachmentRecord] = None


@dataclass
class ProvenanceRecord:
    method: str = ""
    run_id: str = ""
    discovered_at: str = ""
    software_revision: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class QualityRecord:
    center_validated: bool = False
    exact_period_validated: bool = False
    polygon_converged: bool = False
    area_above_cutoff: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ComponentRecord:
    id: str
    period: int
    center: ComplexValue
    numeric: NumericMetadata = field(default_factory=NumericMetadata)
    family: str = "quadratic-z2-plus-c"
    geometry: GeometryRecord = field(default_factory=GeometryRecord)
    classification: ClassificationRecord = field(default_factory=ClassificationRecord)
    symmetry: SymmetryRecord = field(default_factory=SymmetryRecord)
    hierarchy: HierarchyRecord = field(default_factory=HierarchyRecord)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    quality: QualityRecord = field(default_factory=QualityRecord)


@dataclass
class Manifest:
    catalogue_revision: int = 0
    family: str = "z^2+c"
    canonical_half_plane: str = "imaginary>=0"
    component_count_stored: int = 0
    component_count_with_symmetry: int = 0
    minimum_area: Decimal = D(0)
    exact_through_period: int = 0
    created_at: str = ""
    updated_at: str = ""
    software_revision: str = ""


@dataclass(frozen=True, slots=True)
class ComponentKey:
    period: int
    center_re: int
    center_im: int

    @classmethod
    def from_center(cls, period: int, center: "ComplexValue", bits: int = 60) -> "ComponentKey":
        if not 1 <= bits <= 60:
            raise ValueError("ComponentKey bits must lie in 1..60")
        scale = Decimal(2) ** bits
        imag = abs(center.im)
        return cls(
            period,
            int((center.re * scale + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR)),
            int((imag * scale + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR)),
        )


@dataclass(slots=True)
class ComponentQuery:
    min_period: int = 1
    max_period: int = 2**31 - 1
    min_area: Optional[Decimal] = None
    max_area: Optional[Decimal] = None
    require_polygon: bool = False
    require_center_validated: bool = False
    require_exact_period_validated: bool = False
    require_polygon_converged: bool = False
    provenance_method: Optional[str] = None
    hierarchy_root: Optional[str] = None


@dataclass(slots=True)
class CatalogueSnapshot:
    manifest: "Manifest"
    periods: list["PeriodRecord"]
    components: list["ComponentRecord"]
    by_id: dict[str, "ComponentRecord"] = field(init=False)
    by_key: dict[ComponentKey, list["ComponentRecord"]] = field(init=False)
    by_period: dict[int, list["ComponentRecord"]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {component.id: component for component in self.components}
        self.by_key = {}
        self.by_period = {}
        for component in self.components:
            self.by_key.setdefault(ComponentKey.from_center(component.period, component.center), []).append(component)
            self.by_period.setdefault(component.period, []).append(component)

    def find_near_center(self, period: int, center: "ComplexValue", tolerance: Decimal) -> Optional["ComponentRecord"]:
        tolerance = decimal(tolerance)
        candidates = self.by_key.get(ComponentKey.from_center(period, center), [])
        candidates = [*candidates, *[c for c in self.by_period.get(period, []) if c not in candidates]]
        for component in candidates:
            dx = component.center.re - center.re
            dy = component.center.im - center.im
            with localcontext() as context:
                context.prec = max(80, component.numeric.working_precision_digits or 80)
                if (dx * dx + dy * dy).sqrt() <= tolerance:
                    return component
        return None


@dataclass(slots=True)
class UpsertOptions:
    center_tolerance: Decimal = D("1e-15")
    merge_existing: bool = True
    bump_revision: bool = True


@dataclass(slots=True)
class UpsertResult:
    component: "ComponentRecord"
    inserted: bool = False
    updated: bool = False


@dataclass(slots=True)
class AreaScanCenterRecord:
    period: int
    component_index: int
    expected_period_count: int
    center: ComplexValue
    center_residual: Decimal = D(0)
    detected_exact_period: int = 0
    conjugate_index: int = 0
    center_newton_iterations: int = 0
    center_refinement_method: str = "cpp-long-double"
    center_refinement_dps: int = 0


@dataclass(slots=True)
class AreaMeasurementRecord:
    period: int
    component_index: int
    conjugate_index: int
    symmetry_source_component_index: int
    center: ComplexValue
    rho: Decimal
    theta_points: int = 0
    area_polygon: Decimal = D(0)
    area_derivative: Decimal = D(0)
    area_fourier: Optional[Decimal] = None
    area_estimate: Decimal = D(0)
    method_spread: Decimal = D(0)
    spectral_spread: Optional[Decimal] = None
    resolution_delta: Decimal = D(0)
    error_estimate: Decimal = D(0)
    fourier_tail_ratio: Optional[Decimal] = None
    negative_mode_ratio: Optional[Decimal] = None
    closure_error: Decimal = D(0)
    marked_z_closure_error: Decimal = D(0)
    max_residual: Decimal = D(0)
    solve_calls: int = 0
    failed_attempts: int = 0
    newton_iterations: int = 0
    max_subdivision_depth: int = 0
    rejected_branch_candidates: int = 0
    cyclic_seed_attempts: int = 0
    cyclic_recoveries: int = 0
    mp_solve_calls: int = 0
    mp_recoveries: int = 0
    max_mp_dps: int = 0
    seed_rho: Optional[Decimal] = None
    converged: bool = False
    exact_area_at_rho: Optional[Decimal] = None
    exact_relative_error: Optional[Decimal] = None
    failure_reason: str = ""


@dataclass(slots=True)
class AreaPeriodSummaryRecord:
    period: int
    rho: Decimal
    expected_components: int = 0
    completed_components: int = 0
    converged_components: int = 0
    missing_or_unconverged_components: int = 0
    period_complete: bool = False
    min_area: Optional[Decimal] = None
    p10_area: Optional[Decimal] = None
    median_area: Optional[Decimal] = None
    mean_area: Optional[Decimal] = None
    p90_area: Optional[Decimal] = None
    max_area: Optional[Decimal] = None
    period_area: Decimal = D(0)
    cumulative_area: Decimal = D(0)
    cumulative_complete_through_period: bool = False
    summed_error_estimate: Decimal = D(0)
    radial_increment_from_previous_rho: Optional[Decimal] = None


@dataclass
class PeriodRecord:
    period: int
    theoretical_component_count: str = ""
    known_representative_count: int = 0
    known_component_count_with_symmetry: int = 0
    catalogue_complete: bool = False
    known_area: Decimal = D(0)
    known_area_error: Decimal = D(0)
    area_cutoff: Decimal = D(0)
    exact_geometry_complete: bool = False
    polygon_rho: Decimal = D(0)
    area_rho: Decimal = D(0)
    polygon_points: int = 0
    component_ids: list[str] = field(default_factory=list)
    generated_from_catalogue_revision: int = 0


@dataclass
class HierarchyNode:
    id: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)


@dataclass
class HierarchyTree:
    root: str
    nodes: list[HierarchyNode] = field(default_factory=list)
    node_count: int = 0
    maximum_known_generation: int = 0
    known_area: Decimal = D(0)
    minimum_stored_area: Decimal = D(0)
    complete_above_cutoff: bool = False
    generated_from_catalogue_revision: int = 0


class Catalogue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._connection = _open_database(self.root)

    @property
    def database_path(self) -> Path:
        return self.root / "component_catalogue.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def component_path(self, component_id: str) -> Path:
        if len(component_id) < 2:
            raise ValueError("Component ID too short")
        return self.root / "catalogue" / "components" / component_id[:2] / f"{component_id}.json"

    def period_path(self, period: int) -> Path:
        return self.root / "catalogue" / "periods" / f"{period:06d}.json"

    def hierarchy_path(self, root_id: str) -> Path:
        return self.root / "catalogue" / "hierarchies" / f"{root_id}.json"

    @property
    def runs_path(self) -> Path:
        return self.root / "runs"

    @property
    def exports_path(self) -> Path:
        return self.root / "exports"

    @property
    def indexes_path(self) -> Path:
        return self.root / "catalogue" / "indexes"

    def export_path(self, name: str) -> Path:
        if not name:
            raise ValueError("Export name must not be empty")
        return self.exports_path / name

    def run_path(self, algorithm: str, run_name: str, name: str = "") -> Path:
        if not algorithm or not run_name:
            raise ValueError("Algorithm and run name must not be empty")
        path = self.runs_path / algorithm / run_name
        return path / name if name else path

    def ensure_layout(self) -> None:
        for path in (
            self.runs_path,
            self.exports_path,
            self.root / "legacy",
        ):
            path.mkdir(parents=True, exist_ok=True)
        with _transaction(self._connection):
            exists = self._connection.execute(
                "SELECT 1 FROM catalogue_manifest WHERE singleton = 1"
            ).fetchone()
            if exists is None:
                now = utc_timestamp()
                self.save_manifest(Manifest(created_at=now, updated_at=now))

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Catalogue":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    @staticmethod
    def generate_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def stable_id(identity: str) -> str:
        def fnv(text: str, seed: int) -> int:
            value = seed
            for byte in text.encode("utf-8"):
                value ^= byte
                value = (value * 1099511628211) & ((1 << 64) - 1)
            return value
        high = fnv(identity, 14695981039346656037)
        low = fnv("mandelbrot-component:" + identity, 1099511628211)
        raw = bytearray(high.to_bytes(8, "big") + low.to_bytes(8, "big"))
        raw[6] = (raw[6] & 0x0F) | 0x50
        raw[8] = (raw[8] & 0x3F) | 0x80
        return str(uuid.UUID(bytes=bytes(raw)))

    @staticmethod
    def exact_period_index(component: ComponentRecord) -> Optional[ExactPeriodIndex]:
        """Return the typed exact-scanner identity encoded by provenance."""
        prefix = "period-index:"
        for alias in component.provenance.aliases:
            if not alias.startswith(prefix):
                continue
            fields = alias[len(prefix):].split(":", 1)
            if len(fields) != 2:
                continue
            try:
                period = int(fields[0])
                component_index = int(fields[1])
            except ValueError:
                continue
            if period == component.period and component_index >= 0:
                return ExactPeriodIndex(period, component_index)
        return None

    @staticmethod
    def set_exact_period_index(
        component: ComponentRecord,
        period: int,
        component_index: int,
    ) -> None:
        """Replace the exact-scanner identity without exposing alias syntax."""
        if period <= 0 or component_index < 0:
            raise ValueError(
                "Exact-period identity requires a positive period and non-negative index"
            )
        if component.period not in {0, period}:
            raise ValueError(
                "Exact-period identity period does not match ComponentRecord.period"
            )
        component.period = period
        component.provenance.aliases = [
            alias
            for alias in component.provenance.aliases
            if not alias.startswith("period-index:")
        ]
        component.provenance.aliases.insert(
            0, f"period-index:{period}:{component_index}"
        )

    @staticmethod
    def canonicalize_symmetry(component: ComponentRecord,
                              real_axis_tolerance: Decimal = D("1e-50")) -> ComponentRecord:
        tolerance = decimal(real_axis_tolerance)
        if component.classification.shape_class == "circle":
            fit = component.classification.circle_fit
            component.classification.shape_class = (
                "disk"
                if fit is not None and fit.center_centered is not None and fit.radius is not None
                else "unknown"
            )
        if component.center.im < -tolerance:
            component.center.im = -component.center.im
            for point in component.geometry.polygon:
                point.im = -point.im
            component.geometry.polygon.reverse()
            if (component.classification.circle_fit is not None and
                    component.classification.circle_fit.center_centered is not None):
                component.classification.circle_fit.center_centered.im = (
                    -component.classification.circle_fit.center_centered.im
                )
            if component.classification.cardioid_fit is not None:
                fit = component.classification.cardioid_fit
                if fit.center_centered is not None:
                    fit.center_centered.im = -fit.center_centered.im
                fit.angle = -fit.angle
                fit.xi = -fit.xi
            if component.hierarchy.attachment:
                attachment = component.hierarchy.attachment
                if attachment.parent_point:
                    attachment.parent_point.im = -attachment.parent_point.im
                if attachment.child_point_centered:
                    attachment.child_point_centered.im = -attachment.child_point_centered.im
        if abs(component.center.im) <= tolerance:
            component.center.im = D(0)
            component.symmetry = SymmetryRecord("real-axis", 1)
        else:
            component.symmetry = SymmetryRecord("has-conjugate", 2)
        return component

    @staticmethod
    def validate_component(component: ComponentRecord) -> None:
        if len(component.id) < 2:
            raise ValueError("Component ID must contain at least two characters")
        if component.period <= 0:
            raise ValueError("Component period must be positive")
        if component.numeric.encoding != NUMERIC_ENCODING:
            raise ValueError("Canonical numeric encoding must be decimal-string")
        if component.numeric.working_precision_digits < 0 or component.numeric.validated_digits < 0:
            raise ValueError("Precision digits must be non-negative")
        if (component.numeric.working_precision_digits and
            component.numeric.validated_digits > component.numeric.working_precision_digits):
            raise ValueError("validated_digits cannot exceed working_precision_digits")
        if component.geometry.coordinate_frame != "centered":
            raise ValueError("Only centered polygons are canonical")
        if len(component.geometry.polygon) < 3:
            raise ValueError("Component polygon requires at least three points")
        if component.center.im < 0:
            raise ValueError("Canonical center must lie in the upper half-plane")
        if component.symmetry.relation == "real-axis" and component.symmetry.multiplicity != 1:
            raise ValueError("real-axis multiplicity must be 1")
        if component.symmetry.relation == "has-conjugate" and component.symmetry.multiplicity != 2:
            raise ValueError("has-conjugate multiplicity must be 2")
        classification = component.classification
        if classification.shape_class not in {"unknown", "disk", "cardioid"}:
            raise ValueError("shape_class must be unknown, disk, or cardioid")
        if not D(0) <= classification.shape_confidence <= D(1):
            raise ValueError("shape_confidence must lie in [0,1]")
        if classification.circle_fit is not None:
            fit = classification.circle_fit
            if fit.radius is not None and fit.radius <= 0:
                raise ValueError("Circle fit radius must be positive")
            if fit.rms < 0 or (fit.max_error is not None and fit.max_error < 0):
                raise ValueError("Circle fit errors must be non-negative")
        if classification.cardioid_fit is not None:
            fit = classification.cardioid_fit
            if fit.size is not None and fit.size <= 0:
                raise ValueError("Cardioid fit size must be positive")
            if abs(fit.xi) > D("0.5"):
                raise ValueError("Cardioid slant xi must lie in [-0.5,0.5]")
            if fit.rms < 0 or (fit.max_error is not None and fit.max_error < 0):
                raise ValueError("Cardioid fit errors must be non-negative")
        if classification.shape_class == "disk":
            fit = classification.circle_fit
            if fit is None or fit.center_centered is None or fit.radius is None:
                raise ValueError(
                    "Disk classification requires a fitted geometric centre and radius"
                )
        if classification.shape_class == "cardioid":
            fit = classification.cardioid_fit
            if fit is None or fit.center_centered is None or fit.size is None:
                raise ValueError(
                    "Cardioid classification requires a fitted centre and size"
                )

    def _save_component_row(self, component: ComponentRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO components(
                uuid, period, center_re_text, center_im_text,
                center_re_real, center_im_real,
                area_estimate_text, area_estimate_real, area_error_text,
                characteristic_size_real, polygon_points, multiplicity,
                shape_class, provenance_method, hierarchy_root_uuid,
                geometric_parent_uuid, center_validated,
                exact_period_validated, polygon_converged, area_above_cutoff
            ) VALUES(
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(uuid) DO UPDATE SET
                period = excluded.period,
                center_re_text = excluded.center_re_text,
                center_im_text = excluded.center_im_text,
                center_re_real = excluded.center_re_real,
                center_im_real = excluded.center_im_real,
                area_estimate_text = excluded.area_estimate_text,
                area_estimate_real = excluded.area_estimate_real,
                area_error_text = excluded.area_error_text,
                characteristic_size_real = excluded.characteristic_size_real,
                polygon_points = excluded.polygon_points,
                multiplicity = excluded.multiplicity,
                shape_class = excluded.shape_class,
                provenance_method = excluded.provenance_method,
                hierarchy_root_uuid = excluded.hierarchy_root_uuid,
                geometric_parent_uuid = excluded.geometric_parent_uuid,
                center_validated = excluded.center_validated,
                exact_period_validated = excluded.exact_period_validated,
                polygon_converged = excluded.polygon_converged,
                area_above_cutoff = excluded.area_above_cutoff,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                component.id,
                component.period,
                decimal_string(component.center.re),
                decimal_string(component.center.im),
                float(component.center.re),
                float(component.center.im),
                decimal_string(component.geometry.area_estimate),
                float(component.geometry.area_estimate),
                decimal_string(component.geometry.area_error),
                float(component.geometry.characteristic_size),
                len(component.geometry.polygon),
                component.symmetry.multiplicity,
                component.classification.shape_class,
                component.provenance.method,
                component.hierarchy.hierarchy_root,
                component.hierarchy.geometric_parent,
                int(component.quality.center_validated),
                int(component.quality.exact_period_validated),
                int(component.quality.polygon_converged),
                int(component.quality.area_above_cutoff),
            ),
        )
        self._connection.execute(
            """
            INSERT INTO component_records(component_id, record_json)
            SELECT id, ? FROM components WHERE uuid = ?
            ON CONFLICT(component_id) DO UPDATE SET
                record_json = excluded.record_json
            """,
            (_json_text(self._component_to_json(component)), component.id),
        )

    def save_component(self, component: ComponentRecord, *, bump_revision: bool = True) -> None:
        self.ensure_layout()
        component = self.canonicalize_symmetry(component)
        self.validate_component(component)
        with _transaction(self._connection):
            self._save_component_row(component)
            if bump_revision:
                manifest = self.load_manifest()
                manifest.catalogue_revision += 1
                manifest.updated_at = utc_timestamp()
                self.save_manifest(manifest)

    def save_components(self, components: Iterable[ComponentRecord], *, bump_revision: bool = True) -> None:
        self.ensure_layout()
        records = list(components)
        with _transaction(self._connection):
            for component in records:
                component = self.canonicalize_symmetry(component)
                self.validate_component(component)
                self._save_component_row(component)
            if records and bump_revision:
                manifest = self.load_manifest()
                manifest.catalogue_revision += 1
                manifest.updated_at = utc_timestamp()
                self.save_manifest(manifest)

    def delete_component(self, component_id: str, *, bump_revision: bool = True) -> None:
        with _transaction(self._connection):
            cursor = self._connection.execute(
                "DELETE FROM components WHERE uuid = ?", (component_id,)
            )
            if cursor.rowcount and bump_revision:
                manifest = self.load_manifest()
                manifest.catalogue_revision += 1
                manifest.updated_at = utc_timestamp()
                self.save_manifest(manifest)

    def load_component(self, component_id: str) -> ComponentRecord:
        row = self._connection.execute(
            """
            SELECT r.record_json
            FROM components AS c
            JOIN component_records AS r ON r.component_id = c.id
            WHERE c.uuid = ?
            """,
            (component_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown component ID: {component_id}")
        return self._component_from_json(json.loads(row["record_json"]))

    def component_exists(self, component_id: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM components WHERE uuid = ? LIMIT 1",
            (component_id,),
        ).fetchone() is not None

    def list_component_ids(self) -> list[str]:
        return [
            row["uuid"]
            for row in self._connection.execute(
                "SELECT uuid FROM components ORDER BY uuid"
            )
        ]

    def iter_components(self) -> Iterator[ComponentRecord]:
        yield from self.query_components()

    @staticmethod
    def _matches_query(component: ComponentRecord, query: ComponentQuery) -> bool:
        if not query.min_period <= component.period <= query.max_period:
            return False
        if query.min_area is not None and component.geometry.area_estimate < decimal(query.min_area):
            return False
        if query.max_area is not None and component.geometry.area_estimate > decimal(query.max_area):
            return False
        if query.require_polygon and len(component.geometry.polygon) < 3:
            return False
        if query.require_center_validated and not component.quality.center_validated:
            return False
        if query.require_exact_period_validated and not component.quality.exact_period_validated:
            return False
        if query.require_polygon_converged and not component.quality.polygon_converged:
            return False
        if query.provenance_method is not None and component.provenance.method != query.provenance_method:
            return False
        if query.hierarchy_root is not None and (component.hierarchy.hierarchy_root or component.id) != query.hierarchy_root:
            return False
        return True

    def query_components(self, query: Optional[ComponentQuery] = None) -> list[ComponentRecord]:
        query = query or ComponentQuery()
        sql = """
            SELECT r.record_json
            FROM components AS c
            JOIN component_records AS r ON r.component_id = c.id
            WHERE c.period >= ? AND c.period <= ?
        """
        parameters: list[Any] = [query.min_period, query.max_period]
        if query.require_polygon:
            sql += " AND c.polygon_points >= 3"
        if query.require_center_validated:
            sql += " AND c.center_validated = 1"
        if query.require_exact_period_validated:
            sql += " AND c.exact_period_validated = 1"
        if query.require_polygon_converged:
            sql += " AND c.polygon_converged = 1"
        if query.provenance_method is not None:
            sql += " AND c.provenance_method = ?"
            parameters.append(query.provenance_method)
        if query.hierarchy_root is not None:
            sql += " AND COALESCE(c.hierarchy_root_uuid, c.uuid) = ?"
            parameters.append(query.hierarchy_root)
        sql += " ORDER BY c.period, c.center_re_real, c.center_im_real, c.uuid"

        result: list[ComponentRecord] = []
        for row in self._connection.execute(sql, parameters):
            component = self._component_from_json(json.loads(row["record_json"]))
            if self._matches_query(component, query):
                result.append(component)
        result.sort(key=lambda component: (
            component.period, component.center.re, component.center.im, component.id))
        return result

    def load_snapshot(self, query: Optional[ComponentQuery] = None) -> CatalogueSnapshot:
        query = query or ComponentQuery()
        periods = []
        for period in self.list_periods():
            if query.min_period <= period <= query.max_period:
                try:
                    periods.append(self.load_period(period))
                except (OSError, ValueError, KeyError):
                    pass
        return CatalogueSnapshot(self.load_manifest(), periods, self.query_components(query))

    def load_components_for_period(self, period: int) -> list[ComponentRecord]:
        return self.query_components(ComponentQuery(min_period=period, max_period=period))

    def list_periods(self) -> list[int]:
        return [
            int(row["period"])
            for row in self._connection.execute(
                "SELECT period FROM period_records ORDER BY period"
            )
        ]

    def load_period(self, period: int) -> PeriodRecord:
        row = self._connection.execute(
            "SELECT record_json FROM period_records WHERE period = ?",
            (period,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Catalogue has no period record for {period}")
        data = json.loads(row["record_json"])
        schema = data.get("schema")
        if schema not in {PERIOD_SCHEMA, "mandelbrot-period-v2", "mandelbrot-period-v1"}:
            raise ValueError(f"Unsupported period schema: {schema}")
        return PeriodRecord(
            period=int(data["period"]),
            theoretical_component_count=data.get("theoreticalComponentCount", ""),
            known_representative_count=int(data["knownRepresentativeCount"]),
            known_component_count_with_symmetry=int(data["knownComponentCountWithSymmetry"]),
            catalogue_complete=bool(data["catalogueComplete"]),
            known_area=decimal(data["knownArea"]),
            known_area_error=decimal(data["knownAreaError"]),
            area_cutoff=decimal(data["areaCutoff"]),
            exact_geometry_complete=bool(data.get("exactGeometryComplete", False)),
            polygon_rho=decimal(data.get("polygonRho", "0")),
            area_rho=decimal(data.get("areaRho", "0")),
            polygon_points=int(data.get("polygonPoints", 0)),
            component_ids=list(data.get("componentIds", [])),
            generated_from_catalogue_revision=int(data["generatedFromCatalogueRevision"]),
        )

    def save_period(self, period: PeriodRecord) -> None:
        data = {
            "schema": PERIOD_SCHEMA,
            "period": str(period.period),
            "theoreticalComponentCount": period.theoretical_component_count,
            "knownRepresentativeCount": str(period.known_representative_count),
            "knownComponentCountWithSymmetry": str(period.known_component_count_with_symmetry),
            "catalogueComplete": period.catalogue_complete,
            "knownArea": decimal_string(period.known_area),
            "knownAreaError": decimal_string(period.known_area_error),
            "areaCutoff": decimal_string(period.area_cutoff),
            "exactGeometryComplete": period.exact_geometry_complete,
            "polygonRho": decimal_string(period.polygon_rho),
            "areaRho": decimal_string(period.area_rho),
            "polygonPoints": str(period.polygon_points),
            "componentIds": sorted(period.component_ids),
            "generatedFromCatalogueRevision": str(period.generated_from_catalogue_revision),
        }
        with _transaction(self._connection):
            self._connection.execute(
                """
                INSERT INTO period_records(period, record_json) VALUES(?, ?)
                ON CONFLICT(period) DO UPDATE SET
                    record_json = excluded.record_json
                """,
                (period.period, _json_text(data)),
            )

    def period_exists(self, period: int) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM period_records WHERE period = ?",
            (period,),
        ).fetchone() is not None

    def load_hierarchy(self, root_id: str) -> HierarchyTree:
        row = self._connection.execute(
            "SELECT record_json FROM hierarchy_records WHERE root_uuid = ?",
            (root_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Catalogue has no hierarchy record for {root_id}")
        data = json.loads(row["record_json"])
        schema = data.get("schema")
        if schema not in {HIERARCHY_SCHEMA, "mandelbrot-hierarchy-v1"}:
            raise ValueError(f"Unsupported hierarchy schema: {schema}")
        statistics = data["statistics"]
        return HierarchyTree(
            root=data["root"],
            nodes=[
                HierarchyNode(
                    id=node["id"],
                    parent=node.get("parent"),
                    children=list(node.get("children", [])),
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

    def save_hierarchy(self, tree: HierarchyTree) -> None:
        data = {
            "schema": HIERARCHY_SCHEMA,
            "root": tree.root,
            "nodes": [
                {
                    "id": node.id,
                    "parent": node.parent,
                    "children": list(node.children),
                }
                for node in tree.nodes
            ],
            "statistics": {
                "nodeCount": str(tree.node_count),
                "maximumKnownGeneration": str(
                    tree.maximum_known_generation
                ),
                "knownArea": decimal_string(tree.known_area),
                "minimumStoredArea": decimal_string(
                    tree.minimum_stored_area
                ),
                "completeAboveCutoff": tree.complete_above_cutoff,
            },
            "generatedFromCatalogueRevision": str(
                tree.generated_from_catalogue_revision
            ),
        }
        with _transaction(self._connection):
            self._connection.execute(
                """
                INSERT INTO hierarchy_records(root_uuid, record_json)
                VALUES(?, ?)
                ON CONFLICT(root_uuid) DO UPDATE SET
                    record_json = excluded.record_json
                """,
                (tree.root, _json_text(data)),
            )

    def find_near_center(self, period: int, center: ComplexValue,
                         tolerance: Decimal) -> Optional[ComponentRecord]:
        snapshot = self.load_snapshot(ComponentQuery(min_period=period, max_period=period))
        return snapshot.find_near_center(period, center, decimal(tolerance))

    @staticmethod
    def _geometry_source_priority(method: str) -> int:
        return {
            "exact-period-area-scan": 100,
            "boundary-hunter": 70,
            "satellite-hunter": 60,
            "quadtree-hunter": 50,
        }.get(method, 0 if not method else 10)

    @classmethod
    def _should_replace_geometry(
        cls, existing: ComponentRecord, incoming: ComponentRecord
    ) -> bool:
        existing_valid = existing.quality.polygon_converged and len(existing.geometry.polygon) >= 3
        incoming_valid = incoming.quality.polygon_converged and len(incoming.geometry.polygon) >= 3
        if not incoming_valid:
            return False
        if not existing_valid:
            return True
        existing_priority = cls._geometry_source_priority(existing.provenance.method)
        incoming_priority = cls._geometry_source_priority(incoming.provenance.method)
        if incoming_priority != existing_priority:
            return incoming_priority > existing_priority
        if incoming.provenance.method and incoming.provenance.method == existing.provenance.method:
            return incoming.numeric.validated_digits >= existing.numeric.validated_digits
        if incoming.numeric.validated_digits != existing.numeric.validated_digits:
            return incoming.numeric.validated_digits > existing.numeric.validated_digits
        if incoming.numeric.working_precision_digits != existing.numeric.working_precision_digits:
            return incoming.numeric.working_precision_digits > existing.numeric.working_precision_digits
        return len(incoming.geometry.polygon) > len(existing.geometry.polygon)

    @classmethod
    def merge_component_records(cls, existing: ComponentRecord, incoming: ComponentRecord) -> ComponentRecord:
        if existing.period != incoming.period:
            raise ValueError("Cannot merge components with different periods")
        result = copy.deepcopy(existing)
        geometry_better = cls._should_replace_geometry(existing, incoming)
        if geometry_better:
            result.center = copy.deepcopy(incoming.center)
            result.numeric = copy.deepcopy(incoming.numeric)
            result.geometry = copy.deepcopy(incoming.geometry)
            if incoming.provenance.method:
                result.provenance.method = incoming.provenance.method
                result.provenance.run_id = incoming.provenance.run_id
                result.provenance.discovered_at = incoming.provenance.discovered_at
                result.provenance.software_revision = incoming.provenance.software_revision
        if result.classification.shape_class == "unknown" and incoming.classification.shape_class != "unknown":
            result.classification = copy.deepcopy(incoming.classification)
        for name in ("geometric_parent", "renormalization_parent", "hierarchy_root", "generation", "attachment"):
            if getattr(result.hierarchy, name) is None and getattr(incoming.hierarchy, name) is not None:
                setattr(result.hierarchy, name, copy.deepcopy(getattr(incoming.hierarchy, name)))
        if not result.provenance.method:
            result.provenance.method = incoming.provenance.method
        if not result.provenance.run_id:
            result.provenance.run_id = incoming.provenance.run_id
        if not result.provenance.discovered_at:
            result.provenance.discovered_at = incoming.provenance.discovered_at
        aliases = (
            set(result.provenance.aliases)
            | set(existing.provenance.aliases)
            | set(incoming.provenance.aliases)
        )
        if existing.provenance.method and existing.provenance.method != result.provenance.method:
            aliases.add(f"discovery-method:{existing.provenance.method}")
        if incoming.provenance.method and incoming.provenance.method != result.provenance.method:
            aliases.add(f"discovery-method:{incoming.provenance.method}")
        result.provenance.aliases = sorted(aliases)
        result.quality.center_validated |= incoming.quality.center_validated
        result.quality.exact_period_validated |= incoming.quality.exact_period_validated
        result.quality.polygon_converged |= incoming.quality.polygon_converged
        result.quality.area_above_cutoff |= incoming.quality.area_above_cutoff
        result.quality.warnings = sorted(set(result.quality.warnings) | set(incoming.quality.warnings))
        return result

    @staticmethod
    def _key_bits_for_tolerance(tolerance: Decimal) -> int:
        tolerance = decimal(tolerance)
        if tolerance <= 0:
            return 60
        value = float(tolerance)
        if not math.isfinite(value) or value <= 0:
            return 60
        return max(1, min(60, math.floor(-math.log2(2.0 * value))))

    @staticmethod
    def _locate_component(
        records: list[ComponentRecord],
        index: dict[ComponentKey, list[int]],
        period: int,
        center: ComplexValue,
        tolerance: Decimal,
        bits: int,
    ) -> Optional[int]:
        base = ComponentKey.from_center(period, center, bits)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = ComponentKey(period, base.center_re + dx, base.center_im + dy)
                for position in index.get(key, []):
                    component = records[position]
                    delta_re = component.center.re - center.re
                    delta_im = component.center.im - center.im
                    with localcontext() as context:
                        context.prec = max(80, component.numeric.working_precision_digits or 80)
                        if (delta_re * delta_re + delta_im * delta_im).sqrt() <= tolerance:
                            return position
        return None

    def upsert_component(self, component: ComponentRecord,
                         options: Optional[UpsertOptions] = None) -> UpsertResult:
        return self.upsert_components([component], options)[0]

    def upsert_components(self, components: Iterable[ComponentRecord],
                          options: Optional[UpsertOptions] = None) -> list[UpsertResult]:
        options = options or UpsertOptions()
        incoming = [self.canonicalize_symmetry(component) for component in components]
        if not incoming:
            return []
        self.ensure_layout()
        with _transaction(self._connection):
            for component in incoming:
                if not component.id:
                    component.id = self.stable_id(
                        f"component:{component.period}:"
                        f"{decimal_string(component.center.re, 50)}:"
                        f"{decimal_string(component.center.im, 50)}"
                    )

            # Period records are rebuildable views and may lag after a crash.
            # Deduplication therefore always reads authoritative component rows.
            records = [
                record
                for period in sorted({component.period for component in incoming})
                for record in self.load_components_for_period(period)
            ]
            bits = self._key_bits_for_tolerance(options.center_tolerance)
            index: dict[ComponentKey, list[int]] = {}
            for position, component in enumerate(records):
                index.setdefault(
                    ComponentKey.from_center(
                        component.period, component.center, bits
                    ),
                    [],
                ).append(position)

            results: list[UpsertResult] = []
            changed = False
            for component in incoming:
                position = self._locate_component(
                    records,
                    index,
                    component.period,
                    component.center,
                    options.center_tolerance,
                    bits,
                )
                if position is None:
                    self.validate_component(component)
                    self._save_component_row(component)
                    position = len(records)
                    records.append(component)
                    index.setdefault(
                        ComponentKey.from_center(
                            component.period, component.center, bits
                        ),
                        [],
                    ).append(position)
                    results.append(UpsertResult(component, inserted=True))
                    changed = True
                    continue

                existing = records[position]
                component.id = existing.id
                if not options.merge_existing:
                    results.append(UpsertResult(existing))
                    continue
                merged = self.merge_component_records(existing, component)
                self.validate_component(merged)
                if merged != existing:
                    self._save_component_row(merged)
                    records[position] = merged
                    index.setdefault(
                        ComponentKey.from_center(
                            merged.period, merged.center, bits
                        ),
                        [],
                    ).append(position)
                    results.append(UpsertResult(merged, updated=True))
                    changed = True
                else:
                    results.append(UpsertResult(existing))

            if changed and options.bump_revision:
                manifest = self.load_manifest()
                manifest.catalogue_revision += 1
                manifest.updated_at = utc_timestamp()
                self.save_manifest(manifest)
            return results

    def update_component_classification(
        self,
        component_id: str,
        classification: ClassificationRecord,
        *,
        bump_revision: bool = True,
    ) -> bool:
        return bool(self.update_component_classifications(
            [ClassificationUpdate(component_id, classification)],
            bump_revision=bump_revision,
        ))

    def update_component_classifications(
        self,
        updates: Iterable[ClassificationUpdate],
        *,
        bump_revision: bool = True,
    ) -> int:
        """Replace typed classification records and bump the manifest once.

        Consumers never edit component persistence. This method preserves dynamics,
        geometry, hierarchy, provenance, and quality exactly as stored.
        """
        self.ensure_layout()
        unique: dict[str, ClassificationRecord] = {}
        for update in updates:
            if not update.component_id:
                raise ValueError("Classification update requires a component ID")
            unique[update.component_id] = copy.deepcopy(update.classification)

        with _transaction(self._connection):
            changed = 0
            for component_id, classification in sorted(unique.items()):
                component = self.load_component(component_id)
                updated = copy.deepcopy(component)
                updated.classification = classification
                updated = self.canonicalize_symmetry(updated)
                self.validate_component(updated)
                if updated != component:
                    self._save_component_row(updated)
                    changed += 1
            if changed and bump_revision:
                manifest = self.load_manifest()
                manifest.catalogue_revision += 1
                manifest.updated_at = utc_timestamp()
                self.save_manifest(manifest)
            return changed

    def load_manifest(self) -> Manifest:
        row = self._connection.execute(
            "SELECT record_json FROM catalogue_manifest WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("SQLite catalogue has no manifest")
        data = json.loads(row["record_json"])
        return Manifest(
            catalogue_revision=int(data["catalogueRevision"]),
            family=data["family"],
            canonical_half_plane=data["canonicalHalfPlane"],
            component_count_stored=int(data["componentCountStored"]),
            component_count_with_symmetry=int(data["componentCountWithSymmetry"]),
            minimum_area=decimal(data["minimumArea"]),
            exact_through_period=int(data["exactThroughPeriod"]),
            created_at=data["createdAt"], updated_at=data["updatedAt"],
            software_revision=data.get("softwareRevision", ""),
        )

    def save_manifest(self, manifest: Manifest) -> None:
        data = {
            "schema": MANIFEST_SCHEMA,
            "catalogueRevision": str(manifest.catalogue_revision),
            "family": manifest.family,
            "canonicalHalfPlane": manifest.canonical_half_plane,
            "componentCountStored": str(manifest.component_count_stored),
            "componentCountWithSymmetry": str(manifest.component_count_with_symmetry),
            "minimumArea": decimal_string(manifest.minimum_area),
            "exactThroughPeriod": str(manifest.exact_through_period),
            "createdAt": manifest.created_at,
            "updatedAt": manifest.updated_at,
            "softwareRevision": manifest.software_revision,
            "numericEncoding": NUMERIC_ENCODING,
            "paths": {
                "database": "component_catalogue.sqlite",
                "runs": "runs",
                "exports": "exports",
            },
        }
        with _transaction(self._connection):
            self._connection.execute(
                """
                INSERT INTO catalogue_manifest(singleton, record_json)
                VALUES(1, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    record_json = excluded.record_json
                """,
                (_json_text(data),),
            )

    def rebuild_period_indexes(
        self,
        periods_or_area_cutoff: Optional[Iterable[int] | Decimal | int | str | float] = None,
        area_cutoff: Decimal = D(0),
    ) -> None:
        """Rebuild typed period indexes.

        ``rebuild_period_indexes(cutoff)`` retains the original full-catalogue
        calling convention. ``rebuild_period_indexes([3, 7], cutoff)`` updates
        only the requested periods and preserves their scientific completion
        metadata. Consumers never need to inspect period JSON directly.
        """
        self.ensure_layout()

        full_rebuild = periods_or_area_cutoff is None or isinstance(
            periods_or_area_cutoff, (Decimal, int, str, float)
        )
        if full_rebuild:
            cutoff = (
                decimal(periods_or_area_cutoff)
                if periods_or_area_cutoff is not None
                else decimal(area_cutoff)
            )
            manifest = self.load_manifest()
            by_period: dict[int, PeriodRecord] = {}
            for component in self.iter_components():
                period = by_period.setdefault(
                    component.period,
                    PeriodRecord(period=component.period, area_cutoff=cutoff),
                )
                period.component_ids.append(component.id)
                period.known_representative_count += 1
                period.known_component_count_with_symmetry += component.symmetry.multiplicity
                period.known_area += (
                    component.geometry.area_estimate * component.symmetry.multiplicity
                )
                period.known_area_error += (
                    abs(component.geometry.area_error) * component.symmetry.multiplicity
                )
                period.generated_from_catalogue_revision = manifest.catalogue_revision
            with _transaction(self._connection):
                self._connection.execute("DELETE FROM period_records")
                for period in by_period.values():
                    period.component_ids.sort()
                    self.save_period(period)
            return

        cutoff = decimal(area_cutoff)
        requested_periods = sorted({int(period) for period in periods_or_area_cutoff})
        manifest = self.load_manifest()
        with _transaction(self._connection):
            for period_number in requested_periods:
                try:
                    period = self.load_period(period_number)
                except (OSError, ValueError, KeyError):
                    period = PeriodRecord(period=period_number)

                components = self.load_components_for_period(period_number)
                period.period = period_number
                period.component_ids = []
                period.known_representative_count = 0
                period.known_component_count_with_symmetry = 0
                period.known_area = D(0)
                period.known_area_error = D(0)
                period.area_cutoff = cutoff
                for component in components:
                    period.component_ids.append(component.id)
                    period.known_representative_count += 1
                    period.known_component_count_with_symmetry += component.symmetry.multiplicity
                    period.known_area += (
                        component.geometry.area_estimate
                        * component.symmetry.multiplicity
                    )
                    period.known_area_error += (
                        abs(component.geometry.area_error)
                        * component.symmetry.multiplicity
                    )
                period.component_ids.sort()
                period.generated_from_catalogue_revision = (
                    manifest.catalogue_revision
                )
                self.save_period(period)

    def rebuild_hierarchy_indexes(self, minimum_stored_area: Decimal = D(0)) -> None:
        self.ensure_layout()
        manifest = self.load_manifest()
        records = {component.id: component for component in self.iter_components()}
        children: dict[str, list[str]] = {}
        roots: set[str] = set()
        for component in records.values():
            if component.hierarchy.geometric_parent:
                children.setdefault(component.hierarchy.geometric_parent, []).append(component.id)
            else:
                roots.add(component.hierarchy.hierarchy_root or component.id)
        trees: list[HierarchyTree] = []
        for root in sorted(roots):
            if root not in records:
                continue
            stack = [root]
            seen: set[str] = set()
            nodes: list[HierarchyNode] = []
            known_area = D(0)
            maximum_generation = 0
            while stack:
                component_id = stack.pop()
                if component_id in seen or component_id not in records:
                    continue
                seen.add(component_id)
                component = records[component_id]
                child_ids = sorted(children.get(component_id, []))
                nodes.append(HierarchyNode(
                    id=component_id,
                    parent=component.hierarchy.geometric_parent,
                    children=child_ids,
                ))
                stack.extend(reversed(child_ids))
                known_area += component.geometry.area_estimate * component.symmetry.multiplicity
                maximum_generation = max(maximum_generation, component.hierarchy.generation or 0)
            trees.append(HierarchyTree(
                root=root,
                nodes=nodes,
                node_count=len(nodes),
                maximum_known_generation=maximum_generation,
                known_area=known_area,
                minimum_stored_area=decimal(minimum_stored_area),
                complete_above_cutoff=False,
                generated_from_catalogue_revision=manifest.catalogue_revision,
            ))
        with _transaction(self._connection):
            self._connection.execute("DELETE FROM hierarchy_records")
            for tree in trees:
                self.save_hierarchy(tree)

    def rebuild_manifest(self, *, exact_through_period: int = 0,
                         minimum_area: Decimal = D(0),
                         software_revision: str = "") -> None:
        self.ensure_layout()
        manifest = self.load_manifest()
        components = list(self.iter_components())
        manifest.component_count_stored = len(components)
        manifest.component_count_with_symmetry = sum(c.symmetry.multiplicity for c in components)
        manifest.minimum_area = decimal(minimum_area)
        manifest.exact_through_period = exact_through_period
        if software_revision:
            manifest.software_revision = software_revision
        manifest.updated_at = utc_timestamp()
        self.save_manifest(manifest)

    def rebuild_manifest_from_period_indexes(self, *, exact_through_period: int = 0,
                                             minimum_area: Decimal = D(0),
                                             software_revision: str = "") -> None:
        manifest = self.load_manifest()
        manifest.component_count_stored = 0
        manifest.component_count_with_symmetry = 0
        for period_number in self.list_periods():
            try:
                period = self.load_period(period_number)
            except (OSError, ValueError, KeyError):
                continue
            manifest.component_count_stored += period.known_representative_count
            manifest.component_count_with_symmetry += period.known_component_count_with_symmetry
        manifest.minimum_area = decimal(minimum_area)
        manifest.exact_through_period = exact_through_period
        if software_revision:
            manifest.software_revision = software_revision
        manifest.updated_at = utc_timestamp()
        self.save_manifest(manifest)

    def absolute_polygon(self, component: ComponentRecord) -> list[ComplexValue]:
        return [ComplexValue(component.center.re + point.re, component.center.im + point.im)
                for point in component.geometry.polygon]

    def write_component_export(self, path: str | Path, *,
                               query: Optional[ComponentQuery] = None,
                               format_name: str = "mandelbrot-component-export-v2",
                               complete: bool = False,
                               coordinate_digits: int = 0) -> None:
        rows: list[dict[str, Any]] = []
        for component in self.query_components(query):
            row = self._component_to_json(component)
            row["absolutePolygon"] = [
                point.to_json(coordinate_digits)
                for point in self.absolute_polygon(component)
            ]
            rows.append(row)
        self._atomic_write_json(Path(path), {
            "format": format_name,
            "complete": complete,
            "catalogueRevision": str(self.load_manifest().catalogue_revision),
            "components": rows,
        })

    def write_skeleton_export(self, path: str | Path, *,
                              query: Optional[ComponentQuery] = None) -> None:
        components = self.query_components(query)
        self._atomic_write_json(Path(path), {
            "format": "mandelbrot-component-skeleton-v1",
            "nodes": [{
                "id": component.id,
                "period": str(component.period),
                "center": component.center.to_json(component.numeric.working_precision_digits),
                "generation": str(component.hierarchy.generation)
                    if component.hierarchy.generation is not None else None,
            } for component in components],
            "edges": [[component.hierarchy.geometric_parent, component.id]
                      for component in components
                      if component.hierarchy.geometric_parent],
        })

    def verify_integrity(self) -> None:
        messages = [
            str(row[0])
            for row in self._connection.execute("PRAGMA integrity_check")
        ]
        if messages != ["ok"]:
            raise RuntimeError(
                "SQLite catalogue integrity check failed: "
                + "; ".join(messages)
            )
        violation = self._connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchone()
        if violation is not None:
            raise RuntimeError(
                "SQLite catalogue foreign-key check failed: "
                + ", ".join(str(value) for value in violation)
            )

    def area_scan_store(self, run_name: str = "default") -> "AreaScanStore":
        return AreaScanStore(self.root, run_name)

    @staticmethod
    def _component_to_json(component: ComponentRecord) -> dict[str, Any]:
        digits = component.numeric.working_precision_digits
        ds = lambda value: decimal_string(value, digits)
        attachment = None
        if component.hierarchy.attachment:
            value = component.hierarchy.attachment
            attachment = {
                "parentPoint": value.parent_point.to_json(digits) if value.parent_point else None,
                "childPointCentered": value.child_point_centered.to_json(digits) if value.child_point_centered else None,
                "gap": ds(value.gap) if value.gap is not None else None,
                "gapRelativeToChildSize": ds(value.gap_relative_to_child_size)
                    if value.gap_relative_to_child_size is not None else None,
                "verified": value.verified,
            }
        circle_fit = None
        if component.classification.circle_fit is not None:
            fit = component.classification.circle_fit
            circle_fit = {
                "centerCentered": fit.center_centered.to_json(digits)
                    if fit.center_centered is not None else None,
                "radius": ds(fit.radius) if fit.radius is not None else None,
                "rms": ds(fit.rms),
                "maxError": ds(fit.max_error) if fit.max_error is not None else None,
            }
        cardioid_fit = None
        if component.classification.cardioid_fit is not None:
            fit = component.classification.cardioid_fit
            cardioid_fit = {
                "centerCentered": fit.center_centered.to_json(digits)
                    if fit.center_centered is not None else None,
                "size": ds(fit.size) if fit.size is not None else None,
                "angle": ds(fit.angle),
                "xi": ds(fit.xi),
                "rms": ds(fit.rms),
                "maxError": ds(fit.max_error) if fit.max_error is not None else None,
            }
        return {
            "schema": COMPONENT_SCHEMA,
            "id": component.id,
            "numeric": {
                "encoding": component.numeric.encoding,
                "workingPrecisionDigits": str(component.numeric.working_precision_digits),
                "validatedDigits": str(component.numeric.validated_digits),
            },
            "dynamics": {
                "period": str(component.period),
                "center": component.center.to_json(digits),
                "multiplierFamily": component.family,
            },
            "geometry": {
                "coordinateFrame": component.geometry.coordinate_frame,
                "polygonRho": ds(component.geometry.polygon_rho),
                "polygon": [point.to_json(digits) for point in component.geometry.polygon],
                "polygonArea": ds(component.geometry.polygon_area),
                "areaEstimate": ds(component.geometry.area_estimate),
                "areaError": ds(component.geometry.area_error),
                "areaRho": ds(component.geometry.area_rho),
                "characteristicSize": ds(component.geometry.characteristic_size),
                "bboxCentered": [ds(value) for value in component.geometry.bbox_centered],
            },
            "classification": {
                "shapeClass": component.classification.shape_class,
                "shapeConfidence": ds(component.classification.shape_confidence),
                "circleFit": circle_fit,
                "cardioidFit": cardioid_fit,
            },
            "symmetry": {"relation": component.symmetry.relation,
                         "multiplicity": str(component.symmetry.multiplicity)},
            "hierarchy": {
                "geometricParent": component.hierarchy.geometric_parent,
                "renormalizationParent": component.hierarchy.renormalization_parent,
                "hierarchyRoot": component.hierarchy.hierarchy_root,
                "generation": str(component.hierarchy.generation)
                    if component.hierarchy.generation is not None else None,
                "attachment": attachment,
            },
            "provenance": {
                "method": component.provenance.method,
                "runId": component.provenance.run_id,
                "discoveredAt": component.provenance.discovered_at,
                "softwareRevision": component.provenance.software_revision,
                "aliases": component.provenance.aliases,
            },
            "quality": {
                "centerValidated": component.quality.center_validated,
                "exactPeriodValidated": component.quality.exact_period_validated,
                "polygonConverged": component.quality.polygon_converged,
                "areaAboveCutoff": component.quality.area_above_cutoff,
                "warnings": component.quality.warnings,
            },
        }

    @staticmethod
    def _component_from_json(data: dict[str, Any]) -> ComponentRecord:
        schema = data["schema"]
        legacy = schema == "mandelbrot-component-v1"
        if not legacy and schema not in {"mandelbrot-component-v2", "mandelbrot-component-v3", COMPONENT_SCHEMA}:
            raise ValueError(f"Unsupported component schema: {schema}")
        dynamics = data["dynamics"]
        numeric = data.get("numeric") or {
            "encoding": NUMERIC_ENCODING,
            "workingPrecisionDigits": dynamics.get("centerPrecisionDigits", "0"),
            "validatedDigits": dynamics.get("centerPrecisionDigits", "0"),
        }
        geometry = data["geometry"]
        classification = data["classification"]
        shape_class = classification.get("shapeClass", "unknown")
        circle_fit_data = classification.get("circleFit")
        circle_fit = None
        if circle_fit_data is not None:
            circle_fit = CircleFitRecord(
                center_centered=ComplexValue.from_json(circle_fit_data["centerCentered"])
                    if circle_fit_data.get("centerCentered") is not None else None,
                radius=decimal(circle_fit_data["radius"])
                    if circle_fit_data.get("radius") is not None else None,
                rms=decimal(circle_fit_data.get("rms", 0)),
                max_error=decimal(circle_fit_data["maxError"])
                    if circle_fit_data.get("maxError") is not None else None,
            )
        elif classification.get("circleFitRms") is not None:
            circle_fit = CircleFitRecord(rms=decimal(classification["circleFitRms"]))
        cardioid_fit_data = classification.get("cardioidFit")
        cardioid_fit = None
        if cardioid_fit_data is not None:
            cardioid_fit = CardioidFitRecord(
                center_centered=ComplexValue.from_json(cardioid_fit_data["centerCentered"])
                    if cardioid_fit_data.get("centerCentered") is not None else None,
                size=decimal(cardioid_fit_data["size"])
                    if cardioid_fit_data.get("size") is not None else None,
                angle=decimal(cardioid_fit_data.get("angle", 0)),
                xi=decimal(cardioid_fit_data.get("xi", 0)),
                rms=decimal(cardioid_fit_data.get("rms", 0)),
                max_error=decimal(cardioid_fit_data["maxError"])
                    if cardioid_fit_data.get("maxError") is not None else None,
            )
        elif classification.get("cardioidFitRms") is not None:
            cardioid_fit = CardioidFitRecord(rms=decimal(classification["cardioidFitRms"]))
        if shape_class == "circle":
            shape_class = (
                "disk"
                if circle_fit is not None
                and circle_fit.center_centered is not None
                and circle_fit.radius is not None
                else "unknown"
            )
        hierarchy = data["hierarchy"]
        attachment = hierarchy.get("attachment")
        return ComponentRecord(
            id=data["id"], period=int(dynamics["period"]),
            center=ComplexValue.from_json(dynamics["center"]),
            numeric=NumericMetadata(numeric.get("encoding", NUMERIC_ENCODING),
                                    int(numeric.get("workingPrecisionDigits", 0)),
                                    int(numeric.get("validatedDigits", 0))),
            family=dynamics["multiplierFamily"],
            geometry=GeometryRecord(
                coordinate_frame=geometry["coordinateFrame"],
                polygon_rho=decimal(geometry["polygonRho"]),
                polygon=[ComplexValue.from_json(point) for point in geometry["polygon"]],
                polygon_area=decimal(geometry["polygonArea"]),
                area_estimate=decimal(geometry["areaEstimate"]),
                area_error=decimal(geometry["areaError"]),
                area_rho=decimal(geometry["areaRho"]),
                characteristic_size=decimal(geometry["characteristicSize"]),
                bbox_centered=[decimal(value) for value in geometry["bboxCentered"]],
            ),
            classification=ClassificationRecord(
                shape_class=shape_class,
                shape_confidence=decimal(classification.get("shapeConfidence", 0)),
                circle_fit=circle_fit,
                cardioid_fit=cardioid_fit,
            ),
            symmetry=SymmetryRecord(data["symmetry"]["relation"],
                                    int(data["symmetry"]["multiplicity"])),
            hierarchy=HierarchyRecord(
                geometric_parent=hierarchy.get("geometricParent"),
                renormalization_parent=hierarchy.get("renormalizationParent"),
                hierarchy_root=hierarchy.get("hierarchyRoot"),
                generation=int(hierarchy["generation"])
                    if hierarchy.get("generation") is not None else None,
                attachment=AttachmentRecord(
                    parent_point=ComplexValue.from_json(attachment["parentPoint"])
                        if attachment and attachment.get("parentPoint") else None,
                    child_point_centered=ComplexValue.from_json(attachment["childPointCentered"])
                        if attachment and attachment.get("childPointCentered") else None,
                    gap=decimal(attachment["gap"])
                        if attachment and attachment.get("gap") is not None else None,
                    gap_relative_to_child_size=decimal(attachment["gapRelativeToChildSize"])
                        if attachment and attachment.get("gapRelativeToChildSize") is not None else None,
                    verified=bool(attachment.get("verified", False)) if attachment else False,
                ) if attachment else None,
            ),
            provenance=ProvenanceRecord(
                method=data["provenance"].get("method", ""),
                run_id=data["provenance"].get("runId", ""),
                discovered_at=data["provenance"].get("discoveredAt", ""),
                software_revision=data["provenance"].get("softwareRevision", ""),
                aliases=list(data["provenance"].get("aliases", [])),
            ),
            quality=QualityRecord(
                center_validated=bool(data["quality"].get("centerValidated", False)),
                exact_period_validated=bool(data["quality"].get("exactPeriodValidated", False)),
                polygon_converged=bool(data["quality"].get("polygonConverged", False)),
                area_above_cutoff=bool(data["quality"].get("areaAboveCutoff", False)),
                warnings=list(data["quality"].get("warnings", [])),
            ),
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_float=Decimal, parse_int=int)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        def render(value: Any, level: int = 0) -> str:
            indent = "  " * level
            child_indent = "  " * (level + 1)
            if isinstance(value, dict):
                if not value:
                    return "{}"
                parts = [
                    f"{child_indent}{json.dumps(str(key), ensure_ascii=False)}: "
                    f"{render(item, level + 1)}"
                    for key, item in value.items()
                ]
                return "{\n" + ",\n".join(parts) + f"\n{indent}}}"
            if isinstance(value, list):
                if not value:
                    return "[]"
                if len(value) <= 4 and all(
                    item is None or isinstance(item, (bool, int, float, str))
                    for item in value
                ):
                    return "[" + ", ".join(
                        json.dumps(item, ensure_ascii=False, allow_nan=False)
                        for item in value
                    ) + "]"
                parts = [f"{child_indent}{render(item, level + 1)}" for item in value]
                return "[\n" + ",\n".join(parts) + f"\n{indent}]"
            return json.dumps(value, ensure_ascii=False, allow_nan=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(render(data))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


class AreaScanStore:
    CENTER_FIELDS = [
        "period", "component_index", "expected_period_count", "center_re", "center_im",
        "center_residual", "detected_exact_period", "conjugate_index",
        "center_newton_iterations", "center_refinement_method", "center_refinement_dps",
    ]
    MEASUREMENT_FIELDS = [
        "period", "component_index", "conjugate_index", "symmetry_source_component_index",
        "center_re", "center_im", "rho", "theta_points", "area_polygon", "area_derivative",
        "area_fourier", "area_estimate", "method_spread", "spectral_spread", "resolution_delta",
        "error_estimate", "fourier_tail_ratio", "negative_mode_ratio", "closure_error",
        "marked_z_closure_error", "max_residual", "solve_calls", "failed_attempts",
        "newton_iterations", "max_subdivision_depth", "rejected_branch_candidates",
        "cyclic_seed_attempts", "cyclic_recoveries", "mp_solve_calls", "mp_recoveries",
        "max_mp_dps", "seed_rho", "converged", "exact_area_at_rho",
        "exact_relative_error", "failure_reason",
    ]
    SUMMARY_FIELDS = [
        "period", "rho", "expected_components", "completed_components",
        "converged_components", "missing_or_unconverged_components",
        "period_complete", "min_area", "p10_area", "median_area", "mean_area",
        "p90_area", "max_area", "period_area", "cumulative_area",
        "cumulative_complete_through_period", "summed_error_estimate",
        "radial_increment_from_previous_rho",
    ]


    def __init__(self, catalogue_root: str | Path, run_name: str):
        self.root = Path(catalogue_root)
        if not run_name:
            raise ValueError("Area-scan run name must not be empty")
        self.run_name = run_name
        self.exports = self.root / "exports"
        self.run_directory = self.root / "runs" / "area_scan" / run_name
        self._connection = _open_database(self.root)
        self.exports.mkdir(parents=True, exist_ok=True)
        (self.run_directory / "root_checkpoints").mkdir(parents=True, exist_ok=True)

    @property
    def centers_path(self) -> Path:
        return self.exports / "centers.csv"

    @property
    def measurements_path(self) -> Path:
        return self.exports / "components.csv"

    @property
    def summary_path(self) -> Path:
        return self.exports / "period_summary.csv"

    def root_checkpoint_path(self, period: int) -> Path:
        return self.run_directory / "root_checkpoints" / f"period_{period:02d}.chk"

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "AreaScanStore":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    @staticmethod
    def _optional_decimal(value: str | None) -> Optional[Decimal]:
        if value is None or value in {"", "nan", "NaN", "null"}:
            return None
        return decimal(value)

    @staticmethod
    def _atomic_write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _encode_row(fields: list[str], row: dict[str, Any]) -> str:
        output = io.StringIO(newline="")
        csv.DictWriter(
            output,
            fieldnames=fields,
            lineterminator="",
        ).writerow(row)
        return output.getvalue()

    @staticmethod
    def _decode_row(fields: list[str], text: str) -> dict[str, str]:
        values = next(csv.reader([text]))
        return {
            field: values[index] if index < len(values) else ""
            for index, field in enumerate(fields)
        }

    def load_centers(
        self, period: Optional[int] = None
    ) -> dict[int, list[AreaScanCenterRecord]]:
        result: dict[int, list[AreaScanCenterRecord]] = {}
        if period is None:
            rows = self._connection.execute(
                """
                SELECT row_csv FROM area_scan_centers
                WHERE run_name = ?
                ORDER BY period, component_index
                """,
                (self.run_name,),
            )
        else:
            rows = self._connection.execute(
                """
                SELECT row_csv FROM area_scan_centers
                WHERE run_name = ? AND period = ?
                ORDER BY component_index
                """,
                (self.run_name, period),
            )
        for stored in rows:
            row = self._decode_row(self.CENTER_FIELDS, stored["row_csv"])
            try:
                record = AreaScanCenterRecord(
                    period=int(row["period"]),
                    component_index=int(row["component_index"]),
                    expected_period_count=int(row["expected_period_count"]),
                    center=ComplexValue(decimal(row["center_re"]), decimal(row["center_im"])),
                    center_residual=decimal(row["center_residual"]),
                    detected_exact_period=int(row["detected_exact_period"]),
                    conjugate_index=int(row["conjugate_index"]),
                    center_newton_iterations=int(row.get("center_newton_iterations") or 0),
                    center_refinement_method=row.get("center_refinement_method") or "cpp-long-double",
                    center_refinement_dps=int(row.get("center_refinement_dps") or 0),
                )
            except (KeyError, ValueError, InvalidOperation):
                continue
            result.setdefault(record.period, []).append(record)
        for records in result.values():
            records.sort(key=lambda record: record.component_index)
        return result

    def load_center_period(self, period: int) -> list[AreaScanCenterRecord]:
        return self.load_centers(period).get(period, [])

    def save_centers(
        self,
        centers: dict[int, list[AreaScanCenterRecord]],
        period: Optional[int] = None,
    ) -> None:
        rows = []
        for row_period in sorted(centers):
            for record in centers[row_period]:
                if period is not None and record.period != period:
                    raise ValueError(
                        "Period-scoped center save received a row from another period"
                    )
                rows.append({
                    "period": record.period,
                    "component_index": record.component_index,
                    "expected_period_count": record.expected_period_count,
                    "center_re": decimal_string(record.center.re),
                    "center_im": decimal_string(record.center.im),
                    "center_residual": decimal_string(record.center_residual),
                    "detected_exact_period": record.detected_exact_period,
                    "conjugate_index": record.conjugate_index,
                    "center_newton_iterations": record.center_newton_iterations,
                    "center_refinement_method": record.center_refinement_method,
                    "center_refinement_dps": record.center_refinement_dps,
                })
        with _transaction(self._connection):
            if period is None:
                self._connection.execute(
                    "DELETE FROM area_scan_centers WHERE run_name = ?",
                    (self.run_name,),
                )
            else:
                self._connection.execute(
                    """
                    DELETE FROM area_scan_centers
                    WHERE run_name = ? AND period = ?
                    """,
                    (self.run_name, period),
                )
            self._connection.executemany(
                """
                INSERT INTO area_scan_centers(
                    run_name, period, component_index, row_csv
                ) VALUES(?, ?, ?, ?)
                ON CONFLICT(run_name, period, component_index)
                DO UPDATE SET row_csv = excluded.row_csv
                """,
                [
                    (
                        self.run_name,
                        int(row["period"]),
                        int(row["component_index"]),
                        self._encode_row(self.CENTER_FIELDS, row),
                    )
                    for row in rows
                ],
            )

    def save_center_period(
        self, period: int, centers: Iterable[AreaScanCenterRecord]
    ) -> None:
        self.save_centers({period: list(centers)}, period=period)

    def _measurement_from_row(
        self, row: dict[str, str]
    ) -> AreaMeasurementRecord:
        return AreaMeasurementRecord(
            period=int(row["period"]),
            component_index=int(row["component_index"]),
            conjugate_index=int(row["conjugate_index"]),
            symmetry_source_component_index=int(
                row["symmetry_source_component_index"]
            ),
            center=ComplexValue(
                decimal(row["center_re"]), decimal(row["center_im"])
            ),
            rho=decimal(row["rho"]),
            theta_points=int(row["theta_points"]),
            area_polygon=decimal(row["area_polygon"]),
            area_derivative=decimal(row["area_derivative"]),
            area_fourier=self._optional_decimal(row.get("area_fourier")),
            area_estimate=decimal(row["area_estimate"]),
            method_spread=decimal(row["method_spread"]),
            spectral_spread=self._optional_decimal(row.get("spectral_spread")),
            resolution_delta=decimal(row["resolution_delta"]),
            error_estimate=decimal(row["error_estimate"]),
            fourier_tail_ratio=self._optional_decimal(
                row.get("fourier_tail_ratio")
            ),
            negative_mode_ratio=self._optional_decimal(
                row.get("negative_mode_ratio")
            ),
            closure_error=decimal(row["closure_error"]),
            marked_z_closure_error=decimal(row["marked_z_closure_error"]),
            max_residual=decimal(row["max_residual"]),
            solve_calls=int(row["solve_calls"]),
            failed_attempts=int(row["failed_attempts"]),
            newton_iterations=int(row["newton_iterations"]),
            max_subdivision_depth=int(row["max_subdivision_depth"]),
            rejected_branch_candidates=int(
                row["rejected_branch_candidates"]
            ),
            cyclic_seed_attempts=int(row["cyclic_seed_attempts"]),
            cyclic_recoveries=int(row["cyclic_recoveries"]),
            mp_solve_calls=int(row["mp_solve_calls"]),
            mp_recoveries=int(row["mp_recoveries"]),
            max_mp_dps=int(row["max_mp_dps"]),
            seed_rho=self._optional_decimal(row.get("seed_rho")),
            converged=str(row.get("converged", "")).lower()
            in {"true", "1", "yes"},
            exact_area_at_rho=self._optional_decimal(
                row.get("exact_area_at_rho")
            ),
            exact_relative_error=self._optional_decimal(
                row.get("exact_relative_error")
            ),
            failure_reason=row.get("failure_reason", ""),
        )

    def load_measurements(
        self,
        period: Optional[int] = None,
        rho: Optional[Decimal] = None,
    ) -> list[AreaMeasurementRecord]:
        result: list[AreaMeasurementRecord] = []
        if rho is not None and period is None:
            raise ValueError("A rho-scoped measurement load also requires a period")
        if period is None:
            rows = self._connection.execute(
                """
                SELECT row_csv FROM area_scan_measurements
                WHERE run_name = ?
                ORDER BY period, component_index, rho_text
                """,
                (self.run_name,),
            )
        elif rho is None:
            rows = self._connection.execute(
                """
                SELECT row_csv FROM area_scan_measurements
                WHERE run_name = ? AND period = ?
                ORDER BY component_index, rho_text
                """,
                (self.run_name, period),
            )
        else:
            rows = self._connection.execute(
                """
                SELECT row_csv FROM area_scan_measurements
                WHERE run_name = ? AND period = ? AND rho_text = ?
                ORDER BY component_index
                """,
                (self.run_name, period, decimal_string(rho)),
            )
        for stored in rows:
            try:
                result.append(self._measurement_from_row(
                    self._decode_row(
                        self.MEASUREMENT_FIELDS, stored["row_csv"]
                    )
                ))
            except (KeyError, ValueError, InvalidOperation):
                continue
        return result

    def load_measurements_from(
        self, path: str | Path
    ) -> list[AreaMeasurementRecord]:
        path = Path(path)
        if path.resolve() == self.measurements_path.resolve():
            return self.load_measurements()
        if not path.is_file():
            return []
        result: list[AreaMeasurementRecord] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    result.append(self._measurement_from_row(row))
                except (KeyError, ValueError, InvalidOperation):
                    continue
        return result

    def save_measurements(
        self,
        measurements: Iterable[AreaMeasurementRecord],
        period: Optional[int] = None,
    ) -> None:
        def optional(value: Optional[Decimal]) -> str:
            return decimal_string(value) if value is not None else "nan"
        rows = []
        for record in measurements:
            if period is not None and record.period != period:
                raise ValueError(
                    "Period-scoped measurement save received a row from another period"
                )
            rows.append({
                "period": record.period,
                "component_index": record.component_index,
                "conjugate_index": record.conjugate_index,
                "symmetry_source_component_index": record.symmetry_source_component_index,
                "center_re": decimal_string(record.center.re),
                "center_im": decimal_string(record.center.im),
                "rho": decimal_string(record.rho),
                "theta_points": record.theta_points,
                "area_polygon": decimal_string(record.area_polygon),
                "area_derivative": decimal_string(record.area_derivative),
                "area_fourier": optional(record.area_fourier),
                "area_estimate": decimal_string(record.area_estimate),
                "method_spread": decimal_string(record.method_spread),
                "spectral_spread": optional(record.spectral_spread),
                "resolution_delta": decimal_string(record.resolution_delta),
                "error_estimate": decimal_string(record.error_estimate),
                "fourier_tail_ratio": optional(record.fourier_tail_ratio),
                "negative_mode_ratio": optional(record.negative_mode_ratio),
                "closure_error": decimal_string(record.closure_error),
                "marked_z_closure_error": decimal_string(record.marked_z_closure_error),
                "max_residual": decimal_string(record.max_residual),
                "solve_calls": record.solve_calls,
                "failed_attempts": record.failed_attempts,
                "newton_iterations": record.newton_iterations,
                "max_subdivision_depth": record.max_subdivision_depth,
                "rejected_branch_candidates": record.rejected_branch_candidates,
                "cyclic_seed_attempts": record.cyclic_seed_attempts,
                "cyclic_recoveries": record.cyclic_recoveries,
                "mp_solve_calls": record.mp_solve_calls,
                "mp_recoveries": record.mp_recoveries,
                "max_mp_dps": record.max_mp_dps,
                "seed_rho": optional(record.seed_rho),
                "converged": "True" if record.converged else "False",
                "exact_area_at_rho": optional(record.exact_area_at_rho),
                "exact_relative_error": optional(record.exact_relative_error),
                "failure_reason": record.failure_reason,
            })
        with _transaction(self._connection):
            if period is None:
                self._connection.execute(
                    "DELETE FROM area_scan_measurements WHERE run_name = ?",
                    (self.run_name,),
                )
            else:
                self._connection.execute(
                    """
                    DELETE FROM area_scan_measurements
                    WHERE run_name = ? AND period = ?
                    """,
                    (self.run_name, period),
                )
            self._connection.executemany(
                """
                INSERT INTO area_scan_measurements(
                    run_name, period, component_index, rho_text, row_csv
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(run_name, period, component_index, rho_text)
                DO UPDATE SET row_csv = excluded.row_csv
                """,
                [
                    (
                        self.run_name,
                        int(row["period"]),
                        int(row["component_index"]),
                        str(row["rho"]),
                        self._encode_row(self.MEASUREMENT_FIELDS, row),
                    )
                    for row in rows
                ],
            )

    def save_measurement_period(
        self, period: int, measurements: Iterable[AreaMeasurementRecord]
    ) -> None:
        self.save_measurements(measurements, period=period)

    def save_measurements_to(
        self,
        path: str | Path,
        measurements: Iterable[AreaMeasurementRecord],
    ) -> None:
        path = Path(path)
        records = list(measurements)
        if path.resolve() == self.measurements_path.resolve():
            self.save_measurements(records)
            return

        # External area-checkpoint batches intentionally remain simple atomic
        # CSV files so an older scanner can also recover them.
        def optional(value: Optional[Decimal]) -> str:
            return decimal_string(value) if value is not None else "nan"

        rows = [{
            "period": record.period,
            "component_index": record.component_index,
            "conjugate_index": record.conjugate_index,
            "symmetry_source_component_index":
                record.symmetry_source_component_index,
            "center_re": decimal_string(record.center.re),
            "center_im": decimal_string(record.center.im),
            "rho": decimal_string(record.rho),
            "theta_points": record.theta_points,
            "area_polygon": decimal_string(record.area_polygon),
            "area_derivative": decimal_string(record.area_derivative),
            "area_fourier": optional(record.area_fourier),
            "area_estimate": decimal_string(record.area_estimate),
            "method_spread": decimal_string(record.method_spread),
            "spectral_spread": optional(record.spectral_spread),
            "resolution_delta": decimal_string(record.resolution_delta),
            "error_estimate": decimal_string(record.error_estimate),
            "fourier_tail_ratio": optional(record.fourier_tail_ratio),
            "negative_mode_ratio": optional(record.negative_mode_ratio),
            "closure_error": decimal_string(record.closure_error),
            "marked_z_closure_error":
                decimal_string(record.marked_z_closure_error),
            "max_residual": decimal_string(record.max_residual),
            "solve_calls": record.solve_calls,
            "failed_attempts": record.failed_attempts,
            "newton_iterations": record.newton_iterations,
            "max_subdivision_depth": record.max_subdivision_depth,
            "rejected_branch_candidates":
                record.rejected_branch_candidates,
            "cyclic_seed_attempts": record.cyclic_seed_attempts,
            "cyclic_recoveries": record.cyclic_recoveries,
            "mp_solve_calls": record.mp_solve_calls,
            "mp_recoveries": record.mp_recoveries,
            "max_mp_dps": record.max_mp_dps,
            "seed_rho": optional(record.seed_rho),
            "converged": "True" if record.converged else "False",
            "exact_area_at_rho": optional(record.exact_area_at_rho),
            "exact_relative_error": optional(record.exact_relative_error),
            "failure_reason": record.failure_reason,
        } for record in records]
        self._atomic_write_csv(path, self.MEASUREMENT_FIELDS, rows)


    def _summary_from_row(
        self, row: dict[str, str]
    ) -> AreaPeriodSummaryRecord:
        return AreaPeriodSummaryRecord(
            period=int(row["period"]),
            rho=decimal(row["rho"]),
            expected_components=int(row["expected_components"]),
            completed_components=int(row["completed_components"]),
            converged_components=int(row["converged_components"]),
            missing_or_unconverged_components=int(
                row["missing_or_unconverged_components"]
            ),
            period_complete=str(row["period_complete"]).lower()
            in {"true", "1", "yes"},
            min_area=self._optional_decimal(row.get("min_area")),
            p10_area=self._optional_decimal(row.get("p10_area")),
            median_area=self._optional_decimal(row.get("median_area")),
            mean_area=self._optional_decimal(row.get("mean_area")),
            p90_area=self._optional_decimal(row.get("p90_area")),
            max_area=self._optional_decimal(row.get("max_area")),
            period_area=decimal(row["period_area"]),
            cumulative_area=decimal(row["cumulative_area"]),
            cumulative_complete_through_period=str(
                row["cumulative_complete_through_period"]
            ).lower()
            in {"true", "1", "yes"},
            summed_error_estimate=decimal(row["summed_error_estimate"]),
            radial_increment_from_previous_rho=self._optional_decimal(
                row.get("radial_increment_from_previous_rho")
            ),
        )

    def load_summaries(self) -> list[AreaPeriodSummaryRecord]:
        result: list[AreaPeriodSummaryRecord] = []
        rows = self._connection.execute(
            """
            SELECT row_csv FROM area_scan_summaries
            WHERE run_name = ?
            ORDER BY period, rho_text
            """,
            (self.run_name,),
        )
        for stored in rows:
            try:
                result.append(self._summary_from_row(
                    self._decode_row(self.SUMMARY_FIELDS, stored["row_csv"])
                ))
            except (KeyError, ValueError, InvalidOperation):
                continue
        return result

    def has_summaries(self, period: Optional[int] = None) -> bool:
        if period is None:
            row = self._connection.execute(
                """
                SELECT 1 FROM area_scan_summaries
                WHERE run_name = ? LIMIT 1
                """,
                (self.run_name,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT 1 FROM area_scan_summaries
                WHERE run_name = ? AND period = ? LIMIT 1
                """,
                (self.run_name, period),
            ).fetchone()
        return row is not None

    def save_summaries(
        self,
        summaries: Iterable[AreaPeriodSummaryRecord],
        period: Optional[int] = None,
    ) -> None:
        def optional(value: Optional[Decimal]) -> str:
            return decimal_string(value) if value is not None else "nan"
        records = list(summaries)
        if period is not None and any(record.period != period for record in records):
            raise ValueError(
                "Period-scoped summary save received a row from another period"
            )
        rows = [{
            "period": record.period,
            "rho": decimal_string(record.rho),
            "expected_components": record.expected_components,
            "completed_components": record.completed_components,
            "converged_components": record.converged_components,
            "missing_or_unconverged_components": record.missing_or_unconverged_components,
            "period_complete": "True" if record.period_complete else "False",
            "min_area": optional(record.min_area),
            "p10_area": optional(record.p10_area),
            "median_area": optional(record.median_area),
            "mean_area": optional(record.mean_area),
            "p90_area": optional(record.p90_area),
            "max_area": optional(record.max_area),
            "period_area": decimal_string(record.period_area),
            "cumulative_area": decimal_string(record.cumulative_area),
            "cumulative_complete_through_period": (
                "True" if record.cumulative_complete_through_period else "False"
            ),
            "summed_error_estimate": decimal_string(record.summed_error_estimate),
            "radial_increment_from_previous_rho": optional(
                record.radial_increment_from_previous_rho
            ),
        } for record in records]
        with _transaction(self._connection):
            if period is None:
                self._connection.execute(
                    "DELETE FROM area_scan_summaries WHERE run_name = ?",
                    (self.run_name,),
                )
            else:
                self._connection.execute(
                    """
                    DELETE FROM area_scan_summaries
                    WHERE run_name = ? AND period = ?
                    """,
                    (self.run_name, period),
                )
            self._connection.executemany(
                """
                INSERT INTO area_scan_summaries(
                    run_name, period, rho_text, row_csv
                ) VALUES(?, ?, ?, ?)
                ON CONFLICT(run_name, period, rho_text)
                DO UPDATE SET row_csv = excluded.row_csv
                """,
                [
                    (
                        self.run_name,
                        int(row["period"]),
                        str(row["rho"]),
                        self._encode_row(self.SUMMARY_FIELDS, row),
                    )
                    for row in rows
                ],
            )

    def save_summary_period(
        self, period: int, summaries: Iterable[AreaPeriodSummaryRecord]
    ) -> None:
        self.save_summaries(summaries, period=period)


__all__ = [
    "AreaMeasurementRecord", "AreaPeriodSummaryRecord", "AreaScanCenterRecord", "AreaScanStore",
    "AttachmentRecord", "Catalogue", "CatalogueSnapshot", "ClassificationRecord",
    "ComplexValue", "ComponentKey", "ComponentQuery", "ComponentRecord",
    "GeometryRecord", "HierarchyNode", "HierarchyRecord", "HierarchyTree",
    "Manifest", "NumericMetadata",
    "PeriodRecord", "ProvenanceRecord", "QualityRecord", "SymmetryRecord",
    "UpsertOptions", "UpsertResult", "decimal", "decimal_string", "utc_timestamp",
]
