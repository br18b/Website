#!/usr/bin/env python3
"""Build the browser atlas exclusively from the typed component catalogue.

Canonical JSON, period indexes, UUID sharding, decimal parsing, symmetry, and
catalogue queries belong to :mod:`components.catalogue.component_catalogue`.
This script never opens a canonical component or period file.  It receives
``ComponentRecord`` and ``PeriodRecord`` objects, converts selected records to
browser data, fits compact display shapes, and writes demo-only assets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.project_root import expand_project_vars
from common.repo_config import RepoConfig, add_config_argument
from components.catalogue.component_catalogue import (
    Catalogue,
    ComponentKey,
    ComponentQuery,
    ComponentRecord,
    PeriodRecord,
    decimal,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "components": {
        "max_period": 50,
        "min_area": 1.0e-10,
        "coordinate_digits": 11,
        "analytic_shapes": True,
        "circle_fit_rms_tolerance": 0.055,
        "circle_fit_max_tolerance": 0.16,
        "cardioid_cusp_candidates": 9,
        "fit_warning_rms": 0.10,
        "fit_report_worst": 12,
        "cache": True,
    },
    "view": {
        "width": 1200,
        "height": 900,
        "xmin": -2.25,
        "xmax": 0.85,
        "ymin": -1.18,
        "ymax": 1.18,
        "min_span_x": 1.0e-2,
        "max_span_x": 3.40,
    },
    "render": {
        "renderer": "auto",
        "max_width": 1600,
        "max_height": 1200,
        "device_pixel_ratio_cap": 1.25,
        "max_iter": 700,
        "escape_radius": 2.0,
        "potential_stability_tolerance": 1.0e-4,
        "potential_stability_steps": 2,
        "post_escape_max_steps": 24,
        "interaction_resolution_scale": 0.5,
        "idle_resolution_scale": 1.0,
        "webgl_double_single_threshold": 1.0e-5,
        "passes": [8, 4, 2, 1],
        "interaction_passes": [8, 4],
        "settle_delay_ms": 140,
        "worker_chunk_rows": 8,
        "interior_spatial_grid": [112, 88],
    },
    "potential_color": {
        "scheme": "cividis",
        "reverse": False,
        "mapping": "boundary",
        "gmin": 1.0e-50,
        "gmax": 1.0,
        "gamma": 0.62,
        "interior_rgb": [0, 0, 0],
    },
    "interaction": {
        "wheel_zoom_speed": 0.00145,
        "drag_threshold_pixels": 5.0,
        "component_fit_fraction": 0.58,
        "transition_ms": 900,
        "click_test_iterations": 350,
        "mandelbrot_area_reference": 1.5065918849,
        "hover_fill": "rgba(44, 132, 255, 0.16)",
        "hover_stroke": "rgba(74, 161, 255, 0.95)",
        "active_fill": "rgba(69, 225, 133, 0.24)",
        "active_stroke": "rgba(89, 255, 151, 1.0)",
        "period_line_width": 1.6,
        "active_line_width": 2.8,
    },
    "preview": {"width": 1100, "height": 720, "max_iter": 350},
    "text": {
        "title": "Mandelbrot component atlas",
        "subtitle": "Explore the largest known hyperbolic components, including higher-period satellites",
    },
}


# -----------------------------------------------------------------------------
# Filesystem/config helpers
# -----------------------------------------------------------------------------

def expand_path(value: str, *, code_root: Path, project_root: Path) -> Path:
    return expand_project_vars(value, code_root=code_root, project_root=project_root)


def temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    try:
        image.save(temporary, format="PNG", optimize=True)
        with Image.open(temporary) as check:
            check.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merged(defaults: dict[str, Any], raw: Any) -> dict[str, Any]:
    result = dict(defaults)
    if raw is not None:
        if not isinstance(raw, dict):
            raise TypeError("Expected a JSON object in the configuration.")
        result.update(raw)
    return result


class ConsoleProgress:
    """Compact in-place progress for long preprocessing stages."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.terminal = sys.stdout.isatty()
        self.next_percent = 0
        self.started = time.monotonic()

    @staticmethod
    def human(value: int) -> str:
        number = float(value)
        for suffix in ("", "k", "M", "G", "T"):
            if abs(number) < 1000.0 or suffix == "T":
                if suffix == "":
                    return str(int(number))
                digits = 2 if abs(number) < 10 else 1 if abs(number) < 100 else 0
                return f"{number:.{digits}f}{suffix}"
            number /= 1000.0
        return str(value)

    def update(self, done: int, detail: str = "") -> None:
        done = max(0, min(int(done), self.total))
        percent = int(done * 100 / self.total)
        if not self.terminal and done != self.total and percent < self.next_percent:
            return
        if not self.terminal:
            self.next_percent = min(100, ((percent // 10) + 1) * 10)
        width = 28
        filled = round(width * done / self.total)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        eta = (self.total - done) / (done / elapsed) if done else None
        eta_text = "--" if eta is None else format_seconds(eta)
        line = (
            f"  {self.label} [{'#' * filled}{'-' * (width - filled)}] "
            f"{percent:3d}% {self.human(done)}/{self.human(self.total)} "
            f"ETA {eta_text}"
        )
        if detail:
            line += f"  {detail}"
        if self.terminal:
            print("\r" + line + "\x1b[K", end="", flush=True)
            if done == self.total:
                print()
        else:
            print(line)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"




# -----------------------------------------------------------------------------
# Typed catalogue -> demo conversion
# -----------------------------------------------------------------------------


def load_prefitted_components(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a demo-only fit cache, never a canonical catalogue file."""
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("components", []) if isinstance(payload, dict) else []
    return {str(row["id"]): row for row in rows if isinstance(row, dict) and "id" in row}


def _fit_signature(catalogue: Catalogue, component_cfg: dict[str, Any], binary: Path) -> str:
    manifest = catalogue.load_manifest()
    payload = {
        "catalogueRevision": manifest.catalogue_revision,
        "binaryMtimeNs": binary.stat().st_mtime_ns,
        "minArea": str(component_cfg.get("min_area", 0.0)),
        "coordinateDigits": int(component_cfg.get("coordinate_digits", 11)),
        "circleRms": float(component_cfg["circle_fit_rms_tolerance"]),
        "circleMax": float(component_cfg["circle_fit_max_tolerance"]),
        "cuspCandidates": int(component_cfg["cardioid_cusp_candidates"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_cpp_prefits(
    repo_cfg: RepoConfig,
    catalogue: Catalogue,
    component_cfg: dict[str, Any],
    output_path: Path,
    *,
    force: bool,
) -> Path | None:
    """Run the typed C++ fitter when the catalogue revision changed."""
    if not bool(component_cfg.get("analytic_shapes", True)):
        return None
    binary = CODE_ROOT / "bin" / "fit_for_demo"
    if not binary.is_file():
        print(f"C++ analytic fitter not found at {binary}; using Python fallback.")
        return None

    signature = _fit_signature(catalogue, component_cfg, binary)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if not force and output_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                print(f"reusing C++ analytic fits: {output_path}")
                return output_path
        except (OSError, ValueError, TypeError):
            pass

    print("running C++ analytic component fitter from the canonical catalogue...")
    subprocess.run(
        [
            str(binary),
            "--config", str(repo_cfg.paths.config_path),
            "--output", str(output_path),
        ],
        check=True,
    )
    atomic_write_text(
        metadata_path,
        json.dumps({"signature": signature}, separators=(",", ":")) + "\n",
    )
    return output_path


def _component_index(component: ComponentRecord) -> int:
    exact_index = Catalogue.exact_period_index(component)
    return exact_index.component_index if exact_index is not None else -1


def _absolute_polygon(component: ComponentRecord, *, conjugate: bool = False) -> list[complex]:
    center = complex(float(component.center.re), float(component.center.im))
    points = [
        center + complex(float(point.re), float(point.im))
        for point in component.geometry.polygon
    ]
    if conjugate:
        # Conjugation reverses orientation; reverse once more to retain CCW order.
        points = [value.conjugate() for value in reversed(points)]
    return points


def component_to_demo_source(
    component: ComponentRecord,
    *,
    conjugate: bool = False,
) -> dict[str, Any]:
    """Convert one typed canonical record to a browser-oriented polygon record."""
    center = complex(float(component.center.re), float(component.center.im))
    if conjugate:
        center = center.conjugate()
    points = _absolute_polygon(component, conjugate=conjugate)
    component_id = component.id + (":conjugate" if conjugate else "")
    index = _component_index(component)
    return {
        "id": component_id,
        "canonicalId": component.id,
        "conjugate": conjugate,
        "period": component.period,
        "index": index,
        "shape": "polygon",
        "center": [center.real, center.imag],
        "bbox": polygon_bbox(points),
        "points": flatten_polygon(points),
        "polygonRho": float(component.geometry.polygon_rho),
        "polygonArea": float(component.geometry.polygon_area),
        "area": float(component.geometry.area_estimate),
        "areaError": float(component.geometry.area_error),
        "areaRho": float(component.geometry.area_rho),
        "areaSource": component.provenance.method or "catalogue",
        "source": component.provenance.method or "catalogue",
        "generation": component.hierarchy.generation,
        "parentId": component.hierarchy.geometric_parent,
    }


def exact_analytic_component(component: ComponentRecord) -> dict[str, Any] | None:
    center = complex(float(component.center.re), float(component.center.im))
    base = {
        "id": component.id,
        "canonicalId": component.id,
        "conjugate": False,
        "period": component.period,
        "index": _component_index(component),
        "center": [center.real, center.imag],
        "area": float(component.geometry.area_estimate),
        "areaError": float(component.geometry.area_error),
        "areaRho": float(component.geometry.area_rho),
        "source": component.provenance.method or "catalogue",
    }
    classification = component.classification
    if classification.shape_class == "disk" and classification.circle_fit is not None:
        fit = classification.circle_fit
        if fit.center_centered is not None and fit.radius is not None:
            shape_center = center + complex(
                float(fit.center_centered.re),
                float(fit.center_centered.im),
            )
            result = {**base, "shape": "circle", "size": float(fit.radius)}
            if abs(shape_center - center) > 1.0e-16:
                result["shapeCenter"] = [shape_center.real, shape_center.imag]
            return result
    if classification.shape_class == "cardioid" and classification.cardioid_fit is not None:
        fit = classification.cardioid_fit
        if fit.center_centered is not None and fit.size is not None:
            shape_center = center + complex(
                float(fit.center_centered.re),
                float(fit.center_centered.im),
            )
            # Browser cardioids use size*(2e^it-e^2it), while the canonical
            # fit record uses size*(e^it-0.5e^2it).
            result = {
                **base,
                "shape": "cardioid",
                "size": 0.5 * float(fit.size),
                "angle": float(fit.angle),
                "xi": float(fit.xi),
            }
            if abs(shape_center - center) > 1.0e-16:
                result["shapeCenter"] = [shape_center.real, shape_center.imag]
            return result
    if component.period == 1 and abs(center) < 1.0e-12:
        return {**base, "shape": "cardioid", "size": 0.25, "angle": 0.0, "xi": 0.0}
    if component.period == 2 and abs(center + 1.0) < 1.0e-12:
        return {**base, "shape": "circle", "size": 0.25}
    return None


def enrich_compact_component(compact: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = dict(compact)
    for key in (
        "id", "canonicalId", "conjugate", "period", "index", "area",
        "areaError", "areaRho", "polygonRho", "polygonArea", "source",
        "generation", "parentId",
    ):
        if key in source and source[key] is not None:
            result[key] = source[key]
    return result


def conjugate_compact_component(compact: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(compact)
    result["id"] = source["id"]
    result["canonicalId"] = source["canonicalId"]
    result["conjugate"] = True
    if "center" in result:
        result["center"][1] = -float(result["center"][1])
    if "shapeCenter" in result:
        result["shapeCenter"][1] = -float(result["shapeCenter"][1])
    if "angle" in result:
        result["angle"] = -float(result["angle"])
    if "xi" in result:
        result["xi"] = -float(result["xi"])
    return enrich_compact_component(result, source)


def _prefer_component(lhs: ComponentRecord, rhs: ComponentRecord) -> ComponentRecord:
    """Choose the stronger record when a malformed old catalogue has duplicates."""
    lhs_score = (
        lhs.quality.polygon_converged,
        lhs.quality.exact_period_validated,
        lhs.numeric.validated_digits,
        lhs.numeric.working_precision_digits,
        len(lhs.geometry.polygon),
    )
    rhs_score = (
        rhs.quality.polygon_converged,
        rhs.quality.exact_period_validated,
        rhs.numeric.validated_digits,
        rhs.numeric.working_precision_digits,
        len(rhs.geometry.polygon),
    )
    return rhs if rhs_score > lhs_score else lhs


def unique_components(records: list[ComponentRecord]) -> list[ComponentRecord]:
    """Deduplicate through the catalogue's stable 60-bit component key."""
    by_key: dict[ComponentKey, ComponentRecord] = {}
    for component in records:
        key = ComponentKey.from_center(component.period, component.center)
        previous = by_key.get(key)
        by_key[key] = component if previous is None else _prefer_component(previous, component)
    return sorted(
        by_key.values(),
        key=lambda value: (value.period, value.center.re, value.center.im, value.id),
    )


def cache_signature(
    component_cfg: dict[str, Any],
    catalogue: Catalogue,
    prefit_path: Path | None,
) -> str:
    manifest = catalogue.load_manifest()
    payload: dict[str, Any] = {
        "componentConfig": component_cfg,
        "catalogueRevision": manifest.catalogue_revision,
        "componentCountStored": manifest.component_count_stored,
    }
    if prefit_path and prefit_path.is_file():
        stat = prefit_path.stat()
        payload["prefit"] = {"size": stat.st_size, "mtimeNs": stat.st_mtime_ns}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def signed_polygon_area(points: list[complex]) -> float:
    total = 0.0
    for index, value in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += value.real * following.imag - value.imag * following.real
    return 0.5 * total


def polygon_bbox(points: list[complex]) -> list[float]:
    xs = [value.real for value in points]
    ys = [value.imag for value in points]
    return [min(xs), max(xs), min(ys), max(ys)]


def flatten_polygon(points: list[complex]) -> list[float]:
    flat: list[float] = []
    for value in points:
        flat.extend([float(f"{value.real:.17g}"), float(f"{value.imag:.17g}")])
    return flat


def quantize_number(value: float, digits: int) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def quantize_flat_points(values: list[float], digits: int) -> list[float]:
    return [quantize_number(float(value), digits) for value in values]


@dataclass(frozen=True)
class AnalyticShapeFit:
    shape: str
    shape_center: complex
    size: float
    angle: float
    rms_relative: float
    max_relative: float
    circle_rms_relative: float
    circle_max_relative: float
    origin_shift_relative: float


def _flat_to_complex(values: list[float]) -> np.ndarray:
    if len(values) < 6 or len(values) % 2:
        raise ValueError("A component polygon needs at least three complex points.")
    array = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    points = array[:, 0] + 1j * array[:, 1]
    if not np.all(np.isfinite(array)):
        raise ValueError("Component polygon contains non-finite coordinates.")
    return points


def _signed_area_array(points: np.ndarray) -> float:
    x = points.real
    y = points.imag
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _fit_circle(points: np.ndarray, component_center: complex) -> AnalyticShapeFit:
    """Geometrically fit a circle, including its centre.

    The superattracting parameter is the *dynamical* centre of a hyperbolic
    component.  It is generally not the Euclidean centre of the nearly round
    boundary.  Holding it fixed was the source of the conspicuous shifted and
    oversized circles in the first analytic-atlas version.

    A linear algebraic fit provides a stable initial estimate, followed by a
    few Gauss--Newton iterations on the true radial residuals.
    """
    x = points.real
    y = points.imag
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    rhs = x * x + y * y
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    cx, cy, constant = (float(value) for value in solution)
    radius_squared = constant + cx * cx + cy * cy
    if not (math.isfinite(radius_squared) and radius_squared > 0.0):
        # Extremely tiny, badly conditioned shapes are better initialized from
        # their centroid than rejected outright.
        shape_center = complex(np.mean(points))
        radius = float(np.mean(np.abs(points - shape_center)))
        cx, cy = shape_center.real, shape_center.imag
    else:
        radius = math.sqrt(radius_squared)

    for _ in range(12):
        dx = x - cx
        dy = y - cy
        distances = np.hypot(dx, dy)
        safe = np.maximum(distances, np.finfo(float).tiny)
        residual_signed = distances - radius
        jacobian = np.column_stack((-dx / safe, -dy / safe, -np.ones(len(points))))
        delta, *_ = np.linalg.lstsq(jacobian, -residual_signed, rcond=None)
        if not np.all(np.isfinite(delta)):
            break
        cx += float(delta[0])
        cy += float(delta[1])
        radius += float(delta[2])
        if radius <= 0.0:
            radius = float(np.mean(distances))
        if float(np.linalg.norm(delta)) <= 2.0e-14 * max(1.0, radius):
            break

    shape_center = complex(cx, cy)
    radii = np.abs(points - shape_center)
    radius = float(np.mean(radii))
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("Cannot fit a zero-sized circle.")
    residual = np.abs(radii - radius)
    reference = max(radius, np.finfo(float).tiny)
    rms = float(np.sqrt(np.mean(residual * residual)) / reference)
    maximum = float(np.max(residual) / reference)
    return AnalyticShapeFit(
        shape="circle",
        shape_center=shape_center,
        size=radius,
        angle=0.0,
        rms_relative=rms,
        max_relative=maximum,
        circle_rms_relative=rms,
        circle_max_relative=maximum,
        origin_shift_relative=abs(shape_center - component_center) / reference,
    )


def _fit_rotated_cardioid(
    points: np.ndarray,
    component_center: complex,
    *,
    cusp_candidates: int,
    circle_fit: AnalyticShapeFit,
) -> AnalyticShapeFit:
    """Fit beta + alpha * (2 exp(it) - exp(2it)).

    The closest polygon samples to the dynamical centre provide candidate cusp
    locations; for each candidate we test both traversal directions.  Both the
    complex translation ``beta`` and coefficient ``alpha`` are fitted.  For an
    exact cardioid beta equals the superattracting centre, while allowing a
    small translation noticeably improves real high-period outlines.
    """
    if _signed_area_array(points) < 0.0:
        points = points[::-1].copy()

    offsets_from_component_center = points - component_center
    count = len(points)
    candidate_count = max(1, min(int(cusp_candidates), count))
    cusp_indices = np.argsort(np.abs(offsets_from_component_center))[:candidate_count]
    indices = np.arange(count, dtype=np.float64)

    best: tuple[float, float, float, complex, complex] | None = None
    for cusp_index_value in cusp_indices:
        cusp_index = int(cusp_index_value)
        relative = np.mod(indices - cusp_index, count)
        for direction in (1, -1):
            phase = direction * (2.0 * math.pi / count) * relative
            unit = np.exp(1j * phase)
            basis = 2.0 * unit - unit * unit
            basis_mean = complex(np.mean(basis))
            points_mean = complex(np.mean(points))
            centered_basis = basis - basis_mean
            centered_points = points - points_mean
            denominator = float(np.vdot(centered_basis, centered_basis).real)
            if denominator <= np.finfo(float).tiny:
                continue
            alpha = complex(np.vdot(centered_basis, centered_points) / denominator)
            beta = points_mean - alpha * basis_mean
            size = abs(alpha)
            if not (math.isfinite(size) and size > 0.0):
                continue
            residual = np.abs(points - (beta + alpha * basis))
            reference = max(size, np.finfo(float).tiny)
            rms = float(np.sqrt(np.mean(residual * residual)) / reference)
            maximum = float(np.max(residual) / reference)
            score = rms + 0.05 * maximum
            if best is None or score < best[0]:
                best = (score, rms, maximum, alpha, beta)

    if best is None:
        raise ValueError("Could not fit a rotated cardioid.")
    _, rms, maximum, alpha, beta = best
    angle = math.atan2(alpha.imag, alpha.real)
    return AnalyticShapeFit(
        shape="cardioid",
        shape_center=beta,
        size=abs(alpha),
        angle=angle,
        rms_relative=rms,
        max_relative=maximum,
        circle_rms_relative=circle_fit.rms_relative,
        circle_max_relative=circle_fit.max_relative,
        origin_shift_relative=abs(beta - component_center) / max(abs(alpha), np.finfo(float).tiny),
    )


def fit_component_shape(
    flat_points: list[float],
    center: complex,
    *,
    circle_rms_tolerance: float,
    circle_max_tolerance: float,
    cardioid_cusp_candidates: int,
) -> AnalyticShapeFit:
    points = _flat_to_complex(flat_points)
    circle = _fit_circle(points, center)
    if (
        circle.rms_relative <= circle_rms_tolerance
        and circle.max_relative <= circle_max_tolerance
    ):
        return circle

    cardioid = _fit_rotated_cardioid(
        points,
        center,
        cusp_candidates=cardioid_cusp_candidates,
        circle_fit=circle,
    )
    # A badly distorted domain can occasionally fail both idealisations.  We
    # still publish an analytic shape, but use whichever of the two is the
    # better geometric approximation rather than forcing an obviously worse
    # cardioid.
    if circle.rms_relative < cardioid.rms_relative:
        return circle
    return cardioid


def analytic_outline(
    shape: str,
    shape_center: complex,
    size: float,
    angle: float = 0.0,
    xi: float = 0.0,
    *,
    samples: int = 256,
) -> np.ndarray:
    if shape == "circle":
        phase = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        return shape_center + size * np.exp(1j * phase)
    if shape == "cardioid":
        phase = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        unit = np.exp(1j * phase)
        slant = 1.0 - xi * np.sin(phase)
        return (
            shape_center
            + size * complex(math.cos(angle), math.sin(angle))
            * (2.0 * unit - unit * unit) * slant
        )
    raise ValueError(f"Unknown analytic component shape: {shape!r}")


def compact_analytic_component(
    source: dict[str, Any],
    component_cfg: dict[str, Any],
) -> tuple[dict[str, Any], AnalyticShapeFit]:
    center = complex(float(source["center"][0]), float(source["center"][1]))
    fit = fit_component_shape(
        [float(value) for value in source["points"]],
        center,
        circle_rms_tolerance=float(component_cfg["circle_fit_rms_tolerance"]),
        circle_max_tolerance=float(component_cfg["circle_fit_max_tolerance"]),
        cardioid_cusp_candidates=int(component_cfg["cardioid_cusp_candidates"]),
    )
    digits = int(component_cfg["coordinate_digits"])
    compact: dict[str, Any] = {
        "id": str(source["id"]),
        "period": int(source["period"]),
        "shape": fit.shape,
        "center": [
            quantize_number(center.real, digits),
            quantize_number(center.imag, digits),
        ],
        "size": quantize_number(fit.size, digits),
        "area": quantize_number(float(source["area"]), digits),
    }
    shape_center = fit.shape_center
    # Keep the mathematical centre for labels and identity.  Only publish the
    # separate geometric origin when the fit actually needs it; exact period-1
    # and period-2 records therefore remain as compact as before.
    if abs(shape_center - center) > 2.0 * 10.0 ** (-digits):
        compact["shapeCenter"] = [
            quantize_number(shape_center.real, digits),
            quantize_number(shape_center.imag, digits),
        ]
    source_index = int(source.get("index", -1))
    if source_index >= 0:
        compact["index"] = source_index
    if fit.shape == "cardioid":
        compact["angle"] = quantize_number(fit.angle, digits)
    return compact, fit


def report_shape_fits(
    diagnostics: list[tuple[str, int, int, AnalyticShapeFit]],
    component_cfg: dict[str, Any],
) -> None:
    if not diagnostics:
        return
    circles = sum(fit.shape == "circle" for _, _, _, fit in diagnostics)
    cardioids = len(diagnostics) - circles
    rms_values = np.asarray([fit.rms_relative for _, _, _, fit in diagnostics], dtype=float)
    warning = float(component_cfg["fit_warning_rms"])
    print(
        "analytic shape fits: "
        f"circles={circles}, cardioids={cardioids}, "
        f"median RMS={np.median(rms_values):.3%}, "
        f"p95 RMS={np.percentile(rms_values, 95):.3%}, "
        f"max RMS={np.max(rms_values):.3%}"
    )
    poor = sum(value > warning for value in rms_values)
    if poor:
        print(f"  warning: {poor} fit(s) exceed fit_warning_rms={warning:.3%}")
    report_count = max(0, int(component_cfg["fit_report_worst"]))
    if report_count:
        worst = sorted(diagnostics, key=lambda row: row[3].rms_relative, reverse=True)[:report_count]
        print("  worst analytic fits:")
        for component_id, period, index, fit in worst:
            index_text = str(index) if index >= 0 else "discovered"
            print(
                f"    {component_id}: period={period} index={index_text} "
                f"shape={fit.shape} rms={fit.rms_relative:.3%} "
                f"max={fit.max_relative:.3%} circle_rms={fit.circle_rms_relative:.3%} "
                f"origin_shift={fit.origin_shift_relative:.2%} of size"
            )


# -----------------------------------------------------------------------------
# Component assembly
# -----------------------------------------------------------------------------


def build_components(
    component_cfg: dict[str, Any],
    *,
    catalogue: Catalogue,
    prefitted_components: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_period = int(component_cfg["max_period"])
    min_area = decimal(str(component_cfg.get("min_area", 0.0)))
    coordinate_digits = int(component_cfg.get("coordinate_digits", 11))
    analytic_shapes = bool(component_cfg.get("analytic_shapes", True))
    prefitted_components = prefitted_components or {}

    if max_period < 1:
        raise ValueError("components.max_period must be positive.")
    if min_area < 0:
        raise ValueError("components.min_area must be non-negative.")
    if not 8 <= coordinate_digits <= 17:
        raise ValueError("components.coordinate_digits must lie between 8 and 17.")

    # One typed catalogue query replaces centers.csv, components.csv, and every
    # atlas_components*.json parser that used to live in this script.
    snapshot = catalogue.load_snapshot(ComponentQuery(
        min_period=1,
        max_period=max_period,
        require_polygon=True,
        require_polygon_converged=True,
    ))
    canonical = unique_components(snapshot.components)
    selected = [
        component for component in canonical
        if component.geometry.area_estimate >= min_area
    ]
    print(
        f"catalogue query: {len(canonical)} canonical polygon record(s), "
        f"{len(selected)} at area >= {min_area}"
    )

    components: list[dict[str, Any]] = []
    fit_diagnostics: list[tuple[str, int, int, AnalyticShapeFit]] = []
    progress = ConsoleProgress("building demo components", len(selected))

    for position, component in enumerate(selected, start=1):
        canonical_source = component_to_demo_source(component)
        exact = exact_analytic_component(component)
        if exact is not None:
            canonical_compact = exact
        elif analytic_shapes:
            prefit = prefitted_components.get(component.id)
            if prefit is not None:
                canonical_compact = enrich_compact_component(prefit, canonical_source)
            else:
                canonical_compact, fit = compact_analytic_component(canonical_source, component_cfg)
                canonical_compact = enrich_compact_component(canonical_compact, canonical_source)
                fit_diagnostics.append(
                    (component.id, component.period, _component_index(component), fit)
                )
        else:
            canonical_source["points"] = quantize_flat_points(
                canonical_source["points"], coordinate_digits
            )
            canonical_source["bbox"] = [
                quantize_number(float(value), coordinate_digits)
                for value in canonical_source["bbox"]
            ]
            canonical_compact = canonical_source

        components.append(canonical_compact)
        if component.symmetry.multiplicity == 2 and component.center.im != 0:
            conjugate_source = component_to_demo_source(component, conjugate=True)
            if analytic_shapes:
                components.append(
                    conjugate_compact_component(canonical_compact, conjugate_source)
                )
            else:
                conjugate_source["points"] = quantize_flat_points(
                    conjugate_source["points"], coordinate_digits
                )
                conjugate_source["bbox"] = [
                    quantize_number(float(value), coordinate_digits)
                    for value in conjugate_source["bbox"]
                ]
                components.append(conjugate_source)
        if position % 100 == 0 or position == len(selected):
            progress.update(position, f"published {len(components)} full-plane")

    if analytic_shapes:
        report_shape_fits(fit_diagnostics, component_cfg)

    period_records: dict[int, PeriodRecord] = {
        period.period: period for period in snapshot.periods
    }
    all_by_period: dict[int, list[ComponentRecord]] = {}
    selected_by_period: dict[int, list[ComponentRecord]] = {}
    for component in canonical:
        all_by_period.setdefault(component.period, []).append(component)
    for component in selected:
        selected_by_period.setdefault(component.period, []).append(component)

    summaries: list[dict[str, Any]] = []
    for period in sorted(set(all_by_period) | set(period_records)):
        all_rows = all_by_period.get(period, [])
        selected_rows = selected_by_period.get(period, [])
        index = period_records.get(period)
        known_count = (
            index.known_component_count_with_symmetry
            if index else sum(row.symmetry.multiplicity for row in all_rows)
        )
        included_count = sum(row.symmetry.multiplicity for row in selected_rows)
        known_total_area = (
            float(index.known_area)
            if index else math.fsum(
                float(row.geometry.area_estimate) * row.symmetry.multiplicity
                for row in all_rows
            )
        )
        all_areas = [float(row.geometry.area_estimate) for row in all_rows]
        selected_areas = [float(row.geometry.area_estimate) for row in selected_rows]
        summaries.append({
            "period": period,
            "count": known_count,
            "includedCount": included_count,
            "omittedCount": max(0, known_count - included_count),
            "totalArea": known_total_area,
            "knownTotalArea": known_total_area,
            "minArea": min(all_areas) if all_areas else 0.0,
            "maxArea": max(all_areas) if all_areas else 0.0,
            "minimumIncludedArea": min(selected_areas) if selected_areas else None,
            "areaCutoff": float(min_area),
            "complete": bool(index.catalogue_complete) if index else False,
        })

    components.sort(key=lambda row: (
        int(row["period"]), float(row["center"][0]), float(row["center"][1]), str(row["id"])
    ))
    return components, summaries


def make_palette(color_cfg: dict[str, Any]) -> list[list[int]]:
    scheme = str(color_cfg.get("scheme", "turbo"))
    reverse = bool(color_cfg.get("reverse", False))
    values = np.linspace(0.0, 1.0, 256)
    if reverse:
        values = 1.0 - values
    try:
        cmap = matplotlib.colormaps[scheme]
    except KeyError as exc:
        raise ValueError(f"Unknown Matplotlib color scheme: {scheme!r}") from exc
    rgb = np.asarray(cmap(values))[:, :3]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8).tolist()


def render_preview(
    out_dir: Path,
    data: dict[str, Any],
    preview_cfg: dict[str, Any],
) -> None:
    width = int(preview_cfg["width"])
    height = int(preview_cfg["height"])
    max_iter = int(preview_cfg["max_iter"])
    view = data["view"]
    xmin, xmax = float(view["xmin"]), float(view["xmax"])
    ymin, ymax = float(view["ymin"]), float(view["ymax"])

    xs = np.linspace(xmin, xmax, width, endpoint=False) + (xmax - xmin) / (2 * width)
    ys = np.linspace(ymax, ymin, height, endpoint=False) + (ymin - ymax) / (2 * height)
    X, Y = np.meshgrid(xs, ys)
    C = (X + 1j * Y).reshape(-1)
    Z = np.zeros_like(C)
    active = np.ones(C.size, dtype=bool)
    escaped_once = np.zeros(C.size, dtype=bool)
    escape_iteration = np.zeros(C.size, dtype=np.int32)
    previous_log_g = np.full(C.size, np.nan, dtype=np.float64)
    stable_count = np.zeros(C.size, dtype=np.int16)
    log_g = np.full(C.size, np.nan, dtype=np.float64)
    escape2 = float(data["render"].get("escape_radius", 2.0)) ** 2
    log10_2 = math.log10(2.0)
    gmax_log = math.log10(float(data["color"]["gmax"]))
    stability_tol = float(data["render"].get("potential_stability_tolerance", 1.0e-4))
    stability_steps = max(1, int(data["render"].get("potential_stability_steps", 2)))
    post_escape_max_steps = max(1, int(data["render"].get("post_escape_max_steps", 24)))

    for iteration in range(1, max_iter + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break

        with np.errstate(over="ignore", invalid="ignore"):
            Zi = Z[indices] * Z[indices] + C[indices]
            abs2 = Zi.real * Zi.real + Zi.imag * Zi.imag
        Z[indices] = Zi

        finite_mask = np.isfinite(abs2)
        if (~finite_mask).any():
            bad_indices = indices[~finite_mask]
            log_g[bad_indices] = gmax_log
            active[bad_indices] = False

        good_indices = indices[finite_mask]
        if good_indices.size == 0:
            continue
        good_abs2 = abs2[finite_mask]

        newly_escaped = (~escaped_once[good_indices]) & (good_abs2 > escape2)
        if newly_escaped.any():
            fresh = good_indices[newly_escaped]
            escaped_once[fresh] = True
            escape_iteration[fresh] = iteration

        pending_mask = escaped_once[good_indices]
        if not pending_mask.any():
            continue

        pending_indices = good_indices[pending_mask]
        pending_abs2 = good_abs2[pending_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            current = np.log10(0.5 * np.log(pending_abs2)) - iteration * log10_2
        current = np.where(np.isfinite(current), current, gmax_log)

        previous = previous_log_g[pending_indices]
        close = np.isfinite(previous) & (np.abs(current - previous) <= stability_tol)
        stable_count[pending_indices] = np.where(
            close,
            stable_count[pending_indices] + 1,
            0,
        )
        previous_log_g[pending_indices] = current

        done = (
            (stable_count[pending_indices] >= stability_steps)
            | ((iteration - escape_iteration[pending_indices]) >= post_escape_max_steps)
            | (iteration >= max_iter)
        )
        if done.any():
            finished = pending_indices[done]
            log_g[finished] = current[done]
            active[finished] = False

    palette = np.asarray(data["color"]["palette"], dtype=np.uint8)
    gmin_log = math.log10(float(data["color"]["gmin"]))
    gmax_log = math.log10(float(data["color"]["gmax"]))
    gamma = float(data["color"]["gamma"])
    rgb = np.zeros((C.size, 3), dtype=np.uint8)
    rgb[:] = np.asarray(data["color"]["interiorRgb"], dtype=np.uint8)
    escaped = np.isfinite(log_g)
    t = np.clip((log_g[escaped] - gmin_log) / (gmax_log - gmin_log), 0.0, 1.0)
    mapping = str(data["color"].get("mapping", "boundary")).strip().lower()
    if mapping == "boundary":
        palette_position = np.power(1.0 - t, gamma)
    elif mapping == "exterior":
        palette_position = np.power(t, gamma)
    else:
        raise ValueError("potential_color.mapping must be 'boundary' or 'exterior'.")
    palette_index = np.clip(np.rint(palette_position * 255.0), 0, 255).astype(int)
    rgb[escaped] = palette[palette_index]
    image = Image.fromarray(rgb.reshape(height, width, 3), "RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    for component in data["components"]:
        shape = str(component["shape"])
        if shape == "polygon":
            flat = component["points"]
            outline = np.asarray(
                [complex(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)],
                dtype=np.complex128,
            )
        else:
            outline = analytic_outline(
                shape,
                complex(*component.get("shapeCenter", component["center"])),
                float(component["size"]),
                float(component.get("angle", 0.0)),
                float(component.get("xi", 0.0)),
                samples=96 if shape == "circle" else 160,
            )
        pixels = [
            (
                (value.real - xmin) / (xmax - xmin) * width,
                (ymax - value.imag) / (ymax - ymin) * height,
            )
            for value in outline
        ]
        if pixels:
            draw.line(pixels + [pixels[0]], fill=(255, 255, 255, 70), width=1)

    save_png_atomic(image, out_dir / "preview.png")


def render_templates(template_dir: Path, out_dir: Path, text_cfg: dict[str, Any]) -> None:
    required = [
        "index.html",
        "demo.css",
        "mandelbrot_webgl.js",
        "demo.js",
        "mandelbrot_worker.js",
    ]
    missing = [name for name in required if not (template_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing template files in {template_dir}: {', '.join(missing)}"
        )

    replacements = {
        "__DEMO_TITLE__": str(text_cfg.get("title", "Mandelbrot component atlas")),
        "__DEMO_SUBTITLE__": str(text_cfg.get("subtitle", "Explore hyperbolic components")),
    }
    for name in required:
        text = (template_dir / name).read_text(encoding="utf-8")
        for token, replacement in replacements.items():
            text = text.replace(token, replacement)
        atomic_write_text(out_dir / name, text)




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--template-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--recompute-components", action="store_true")
    args = parser.parse_args()

    repo_cfg = RepoConfig.load(args.config, start=SCRIPT_DIR)
    config = repo_cfg.section("demo.atlas")
    component_cfg = merged(DEFAULT_CONFIG["components"], config.get("components"))
    view_cfg = merged(DEFAULT_CONFIG["view"], config.get("view"))
    render_cfg = merged(DEFAULT_CONFIG["render"], config.get("render"))
    color_cfg = merged(DEFAULT_CONFIG["potential_color"], config.get("potential_color"))
    interaction_cfg = merged(DEFAULT_CONFIG["interaction"], config.get("interaction"))
    preview_cfg = merged(DEFAULT_CONFIG["preview"], config.get("preview"))
    text_cfg = merged(DEFAULT_CONFIG["text"], config.get("text"))

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_cfg.path("paths.atlas_demo_output")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    template_dir = args.template_dir
    if not template_dir.is_absolute():
        template_dir = (SCRIPT_DIR / template_dir).resolve()

    catalogue = Catalogue(repo_cfg.path("paths.catalogue_root"))
    catalogue.ensure_layout()
    manifest = catalogue.load_manifest()

    print(f"project root: {repo_cfg.paths.project_root}")
    print(f"catalogue:    {catalogue.root}")
    print(f"revision:     {manifest.catalogue_revision}")
    print(f"output:       {out_dir}")
    print(f"templates:    {template_dir}")
    print(f"periods:      1..{component_cfg['max_period']}")

    prefit_path = out_dir / ".atlas_analytic_prefit.json"
    prefit_path = prepare_cpp_prefits(
        repo_cfg,
        catalogue,
        component_cfg,
        prefit_path,
        force=args.recompute_components,
    )
    prefitted_components = load_prefitted_components(prefit_path)
    if prefitted_components:
        print(f"loaded {len(prefitted_components)} typed-catalogue C++ fit(s)")

    signature = cache_signature(component_cfg, catalogue, prefit_path)
    cache_path = out_dir / ".atlas_component_cache.json"
    components: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    use_cache = bool(component_cfg.get("cache", True)) and not args.recompute_components
    if use_cache and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") != signature:
                raise ValueError("signature mismatch")
            components = cached["components"]
            summaries = cached["periodSummary"]
            print(f"reusing component cache: {cache_path}")
        except (OSError, ValueError, TypeError, KeyError):
            components, summaries = build_components(
                component_cfg,
                catalogue=catalogue,
                prefitted_components=prefitted_components,
            )
    else:
        components, summaries = build_components(
            component_cfg,
            catalogue=catalogue,
            prefitted_components=prefitted_components,
        )

    if not components:
        raise RuntimeError(
            "The catalogue query returned no polygon components at the configured cutoff."
        )

    atomic_write_text(
        cache_path,
        json.dumps({
            "signature": signature,
            "components": components,
            "periodSummary": summaries,
        }, separators=(",", ":"), allow_nan=False),
    )
    render_templates(template_dir, out_dir, text_cfg)

    gmin = float(color_cfg["gmin"])
    gmax = float(color_cfg["gmax"])
    if not 0.0 < gmin < gmax:
        raise ValueError("Require 0 < potential_color.gmin < potential_color.gmax.")

    data = {
        "assetVersion": str(time.time_ns()),
        "catalogueRevision": str(manifest.catalogue_revision),
        "title": text_cfg["title"],
        "subtitle": text_cfg["subtitle"],
        "maxPeriod": max(int(component["period"]) for component in components),
        "view": view_cfg,
        "render": render_cfg,
        "interaction": interaction_cfg,
        "color": {
            "scheme": color_cfg["scheme"],
            "reverse": bool(color_cfg["reverse"]),
            "mapping": str(color_cfg.get("mapping", "boundary")),
            "gmin": gmin,
            "gmax": gmax,
            "gamma": float(color_cfg["gamma"]),
            "interiorRgb": list(color_cfg["interior_rgb"]),
            "palette": make_palette(color_cfg),
        },
        "mandelbrotAreaReference": float(
            interaction_cfg.get("mandelbrot_area_reference", 1.5065918849)
        ),
        "components": components,
        "periodSummary": summaries,
    }

    render_preview(out_dir, data, preview_cfg)
    atomic_write_text(
        out_dir / "data.json",
        json.dumps(data, separators=(",", ":"), allow_nan=False),
    )
    atomic_write_text(
        out_dir / "README.md",
        "# Mandelbrot component atlas demo\n\n"
        "Generated from typed canonical `ComponentRecord` objects by "
        "`make_atlas_demo.py`. Canonical JSON and indexes are accessed only "
        "through the catalogue module.\n",
    )

    shape_counts: dict[str, int] = {}
    for component in components:
        shape = str(component.get("shape", "unknown"))
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    print("done")
    print(f"components: {len(components)}")
    print("analytic shapes: " + ", ".join(
        f"{shape}={count}" for shape, count in sorted(shape_counts.items())
    ))
    for name in (
        "data.json", "preview.png", "index.html", "mandelbrot_webgl.js",
        "demo.js", "mandelbrot_worker.js",
    ):
        path = out_dir / name
        print(f"{name}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
