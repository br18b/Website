#!/usr/bin/env python3
"""
Plot and summarize Mandelbrot hyperbolic-component areas produced by
component_area_scan.cpp.

This script performs no center finding, continuation, or multiplier-ring
tracing. It receives typed center and measurement records from ``AreaScanStore``,
selects one complete multiplier radius, computes per-period statistics, and
writes static PNG/SVG figures plus a statistics CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument
from components.catalogue.component_catalogue import (
    Catalogue,
)


# -----------------------------------------------------------------------------
# Config and paths
# -----------------------------------------------------------------------------

def trim(value: str) -> str:
    return value.strip()


def parse_bool(value: str) -> bool:
    return trim(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_kv_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key:
                raise ValueError(f"Empty config key at {path}:{line_number}")
            result[key] = value
    return result


def find_project_root_from(start: Path) -> Path:
    try:
        current = start.resolve()
    except OSError:
        current = start.absolute()

    while True:
        if (current / ".git").exists() or (current / ".root").exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def resolve_roots(config: dict[str, str]) -> tuple[Path, Path]:
    raw_code_root = config.get("code_root", "auto")
    if raw_code_root.strip().lower() == "auto":
        code_root = SCRIPT_DIR
    else:
        candidate = Path(raw_code_root).expanduser()
        if not candidate.is_absolute():
            candidate = SCRIPT_DIR / candidate
        code_root = candidate.resolve()

    raw_project_root = config.get("project_root", "auto")
    if raw_project_root.strip().lower() == "auto":
        project_root = find_project_root_from(code_root)
    else:
        text = raw_project_root
        text = text.replace("${code_root}", str(code_root))
        text = text.replace("$code_root", str(code_root))
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = code_root / candidate
        project_root = candidate.resolve()

    return code_root, project_root


def expand_path(
    raw: str,
    *,
    code_root: Path,
    project_root: Path,
    output_dir: Path | None = None,
) -> Path:
    text = raw
    replacements = {
        "${code_root}": str(code_root),
        "$code_root": str(code_root),
        "${project_root}": str(project_root),
        "$project_root": str(project_root),
    }
    if output_dir is not None:
        replacements["${output_dir}"] = str(output_dir)
        replacements["$output_dir"] = str(output_dir)

    # Longest first avoids replacing $output_dir inside ${output_dir}.
    for token in sorted(replacements, key=len, reverse=True):
        text = text.replace(token, replacements[token])

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = code_root / path
    return path.resolve()


@dataclass(frozen=True)
class PlotConfig:
    config_path: Path
    output_dir: Path
    plot_output_dir: Path
    rho_request: str
    min_period: int | None
    max_period: int | None
    colormap: str
    line_color: str
    range_max_color: str
    range_geometric_color: str
    range_min_color: str
    configured_radii: tuple[float, ...]
    formats: tuple[str, ...]
    dpi: int
    equal_area_rtol: float
    equal_area_error_factor: float
    equal_area_point_spacing: float
    equal_area_max_half_width: float
    marker_size: float
    alpha: float
    vertical_large_marker_max_period: int
    vertical_violin_disks_period_start: int
    vertical_violin_cardioids_period_start: int
    vertical_violin_disks_min_points: int
    vertical_violin_cardioids_min_points: int
    vertical_large_marker_scale: float
    vertical_regular_marker_scale: float
    vertical_outlier_marker_scale: float
    vertical_violin_quantile_low: float
    vertical_violin_quantile_high: float
    vertical_violin_bins: int
    vertical_violin_smoothing_bins: float
    vertical_violin_min_half_width: float
    vertical_violin_max_half_width: float
    vertical_cardioid_color: str
    vertical_cardioid_edge_color: str
    vertical_cardioid_violin_outer_color: str
    vertical_cardioid_violin_inner_color: str
    vertical_disk_violin_outer_color: str
    vertical_disk_violin_inner_color: str
    vertical_disk_color: str
    vertical_violin_count_labels: bool
    vertical_violin_count_font_size: float
    vertical_violin_count_box_mode: str
    vertical_violin_count_box_padding: float
    vertical_violin_count_min_gap_points: float
    show: bool


def optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"", "auto", "none"}:
        return None
    return int(text)


def load_plot_config(args: argparse.Namespace) -> PlotConfig:
    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    raw = repo.section("component_area_scan")
    catalogue = Catalogue(repo.path("paths.catalogue_root"))
    output_dir = catalogue.exports_path

    raw_plot_output = str(raw.get("plot_output_dir", "$output_dir/plots"))
    raw_plot_output = raw_plot_output.replace("${output_dir}", str(output_dir))
    raw_plot_output = raw_plot_output.replace("$output_dir", str(output_dir))
    plot_output_dir = Path(raw_plot_output).expanduser()
    if not plot_output_dir.is_absolute():
        plot_output_dir = repo.paths.code_root / plot_output_dir
    plot_output_dir = plot_output_dir.resolve()

    formats_value = args.formats if args.formats is not None else raw.get("plot_formats", ["png", "svg"])
    if isinstance(formats_value, str):
        formats = tuple(item.strip().lower() for item in formats_value.split(",") if item.strip())
    else:
        formats = tuple(str(item).strip().lower() for item in formats_value if str(item).strip())
    unsupported = [item for item in formats if item not in {"png", "svg", "pdf"}]
    if unsupported:
        raise ValueError("Unsupported plot formats: " + ", ".join(unsupported))
    if not formats:
        raise ValueError("At least one plot format is required.")

    configured_radii = tuple(sorted({float(value) for value in raw.get("radii", [])}))
    config = PlotConfig(
        config_path=repo.paths.config_path,
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        rho_request=args.rho or str(raw.get("plot_rho", "auto")),
        min_period=args.min_period if args.min_period is not None else optional_int(str(raw.get("plot_min_period", "1"))),
        max_period=args.max_period if args.max_period is not None else optional_int(str(raw.get("plot_max_period", "auto"))),
        colormap=args.colormap or str(raw.get("plot_colormap", "viridis")),
        line_color=args.line_color or str(raw.get("plot_line_color", "tab:blue")),
        range_max_color=str(raw.get("plot_range_max_color", "tab:red")),
        range_geometric_color=str(raw.get("plot_range_geometric_color", "tab:purple")),
        range_min_color=str(raw.get("plot_range_min_color", "tab:blue")),
        configured_radii=configured_radii,
        formats=formats,
        dpi=max(72, int(args.dpi or raw.get("plot_dpi", 180))),
        equal_area_rtol=float(raw.get("plot_equal_area_rtol", 1e-8)),
        equal_area_error_factor=float(raw.get("plot_equal_area_error_factor", 4)),
        equal_area_point_spacing=float(raw.get("plot_equal_area_point_spacing", 0.045)),
        equal_area_max_half_width=float(raw.get("plot_equal_area_max_half_width", 0.18)),
        marker_size=float(raw.get("plot_marker_size", 18)),
        alpha=float(raw.get("plot_alpha", 0.78)),
        vertical_large_marker_max_period=max(
            1, int(raw.get("plot_vertical_large_marker_max_period", 3))
        ),
        vertical_violin_disks_period_start=max(
            1, int(raw.get("plot_vertical_violin_disks_period_start", 999999))
        ),
        vertical_violin_cardioids_period_start=max(
            1, int(raw.get("plot_vertical_violin_cardioids_period_start", 8))
        ),
        vertical_violin_disks_min_points=max(
            4, int(raw.get("plot_vertical_violin_disks_min_points", 50))
        ),
        vertical_violin_cardioids_min_points=max(
            4, int(raw.get("plot_vertical_violin_cardioids_min_points", 50))
        ),
        vertical_large_marker_scale=max(
            0.01, float(raw.get("plot_vertical_large_marker_scale", 3.2))
        ),
        vertical_regular_marker_scale=max(
            0.01, float(raw.get("plot_vertical_regular_marker_scale", 1.0))
        ),
        vertical_outlier_marker_scale=max(
            0.01, float(raw.get("plot_vertical_outlier_marker_scale", 0.65))
        ),
        vertical_violin_quantile_low=float(
            raw.get("plot_vertical_violin_quantile_low", 0.05)
        ),
        vertical_violin_quantile_high=float(
            raw.get("plot_vertical_violin_quantile_high", 0.95)
        ),
        vertical_violin_bins=max(
            12, int(raw.get("plot_vertical_violin_bins", 128))
        ),
        vertical_violin_smoothing_bins=max(
            0.0, float(raw.get("plot_vertical_violin_smoothing_bins", 1.5))
        ),
        vertical_violin_min_half_width=max(
            0.01, float(raw.get("plot_vertical_violin_min_half_width", 0.10))
        ),
        vertical_violin_max_half_width=max(
            0.01, float(raw.get("plot_vertical_violin_max_half_width", 0.34))
        ),
        vertical_cardioid_color=str(
            raw.get("plot_vertical_cardioid_color", "#159bd7")
        ),
        vertical_cardioid_edge_color=str(
            raw.get("plot_vertical_cardioid_edge_color", "black")
        ),
        vertical_cardioid_violin_outer_color=str(
            raw.get(
                "plot_vertical_cardioid_violin_outer_color",
                raw.get("plot_vertical_violin_outer_color", "#0072b2"),
            )
        ),
        vertical_cardioid_violin_inner_color=str(
            raw.get(
                "plot_vertical_cardioid_violin_inner_color",
                raw.get("plot_vertical_violin_inner_color", "#8bd3f7"),
            )
        ),
        vertical_disk_violin_outer_color=str(
            raw.get("plot_vertical_disk_violin_outer_color", "#bd6f22")
        ),
        vertical_disk_violin_inner_color=str(
            raw.get("plot_vertical_disk_violin_inner_color", "#f8c48f")
        ),
        vertical_disk_color=str(
            raw.get("plot_vertical_disk_color", "#f28e2b")
        ),
        vertical_violin_count_labels=bool(
            raw.get("plot_vertical_violin_count_labels", True)
        ),
        vertical_violin_count_font_size=max(
            1.0, float(raw.get("plot_vertical_violin_count_font_size", 8.0))
        ),
        vertical_violin_count_box_mode=str(
            raw.get("plot_vertical_violin_count_box_mode", "fit")
        ).strip().lower(),
        vertical_violin_count_box_padding=max(
            0.0, float(raw.get("plot_vertical_violin_count_box_padding", 0.28))
        ),
        vertical_violin_count_min_gap_points=max(
            0.0, float(raw.get("plot_vertical_violin_count_min_gap_points", 5.0))
        ),
        show=args.show or bool(raw.get("plot_show", False)),
    )
    if not (
        0.0 <= config.vertical_violin_quantile_low
        < config.vertical_violin_quantile_high <= 1.0
    ):
        raise ValueError(
            "Require 0 <= plot_vertical_violin_quantile_low < "
            "plot_vertical_violin_quantile_high <= 1."
        )
    if config.vertical_violin_max_half_width < config.vertical_violin_min_half_width:
        raise ValueError(
            "plot_vertical_violin_max_half_width cannot be smaller than "
            "plot_vertical_violin_min_half_width."
        )
    if config.vertical_violin_count_box_mode not in {"fit", "uniform"}:
        raise ValueError(
            "plot_vertical_violin_count_box_mode must be 'fit' or 'uniform'."
        )
    return config


# -----------------------------------------------------------------------------
# CSV loading and completeness
# -----------------------------------------------------------------------------

class InlineProgress:
    """One-line terminal progress display with a quiet redirected fallback."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(0, total)
        self.started = time.monotonic()
        self.terminal = sys.stdout.isatty()
        self.last_render = 0.0

    def update(self, current: int, detail: str = "", *, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - self.last_render < 0.12:
            return
        self.last_render = now
        current = max(0, min(current, self.total)) if self.total else current
        fraction = current / self.total if self.total else 1.0
        width = 30
        filled = max(0, min(width, round(fraction * width)))
        elapsed = now - self.started
        line = (
            f"  {self.label} [{'#' * filled}{'-' * (width - filled)}] "
            f"{100.0 * fraction:5.1f}% {current:,}/{self.total:,}"
        )
        if detail:
            line += f" | {detail}"
        line += f" | elapsed={format_seconds(elapsed)}"
        if self.terminal:
            print(f"\r\033[2K{line}", end="\n" if final else "", flush=True)
        elif final:
            print(line, flush=True)


def format_seconds(seconds: float) -> str:
    whole = max(0, int(seconds))
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}:{whole % 60:02d}"
    return f"{whole // 3600}:{(whole // 60) % 60:02d}:{whole % 60:02d}"


def load_csv_rows(
    path: Path,
    label: str,
    *,
    keep_fields: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    total_bytes = max(1, path.stat().st_size)
    progress = InlineProgress(label, total_bytes)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, source in enumerate(reader, start=1):
            if keep_fields is None:
                rows.append(dict(source))
            else:
                rows.append({name: source.get(name, "") for name in keep_fields})
            if row_number % 2048 == 0:
                try:
                    position = min(total_bytes, handle.buffer.tell())
                except (AttributeError, OSError):
                    position = 0
                progress.update(position, f"rows={row_number:,}")
    progress.update(total_bytes, f"rows={len(rows):,}", final=True)
    return rows

def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def expected_counts_from_centers(
    center_rows: Iterable[dict[str, str]],
) -> dict[int, int]:
    indices: dict[int, set[int]] = defaultdict(set)
    declared: dict[int, int] = {}

    for row in center_rows:
        period = parse_int(row.get("period"))
        component = parse_int(row.get("component_index"))
        if period < 1 or component < 0:
            continue
        indices[period].add(component)
        expected = parse_int(row.get("expected_period_count"), -1)
        if expected > 0:
            declared[period] = max(declared.get(period, 0), expected)

    result: dict[int, int] = {}
    for period, components in indices.items():
        actual = len(components)
        expected = declared.get(period, actual)
        if actual != expected:
            raise RuntimeError(
                f"The center store is incomplete for period {period}: "
                f"contains {actual} unique centers, declares {expected}."
            )
        result[period] = expected
    return result


def canonical_rho(
    rho: float,
    configured_radii: tuple[float, ...],
) -> float:
    if not configured_radii or not math.isfinite(rho):
        return rho
    nearest = min(configured_radii, key=lambda value: abs(value - rho))
    tolerance = max(
        1.0e-14,
        1024.0 * np.finfo(float).eps * max(1.0, abs(nearest)),
    )
    return nearest if abs(rho - nearest) <= tolerance else rho


def row_relative_error(row: dict[str, str]) -> float:
    area = abs(parse_float(row.get("area_estimate")))
    error = abs(parse_float(row.get("error_estimate")))
    if math.isfinite(area) and area > 0 and math.isfinite(error):
        return error / area
    return math.inf


def row_quality(row: dict[str, str]) -> tuple[int, float, int]:
    area = parse_float(row.get("area_estimate"))
    usable = int(
        is_true(row.get("converged"))
        and math.isfinite(area)
        and area > 0
    )
    # Higher tuple wins: usable, smaller relative error, larger theta grid.
    return (
        usable,
        -row_relative_error(row),
        parse_int(row.get("theta_points"), 0),
    )


def deduplicate_measurements(
    rows: Iterable[dict[str, str]],
    configured_radii: tuple[float, ...],
) -> tuple[list[dict[str, str]], int]:
    if not isinstance(rows, list):
        rows = list(rows)
    progress = InlineProgress("canonicalizing measurements", len(rows))
    by_key: dict[tuple[int, int, float], dict[str, str]] = {}
    total_valid_rows = 0

    for index, source_row in enumerate(rows, start=1):
        period = parse_int(source_row.get("period"))
        component = parse_int(source_row.get("component_index"))
        rho = parse_float(source_row.get("rho"))
        if period >= 1 and component >= 0 and math.isfinite(rho):
            total_valid_rows += 1
            rho = canonical_rho(rho, configured_radii)
            row = source_row
            row["rho"] = f"{rho:.17g}"
            key = (period, component, rho)

            current = by_key.get(key)
            if current is None or row_quality(row) >= row_quality(current):
                by_key[key] = row
        if index % 4096 == 0 or index == len(rows):
            progress.update(
                index,
                f"unique={len(by_key):,}",
                final=index == len(rows),
            )

    rows_out = list(by_key.values())
    sort_started = time.monotonic()
    rows_out.sort(
        key=lambda row: (
            parse_int(row.get("period")),
            parse_int(row.get("component_index")),
            parse_float(row.get("rho")),
        )
    )
    if rows_out:
        print(
            f"  sorted {len(rows_out):,} canonical measurement rows "
            f"in {format_seconds(time.monotonic() - sort_started)}",
            flush=True,
        )
    return rows_out, total_valid_rows - len(rows_out)


def usable_row(row: dict[str, str]) -> bool:
    area = parse_float(row.get("area_estimate"))
    return is_true(row.get("converged")) and math.isfinite(area) and area > 0


def requested_periods(
    expected_counts: dict[int, int],
    config: PlotConfig,
) -> list[int]:
    periods = sorted(expected_counts)
    if config.min_period is not None:
        periods = [period for period in periods if period >= config.min_period]
    if config.max_period is not None:
        periods = [period for period in periods if period <= config.max_period]
    if not periods:
        raise RuntimeError("No center periods match the requested plotting range.")
    return periods


def complete_rows_for_rho(
    rows: list[dict[str, str]],
    periods: list[int],
    expected_counts: dict[int, int],
    rho: float,
) -> list[dict[str, str]] | None:
    selected = [
        row
        for row in rows
        if parse_int(row.get("period")) in periods
        and math.isclose(
            parse_float(row.get("rho")),
            rho,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        and usable_row(row)
    ]

    grouped: dict[int, set[int]] = defaultdict(set)
    for row in selected:
        grouped[parse_int(row.get("period"))].add(
            parse_int(row.get("component_index"))
        )

    incomplete = {
        period: (len(grouped.get(period, set())), expected_counts[period])
        for period in periods
        if len(grouped.get(period, set())) != expected_counts[period]
    }
    if incomplete:
        return None
    return selected


def choose_rho(
    rows: list[dict[str, str]],
    periods: list[int],
    expected_counts: dict[int, int],
    request: str,
    configured_radii: tuple[float, ...],
) -> tuple[float, list[dict[str, str]]]:
    period_set = set(periods)
    usable_by_rho: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        period = parse_int(row.get("period"))
        rho = parse_float(row.get("rho"))
        if period in period_set and math.isfinite(rho) and usable_row(row):
            usable_by_rho[rho].append(row)

    def complete(candidate: float) -> list[dict[str, str]] | None:
        selected = usable_by_rho.get(candidate, [])
        counts: dict[int, set[int]] = defaultdict(set)
        for row in selected:
            counts[parse_int(row.get("period"))].add(
                parse_int(row.get("component_index"))
            )
        if any(
            len(counts.get(period, set())) != expected_counts[period]
            for period in periods
        ):
            return None
        return selected

    if request.strip().lower() != "auto":
        rho = float(request)
        nearest = min(usable_by_rho, key=lambda value: abs(value - rho), default=rho)
        if not math.isclose(nearest, rho, rel_tol=0.0, abs_tol=1.0e-14):
            nearest = rho
        selected = complete(nearest)
        if selected is None:
            raise RuntimeError(
                f"Requested rho={rho:.12g} is not complete and converged "
                "for every selected period."
            )
        return nearest, selected

    available = set(usable_by_rho)
    candidates = [
        rho for rho in sorted(configured_radii, reverse=True)
        if rho in available
    ]
    candidates.extend(
        sorted(available.difference(candidates), reverse=True)
    )

    for rho in candidates:
        selected = complete(rho)
        if selected is not None:
            return rho, selected

    diagnostics: list[str] = []
    for rho in candidates[:8]:
        per_period: list[str] = []
        for period in periods:
            count = len({
                parse_int(row.get("component_index"))
                for row in usable_by_rho.get(rho, [])
                if parse_int(row.get("period")) == period
            })
            per_period.append(
                f"p{period}:{count}/{expected_counts[period]}"
            )
        diagnostics.append(f"rho={rho:.12g} " + " ".join(per_period))

    raise RuntimeError(
        "No multiplier radius is complete across all selected periods.\n"
        + "\n".join(diagnostics)
    )


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodStats:
    period: int
    components: int
    total_area: float
    cumulative_area: float
    min_area: float
    max_area: float
    arithmetic_mean_area: float
    geometric_mean_area: float
    median_area: float
    p10_area: float
    p90_area: float


def compute_statistics(
    rows: list[dict[str, str]],
    periods: list[int],
) -> list[PeriodStats]:
    areas_by_period: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        period = parse_int(row.get("period"))
        area = parse_float(row.get("area_estimate"))
        if period in periods and math.isfinite(area) and area > 0:
            areas_by_period[period].append(area)

    result: list[PeriodStats] = []
    cumulative = 0.0
    for period in periods:
        values = np.asarray(areas_by_period[period], dtype=float)
        if values.size == 0:
            raise RuntimeError(f"No positive areas available for period {period}.")
        total = float(np.sum(values))
        cumulative += total
        geometric = float(np.exp(np.mean(np.log(values))))
        result.append(
            PeriodStats(
                period=period,
                components=int(values.size),
                total_area=total,
                cumulative_area=cumulative,
                min_area=float(np.min(values)),
                max_area=float(np.max(values)),
                arithmetic_mean_area=float(np.mean(values)),
                geometric_mean_area=geometric,
                median_area=float(np.median(values)),
                p10_area=float(np.quantile(values, 0.10)),
                p90_area=float(np.quantile(values, 0.90)),
            )
        )
    return result


def rho_tag(rho: float) -> str:
    return f"{rho:.12g}".replace("-", "m").replace(".", "p")


def atomic_write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it in the active environment."
        ) from exc
    return plt


def get_colormap(name: str):
    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except KeyError as exc:
        available = ", ".join(sorted(matplotlib.colormaps))
        raise RuntimeError(
            f"Unknown Matplotlib colormap {name!r}. Available names include:\n"
            f"{available}"
        ) from exc


def normalized_period_colors(periods: list[int], colormap: str) -> dict[int, Any]:
    cmap = get_colormap(colormap)
    if len(periods) == 1:
        return {periods[0]: cmap(0.62)}
    low = min(periods)
    span = max(periods) - low
    return {
        period: cmap((period - low) / span)
        for period in periods
    }


def cardioid_marker_path(samples: int = 48) -> Any:
    """Return a normalized right-facing cardioid for use as a scatter marker."""
    from matplotlib.path import Path as MatplotlibPath

    count = max(16, int(samples))
    phase = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    x = np.sin(0.5 * phase) + np.cos(phase) - 0.5 * np.cos(2.0 * phase)
    y = 0.9 * (np.sin(phase) - 0.5 * np.sin(2.0 * phase))

    # Center the marker's bounding box on the scatter coordinate, then scale it
    # to Matplotlib's conventional unit marker box.
    x -= 0.5 * (float(np.min(x)) + float(np.max(x)))
    y -= 0.5 * (float(np.min(y)) + float(np.max(y)))
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0)
    vertices = np.column_stack((x / scale, y / scale))
    vertices = np.vstack((vertices, vertices[0]))

    codes = np.full(vertices.shape[0], MatplotlibPath.LINETO, dtype=np.uint8)
    codes[0] = MatplotlibPath.MOVETO
    codes[-1] = MatplotlibPath.CLOSEPOLY
    return MatplotlibPath(vertices, codes)


def row_error(row: dict[str, str]) -> float:
    value = parse_float(row.get("error_estimate"), 0.0)
    return value if math.isfinite(value) and value >= 0 else 0.0


def same_area(
    row_a: dict[str, str],
    row_b: dict[str, str],
    *,
    relative_tolerance: float,
    error_factor: float,
) -> bool:
    area_a = parse_float(row_a.get("area_estimate"))
    area_b = parse_float(row_b.get("area_estimate"))
    uncertainty = error_factor * (row_error(row_a) + row_error(row_b))
    relative = relative_tolerance * max(abs(area_a), abs(area_b))
    return abs(area_a - area_b) <= max(uncertainty, relative)


def dispersed_x_positions(
    rows: list[dict[str, str]],
    periods: list[int],
    config: PlotConfig,
) -> dict[tuple[int, int], float]:
    positions: dict[tuple[int, int], float] = {}

    for period in periods:
        period_rows = sorted(
            [
                row
                for row in rows
                if parse_int(row.get("period")) == period
            ],
            key=lambda row: (
                parse_float(row.get("area_estimate")),
                parse_int(row.get("component_index")),
            ),
        )

        groups: list[list[dict[str, str]]] = []
        for row in period_rows:
            if groups and same_area(
                groups[-1][0],
                row,
                relative_tolerance=config.equal_area_rtol,
                error_factor=config.equal_area_error_factor,
            ):
                groups[-1].append(row)
            else:
                groups.append([row])

        for group in groups:
            count = len(group)
            if count == 1:
                offsets = np.zeros(1)
            else:
                offsets = (
                    np.arange(count, dtype=float) - 0.5 * (count - 1)
                ) * config.equal_area_point_spacing
                maximum = float(np.max(np.abs(offsets)))
                if (
                    maximum > config.equal_area_max_half_width
                    and maximum > 0
                ):
                    offsets *= config.equal_area_max_half_width / maximum

            for row, offset in zip(group, offsets):
                key = (
                    period,
                    parse_int(row.get("component_index")),
                )
                positions[key] = period + float(offset)

    return positions


def save_figure(
    fig: Any,
    stem: Path,
    config: PlotConfig,
) -> list[Path]:
    output_paths: list[Path] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in config.formats:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = config.dpi
        fig.savefig(path, **kwargs)
        output_paths.append(path)
    return output_paths


def decorate_period_axis(ax: Any, periods: list[int]) -> None:
    ax.set_xticks(periods)
    ax.set_xlim(min(periods) - 0.45, max(periods) + 0.45)
    ax.grid(True, which="both", alpha=0.22)


SHAPE_INDEX_SCHEMA = "mandelbrot-shape-index-v1"


def load_shape_index(
    path: Path,
    *,
    catalogue_revision: int,
    periods: list[int],
) -> dict[tuple[int, int], str] | None:
    if not path.is_file():
        return None
    result: dict[tuple[int, int], str] = {}
    metadata_checked = False
    requested_min = min(periods)
    requested_max = max(periods)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not metadata_checked:
                    if row.get("schema") != SHAPE_INDEX_SCHEMA:
                        return None
                    if parse_int(row.get("catalogue_revision"), -1) != catalogue_revision:
                        return None
                    if parse_int(row.get("index_min_period"), requested_min + 1) > requested_min:
                        return None
                    if parse_int(row.get("index_max_period"), requested_max - 1) < requested_max:
                        return None
                    metadata_checked = True
                period = parse_int(row.get("period"))
                component_index = parse_int(row.get("component_index"))
                if period in periods and component_index >= 0:
                    result[(period, component_index)] = (
                        row.get("shape_class") or "unknown"
                    )
    except (OSError, csv.Error, ValueError):
        return None
    return result if metadata_checked else None


def rebuild_shape_index_python(
    catalogue: Catalogue,
    periods: list[int],
    path: Path,
    catalogue_revision: int,
) -> dict[tuple[int, int], str]:
    """Slow fallback when the C++ catalogue helper has not been rebuilt yet."""
    ids: list[str] = []
    for period in periods:
        try:
            ids.extend(catalogue.load_period(period).component_ids)
        except (OSError, ValueError, KeyError):
            continue
    ids = sorted(set(ids))

    progress = InlineProgress("loading component classifications", len(ids))
    result: dict[tuple[int, int], str] = {}
    index_rows: list[dict[str, Any]] = []
    for index, component_id in enumerate(ids, start=1):
        try:
            component = catalogue.load_component(component_id)
            exact = catalogue.exact_period_index(component)
            if exact is not None and exact.period in periods:
                shape = component.classification.shape_class or "unknown"
                result[(exact.period, exact.component_index)] = shape
                index_rows.append({
                    "schema": SHAPE_INDEX_SCHEMA,
                    "catalogue_revision": catalogue_revision,
                    "index_min_period": min(periods),
                    "index_max_period": max(periods),
                    "period": exact.period,
                    "component_index": exact.component_index,
                    "shape_class": shape,
                })
        except (OSError, ValueError, KeyError):
            pass
        if index % 64 == 0 or index == len(ids):
            progress.update(
                index,
                f"classes={len(result):,}",
                final=index == len(ids),
            )

    atomic_write_csv(
        path,
        index_rows,
        [
            "schema",
            "catalogue_revision",
            "index_min_period",
            "index_max_period",
            "period",
            "component_index",
            "shape_class",
        ],
    )
    return result


def catalogue_shape_classes(
    catalogue: Catalogue,
    periods: list[int],
    config_path: Path,
) -> dict[tuple[int, int], str]:
    """Load a compact exact-index→shape map, rebuilding it only when stale."""
    manifest = catalogue.load_manifest()
    index_path = catalogue.exports_path / "component_shape_classes.csv"
    cached = load_shape_index(
        index_path,
        catalogue_revision=manifest.catalogue_revision,
        periods=periods,
    )
    if cached is not None:
        print(
            f"  shape index: {index_path} ({len(cached):,} classifications, cached)",
            flush=True,
        )
        return cached

    tool = CODE_ROOT / "bin" / "catalogue_tool"
    if tool.is_file() and os.access(tool, os.X_OK):
        print("  compact shape index is missing or stale; rebuilding it in C++...", flush=True)
        command = [
            str(tool),
            "--config",
            str(config_path),
            "--command",
            "export-shape-index",
            "--min-period",
            str(min(periods)),
            "--max-period",
            str(max(periods)),
            "--output",
            str(index_path),
        ]
        try:
            subprocess.run(command, check=True)
            cached = load_shape_index(
                index_path,
                catalogue_revision=manifest.catalogue_revision,
                periods=periods,
            )
            if cached is not None:
                return cached
        except (OSError, subprocess.CalledProcessError):
            print(
                "  C++ shape-index export unavailable; using the slower Python fallback.",
                flush=True,
            )

    return rebuild_shape_index_python(
        catalogue,
        periods,
        index_path,
        manifest.catalogue_revision,
    )


def measurement_catalogue_key(row: dict[str, str]) -> tuple[int, int]:
    """Return the canonical upper-half-plane exact-scanner identity.

    AreaScanStore contains both members of a conjugate pair, while the
    catalogue stores only the upper-half-plane representative.  The row's
    typed centre and conjugate index tell us which exact index owns the
    canonical classification.
    """
    period = parse_int(row.get("period"))
    component_index = parse_int(row.get("component_index"))
    if parse_float(row.get("center_im")) < 0:
        conjugate_index = parse_int(row.get("conjugate_index"), -1)
        if conjugate_index >= 0:
            component_index = conjugate_index
    return period, component_index


def smooth_histogram(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth a one-dimensional histogram without requiring SciPy."""
    result = np.asarray(values, dtype=float)
    if result.size < 2 or sigma_bins <= 0:
        return result.copy()

    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= float(np.sum(kernel))
    mode = "reflect" if result.size > 1 else "edge"
    padded = np.pad(result, radius, mode=mode)
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def log_violin_profile(
    areas: np.ndarray,
    config: PlotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return y coordinates, normalized density, and an outlier mask.

    The histogram is built directly in log10(area), because that transformed
    coordinate is the meaningful visual space on a logarithmic y axis.  Only
    the configured central quantile interval contributes to the violin; rows
    outside it are returned as explicit cardioid-marker outliers.
    """
    values = np.asarray(areas, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    # Period-start triggers deliberately override the configured point-count
    # threshold. Still require enough samples to form a meaningful profile.
    if int(np.count_nonzero(valid)) < 4:
        return None

    logs = np.full(values.shape, np.nan, dtype=float)
    logs[valid] = np.log10(values[valid])
    valid_logs = logs[valid]
    lower, upper = np.quantile(
        valid_logs,
        [
            config.vertical_violin_quantile_low,
            config.vertical_violin_quantile_high,
        ],
    )
    outlier_mask = valid & ((logs < lower) | (logs > upper))
    central = valid_logs[(valid_logs >= lower) & (valid_logs <= upper)]
    if central.size < 4:
        return None

    span = float(upper - lower)
    if not math.isfinite(span) or span <= 1.0e-14:
        return None

    # More samples support more detail, but cap the bins so the profile remains
    # smooth and cheap even for the very large period-17 cloud.
    adaptive_bins = max(12, int(round(2.5 * math.sqrt(float(central.size)))))
    bin_count = min(config.vertical_violin_bins, adaptive_bins)
    counts, edges = np.histogram(
        central,
        bins=bin_count,
        range=(float(lower), float(upper)),
    )
    density = smooth_histogram(
        counts.astype(float),
        config.vertical_violin_smoothing_bins,
    )
    maximum = float(np.max(density)) if density.size else 0.0
    if not math.isfinite(maximum) or maximum <= 0:
        return None
    density /= maximum

    centers = 0.5 * (edges[:-1] + edges[1:])
    log_y = np.concatenate(([edges[0]], centers, [edges[-1]]))
    normalized_density = np.concatenate(([0.0], density, [0.0]))
    return np.power(10.0, log_y), normalized_density, outlier_mask


def violin_half_width(
    period: int,
    violin_periods: list[int],
    config: PlotConfig,
) -> float:
    if len(violin_periods) < 2:
        return config.vertical_violin_max_half_width
    first = min(violin_periods)
    last = max(violin_periods)
    fraction = (period - first) / max(1, last - first)
    return (
        config.vertical_violin_min_half_width
        + fraction
        * (
            config.vertical_violin_max_half_width
            - config.vertical_violin_min_half_width
        )
    )


def blend_color(color: str, target: str, fraction: float) -> tuple[float, float, float]:
    """Blend a Matplotlib color toward another color."""
    from matplotlib.colors import to_rgb

    base = np.asarray(to_rgb(color), dtype=float)
    destination = np.asarray(to_rgb(target), dtype=float)
    amount = float(np.clip(fraction, 0.0, 1.0))
    mixed = (1.0 - amount) * base + amount * destination
    return tuple(float(value) for value in mixed)


def _rgb_array(color: Any) -> np.ndarray:
    """Return an RGB color as a float NumPy array in [0, 1]."""
    from matplotlib.colors import to_rgb

    return np.asarray(to_rgb(color), dtype=float)


def _radial_gradient_rgb(
    outer_color: Any,
    inner_color: Any,
    radial_fraction: np.ndarray,
) -> np.ndarray:
    """Interpolate from inner at the centre to outer at the violin edge."""
    outer = _rgb_array(outer_color)
    inner = _rgb_array(inner_color)
    fraction = np.clip(radial_fraction, 0.0, 1.0)[..., None]
    return inner + fraction * (outer - inner)


def _profile_widths_on_log_grid(
    y_values: np.ndarray,
    density: np.ndarray,
    half_width: float,
    log_y_grid: np.ndarray,
) -> np.ndarray:
    log_profile_y = np.log10(np.asarray(y_values, dtype=float))
    widths = half_width * np.asarray(density, dtype=float)
    return np.interp(log_y_grid, log_profile_y, widths, left=0.0, right=0.0)


def plot_solid_blended_violins(
    ax: Any,
    *,
    period: int,
    cardioid_profile: tuple[np.ndarray, np.ndarray, float] | None,
    disk_profile: tuple[np.ndarray, np.ndarray, float] | None,
    config: PlotConfig,
) -> None:
    """Draw opaque violin gradients and explicitly average overlap colors.

    Each family's local color is determined from its own normalized horizontal
    position: the configured inner color at the centre and outer color at the
    edge. Where both families occupy a pixel, the displayed RGB value is the
    arithmetic mean of those two independently evaluated local colors. Alpha is
    one throughout every violin and zero outside, so neither the white
    background nor draw order affects the mixed color.
    """
    from matplotlib.image import NonUniformImage

    profiles = [profile for profile in (cardioid_profile, disk_profile) if profile]
    if not profiles:
        return

    log_min = min(float(np.log10(np.min(profile[0]))) for profile in profiles)
    log_max = max(float(np.log10(np.max(profile[0]))) for profile in profiles)
    max_half_width = max(float(profile[2]) for profile in profiles)
    if not (
        math.isfinite(log_min)
        and math.isfinite(log_max)
        and log_max > log_min
        and max_half_width > 0
    ):
        return

    # The raster is defined directly on a uniform log-area grid, matching the
    # visual coordinate system of the logarithmic y axis. NonUniformImage then
    # maps those samples to the corresponding positive area coordinates.
    y_samples = max(
        320,
        min(768, 4 * max(len(profile[0]) for profile in profiles)),
    )
    x_samples = 241
    log_y = np.linspace(log_min, log_max, y_samples)
    y = np.power(10.0, log_y)
    x = np.linspace(
        period - 1.03 * max_half_width,
        period + 1.03 * max_half_width,
        x_samples,
    )
    distance = np.abs(x[None, :] - float(period))

    rgba = np.zeros((y_samples, x_samples, 4), dtype=float)
    family_rgb: dict[str, np.ndarray] = {}
    family_mask: dict[str, np.ndarray] = {}

    if cardioid_profile is not None:
        cardioid_y, cardioid_density, cardioid_half_width = cardioid_profile
        widths = _profile_widths_on_log_grid(
            cardioid_y,
            cardioid_density,
            cardioid_half_width,
            log_y,
        )[:, None]
        mask = (widths > 0.0) & (distance <= widths)
        radial = np.divide(
            distance,
            widths,
            out=np.ones((y_samples, x_samples), dtype=float),
            where=widths > 0.0,
        )
        family_rgb["cardioid"] = _radial_gradient_rgb(
            config.vertical_cardioid_violin_outer_color,
            config.vertical_cardioid_violin_inner_color,
            radial,
        )
        family_mask["cardioid"] = mask

    if disk_profile is not None:
        disk_y, disk_density, disk_half_width = disk_profile
        widths = _profile_widths_on_log_grid(
            disk_y,
            disk_density,
            disk_half_width,
            log_y,
        )[:, None]
        mask = (widths > 0.0) & (distance <= widths)
        radial = np.divide(
            distance,
            widths,
            out=np.ones((y_samples, x_samples), dtype=float),
            where=widths > 0.0,
        )
        family_rgb["disk"] = _radial_gradient_rgb(
            config.vertical_disk_violin_outer_color,
            config.vertical_disk_violin_inner_color,
            radial,
        )
        family_mask["disk"] = mask

    cardioid_mask = family_mask.get(
        "cardioid", np.zeros((y_samples, x_samples), dtype=bool)
    )
    disk_mask = family_mask.get(
        "disk", np.zeros((y_samples, x_samples), dtype=bool)
    )
    cardioid_only = cardioid_mask & ~disk_mask
    disk_only = disk_mask & ~cardioid_mask
    overlap = cardioid_mask & disk_mask

    if np.any(cardioid_only):
        rgba[cardioid_only, :3] = family_rgb["cardioid"][cardioid_only]
    if np.any(disk_only):
        rgba[disk_only, :3] = family_rgb["disk"][disk_only]
    if np.any(overlap):
        rgba[overlap, :3] = 0.5 * (
            family_rgb["cardioid"][overlap] + family_rgb["disk"][overlap]
        )
    rgba[cardioid_mask | disk_mask, 3] = 1.0

    image = NonUniformImage(
        ax,
        interpolation="nearest",
        extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        zorder=1.0,
    )
    image.set_data(x, y, rgba)
    ax.add_image(image)

    # Keep crisp vector outlines over the opaque raster body.
    if cardioid_profile is not None:
        profile_y, profile_density, half_width = cardioid_profile
        outer = half_width * profile_density
        ax.plot(
            period - outer,
            profile_y,
            color=config.vertical_cardioid_edge_color,
            linewidth=0.85,
            zorder=3.0,
        )
        ax.plot(
            period + outer,
            profile_y,
            color=config.vertical_cardioid_edge_color,
            linewidth=0.85,
            zorder=3.0,
        )

    if disk_profile is not None:
        profile_y, profile_density, half_width = disk_profile
        outer = half_width * profile_density
        disk_edge = blend_color(
            config.vertical_disk_violin_outer_color, "black", 0.30
        )
        ax.plot(
            period - outer,
            profile_y,
            color=disk_edge,
            linewidth=0.85,
            zorder=3.1,
        )
        ax.plot(
            period + outer,
            profile_y,
            color=disk_edge,
            linewidth=0.85,
            zorder=3.1,
        )


def violin_requested(
    period: int,
    point_count: int,
    *,
    period_start: int,
    min_points: int,
) -> bool:
    """Period-start is the explicit override; point count is the fallback."""
    return period >= period_start or point_count >= min_points


@dataclass(frozen=True)
class ViolinCountLabel:
    period: int
    family: str
    hidden_count: int
    anchor_y: float
    box_facecolor: Any
    box_edgecolor: Any


def violin_hidden_count(outlier_mask: np.ndarray) -> int:
    mask = np.asarray(outlier_mask, dtype=bool)
    return int(mask.size - np.count_nonzero(mask))


def violin_bottom_tip_y(y_values: np.ndarray) -> float:
    values = np.asarray(y_values, dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.min(valid)) if valid.size else math.nan


def violin_top_tip_y(y_values: np.ndarray) -> float:
    values = np.asarray(y_values, dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.max(valid)) if valid.size else math.nan


def place_violin_count_labels(
    fig: Any,
    ax: Any,
    labels: list[ViolinCountLabel],
    config: PlotConfig,
) -> None:
    """Place cardioid counts at lower tips and disk counts above violins."""
    if not config.vertical_violin_count_labels or not labels:
        return

    formatted = [f"{label.hidden_count:,}" for label in labels]
    maximum_characters = max(len(value) for value in formatted)
    if config.vertical_violin_count_box_mode == "uniform":
        display_text = [value.center(maximum_characters) for value in formatted]
        font_family = "DejaVu Sans Mono"
    else:
        display_text = formatted
        font_family = None

    # This knob now controls the breathing room around the natural placements:
    # cardioid labels sit left/below the lower tip; disk labels sit directly
    # above the upper tip at their integer period.
    gap = config.vertical_violin_count_min_gap_points
    cardioid_dx = -(10.0 + gap)
    cardioid_dy = -(5.0 + 0.5 * gap)
    disk_dy = 5.0 + gap

    ordered = sorted(
        zip(labels, display_text),
        key=lambda item: (item[0].period, item[0].family != "cardioid"),
    )
    for label, text in ordered:
        if not math.isfinite(label.anchor_y) or label.anchor_y <= 0:
            continue

        common = {
            "fontsize": config.vertical_violin_count_font_size,
            "fontfamily": font_family,
            "color": "black",
            "zorder": 12,
            "clip_on": False,
            "bbox": {
                "boxstyle": (
                    "round,pad="
                    f"{config.vertical_violin_count_box_padding:.6g},"
                    "rounding_size=0.08"
                ),
                "facecolor": label.box_facecolor,
                "edgecolor": label.box_edgecolor,
                "linewidth": 0.9,
                "alpha": 0.92,
            },
        }

        if label.family == "cardioid":
            # The annotation target is the bottom tip. With the text anchored by
            # its top-right corner to the left and below, Matplotlib connects
            # that corner naturally to the tip without crossing the violin.
            ax.annotate(
                text,
                xy=(label.period, label.anchor_y),
                xytext=(cardioid_dx, cardioid_dy),
                textcoords="offset points",
                ha="right",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": label.box_edgecolor,
                    "linewidth": 0.8,
                    "shrinkA": 0.0,
                    "shrinkB": 1.5,
                    "connectionstyle": "arc3,rad=0.0",
                },
                **common,
            )
        else:
            # Disk labels are centered at integer n just above the violin and
            # deliberately have no connector line.
            ax.annotate(
                text,
                xy=(label.period, label.anchor_y),
                xytext=(0.0, disk_dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                **common,
            )


def plot_vertical_stacks(
    rows: list[dict[str, str]],
    periods: list[int],
    rho: float,
    config: PlotConfig,
    shape_classes: dict[tuple[int, int], str],
) -> list[Path]:
    plt = get_pyplot()
    positions = dispersed_x_positions(rows, periods, config)

    grouped: dict[int, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    cardioid_violin_periods: list[int] = []
    disk_violin_periods: list[int] = []
    for period in periods:
        disk_rows: list[dict[str, str]] = []
        cardioid_rows: list[dict[str, str]] = []
        for row in rows:
            if parse_int(row.get("period")) != period:
                continue
            key = measurement_catalogue_key(row)
            if shape_classes.get(key, "unknown") == "disk":
                disk_rows.append(row)
            else:
                # Unknowns stay with the cardioid family so the plot remains
                # complete while classification work continues.
                cardioid_rows.append(row)
        grouped[period] = (cardioid_rows, disk_rows)
        if cardioid_rows and violin_requested(
            period,
            len(cardioid_rows),
            period_start=config.vertical_violin_cardioids_period_start,
            min_points=config.vertical_violin_cardioids_min_points,
        ):
            cardioid_violin_periods.append(period)
        if disk_rows and violin_requested(
            period,
            len(disk_rows),
            period_start=config.vertical_violin_disks_period_start,
            min_points=config.vertical_violin_disks_min_points,
        ):
            disk_violin_periods.append(period)

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    cardioid_marker = cardioid_marker_path()
    cardioid_drawn = False
    disk_drawn = False
    cardioid_violin_drawn = False
    disk_violin_drawn = False
    violin_count_labels: list[ViolinCountLabel] = []

    disk_edge = blend_color(
        config.vertical_disk_violin_outer_color, "black", 0.30
    )

    for period in periods:
        cardioid_rows, disk_rows = grouped[period]
        marker_scale = (
            config.vertical_large_marker_scale
            if period <= config.vertical_large_marker_max_period
            else config.vertical_regular_marker_scale
        )

        visible_cardioids = cardioid_rows
        visible_disks = disk_rows
        cardioid_profile: tuple[np.ndarray, np.ndarray, float] | None = None
        disk_profile: tuple[np.ndarray, np.ndarray, float] | None = None

        if period in cardioid_violin_periods:
            cardioid_areas = np.asarray(
                [parse_float(row.get("area_estimate")) for row in cardioid_rows],
                dtype=float,
            )
            profile = log_violin_profile(cardioid_areas, config)
            if profile is not None:
                y_values, density, outlier_mask = profile
                cardioid_profile = (
                    y_values,
                    density,
                    violin_half_width(period, cardioid_violin_periods, config),
                )
                cardioid_violin_drawn = True
                hidden_count = violin_hidden_count(outlier_mask)
                anchor_y = violin_bottom_tip_y(y_values)
                if hidden_count > 0 and math.isfinite(anchor_y):
                    violin_count_labels.append(
                        ViolinCountLabel(
                            period=period,
                            family="cardioid",
                            hidden_count=hidden_count,
                            anchor_y=anchor_y,
                            box_facecolor=blend_color(
                                config.vertical_cardioid_color, "white", 0.72
                            ),
                            box_edgecolor=config.vertical_cardioid_edge_color,
                        )
                    )
                visible_cardioids = [
                    row
                    for row, is_outlier in zip(cardioid_rows, outlier_mask)
                    if bool(is_outlier)
                ]
                marker_scale = config.vertical_outlier_marker_scale

        disk_marker_scale = marker_scale
        if period in disk_violin_periods:
            disk_areas = np.asarray(
                [parse_float(row.get("area_estimate")) for row in disk_rows],
                dtype=float,
            )
            profile = log_violin_profile(disk_areas, config)
            if profile is not None:
                y_values, density, outlier_mask = profile
                disk_profile = (
                    y_values,
                    density,
                    violin_half_width(period, disk_violin_periods, config),
                )
                disk_violin_drawn = True
                hidden_count = violin_hidden_count(outlier_mask)
                anchor_y = violin_top_tip_y(y_values)
                if hidden_count > 0 and math.isfinite(anchor_y):
                    violin_count_labels.append(
                        ViolinCountLabel(
                            period=period,
                            family="disk",
                            hidden_count=hidden_count,
                            anchor_y=anchor_y,
                            box_facecolor=blend_color(
                                config.vertical_disk_color, "white", 0.78
                            ),
                            box_edgecolor=disk_edge,
                        )
                    )
                visible_disks = [
                    row
                    for row, is_outlier in zip(disk_rows, outlier_mask)
                    if bool(is_outlier)
                ]
                disk_marker_scale = config.vertical_outlier_marker_scale

        plot_solid_blended_violins(
            ax,
            period=period,
            cardioid_profile=cardioid_profile,
            disk_profile=disk_profile,
            config=config,
        )

        if visible_cardioids:
            ax.scatter(
                [
                    positions[(period, parse_int(row.get("component_index")))]
                    for row in visible_cardioids
                ],
                [parse_float(row.get("area_estimate")) for row in visible_cardioids],
                s=1.15 * config.marker_size * marker_scale,
                marker=cardioid_marker,
                alpha=1.0,
                facecolors=config.vertical_cardioid_color,
                edgecolors=config.vertical_cardioid_edge_color,
                linewidths=0.65,
                zorder=5,
            )
            cardioid_drawn = True

        if visible_disks:
            ax.scatter(
                [
                    positions[(period, parse_int(row.get("component_index")))]
                    for row in visible_disks
                ],
                [parse_float(row.get("area_estimate")) for row in visible_disks],
                s=0.78 * config.marker_size * disk_marker_scale,
                marker="o",
                alpha=0.96,
                facecolors=config.vertical_disk_color,
                edgecolors="none",
                zorder=6,
            )
            disk_drawn = True

    ax.set_yscale("log")
    ax.set_xlabel("Exact period n")
    ax.set_ylabel("Component area")
    ax.set_title("Mandelbrot hyperbolic components")
    decorate_period_axis(ax, periods)

    if cardioid_drawn or cardioid_violin_drawn or disk_drawn or disk_violin_drawn:
        from matplotlib.lines import Line2D

        handles: list[Any] = []
        if cardioid_drawn or cardioid_violin_drawn:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=cardioid_marker,
                    markersize=8.5,
                    markerfacecolor=config.vertical_cardioid_color,
                    markeredgecolor=config.vertical_cardioid_edge_color,
                    label="Cardioid",
                )
            )
        if disk_drawn or disk_violin_drawn:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=6.5,
                    markerfacecolor=config.vertical_disk_color,
                    markeredgecolor="none",
                    label="Disk",
                )
            )
        ax.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    place_violin_count_labels(fig, ax, violin_count_labels, config)
    paths = save_figure(
        fig,
        config.plot_output_dir
        / f"component_area_vertical_stacks_rho_{rho_tag(rho)}",
        config,
    )
    if config.show:
        plt.show()
    plt.close(fig)
    return paths


def plot_period_series(
    stats: list[PeriodStats],
    rho: float,
    config: PlotConfig,
    *,
    attribute: str,
    filename: str,
    title: str,
    ylabel: str,
    logarithmic: bool,
) -> list[Path]:
    plt = get_pyplot()
    periods = [item.period for item in stats]
    values = [float(getattr(item, attribute)) for item in stats]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.plot(
        periods,
        values,
        linewidth=1.25,
        alpha=0.72,
        color=config.line_color,
        zorder=1,
    )
    ax.scatter(
        periods,
        values,
        s=42,
        color=config.line_color,
        edgecolors="none",
        zorder=2,
    )
    if logarithmic:
        ax.set_yscale("log")
    ax.set_xlabel("Exact period n")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nMultiplier radius ρ = {rho:.8g}")
    decorate_period_axis(ax, periods)
    fig.tight_layout()
    paths = save_figure(
        fig,
        config.plot_output_dir / f"{filename}_rho_{rho_tag(rho)}",
        config,
    )
    if config.show:
        plt.show()
    plt.close(fig)
    return paths



def plot_area_range(
    stats: list[PeriodStats],
    rho: float,
    config: PlotConfig,
) -> list[Path]:
    plt = get_pyplot()
    periods = [item.period for item in stats]

    series = [
        (
            "Maximum",
            [item.max_area for item in stats],
            config.range_max_color,
            "o",
        ),
        (
            "Geometric mean",
            [item.geometric_mean_area for item in stats],
            config.range_geometric_color,
            "s",
        ),
        (
            "Minimum",
            [item.min_area for item in stats],
            config.range_min_color,
            "^",
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    for label, values, color, marker in series:
        ax.plot(
            periods,
            values,
            linewidth=1.35,
            marker=marker,
            markersize=5.5,
            color=color,
            label=label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exact period n")
    ax.set_ylabel("Component area")
    ax.set_title(
        "Component-area range at each exact period\\n"
        f"Multiplier radius ρ = {rho:.8g}"
    )
    decorate_period_axis(ax, periods)
    ax.legend()
    fig.tight_layout()

    paths = save_figure(
        fig,
        config.plot_output_dir
        / f"component_area_range_rho_{rho_tag(rho)}",
        config,
    )
    if config.show:
        plt.show()
    plt.close(fig)
    return paths


def write_numerical_audit(
    selected_rows: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    periods: list[int],
    rho: float,
    config: PlotConfig,
) -> tuple[Path, int]:
    fields = [
        "period",
        "rho",
        "components",
        "total_area",
        "summed_error_estimate",
        "relative_summed_error",
        "max_relative_error",
        "max_theta_points",
        "radial_monotonicity_violations",
    ]

    violations_by_period: dict[int, int] = defaultdict(int)
    grouped: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    audit_progress = InlineProgress("building numerical audit", len(all_rows))
    for index, row in enumerate(all_rows, start=1):
        period = parse_int(row.get("period"))
        component = parse_int(row.get("component_index"))
        value_rho = parse_float(row.get("rho"))
        area = parse_float(row.get("area_estimate"))
        error = abs(parse_float(row.get("error_estimate"), 0.0))
        if (
            period in periods
            and component >= 0
            and usable_row(row)
            and math.isfinite(value_rho)
            and math.isfinite(area)
        ):
            grouped[(period, component)].append((value_rho, area, error))
        if index % 4096 == 0 or index == len(all_rows):
            audit_progress.update(
                index,
                f"component series={len(grouped):,}",
                final=index == len(all_rows),
            )

    for (period, _component), values in grouped.items():
        values.sort()
        for (_rho_a, area_a, error_a), (_rho_b, area_b, error_b) in zip(
            values, values[1:]
        ):
            tolerance = 4.0 * (error_a + error_b)
            if area_b + tolerance < area_a:
                violations_by_period[period] += 1

    rows_out: list[dict[str, Any]] = []
    for period in periods:
        rows = [
            row for row in selected_rows
            if parse_int(row.get("period")) == period
        ]
        areas = np.asarray(
            [parse_float(row.get("area_estimate")) for row in rows],
            dtype=float,
        )
        errors = np.asarray(
            [
                max(0.0, parse_float(row.get("error_estimate"), 0.0))
                for row in rows
            ],
            dtype=float,
        )
        relative = np.divide(
            errors,
            areas,
            out=np.zeros_like(errors),
            where=areas > 0,
        )
        total = float(np.sum(areas))
        summed_error = float(np.sum(errors))
        rows_out.append(
            {
                "period": period,
                "rho": f"{rho:.17g}",
                "components": len(rows),
                "total_area": f"{total:.17g}",
                "summed_error_estimate": f"{summed_error:.17g}",
                "relative_summed_error": (
                    f"{summed_error / total:.17g}" if total > 0 else "nan"
                ),
                "max_relative_error": (
                    f"{float(np.max(relative)):.17g}"
                    if relative.size else "nan"
                ),
                "max_theta_points": max(
                    (parse_int(row.get("theta_points"), 0) for row in rows),
                    default=0,
                ),
                "radial_monotonicity_violations":
                    violations_by_period.get(period, 0),
            }
        )

    path = (
        config.plot_output_dir
        / f"component_area_numerical_audit_rho_{rho_tag(rho)}.csv"
    )
    atomic_write_csv(path, rows_out, fields)
    return path, sum(violations_by_period.values())


def write_statistics(
    stats: list[PeriodStats],
    rho: float,
    config: PlotConfig,
) -> Path:
    fields = [
        "period",
        "rho",
        "components",
        "total_area",
        "cumulative_area",
        "min_area",
        "max_area",
        "arithmetic_mean_area",
        "geometric_mean_area",
        "median_area",
        "p10_area",
        "p90_area",
    ]
    rows = [
        {
            "period": item.period,
            "rho": f"{rho:.17g}",
            "components": item.components,
            "total_area": f"{item.total_area:.17g}",
            "cumulative_area": f"{item.cumulative_area:.17g}",
            "min_area": f"{item.min_area:.17g}",
            "max_area": f"{item.max_area:.17g}",
            "arithmetic_mean_area": f"{item.arithmetic_mean_area:.17g}",
            "geometric_mean_area": f"{item.geometric_mean_area:.17g}",
            "median_area": f"{item.median_area:.17g}",
            "p10_area": f"{item.p10_area:.17g}",
            "p90_area": f"{item.p90_area:.17g}",
        }
        for item in stats
    ]
    path = (
        config.plot_output_dir
        / f"component_area_statistics_rho_{rho_tag(rho)}.csv"
    )
    atomic_write_csv(path, rows, fields)
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot typed component-area records produced by component_area_scan."
        )
    )
    add_config_argument(parser)
    parser.add_argument(
        "--rho",
        help="Multiplier radius to plot, or omit for config/auto selection.",
    )
    parser.add_argument("--min-period", type=int)
    parser.add_argument("--max-period", type=int)
    parser.add_argument(
        "--colormap",
        help="Matplotlib colormap override, e.g. viridis, turbo, rainbow.",
    )
    parser.add_argument(
        "--line-color",
        help="Color override for single-series plots, e.g. tab:blue or #336699.",
    )
    parser.add_argument(
        "--formats",
        help="Comma-separated output formats, e.g. png,svg.",
    )
    parser.add_argument("--dpi", type=int)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_plot_config(args)

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    catalogue = Catalogue(repo.path("paths.catalogue_root"))
    run_name = repo.string("component_area_scan.run_name", "default")
    store = catalogue.area_scan_store(run_name)

    print("Mandelbrot component-area postprocessor")
    print(f"  config:      {config.config_path}")
    print(f"  catalogue:   {catalogue.root}")
    print(f"  scan run:    {run_name}")
    print(f"  plots:       {config.plot_output_dir}")
    print("  loading area-scan records (read-only)...", flush=True)

    center_rows = load_csv_rows(
        store.centers_path,
        "loading scanner centers",
        keep_fields=(
            "period",
            "component_index",
            "expected_period_count",
            "center_re",
            "center_im",
            "conjugate_index",
        ),
    )
    raw_measurement_rows = load_csv_rows(
        store.measurements_path,
        "loading area measurements",
        keep_fields=(
            "period",
            "component_index",
            "conjugate_index",
            "center_im",
            "rho",
            "theta_points",
            "area_estimate",
            "error_estimate",
            "converged",
        ),
    )

    print("  selecting complete radius and computing statistics...", flush=True)
    measurement_rows, duplicate_rows_removed = deduplicate_measurements(
        raw_measurement_rows,
        config.configured_radii,
    )
    expected_counts = expected_counts_from_centers(center_rows)
    periods = requested_periods(expected_counts, config)
    rho, selected_rows = choose_rho(
        measurement_rows,
        periods,
        expected_counts,
        config.rho_request,
        config.configured_radii,
    )
    stats = compute_statistics(selected_rows, periods)

    print("  loading stored catalogue classifications (read-only)...", flush=True)
    shape_classes = catalogue_shape_classes(
        catalogue,
        periods,
        config.config_path,
    )
    classified_disks = sum(
        shape_classes.get(measurement_catalogue_key(row), "unknown") == "disk"
        for row in selected_rows
    )
    plotted_cardioids = len(selected_rows) - classified_disks

    print(f"  periods:     {periods[0]}..{periods[-1]}")
    print(f"  rho:         {rho:.12g}")
    print(f"  components:  {len(selected_rows):,}")
    print(
        "  stack cardioids: violin at "
        f"p{config.vertical_violin_cardioids_period_start}+ or "
        f"{config.vertical_violin_cardioids_min_points}+ points"
    )
    print(
        "  stack disks:     violin at "
        f"p{config.vertical_violin_disks_period_start}+ or "
        f"{config.vertical_violin_disks_min_points}+ points"
    )
    print(
        "  stack violin:    "
        f"q{100 * config.vertical_violin_quantile_low:.0f}–"
        f"q{100 * config.vertical_violin_quantile_high:.0f}; "
        "outside values remain explicit markers"
    )
    print(
        "  stack colors:    opaque local gradients; overlaps use explicit "
        "50/50 RGB mixing"
    )
    print(f"  line color:  {config.line_color}")
    print(f"  duplicate rows canonicalized: {duplicate_rows_removed:,}")
    print(f"  disks:       {classified_disks:,}")
    print(f"  cardioids:   {plotted_cardioids:,}")
    print(
        "  symmetry:    full-plane area rows; lower-half rows reuse the "
        "upper representative classification"
    )

    outputs: list[Path] = []
    plot_steps = 1 + 3 + 1 + 1 + 1
    plot_progress = InlineProgress("rendering postprocess outputs", plot_steps)
    completed_plot_steps = 0
    outputs.extend(
        plot_vertical_stacks(
            selected_rows,
            periods,
            rho,
            config,
            shape_classes,
        )
    )
    completed_plot_steps += 1
    plot_progress.update(completed_plot_steps, "component stack")

    plot_specs = [
        (
            "cumulative_area",
            "component_area_cumulative",
            "Cumulative summed hyperbolic-component area through period n",
            "Cumulative area",
            False,
        ),
        (
            "total_area",
            "component_area_period_total",
            "Total component area contributed by exact period n",
            "Exact-period total area",
            True,
        ),
        (
            "arithmetic_mean_area",
            "component_area_arithmetic_mean",
            "Arithmetic mean component area at each exact period",
            "Arithmetic mean area",
            True,
        ),
    ]

    for attribute, filename, title, ylabel, logarithmic in plot_specs:
        outputs.extend(
            plot_period_series(
                stats,
                rho,
                config,
                attribute=attribute,
                filename=filename,
                title=title,
                ylabel=ylabel,
                logarithmic=logarithmic,
            )
        )
        completed_plot_steps += 1
        plot_progress.update(completed_plot_steps, filename)

    outputs.extend(plot_area_range(stats, rho, config))
    completed_plot_steps += 1
    plot_progress.update(completed_plot_steps, "area range")

    statistics_path = write_statistics(stats, rho, config)
    completed_plot_steps += 1
    plot_progress.update(completed_plot_steps, "statistics CSV")
    audit_path, monotonicity_violations = write_numerical_audit(
        selected_rows,
        measurement_rows,
        periods,
        rho,
        config,
    )
    completed_plot_steps += 1
    plot_progress.update(
        completed_plot_steps,
        "numerical audit",
        final=True,
    )

    print(
        f"  radial monotonicity violations: {monotonicity_violations:,}"
    )
    print("\nCreated:")
    for path in outputs:
        print(f"  {path}")
    print(f"  {statistics_path}")
    print(f"  {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
