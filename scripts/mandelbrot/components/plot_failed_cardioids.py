#!/usr/bin/env python3
"""Plot rejected slanted-cardioid fits exported by classify_component_shapes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay failed Mandelbrot component polygons with their best "
            "rejected slanted-cardioid fits."
        )
    )
    add_config_argument(parser)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="failed_cardioids.ndjson exported by the classifier",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="plot directory (default: <diagnostic-dir>/plots)",
    )
    parser.add_argument(
        "--component-id",
        default=None,
        help="plot one exact component ID or unique ID prefix",
    )
    parser.add_argument(
        "--sort",
        choices=("score", "rms", "max", "period", "abs-xi"),
        default="score",
        help="ranking used for the contact sheet",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=36,
        help="maximum components in the contact sheet (default: 36)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip this many ranked failures before plotting (default: 0)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="contact-sheet columns (default: 4)",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=768,
        help="points sampled on each fitted cardioid (default: 768)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="also save one image per selected component",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open figures interactively after saving",
    )
    args = parser.parse_args()
    if (
        args.limit < 1
        or args.offset < 0
        or args.columns < 1
        or args.curve_points < 32
    ):
        parser.error(
            "--limit/--columns must be positive, --offset non-negative, "
            "and --curve-points >= 32"
        )
    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config = RepoConfig.load(args.config, start=__file__)
    catalogue_root = config.path("paths.catalogue_root")
    relative_dir = config.string(
        "component_shape_classifier.failed_cardioid_export_dir",
        "shape_debug/failed_cardioids",
    )
    input_path = args.input
    if input_path is None:
        input_path = catalogue_root / "exports" / relative_dir / "failed_cardioids.ndjson"
    elif not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_path.parent / "plots"
    elif not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    return input_path, output_dir


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Diagnostic export not found: {path}\n"
            "Run ./bin/classify_component_shapes first."
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid NDJSON at {path}:{line_number}: {error}") from error
            records.append(record)
    return records


def cardioid_curve(record: dict[str, Any], count: int) -> np.ndarray:
    fit = record["cardioid_fit"]
    center = np.asarray(fit["center_centered"], dtype=float)
    size = float(fit["size"])
    angle = float(fit["angle"])
    xi = float(fit["xi"])
    phi = np.linspace(0.0, 2.0 * math.pi, count, endpoint=True)
    factor = 1.0 - xi * np.sin(phi)
    x = size * (np.cos(phi) - 0.5 * np.cos(2.0 * phi)) * factor
    y = size * (np.sin(phi) - 0.5 * np.sin(2.0 * phi)) * factor
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.column_stack(
        (center[0] + cosine * x - sine * y,
         center[1] + sine * x + cosine * y)
    )


def record_key(record: dict[str, Any], key: str) -> tuple[float, str]:
    fit = record["cardioid_fit"]
    if key == "score":
        value = float(record["score"])
    elif key == "rms":
        value = float(fit["rms"])
    elif key == "max":
        value = float(fit["max_error"])
    elif key == "period":
        value = float(record["period"])
    elif key == "abs-xi":
        value = abs(float(fit["xi"]))
    else:
        raise ValueError(key)
    return value, str(record["component_id"])


def select_component(
    records: Iterable[dict[str, Any]], component_id: str
) -> dict[str, Any]:
    exact = [record for record in records if record["component_id"] == component_id]
    if len(exact) == 1:
        return exact[0]
    matches = [
        record for record in records
        if str(record["component_id"]).startswith(component_id)
    ]
    if not matches:
        raise KeyError(f"No failed component matches {component_id!r}")
    if len(matches) > 1:
        ids = ", ".join(str(item["component_id"]) for item in matches[:8])
        raise KeyError(f"Component prefix is ambiguous: {ids}")
    return matches[0]


def short_id(component_id: str) -> str:
    return component_id if len(component_id) <= 14 else component_id[:12] + "…"


def set_equal_limits(ax: plt.Axes, arrays: Iterable[np.ndarray]) -> None:
    combined = np.vstack([array for array in arrays if array.size])
    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    span = max(x_max - x_min, y_max - y_min, 1e-30)
    padding = 0.08 * span
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * span + padding
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")


def draw_record(
    ax: plt.Axes,
    record: dict[str, Any],
    curve_points: int,
    *,
    detailed: bool,
) -> None:
    polygon = np.asarray(record["polygon_centered"], dtype=float)
    curve = cardioid_curve(record, curve_points)
    fit = record["cardioid_fit"]

    ax.scatter(polygon[:, 0], polygon[:, 1], s=7, label="traced polygon")
    ax.plot(curve[:, 0], curve[:, 1], linewidth=1.2, label="best rejected fit")

    center = np.asarray(fit["center_centered"], dtype=float)
    ax.scatter([center[0]], [center[1]], marker="+", s=44, label="fit center")
    cusp_index = int(fit.get("cusp_index", -1))
    if 0 <= cusp_index < len(polygon):
        ax.scatter(
            [polygon[cusp_index, 0]],
            [polygon[cusp_index, 1]],
            marker="x",
            s=36,
            label="chosen cusp",
        )

    set_equal_limits(ax, (polygon, curve))
    ax.tick_params(labelsize=7)
    title = (
        f"p{record['period']} {short_id(str(record['component_id']))}\n"
        f"RMS={float(fit['rms']):.5g}  max={float(fit['max_error']):.5g}  "
        f"ξ={float(fit['xi']):+.4f}  score={float(record['score']):.4f}"
    )
    ax.set_title(title, fontsize=8)
    if detailed:
        ax.set_xlabel("Re(c − center)")
        ax.set_ylabel("Im(c − center)")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def save_contact_sheet(
    records: list[dict[str, Any]],
    output: Path,
    columns: int,
    curve_points: int,
) -> None:
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 4.1 * rows),
        squeeze=False,
    )
    for index, ax in enumerate(axes.flat):
        if index < len(records):
            draw_record(ax, records[index], curve_points, detailed=False)
        else:
            ax.axis("off")
    figure.suptitle(
        f"Best rejected slanted-cardioid fits ({len(records)} shown)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_individual(
    record: dict[str, Any], output: Path, curve_points: int
) -> None:
    figure, ax = plt.subplots(figsize=(7, 7))
    draw_record(ax, record, curve_points, detailed=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def save_error_scatter(records: list[dict[str, Any]], output: Path) -> None:
    rms = np.asarray([float(item["cardioid_fit"]["rms"]) for item in records])
    maximum = np.asarray(
        [float(item["cardioid_fit"]["max_error"]) for item in records]
    )
    periods = np.asarray([int(item["period"]) for item in records])
    threshold = records[0]["thresholds"]

    figure, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(rms, maximum, c=periods, s=18)
    ax.axvline(float(threshold["rms"]), linestyle="--", linewidth=1)
    ax.axhline(float(threshold["max"]), linestyle="--", linewidth=1)
    ax.set_xlabel("Relative RMS error")
    ax.set_ylabel("Relative maximum error")
    ax.set_title("Rejected slanted-cardioid fits")
    figure.colorbar(points, ax=ax, label="Exact period")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def safe_filename(component_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in component_id)


def main() -> int:
    args = parse_args()
    input_path, output_dir = resolve_paths(args)
    records = load_records(input_path)
    if not records:
        print(f"No rejected cardioid fits in {input_path}")
        return 0

    records.sort(key=lambda item: record_key(item, args.sort))
    if args.component_id:
        selected = [select_component(records, args.component_id)]
    else:
        selected = records[args.offset : args.offset + args.limit]
        if not selected:
            raise IndexError(
                f"--offset {args.offset} is beyond the {len(records)} failures"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Mandelbrot failed-cardioid fit inspector")
    print(f"  input:       {input_path}")
    print(f"  failures:    {len(records)}")
    first_rank = args.offset + 1 if not args.component_id else "component ID"
    last_rank = args.offset + len(selected) if not args.component_id else "component ID"
    print(
        f"  selected:    {len(selected)} (sorted by {args.sort}; "
        f"rank {first_rank}..{last_rank})"
    )
    print(f"  output:      {output_dir}")

    if args.component_id:
        record = selected[0]
        output = output_dir / f"failed_{safe_filename(str(record['component_id']))}.png"
        save_individual(record, output, args.curve_points)
        print(f"  wrote:       {output}")
    else:
        first = args.offset + 1
        last = args.offset + len(selected)
        sheet = output_dir / (
            f"failed_cardioids_contact_sheet_{first:04d}_{last:04d}.png"
        )
        save_contact_sheet(selected, sheet, args.columns, args.curve_points)
        print(f"  wrote:       {sheet}")

    scatter = output_dir / "failed_cardioids_error_scatter.png"
    save_error_scatter(records, scatter)
    print(f"  wrote:       {scatter}")

    if args.individual and not args.component_id:
        for index, record in enumerate(selected, start=1):
            output = output_dir / (
                f"{index:04d}_p{int(record['period']):02d}_"
                f"{safe_filename(str(record['component_id']))}.png"
            )
            save_individual(record, output, args.curve_points)
            print(f"\r  individual: {index}/{len(selected)}", end="", flush=True)
        print()

    if args.show:
        # Recreate only the selected detailed view for interactive inspection.
        for record in selected:
            figure, ax = plt.subplots(figsize=(7, 7))
            draw_record(ax, record, args.curve_points, detailed=True)
            figure.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
