#!/usr/bin/env python3
"""
Measure how Mandelbrot equipotential contour geometry changes as G -> 0+.

For log-spaced target_G values, this script:
  1. traces the contour G(c) = target_G using mandelbrot_tools.trace_mandelbrot_contour,
  2. computes closed-contour length and enclosed area,
  3. writes a CSV row immediately after each contour finishes,
  4. optionally saves the raw contour as JSON,
  5. writes log-scale plots of length(G), area(G), and both together.

Put this file next to your current mandelbrot_tools.py, the one with
trace_mandelbrot_contour(...).

Example:

    python3 mandelbrot_contour_scaling.py \
      --g-start 0.25 \
      --g-stop 1e-6 \
      --n 18 \
      --output-dir work/mandelbrot/G_scaling \
      --save-contours-dir work/mandelbrot/G_contours

For a quick test:

    python3 mandelbrot_contour_scaling.py --g-stop 1e-4 --n 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    # Prefer the file the user usually replaces/copies into the working tree.
    from mandelbrot_tools import trace_mandelbrot_contour
except Exception:
    try:
        from mandelbrot_tools_v8 import trace_mandelbrot_contour
    except Exception:
        try:
            from mandelbrot_tools_v7 import trace_mandelbrot_contour
        except Exception:
            try:
                from mandelbrot_tools_v6 import trace_mandelbrot_contour
            except Exception:
                try:
                    from mandelbrot_tools_v5 import trace_mandelbrot_contour
                except Exception:
                    try:
                        from mandelbrot_tools_v4 import trace_mandelbrot_contour
                    except Exception:
                        try:
                            from mandelbrot_tools_v3 import trace_mandelbrot_contour
                        except Exception:
                            try:
                                from mandelbrot_tools_with_contours import trace_mandelbrot_contour
                            except Exception as exc:
                                raise SystemExit(
                                    "Could not import trace_mandelbrot_contour.\n"
                                    "Put this script next to your newest mandelbrot_tools.py, "
                                    "or next to mandelbrot_tools_v8.py / v7.py / v6.py / v5.py / v4.py / v3.py."
                                ) from exc

import inspect


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("MANDELBROT_DATA_ROOT", PROJECT_ROOT / "work" / "mandelbrot")
).expanduser()
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT


CSV_COLUMNS = [
    "target_G",
    "points",
    "ds",
    "max_turn_angle_rad",
    "length",
    "area_signed",
    "area",
    "seconds",
]


def format_g_level_for_filename(level: float) -> str:
    level = float(level)
    if level == 0.0:
        return "0"

    abs_level = abs(level)
    if 1e-6 <= abs_level < 1e6:
        s = f"{level:.12f}".rstrip("0").rstrip(".")
    else:
        s = f"{level:.12g}"

    # Keep filenames shell-friendly.
    return s.replace("+", "").replace("-", "m")


def drop_repeated_closure(points: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        return pts[:-1].copy()
    return pts.copy()


def close_if_needed(points: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        return np.vstack([pts, pts[0]])
    return pts


def signed_polygon_area(points: np.ndarray) -> float:
    pts = drop_repeated_closure(points)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def closed_polyline_length(points: np.ndarray) -> float:
    pts = close_if_needed(drop_repeated_closure(points))
    if len(pts) < 2:
        return 0.0

    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))


def contour_path_to_xy(path: Iterable[tuple[float, float, float]]) -> np.ndarray:
    return np.asarray([(x, y) for x, y, _g in path], dtype=float)


def choose_ds(
    G: float,
    g_start: float,
    ds_at_start: float,
    ds_power: float,
    ds_min: float,
    ds_max: float | None,
) -> float:
    """
    Choose a contour step size for this G.

    Default behavior is roughly calibrated so:
        G=0.25  -> ds≈0.02
        G=1e-4  -> ds≈0.0019
        G=1e-6  -> ds≈0.0005

    The curvature limiter may still reduce this locally.
    """
    if G <= 0.0 or g_start <= 0.0:
        raise ValueError("G and g_start must be positive")

    ds = float(ds_at_start) * (float(G) / float(g_start)) ** float(ds_power)
    ds = max(float(ds_min), ds)

    if ds_max is not None:
        ds = min(float(ds_max), ds)

    return ds


def load_existing_rows(csv_path: Path) -> dict[float, dict[str, str]]:
    if not csv_path.exists():
        return {}

    out: dict[float, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[float(row["target_G"])] = row
            except Exception:
                continue
    return out


def contour_json_path_for_G(save_dir: Path | None, G: float) -> Path | None:
    if save_dir is None:
        return None
    return save_dir / f"{format_g_level_for_filename(G)}.json"


def should_drop_level(G: float, args: argparse.Namespace) -> bool:
    for val in getattr(args, "drop_g_at", []) or []:
        if abs(G - float(val)) <= 5e-12 * max(abs(G), abs(float(val)), 1.0):
            return True

    if args.drop_g_le is not None and G <= float(args.drop_g_le) * (1.0 + 1e-12):
        return True

    if args.drop_g_ge is not None and G >= float(args.drop_g_ge) * (1.0 - 1e-12):
        return True

    return False


def nearest_previous_higher_G_row(
    rows: list[dict[str, float | int]],
    G: float,
) -> dict[str, float | int] | None:
    higher = [row for row in rows if float(row["target_G"]) > G]
    if not higher:
        return None
    return min(higher, key=lambda row: abs(math.log(float(row["target_G"]) / G)))


def filtered_trace_call(**kwargs):
    """
    Call trace_mandelbrot_contour while tolerating older mandelbrot_tools versions.

    New robust options are passed when the installed function supports them.
    Older local copies silently ignore unsupported keyword arguments, although
    for best reliability you should use mandelbrot_tools_v7.py or newer.
    
    This function deliberately fails fast on code/integration errors such as
    missing helper functions. Those are not numerical contour failures, so
    retrying with smaller ds only wastes hours.
    """
    if "mirror_contour_y" not in getattr(trace_mandelbrot_contour, "__globals__", {}):
        raise NameError(
            "The imported mandelbrot_tools.trace_mandelbrot_contour is missing "
            "mirror_contour_y. Replace mandelbrot_tools.py with mandelbrot_tools_v7.py."
        )

    sig = inspect.signature(trace_mandelbrot_contour)
    supported = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return trace_mandelbrot_contour(**filtered)


def write_rows(csv_path: Path, rows: list[dict[str, float | int]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: float(r["target_G"]), reverse=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


def save_contour_json(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = drop_repeated_closure(points)
    data = [[float(x), float(y)] for x, y in pts]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def read_metric_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(row[k]) for k in CSV_COLUMNS})
    return rows


def make_plots(csv_path: Path, output_dir: Path, invert_x: bool = True) -> None:
    rows = read_metric_rows(csv_path)
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r["target_G"], reverse=True)
    G = np.asarray([r["target_G"] for r in rows], dtype=float)
    length = np.asarray([r["length"] for r in rows], dtype=float)
    area = np.asarray([r["area"] for r in rows], dtype=float)
    points = np.asarray([r["points"] for r in rows], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)

    def finish_axis(ax, ylabel: str, title: str) -> None:
        ax.set_xscale("log")
        ax.set_yscale("log")
        if invert_x:
            ax.invert_xaxis()
        ax.set_xlabel("target escape potential G")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(G, length, marker="o")
    finish_axis(ax, "closed contour length", "Mandelbrot equipotential length vs G")
    fig.tight_layout()
    fig.savefig(output_dir / "length_vs_G_loglog.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(G, area, marker="o")
    finish_axis(ax, "enclosed area", "Mandelbrot equipotential area vs G")
    fig.tight_layout()
    fig.savefig(output_dir / "area_vs_G_loglog.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.3), dpi=160)
    line1, = ax1.plot(G, length, marker="o", label="length")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    if invert_x:
        ax1.invert_xaxis()
    ax1.set_xlabel("target escape potential G")
    ax1.set_ylabel("closed contour length")
    ax1.grid(True, which="both", alpha=0.35)

    ax2 = ax1.twinx()
    line2, = ax2.plot(G, area, marker="s", linestyle="--", label="area")
    ax2.set_yscale("log")
    ax2.set_ylabel("enclosed area")

    ax1.legend([line1, line2], ["length", "area"], loc="best")
    ax1.set_title("Mandelbrot equipotential length and area vs G")
    fig.tight_layout()
    fig.savefig(output_dir / "length_area_vs_G_loglog.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(G, points, marker="o")
    finish_axis(ax, "traced contour points", "Traced point count vs G")
    fig.tight_layout()
    fig.savefig(output_dir / "points_vs_G_loglog.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trace log-spaced Mandelbrot G-contours and plot length/area scaling."
    )

    p.add_argument("--g-start", type=float, default=0.25, help="Largest G value.")
    p.add_argument("--g-stop", type=float, default=1e-6, help="Smallest G value.")
    p.add_argument("--n", type=int, default=18, help="Number of log-spaced G values.")

    p.add_argument("--output-dir", type=Path, default=DATA_ROOT / "G_scaling")
    p.add_argument("--csv", type=Path, default=None, help="CSV output path. Default: output-dir/contour_scaling.csv")
    p.add_argument("--save-contours-dir", type=Path, default=None, help="Optional directory for raw contour JSON files.")
    p.add_argument("--resume", action="store_true", help="Skip G values already present in the CSV.")

    p.add_argument("--ds-at-start", type=float, default=0.02, help="Base ds used at g-start.")
    p.add_argument("--ds-power", type=float, default=0.30, help="Scale ds as ds_at_start*(G/g_start)^ds_power.")
    p.add_argument("--ds-min", type=float, default=1e-6, help="Floor for the initial per-contour ds chosen from G.")
    p.add_argument("--trace-ds-min", type=float, default=1e-10, help="Internal minimum step allowed inside the adaptive contour tracer.")
    p.add_argument("--ds-max", type=float, default=None)

    p.add_argument("--max-turn-angle", type=float, default=0.02, help="Curvature limiter in radians.")
    p.add_argument("--derivative-epsilon", type=float, default=1e-12)
    p.add_argument("--derivative-max-iter", type=int, default=20000)
    p.add_argument("--escape-epsilon", type=float, default=1e-12)
    p.add_argument("--escape-max-iter", type=int, default=20000)
    p.add_argument("--root-epsilon", type=float, default=1e-12)
    p.add_argument("--tol-G", type=float, default=1e-12)
    p.add_argument("--max-steps", type=int, default=20_000_000, help="Huge practical guard, not a contour simplification limit.")
    p.add_argument("--max-step-halvings", type=int, default=32)
    p.add_argument("--contour-progress", action="store_true", help="Show progress while tracing each contour.")
    p.add_argument("--contour-progress-every", type=int, default=500, help="Accepted steps between progress updates.")
    p.add_argument("--contour-progress-width", type=int, default=32, help="Width of the contour progress bar.")
    p.add_argument("--no-symmetry", action="store_true", help="Disable y-symmetry contour construction.")
    p.add_argument("--no-project", action="store_true", help="Disable Newton projection after RK contour steps.")
    p.add_argument("--no-invert-x", action="store_true", help="Do not invert plot x-axis.")

    # Resume / repair controls.
    p.add_argument("--drop-g-at", type=float, action="append", default=[], help="Drop/recompute one specific G row from an existing CSV. Can be repeated.")
    p.add_argument("--drop-g-le", type=float, default=None, help="Drop/recompute existing rows with G <= this value.")
    p.add_argument("--drop-g-ge", type=float, default=None, help="Drop/recompute existing rows with G >= this value.")
    p.add_argument("--delete-dropped-contours", action="store_true", help="Also delete matching saved contour JSON files when rows are dropped.")

    # Metric sanity / retry controls.
    p.add_argument("--retry-count", type=int, default=4, help="Retry suspicious contours this many times with stricter settings.")
    p.add_argument("--retry-ds-factor", type=float, default=0.5, help="Each retry multiplies ds by this factor.")
    p.add_argument("--retry-turn-factor", type=float, default=0.65, help="Each retry multiplies max_turn_angle by this factor.")
    p.add_argument("--min-area-ratio-to-previous", type=float, default=0.70, help="Reject if area collapses below this fraction of the previous higher-G area.")
    p.add_argument("--max-area-ratio-to-previous", type=float, default=1.03, help="Reject if area exceeds previous higher-G area by more than this factor.")
    p.add_argument("--length-monotone-after-G", type=float, default=0.04, help="For G below this value, reject contours whose length collapses relative to the previous higher-G contour. Use <=0 to disable.")
    p.add_argument("--min-length-ratio-to-previous-after-min", type=float, default=0.98, help="Low-G length guard: reject if length < this * previous length. Use a lower value like 0.90 if you expect mild wiggles.")

    # Robust adaptive contour controls forwarded to mandelbrot_tools_v5 when available.
    p.add_argument("--no-adaptive-ds", action="store_true", help="Disable adaptive ds rollback/growth inside the contour tracer.")
    p.add_argument("--ds-growth", type=float, default=1.015)
    p.add_argument("--ds-shrink", type=float, default=0.5)
    p.add_argument("--grow-after-successes", type=int, default=12)
    p.add_argument("--rollback-points", type=int, default=64)
    p.add_argument("--rollback-shrink", type=float, default=0.35)
    p.add_argument("--max-rollbacks", type=int, default=200)
    p.add_argument("--no-step-doubling", action="store_true", help="Disable RK one-step vs two-half-step error check.")
    p.add_argument("--step-error-factor", type=float, default=0.025)
    p.add_argument("--max-actual-turn-angle", type=float, default=None, help="Reject actual polyline turns above this angle in radians. Default derives from max_turn_angle.")
    p.add_argument("--self-intersection-window", type=int, default=4096)
    p.add_argument("--seam-tol-factor", type=float, default=6.0)
    p.add_argument("--seam-tol-min", type=float, default=1e-8)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.g_start <= 0.0 or args.g_stop <= 0.0:
        raise SystemExit("G values must be positive.")

    if args.n < 2:
        raise SystemExit("--n must be at least 2.")

    output_dir: Path = args.output_dir
    csv_path: Path = args.csv if args.csv is not None else output_dir / "contour_scaling.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = np.geomspace(float(args.g_start), float(args.g_stop), int(args.n))

    rows: list[dict[str, float | int]] = []
    existing_raw = load_existing_rows(csv_path) if args.resume else {}

    dropped = 0
    if existing_raw:
        for row in existing_raw.values():
            G_old = float(row["target_G"])
            if should_drop_level(G_old, args):
                dropped += 1
                if args.delete_dropped_contours:
                    p_json = contour_json_path_for_G(args.save_contours_dir, G_old)
                    if p_json is not None and p_json.exists():
                        p_json.unlink()
                        print(f"Deleted dropped contour JSON: {p_json}", flush=True)
                continue
            rows.append({k: float(row[k]) for k in CSV_COLUMNS})

        print(
            f"Loaded {len(rows)} existing rows from {csv_path}"
            + (f"; dropped {dropped} row(s) for recomputation" if dropped else ""),
            flush=True,
        )

    existing = {float(row["target_G"]): row for row in rows}

    for idx, G in enumerate(levels, start=1):
        G = float(G)

        # Resume with fuzzy-ish relative match because geomspace formatting can vary slightly.
        if args.resume and any(abs(G - old_G) <= 1e-14 * max(abs(G), abs(old_G), 1.0) for old_G in existing):
            print(f"[{idx}/{len(levels)}] G={G:g}: already in CSV, skipping", flush=True)
            continue

        ds = choose_ds(
            G=G,
            g_start=float(args.g_start),
            ds_at_start=float(args.ds_at_start),
            ds_power=float(args.ds_power),
            ds_min=float(args.ds_min),
            ds_max=args.ds_max,
        )

        print(
            f"[{idx}/{len(levels)}] Tracing G={G:g}: ds≈{ds:g}, "
            f"max_turn_angle={args.max_turn_angle:g} rad",
            flush=True,
        )

        previous_row = nearest_previous_higher_G_row(rows, G)

        last_reject_reason = None
        accepted_metrics = None

        for attempt in range(int(args.retry_count) + 1):
            attempt_ds = ds * (float(args.retry_ds_factor) ** attempt)
            attempt_turn = float(args.max_turn_angle) * (float(args.retry_turn_factor) ** attempt)

            print(
                f"    attempt {attempt + 1}/{int(args.retry_count) + 1}: "
                f"ds≈{attempt_ds:g}, max_turn_angle={attempt_turn:g} rad",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                path = filtered_trace_call(
                    target_G=G,
                    derivative_epsilon=float(args.derivative_epsilon),
                    derivative_max_iter=int(args.derivative_max_iter),
                    escape_epsilon=float(args.escape_epsilon),
                    escape_max_iter=int(args.escape_max_iter),
                    root_epsilon=float(args.root_epsilon),
                    ds=attempt_ds,
                    max_turn_angle=attempt_turn,
                    tol_G=float(args.tol_G),
                    max_steps=int(args.max_steps),
                    use_y_symmetry=not bool(args.no_symmetry),
                    close_loop=True,
                    project_each_step=not bool(args.no_project),
                    max_step_halvings=int(args.max_step_halvings),
                    adaptive_ds=not bool(args.no_adaptive_ds),
                    ds_min=float(args.trace_ds_min),
                    ds_growth=float(args.ds_growth),
                    ds_shrink=float(args.ds_shrink),
                    grow_after_successes=int(args.grow_after_successes),
                    rollback_points=int(args.rollback_points),
                    rollback_shrink=float(args.rollback_shrink),
                    max_rollbacks=int(args.max_rollbacks),
                    use_step_doubling=not bool(args.no_step_doubling),
                    step_error_factor=float(args.step_error_factor),
                    max_actual_turn_angle=args.max_actual_turn_angle,
                    self_intersection_window=int(args.self_intersection_window),
                    seam_tol_factor=float(args.seam_tol_factor),
                    seam_tol_min=float(args.seam_tol_min),
                    progress=bool(args.contour_progress),
                    progress_every=int(args.contour_progress_every),
                    progress_width=int(args.contour_progress_width),
                )
            except (NameError, AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Non-retryable code/integration error while tracing. "
                    "This is not a numerical contour failure, so retrying with smaller ds "
                    "would only waste time. Check that mandelbrot_tools.py is the newest "
                    "patched version."
                ) from exc
            except Exception as exc:
                last_reject_reason = f"trace failed: {exc}"
                print(f"    rejected attempt: {last_reject_reason}", flush=True)
                continue

            seconds = time.perf_counter() - t0

            xy = contour_path_to_xy(path)
            xy_open = drop_repeated_closure(xy)
            length = closed_polyline_length(xy_open)
            area_signed = signed_polygon_area(xy_open)
            area = abs(area_signed)

            reject_reason = None
            if previous_row is not None:
                prev_area = float(previous_row["area"])
                prev_G = float(previous_row["target_G"])

                if area < float(args.min_area_ratio_to_previous) * prev_area:
                    reject_reason = (
                        f"area collapsed relative to previous higher-G contour: "
                        f"area={area:.10g}, previous area={prev_area:.10g} at G={prev_G:g}"
                    )
                elif area > float(args.max_area_ratio_to_previous) * prev_area:
                    reject_reason = (
                        f"area increased relative to previous higher-G contour: "
                        f"area={area:.10g}, previous area={prev_area:.10g} at G={prev_G:g}"
                    )

            if reject_reason is None and previous_row is not None:
                prev_length = float(previous_row["length"])
                prev_G = float(previous_row["target_G"])
                length_guard_G = float(args.length_monotone_after_G)
                if length_guard_G > 0.0 and G <= length_guard_G and prev_G <= length_guard_G:
                    min_len = float(args.min_length_ratio_to_previous_after_min) * prev_length
                    if length < min_len:
                        reject_reason = (
                            f"low-G length collapsed relative to previous contour: "
                            f"length={length:.10g}, previous length={prev_length:.10g} "
                            f"at G={prev_G:g}; required >= {min_len:.10g}"
                        )

            if reject_reason is None:
                accepted_metrics = (xy_open, length, area_signed, area, seconds, attempt_ds, attempt_turn)
                break

            last_reject_reason = reject_reason
            print(f"    rejected attempt: {reject_reason}", flush=True)

        if accepted_metrics is None:
            raise RuntimeError(
                f"Could not trace a sane contour for G={G:g} after {int(args.retry_count) + 1} attempts. "
                f"Last rejection: {last_reject_reason}"
            )

        xy_open, length, area_signed, area, seconds, used_ds, used_turn = accepted_metrics

        row: dict[str, float | int] = {
            "target_G": G,
            "points": int(len(xy_open)),
            "ds": used_ds,
            "max_turn_angle_rad": used_turn,
            "length": length,
            "area_signed": area_signed,
            "area": area,
            "seconds": seconds,
        }
        rows.append(row)
        write_rows(csv_path, rows)

        print(
            f"    points={len(xy_open):,}, length≈{length:.10g}, "
            f"area≈{area:.10g}, signed≈{area_signed:.10g}, "
            f"time={seconds:.2f}s",
            flush=True,
        )
        print(f"    updated CSV: {csv_path}", flush=True)

        if args.save_contours_dir is not None:
            fname = f"{format_g_level_for_filename(G)}.json"
            contour_path = args.save_contours_dir / fname
            save_contour_json(contour_path, xy_open)
            print(f"    saved contour: {contour_path} ({len(xy_open):,} points)", flush=True)

        make_plots(csv_path, output_dir, invert_x=not bool(args.no_invert_x))
        print(f"    updated plots in: {output_dir}", flush=True)

    make_plots(csv_path, output_dir, invert_x=not bool(args.no_invert_x))
    print("Done.", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Plots: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
