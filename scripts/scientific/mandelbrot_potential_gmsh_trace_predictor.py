#!/usr/bin/env python3
"""
Mandelbrot escape-potential surface using Gmsh.

This version does NOT stitch contour rings by hand.

Workflow:
  1. Compute G(c) on a grid.
  2. Extract two contours: G = gmin and G = gmax.
  3. Feed those contours to Gmsh as an annular planar region with a hole.
  4. Let Gmsh triangulate the 2D region.
  5. Interpolate G at mesh vertices.
  6. Lift the mesh to z = height_mode(G).
  7. Render with Plotly Mesh3d.

Install dependencies:

    pip install numpy scipy matplotlib plotly gmsh

On Linux/WSL, the gmsh wheel may also need OpenGL runtime libraries. If import
fails with libGL errors, install the relevant system package, e.g.

    sudo apt install libglu1-mesa
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import plotly.graph_objects as go


HeightMode = Literal["G", "logG", "invG"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("MANDELBROT_DATA_ROOT", PROJECT_ROOT / "work" / "mandelbrot")
).expanduser()
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT


# ============================================================
# Basic geometry
# ============================================================

def polygon_area(points: np.ndarray) -> float:
    """Signed polygon area. Positive means CCW in the x-y plane."""
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts[::-1].copy() if polygon_area(pts) < 0 else pts.copy()


def ensure_cw(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts[::-1].copy() if polygon_area(pts) > 0 else pts.copy()


def close_if_needed(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        return np.vstack([pts, pts[0]])
    return pts


def polyline_length_closed(points: np.ndarray) -> float:
    pts = close_if_needed(points)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs * diffs).sum(axis=1))))


def polyline_length_open(points: list[np.ndarray] | np.ndarray) -> float:
    """Length of an open polyline without closing the final point to the first."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs * diffs).sum(axis=1))))


def polyline_order_diagnostics(points: np.ndarray) -> dict:
    """
    Diagnose whether array order still resembles geometric contour order.

    If adjacent point distances are huge compared with each point's nearest
    spatial neighbor, the saved object is likely a point cloud / branch-jumped
    contour rather than an ordered polyline.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return {"n": int(len(pts)), "ok": True}

    edges = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    try:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=min(4, len(pts)))
        # d[:,0] is self. Use first positive-ish neighbor when possible.
        nn = []
        for row in d:
            val = None
            for x in row[1:]:
                if x > 1e-15:
                    val = float(x)
                    break
            if val is None:
                val = float(row[-1])
            nn.append(val)
        nn = np.asarray(nn, dtype=float)
        ratios = edges / np.maximum(nn[:-1], 1e-300)
    except Exception:
        nn = np.full(len(pts), np.nan)
        ratios = np.full(len(edges), np.nan)

    finite_ratios = ratios[np.isfinite(ratios)]
    finite_edges = edges[np.isfinite(edges)]

    if len(finite_ratios) == 0 or len(finite_edges) == 0:
        return {"n": int(len(pts)), "ok": True}

    out = {
        "n": int(len(pts)),
        "edge_median": float(np.median(finite_edges)),
        "edge_p99": float(np.quantile(finite_edges, 0.99)),
        "edge_max": float(np.max(finite_edges)),
        "ratio_median": float(np.median(finite_ratios)),
        "ratio_p95": float(np.quantile(finite_ratios, 0.95)),
        "ratio_p99": float(np.quantile(finite_ratios, 0.99)),
        "ratio_max": float(np.max(finite_ratios)),
    }

    # Conservative warning thresholds. A valid contour can have close
    # non-adjacent branches, so don't fail purely on this, but these values are
    # excellent smoke alarms.
    out["ok"] = bool(out["ratio_median"] < 100.0 and out["ratio_p99"] < 5000.0)
    return out




def format_g_level_for_filename(level: float) -> str:
    """
    Human-readable filename stem for a natural-G contour level.

    Examples:
      0.25   -> "0.25"
      0.001  -> "0.001"
      1e-04  -> "0.0001"
    """
    level = float(level)
    if level == 0:
        return "0"

    abs_level = abs(level)
    if 1e-6 <= abs_level < 1e6:
        s = f"{level:.12f}".rstrip("0").rstrip(".")
    else:
        s = f"{level:.12g}"

    return s.replace("+", "")


def save_contour_json(
    level: float,
    points: np.ndarray,
    out_dir: str | os.PathLike | None,
    *,
    verbose: bool = True,
) -> str | None:
    """
    Save a contour as a JSON array of [x, y] pairs.

    The filename uses the natural G value, e.g.
      $MANDELBROT_DATA_ROOT/G_contours/0.25.json
      $MANDELBROT_DATA_ROOT/G_contours/0.001.json
    """
    if not out_dir:
        return None

    pts = np.asarray(points, dtype=float)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fname = f"{format_g_level_for_filename(level)}.json"
    path = out_path / fname

    data = [[float(x), float(y)] for x, y in pts]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    if verbose:
        print(f"    saved G={float(level):g} contour: {path} ({len(pts):,} points)", flush=True)

    return str(path)


@dataclass
class SimplifyStats:
    before: int = 0
    after: int = 0
    removed: int = 0
    passes: int = 0


def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Perpendicular distance from p to the infinite line through a--b.
    Falls back to distance to a for degenerate lines.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    lab = float(np.linalg.norm(ab))
    if lab <= 0:
        return float(np.linalg.norm(p - a))
    # NumPy 2.0 deprecates np.cross() on 2D vectors. Use the scalar
    # z-component of the 2D cross product directly.
    ap = p - a
    cross_z = ab[0] * ap[1] - ab[1] * ap[0]
    return abs(float(cross_z)) / lab


def _three_point_turn_degrees(a: np.ndarray, p: np.ndarray, b: np.ndarray) -> float:
    """
    Local polyline direction-change angle at p.

    For ordinary path vectors
        a -> p
        p -> b
    a straight continuation has angle 0 degrees. The old implementation used
    abs(180-angle), which is correct for two vectors both pointing away from an
    inserted midpoint, but wrong for a normal polyline triple.
    """
    v0 = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    v1 = np.asarray(b, dtype=float) - np.asarray(p, dtype=float)
    return _angle_between_degrees(v0, v1)


def simplify_open_polyline_by_sagitta_angle(
    points: np.ndarray,
    *,
    max_sagitta_factor: float,
    max_turn_degrees: float,
    max_merged_segment_length: float,
    min_keep_points: int,
    max_passes: int,
) -> tuple[np.ndarray, SimplifyStats]:
    """
    Simplify an open polyline by removing locally redundant interior points.

    A point p_i is removed only when replacing
        p_{i-1} -> p_i -> p_{i+1}
    by the chord
        p_{i-1} -> p_{i+1}
    is geometrically harmless:

      * p_i has small sagitta relative to the merged chord,
      * the local turn angle is small,
      * the merged chord is not too long.

    The previous defaults were too timid for the traced seed: the original
    contour step is about 0.035, so a merged segment is often around 0.07.
    A 0.03 length cap prevented almost all removals before the test even got
    to sagitta/angle.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) <= max(3, int(min_keep_points)):
        return pts.copy(), SimplifyStats(before=len(pts), after=len(pts), removed=0, passes=0)

    max_sagitta_factor = float(max_sagitta_factor)
    max_turn_degrees = float(max_turn_degrees)
    max_merged_segment_length = float(max_merged_segment_length)
    min_keep_points = max(3, int(min_keep_points))
    max_passes = max(0, int(max_passes))

    current = pts.copy()
    before = len(current)
    total_removed = 0
    passes_done = 0

    for _pass in range(max_passes):
        if len(current) <= min_keep_points:
            break

        keep = np.ones(len(current), dtype=bool)
        removed_this_pass = 0

        # Greedy alternating pass: once a point is removed, skip its immediate
        # neighbor in this pass. This avoids collapsing multiple points around
        # the same curve feature at once.
        i = 1
        while i < len(current) - 1:
            if len(current) - removed_this_pass <= min_keep_points:
                break

            a = current[i - 1]
            p = current[i]
            b = current[i + 1]

            merged_len = float(np.linalg.norm(b - a))
            if merged_len <= 0:
                keep[i] = False
                removed_this_pass += 1
                i += 2
                continue

            if max_merged_segment_length > 0 and merged_len > max_merged_segment_length:
                i += 1
                continue

            sagitta = _point_line_distance(p, a, b)
            sagitta_factor = sagitta / max(merged_len, 1e-300)
            turn_deg = _three_point_turn_degrees(a, p, b)

            if sagitta_factor <= max_sagitta_factor and turn_deg <= max_turn_degrees:
                keep[i] = False
                removed_this_pass += 1
                i += 2
            else:
                i += 1

        if removed_this_pass == 0:
            break

        current = current[keep]
        total_removed += removed_this_pass
        passes_done += 1

    return current, SimplifyStats(
        before=before,
        after=len(current),
        removed=total_removed,
        passes=passes_done,
    )


def simplify_upper_contour_for_marching(
    upper: np.ndarray,
    *,
    max_sagitta_factor: float,
    max_turn_degrees: float,
    max_merged_segment_length: float,
    min_keep_points: int,
    max_passes: int,
    verbose: bool,
    label: str,
) -> np.ndarray:
    """
    Simplify an upper-half contour while preserving the real-axis endpoints.
    """
    upper = np.asarray(upper, dtype=float)
    if len(upper) < 4:
        return upper.copy()

    simplified, stats = simplify_open_polyline_by_sagitta_angle(
        upper,
        max_sagitta_factor=max_sagitta_factor,
        max_turn_degrees=max_turn_degrees,
        max_merged_segment_length=max_merged_segment_length,
        min_keep_points=min_keep_points,
        max_passes=max_passes,
    )

    # Keep endpoints exactly on the real axis.
    simplified = simplified.copy()
    simplified[0] = upper[0]
    simplified[-1] = upper[-1]
    simplified[0, 1] = 0.0
    simplified[-1, 1] = 0.0

    if verbose and (stats.removed > 0 or stats.before >= 1000):
        print(
            f"    simplified {label}: points {stats.before:,} -> {stats.after:,} "
            f"(removed {stats.removed:,}, passes={stats.passes}; "
            f"sag≤{max_sagitta_factor:g}, turn≤{max_turn_degrees:g}°, "
            f"merge≤{max_merged_segment_length:g})",
            flush=True,
        )

    return simplified


def resample_closed_polyline_by_count(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample a closed polyline to exactly n_samples points.
    Returns an open ring: the first point is NOT repeated at the end.
    """
    pts = close_if_needed(np.asarray(points, dtype=float))

    if len(pts) < 4:
        raise ValueError("Need at least 3 points to resample a closed contour.")

    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs * diffs).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])

    if total <= 0:
        raise ValueError("Degenerate contour with zero length.")

    target = np.linspace(0.0, total, int(n_samples), endpoint=False)
    xr = np.interp(target, s, pts[:, 0])
    yr = np.interp(target, s, pts[:, 1])

    return np.column_stack([xr, yr])


def circular_smooth_ring(points: np.ndarray, passes: int = 0, alpha: float = 0.35) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    for _ in range(max(0, int(passes))):
        avg = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pts = (1.0 - alpha) * pts + alpha * avg
    return pts


def triangle_orientation_stats(vertices: np.ndarray, tris: np.ndarray) -> dict:
    vals = []
    for a, b, c in tris:
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn > 0:
            vals.append(n[2] / nn)
    if not vals:
        return {}
    arr = np.asarray(vals)
    return {
        "count": int(len(arr)),
        "min_unit_z": float(arr.min()),
        "max_unit_z": float(arr.max()),
        "negative_z": int(np.sum(arr < 0)),
        "nonpositive_z": int(np.sum(arr <= 0)),
    }


def orient_triangles_up(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tris = np.asarray(tris, dtype=int).copy()
    for t, (a, b, c) in enumerate(tris):
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        n = np.cross(p1 - p0, p2 - p0)
        if n[2] < 0:
            tris[t] = (a, c, b)
    return tris


# ============================================================
# Mandelbrot potential
# ============================================================

def compute_mandelbrot_potential_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    max_iter: int,
    escape_radius: float,
):
    """
    Approximate the escape potential

        G(c) = lim 2^{-n} log |z_n|,
        z_{n+1}=z_n^2+c, z_0=0.

    Points that do not escape within max_iter are left at G=0.
    """
    x = np.linspace(xmin, xmax, int(nx))
    y = np.linspace(ymin, ymax, int(ny))

    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=np.complex128)
    active = np.ones(C.shape, dtype=bool)
    escaped = np.zeros(C.shape, dtype=bool)
    G = np.zeros(C.shape, dtype=np.float64)

    for n in range(1, int(max_iter) + 1):
        Z[active] = Z[active] * Z[active] + C[active]
        absZ = np.abs(Z)

        newly_escaped = active & (absZ > escape_radius)
        if np.any(newly_escaped):
            G[newly_escaped] = np.ldexp(np.log(absZ[newly_escaped]), -n)
            escaped[newly_escaped] = True
            active[newly_escaped] = False

        if not np.any(active):
            break

    return x, y, G, escaped



def compute_mandelbrot_potential_points(
    xy: np.ndarray,
    max_iter: int,
    escape_radius: float,
    batch_size: int = 50000,
) -> np.ndarray:
    """
    Directly evaluate the escape potential at arbitrary xy points.

    This avoids the worst artifact of grid interpolation near the Mandelbrot
    boundary: the grid stores unresolved/interior points as G=0, so bilinear
    interpolation near the boundary can mix exterior values with artificial
    zeros and create jagged height spikes.
    """
    xy = np.asarray(xy, dtype=float)
    out = np.zeros(len(xy), dtype=float)

    n_total = len(xy)
    batch_size = max(1, int(batch_size))

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        C = xy[start:end, 0] + 1j * xy[start:end, 1]
        Z = np.zeros_like(C, dtype=np.complex128)
        active = np.ones(C.shape, dtype=bool)
        G = np.zeros(C.shape, dtype=float)

        active_indices = np.arange(len(C))

        for n in range(1, int(max_iter) + 1):
            if not np.any(active):
                break

            Z[active] = Z[active] * Z[active] + C[active]
            abs_active = np.abs(Z[active])
            newly_local = abs_active > escape_radius

            if np.any(newly_local):
                global_local_idx = active_indices[active][newly_local]
                G[global_local_idx] = np.ldexp(np.log(abs_active[newly_local]), -n)

                # Update active mask
                tmp = np.zeros_like(active)
                tmp[active] = newly_local
                active[tmp] = False

        out[start:end] = G

        print(f"  direct G: {end:,}/{n_total:,} vertices", flush=True)

    return out


def clamp_boundary_g_values(
    xy: np.ndarray,
    Gv: np.ndarray,
    inner_boundary: np.ndarray,
    outer_boundary: np.ndarray,
    gmin: float,
    gmax: float,
    tol: float,
) -> np.ndarray:
    """
    Optionally force vertices very close to the polygonized boundary loops to
    have the exact boundary potential. Useful because polygonization/resampling
    can move points a tiny bit off the extracted contour.
    """
    if tol <= 0:
        return Gv

    xy = np.asarray(xy, dtype=float)
    Gv = np.asarray(Gv, dtype=float).copy()

    inner_tree = cKDTree(np.asarray(inner_boundary, dtype=float))
    outer_tree = cKDTree(np.asarray(outer_boundary, dtype=float))

    d_inner, _ = inner_tree.query(xy, k=1)
    d_outer, _ = outer_tree.query(xy, k=1)

    inner_mask = d_inner <= tol
    outer_mask = d_outer <= tol

    Gv[inner_mask] = gmin
    Gv[outer_mask] = gmax

    print(
        f"Boundary G clamp: {int(inner_mask.sum()):,} inner vertices, "
        f"{int(outer_mask.sum()):,} outer vertices within tol={tol:g}"
    )

    return Gv


def print_g_stats(label: str, Gv: np.ndarray) -> None:
    finite = np.asarray(Gv, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        print(f"{label}: no finite values")
        return
    print(
        f"{label}: min={finite.min():.6g}, "
        f"p01={np.quantile(finite, 0.01):.6g}, "
        f"median={np.quantile(finite, 0.5):.6g}, "
        f"p99={np.quantile(finite, 0.99):.6g}, "
        f"max={finite.max():.6g}"
    )





# ============================================================
# Direct contour tracing
# ============================================================

class PointPotentialCache:
    """
    Cached evaluator for G(c) at arbitrary xy points.

    The contour tracer asks for many nearby points on small circles, and some
    points repeat during bisection / loop closure checks. Caching avoids
    recomputing the exact same dyadic-ish samples.
    """
    def __init__(
        self,
        max_iter: int,
        escape_radius: float,
        batch_size: int = 50000,
        key_digits: int = 14,
    ):
        self.max_iter = int(max_iter)
        self.escape_radius = float(escape_radius)
        self.batch_size = int(batch_size)
        self.key_digits = int(key_digits)
        self.cache: dict[tuple[float, float], float] = {}

    def _key(self, x: float, y: float) -> tuple[float, float]:
        return (round(float(x), self.key_digits), round(float(y), self.key_digits))

    def get_many(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        out = np.empty(len(xy), dtype=float)

        missing_points = []
        missing_keys = []
        missing_indices = []

        for i, (x, y) in enumerate(xy):
            key = self._key(x, y)
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                missing_points.append((float(x), float(y)))
                missing_keys.append(key)
                missing_indices.append(i)

        if missing_points:
            vals = compute_mandelbrot_potential_points_quiet(
                np.asarray(missing_points, dtype=float),
                max_iter=self.max_iter,
                escape_radius=self.escape_radius,
                batch_size=self.batch_size,
            )
            for idx, key, val in zip(missing_indices, missing_keys, vals):
                v = float(val)
                self.cache[key] = v
                out[idx] = v

        return out

    def get(self, x: float, y: float) -> float:
        return float(self.get_many(np.array([[x, y]], dtype=float))[0])


def compute_mandelbrot_potential_points_quiet(
    xy: np.ndarray,
    max_iter: int,
    escape_radius: float,
    batch_size: int = 50000,
) -> np.ndarray:
    """
    Quiet direct evaluator used internally by the contour tracer.
    """
    xy = np.asarray(xy, dtype=float)
    out = np.zeros(len(xy), dtype=float)
    batch_size = max(1, int(batch_size))

    for start in range(0, len(xy), batch_size):
        end = min(start + batch_size, len(xy))
        C = xy[start:end, 0] + 1j * xy[start:end, 1]
        Z = np.zeros_like(C, dtype=np.complex128)
        active = np.ones(C.shape, dtype=bool)
        G = np.zeros(C.shape, dtype=float)
        active_indices = np.arange(len(C))

        for n in range(1, int(max_iter) + 1):
            if not np.any(active):
                break

            Z[active] = Z[active] * Z[active] + C[active]
            abs_active = np.abs(Z[active])
            newly_local = abs_active > escape_radius

            if np.any(newly_local):
                global_local_idx = active_indices[active][newly_local]
                G[global_local_idx] = np.ldexp(np.log(abs_active[newly_local]), -n)

                tmp = np.zeros_like(active)
                tmp[active] = newly_local
                active[tmp] = False

        out[start:end] = G

    return out


def find_trace_start_on_ray(
    level: float,
    cache: PointPotentialCache,
    start_angle: float = 0.0,
    ray_origin: tuple[float, float] = (0.0, 0.0),
    initial_high: float = 0.5,
    max_high: float = 10.0,
    bisect_steps: int = 80,
) -> np.ndarray:
    """
    Find a point on G(c)=level by shooting a ray from an interior point.

    Default ray is from c=0 in the positive real direction, which intersects
    the exterior contour on the right side of the main cardioid.
    """
    ox, oy = ray_origin
    direction = np.array([math.cos(start_angle), math.sin(start_angle)], dtype=float)

    low_t = 0.0
    low_p = np.array([ox, oy], dtype=float)
    low_f = cache.get(low_p[0], low_p[1]) - level

    if low_f > 0:
        raise RuntimeError(
            "Ray origin is already outside the requested contour. "
            "Use a different --trace-start-angle or ray origin in code."
        )

    high_t = float(initial_high)
    while high_t <= max_high:
        high_p = np.array([ox, oy], dtype=float) + high_t * direction
        high_f = cache.get(high_p[0], high_p[1]) - level
        if high_f >= 0:
            break
        high_t *= 2.0
    else:
        raise RuntimeError(f"Could not bracket contour level {level:g} along the start ray.")

    lo = low_t
    hi = high_t
    for _ in range(int(bisect_steps)):
        mid = 0.5 * (lo + hi)
        p = np.array([ox, oy], dtype=float) + mid * direction
        f = cache.get(p[0], p[1]) - level
        if f >= 0:
            hi = mid
        else:
            lo = mid

    return np.array([ox, oy], dtype=float) + hi * direction


def circle_level_roots(
    center: np.ndarray,
    radius: float,
    level: float,
    cache: PointPotentialCache,
    circle_samples: int,
    bisect_steps: int,
) -> list[np.ndarray]:
    """
    Find intersections of G(c)=level with a circle around center.

    Returns zero or more root points on the circle.
    """
    center = np.asarray(center, dtype=float)
    n = max(16, int(circle_samples))
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    circle = center[None, :] + radius * np.column_stack([np.cos(theta), np.sin(theta)])

    vals = cache.get_many(circle) - level
    roots: list[np.ndarray] = []

    def point_at_angle(a: float) -> np.ndarray:
        return center + radius * np.array([math.cos(a), math.sin(a)], dtype=float)

    for i in range(n):
        j = (i + 1) % n
        f0 = float(vals[i])
        f1 = float(vals[j])
        t0 = float(theta[i])
        t1 = float(theta[j] if j != 0 else 2.0 * math.pi)

        if not np.isfinite(f0) or not np.isfinite(f1):
            continue

        # Exact-ish sample hit.
        if abs(f0) < 1e-14:
            roots.append(circle[i])
            continue

        if f0 * f1 > 0:
            continue

        # Bisection in angle.
        a = t0
        b = t1
        fa = f0
        fb = f1

        for _ in range(int(bisect_steps)):
            m = 0.5 * (a + b)
            pm = point_at_angle(m)
            fm = cache.get(pm[0], pm[1]) - level

            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm

        roots.append(point_at_angle(0.5 * (a + b)))

    # Deduplicate nearby roots.
    deduped: list[np.ndarray] = []
    eps = max(1e-12, radius * 1e-5)
    for r in roots:
        if not any(np.linalg.norm(r - q) < eps for q in deduped):
            deduped.append(r)

    return deduped


def choose_next_trace_root(
    roots: list[np.ndarray],
    current: np.ndarray,
    previous: np.ndarray | None,
    start: np.ndarray,
    tangent_hint: np.ndarray,
    radius: float,
    min_separation_factor: float,
) -> np.ndarray | None:
    """
    Choose the forward root among circle-contour intersections.
    """
    if not roots:
        return None

    current = np.asarray(current, dtype=float)

    if previous is None:
        # First step: use tangent hint, e.g. +y direction at rightmost start.
        hint = tangent_hint / max(1e-15, np.linalg.norm(tangent_hint))
        scored = []
        for r in roots:
            v = r - current
            vn = np.linalg.norm(v)
            if vn <= 0:
                continue
            scored.append((float(np.dot(v / vn, hint)), r))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    previous = np.asarray(previous, dtype=float)
    prev_dir = current - previous
    prev_norm = np.linalg.norm(prev_dir)
    if prev_norm == 0:
        return None
    prev_dir = prev_dir / prev_norm

    candidates = []
    for r in roots:
        # Avoid taking the root we came from.
        dprev = np.linalg.norm(r - previous)
        if dprev < min_separation_factor * radius:
            continue

        v = r - current
        vn = np.linalg.norm(v)
        if vn == 0:
            continue

        forward_score = float(np.dot(v / vn, prev_dir))
        candidates.append((forward_score, dprev, r))

    if not candidates:
        return None

    # Prefer continuation of tangent direction; tie-break by distance from previous.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]



class TraceQualityFailure(RuntimeError):
    def __init__(self, message: str, point_count: int, length: float, current: np.ndarray, suggested_radius: float):
        super().__init__(message)
        self.point_count = int(point_count)
        self.length = float(length)
        self.current = np.asarray(current, dtype=float)
        self.suggested_radius = float(suggested_radius)


def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    x = float(np.dot(a, b) / (na * nb))
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))


def _line_level_roots_near_midpoint(
    midpoint: np.ndarray,
    normal: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    span: float,
    samples: int,
    bisect_steps: int,
) -> list[tuple[float, np.ndarray]]:
    """
    Find roots of G(midpoint + s normal) = level for s in [-span, +span].

    Returns [(s, point), ...]. The caller usually wants the root with smallest
    |s|, i.e. the contour point nearest the chord midpoint.
    """
    midpoint = np.asarray(midpoint, dtype=float)
    normal = np.asarray(normal, dtype=float)
    nn = float(np.linalg.norm(normal))
    if nn <= 0:
        return []
    normal = normal / nn

    span = float(span)
    samples = max(5, int(samples))
    if samples % 2 == 0:
        samples += 1

    s_vals = np.linspace(-span, span, samples)
    pts = midpoint[None, :] + s_vals[:, None] * normal[None, :]
    vals = cache.get_many(pts) - level

    roots: list[tuple[float, np.ndarray]] = []

    def point_at_s(s: float) -> np.ndarray:
        return midpoint + float(s) * normal

    for i in range(samples - 1):
        s0 = float(s_vals[i])
        s1 = float(s_vals[i + 1])
        f0 = float(vals[i])
        f1 = float(vals[i + 1])

        if not np.isfinite(f0) or not np.isfinite(f1):
            continue

        if abs(f0) < 1e-14:
            roots.append((s0, point_at_s(s0)))
            continue

        if f0 * f1 > 0:
            continue

        a = s0
        b = s1
        fa = f0

        for _ in range(int(bisect_steps)):
            m = 0.5 * (a + b)
            fm = cache.get(*point_at_s(m)) - level
            if fa * fm <= 0:
                b = m
            else:
                a = m
                fa = fm

        s = 0.5 * (a + b)
        roots.append((s, point_at_s(s)))

    # Deduplicate roots that are almost identical in s.
    roots.sort(key=lambda t: t[0])
    deduped: list[tuple[float, np.ndarray]] = []
    eps = max(1e-12, span * 1e-5)
    for s, p in roots:
        if not deduped or abs(s - deduped[-1][0]) > eps:
            deduped.append((s, p))

    return deduped


def check_trace_segment_quality(
    p0: np.ndarray,
    p1: np.ndarray,
    previous: np.ndarray | None,
    level: float,
    cache: PointPotentialCache,
    normal_span_factor: float,
    normal_samples: int,
    bisect_steps: int,
    max_sagitta_factor: float,
    max_turn_degrees: float,
) -> tuple[bool, str, dict]:
    """
    Check whether segment p0 -> p1 is a locally trustworthy approximation
    to the contour.

    The clean geometric test is not "search along the chord" because the chord
    already has contour roots at the endpoints. Instead we use the perpendicular
    line through the chord midpoint. If the actual contour point nearest that
    midpoint is far from the midpoint, then the chord is skipping unresolved
    curvature/wrinkles.

    Returns (ok, message, info).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 0:
        return False, "zero-length step", {"chord_len": chord_len}

    info = {"chord_len": chord_len}

    # Turn check: if the local direction changes wildly, this is likely either
    # a sharp underresolved wrinkle or a branch hop.
    if previous is not None:
        prev_vec = p0 - np.asarray(previous, dtype=float)
        turn_deg = _angle_between_degrees(prev_vec, chord)
        info["turn_deg"] = turn_deg
        if turn_deg > max_turn_degrees:
            return False, f"turn angle too large: {turn_deg:.2f} deg > {max_turn_degrees:.2f} deg", info

    midpoint = 0.5 * (p0 + p1)
    normal = np.array([-chord[1], chord[0]], dtype=float)
    span = max(1e-14, float(normal_span_factor) * chord_len)

    roots = _line_level_roots_near_midpoint(
        midpoint=midpoint,
        normal=normal,
        level=level,
        cache=cache,
        span=span,
        samples=normal_samples,
        bisect_steps=bisect_steps,
    )

    if not roots:
        # This is not always fatal in perfectly flat regions because the chord
        # midpoint can sit on one side and the normal segment might be too short.
        # But for our use, no nearby midpoint crossing means the step is not
        # locally resolving the level set.
        return False, f"no midpoint-normal contour crossing within span={span:.4g}", info

    s_best, p_best = min(roots, key=lambda t: abs(t[0]))
    sagitta = abs(float(s_best))
    sagitta_factor = sagitta / chord_len

    info["sagitta"] = sagitta
    info["sagitta_factor"] = sagitta_factor
    info["normal_roots"] = len(roots)

    if sagitta_factor > max_sagitta_factor:
        return (
            False,
            f"midpoint sagitta too large: {sagitta_factor:.3f} > {max_sagitta_factor:.3f}",
            info,
        )

    # Multiple nearby roots along the midpoint normal means the contour is
    # locally folded at the scale of this step. That is a strong hint to use
    # a smaller radius, but do not fail solely on it by default.
    return True, "ok", info


def _trace_potential_contour_once(
    level: float,
    cache: PointPotentialCache,
    step_radius: float,
    start_angle: float = 0.0,
    direction: str = "ccw",
    circle_samples: int = 72,
    bisect_steps: int = 28,
    max_points: int = 20000,
    min_points_before_close: int = 50,
    close_factor: float = 1.25,
    min_separation_factor: float = 0.35,
    max_radius_adjust: int = 8,
    radius_shrink: float = 0.65,
    verbose: bool = True,
    progress_every: int = 250,
    quality_check: bool = True,
    quality_every: int = 1,
    quality_start_after: int = 5,
    quality_normal_span_factor: float = 3.0,
    quality_normal_samples: int = 15,
    quality_max_sagitta_factor: float = 0.30,
    quality_max_turn_degrees: float = 135.0,
    quality_action: str = "rollback",
    quality_rollback_points: int = 20,
    quality_restart_shrink: float = 0.5,
    quality_min_radius: float = 1e-6,
    quality_max_local_failures: int = 100,
    stop_at_point: np.ndarray | None = None,
    stop_point_close_factor: float = 2.0,
    stop_point_min_x: float = -0.25,
    stop_label: str = "target",
) -> np.ndarray:
    """
    Trace a closed contour G(c)=level by local circle intersections.

    From the current point, draw a small circle. If the radius is small enough,
    the contour should cross it in two places: the previous point and the next
    point. We choose the root that continues the local direction.
    """
    level = float(level)
    base_radius = float(step_radius)
    current_radius = base_radius
    local_quality_failures = 0

    start = find_trace_start_on_ray(
        level=level,
        cache=cache,
        start_angle=float(start_angle),
    )

    radial = np.array([math.cos(start_angle), math.sin(start_angle)], dtype=float)
    tangent = np.array([-radial[1], radial[0]], dtype=float)
    if direction == "cw":
        tangent = -tangent

    points = [start]
    previous = None
    current = start
    accumulated_length = 0.0
    progress_every = int(progress_every)

    if verbose:
        print(
            f"    start c = {start[0]:+.10f} {start[1]:+.10f}i",
            flush=True,
        )

    for step_idx in range(int(max_points)):
        found = None
        used_radius = base_radius

        for attempt in range(int(max_radius_adjust)):
            r = current_radius * (radius_shrink ** attempt)
            roots = circle_level_roots(
                center=current,
                radius=r,
                level=level,
                cache=cache,
                circle_samples=circle_samples,
                bisect_steps=bisect_steps,
            )

            cand = choose_next_trace_root(
                roots=roots,
                current=current,
                previous=previous,
                start=start,
                tangent_hint=tangent,
                radius=r,
                min_separation_factor=min_separation_factor,
            )

            if cand is not None:
                found = cand
                used_radius = r
                break

        if found is None:
            raise RuntimeError(
                f"Contour tracing failed at level {level:g}, step {step_idx}, "
                f"point {current}. Try smaller --trace-inner-step/--trace-outer-step "
                f"or larger --trace-circle-samples."
            )

        if quality_check and previous is not None:
            check_index = len(points)
            if (
                check_index >= int(quality_start_after)
                and int(quality_every) > 0
                and check_index % int(quality_every) == 0
            ):
                ok, reason, info = check_trace_segment_quality(
                    p0=current,
                    p1=found,
                    previous=previous,
                    level=level,
                    cache=cache,
                    normal_span_factor=quality_normal_span_factor,
                    normal_samples=quality_normal_samples,
                    bisect_steps=bisect_steps,
                    max_sagitta_factor=quality_max_sagitta_factor,
                    max_turn_degrees=quality_max_turn_degrees,
                )
                if not ok:
                    suggested = max(used_radius * float(quality_restart_shrink), 1e-15)
                    msg = (
                        f"quality check failed at G={level:g}, points={len(points):,}, "
                        f"length≈{accumulated_length:.8g}, "
                        f"current c={current[0]:+.10f} {current[1]:+.10f}i, "
                        f"candidate c={found[0]:+.10f} {found[1]:+.10f}i, "
                        f"radius={used_radius:g}: {reason}; info={info}"
                    )

                    if quality_action == "rollback":
                        local_quality_failures += 1
                        if local_quality_failures > int(quality_max_local_failures):
                            raise RuntimeError(
                                f"Exceeded --trace-quality-max-local-failures="
                                f"{quality_max_local_failures}. Last failure: {msg}"
                            )

                        new_radius = max(float(quality_min_radius), suggested)
                        if new_radius >= current_radius:
                            new_radius = max(float(quality_min_radius), current_radius * float(quality_restart_shrink))

                        if new_radius <= float(quality_min_radius) and current_radius <= float(quality_min_radius) * 1.0001:
                            raise RuntimeError(
                                f"Quality check wants radius below minimum {quality_min_radius:g}. "
                                f"Last failure: {msg}"
                            )

                        # Rewind a small local suffix only. This keeps the good
                        # part of the contour and avoids restarting from c_start
                        # after reaching the left side.
                        keep_last_index = max(0, len(points) - 1 - int(quality_rollback_points))
                        points = points[:keep_last_index + 1]
                        current = points[-1]
                        previous = points[-2] if len(points) >= 2 else None
                        accumulated_length = polyline_length_open(points)
                        current_radius = new_radius

                        if verbose:
                            print(
                                f"  {msg}\n"
                                f"  local rollback: kept {len(points):,} points, "
                                f"length≈{accumulated_length:.8g}, "
                                f"new radius {current_radius:g}; "
                                f"cache preserved ({len(cache.cache):,} values)",
                                flush=True,
                            )

                        # Discard this candidate and continue from the rewound point.
                        continue

                    else:
                        raise TraceQualityFailure(
                            msg,
                            point_count=len(points),
                            length=accumulated_length,
                            current=current,
                            suggested_radius=suggested,
                        )

        if len(points) >= int(min_points_before_close):
            if stop_at_point is not None:
                target = np.asarray(stop_at_point, dtype=float)
                if (
                    current[0] < float(stop_point_min_x)
                    and found[0] < float(stop_point_min_x)
                    and np.linalg.norm(found - target) < float(stop_point_close_factor) * used_radius
                ):
                    accumulated_length += float(np.linalg.norm(current - target))
                    points.append(target.copy())
                    if verbose:
                        print(
                            f"  traced G={level:g}: reached {stop_label} after {len(points)-1:,} points, "
                            f"upper length≈{accumulated_length:.8g}, "
                            f"target c={target[0]:+.10f} {target[1]:+.10f}i, "
                            f"radius≈{base_radius:g}, cache={len(cache.cache):,}",
                            flush=True,
                        )
                    break
            else:
                if np.linalg.norm(found - start) < close_factor * used_radius:
                    accumulated_length += float(np.linalg.norm(current - start))
                    points.append(start.copy())
                    if verbose:
                        print(
                            f"  traced G={level:g}: closed after {len(points)-1:,} points, "
                            f"length≈{accumulated_length:.8g}, "
                            f"final c={start[0]:+.10f} {start[1]:+.10f}i, "
                            f"radius≈{base_radius:g}, cache={len(cache.cache):,}",
                            flush=True,
                        )
                    break

        step_len = float(np.linalg.norm(found - current))
        accumulated_length += step_len

        previous = current
        current = found
        points.append(current)

        # After successful steps, gently grow back toward the original requested radius.
        # This makes the radius small only where the contour actually demanded it.
        if quality_action == "rollback" and current_radius < base_radius:
            current_radius = min(base_radius, current_radius * 1.01)

        if verbose and progress_every > 0 and (len(points) % progress_every == 0):
            print(
                f"    G={level:g}: points={len(points):,}, "
                f"length≈{accumulated_length:.8g}, "
                f"current c={current[0]:+.10f} {current[1]:+.10f}i, "
                f"step≈{step_len:.4g}, radius used≈{used_radius:.4g}, "
                f"base radius≈{current_radius:.4g}, cache={len(cache.cache):,}",
                flush=True,
            )

    else:
        raise RuntimeError(
            f"Contour tracing hit --trace-max-points={max_points} for level {level:g} "
            "without closing."
        )

    pts = np.asarray(points, dtype=float)
    # Drop repeated closure point for downstream routines, then enforce orientation.
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-10:
        pts = pts[:-1]

    pts = ensure_ccw(pts)
    return pts



def trace_potential_contour(
    level: float,
    cache: PointPotentialCache,
    step_radius: float,
    start_angle: float = 0.0,
    direction: str = "ccw",
    circle_samples: int = 72,
    bisect_steps: int = 28,
    max_points: int = 20000,
    min_points_before_close: int = 50,
    close_factor: float = 1.25,
    min_separation_factor: float = 0.35,
    max_radius_adjust: int = 8,
    radius_shrink: float = 0.65,
    verbose: bool = True,
    progress_every: int = 250,
    quality_check: bool = True,
    quality_every: int = 1,
    quality_start_after: int = 5,
    quality_normal_span_factor: float = 3.0,
    quality_normal_samples: int = 15,
    quality_max_sagitta_factor: float = 0.30,
    quality_max_turn_degrees: float = 135.0,
    quality_action: str = "rollback",
    quality_rollback_points: int = 20,
    quality_restart_shrink: float = 0.5,
    quality_max_restarts: int = 8,
    quality_min_radius: float = 1e-6,
    quality_max_local_failures: int = 100,
    stop_at_point: np.ndarray | None = None,
    stop_point_close_factor: float = 2.0,
    stop_point_min_x: float = -0.25,
    stop_label: str = "target",
) -> np.ndarray:
    """
    Trace with restart-on-quality-failure.

    If the midpoint/turn checker says the current radius is too coarse, throw
    away the traced points, keep the cache, shrink the circle radius, and start
    the same contour again from the original ray hit.
    """
    radius = float(step_radius)

    for restart in range(int(quality_max_restarts) + 1):
        try:
            if verbose and restart > 0:
                print(
                    f"  restart {restart}/{quality_max_restarts} for G={level:g}: "
                    f"new step radius {radius:g}; cache preserved ({len(cache.cache):,} values)",
                    flush=True,
                )

            return _trace_potential_contour_once(
                level=level,
                cache=cache,
                step_radius=radius,
                start_angle=start_angle,
                direction=direction,
                circle_samples=circle_samples,
                bisect_steps=bisect_steps,
                max_points=max_points,
                min_points_before_close=min_points_before_close,
                close_factor=close_factor,
                min_separation_factor=min_separation_factor,
                max_radius_adjust=max_radius_adjust,
                radius_shrink=radius_shrink,
                verbose=verbose,
                progress_every=progress_every,
                quality_check=quality_check,
                quality_every=quality_every,
                quality_start_after=quality_start_after,
                quality_normal_span_factor=quality_normal_span_factor,
                quality_normal_samples=quality_normal_samples,
                quality_max_sagitta_factor=quality_max_sagitta_factor,
                quality_max_turn_degrees=quality_max_turn_degrees,
                quality_action=quality_action,
                quality_rollback_points=quality_rollback_points,
                quality_restart_shrink=quality_restart_shrink,
                quality_min_radius=quality_min_radius,
                quality_max_local_failures=quality_max_local_failures,
                stop_at_point=stop_at_point,
                stop_point_close_factor=stop_point_close_factor,
                stop_point_min_x=stop_point_min_x,
                stop_label=stop_label,
            )

        except TraceQualityFailure as exc:
            if not quality_check:
                raise

            if verbose:
                print(f"  {exc}", flush=True)

            new_radius = min(float(exc.suggested_radius), radius * float(quality_restart_shrink))
            if new_radius < float(quality_min_radius):
                raise RuntimeError(
                    f"Quality restart wants radius {new_radius:g}, below --trace-quality-min-radius="
                    f"{quality_min_radius:g}. Last failure: {exc}"
                ) from exc

            radius = new_radius

    raise RuntimeError(
        f"Exceeded --trace-quality-max-restarts={quality_max_restarts} while tracing G={level:g}. "
        f"Last radius={radius:g}."
    )



def mirror_upper_half_contour_to_full(upper: np.ndarray, drop_duplicate_eps: float = 1e-10) -> np.ndarray:
    """
    Given an upper-half contour arc from right real-axis endpoint to left
    real-axis endpoint, mirror it across the real axis to get the full loop.

    Input is expected open: [right, ..., left].
    Output is open: [right, ..., left, mirrored points ..., near right].
    Downstream close_if_needed / Gmsh closes the final edge implicitly.
    """
    upper = np.asarray(upper, dtype=float)
    if len(upper) < 3:
        raise ValueError("Need at least 3 points for symmetry mirroring.")

    # Force endpoints exactly onto the real axis; this prevents tiny zippers
    # when the mirrored lower half meets the upper half.
    upper = upper.copy()
    upper[0, 1] = 0.0
    upper[-1, 1] = 0.0

    # Mirror the interior points in reverse order: after reaching the left
    # endpoint, walk along the lower half back toward the right endpoint.
    lower = upper[-2:0:-1].copy()
    lower[:, 1] *= -1.0

    full = np.vstack([upper, lower])

    # Remove accidental near-duplicate consecutive points.
    cleaned = [full[0]]
    for p in full[1:]:
        if np.linalg.norm(p - cleaned[-1]) > drop_duplicate_eps:
            cleaned.append(p)

    return np.asarray(cleaned, dtype=float)


def trace_potential_contour_symmetric(
    level: float,
    cache: PointPotentialCache,
    step_radius: float,
    start_angle: float = 0.0,
    direction: str = "ccw",
    circle_samples: int = 72,
    bisect_steps: int = 28,
    max_points: int = 20000,
    min_points_before_close: int = 50,
    close_factor: float = 1.25,
    min_separation_factor: float = 0.35,
    max_radius_adjust: int = 8,
    radius_shrink: float = 0.65,
    verbose: bool = True,
    progress_every: int = 250,
    quality_check: bool = True,
    quality_every: int = 1,
    quality_start_after: int = 5,
    quality_normal_span_factor: float = 3.0,
    quality_normal_samples: int = 15,
    quality_max_sagitta_factor: float = 0.30,
    quality_max_turn_degrees: float = 135.0,
    quality_action: str = "rollback",
    quality_rollback_points: int = 20,
    quality_restart_shrink: float = 0.5,
    quality_max_restarts: int = 8,
    quality_min_radius: float = 1e-6,
    quality_max_local_failures: int = 100,
    symmetry_target_factor: float = 2.0,
    symmetry_min_x: float = -0.25,
) -> np.ndarray:
    """
    Trace only the upper half of the contour and mirror it.

    Because G(x+iy) = G(x-iy), the full equipotential can be built from the
    upper half. We start at the right real-axis intersection and trace CCW
    until we reach the left real-axis intersection found independently by
    bisection on the negative real ray.
    """
    if abs(float(start_angle)) > 1e-14:
        raise ValueError("Symmetry mode currently expects --trace-start-angle 0.0.")

    if direction != "ccw":
        raise ValueError("Symmetry mode currently expects --trace-direction ccw.")

    left_target = find_trace_start_on_ray(
        level=level,
        cache=cache,
        start_angle=math.pi,
    )
    left_target[1] = 0.0

    if verbose:
        print(
            f"    symmetry target left real-axis c = {left_target[0]:+.10f} {left_target[1]:+.10f}i",
            flush=True,
        )

    # Use the same robust tracer, but ask it to stop at the left real-axis
    # endpoint instead of closing back to the right endpoint.
    upper = trace_potential_contour(
        level=level,
        cache=cache,
        step_radius=step_radius,
        start_angle=start_angle,
        direction=direction,
        circle_samples=circle_samples,
        bisect_steps=bisect_steps,
        max_points=max_points,
        min_points_before_close=min_points_before_close,
        close_factor=close_factor,
        min_separation_factor=min_separation_factor,
        max_radius_adjust=max_radius_adjust,
        radius_shrink=radius_shrink,
        verbose=verbose,
        progress_every=progress_every,
        quality_check=quality_check,
        quality_every=quality_every,
        quality_start_after=quality_start_after,
        quality_normal_span_factor=quality_normal_span_factor,
        quality_normal_samples=quality_normal_samples,
        quality_max_sagitta_factor=quality_max_sagitta_factor,
        quality_max_turn_degrees=quality_max_turn_degrees,
        quality_action=quality_action,
        quality_rollback_points=quality_rollback_points,
        quality_restart_shrink=quality_restart_shrink,
        quality_max_restarts=quality_max_restarts,
        quality_min_radius=quality_min_radius,
        quality_max_local_failures=quality_max_local_failures,
        stop_at_point=left_target,
        stop_point_close_factor=symmetry_target_factor,
        stop_point_min_x=symmetry_min_x,
        stop_label="left real-axis target",
    )

    # Replace final near-target point with the exact bisection target. This
    # makes the mirror seam perfectly sit on Im(c)=0.
    upper = np.asarray(upper, dtype=float)
    if len(upper) >= 2:
        upper[-1] = left_target

    full = mirror_upper_half_contour_to_full(upper)

    if verbose:
        print(
            f"  mirrored G={level:g}: upper points={len(upper):,}, "
            f"full points={len(full):,}, full length≈{polyline_length_closed(full):.8g}",
            flush=True,
        )

    return ensure_ccw(full)


def trace_step_for_level(
    level: float,
    gmin: float,
    gmax: float,
    inner_step: float,
    outer_step: float,
) -> float:
    """
    Geometric interpolation of tracing radius between inner and outer levels.
    """
    if gmax <= gmin:
        return float(inner_step)
    t = (math.log(level) - math.log(gmin)) / (math.log(gmax) - math.log(gmin))
    t = max(0.0, min(1.0, t))
    return float(inner_step) * (float(outer_step) / float(inner_step)) ** t



# ============================================================
# Adaptive perpendicular-bisector contour construction
# ============================================================

@dataclass
class BisectorStats:
    processed: int = 0
    accepted: int = 0
    subdivided: int = 0
    forced_accept: int = 0
    no_root: int = 0
    multi_root: int = 0
    max_depth_seen: int = 0


def _unit_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v.copy()
    return v / n


def _dedupe_line_roots(
    roots: list[tuple[float, np.ndarray]],
    span: float,
) -> list[tuple[float, np.ndarray]]:
    roots.sort(key=lambda item: item[0])
    deduped: list[tuple[float, np.ndarray]] = []
    eps = max(1e-13, float(span) * 1e-7)
    for s, p in roots:
        if not deduped or abs(float(s) - float(deduped[-1][0])) > eps:
            deduped.append((float(s), np.asarray(p, dtype=float)))
    return deduped


def _bisect_root_on_line(
    center: np.ndarray,
    normal: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    s0: float,
    s1: float,
    f0: float,
    f1: float,
    bisect_steps: int,
) -> tuple[float, np.ndarray]:
    """
    Refine a sign-changing root of G(center+s normal)-level.
    """
    center = np.asarray(center, dtype=float)
    normal = _unit_vec(normal)

    a = float(s0)
    b = float(s1)
    fa = float(f0)

    def point_at_s(s: float) -> np.ndarray:
        return center + float(s) * normal

    for _ in range(int(bisect_steps)):
        m = 0.5 * (a + b)
        fm = cache.get(*point_at_s(m)) - level
        if fa * fm <= 0:
            b = m
        else:
            a = m
            fa = fm

    s = 0.5 * (a + b)
    return s, point_at_s(s)


def _line_roots_from_samples(
    center: np.ndarray,
    normal: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    s_vals: np.ndarray,
    vals: np.ndarray,
    bisect_steps: int,
) -> list[tuple[float, np.ndarray]]:
    roots: list[tuple[float, np.ndarray]] = []
    center = np.asarray(center, dtype=float)
    normal = _unit_vec(normal)

    def point_at_s(s: float) -> np.ndarray:
        return center + float(s) * normal

    for i in range(len(s_vals) - 1):
        s0 = float(s_vals[i])
        s1 = float(s_vals[i + 1])
        f0 = float(vals[i])
        f1 = float(vals[i + 1])

        if not np.isfinite(f0) or not np.isfinite(f1):
            continue

        if abs(f0) < 1e-14:
            roots.append((s0, point_at_s(s0)))
            continue

        if f0 * f1 <= 0:
            roots.append(
                _bisect_root_on_line(
                    center=center,
                    normal=normal,
                    level=level,
                    cache=cache,
                    s0=s0,
                    s1=s1,
                    f0=f0,
                    f1=f1,
                    bisect_steps=bisect_steps,
                )
            )

    return roots


def line_level_roots(
    center: np.ndarray,
    normal: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    span: float,
    samples: int,
    bisect_steps: int,
) -> list[tuple[float, np.ndarray]]:
    """
    Find roots of G(center + s normal) = level for s in [-span, span].

    Important detail: a uniform 81-point scan over a long line can miss a very
    narrow sign-changing pocket near the Mandelbrot boundary. That is exactly
    what happens near the left cusp. So this function first does a normal
    sign-change scan, then adaptively resamples small windows around the most
    promising |G-level| minima.

    This is still cheap compared with walking thousands of contour points,
    because it only refines a few one-dimensional windows.
    """
    center = np.asarray(center, dtype=float)
    normal = _unit_vec(normal)
    span = float(span)
    samples = max(9, int(samples))
    if samples % 2 == 0:
        samples += 1

    s_vals = np.linspace(-span, span, samples)
    pts = center[None, :] + s_vals[:, None] * normal[None, :]
    vals = cache.get_many(pts) - level

    roots = _line_roots_from_samples(
        center=center,
        normal=normal,
        level=level,
        cache=cache,
        s_vals=s_vals,
        vals=vals,
        bisect_steps=bisect_steps,
    )

    # Adaptive rescue: if the coarse grid missed a narrow crossing, refine
    # around local minima of |f|. We do this even if roots were found, because
    # a perpendicular line can legitimately cross the contour multiple times.
    abs_vals = np.where(np.isfinite(vals), np.abs(vals), np.inf)

    candidate_indices: set[int] = set()

    # Local minima of |f|.
    for i in range(1, len(abs_vals) - 1):
        if abs_vals[i] <= abs_vals[i - 1] and abs_vals[i] <= abs_vals[i + 1]:
            candidate_indices.add(i)

    # Also include the globally smallest few samples.
    finite_idx = np.where(np.isfinite(abs_vals))[0]
    if len(finite_idx):
        order = finite_idx[np.argsort(abs_vals[finite_idx])]
        for i in order[:8]:
            if 0 < int(i) < len(abs_vals) - 1:
                candidate_indices.add(int(i))

    # Refinement rounds. Each round resamples the best-looking small windows.
    # The first round window is one coarse-grid cell on each side. Further
    # rounds are smaller because they are centered around refined minima.
    windows: list[tuple[float, float]] = []
    for i in sorted(candidate_indices):
        a = float(s_vals[max(0, i - 1)])
        b = float(s_vals[min(len(s_vals) - 1, i + 1)])
        if b > a:
            windows.append((a, b))

    refine_samples = max(41, min(201, int(samples)))
    for _round in range(3):
        new_windows: list[tuple[float, float]] = []

        for a, b in windows:
            if b - a <= max(1e-14, span * 1e-12):
                continue

            ss = np.linspace(a, b, refine_samples)
            pp = center[None, :] + ss[:, None] * normal[None, :]
            ff = cache.get_many(pp) - level

            roots.extend(
                _line_roots_from_samples(
                    center=center,
                    normal=normal,
                    level=level,
                    cache=cache,
                    s_vals=ss,
                    vals=ff,
                    bisect_steps=bisect_steps,
                )
            )

            aa = np.where(np.isfinite(ff), np.abs(ff), np.inf)
            if len(aa) >= 3 and np.any(np.isfinite(aa)):
                # Keep a few best local minima for another round.
                locs: set[int] = set()
                for j in range(1, len(aa) - 1):
                    if aa[j] <= aa[j - 1] and aa[j] <= aa[j + 1]:
                        locs.add(j)

                finite_j = np.where(np.isfinite(aa))[0]
                if len(finite_j):
                    order_j = finite_j[np.argsort(aa[finite_j])]
                    for j in order_j[:4]:
                        if 0 < int(j) < len(aa) - 1:
                            locs.add(int(j))

                for j in sorted(locs):
                    lo = float(ss[max(0, j - 1)])
                    hi = float(ss[min(len(ss) - 1, j + 1)])
                    if hi > lo:
                        new_windows.append((lo, hi))

        if roots:
            # If roots are now found, one additional round is usually enough;
            # further refinement just burns time. Still dedupe robustly below.
            if _round >= 1:
                break

        # Limit explosion if many local minima appear on a line.
        new_windows.sort(key=lambda w: (w[1] - w[0], w[0]))
        windows = new_windows[:16]
        if not windows:
            break

    return _dedupe_line_roots(roots, span)




def bracketed_root_on_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    initial_step: float,
    grow: float,
    max_step: float,
    bisect_steps: int,
) -> np.ndarray | None:
    """
    Find G(origin + t direction) = level for t >= 0 by monotone-ish bracketing.

    This is much cheaper than sampling a whole long line. It is meant for the
    controlled cases where we already know which side of the segment the contour
    should lie on.
    """
    origin = np.asarray(origin, dtype=float)
    direction = _unit_vec(direction)
    if float(np.linalg.norm(direction)) <= 0:
        return None

    f0 = cache.get(origin[0], origin[1]) - float(level)
    if abs(f0) < 1e-14:
        return origin.copy()

    step = max(1e-12, float(initial_step))
    grow = max(1.05, float(grow))
    max_step = max(step, float(max_step))

    lo_t = 0.0
    lo_f = f0
    hi_t = step

    while hi_t <= max_step * (1.0 + 1e-12):
        p_hi = origin + hi_t * direction
        hi_f = cache.get(p_hi[0], p_hi[1]) - float(level)

        if lo_f * hi_f <= 0:
            a = lo_t
            b = hi_t
            fa = lo_f

            for _ in range(int(bisect_steps)):
                m = 0.5 * (a + b)
                p_m = origin + m * direction
                fm = cache.get(p_m[0], p_m[1]) - float(level)
                if fa * fm <= 0:
                    b = m
                else:
                    a = m
                    fa = fm

            return origin + 0.5 * (a + b) * direction

        lo_t = hi_t
        lo_f = hi_f
        hi_t *= grow

    return None


def find_bisector_point_outward_fast(
    p0: np.ndarray,
    p1: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    initial_step_factor: float,
    max_step: float,
    grow: float,
    bisect_steps: int,
) -> tuple[np.ndarray | None, dict]:
    """
    Fast controlled perpendicular-bisector root.

    For an upper-half edge ordered left -> right, the contour arc should lie on
    the left-normal side of the chord. The chord midpoint is normally lower-G
    than the contour, so we search only outward from the midpoint, not across a
    giant sampled line.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 0:
        return None, {"reason": "zero chord", "chord_len": chord_len}

    midpoint = 0.5 * (p0 + p1)
    normal = np.array([-chord[1], chord[0]], dtype=float)
    normal = _unit_vec(normal)

    # For left -> right upper arcs, the wanted side is usually +y.
    # If the normal points down, flip it.
    if normal[1] < 0:
        normal = -normal

    q = bracketed_root_on_ray(
        origin=midpoint,
        direction=normal,
        level=level,
        cache=cache,
        initial_step=max(1e-8, float(initial_step_factor) * chord_len),
        grow=grow,
        max_step=max(float(max_step), 0.5 * chord_len),
        bisect_steps=bisect_steps,
    )

    if q is None:
        return None, {
            "reason": "outward bracket failed",
            "chord_len": chord_len,
            "max_step": max(float(max_step), 0.5 * chord_len),
        }

    sagitta = float(np.linalg.norm(q - midpoint))
    sagitta_factor = sagitta / max(chord_len, 1e-300)
    v0 = p0 - q
    v1 = p1 - q
    angle_at_q = _angle_between_degrees(v0, v1)
    turn_deg = 180.0 - angle_at_q

    return q, {
        "reason": "ok-fast",
        "chord_len": chord_len,
        "sagitta": sagitta,
        "sagitta_factor": sagitta_factor,
        "angle_at_p": angle_at_q,
        "turn_deg": turn_deg,
    }


def find_bisector_point_nearest_fast(
    p0: np.ndarray,
    p1: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    initial_step_factor: float,
    max_step: float,
    grow: float,
    bisect_steps: int,
    upper_only: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """
    Fast local perpendicular-bisector root for an already ordered contour edge.

    Important difference from find_bisector_point_outward_fast():
      - this searches BOTH sides of the perpendicular bisector,
      - then chooses the nearest usable root to the chord midpoint.

    The older +y-only version is fine for the very first real-axis chord, but
    it is dangerous during marching. A local arc is not always above its chord.
    Forcing +y can jump to another nearby branch; the array order is preserved
    syntactically, but the polyline order becomes geometrically wrong.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 0:
        return None, {"reason": "zero chord", "chord_len": chord_len}

    midpoint = 0.5 * (p0 + p1)
    normal = _unit_vec(np.array([-chord[1], chord[0]], dtype=float))
    if float(np.linalg.norm(normal)) <= 0:
        return None, {"reason": "zero normal", "chord_len": chord_len}

    initial_step = max(1e-10, float(initial_step_factor) * chord_len)
    max_step_eff = max(float(max_step), 0.5 * chord_len)

    candidates: list[tuple[float, float, np.ndarray]] = []

    for sign in (1.0, -1.0):
        direction = sign * normal
        q = bracketed_root_on_ray(
            origin=midpoint,
            direction=direction,
            level=level,
            cache=cache,
            initial_step=initial_step,
            grow=grow,
            max_step=max_step_eff,
            bisect_steps=bisect_steps,
        )
        if q is None:
            continue

        q = np.asarray(q, dtype=float)
        if upper_only and q[1] < -1e-10:
            continue

        s = float(np.dot(q - midpoint, normal))
        d = float(np.linalg.norm(q - midpoint))
        candidates.append((d, s, q))

    if not candidates:
        return None, {
            "reason": "nearest bracket failed",
            "chord_len": chord_len,
            "max_step": max_step_eff,
        }

    candidates.sort(key=lambda item: item[0])
    sagitta, s, q = candidates[0]
    sagitta_factor = sagitta / max(chord_len, 1e-300)

    v0 = p0 - q
    v1 = p1 - q
    angle_at_q = _angle_between_degrees(v0, v1)
    turn_deg = 180.0 - angle_at_q

    return q, {
        "reason": "ok-nearest-fast",
        "chord_len": chord_len,
        "sagitta": sagitta,
        "sagitta_factor": sagitta_factor,
        "s": float(s),
        "roots": len(candidates),
        "angle_at_p": angle_at_q,
        "turn_deg": turn_deg,
    }


def choose_bisector_root(
    roots: list[tuple[float, np.ndarray]],
    center: np.ndarray,
    normal: np.ndarray,
    chord_len: float,
    root_choice: str,
    upper_only: bool = True,
    y_tol: float = 1e-10,
) -> tuple[float, np.ndarray] | None:
    """
    Choose the contour root to insert on a perpendicular line.

    For upper-half construction, the most useful default is "highest": among
    roots in y>=0, choose the one with largest y. This follows the upper
    exterior arc and avoids accidentally choosing a lower/lower-return branch.
    """
    if not roots:
        return None

    candidates = []
    for s, p in roots:
        p = np.asarray(p, dtype=float)
        if upper_only and p[1] < -abs(y_tol):
            continue
        d_mid = float(np.linalg.norm(p - center))
        candidates.append((float(s), p, d_mid))

    if not candidates:
        return None

    if root_choice == "nearest":
        candidates.sort(key=lambda item: abs(item[0]))
        s, p, _ = candidates[0]
        return s, p

    if root_choice == "positive-nearest":
        positive = [item for item in candidates if item[0] >= -1e-14]
        if positive:
            positive.sort(key=lambda item: abs(item[0]))
            s, p, _ = positive[0]
            return s, p
        candidates.sort(key=lambda item: abs(item[0]))
        s, p, _ = candidates[0]
        return s, p

    if root_choice == "outermost":
        candidates.sort(key=lambda item: item[0], reverse=True)
        s, p, _ = candidates[0]
        return s, p

    # Default: highest upper-half point.
    candidates.sort(key=lambda item: (item[1][1], item[0]), reverse=True)
    s, p, _ = candidates[0]
    return s, p


def find_bisector_point(
    p0: np.ndarray,
    p1: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    span_factor: float,
    min_span: float,
    max_span: float,
    samples: int,
    bisect_steps: int,
    root_choice: str,
    upper_only: bool,
    stats: BisectorStats | None = None,
) -> tuple[np.ndarray | None, dict]:
    """
    For an edge p0--p1, intersect the level set with its perpendicular bisector.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    midpoint = 0.5 * (p0 + p1)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))

    if chord_len <= 0:
        return None, {"reason": "zero chord", "chord_len": chord_len}

    # Left normal for p0 -> p1. For the initial left-to-right real-axis chord
    # this is +y, exactly what we want for upper-half construction.
    normal = np.array([-chord[1], chord[0]], dtype=float)
    normal = _unit_vec(normal)

    # Prefer the normal that has positive y where possible. This stabilizes the
    # upper-half branch choice for strongly curved / almost vertical segments.
    if upper_only and normal[1] < 0:
        normal = -normal

    span = min(float(max_span), max(float(min_span), float(span_factor) * chord_len))

    roots: list[tuple[float, np.ndarray]] = []
    for _ in range(8):
        roots = line_level_roots(
            center=midpoint,
            normal=normal,
            level=level,
            cache=cache,
            span=span,
            samples=samples,
            bisect_steps=bisect_steps,
        )
        if roots or span >= float(max_span) * 0.999:
            break
        span = min(float(max_span), span * 2.0)

    if stats is not None:
        if len(roots) == 0:
            stats.no_root += 1
        if len(roots) > 1:
            stats.multi_root += 1

    chosen = choose_bisector_root(
        roots=roots,
        center=midpoint,
        normal=normal,
        chord_len=chord_len,
        root_choice=root_choice,
        upper_only=upper_only,
    )

    if chosen is None:
        return None, {
            "reason": "no usable root",
            "chord_len": chord_len,
            "span": span,
            "roots": len(roots),
        }

    s, p = chosen
    sagitta = float(np.linalg.norm(p - midpoint))
    sagitta_factor = sagitta / max(chord_len, 1e-300)

    v0 = p0 - p
    v1 = p1 - p
    angle_at_p = _angle_between_degrees(v0, v1)
    turn_deg = 180.0 - angle_at_p

    return p, {
        "reason": "ok",
        "chord_len": chord_len,
        "span": span,
        "roots": len(roots),
        "s": float(s),
        "sagitta": sagitta,
        "sagitta_factor": sagitta_factor,
        "angle_at_p": angle_at_p,
        "turn_deg": turn_deg,
    }


def bisector_segment_is_straight_enough(
    p0: np.ndarray,
    p1: np.ndarray,
    q: np.ndarray,
    info: dict,
    level: float,
    cache: PointPotentialCache,
    max_sagitta_factor: float,
    max_turn_degrees: float,
    min_segment_length: float,
    confirm: bool,
    confirm_fractions: tuple[float, ...],
    span_factor: float,
    min_span: float,
    max_span: float,
    samples: int,
    bisect_steps: int,
    root_choice: str,
    upper_only: bool,
    stats: BisectorStats,
) -> tuple[bool, str]:
    chord_len = float(info.get("chord_len", np.linalg.norm(np.asarray(p1) - np.asarray(p0))))

    if chord_len <= float(min_segment_length):
        return True, "min segment length"

    if float(info.get("sagitta_factor", np.inf)) > float(max_sagitta_factor):
        return False, f"sagitta {info.get('sagitta_factor'):.4g}"

    if float(info.get("turn_deg", np.inf)) > float(max_turn_degrees):
        return False, f"turn {info.get('turn_deg'):.4g} deg"

    if not confirm:
        return True, "midpoint ok"

    # Extra safety: before accepting a segment, test a few additional normal
    # slices along the same chord. This catches wrinkles that the exact midpoint
    # would miss.
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 0:
        return True, "degenerate"

    normal = np.array([-chord[1], chord[0]], dtype=float)
    normal = _unit_vec(normal)
    if upper_only and normal[1] < 0:
        normal = -normal

    span = min(float(max_span), max(float(min_span), float(span_factor) * chord_len))

    for frac in confirm_fractions:
        frac = float(frac)
        if abs(frac - 0.5) < 1e-12:
            continue

        center = p0 + frac * chord
        roots = line_level_roots(
            center=center,
            normal=normal,
            level=level,
            cache=cache,
            span=span,
            samples=samples,
            bisect_steps=bisect_steps,
        )
        chosen = choose_bisector_root(
            roots=roots,
            center=center,
            normal=normal,
            chord_len=chord_len,
            root_choice=root_choice,
            upper_only=upper_only,
        )

        if chosen is None:
            return False, f"confirm no root at t={frac:g}"

        _, p = chosen
        dev = float(np.linalg.norm(p - center))
        dev_factor = dev / max(chord_len, 1e-300)
        if dev_factor > float(max_sagitta_factor):
            return False, f"confirm sagitta {dev_factor:.4g} at t={frac:g}"

    return True, "confirmed"


def build_upper_contour_by_bisectors(
    level: float,
    cache: PointPotentialCache,
    max_depth: int = 26,
    min_segment_length: float = 1e-5,
    max_points: int = 50000,
    max_sagitta_factor: float = 0.035,
    max_turn_degrees: float = 8.0,
    span_factor: float = 4.0,
    min_span: float = 1e-6,
    max_span: float = 4.0,
    line_samples: int = 81,
    bisect_steps: int = 28,
    root_choice: str = "highest",
    confirm: bool = True,
    confirm_fractions: tuple[float, ...] = (0.25, 0.75),
    progress_every: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build the upper half of G(c)=level by recursive perpendicular-bisector
    refinement.
    """
    left = find_trace_start_on_ray(level=level, cache=cache, start_angle=math.pi)
    right = find_trace_start_on_ray(level=level, cache=cache, start_angle=0.0)
    left[1] = 0.0
    right[1] = 0.0

    stats = BisectorStats()

    if verbose:
        print(
            f"    bisector endpoints: left={left[0]:+.10f} {left[1]:+.10f}i, "
            f"right={right[0]:+.10f} {right[1]:+.10f}i",
            flush=True,
        )

    def recurse(p0: np.ndarray, p1: np.ndarray, depth: int) -> list[np.ndarray]:
        stats.processed += 1
        stats.max_depth_seen = max(stats.max_depth_seen, int(depth))

        if progress_every > 0 and stats.processed % int(progress_every) == 0 and verbose:
            print(
                f"    bisector G={level:g}: processed={stats.processed:,}, "
                f"accepted={stats.accepted:,}, subdivided={stats.subdivided:,}, "
                f"depth≤{stats.max_depth_seen}, cache={len(cache.cache):,}",
                flush=True,
            )

        if stats.accepted > int(max_points):
            raise RuntimeError(
                f"Bisector contour produced too many segments at G={level:g}. "
                f"accepted>{max_points:,}. Relax tolerance or increase --bisector-max-points."
            )

        q, info = find_bisector_point_outward_fast(
            p0=p0,
            p1=p1,
            level=level,
            cache=cache,
            initial_step_factor=0.10,
            max_step=max_span,
            grow=1.8,
            bisect_steps=bisect_steps,
        )

        if q is None:
            q, info = find_bisector_point(
                p0=p0,
                p1=p1,
                level=level,
                cache=cache,
                span_factor=span_factor,
                min_span=min_span,
                max_span=max_span,
                samples=line_samples,
                bisect_steps=bisect_steps,
                root_choice=root_choice,
                upper_only=True,
                stats=stats,
            )

        chord_len = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))

        if q is None:
            if depth >= int(max_depth) or chord_len <= float(min_segment_length):
                stats.forced_accept += 1
                stats.accepted += 1
                return [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]

            raise RuntimeError(
                f"Bisector could not find contour point at G={level:g}, depth={depth}, "
                f"p0={p0}, p1={p1}, info={info}. Try larger --bisector-max-span "
                f"or --bisector-line-samples."
            )

        straight, reason = bisector_segment_is_straight_enough(
            p0=p0,
            p1=p1,
            q=q,
            info=info,
            level=level,
            cache=cache,
            max_sagitta_factor=max_sagitta_factor,
            max_turn_degrees=max_turn_degrees,
            min_segment_length=min_segment_length,
            confirm=confirm,
            confirm_fractions=confirm_fractions,
            span_factor=span_factor,
            min_span=min_span,
            max_span=max_span,
            samples=line_samples,
            bisect_steps=bisect_steps,
            root_choice=root_choice,
            upper_only=True,
            stats=stats,
        )

        if straight or depth >= int(max_depth):
            if depth >= int(max_depth) and not straight:
                stats.forced_accept += 1
            stats.accepted += 1
            return [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]

        stats.subdivided += 1
        left_part = recurse(p0, q, depth + 1)
        right_part = recurse(q, p1, depth + 1)
        return left_part[:-1] + right_part

    upper = recurse(left, right, 0)

    cleaned: list[np.ndarray] = []
    eps = max(1e-14, float(min_segment_length) * 1e-3)
    for p in upper:
        p = np.asarray(p, dtype=float)
        if p[1] < 0 and p[1] > -1e-10:
            p = p.copy()
            p[1] = 0.0
        if not cleaned or np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)

    upper_arr = np.asarray(cleaned, dtype=float)
    upper_arr[0] = left
    upper_arr[-1] = right

    if len(upper_arr) > int(max_points):
        raise RuntimeError(
            f"Bisector contour produced {len(upper_arr):,} upper-half points, "
            f"above --bisector-max-points={max_points:,}. Relax tolerance or increase the limit."
        )

    if verbose:
        print(
            f"  bisector G={level:g}: upper points={len(upper_arr):,}, "
            f"upper length≈{polyline_length_open(upper_arr):.8g}, "
            f"processed={stats.processed:,}, accepted={stats.accepted:,}, "
            f"subdivided={stats.subdivided:,}, forced={stats.forced_accept:,}, "
            f"multi-root slices={stats.multi_root:,}, no-root slices={stats.no_root:,}, "
            f"max depth={stats.max_depth_seen}, cache={len(cache.cache):,}",
            flush=True,
        )

    return upper_arr


def trace_potential_contour_bisector(
    level: float,
    cache: PointPotentialCache,
    max_depth: int = 26,
    min_segment_length: float = 1e-5,
    max_points: int = 50000,
    max_sagitta_factor: float = 0.035,
    max_turn_degrees: float = 8.0,
    span_factor: float = 4.0,
    min_span: float = 1e-6,
    max_span: float = 4.0,
    line_samples: int = 81,
    bisect_steps: int = 28,
    root_choice: str = "highest",
    confirm: bool = True,
    confirm_fractions: tuple[float, ...] = (0.25, 0.75),
    progress_every: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """Full closed contour from adaptive upper-half construction + mirror."""
    upper = build_upper_contour_by_bisectors(
        level=level,
        cache=cache,
        max_depth=max_depth,
        min_segment_length=min_segment_length,
        max_points=max_points,
        max_sagitta_factor=max_sagitta_factor,
        max_turn_degrees=max_turn_degrees,
        span_factor=span_factor,
        min_span=min_span,
        max_span=max_span,
        line_samples=line_samples,
        bisect_steps=bisect_steps,
        root_choice=root_choice,
        confirm=confirm,
        confirm_fractions=confirm_fractions,
        progress_every=progress_every,
        verbose=verbose,
    )

    # Upper was built left -> right. For CCW full loop, use right -> left upper,
    # then mirror back left -> right on the lower half.
    upper_right_to_left = upper[::-1].copy()
    full = mirror_upper_half_contour_to_full(upper_right_to_left)

    if verbose:
        print(
            f"  bisector mirrored G={level:g}: full points={len(full):,}, "
            f"full length≈{polyline_length_closed(full):.8g}",
            flush=True,
        )

    return ensure_ccw(full)



# ============================================================
# Outward-to-inward log(G) contour marching
# ============================================================

def make_log_march_levels(start_level: float, target_level: float, log10_step: float) -> np.ndarray:
    start_level = float(start_level)
    target_level = float(target_level)
    if target_level > start_level:
        raise ValueError("Contour marching expects target_level <= start_level.")
    if target_level == start_level:
        return np.array([start_level], dtype=float)
    step = max(1e-6, float(log10_step))
    n = max(1, int(math.ceil(abs(math.log10(target_level / start_level)) / step)))
    return np.exp(np.linspace(math.log(start_level), math.log(target_level), n + 1))


def upper_polyline_normals_inward(upper: np.ndarray, level: float, cache: PointPotentialCache) -> np.ndarray:
    """Normals for an upper-half contour ordered left -> right; choose side where G decreases."""
    pts = np.asarray(upper, dtype=float)
    n = len(pts)
    normals = np.zeros_like(pts)
    if n < 2:
        return normals

    for i in range(n):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]
        tn = float(np.linalg.norm(tangent))
        if tn <= 0:
            normals[i] = np.array([0.0, -1.0])
            continue
        tangent = tangent / tn
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        if i == 0:
            local_len = float(np.linalg.norm(pts[1] - pts[0]))
        elif i == n - 1:
            local_len = float(np.linalg.norm(pts[-1] - pts[-2]))
        else:
            local_len = 0.5 * (float(np.linalg.norm(pts[i] - pts[i - 1])) + float(np.linalg.norm(pts[i + 1] - pts[i])))
        eps = max(1e-8, min(1e-3, 0.05 * local_len))
        gp = cache.get(*(pts[i] + eps * normal))
        gm = cache.get(*(pts[i] - eps * normal))
        if gm < gp:
            normal = -normal
        normals[i] = normal

    # For the upper arc, endpoint inward normals should not point upward.
    if normals[0, 1] > 0:
        normals[0] *= -1
    if normals[-1, 1] > 0:
        normals[-1] *= -1
    return normals


def project_point_to_lower_level_along_normal(
    p: np.ndarray,
    normal: np.ndarray,
    new_level: float,
    cache: PointPotentialCache,
    initial_step: float,
    grow: float,
    max_step: float,
    bisect_steps: int,
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    normal = _unit_vec(normal)
    f0 = cache.get(p[0], p[1]) - float(new_level)
    if f0 <= 0:
        return p.copy()
    step = max(1e-12, float(initial_step))
    max_step = max(step, float(max_step))
    grow = max(1.05, float(grow))
    lo = 0.0
    hi = step
    while hi <= max_step * (1.0 + 1e-12):
        q = p + hi * normal
        if cache.get(q[0], q[1]) - float(new_level) <= 0:
            break
        lo = hi
        hi *= grow
    else:
        raise RuntimeError(f"Could not bracket lower contour along normal from p={p}, new_level={new_level:g}, max_step={max_step:g}.")
    a = lo
    b = hi
    fa = cache.get(*(p + a * normal)) - float(new_level)
    for _ in range(int(bisect_steps)):
        m = 0.5 * (a + b)
        fm = cache.get(*(p + m * normal)) - float(new_level)
        if fa * fm <= 0:
            b = m
        else:
            a = m
            fa = fm
    return p + 0.5 * (a + b) * normal


def project_upper_contour_to_lower_level(
    upper: np.ndarray,
    old_level: float,
    new_level: float,
    cache: PointPotentialCache,
    initial_step_factor: float,
    max_step: float,
    grow: float,
    bisect_steps: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Project an already-known upper contour to a nearby lower level.

    v4: cheap directional projection. Normals are computed from the old contour
    and then kept fixed. For each point we try exactly what the idea suggests:
    move a small distance inward along that remembered normal and bisection-
    bracket the new contour. Only if that fails do we fall back to a full normal
    line search.

    This should keep cache growth sane when dlogG is small enough.
    """
    upper = np.asarray(upper, dtype=float)
    normals = upper_polyline_normals_inward(upper, old_level, cache)

    projected = []
    failures = 0
    fallback_line_searches = 0
    fallback_successes = 0
    total_bisect_projections = 0

    left_new = find_trace_start_on_ray(level=new_level, cache=cache, start_angle=math.pi)
    right_new = find_trace_start_on_ray(level=new_level, cache=cache, start_angle=0.0)
    left_new[1] = 0.0
    right_new[1] = 0.0

    for i, p in enumerate(upper):
        if i == 0:
            projected.append(left_new)
            continue
        if i == len(upper) - 1:
            projected.append(right_new)
            continue

        p = np.asarray(p, dtype=float)
        normal = normals[i]

        local_len = 0.5 * (
            float(np.linalg.norm(upper[i] - upper[i - 1])) +
            float(np.linalg.norm(upper[i + 1] - upper[i]))
        )
        initial_step = max(1e-10, float(initial_step_factor) * local_len)

        q = bracketed_root_on_ray(
            origin=p,
            direction=normal,
            level=new_level,
            cache=cache,
            initial_step=initial_step,
            grow=grow,
            max_step=max_step,
            bisect_steps=bisect_steps,
        )

        if q is not None and q[1] >= -1e-8:
            total_bisect_projections += 1
        else:
            # Fallback only. This should be rare. It catches bad normal
            # orientation/twisty regions without making every point expensive.
            fallback_line_searches += 1
            roots = line_level_roots(
                center=p,
                normal=normal,
                level=new_level,
                cache=cache,
                span=max_step,
                samples=31,
                bisect_steps=bisect_steps,
            )
            roots = [(s, r) for (s, r) in roots if r[1] >= -1e-8]
            if roots:
                _, q = min(roots, key=lambda item: abs(item[0]))
                fallback_successes += 1
            else:
                failures += 1
                q = p.copy()

        q = np.asarray(q, dtype=float)
        if q[1] < 0 and q[1] > -1e-8:
            q = q.copy()
            q[1] = 0.0
        if q[1] < -1e-8:
            failures += 1
            q = p.copy()

        projected.append(q)

    arr = np.asarray(projected, dtype=float)
    arr[0] = left_new
    arr[-1] = right_new

    if verbose:
        print(
            f"    projected {old_level:g} -> {new_level:g}: "
            f"points={len(arr):,}, failures={failures}, "
            f"ray projections={total_bisect_projections}, "
            f"fallback line searches={fallback_line_searches}, "
            f"fallback successes={fallback_successes}, "
            f"cache={len(cache.cache):,}",
            flush=True,
        )

    return arr


def refine_upper_contour_edges(
    upper: np.ndarray,
    level: float,
    cache: PointPotentialCache,
    max_depth: int,
    min_segment_length: float,
    max_points: int,
    max_sagitta_factor: float,
    max_turn_degrees: float,
    span_factor: float,
    min_span: float,
    max_span: float,
    line_samples: int,
    bisect_steps: int,
    root_choice: str,
    confirm: bool,
    progress_every: int,
    verbose: bool,
) -> np.ndarray:
    upper = np.asarray(upper, dtype=float)
    if len(upper) < 2:
        return upper.copy()
    stats = BisectorStats()

    def recurse(p0: np.ndarray, p1: np.ndarray, depth: int) -> list[np.ndarray]:
        stats.processed += 1
        stats.max_depth_seen = max(stats.max_depth_seen, int(depth))
        if progress_every > 0 and stats.processed % int(progress_every) == 0 and verbose:
            print(f"    refine G={level:g}: processed={stats.processed:,}, accepted={stats.accepted:,}, subdivided={stats.subdivided:,}, depth≤{stats.max_depth_seen}, cache={len(cache.cache):,}", flush=True)
        chord_len = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        if chord_len <= float(min_segment_length) or depth >= int(max_depth):
            if depth >= int(max_depth) and chord_len > float(min_segment_length):
                stats.forced_accept += 1
            stats.accepted += 1
            return [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]
        # Marching refinement is local. The correct root is the nearest
        # perpendicular-bisector crossing, not necessarily the highest/+y one.
        # Forcing +y here can jump to another branch and corrupt point order.
        q, info = find_bisector_point_nearest_fast(
            p0=p0,
            p1=p1,
            level=level,
            cache=cache,
            initial_step_factor=0.10,
            max_step=max_span,
            grow=1.8,
            bisect_steps=bisect_steps,
            upper_only=True,
        )

        if q is None:
            q, info = find_bisector_point(
                p0, p1, level, cache,
                span_factor, min_span, max_span,
                line_samples, bisect_steps, "nearest", True, stats
            )
        if q is None:
            # Already have a projected contour. If a local bisector misses, accept this segment
            # unless it is very long; this avoids geometric-midpoint drift off the contour.
            if chord_len <= 10.0 * float(min_segment_length) or depth + 1 >= int(max_depth):
                stats.forced_accept += 1; stats.accepted += 1
                return [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]
            mid = 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
            stats.subdivided += 1
            left_part = recurse(p0, mid, depth + 1)
            right_part = recurse(mid, p1, depth + 1)
            return left_part[:-1] + right_part
        straight, _reason = bisector_segment_is_straight_enough(
            p0, p1, q, info, level, cache, max_sagitta_factor, max_turn_degrees,
            min_segment_length, confirm, (0.25, 0.75), span_factor, min_span,
            max_span, line_samples, bisect_steps, root_choice, True, stats)
        if straight:
            stats.accepted += 1
            return [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]
        stats.subdivided += 1
        left_part = recurse(p0, q, depth + 1)
        right_part = recurse(q, p1, depth + 1)
        return left_part[:-1] + right_part

    pieces = []
    for i in range(len(upper) - 1):
        part = recurse(upper[i], upper[i + 1], 0)
        pieces.extend(part if i == 0 else part[1:])
        if len(pieces) > int(max_points):
            raise RuntimeError(f"Refined contour at G={level:g} exceeded --march-max-points={max_points:,}.")

    cleaned = []
    eps = max(1e-14, float(min_segment_length) * 1e-3)
    for p in pieces:
        p = np.asarray(p, dtype=float)
        if p[1] < 0 and p[1] > -1e-10:
            p = p.copy(); p[1] = 0.0
        if not cleaned or np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)
    refined = np.asarray(cleaned, dtype=float)
    if verbose:
        print(f"    refined G={level:g}: points {len(upper):,} -> {len(refined):,}, processed={stats.processed:,}, accepted={stats.accepted:,}, subdivided={stats.subdivided:,}, forced={stats.forced_accept:,}, multi-root slices={stats.multi_root:,}, no-root slices={stats.no_root:,}, max depth={stats.max_depth_seen}, cache={len(cache.cache):,}", flush=True)
    return refined



def build_upper_contour_seed_by_trace(
    level: float,
    cache: PointPotentialCache,
    seed_step: float,
    circle_samples: int,
    bisect_steps: int,
    max_points: int,
    progress_every: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a smooth outer seed contour by ordinary circle tracing, then return
    only the upper half ordered left -> right.
    """
    left_target = find_trace_start_on_ray(level=level, cache=cache, start_angle=math.pi)
    left_target[1] = 0.0

    if verbose:
        print(
            f"    trace seed left real-axis target c = "
            f"{left_target[0]:+.10f} {left_target[1]:+.10f}i",
            flush=True,
        )

    upper_right_to_left = trace_potential_contour(
        level=level,
        cache=cache,
        step_radius=float(seed_step),
        start_angle=0.0,
        direction="ccw",
        circle_samples=circle_samples,
        bisect_steps=bisect_steps,
        max_points=max_points,
        min_points_before_close=20,
        close_factor=1.5,
        max_radius_adjust=8,
        radius_shrink=0.65,
        verbose=verbose,
        progress_every=progress_every,
        quality_check=False,
        stop_at_point=left_target,
        stop_point_close_factor=2.5,
        stop_point_min_x=-0.25,
        stop_label="outer seed left real-axis target",
    )

    upper_right_to_left = np.asarray(upper_right_to_left, dtype=float)
    if len(upper_right_to_left) >= 2:
        upper_right_to_left[-1] = left_target

    upper = upper_right_to_left[::-1].copy()
    upper[0, 1] = 0.0
    upper[-1, 1] = 0.0

    if verbose:
        print(
            f"    trace seed G={level:g}: upper points={len(upper):,}, "
            f"upper length≈{polyline_length_open(upper):.8g}",
            flush=True,
        )

    return upper


def trace_potential_contour_march(
    target_level: float,
    start_level: float,
    cache: PointPotentialCache,
    log10_step: float = 0.10,
    max_depth: int = 14,
    min_segment_length: float = 3e-5,
    max_points: int = 50000,
    max_sagitta_factor: float = 0.12,
    max_turn_degrees: float = 30.0,
    span_factor: float = 2.0,
    min_span: float = 1e-6,
    max_span: float = 0.15,
    line_samples: int = 31,
    bisect_steps: int = 28,
    root_choice: str = "nearest",
    confirm: bool = False,
    project_initial_step_factor: float = 1.0,
    project_max_step: float = 0.25,
    project_grow: float = 1.6,
    progress_every: int = 1000,
    verbose: bool = True,
    seed_mode: str = "trace",
    seed_step: float = 0.035,
    seed_max_span: float = 4.0,
    save_contours_dir: str | None = None,
    simplify_after_step: bool = True,
    simplify_sagitta_factor: float = 0.02,
    simplify_turn_degrees: float = 8.0,
    simplify_max_merged_segment_length: float = 0.12,
    simplify_min_keep_points: int = 100,
    simplify_max_passes: int = 12,
    simplify_working_contour: bool = False,
    save_simplified_contours: bool = True,
) -> np.ndarray:
    target_level = float(target_level)
    start_level = float(start_level)
    if target_level > start_level:
        raise ValueError("Marching expects target_level <= start_level.")
    if verbose:
        print(
            f"    march seed: constructing upper G={start_level:g} "
            f"with seed_mode={seed_mode}",
            flush=True,
        )

    if seed_mode == "bisector":
        upper = build_upper_contour_by_bisectors(
            level=start_level, cache=cache,
            max_depth=max(10, min(int(max_depth), 16)),
            min_segment_length=max(float(min_segment_length), 8e-5),
            max_points=max_points,
            max_sagitta_factor=max(float(max_sagitta_factor), 0.12),
            max_turn_degrees=max(float(max_turn_degrees), 30.0),
            span_factor=max(float(span_factor), 2.0),
            min_span=min_span, max_span=max(float(max_span), float(seed_max_span)),
            line_samples=max(25, min(int(line_samples), 61)),
            bisect_steps=bisect_steps, root_choice=root_choice,
            confirm=False, progress_every=progress_every, verbose=verbose)
    else:
        upper = build_upper_contour_seed_by_trace(
            level=start_level,
            cache=cache,
            seed_step=seed_step,
            circle_samples=max(72, int(line_samples) * 2),
            bisect_steps=bisect_steps,
            max_points=max_points,
            progress_every=progress_every,
            verbose=verbose,
        )

    # Keep the marching state at full working resolution by default.
    # Simplifying the state itself can destroy predictor normals and cause
    # later branch/order corruption. We simplify only the saved/exported copy
    # unless --march-simplify-working-contour is explicitly requested.
    if simplify_after_step and simplify_working_contour:
        upper = simplify_upper_contour_for_marching(
            upper,
            max_sagitta_factor=simplify_sagitta_factor,
            max_turn_degrees=simplify_turn_degrees,
            max_merged_segment_length=simplify_max_merged_segment_length,
            min_keep_points=simplify_min_keep_points,
            max_passes=simplify_max_passes,
            verbose=verbose,
            label=f"seed working G={start_level:g}",
        )

    if save_contours_dir:
        upper_save = upper
        if simplify_after_step and save_simplified_contours and not simplify_working_contour:
            upper_save = simplify_upper_contour_for_marching(
                upper,
                max_sagitta_factor=simplify_sagitta_factor,
                max_turn_degrees=simplify_turn_degrees,
                max_merged_segment_length=simplify_max_merged_segment_length,
                min_keep_points=simplify_min_keep_points,
                max_passes=simplify_max_passes,
                verbose=verbose,
                label=f"seed export G={start_level:g}",
            )
        seed_full = mirror_upper_half_contour_to_full(upper_save[::-1].copy())
        save_contour_json(start_level, ensure_ccw(seed_full), save_contours_dir, verbose=verbose)

    levels = make_log_march_levels(start_level, target_level, log10_step)
    if verbose:
        print(f"    marching {len(levels)-1} log-spaced steps: {start_level:g} -> {target_level:g}", flush=True)
    old_level = start_level
    for k, new_level in enumerate(levels[1:], start=1):
        if verbose:
            print(f"  march step {k}/{len(levels)-1}: G {old_level:g} -> {new_level:g}", flush=True)
        upper = project_upper_contour_to_lower_level(
            upper, old_level, float(new_level), cache,
            initial_step_factor=project_initial_step_factor,
            max_step=project_max_step, grow=project_grow,
            bisect_steps=bisect_steps, verbose=verbose)
        upper = refine_upper_contour_edges(
            upper, float(new_level), cache,
            max_depth=max_depth, min_segment_length=min_segment_length,
            max_points=max_points, max_sagitta_factor=max_sagitta_factor,
            max_turn_degrees=max_turn_degrees, span_factor=span_factor,
            min_span=min_span, max_span=max_span, line_samples=line_samples,
            bisect_steps=bisect_steps, root_choice=root_choice,
            confirm=confirm, progress_every=progress_every, verbose=verbose)
        if simplify_after_step and simplify_working_contour:
            upper = simplify_upper_contour_for_marching(
                upper,
                max_sagitta_factor=simplify_sagitta_factor,
                max_turn_degrees=simplify_turn_degrees,
                max_merged_segment_length=simplify_max_merged_segment_length,
                min_keep_points=simplify_min_keep_points,
                max_passes=simplify_max_passes,
                verbose=verbose,
                label=f"working G={float(new_level):g}",
            )

        old_level = float(new_level)
        if verbose:
            order_stats = polyline_order_diagnostics(upper)
            if not order_stats.get("ok", True):
                print(
                    f"    WARNING order check G={float(new_level):g}: "
                    f"edge median={order_stats['edge_median']:.4g}, "
                    f"edge p99={order_stats['edge_p99']:.4g}, "
                    f"ratio median={order_stats['ratio_median']:.4g}, "
                    f"ratio p99={order_stats['ratio_p99']:.4g}, "
                    f"ratio max={order_stats['ratio_max']:.4g}. "
                    f"Likely branch/order corruption; reduce --march-log10-step or keep more working contour points.",
                    flush=True,
                )

        if save_contours_dir:
            upper_save = upper
            if simplify_after_step and save_simplified_contours and not simplify_working_contour:
                upper_save = simplify_upper_contour_for_marching(
                    upper,
                    max_sagitta_factor=simplify_sagitta_factor,
                    max_turn_degrees=simplify_turn_degrees,
                    max_merged_segment_length=simplify_max_merged_segment_length,
                    min_keep_points=simplify_min_keep_points,
                    max_passes=simplify_max_passes,
                    verbose=verbose,
                    label=f"export G={float(new_level):g}",
                )
            step_full = mirror_upper_half_contour_to_full(upper_save[::-1].copy())
            save_contour_json(float(new_level), ensure_ccw(step_full), save_contours_dir, verbose=verbose)


    full = mirror_upper_half_contour_to_full(upper[::-1].copy())
    if verbose:
        print(f"  march mirrored G={target_level:g}: upper points={len(upper):,}, full points={len(full):,}, full length≈{polyline_length_closed(full):.8g}", flush=True)
    return ensure_ccw(full)


# ============================================================
# Contour extraction
# ============================================================

def extract_primary_contour(x: np.ndarray, y: np.ndarray, G: np.ndarray, level: float) -> np.ndarray:
    """
    Extract the longest contour segment G(x,y)=level.
    """
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, G, levels=[float(level)])

    try:
        segs = cs.allsegs[0]
    except Exception:
        segs = []

    plt.close(fig)

    if not segs:
        raise RuntimeError(f"No contour found at level {level:g}")

    best = None
    best_len = -1.0
    for seg in segs:
        seg = np.asarray(seg, dtype=float)
        if len(seg) < 4:
            continue
        L = polyline_length_closed(seg)
        if L > best_len:
            best = seg
            best_len = L

    if best is None:
        raise RuntimeError(f"No usable contour found at level {level:g}")

    return ensure_ccw(best)


def make_contour_levels(gmin: float, gmax: float, nlevels: int, density_power: float) -> np.ndarray:
    """
    Levels in log-space, optionally denser near gmin.

    density_power = 1: ordinary geomspace.
    density_power > 1: more levels near gmin.
    """
    u = np.linspace(0.0, 1.0, int(nlevels))
    t = u ** max(1e-8, float(density_power))
    return np.exp(np.log(gmin) + (np.log(gmax) - np.log(gmin)) * t)


# ============================================================
# Transform / interpolation
# ============================================================

def transform_value(g: np.ndarray | float, mode: HeightMode):
    g = np.asarray(g, dtype=float)
    if mode == "G":
        return g
    if mode == "logG":
        return np.log10(g)
    if mode == "invG":
        return 1.0 / g
    raise ValueError(f"Unknown transform mode: {mode}")


def interpolate_G_on_points(xgrid: np.ndarray, ygrid: np.ndarray, Ggrid: np.ndarray, xy: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator(
        (ygrid, xgrid),
        Ggrid,
        bounds_error=False,
        fill_value=np.nan,
    )
    pts_yx = np.column_stack([xy[:, 1], xy[:, 0]])
    return interp(pts_yx)


# ============================================================
# Gmsh meshing
# ============================================================

@dataclass
class GmshMesh:
    xy: np.ndarray
    tris: np.ndarray


def add_gmsh_loop(gmsh, points: np.ndarray, mesh_size: float, name: str) -> int:
    """
    Add a closed polyline loop to gmsh.model.geo and return the curve loop tag.
    """
    point_tags = []
    for x, y in points:
        point_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(mesh_size)))

    line_tags = []
    n = len(point_tags)
    for i in range(n):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n]))

    loop = gmsh.model.geo.addCurveLoop(line_tags)
    return loop


def build_gmsh_annulus_mesh(
    outer_xy_ccw: np.ndarray,
    inner_xy_cw: np.ndarray,
    outer_mesh_size: float,
    inner_mesh_size: float,
    mesh_size_min: float,
    mesh_size_max: float,
    algorithm: int,
    optimize: bool,
    verbose: bool,
    save_msh: str | None = None,
) -> GmshMesh:
    try:
        import gmsh
    except Exception as exc:
        raise RuntimeError(
            "Could not import gmsh. Install it with: pip install gmsh\n"
            "On WSL/Linux you may also need: sudo apt install libglu1-mesa"
        ) from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.Algorithm", int(algorithm))
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)

        gmsh.model.add("mandelbrot_potential_annulus")

        outer_loop = add_gmsh_loop(gmsh, outer_xy_ccw, outer_mesh_size, "outer")
        inner_loop = add_gmsh_loop(gmsh, inner_xy_cw, inner_mesh_size, "inner")

        surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)

        if optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                gmsh.model.mesh.optimize()

        if save_msh:
            gmsh.write(save_msh)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        xy = coords[:, :2]

        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tris = []

        for elem_type, nodes in zip(elem_types, elem_node_tags):
            # Gmsh element type 2 is a 3-node triangle.
            if int(elem_type) != 2:
                continue

            arr = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            for a, b, c in arr:
                tris.append((tag_to_idx[int(a)], tag_to_idx[int(b)], tag_to_idx[int(c)]))

        if not tris:
            raise RuntimeError("Gmsh produced no linear triangle elements. Try --gmsh-algorithm 6 or reduce options.")

        tris = np.asarray(tris, dtype=int)
        return GmshMesh(xy=xy, tris=tris)

    finally:
        gmsh.finalize()


# ============================================================
# Plotly rendering
# ============================================================

def colorbar_title(mode: str) -> str:
    if mode == "G":
        return "G(c)"
    if mode == "logG":
        return "log₁₀ G(c)"
    if mode == "invG":
        return "1 / G(c)"
    return mode


def choose_visible_ring_indices(nrings: int, nvisible: int) -> list[int]:
    if nvisible <= 0 or nrings <= 0:
        return []
    nvisible = min(int(nvisible), int(nrings))
    return sorted(set(int(i) for i in np.linspace(0, nrings - 1, nvisible, dtype=int)))


def build_plotly_figure(
    xy: np.ndarray,
    tris: np.ndarray,
    Gv: np.ndarray,
    height_mode: HeightMode,
    color_mode: HeightMode,
    invert: bool,
    z_scale: float,
    colorscale: str,
    contours: list[tuple[float, np.ndarray]],
    visible_contours: int,
    contour_color: str,
    contour_width: float,
    contour_opacity: float,
    unlit: bool,
    hard_lighting: bool,
    flat_shading: bool,
    hover_mode: str,
    title: str,
):
    Gsafe = np.asarray(Gv, dtype=float)
    H = z_scale * transform_value(Gsafe, height_mode)
    if invert:
        H = -H

    intensity = transform_value(Gsafe, color_mode)

    vertices = np.column_stack([xy[:, 0], xy[:, 1], H])
    tris = orient_triangles_up(vertices, tris)

    if unlit:
        lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    elif hard_lighting:
        lighting = dict(ambient=0.5, diffuse=0.8, specular=0.15, roughness=0.7, fresnel=0.05)
    else:
        lighting = dict(ambient=0.92, diffuse=0.12, specular=0.0, roughness=1.0, fresnel=0.0)

    if hover_mode == "none":
        hovertemplate = None
        hoverinfo = "skip"
    elif hover_mode == "full":
        hovertemplate = (
            "Re(c)=%{x:.6f}<br>"
            "Im(c)=%{y:.6f}<br>"
            "G(c)=%{customdata:.6g}<br>"
            "height=%{z:.6f}<extra></extra>"
        )
        hoverinfo = None
    else:
        hovertemplate = "G(c)=%{customdata:.6g}<extra></extra>"
        hoverinfo = None

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=tris[:, 0],
        j=tris[:, 1],
        k=tris[:, 2],
        intensity=intensity,
        intensitymode="vertex",
        customdata=Gsafe,
        colorscale=colorscale,
        flatshading=flat_shading,
        showscale=True,
        colorbar=dict(title=colorbar_title(color_mode)),
        hovertemplate=hovertemplate,
        hoverinfo=hoverinfo,
        lighting=lighting,
        lightposition=dict(x=80, y=80, z=140),
        name="Gmsh potential mesh",
    ))

    visible_idx = choose_visible_ring_indices(len(contours), visible_contours)
    for idx in visible_idx:
        level, pts = contours[idx]
        g = np.full(len(pts) + 1, float(level))
        z = z_scale * transform_value(g, height_mode)
        if invert:
            z = -z

        closed = close_if_needed(pts)
        fig.add_trace(go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=z,
            mode="lines",
            line=dict(color=contour_color, width=contour_width),
            opacity=contour_opacity,
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        paper_bgcolor="white",
        scene=dict(
            bgcolor="white",
            aspectmode="data",
            xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=0.95)),
        ),
    )

    return fig, triangle_orientation_stats(vertices, tris)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Mandelbrot potential surface using Gmsh annulus meshing")

    # Potential grid / contour extraction
    parser.add_argument("--xmin", type=float, default=-2.5)
    parser.add_argument("--xmax", type=float, default=1.5)
    parser.add_argument("--ymin", type=float, default=-1.8)
    parser.add_argument("--ymax", type=float, default=1.8)
    parser.add_argument("--nx", type=int, default=1400)
    parser.add_argument("--ny", type=int, default=1200)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--escape-radius", type=float, default=1e6)
    parser.add_argument("--contour-source", choices=["trace", "bisector", "march", "uniform"], default="march",
                        help="trace = local contour follower; bisector = adaptive perpendicular-bisector contour; march = log(G) continuation from outer contour; uniform = old grid/matplotlib contour.")
    parser.add_argument("--trace-symmetry", action="store_true",
                        help="Trace only upper half and mirror using G(x+iy)=G(x-iy).")
    parser.add_argument("--trace-symmetry-target-factor", type=float, default=2.0,
                        help="Stop upper-half trace when within factor*radius of left real-axis target.")
    parser.add_argument("--trace-symmetry-min-x", type=float, default=-0.25,
                        help="Only allow symmetry stop after x is less than this value.")


    # Adaptive perpendicular-bisector contour construction
    parser.add_argument("--bisector-max-depth", type=int, default=26)
    parser.add_argument("--bisector-min-segment-length", type=float, default=1e-5)
    parser.add_argument("--bisector-max-points", type=int, default=50000)
    parser.add_argument("--bisector-max-sagitta-factor", type=float, default=0.035,
                        help="Accept a segment only if midpoint deviation/chord length is below this.")
    parser.add_argument("--bisector-max-turn-deg", type=float, default=8.0,
                        help="Accept a segment only if the midpoint turn angle is below this.")
    parser.add_argument("--bisector-span-factor", type=float, default=4.0,
                        help="Perpendicular search span as a multiple of chord length.")
    parser.add_argument("--bisector-min-span", type=float, default=1e-6)
    parser.add_argument("--bisector-max-span", type=float, default=4.0)
    parser.add_argument("--bisector-line-samples", type=int, default=81)
    parser.add_argument("--bisector-root-choice",
                        choices=["highest", "outermost", "nearest", "positive-nearest"],
                        default="highest")
    parser.add_argument("--no-bisector-confirm", action="store_true",
                        help="Disable extra quarter-point checks before accepting a segment.")
    parser.add_argument("--bisector-progress-every", type=int, default=1000)


    # Log(G) contour marching controls
    parser.add_argument("--march-log10-step", type=float, default=0.01,
                        help="Step size in log10(G) when marching from gmax down to gmin.")
    parser.add_argument("--march-max-depth", type=int, default=16)
    parser.add_argument("--march-min-segment-length", type=float, default=3e-5)
    parser.add_argument("--march-max-points", type=int, default=50000)
    parser.add_argument("--march-max-sagitta-factor", type=float, default=0.12)
    parser.add_argument("--march-max-turn-deg", type=float, default=15.0)
    parser.add_argument("--march-span-factor", type=float, default=2.0)
    parser.add_argument("--march-min-span", type=float, default=1e-6)
    parser.add_argument("--march-max-span", type=float, default=0.15)
    parser.add_argument("--march-line-samples", type=int, default=31)
    parser.add_argument("--march-root-choice",
                        choices=["highest", "outermost", "nearest", "positive-nearest"],
                        default="nearest")
    parser.add_argument("--march-confirm", action="store_true",
                        help="Enable extra quarter-point confirmation checks during march refinement.")
    parser.add_argument("--march-project-initial-step-factor", type=float, default=1.0)
    parser.add_argument("--march-project-max-step", type=float, default=0.20)
    parser.add_argument("--march-project-grow", type=float, default=1.6)
    parser.add_argument("--march-progress-every", type=int, default=1000)
    parser.add_argument("--no-march-simplify", action="store_true",
                        help="Disable geometric simplification of marched contours after each step.")
    parser.add_argument("--march-simplify-sagitta-factor", type=float, default=0.02,
                        help="Remove a point if its sagitta relative to the merged chord is below this value. Larger = more decimation.")
    parser.add_argument("--march-simplify-turn-deg", type=float, default=8.0,
                        help="Remove a point if its local direction-change angle is below this value. Larger = more decimation.")
    parser.add_argument("--march-simplify-max-merged-segment-length", type=float, default=0.12,
                        help="Do not simplify across a merged segment longer than this. Use <=0 to disable this cap. Larger = more decimation across smooth arcs.")
    parser.add_argument("--march-simplify-min-keep-points", type=int, default=100,
                        help="Never simplify an upper-half contour below this many points.")
    parser.add_argument("--march-simplify-max-passes", type=int, default=12,
                        help="Maximum simplification passes after each march step.")
    parser.add_argument("--march-simplify-working-contour", action="store_true",
                        help="Apply simplification to the internal marching state. Default is off; normally simplify only exported JSON contours.")
    parser.add_argument("--no-save-simplified-contours", "--save-raw-g-contours",
                        action="store_true",
                        dest="no_save_simplified_contours",
                        help="Save raw/full working G contour JSONs instead of simplified export copies.")

    parser.add_argument("--march-seed-mode", choices=["trace", "bisector"], default="trace",
                        help="How to build the smooth outer seed contour for marching.")
    parser.add_argument("--march-seed-step", type=float, default=0.035,
                        help="Circle-trace step radius for the smooth outer seed when --march-seed-mode trace.")
    parser.add_argument("--march-seed-max-span", type=float, default=4.0,
                        help="Max perpendicular search span for --march-seed-mode bisector.")

    # Potential annulus
    parser.add_argument("--gmin", type=float, default=1e-3)
    parser.add_argument("--gmax", type=float, default=0.25)

    # Direct contour tracing controls
    parser.add_argument("--trace-inner-step", type=float, default=0.004,
                        help="Circle radius used to walk the innermost contour.")
    parser.add_argument("--trace-outer-step", type=float, default=0.025,
                        help="Circle radius used to walk the outermost contour.")
    parser.add_argument("--trace-start-angle", type=float, default=0.0,
                        help="Start ray angle in radians; 0 starts on the positive real side.")
    parser.add_argument("--trace-direction", choices=["ccw", "cw"], default="ccw")
    parser.add_argument("--trace-circle-samples", type=int, default=72)
    parser.add_argument("--trace-bisect-steps", type=int, default=28)
    parser.add_argument("--trace-max-points", type=int, default=30000)
    parser.add_argument("--trace-min-points-before-close", type=int, default=80)
    parser.add_argument("--trace-close-factor", type=float, default=1.25)
    parser.add_argument("--trace-radius-shrink", type=float, default=0.65)
    parser.add_argument("--trace-max-radius-adjust", type=int, default=8)
    parser.add_argument("--trace-cache-digits", type=int, default=14)
    parser.add_argument("--trace-verbose", action="store_true")
    parser.add_argument("--trace-progress-every", type=int, default=250,
                        help="During direct contour tracing, print progress every N added points. Use 0 to disable.")
    parser.add_argument("--no-trace-quality-check", action="store_true",
                        help="Disable midpoint/turn quality checks and restart logic.")
    parser.add_argument("--trace-quality-every", type=int, default=1,
                        help="Run the segment quality check every N accepted points.")
    parser.add_argument("--trace-quality-start-after", type=int, default=5)
    parser.add_argument("--trace-quality-normal-span-factor", type=float, default=3.0,
                        help="Normal-line search span as a multiple of chord length.")
    parser.add_argument("--trace-quality-normal-samples", type=int, default=15)
    parser.add_argument("--trace-quality-max-sagitta-factor", type=float, default=0.30,
                        help="Fail if midpoint-normal correction distance exceeds this fraction of chord length.")
    parser.add_argument("--trace-quality-max-turn-deg", type=float, default=135.0,
                        help="Fail if consecutive segment turn angle exceeds this.")
    parser.add_argument("--trace-quality-action", choices=["rollback", "restart"], default="rollback",
                        help="rollback = rewind local suffix and continue smaller; restart = throw away entire contour.")
    parser.add_argument("--trace-quality-rollback-points", type=int, default=20,
                        help="For --trace-quality-action rollback, how many accepted points to rewind after failure.")
    parser.add_argument("--trace-quality-restart-shrink", type=float, default=0.5,
                        help="Factor by which the trace radius is reduced after a quality failure.")
    parser.add_argument("--trace-quality-max-restarts", type=int, default=8)
    parser.add_argument("--trace-quality-min-radius", type=float, default=1e-6)
    parser.add_argument("--trace-quality-max-local-failures", type=int, default=100,
                        help="For rollback mode, stop after this many quality failures on one contour.")
    parser.add_argument("--no-boundary-resample", action="store_true",
                        help="Use raw traced boundary points instead of resampling to inner/outer-boundary-points.")

    # Boundary contour resolution
    parser.add_argument("--inner-boundary-points", type=int, default=2500)
    parser.add_argument("--outer-boundary-points", type=int, default=800)
    parser.add_argument("--boundary-smooth-passes", type=int, default=0)
    parser.add_argument("--boundary-smooth-alpha", type=float, default=0.25)

    # Gmsh mesh control
    parser.add_argument("--inner-mesh-size", type=float, default=0.006)
    parser.add_argument("--outer-mesh-size", type=float, default=0.035)
    parser.add_argument("--mesh-size-min", type=float, default=0.003)
    parser.add_argument("--mesh-size-max", type=float, default=0.08)
    parser.add_argument("--gmsh-algorithm", type=int, default=6,
                        help="Gmsh 2D meshing algorithm. 6 is Frontal-Delaunay, often good.")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--gmsh-verbose", action="store_true")
    parser.add_argument("--save-msh", default="")

    # Mesh vertex scalar evaluation
    parser.add_argument("--mesh-g-source", choices=["direct", "interp"], default="direct",
                        help="How to compute G on Gmsh vertices. direct is cleaner near the boundary; interp is faster.")
    parser.add_argument("--point-batch-size", type=int, default=50000,
                        help="Batch size for --mesh-g-source direct.")
    parser.add_argument("--boundary-g-clamp-tol", type=float, default=0.0,
                        help="If >0, force vertices within this xy distance of the inner/outer boundary to exact gmin/gmax.")
    parser.add_argument("--print-g-stats", action="store_true")

    # Visual
    parser.add_argument("--height-mode", choices=["G", "logG", "invG"], default="logG")
    parser.add_argument("--color-mode", choices=["G", "logG", "invG"], default="logG")
    parser.add_argument("--invert", "--flip-z", action="store_true", dest="invert")
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--colorscale", default="Turbo")
    parser.add_argument("--unlit", action="store_true")
    parser.add_argument("--hard-lighting", action="store_true")
    parser.add_argument("--flat-shading", action="store_true")

    # Visible contour lines, independent of mesh
    parser.add_argument("--visible-contours", type=int, default=16)
    parser.add_argument("--save-g-contours-dir", default=str(DATA_ROOT / "G_contours"),
                        help="Directory for JSON dumps of successfully generated G contours. Use empty string to disable.")
    parser.add_argument("--no-save-g-contours", action="store_true",
                        help="Disable saving G contour JSON files.")
    parser.add_argument("--visible-contour-points", type=int, default=1200)
    parser.add_argument("--level-density-power", type=float, default=1.0)
    parser.add_argument("--contour-color", default="#000000")
    parser.add_argument("--contour-width", type=float, default=2.0)
    parser.add_argument("--contour-opacity", type=float, default=1.0)

    # Hover
    parser.add_argument("--hover-mode", choices=["g", "full", "none"], default="g")
    parser.add_argument("--show-coordinates", dest="hover_mode", action="store_const", const="full")
    parser.add_argument("--hide-coordinate-bubbles", dest="hover_mode", action="store_const", const="none")

    parser.add_argument("--report-normals", action="store_true")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "work"
            / "promote"
            / "mandelbrot"
            / "potential_surface_gmsh.html"
        ),
    )

    args = parser.parse_args()

    if args.gmin <= 0:
        raise ValueError("--gmin must be positive")
    if args.gmax <= args.gmin:
        raise ValueError("--gmax must be larger than --gmin")
    if args.z_scale <= 0:
        raise ValueError("--z-scale must be positive")

    contour_save_dir = None if args.no_save_g_contours else (args.save_g_contours_dir or None)

    x = y = G = escaped = None
    traced_contours: dict[float, np.ndarray] = {}

    if args.contour_source == "uniform":
        print("Computing Mandelbrot potential grid...")
        x, y, G, escaped = compute_mandelbrot_potential_grid(
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
            nx=args.nx,
            ny=args.ny,
            max_iter=args.max_iter,
            escape_radius=args.escape_radius,
        )

        print("Extracting boundary contours from uniform grid...")
        inner = extract_primary_contour(x, y, G, args.gmin)
        outer = extract_primary_contour(x, y, G, args.gmax)

    else:
        print("Tracing/constructing boundary contours directly...")
        trace_cache = PointPotentialCache(
            max_iter=args.max_iter,
            escape_radius=args.escape_radius,
            batch_size=args.point_batch_size,
            key_digits=args.trace_cache_digits,
        )

        def get_traced_contour(level: float) -> np.ndarray:
            level = float(level)
            if level not in traced_contours:
                step_r = trace_step_for_level(
                    level=level,
                    gmin=args.gmin,
                    gmax=args.gmax,
                    inner_step=args.trace_inner_step,
                    outer_step=args.trace_outer_step,
                )
                if args.contour_source == "bisector":
                    print(f"  constructing G={level:g} with adaptive perpendicular bisectors ...", flush=True)
                    traced_contours[level] = trace_potential_contour_bisector(
                        level=level,
                        cache=trace_cache,
                        max_depth=args.bisector_max_depth,
                        min_segment_length=args.bisector_min_segment_length,
                        max_points=args.bisector_max_points,
                        max_sagitta_factor=args.bisector_max_sagitta_factor,
                        max_turn_degrees=args.bisector_max_turn_deg,
                        span_factor=args.bisector_span_factor,
                        min_span=args.bisector_min_span,
                        max_span=args.bisector_max_span,
                        line_samples=args.bisector_line_samples,
                        bisect_steps=args.trace_bisect_steps,
                        root_choice=args.bisector_root_choice,
                        confirm=not args.no_bisector_confirm,
                        progress_every=args.bisector_progress_every,
                        verbose=True,
                    )
                elif args.contour_source == "march":
                    print(f"  constructing G={level:g} by log(G) contour marching from G={args.gmax:g} ...", flush=True)
                    traced_contours[level] = trace_potential_contour_march(
                        target_level=level,
                        start_level=args.gmax,
                        cache=trace_cache,
                        log10_step=args.march_log10_step,
                        max_depth=args.march_max_depth,
                        min_segment_length=args.march_min_segment_length,
                        max_points=args.march_max_points,
                        max_sagitta_factor=args.march_max_sagitta_factor,
                        max_turn_degrees=args.march_max_turn_deg,
                        span_factor=args.march_span_factor,
                        min_span=args.march_min_span,
                        max_span=args.march_max_span,
                        line_samples=args.march_line_samples,
                        bisect_steps=args.trace_bisect_steps,
                        root_choice=args.march_root_choice,
                        confirm=args.march_confirm,
                        project_initial_step_factor=args.march_project_initial_step_factor,
                        project_max_step=args.march_project_max_step,
                        project_grow=args.march_project_grow,
                        progress_every=args.march_progress_every,
                        verbose=True,
                        seed_mode=args.march_seed_mode,
                        seed_step=args.march_seed_step,
                        seed_max_span=args.march_seed_max_span,
                        save_contours_dir=contour_save_dir,
                        simplify_after_step=not args.no_march_simplify,
                        simplify_sagitta_factor=args.march_simplify_sagitta_factor,
                        simplify_turn_degrees=args.march_simplify_turn_deg,
                        simplify_max_merged_segment_length=args.march_simplify_max_merged_segment_length,
                        simplify_min_keep_points=args.march_simplify_min_keep_points,
                        simplify_max_passes=args.march_simplify_max_passes,
                        simplify_working_contour=args.march_simplify_working_contour,
                        save_simplified_contours=not args.no_save_simplified_contours,
                    )
                else:
                    print(f"  tracing G={level:g} with step radius {step_r:g} ...", flush=True)
                    tracer_fn = trace_potential_contour_symmetric if args.trace_symmetry else trace_potential_contour
                    extra_kwargs = {}
                    if args.trace_symmetry:
                        extra_kwargs.update(dict(
                            symmetry_target_factor=args.trace_symmetry_target_factor,
                            symmetry_min_x=args.trace_symmetry_min_x,
                        ))

                    traced_contours[level] = tracer_fn(
                        level=level,
                        cache=trace_cache,
                        step_radius=step_r,
                        start_angle=args.trace_start_angle,
                        direction=args.trace_direction,
                        circle_samples=args.trace_circle_samples,
                        bisect_steps=args.trace_bisect_steps,
                        max_points=args.trace_max_points,
                        min_points_before_close=args.trace_min_points_before_close,
                        close_factor=args.trace_close_factor,
                        max_radius_adjust=args.trace_max_radius_adjust,
                        radius_shrink=args.trace_radius_shrink,
                        verbose=True,
                        progress_every=args.trace_progress_every,
                        quality_check=not args.no_trace_quality_check,
                        quality_every=args.trace_quality_every,
                        quality_start_after=args.trace_quality_start_after,
                        quality_normal_span_factor=args.trace_quality_normal_span_factor,
                        quality_normal_samples=args.trace_quality_normal_samples,
                        quality_max_sagitta_factor=args.trace_quality_max_sagitta_factor,
                        quality_max_turn_degrees=args.trace_quality_max_turn_deg,
                        quality_action=args.trace_quality_action,
                        quality_rollback_points=args.trace_quality_rollback_points,
                        quality_restart_shrink=args.trace_quality_restart_shrink,
                        quality_max_restarts=args.trace_quality_max_restarts,
                        quality_min_radius=args.trace_quality_min_radius,
                        quality_max_local_failures=args.trace_quality_max_local_failures,
                        **extra_kwargs,
                    )
            if contour_save_dir and level in traced_contours:
                save_contour_json(level, traced_contours[level], contour_save_dir, verbose=True)

            return traced_contours[level]

        inner = get_traced_contour(args.gmin)
        outer = get_traced_contour(args.gmax)

    if not args.no_boundary_resample:
        inner = resample_closed_polyline_by_count(inner, args.inner_boundary_points)
        outer = resample_closed_polyline_by_count(outer, args.outer_boundary_points)

    inner = circular_smooth_ring(inner, args.boundary_smooth_passes, args.boundary_smooth_alpha)
    outer = circular_smooth_ring(outer, args.boundary_smooth_passes, args.boundary_smooth_alpha)

    outer_ccw = ensure_ccw(outer)
    inner_cw = ensure_cw(inner)

    outer_len = polyline_length_closed(outer_ccw)
    inner_len = polyline_length_closed(inner_cw)
    print(f"Outer boundary: {len(outer_ccw):,} points, length {outer_len:.6g}")
    print(f"Inner boundary: {len(inner_cw):,} points, length {inner_len:.6g}")
    if args.contour_source == "trace" and inner_len < 0.5 * outer_len:
        print("WARNING: traced inner contour is suspiciously short. Try smaller --trace-inner-step.")

    print("Meshing annulus with Gmsh...")
    mesh = build_gmsh_annulus_mesh(
        outer_xy_ccw=outer_ccw,
        inner_xy_cw=inner_cw,
        outer_mesh_size=args.outer_mesh_size,
        inner_mesh_size=args.inner_mesh_size,
        mesh_size_min=args.mesh_size_min,
        mesh_size_max=args.mesh_size_max,
        algorithm=args.gmsh_algorithm,
        optimize=not args.no_optimize,
        verbose=args.gmsh_verbose,
        save_msh=args.save_msh or None,
    )

    print(f"Gmsh mesh: {len(mesh.xy):,} vertices, {len(mesh.tris):,} triangles")

    if args.mesh_g_source == "direct":
        print("Directly evaluating G on mesh vertices...")
        Gv = compute_mandelbrot_potential_points(
            mesh.xy,
            max_iter=args.max_iter,
            escape_radius=args.escape_radius,
            batch_size=args.point_batch_size,
        )
    else:
        if x is None or y is None or G is None:
            print("No uniform grid available for interpolation; falling back to direct G evaluation.")
            Gv = compute_mandelbrot_potential_points(
                mesh.xy,
                max_iter=args.max_iter,
                escape_radius=args.escape_radius,
                batch_size=args.point_batch_size,
            )
        else:
            print("Interpolating G on mesh vertices...")
            Gv = interpolate_G_on_points(x, y, G, mesh.xy)

    bad = ~np.isfinite(Gv)
    if np.any(bad):
        print(f"Warning: {bad.sum()} mesh vertices had NaN G values; filling with gmin.")
        Gv[bad] = args.gmin

    if args.print_g_stats:
        print_g_stats("Raw mesh G", Gv)

    if args.boundary_g_clamp_tol > 0:
        Gv = clamp_boundary_g_values(
            mesh.xy,
            Gv,
            inner_boundary=inner,
            outer_boundary=outer,
            gmin=args.gmin,
            gmax=args.gmax,
            tol=args.boundary_g_clamp_tol,
        )

    # Vertices are supposed to live in the annulus gmin <= G <= gmax.
    # Clip small leaks due to polygonization / finite iteration.
    Gv = np.clip(Gv, args.gmin, args.gmax)

    if args.print_g_stats:
        print_g_stats("Clipped mesh G", Gv)

    print("Preparing visible contour overlays...")
    levels = make_contour_levels(args.gmin, args.gmax, args.visible_contours, args.level_density_power)
    contours = []

    for level in levels:
        level = float(level)

        if args.contour_source == "trace":
            # Reuse already-traced gmin/gmax contours when possible.
            if level in traced_contours:
                pts = traced_contours[level]
            else:
                step_r = trace_step_for_level(
                    level=level,
                    gmin=args.gmin,
                    gmax=args.gmax,
                    inner_step=args.trace_inner_step,
                    outer_step=args.trace_outer_step,
                )
                if 'trace_cache' not in locals():
                    trace_cache = PointPotentialCache(
                        max_iter=args.max_iter,
                        escape_radius=args.escape_radius,
                        batch_size=args.point_batch_size,
                        key_digits=args.trace_cache_digits,
                    )
                if args.contour_source == "bisector":
                    print(f"  constructing visible G={level:g} with adaptive perpendicular bisectors ...", flush=True)
                    pts = trace_potential_contour_bisector(
                        level=level,
                        cache=trace_cache,
                        max_depth=args.bisector_max_depth,
                        min_segment_length=args.bisector_min_segment_length,
                        max_points=args.bisector_max_points,
                        max_sagitta_factor=args.bisector_max_sagitta_factor,
                        max_turn_degrees=args.bisector_max_turn_deg,
                        span_factor=args.bisector_span_factor,
                        min_span=args.bisector_min_span,
                        max_span=args.bisector_max_span,
                        line_samples=args.bisector_line_samples,
                        bisect_steps=args.trace_bisect_steps,
                        root_choice=args.bisector_root_choice,
                        confirm=not args.no_bisector_confirm,
                        progress_every=0,
                        verbose=False,
                    )
                elif args.contour_source == "march":
                    print(f"  constructing visible G={level:g} by log(G) contour marching ...", flush=True)
                    pts = trace_potential_contour_march(
                        target_level=level,
                        start_level=args.gmax,
                        cache=trace_cache,
                        log10_step=args.march_log10_step,
                        max_depth=args.march_max_depth,
                        min_segment_length=args.march_min_segment_length,
                        max_points=args.march_max_points,
                        max_sagitta_factor=args.march_max_sagitta_factor,
                        max_turn_degrees=args.march_max_turn_deg,
                        span_factor=args.march_span_factor,
                        min_span=args.march_min_span,
                        max_span=args.march_max_span,
                        line_samples=args.march_line_samples,
                        bisect_steps=args.trace_bisect_steps,
                        root_choice=args.march_root_choice,
                        confirm=args.march_confirm,
                        project_initial_step_factor=args.march_project_initial_step_factor,
                        project_max_step=args.march_project_max_step,
                        project_grow=args.march_project_grow,
                        progress_every=0,
                        verbose=False,
                        seed_mode=args.march_seed_mode,
                        seed_step=args.march_seed_step,
                        seed_max_span=args.march_seed_max_span,
                        save_contours_dir=contour_save_dir,
                        simplify_after_step=not args.no_march_simplify,
                        simplify_sagitta_factor=args.march_simplify_sagitta_factor,
                        simplify_turn_degrees=args.march_simplify_turn_deg,
                        simplify_max_merged_segment_length=args.march_simplify_max_merged_segment_length,
                        simplify_min_keep_points=args.march_simplify_min_keep_points,
                        simplify_max_passes=args.march_simplify_max_passes,
                        simplify_working_contour=args.march_simplify_working_contour,
                        save_simplified_contours=not args.no_save_simplified_contours,
                    )
                else:
                    print(f"  tracing visible G={level:g} with step radius {step_r:g} ...", flush=True)
                    tracer_fn = trace_potential_contour_symmetric if args.trace_symmetry else trace_potential_contour
                    extra_kwargs = {}
                    if args.trace_symmetry:
                        extra_kwargs.update(dict(
                            symmetry_target_factor=args.trace_symmetry_target_factor,
                            symmetry_min_x=args.trace_symmetry_min_x,
                        ))

                    pts = tracer_fn(
                        level=level,
                        cache=trace_cache,
                        step_radius=step_r,
                        start_angle=args.trace_start_angle,
                        direction=args.trace_direction,
                        circle_samples=args.trace_circle_samples,
                        bisect_steps=args.trace_bisect_steps,
                        max_points=args.trace_max_points,
                        min_points_before_close=args.trace_min_points_before_close,
                        close_factor=args.trace_close_factor,
                        max_radius_adjust=args.trace_max_radius_adjust,
                        radius_shrink=args.trace_radius_shrink,
                        verbose=False,
                        progress_every=0,
                        quality_check=not args.no_trace_quality_check,
                        quality_every=args.trace_quality_every,
                        quality_start_after=args.trace_quality_start_after,
                        quality_normal_span_factor=args.trace_quality_normal_span_factor,
                        quality_normal_samples=args.trace_quality_normal_samples,
                        quality_max_sagitta_factor=args.trace_quality_max_sagitta_factor,
                        quality_max_turn_degrees=args.trace_quality_max_turn_deg,
                        quality_action=args.trace_quality_action,
                        quality_rollback_points=args.trace_quality_rollback_points,
                        quality_restart_shrink=args.trace_quality_restart_shrink,
                        quality_max_restarts=args.trace_quality_max_restarts,
                        quality_min_radius=args.trace_quality_min_radius,
                        quality_max_local_failures=args.trace_quality_max_local_failures,
                        **extra_kwargs,
                    )
                traced_contours[level] = pts
        else:
            pts = extract_primary_contour(x, y, G, level)

        pts = resample_closed_polyline_by_count(pts, args.visible_contour_points)
        contours.append((level, pts))

    title = (
        f"Mandelbrot potential Gmsh mesh "
        f"({args.height_mode}{', inverted' if args.invert else ''}, z×{args.z_scale:g}; "
        f"{args.gmin:g} ≤ G ≤ {args.gmax:g})"
    )

    print("Building Plotly figure...")
    fig, stats = build_plotly_figure(
        xy=mesh.xy,
        tris=mesh.tris,
        Gv=Gv,
        height_mode=args.height_mode,
        color_mode=args.color_mode,
        invert=args.invert,
        z_scale=args.z_scale,
        colorscale=args.colorscale,
        contours=contours,
        visible_contours=args.visible_contours,
        contour_color=args.contour_color,
        contour_width=args.contour_width,
        contour_opacity=args.contour_opacity,
        unlit=args.unlit,
        hard_lighting=args.hard_lighting,
        flat_shading=args.flat_shading,
        hover_mode=args.hover_mode,
        title=title,
    )

    if args.report_normals:
        print("Triangle orientation stats:", stats)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing {args.output} ...")
    fig.write_html(args.output, include_plotlyjs="cdn")
    print("Done.")


if __name__ == "__main__":
    main()
