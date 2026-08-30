#!/usr/bin/env python3
"""
Axisymmetric vector fractals from 2D profile + marching squares.

This script assumes the distinguished vector W is the z-axis. The fractal is
computed in the cylindrical half-plane (rho, z), where A=(rho, 0, z). The
inside/outside boundary is extracted with marching squares, simplified as 2D
profile curves, then revolved around the z-axis to make a 3D triangle mesh.

This version treats the largest open contour as the main body and closed
contours as rings, so the main body can be smoothed/simplified aggressively
while detached rings are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm
from skimage import measure
from scipy import ndimage as ndi
import open3d as o3d


@dataclass
class RenderConfig:
    # Map choice:
    #   cross_power: V -> V x (V x (...(V x W))) + A
    #                with cross_power repeated crosses by V.
    #                cross_power=2 is V x (V x W).
    #   dot_v:       V -> (V.W) V + A
    #   norm_w:      V -> (V.V) W + A
    #   dot_w:       alias for norm_w, because the name is easy to mix up.
    map_kind: str = "cross_power"
    power: int = 2

    nonlinear_scale: float = 1.0
    max_iter: int = 80
    bailout: float = 4.0

    # 2D cylindrical profile domain.
    rho_extent: float = 3.0

    # If z_bounds is a tuple, use it directly.
    # If z_bounds is None, compute on auto_z_search_bounds, then trim the
    # profile to the detected non-axis portion plus padding.
    z_bounds: tuple[float, float] | None = None
    auto_z_search_bounds: tuple[float, float] = (-3.0, 3.0)
    auto_axis_radius_fraction: float = 0.01
    auto_axis_radius_cells: int = 3
    auto_z_padding_fraction: float = 0.05
    center_z_after_auto_trim: bool = True

    n_rho: int = 1800
    n_z: int = 2400
    z_chunk_rows: int = 256

    # Profile cleanup and smoothing in grid-cell units.
    min_component_pixels: int = 32
    keep_largest_component: bool = False
    profile_smooth_sigma: float = 0.65

    # Marching-square contour filtering / simplification.
    min_contour_points: int = 25
    min_contour_length: float = 0.02
    simplify_tolerance_cells: float = 0.75

    # Role-aware contour simplification. The largest open contour is usually
    # the main body. Closed contours are usually detached rings. This lets us
    # smooth/decimate the main body aggressively while preserving rings.
    main_simplify_tolerance_cells: float | None = 8.0
    ring_simplify_tolerance_cells: float | None = None
    secondary_simplify_tolerance_cells: float | None = None
    main_smooth_iterations: int = 2
    main_smooth_amount: float = 0.35
    max_main_profile_points: int | None = 1800
    max_ring_profile_points: int | None = None
    # Closed ring contours should not be simplified down to 2-3 profile
    # points. That creates crude/self-intersect-looking toroidal strips after
    # revolution. This is a lower floor; max_ring_profile_points is still the
    # upper cap. If both are set and this value is larger than the max, the max
    # wins.
    min_ring_profile_points: int | None = None
    max_secondary_profile_points: int | None = 800
    keep_largest_contours: int | None = None

    # Revolution resolution.
    #
    # If dynamic_n_phi_by_radius=True, the role-specific n_phi values are
    # interpreted as the azimuthal resolution at rho == rho_extent. Points near
    # the axis use fewer samples. This is a big optimization for axisymmetric
    # meshes: small-radius rings do not need the same angular resolution as the
    # wide outer skirt.
    n_phi: int = 320
    main_n_phi: int | None = 48
    ring_n_phi: int | None = 80
    secondary_n_phi: int | None = 64
    dynamic_n_phi_by_radius: bool = True
    min_n_phi: int = 12
    n_phi_radius_power: float = 1.0
    n_phi_round_to: int = 4
    axis_epsilon: float = 1e-7

    # HTML export. Rounding coordinates before Plotly export can noticeably
    # reduce file size without changing the visual result much.
    html_float_precision: int | None = 6

    # Optional final mesh decimation. Usually leave as None; profile
    # simplification is cheaper and better for this axisymmetric case.
    final_target_triangles: int | None = None
    final_reduction_factor: float = 0.85

    # Output.
    output_basename: str = "axisymmetric_vector_fractal"
    colorscale: str = "Turbo"
    show_colorbar: bool = True


def vector_step_axisymmetric(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    A_rho: np.ndarray,
    A_z: np.ndarray,
    cfg: RenderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One vector-iteration step with W fixed to zhat=(0,0,1).

    The parameter is A=(A_rho, 0, A_z). We still keep full V=(vx,vy,vz),
    because odd nested-cross powers can generate a y component.
    """

    kind = cfg.map_kind.lower()
    scale = np.float32(cfg.nonlinear_scale)

    if kind == "cross_power":
        if cfg.power < 1:
            raise ValueError("cross_power must be at least 1")

        # Start with U = W = (0,0,1), then repeatedly U <- V x U.
        ux = np.zeros_like(vx, dtype=np.float32)
        uy = np.zeros_like(vx, dtype=np.float32)
        uz = np.ones_like(vx, dtype=np.float32)

        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(cfg.power):
                nx = vy * uz - vz * uy
                ny = vz * ux - vx * uz
                nz = vx * uy - vy * ux
                ux, uy, uz = nx, ny, nz

            return (
                scale * ux + A_rho,
                scale * uy,
                scale * uz + A_z,
            )

    if kind == "dot_v":
        # W=zhat, so V.W = vz.
        with np.errstate(over="ignore", invalid="ignore"):
            return (
                scale * vz * vx + A_rho,
                scale * vz * vy,
                scale * vz * vz + A_z,
            )

    if kind in {"norm_w", "dot_w"}:
        # W=zhat, so (V.V)W = (0,0,|V|^2).
        with np.errstate(over="ignore", invalid="ignore"):
            norm2 = vx * vx + vy * vy + vz * vz
            return (
                A_rho.copy(),
                np.zeros_like(vx, dtype=np.float32),
                scale * norm2 + A_z,
            )

    raise ValueError(
        f"Unknown map_kind={cfg.map_kind!r}. "
        "Use 'cross_power', 'pow_v', 'dot_v', 'norm_w', or 'dot_w'."
    )


def classify_profile(cfg: RenderConfig):
    """
    Compute the bounded/unbounded mask in the (rho, z) half-plane.

    Chunked in z so very large grids do not allocate the full RHO/Z/vx/vy/vz
    arrays at once.

    Returns:
        mask: boolean array with shape (n_z, n_rho), True = bounded / inside
        rho_values: rho coordinate array
        z_values: z coordinate array
    """

    print(f"Map: {cfg.map_kind}")
    if cfg.map_kind.lower() == "cross_power":
        print(f"Cross power: {cfg.power}")

    if cfg.z_bounds is None:
        z_min, z_max = cfg.auto_z_search_bounds
        print("z_bounds=None -> automatic z trimming enabled")
        print(f"initial z search bounds: [{z_min}, {z_max}]")
    else:
        z_min, z_max = cfg.z_bounds
        print("using explicit z bounds")

    print(f"Grid: {cfg.n_z:,} x {cfg.n_rho:,} = {cfg.n_z * cfg.n_rho:,} profile samples")
    print(f"rho: [0, {cfg.rho_extent}], z: [{z_min}, {z_max}]")

    rho_values = np.linspace(0.0, cfg.rho_extent, cfg.n_rho, dtype=np.float32)
    z_values = np.linspace(z_min, z_max, cfg.n_z, dtype=np.float32)

    # Full final mask is still large, but much smaller than holding all float arrays.
    mask = np.zeros((cfg.n_z, cfg.n_rho), dtype=bool)

    # Tune this. Smaller = less RAM, larger = faster.
    z_chunk_rows = getattr(cfg, "z_chunk_rows", 256)

    bailout2 = np.float32(cfg.bailout * cfg.bailout)

    chunks = list(range(0, cfg.n_z, z_chunk_rows))

    for z0 in tqdm(chunks, desc="Profile z chunks", unit=" chunk"):
        z1 = min(cfg.n_z, z0 + z_chunk_rows)
        z_chunk = z_values[z0:z1]

        # Shape: (chunk_rows, n_rho)
        RHO, Z = np.meshgrid(rho_values, z_chunk, indexing="xy")

        vx = np.zeros_like(RHO, dtype=np.float32)
        vy = np.zeros_like(RHO, dtype=np.float32)
        vz = np.zeros_like(RHO, dtype=np.float32)

        active = np.ones(RHO.shape, dtype=bool)

        for _ in range(cfg.max_iter):
            with np.errstate(over="ignore", invalid="ignore"):
                norm2 = vx * vx + vy * vy + vz * vz

            escaped = ((norm2 > bailout2) | (~np.isfinite(norm2))) & active
            active[escaped] = False

            if not np.any(active):
                break

            idx = active

            vx_new, vy_new, vz_new = vector_step_axisymmetric(
                vx[idx],
                vy[idx],
                vz[idx],
                RHO[idx],
                Z[idx],
                cfg,
            )

            vx[idx] = vx_new
            vy[idx] = vy_new
            vz[idx] = vz_new

        mask[z0:z1, :] = active

        # Help Python release chunky arrays between bands.
        del RHO, Z, vx, vy, vz, active

    print(f"Inside profile samples: {np.count_nonzero(mask):,}")
    return mask, rho_values, z_values


def clean_profile_components(mask: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    if cfg.min_component_pixels <= 0 and not cfg.keep_largest_component:
        return mask

    print("Cleaning 2D profile components...")
    structure = np.ones((3, 3), dtype=bool)
    labels, n_labels = ndi.label(mask, structure=structure)

    if n_labels == 0:
        print("No profile components found.")
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    if cfg.keep_largest_component:
        keep_labels = np.array([int(np.argmax(counts))], dtype=np.int64)
        print(f"Keeping largest 2D component: {counts[keep_labels[0]]:,} pixels")
    else:
        keep_labels = np.flatnonzero(counts >= cfg.min_component_pixels)
        print(
            f"Keeping {len(keep_labels):,}/{n_labels:,} 2D components "
            f"with at least {cfg.min_component_pixels:,} pixels"
        )

    cleaned = np.isin(labels, keep_labels)
    print(f"Profile pixels before cleanup: {np.count_nonzero(mask):,}")
    print(f"Profile pixels after cleanup:  {np.count_nonzero(cleaned):,}")
    return cleaned


def auto_trim_z_bounds(
    mask: np.ndarray,
    rho_values: np.ndarray,
    z_values: np.ndarray,
    cfg: RenderConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    If cfg.z_bounds is None, trim the profile in z to the region where the
    radial profile is not collapsed to the axis, plus a small padding.

    Returns:
        trimmed_mask
        possibly centered z_values
        z_display_center subtracted from z_values, or 0 if no centering
    """

    if cfg.z_bounds is not None:
        return mask, z_values, 0.0

    print("Detecting automatic z bounds from profile width...")

    if mask.shape != (len(z_values), len(rho_values)):
        raise ValueError("mask shape must be (len(z_values), len(rho_values))")

    if len(rho_values) < 2 or len(z_values) < 2:
        print("Not enough grid points for automatic z trimming; keeping full range.")
        return mask, z_values, 0.0

    dr = float(rho_values[1] - rho_values[0])
    dz = float(z_values[1] - z_values[0])

    close_to_axis_radius = max(
        cfg.auto_axis_radius_fraction * cfg.rho_extent,
        cfg.auto_axis_radius_cells * abs(dr),
    )

    # For each z row, find the largest rho that is still inside.
    rho_index = np.arange(len(rho_values), dtype=np.int32)
    max_inside_index = np.where(mask, rho_index[None, :], -1).max(axis=1)
    row_radius = np.where(max_inside_index >= 0, rho_values[max_inside_index], 0.0)

    active_rows = np.flatnonzero(row_radius > close_to_axis_radius)

    if len(active_rows) == 0:
        # Fallback: if only axis-hugging pixels exist, keep the rows with any
        # inside pixels. If even that fails, keep the original range.
        active_rows = np.flatnonzero(np.any(mask, axis=1))

    if len(active_rows) == 0:
        print("Could not detect a nonempty profile; keeping full z range.")
        return mask, z_values, 0.0

    i_min = int(active_rows[0])
    i_max = int(active_rows[-1])

    detected_min = float(z_values[i_min])
    detected_max = float(z_values[i_max])
    detected_height = max(detected_max - detected_min, abs(dz))
    padding = cfg.auto_z_padding_fraction * detected_height

    trim_min = max(float(z_values[0]), detected_min - padding)
    trim_max = min(float(z_values[-1]), detected_max + padding)

    keep_rows = np.flatnonzero((z_values >= trim_min) & (z_values <= trim_max))

    if len(keep_rows) == 0:
        print("Automatic z trimming produced no rows; keeping full z range.")
        return mask, z_values, 0.0

    trimmed_mask = mask[keep_rows, :]
    trimmed_z_values = z_values[keep_rows].astype(np.float32, copy=True)

    z_center = 0.5 * (trim_min + trim_max)

    print(f"axis-collapse radius threshold: {close_to_axis_radius:.6g}")
    print(f"detected non-axis z span: [{detected_min:.6g}, {detected_max:.6g}]")
    print(f"padding: {padding:.6g} ({100 * cfg.auto_z_padding_fraction:.2f}% of detected height)")
    print(f"trimmed z span: [{trim_min:.6g}, {trim_max:.6g}]")
    print(f"kept z rows: {len(keep_rows):,}/{len(z_values):,}")

    if cfg.center_z_after_auto_trim:
        trimmed_z_values -= np.float32(z_center)
        print(f"display z center shifted to zero: subtracted {z_center:.6g}")
        return trimmed_mask, trimmed_z_values, z_center

    return trimmed_mask, trimmed_z_values, 0.0


def polyline_length(points: np.ndarray, closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    length = float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
    if closed:
        d = points[0] - points[-1]
        length += float(np.sqrt(np.dot(d, d)))
    return length


def rdp_open(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for an open polyline."""

    if epsilon <= 0 or len(points) <= 2:
        return points.copy()

    start = points[0]
    end = points[-1]
    line = end - start
    line_len = float(np.sqrt(np.dot(line, line)))

    if line_len == 0:
        dists = np.sqrt(np.sum((points - start) ** 2, axis=1))
    else:
        # Perpendicular distance to the start-end line.
        rel = points - start
        cross = rel[:, 0] * line[1] - rel[:, 1] * line[0]
        dists = np.abs(cross) / line_len

    idx = int(np.argmax(dists))
    dmax = float(dists[idx])

    if dmax > epsilon:
        left = rdp_open(points[: idx + 1], epsilon)
        right = rdp_open(points[idx:], epsilon)
        return np.vstack([left[:-1], right])

    return np.vstack([start, end])


def simplify_contour(points: np.ndarray, epsilon: float, close_tol: float) -> tuple[np.ndarray, bool]:
    """Simplify a contour and return (points, closed)."""

    if len(points) < 2:
        return points, False

    closed = bool(np.linalg.norm(points[0] - points[-1]) <= close_tol)

    if closed:
        # Remove duplicate closure point, simplify as a closed loop by temporarily
        # cutting at a reasonably stable place, then close it again.
        pts = points[:-1].copy()
        if len(pts) <= 3:
            return pts, True

        # Cut opposite the first point to avoid making a tiny seam dominate RDP.
        d = np.sqrt(np.sum((pts - pts[0]) ** 2, axis=1))
        cut = int(np.argmax(d))
        rolled = np.vstack([pts[cut:], pts[:cut], pts[cut]])
        simp = rdp_open(rolled, epsilon)
        if len(simp) > 1 and np.linalg.norm(simp[0] - simp[-1]) <= close_tol:
            simp = simp[:-1]
        return simp, True

    return rdp_open(points, epsilon), False




def smooth_profile_curve(
    points: np.ndarray,
    closed: bool,
    iterations: int,
    amount: float,
) -> np.ndarray:
    """
    Light Laplacian smoothing for 2D profile curves.

    For the main open contour, endpoints are held fixed so axis/pole closure
    does not drift. For closed loops, smoothing wraps around.
    """

    if iterations <= 0 or amount <= 0 or len(points) < 3:
        return points.copy()

    amount = float(amount)
    amount = max(0.0, min(1.0, amount))
    pts = points.astype(np.float64, copy=True)

    for _ in range(iterations):
        old = pts.copy()
        if closed:
            pts = (1.0 - amount) * old + 0.5 * amount * (
                np.roll(old, 1, axis=0) + np.roll(old, -1, axis=0)
            )
        else:
            pts[1:-1] = (1.0 - amount) * old[1:-1] + 0.5 * amount * (
                old[:-2] + old[2:]
            )
            pts[0] = old[0]
            pts[-1] = old[-1]

    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    return pts


def resample_polyline_by_count(points: np.ndarray, max_points: int, closed: bool) -> np.ndarray:
    """
    Last-resort uniform arc-length resampling to enforce a point cap.

    RDP is preferred because it preserves shape adaptively. This only kicks in
    if RDP cannot reach the configured cap.
    """

    if max_points is None or len(points) <= max_points or max_points < 2:
        return points.copy()

    pts = points.astype(np.float64, copy=False)

    if closed:
        work = np.vstack([pts, pts[0]])
        n_out = max(3, max_points)
        sample_s = np.linspace(0.0, 1.0, n_out + 1, endpoint=True)[:-1]
    else:
        work = pts
        n_out = max(2, max_points)
        sample_s = np.linspace(0.0, 1.0, n_out, endpoint=True)

    seg = np.diff(work, axis=0)
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    total = float(np.sum(seg_len))

    if total == 0:
        return pts[:n_out].copy()

    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = sample_s * total

    out = []
    for t in targets:
        idx = int(np.searchsorted(cumulative, t, side="right") - 1)
        idx = min(max(idx, 0), len(seg_len) - 1)
        denom = seg_len[idx]
        if denom == 0:
            alpha = 0.0
        else:
            alpha = (t - cumulative[idx]) / denom
        p = (1.0 - alpha) * work[idx] + alpha * work[idx + 1]
        out.append(p)

    out = np.asarray(out, dtype=np.float64)
    out[:, 0] = np.maximum(out[:, 0], 0.0)
    return out


def simplify_contour_with_cap(
    points: np.ndarray,
    base_epsilon: float,
    close_tol: float,
    max_points: int | None,
    label: str,
) -> tuple[np.ndarray, bool, float]:
    """
    Simplify a contour, increasing epsilon until an optional point cap is met.

    Returns simplified points, closed flag, and the final epsilon used.
    """

    eps = max(float(base_epsilon), 0.0)
    pts_s, closed = simplify_contour(points, eps, close_tol)

    if max_points is None or len(pts_s) <= max_points:
        return pts_s, closed, eps

    # Increase RDP tolerance progressively. This is much cheaper in 2D than
    # decimating the final revolved mesh in 3D.
    for _ in range(20):
        eps *= 1.5
        pts_try, closed_try = simplify_contour(points, eps, close_tol)
        pts_s, closed = pts_try, closed_try
        if len(pts_s) <= max_points:
            return pts_s, closed, eps

    # Last resort: force the cap by arc-length resampling.
    forced = resample_polyline_by_count(pts_s, max_points=max_points, closed=closed)
    print(
        f"  warning: {label} contour still had {len(pts_s):,} points after RDP; "
        f"arc-length resampled to {len(forced):,}"
    )
    return forced, closed, eps


def extract_profile_contours(mask: np.ndarray, rho_values: np.ndarray, z_values: np.ndarray, cfg: RenderConfig):
    """
    Use marching squares to extract all 2D inside/outside boundaries.

    The biggest open contour is treated as the main body. Closed contours are
    treated as rings. This allows aggressive smoothing/simplification of the
    main body while keeping detached rings crisp.
    """

    field = mask.astype(np.float32)
    if cfg.profile_smooth_sigma and cfg.profile_smooth_sigma > 0:
        print(f"Smoothing 2D profile field with sigma={cfg.profile_smooth_sigma} cells...")
        field = ndi.gaussian_filter(field, sigma=cfg.profile_smooth_sigma, mode="nearest")

    print("Running marching squares on 2D profile...")
    raw_contours = measure.find_contours(field, level=0.5)
    print(f"Raw contours found: {len(raw_contours):,}")

    dr = float(rho_values[1] - rho_values[0]) if len(rho_values) > 1 else 1.0
    dz = float(z_values[1] - z_values[0]) if len(z_values) > 1 else 1.0
    rho0 = float(rho_values[0])
    z0 = float(z_values[0])

    close_tol = 2.0 * max(abs(dr), abs(dz))
    base_eps = cfg.simplify_tolerance_cells * min(abs(dr), abs(dz))

    raw_items = []

    for c in raw_contours:
        # skimage contour columns are (row, col) = (z_index, rho_index).
        rows = c[:, 0]
        cols = c[:, 1]

        rho = rho0 + cols * dr
        z = z0 + rows * dz
        pts = np.column_stack([rho, z]).astype(np.float64)
        pts[:, 0] = np.maximum(pts[:, 0], 0.0)

        closed_raw = bool(np.linalg.norm(pts[0] - pts[-1]) <= close_tol)
        length = polyline_length(pts, closed=closed_raw)

        if len(pts) < cfg.min_contour_points:
            continue
        if length < cfg.min_contour_length:
            continue

        raw_items.append({
            "points": pts,
            "closed_raw": closed_raw,
            "length_raw": length,
            "raw_points": len(pts),
        })

    raw_items.sort(key=lambda item: item["length_raw"], reverse=True)

    if cfg.keep_largest_contours is not None:
        raw_items = raw_items[: cfg.keep_largest_contours]

    # Main body: largest open contour if available, otherwise largest contour.
    main_index = None
    for i, item in enumerate(raw_items):
        if not item["closed_raw"]:
            main_index = i
            break
    if main_index is None and raw_items:
        main_index = 0

    print(f"Profile base simplification tolerance: {base_eps:.6g} world units")
    if main_index is not None:
        print(f"Main contour candidate: contour {main_index + 1} after length sorting")

    contours = []

    for i, item in enumerate(raw_items):
        pts = item["points"]
        closed_raw = item["closed_raw"]

        if i == main_index:
            role = "main"
            tol_cells = cfg.main_simplify_tolerance_cells
            if tol_cells is None:
                tol_cells = cfg.simplify_tolerance_cells
            max_points = cfg.max_main_profile_points
            if cfg.main_smooth_iterations > 0:
                pts = smooth_profile_curve(
                    pts,
                    closed=closed_raw,
                    iterations=cfg.main_smooth_iterations,
                    amount=cfg.main_smooth_amount,
                )
        elif closed_raw:
            role = "ring"
            tol_cells = cfg.ring_simplify_tolerance_cells
            if tol_cells is None:
                tol_cells = cfg.simplify_tolerance_cells
            max_points = cfg.max_ring_profile_points
        else:
            role = "secondary"
            tol_cells = cfg.secondary_simplify_tolerance_cells
            if tol_cells is None:
                tol_cells = cfg.simplify_tolerance_cells
            max_points = cfg.max_secondary_profile_points

        eps = tol_cells * min(abs(dr), abs(dz))
        pts_s, closed, eps_used = simplify_contour_with_cap(
            pts,
            base_epsilon=eps,
            close_tol=close_tol,
            max_points=max_points,
            label=role,
        )

        # Closed rings simplified to 2-3 points revolve into ugly wedge-like
        # toroidal strips. Enforce a small lower floor by resampling the
        # original high-resolution contour if RDP became too aggressive.
        if role == "ring" and closed and cfg.min_ring_profile_points is not None:
            min_points = max(4, int(cfg.min_ring_profile_points))

            # If both a min and max are configured, the max is the hard cap.
            if cfg.max_ring_profile_points is not None:
                min_points = min(min_points, int(cfg.max_ring_profile_points))

            if len(pts_s) < min_points:
                pts_s = resample_polyline_by_count(
                    pts,
                    max_points=min_points,
                    closed=True,
                )
                closed = True

        if len(pts_s) < 2:
            continue

        length_s = polyline_length(pts_s, closed=closed)

        contours.append({
            "points": pts_s,
            "closed": closed,
            "role": role,
            "length": length_s,
            "raw_length": item["length_raw"],
            "raw_points": item["raw_points"],
            "simplified_points": len(pts_s),
            "epsilon_used": eps_used,
        })

    # Keep main first, then rings/secondary by length. This is useful for logs.
    role_order = {"main": 0, "ring": 1, "secondary": 2}
    contours.sort(key=lambda item: (role_order.get(item["role"], 99), -item["length"]))

    print(f"Contours kept: {len(contours):,}")
    total_profile_points = sum(item["simplified_points"] for item in contours)
    print(f"Total simplified profile points: {total_profile_points:,}")

    for i, item in enumerate(contours[:15], start=1):
        print(
            f"  contour {i:02d}: "
            f"role={item['role']:<9} "
            f"points {item['raw_points']:,} -> {item['simplified_points']:,}, "
            f"length {item['length']:.4g}, "
            f"closed={item['closed']}, "
            f"eps={item['epsilon_used']:.4g}"
        )
    if len(contours) > 15:
        print(f"  ... {len(contours) - 15:,} more contours")

    return contours


def revolve_contours_to_mesh(contours, cfg: RenderConfig):
    """
    Revolve one or more 2D contours around the z-axis.

    This version supports dynamic azimuthal resolution. Each profile point gets
    its own n_phi based on its local radius rho. The configured role-specific
    n_phi values act as the maximum at rho == cfg.rho_extent.

    Neighboring rings with different vertex counts are connected with a simple
    angular advancing-front triangulation. It is not mathematically fancy, but
    it is exactly the optimization we want here: small-radius rings get fewer
    vertices, large-radius rings keep more circularity.
    """

    print("Revolving profile contours into a 3D mesh...")

    vertices = []
    faces = []

    def add_vertex(v):
        vertices.append(v)
        return len(vertices) - 1

    def max_n_phi_for_role(role: str) -> int:
        if role == "main" and cfg.main_n_phi is not None:
            return int(cfg.main_n_phi)
        if role == "ring" and cfg.ring_n_phi is not None:
            return int(cfg.ring_n_phi)
        if role == "secondary" and cfg.secondary_n_phi is not None:
            return int(cfg.secondary_n_phi)
        return int(cfg.n_phi)

    def quantize_n_phi(n: float, max_n_phi: int) -> int:
        min_n = max(3, int(cfg.min_n_phi))
        max_n = max(min_n, int(max_n_phi))

        n_int = int(np.ceil(float(n)))

        q = int(cfg.n_phi_round_to)
        if q > 1:
            n_int = int(np.ceil(n_int / q) * q)

        n_int = max(min_n, min(max_n, n_int))
        return n_int

    def n_phi_for_radius(role: str, rho: float) -> int:
        max_n = max_n_phi_for_role(role)

        if not cfg.dynamic_n_phi_by_radius:
            return max(3, int(max_n))

        rho_scale = 0.0 if cfg.rho_extent <= 0 else float(rho) / float(cfg.rho_extent)
        rho_scale = max(0.0, min(1.0, rho_scale))
        rho_scale = rho_scale ** float(cfg.n_phi_radius_power)

        min_n = max(3, int(cfg.min_n_phi))
        n = min_n + (float(max_n) - min_n) * rho_scale
        return quantize_n_phi(n, max_n)

    def add_axis_to_ring(axis_idx: int, ring: list[int], reverse: bool = False):
        n_phi = len(ring)
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            if reverse:
                faces.append([ring[j], axis_idx, ring[jn]])
            else:
                faces.append([axis_idx, ring[j], ring[jn]])

    def add_matching_ring_connection(ring_a: list[int], ring_b: list[int]):
        n_phi = len(ring_a)
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            a0 = ring_a[j]
            a1 = ring_a[jn]
            b0 = ring_b[j]
            b1 = ring_b[jn]
            faces.append([a0, b0, b1])
            faces.append([a0, b1, a1])

    def add_mismatched_ring_connection(ring_a: list[int], ring_b: list[int]):
        """
        Connect two circular rings with possibly different vertex counts.

        Think of both rings as polygonal samples of the same angular interval
        [0, 2pi). We advance around the lower-angle next edge. If both next
        angular breaks coincide, we emit a quad split into two triangles.
        Otherwise we emit one triangle. This produces a valid annular strip
        without requiring matching n_phi.
        """

        na = len(ring_a)
        nb = len(ring_b)

        if na < 3 or nb < 3:
            return

        if na == nb:
            add_matching_ring_connection(ring_a, ring_b)
            return

        i = 0
        j = 0

        while i < na or j < nb:
            # One side has finished its full turn. Fan remaining angular
            # intervals to the other ring's zero vertex.
            if i >= na:
                bj = ring_b[j % nb]
                bj1 = ring_b[(j + 1) % nb]
                a0 = ring_a[0]
                faces.append([a0, bj, bj1])
                j += 1
                continue

            if j >= nb:
                ai = ring_a[i % na]
                ai1 = ring_a[(i + 1) % na]
                b0 = ring_b[0]
                faces.append([ai, b0, ai1])
                i += 1
                continue

            # Compare (i+1)/na and (j+1)/nb without floating point.
            next_a_scaled = (i + 1) * nb
            next_b_scaled = (j + 1) * na

            ai = ring_a[i % na]
            ai1 = ring_a[(i + 1) % na]
            bj = ring_b[j % nb]
            bj1 = ring_b[(j + 1) % nb]

            if next_a_scaled < next_b_scaled:
                faces.append([ai, bj, ai1])
                i += 1
            elif next_b_scaled < next_a_scaled:
                faces.append([ai, bj, bj1])
                j += 1
            else:
                # Both edges end at the same angular position: emit a quad.
                faces.append([ai, bj, bj1])
                faces.append([ai, bj1, ai1])
                i += 1
                j += 1

    def add_connection(a, b):
        """Connect two contour sample rings/axis-points."""
        a_axis = isinstance(a, int)
        b_axis = isinstance(b, int)

        if a_axis and b_axis:
            return

        if a_axis and not b_axis:
            add_axis_to_ring(a, b, reverse=False)
            return

        if not a_axis and b_axis:
            add_axis_to_ring(b, a, reverse=True)
            return

        add_mismatched_ring_connection(a, b)

    expected_triangles = 0

    for item in contours:
        pts = item["points"]
        closed = item["closed"]
        role = item.get("role", "secondary")

        face_count_before = len(faces)
        n_phi_values = []
        ring_refs = []

        for rho, z in pts:
            if rho <= cfg.axis_epsilon:
                ring_refs.append(add_vertex([0.0, 0.0, float(z)]))
                continue

            n_phi = n_phi_for_radius(role, float(rho))
            n_phi_values.append(n_phi)

            phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)
            cos_phi = np.cos(phis)
            sin_phi = np.sin(phis)

            x = float(rho) * cos_phi
            y = float(rho) * sin_phi
            zz = np.full_like(x, float(z), dtype=np.float64)

            ring = []
            for xx, yy, zzz in zip(x, y, zz):
                ring.append(add_vertex([float(xx), float(yy), float(zzz)]))
            ring_refs.append(ring)

        for i in range(len(ring_refs) - 1):
            add_connection(ring_refs[i], ring_refs[i + 1])

        if closed and len(ring_refs) > 2:
            add_connection(ring_refs[-1], ring_refs[0])

        actual_triangles = len(faces) - face_count_before
        expected_triangles += actual_triangles

        if n_phi_values:
            n_min = min(n_phi_values)
            n_max = max(n_phi_values)
            n_mean = float(np.mean(n_phi_values))
            n_info = f"n_phi={n_min}..{n_max} (mean {n_mean:.1f})"
        else:
            n_info = "axis-only"

        print(
            f"  revolved {role:<9} contour: {len(pts):,} profile points, "
            f"{n_info}, triangles={actual_triangles:,}"
        )

    if not vertices or not faces:
        raise RuntimeError("No mesh generated. Try a larger domain, lower bailout, or fewer filters.")

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    print(f"Raw revolved mesh: {len(vertices):,} vertices, {len(faces):,} triangles")
    print(f"Expected triangles before cleanup: ~{expected_triangles:,}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    try:
        mesh.orient_triangles()
    except Exception as exc:
        print(f"Could not orient triangles: {exc}")

    mesh.compute_vertex_normals()

    if cfg.final_target_triangles is not None:
        mesh = simplify_mesh_progressive(
            mesh,
            target_triangles=cfg.final_target_triangles,
            reduction_factor=cfg.final_reduction_factor,
        )

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    print(f"Final mesh: {len(verts):,} vertices, {len(tris):,} triangles")

    return mesh, verts, tris


def simplify_mesh_progressive(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int,
    reduction_factor: float = 0.85,
    min_triangle_drop: int = 1000,
):
    """Optional final 3D mesh simplification. Usually not needed here."""

    if not (0 < reduction_factor < 1):
        raise ValueError("reduction_factor must be between 0 and 1")

    current = len(np.asarray(mesh.triangles))
    target_triangles = min(target_triangles, current)

    print(
        f"Final mesh simplification: {current:,} -> {target_triangles:,} triangles "
        f"using factor {reduction_factor:.3f}"
    )

    simplified = mesh
    stage = 0

    with tqdm(desc="Final simplification", unit=" stage") as pbar:
        while True:
            before = len(np.asarray(simplified.triangles))
            if before <= target_triangles:
                break

            next_target = int(before * reduction_factor)
            next_target = max(next_target, target_triangles)
            if before - next_target < min_triangle_drop:
                next_target = target_triangles

            t0 = time.perf_counter()
            simplified_next = simplified.simplify_quadric_decimation(next_target)

            simplified_next.remove_degenerate_triangles()
            simplified_next.remove_duplicated_triangles()
            simplified_next.remove_duplicated_vertices()
            simplified_next.remove_unreferenced_vertices()
            simplified_next.remove_non_manifold_edges()

            after = len(np.asarray(simplified_next.triangles))
            dt = time.perf_counter() - t0
            stage += 1
            tqdm.write(f"  stage {stage:02d}: {before:,} -> {after:,} triangles in {dt:.1f} s")

            simplified = simplified_next
            pbar.update(1)

            if after >= before:
                print("No further simplification achieved; stopping.")
                break

    try:
        simplified.orient_triangles()
    except Exception as exc:
        print(f"Could not orient triangles after simplification: {exc}")

    simplified.compute_vertex_normals()
    return simplified


def plot_mesh_html(
    verts: np.ndarray,
    faces: np.ndarray,
    output_html: str,
    colorscale: str = "Turbo",
    show_colorbar: bool = True,
    cmin: float | None = None,
    cmax: float | None = None,
    html_float_precision: int | None = 6,
):
    print("Preparing Plotly mesh...")

    if html_float_precision is not None:
        verts = np.round(verts.astype(np.float64, copy=False), int(html_float_precision))
        print(f"Rounded vertex coordinates to {html_float_precision} decimals for smaller HTML")

    radius = np.sqrt(
        verts[:, 0] * verts[:, 0]
        + verts[:, 1] * verts[:, 1]
        + verts[:, 2] * verts[:, 2]
    )

    if cmin is None:
        cmin = float(np.nanmin(radius))
    if cmax is None:
        cmax = float(np.nanmax(radius))

    if np.isclose(cmin, cmax):
        eps = 1e-6 if cmin == 0 else abs(cmin) * 1e-6
        cmin -= eps
        cmax += eps

    print(f"Color range: {cmin:.6g} -> {cmax:.6g}")

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=radius,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Radius") if show_colorbar else None,
                showscale=show_colorbar,
                opacity=1.0,
                flatshading=False,
                hoverinfo="skip",
                lighting=dict(
                    ambient=0.35,
                    diffuse=0.75,
                    specular=0.25,
                    roughness=0.75,
                    fresnel=0.05,
                ),
                lightposition=dict(x=100, y=200, z=120),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Wrote {output_html}")


def make_safe_basename(cfg: RenderConfig) -> str:
    if cfg.map_kind.lower() == "cross_power":
        stem = f"{cfg.output_basename}_{cfg.map_kind}_p{cfg.power}"
    else:
        stem = f"{cfg.output_basename}_{cfg.map_kind}"
    return stem.replace("-", "m").replace(".", "p")


def main():
    cfg = RenderConfig(
        # Try these:
        map_kind="cross_power",
        power=2,
        # cross_power=3,
        # cross_power=4,
        # map_kind="dot_v",
        # map_kind="norm_w",
        # map_kind="dot_w",

        nonlinear_scale=1.0,
        max_iter=200,
        bailout=20.0,

        rho_extent=3.0,

        # Use explicit bounds with e.g. z_bounds=(-1.1, 2.05).
        # Leave as None to search first, detect the useful z span, trim it,
        # and recenter the displayed object.
        z_bounds=None,
        auto_z_search_bounds=(-3.0, 3.0),
        auto_axis_radius_fraction=0.01,
        auto_axis_radius_cells=3,
        auto_z_padding_fraction=0.01,
        center_z_after_auto_trim=True,

        n_rho=50_000,
        n_z=50_000,
        z_chunk_rows=256,

        profile_smooth_sigma=0.65,
        min_component_pixels=32,
        keep_largest_component=False,

        # The script classifies the largest open contour as the main body and
        # closed contours as detached rings. This keeps the rings while making
        # the body much cheaper.
        simplify_tolerance_cells=0.75,
        main_simplify_tolerance_cells=12.0,
        ring_simplify_tolerance_cells=12.0,
        secondary_simplify_tolerance_cells=3.0,
        main_smooth_iterations=3,
        main_smooth_amount=0.35,
        max_main_profile_points=1200,
        max_ring_profile_points=16,
        min_ring_profile_points=6,
        max_secondary_profile_points=400,

        min_contour_points=25,
        min_contour_length=0.02,
        keep_largest_contours=None,

        n_phi=80,
        main_n_phi=80,
        ring_n_phi=80,
        secondary_n_phi=80,
        dynamic_n_phi_by_radius=True,
        min_n_phi=12,
        n_phi_radius_power=1.0,
        n_phi_round_to=4,

        # Leave this as None. For this axisymmetric pipeline, simplifying the
        # 2D profile is cleaner than decimating the already-revolved mesh.
        final_target_triangles=None,
        final_reduction_factor=0.85,

        html_float_precision=6,
        output_basename="axisymmetric_vector_fractal_light",
    )

    stem = make_safe_basename(cfg)
    output_ply = Path(f"{stem}.ply")
    output_html = Path(f"{stem}.html")

    mask, rho_values, z_values = classify_profile(cfg)
    mask = clean_profile_components(mask, cfg)
    mask, z_values, z_display_center = auto_trim_z_bounds(mask, rho_values, z_values, cfg)
    contours = extract_profile_contours(mask, rho_values, z_values, cfg)
    mesh, verts, faces = revolve_contours_to_mesh(contours, cfg)

    o3d.io.write_triangle_mesh(str(output_ply), mesh)
    print(f"Wrote {output_ply}")

    plot_mesh_html(
        verts,
        faces,
        output_html=str(output_html),
        colorscale=cfg.colorscale,
        show_colorbar=cfg.show_colorbar,
        html_float_precision=cfg.html_float_precision,
    )


if __name__ == "__main__":
    main()
