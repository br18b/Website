#!/usr/bin/env python3
"""
Mandelbrot escape-potential contour mesh.

Builds a Plotly Mesh3d surface by extracting exterior equipotential contours
G(c)=constant and stitching neighboring contours together.

This version deliberately defaults to a stable zipper mesh with the SAME number
of points on every contour ring. The adaptive / Delaunay attempts are tempting,
but near the Mandelbrot boundary they easily create wrong bridges, holes, and
spikes. Same-count contour rings are boring, but robust.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Geometry helpers
# ============================================================


def polygon_area(points: np.ndarray) -> float:
    """Signed polygon area. Positive means CCW in the x-y plane."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    if polygon_area(points) < 0:
        return points[::-1].copy()
    return points.copy()


def close_if_needed(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        return np.vstack([points, points[0]])
    return points


def polyline_length_closed(points: np.ndarray) -> float:
    pts = close_if_needed(points)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs * diffs).sum(axis=1))))


def rotate_ring(points: np.ndarray, start_index: int) -> np.ndarray:
    return np.vstack([points[start_index:], points[:start_index]])


def align_ring_to_anchor(points: np.ndarray, anchor: np.ndarray | None) -> np.ndarray:
    """
    Rotate ring so its seam is consistent across levels.

    First ring starts at the rightmost point. Later rings start at the point
    closest to the previous ring's seam. This keeps the zipper seam from
    drifting around the shape.
    """
    if anchor is None:
        idx = int(np.argmax(points[:, 0]))
    else:
        d2 = np.sum((points - anchor[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
    return rotate_ring(points, idx)


def best_cyclic_shift_to_reference(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Rotate `points` so point i on the outer ring is as close as possible,
    in a global least-squares sense, to point i on the reference ring.

    This does not change the contour geometry or point order. It only moves
    the ring seam. It helps prevent twisted bands caused by seam drift.
    """
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(points, dtype=float)

    if len(ref) != len(pts):
        raise ValueError("best_cyclic_shift_to_reference requires equal point counts.")

    n = len(ref)
    if n == 0:
        return pts.copy()

    # Brute force is fine for n ~ 1000-3000 and only a few dozen rings.
    # Score shift s as sum_i |ref[i] - pts[(i+s) % n]|^2.
    scores = np.empty(n, dtype=float)
    for s in range(n):
        shifted = np.roll(pts, -s, axis=0)
        d = ref - shifted
        scores[s] = float(np.sum(d * d))

    best = int(np.argmin(scores))
    return np.roll(pts, -best, axis=0)



def resample_closed_polyline_by_count(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample a closed polyline to exactly n_samples points.

    Returns an open ring: first point is NOT repeated at the end.
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

    target = np.linspace(0.0, total, n_samples, endpoint=False)
    xr = np.interp(target, s, pts[:, 0])
    yr = np.interp(target, s, pts[:, 1])
    return np.column_stack([xr, yr])


def circular_smooth_ring(points: np.ndarray, passes: int = 0, alpha: float = 0.5) -> np.ndarray:
    """
    Gentle periodic smoothing for contour rings.

    alpha=0 keeps the original ring, alpha=1 replaces each point by the
    neighbor average. Values around 0.25-0.5 are useful for removing grid
    staircase noise without destroying the overall shape.
    """
    pts = np.asarray(points, dtype=float).copy()
    for _ in range(max(0, passes)):
        avg = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pts = (1.0 - alpha) * pts + alpha * avg
    return pts



def triangle_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.cross(p1 - p0, p2 - p0)


def normal_alignment_score(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, split: str) -> float:
    """
    Score a quad split by comparing the normals of its two triangles.
    Larger is better. 1 means the two triangle normals are parallel.
    """
    if split == "diag_a0_b1":
        n1 = triangle_normal(p0, p2, p3)
        n2 = triangle_normal(p0, p3, p1)
    elif split == "diag_a1_b0":
        n1 = triangle_normal(p0, p2, p1)
        n2 = triangle_normal(p1, p2, p3)
    else:
        raise ValueError(f"Unknown split: {split}")

    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n == 0 or n2n == 0:
        return -1.0

    return float(abs(np.dot(n1, n2)) / (n1n * n2n))


def analyze_triangle_orientation(vertices: np.ndarray, tris: np.ndarray) -> dict:
    normals_z = []
    degenerate = 0

    for a, b, c in np.asarray(tris, dtype=int):
        n = triangle_normal(vertices[a], vertices[b], vertices[c])
        nn = np.linalg.norm(n)
        if nn == 0:
            degenerate += 1
            continue
        normals_z.append(n[2] / nn)

    if not normals_z:
        return {"count": 0, "degenerate": degenerate}

    arr = np.asarray(normals_z, dtype=float)
    return {
        "count": int(arr.size),
        "degenerate": int(degenerate),
        "min_unit_z": float(arr.min()),
        "max_unit_z": float(arr.max()),
        "negative_z": int(np.sum(arr < 0)),
        "nonpositive_z": int(np.sum(arr <= 0)),
    }


def _quantile_dict(values: np.ndarray, name: str) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{name}_count": 0}

    return {
        f"{name}_count": int(values.size),
        f"{name}_min": float(np.min(values)),
        f"{name}_p01": float(np.quantile(values, 0.01)),
        f"{name}_p05": float(np.quantile(values, 0.05)),
        f"{name}_median": float(np.quantile(values, 0.50)),
        f"{name}_p95": float(np.quantile(values, 0.95)),
        f"{name}_p99": float(np.quantile(values, 0.99)),
        f"{name}_max": float(np.max(values)),
    }


def _orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-14) -> bool:
    """
    Strict 2D segment crossing test. Collinear touches are ignored.
    """
    o1 = _orientation_2d(a, b, c)
    o2 = _orientation_2d(a, b, d)
    o3 = _orientation_2d(c, d, a)
    o4 = _orientation_2d(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def geometry_diagnostics(vertices: np.ndarray, tris: np.ndarray, eps: float = 1e-12) -> dict:
    """
    Check for actual mesh/topology problems: duplicate triangles, non-manifold
    edges, degenerate faces, repeated vertices, and edge length outliers.
    """
    V = np.asarray(vertices, dtype=float)
    T = np.asarray(tris, dtype=int)

    sorted_tris = np.sort(T, axis=1)
    tri_keys = [tuple(row) for row in sorted_tris]
    duplicate_triangles = len(tri_keys) - len(set(tri_keys))

    edge_counts = {}
    edge_lengths = []
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            uu, vv = (int(u), int(v)) if u < v else (int(v), int(u))
            edge_counts[(uu, vv)] = edge_counts.get((uu, vv), 0) + 1
            edge_lengths.append(float(np.linalg.norm(V[uu] - V[vv])))

    counts = np.fromiter(edge_counts.values(), dtype=int)
    boundary_edges = int(np.sum(counts == 1))
    interior_edges = int(np.sum(counts == 2))
    nonmanifold_edges = int(np.sum(counts > 2))

    areas = []
    for a, b, c in T:
        areas.append(0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a])))
    areas = np.asarray(areas, dtype=float)
    degenerate_faces = int(np.sum(areas <= eps))

    if eps > 0:
        Q = np.round(V / eps).astype(np.int64)
        vertex_keys = [tuple(row) for row in Q]
        duplicate_vertices = len(vertex_keys) - len(set(vertex_keys))
    else:
        duplicate_vertices = 0

    out = {
        "vertices": int(len(V)),
        "triangles": int(len(T)),
        "duplicate_unordered_triangles": int(duplicate_triangles),
        "unique_edges": int(len(edge_counts)),
        "boundary_edges": boundary_edges,
        "interior_edges": interior_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": degenerate_faces,
        "duplicate_vertices_quantized": int(duplicate_vertices),
    }
    out.update(_quantile_dict(np.asarray(edge_lengths, dtype=float), "edge_length"))
    out.update(_quantile_dict(areas, "triangle_area"))
    return out


def choose_quad_split_for_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, quad_split: str, q: int) -> str:
    if quad_split == "fixed":
        return "diag_a0_b1"
    if quad_split == "alternate":
        return "diag_a0_b1" if (q % 2 == 0) else "diag_a1_b0"
    if quad_split == "planar":
        s1 = normal_alignment_score(p0, p1, p2, p3, "diag_a0_b1")
        s2 = normal_alignment_score(p0, p1, p2, p3, "diag_a1_b0")
        return "diag_a0_b1" if s1 >= s2 else "diag_a1_b0"

    raise ValueError(f"Unknown quad_split: {quad_split}")


def band_diagnostics(vertices: np.ndarray, ring_indices: list[list[int]], quad_split: str) -> dict:
    """
    Diagnostics for ring-to-ring quads. This catches twisted annulus quads,
    crossing radial edges, and strongly non-planar quad splits.
    """
    V = np.asarray(vertices, dtype=float)

    quad_signed_areas_xy = []
    normal_angles_deg = []
    crossing_radial_edges = 0
    negative_area_quads = 0
    near_zero_area_quads = 0

    for inner, outer in zip(ring_indices[:-1], ring_indices[1:]):
        n = len(inner)
        if n != len(outer):
            continue

        for q in range(n):
            a0 = inner[q]
            a1 = inner[(q + 1) % n]
            b0 = outer[q]
            b1 = outer[(q + 1) % n]

            p0 = V[a0]
            p1 = V[a1]
            p2 = V[b0]
            p3 = V[b1]

            xy_poly = np.array([p0[:2], p2[:2], p3[:2], p1[:2]])
            area_xy = polygon_area(xy_poly)
            quad_signed_areas_xy.append(area_xy)

            if area_xy < 0:
                negative_area_quads += 1
            if abs(area_xy) < 1e-14:
                near_zero_area_quads += 1

            if _segments_cross_2d(p0[:2], p2[:2], p1[:2], p3[:2]):
                crossing_radial_edges += 1

            split = choose_quad_split_for_points(p0, p1, p2, p3, quad_split, q)
            if split == "diag_a0_b1":
                n1 = triangle_normal(p0, p2, p3)
                n2 = triangle_normal(p0, p3, p1)
            else:
                n1 = triangle_normal(p0, p2, p1)
                n2 = triangle_normal(p1, p2, p3)

            n1n = np.linalg.norm(n1)
            n2n = np.linalg.norm(n2)
            if n1n > 0 and n2n > 0:
                cosang = abs(float(np.dot(n1, n2) / (n1n * n2n)))
                cosang = max(-1.0, min(1.0, cosang))
                normal_angles_deg.append(float(np.degrees(np.arccos(cosang))))

    area_arr = np.asarray(quad_signed_areas_xy, dtype=float)
    angle_arr = np.asarray(normal_angles_deg, dtype=float)

    out = {
        "quad_count": int(len(quad_signed_areas_xy)),
        "negative_area_quads_xy": int(negative_area_quads),
        "near_zero_area_quads_xy": int(near_zero_area_quads),
        "crossing_radial_edges_xy": int(crossing_radial_edges),
    }
    out.update(_quantile_dict(area_arr, "quad_signed_area_xy"))
    out.update(_quantile_dict(angle_arr, "quad_triangle_normal_angle_deg"))
    return out


def orient_triangles_up(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """
    Ensure all triangles have positive z-normal in their x-y projection.

    For a graph-like surface z=h(x,y), consistent CCW orientation in the x-y
    plane avoids Plotly's alternating dark/bright triangle issue.
    """
    tris = np.asarray(tris, dtype=int).copy()
    for t_idx, (a, b, c) in enumerate(tris):
        p0 = vertices[a]
        p1 = vertices[b]
        p2 = vertices[c]
        normal = np.cross(p1 - p0, p2 - p0)
        if normal[2] < 0:
            tris[t_idx] = (a, c, b)
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
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=np.complex128)
    active = np.ones(C.shape, dtype=bool)
    escaped = np.zeros(C.shape, dtype=bool)
    G = np.zeros(C.shape, dtype=np.float64)

    for n in range(1, max_iter + 1):
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


# ============================================================
# Contours
# ============================================================


def extract_primary_contour(x: np.ndarray, y: np.ndarray, G: np.ndarray, level: float) -> np.ndarray:
    """
    Extract the longest contour segment G(x,y)=level.

    For the chosen exterior levels this should be the main closed equipotential
    loop around the Mandelbrot set.
    """
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, G, levels=[level])

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
            best_len = L
            best = seg

    if best is None:
        raise RuntimeError(f"No usable contour found at level {level:g}")

    return ensure_ccw(best)


@dataclass
class Ring:
    level: float
    xy: np.ndarray
    z: float
    color_value: float


HeightMode = Literal["G", "logG", "invG"]


def transform_value(g: float, mode: HeightMode) -> float:
    if mode == "G":
        return float(g)
    if mode == "logG":
        return float(np.log10(g))
    if mode == "invG":
        return float(1.0 / g)
    raise ValueError(f"Unknown mode: {mode}")


def resolve_inner_mode(user_inner_mode: str, height_mode: str) -> str:
    if user_inner_mode != "auto":
        return user_inner_mode
    return "cap" if height_mode == "G" else "hollow"


def build_contour_rings(
    x: np.ndarray,
    y: np.ndarray,
    G: np.ndarray,
    gmin: float,
    gmax: float,
    nlevels: int,
    ring_points: int,
    height_mode: HeightMode,
    color_mode: HeightMode,
    invert: bool,
    z_scale: float,
    smooth_passes: int,
    smooth_alpha: float,
    ring_align: str,
) -> list[Ring]:
    levels = np.geomspace(gmin, gmax, nlevels)
    rings: list[Ring] = []

    anchor = None
    prev_ring_xy = None

    for level in levels:
        raw = ensure_ccw(extract_primary_contour(x, y, G, level))

        if ring_align in ("seam", "global") and anchor is not None:
            raw = align_ring_to_anchor(raw, anchor)
        elif ring_align in ("seam", "global") and anchor is None:
            raw = align_ring_to_anchor(raw, None)
        elif ring_align == "none":
            pass
        else:
            raise ValueError(f"Unknown ring_align: {ring_align}")

        ring_xy = resample_closed_polyline_by_count(raw, ring_points)
        ring_xy = circular_smooth_ring(ring_xy, passes=smooth_passes, alpha=smooth_alpha)
        ring_xy = ensure_ccw(ring_xy)

        if ring_align == "global" and prev_ring_xy is not None:
            ring_xy = best_cyclic_shift_to_reference(prev_ring_xy, ring_xy)
        elif ring_align == "seam":
            ring_xy = align_ring_to_anchor(ring_xy, anchor)
        elif ring_align == "global" and prev_ring_xy is None:
            ring_xy = align_ring_to_anchor(ring_xy, None)

        anchor = ring_xy[0].copy()
        prev_ring_xy = ring_xy.copy()

        z = z_scale * transform_value(level, height_mode)
        if invert:
            z = -z

        cval = transform_value(level, color_mode)

        rings.append(Ring(
            level=float(level),
            xy=ring_xy,
            z=float(z),
            color_value=float(cval),
        ))

    return rings


# ============================================================
# Mesh construction
# ============================================================


def add_ring_vertices(vertices: list[tuple[float, float, float]], intensities: list[float], ring: Ring) -> list[int]:
    start = len(vertices)
    for p in ring.xy:
        vertices.append((float(p[0]), float(p[1]), float(ring.z)))
        intensities.append(float(ring.color_value))
    return list(range(start, start + len(ring.xy)))


def stitch_equal_count_rings(
    idx_inner: list[int],
    idx_outer: list[int],
    vertices: np.ndarray,
    tris: list[tuple[int, int, int]],
    quad_split: str = "planar",
) -> None:
    """
    Stitch two rings that have the same number of points.

    quad_split:
      - fixed: always use the same diagonal
      - alternate: alternate diagonals around the ring
      - planar: choose, per quad, the diagonal whose two triangle normals agree better
    """
    if len(idx_inner) != len(idx_outer):
        raise ValueError("Equal-count stitcher requires equal point counts on both rings.")

    n = len(idx_inner)
    for q in range(n):
        a0 = idx_inner[q]
        a1 = idx_inner[(q + 1) % n]
        b0 = idx_outer[q]
        b1 = idx_outer[(q + 1) % n]

        p0 = vertices[a0]
        p1 = vertices[a1]
        p2 = vertices[b0]
        p3 = vertices[b1]

        split = choose_quad_split_for_points(p0, p1, p2, p3, quad_split, q)

        if split == "diag_a0_b1":
            tris.append((a0, b0, b1))
            tris.append((a0, b1, a1))
        else:
            tris.append((a0, b0, a1))
            tris.append((a1, b0, b1))

def build_inner_cap_rings(first_ring: Ring, n_cap_rings: int, cap_height_mode: str, color_mode: HeightMode) -> tuple[list[Ring], np.ndarray, float, float]:
    """Build simple shrunken rings for a visual cap. Used by G mode only by default."""
    xy = first_ring.xy
    center = xy.mean(axis=0)

    if cap_height_mode == "zero":
        z = 0.0
    elif cap_height_mode == "match_cutoff":
        z = first_ring.z
    else:
        raise ValueError("cap_height_mode must be zero or match_cutoff")

    cval = first_ring.color_value if cap_height_mode == "match_cutoff" else transform_value(first_ring.level, color_mode)

    factors = np.linspace(0.12, 0.88, max(0, n_cap_rings))
    cap_rings: list[Ring] = []
    for f in factors:
        pts = center[None, :] + f * (xy - center[None, :])
        cap_rings.append(Ring(
            level=first_ring.level,
            xy=pts,
            z=float(z),
            color_value=float(cval),
        ))

    return cap_rings, center, float(z), float(cval)


def build_mesh_from_rings(
    contour_rings: list[Ring],
    inner_mode: str,
    n_cap_rings: int,
    cap_height_mode: str,
    color_mode: HeightMode,
    quad_split: str,
    report_normals: bool,
    debug_geometry: bool,
    debug_eps: float,
):
    if not contour_rings:
        raise ValueError("No contour rings supplied.")

    vertices: list[tuple[float, float, float]] = []
    intensities: list[float] = []
    tris: list[tuple[int, int, int]] = []

    if inner_mode == "cap":
        cap_rings, center, center_z, center_c = build_inner_cap_rings(
            contour_rings[0],
            n_cap_rings=n_cap_rings,
            cap_height_mode=cap_height_mode,
            color_mode=color_mode,
        )
        all_rings = cap_rings + contour_rings
    elif inner_mode == "hollow":
        all_rings = contour_rings
        center = None
        center_z = 0.0
        center_c = 0.0
    else:
        raise ValueError(f"Unknown inner_mode: {inner_mode}")

    ring_indices = [add_ring_vertices(vertices, intensities, ring) for ring in all_rings]
    V_partial = np.asarray(vertices, dtype=float)

    for ra, rb in zip(ring_indices[:-1], ring_indices[1:]):
        stitch_equal_count_rings(ra, rb, V_partial, tris, quad_split=quad_split)

    if debug_geometry:
        print("Band diagnostics:", band_diagnostics(V_partial, ring_indices, quad_split))

    if inner_mode == "cap" and center is not None and ring_indices:
        center_idx = len(vertices)
        vertices.append((float(center[0]), float(center[1]), float(center_z)))
        intensities.append(float(center_c))

        first = ring_indices[0]
        n = len(first)
        for q in range(n):
            tris.append((center_idx, first[q], first[(q + 1) % n]))

    V = np.asarray(vertices, dtype=float)
    I = np.asarray(intensities, dtype=float)
    T = orient_triangles_up(V, np.asarray(tris, dtype=int))

    if report_normals:
        print("Triangle orientation stats:", analyze_triangle_orientation(V, T))

    if debug_geometry:
        print("Mesh diagnostics:", geometry_diagnostics(V, T, eps=debug_eps))

    return V, I, T, all_rings


# ============================================================
# Plotly
# ============================================================


def unique_edges_from_triangles(tris: np.ndarray):
    edges = set()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((int(u), int(v)))
    return sorted(edges)


def colorbar_title(mode: str) -> str:
    if mode == "G":
        return "G(c)"
    if mode == "logG":
        return "log₁₀ G(c)"
    if mode == "invG":
        return "1 / G(c)"
    return mode


def build_figure(
    vertices: np.ndarray,
    intensities: np.ndarray,
    tris: np.ndarray,
    rings: list[Ring],
    color_mode: str,
    colorscale: str,
    wireframe: bool,
    show_contours: bool,
    contour_opacity: float,
    contour_color: str,
    contour_width: float,
    output_title: str,
    soft_lighting: bool,
    unlit: bool,
    flat_shading: bool,
):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    if unlit:
        lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    elif soft_lighting:
        lighting = dict(ambient=0.88, diffuse=0.25, specular=0.0, roughness=1.0, fresnel=0.0)
    else:
        lighting = dict(ambient=0.5, diffuse=0.8, specular=0.15, roughness=0.7, fresnel=0.05)

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=tris[:, 0],
        j=tris[:, 1],
        k=tris[:, 2],
        intensity=intensities,
        intensitymode="vertex",
        colorscale=colorscale,
        flatshading=flat_shading,
        showscale=True,
        colorbar=dict(title=colorbar_title(color_mode)),
        hovertemplate=(
            "Re(c)=%{x:.6f}<br>"
            "Im(c)=%{y:.6f}<br>"
            "height=%{z:.6f}<extra></extra>"
        ),
        lighting=lighting,
        lightposition=dict(x=80, y=80, z=140),
        name="potential mesh",
    ))

    if show_contours:
        for idx, ring in enumerate(rings):
            pts = ring.xy
            zz = np.full(len(pts) + 1, ring.z)
            xx = np.append(pts[:, 0], pts[0, 0])
            yy = np.append(pts[:, 1], pts[0, 1])

            fig.add_trace(go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="lines",
                line=dict(color=contour_color, width=contour_width),
                opacity=contour_opacity,
                hoverinfo="skip",
                showlegend=False,
            ))

    if wireframe:
        edges = unique_edges_from_triangles(tris)
        xe, ye, ze = [], [], []
        for a, b in edges:
            xe.extend([x[a], x[b], None])
            ye.extend([y[a], y[b], None])
            ze.extend([z[a], z[b], None])

        fig.add_trace(go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.28)", width=1),
            hoverinfo="skip",
            showlegend=False,
            name="wireframe",
        ))

    fig.update_layout(
        title=output_title,
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

    return fig


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Contour-stitched Mandelbrot potential mesh")

    # Domain / grid for potential computation
    parser.add_argument("--xmin", type=float, default=-2.5)
    parser.add_argument("--xmax", type=float, default=1.5)
    parser.add_argument("--ymin", type=float, default=-1.8)
    parser.add_argument("--ymax", type=float, default=1.8)
    parser.add_argument("--nx", type=int, default=12000)
    parser.add_argument("--ny", type=int, default=10500)

    # Potential computation
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--escape-radius", type=float, default=1e6)

    # Contour mesh controls
    parser.add_argument("--gmin", type=float, default=1e-4, help="inner cutoff contour")
    parser.add_argument("--gmax", type=float, default=0.25, help="outer cutoff contour")
    parser.add_argument("--nlevels", type=int, default=20, help="number of contour rings")
    parser.add_argument("--ring-points", type=int, default=2600, help="points per contour ring; same on every ring for robust stitching")
    parser.add_argument("--ring-align", choices=["global", "seam", "none"], default="global",
                        help="How to align seams between neighboring rings. 'global' minimizes ring-to-ring point distances.")
    parser.add_argument("--smooth-passes", type=int, default=1, help="gentle circular smoothing passes on resampled rings")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="smoothing strength per pass")

    # Visual modes
    parser.add_argument("--height-mode", choices=["G", "logG", "invG"], default="logG")
    parser.add_argument("--color-mode", choices=["G", "logG", "invG"], default="logG")
    parser.add_argument("--invert", "--flip-z", action="store_true", dest="invert",
                        help="Invert the vertical direction, e.g. plot -log10(G) instead of log10(G).")
    parser.add_argument("--z-scale", type=float, default=1.0,
                        help="Multiply all z-heights by this factor to squish or stretch the surface vertically.")
    parser.add_argument("--inner-mode", choices=["auto", "cap", "hollow"], default="auto")
    parser.add_argument("--cap-height-mode", choices=["zero", "match_cutoff"], default="zero")
    parser.add_argument("--colorscale", default="Turbo")
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--no-contours", action="store_true")
    parser.add_argument("--quad-split", choices=["fixed", "alternate", "planar"], default="planar",
                        help="How to split each quad between neighboring contour rings.")
    parser.add_argument("--report-normals", action="store_true",
                        help="Print a post-process report on triangle orientation consistency.")
    parser.add_argument("--debug-geometry", action="store_true",
                        help="Print topology diagnostics: duplicates, non-manifold edges, twisted quads, crossing radial edges.")
    parser.add_argument("--debug-eps", type=float, default=1e-12,
                        help="Quantization epsilon used by --debug-geometry for duplicate-vertex checks.")
    parser.add_argument("--contour-width", type=float, default=4.0,
                        help="Width of the overlaid contour lines.")
    parser.add_argument("--contour-color", default="#000000",
                        help="Color of the overlaid contour lines, e.g. '#000000', 'white', 'rgba(0,0,0,1)'.")
    parser.add_argument("--contour-opacity", type=float, default=1.0,
                        help="Opacity of the overlaid contour lines.")
    parser.add_argument("--hard-lighting", action="store_true", help="use stronger lighting; default is soft/color-driven lighting")
    parser.add_argument("--unlit", action="store_true",
                        help="Use ambient-only lighting. Best test for whether visible triangles are shading artifacts.")
    parser.add_argument("--flat-shading", action="store_true",
                        help="Force flat face shading. Useful only for debugging facets.")

    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "work" / "promote" / "mandelbrot" / "potential_surface.html"
        ),
    )

    args = parser.parse_args()

    if args.gmin <= 0:
        raise ValueError("--gmin must be positive")
    if args.gmax <= args.gmin:
        raise ValueError("--gmax must be larger than --gmin")
    if args.ring_points < 32:
        raise ValueError("--ring-points should be at least 32")
    if args.nlevels < 2:
        raise ValueError("--nlevels should be at least 2")
    if args.z_scale <= 0:
        raise ValueError("--z-scale must be positive")
    if not (0.0 <= args.contour_opacity <= 1.0):
        raise ValueError("--contour-opacity must lie between 0 and 1")
    if args.contour_width <= 0:
        raise ValueError("--contour-width must be positive")

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

    print("Building contour rings...")
    contour_rings = build_contour_rings(
        x=x,
        y=y,
        G=G,
        gmin=args.gmin,
        gmax=args.gmax,
        nlevels=args.nlevels,
        ring_points=args.ring_points,
        height_mode=args.height_mode,
        color_mode=args.color_mode,
        invert=args.invert,
        z_scale=args.z_scale,
        smooth_passes=args.smooth_passes,
        smooth_alpha=args.smooth_alpha,
        ring_align=args.ring_align,
    )

    resolved_inner_mode = resolve_inner_mode(args.inner_mode, args.height_mode)
    print(f"Inner mode resolved to: {resolved_inner_mode}")

    print("Stitching contour mesh...")
    vertices, intensities, tris, all_rings = build_mesh_from_rings(
        contour_rings=contour_rings,
        inner_mode=resolved_inner_mode,
        n_cap_rings=5,
        cap_height_mode=args.cap_height_mode,
        color_mode=args.color_mode,
        quad_split=args.quad_split,
        report_normals=args.report_normals,
        debug_geometry=args.debug_geometry,
        debug_eps=args.debug_eps,
    )

    print(f"Mesh: {len(vertices):,} vertices, {len(tris):,} triangles, {len(all_rings)} rings")

    title = (
        f"Mandelbrot potential contour mesh "
        f"({args.height_mode}{', inverted' if args.invert else ''}, z×{args.z_scale:g}; "
        f"{args.gmin:g} ≤ G ≤ {args.gmax:g})"
    )

    print("Building Plotly figure...")
    fig = build_figure(
        vertices=vertices,
        intensities=intensities,
        tris=tris,
        rings=all_rings,
        color_mode=args.color_mode,
        colorscale=args.colorscale,
        wireframe=args.wireframe,
        show_contours=not args.no_contours,
        contour_opacity=args.contour_opacity,
        contour_color=args.contour_color,
        contour_width=args.contour_width,
        output_title=title,
        soft_lighting=not args.hard_lighting,
        unlit=args.unlit,
        flat_shading=args.flat_shading,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing {args.output} ...")
    fig.write_html(args.output, include_plotlyjs="cdn")
    print("Done.")


if __name__ == "__main__":
    main()
