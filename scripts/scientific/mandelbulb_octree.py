#!/usr/bin/env python3
"""
Adaptive octree-ish Mandelbulb mesher.

This is an exploratory alternative to the full dense 3D volume pipeline:

    dense cube -> marching cubes -> simplify

Instead, it does:

    coarse grid cells
    -> keep only cells whose corners straddle inside/outside
    -> recursively subdivide those boundary cells
    -> polygonize final boundary cells with marching tetrahedra
    -> optional smoothing/simplification
    -> PLY + Plotly HTML

Notes / caveats:
- This is a practical visual mesher, not a certified fractal solver.
- A cell is refined if its corners disagree. Optional center sampling helps catch
  some features that corners miss, but very thin islands can still be missed.
- The final mesh uses a per-leaf marching-tetrahedra surface. This avoids storing
  a full fine grid, but adaptive cells can still produce some T-junction-ish
  artifacts. Use smoothing/simplification for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import math
import time

import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm
import open3d as o3d


@dataclass
class OctreeConfig:
    powers: tuple[int, ...] = (2, 3, 4)
    base_resolution: int = 64
    max_depth: int = 4
    extent: float = 1.4
    max_iter: int = 50
    bailout: float = 2.0

    # Chunking and detection.
    cell_chunk_size: int = 200_000
    point_batch_size: int = 1_000_000
    use_center_probe: bool = True

    # Mesh cleanup / display.
    merge_close_factor: float = 0.05
    taubin_iterations: int = 0
    target_triangles: int | None = 250_000
    simplification_factor: float = 0.75
    html_float_precision: int | None = 6
    write_html: bool = True
    write_ply: bool = True
    output_prefix: str = "mandelbulb_octree"


# Cube corners, in the order used by marching tetrahedra below.
CORNER_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

# Six tetrahedra sharing the body diagonal 0 -> 7.
TETS = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)

TET_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def mandelbulb_inside_points(points: np.ndarray, power: int, max_iter: int, bailout: float) -> np.ndarray:
    """
    Classify parameter points A=(x,y,z) as bounded/unbounded under a Mandelbulb map.

    points: shape (N, 3), float32/float64.
    returns: bool array, True = did not escape within max_iter.
    """

    A = points.astype(np.float32, copy=False)
    X = A[:, 0]
    Y = A[:, 1]
    Z = A[:, 2]

    x = np.zeros_like(X, dtype=np.float32)
    y = np.zeros_like(Y, dtype=np.float32)
    z = np.zeros_like(Z, dtype=np.float32)

    active = np.ones(len(points), dtype=bool)
    bailout_f = np.float32(bailout)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            r = np.sqrt(x * x + y * y + z * z)
            escaped = ((r > bailout_f) | (~np.isfinite(r))) & active
            active[escaped] = False

            if not np.any(active):
                break

            idx = active
            r_i = r[idx]

            # For r=0, theta/phi are harmlessly 0.
            theta = np.arctan2(np.sqrt(x[idx] * x[idx] + y[idx] * y[idx]), z[idx])
            phi = np.arctan2(y[idx], x[idx])

            rn = r_i ** np.float32(power)
            theta_n = theta * np.float32(power)
            phi_n = phi * np.float32(power)

            x_new = rn * np.sin(theta_n) * np.cos(phi_n) + X[idx]
            y_new = rn * np.sin(theta_n) * np.sin(phi_n) + Y[idx]
            z_new = rn * np.cos(theta_n) + Z[idx]

            x[idx] = x_new
            y[idx] = y_new
            z[idx] = z_new

    return active


class CornerCache:
    def __init__(self, fine_cells: int, extent: float, power: int, max_iter: int, bailout: float, batch_size: int):
        self.fine_cells = int(fine_cells)
        self.stride = int(fine_cells + 1)
        self.stride2 = self.stride * self.stride
        self.extent = float(extent)
        self.power = int(power)
        self.max_iter = int(max_iter)
        self.bailout = float(bailout)
        self.batch_size = int(batch_size)
        self.cache: dict[int, bool] = {}

    def pack_keys(self, idx: np.ndarray) -> np.ndarray:
        idx64 = idx.astype(np.int64, copy=False)
        return idx64[:, 0] * self.stride2 + idx64[:, 1] * self.stride + idx64[:, 2]

    def indices_to_world(self, idx: np.ndarray) -> np.ndarray:
        # Map integer fine-grid coordinates 0..fine_cells to [-extent, extent].
        return (-self.extent + (2.0 * self.extent) * idx.astype(np.float32) / np.float32(self.fine_cells)).astype(np.float32)

    def get(self, idx: np.ndarray) -> np.ndarray:
        """Return inside/outside values for integer fine-grid indices, using a Python dict cache."""
        if len(idx) == 0:
            return np.empty(0, dtype=bool)

        idx = np.asarray(idx, dtype=np.int64)
        keys = self.pack_keys(idx)

        missing_mask = np.fromiter((int(k) not in self.cache for k in keys), dtype=bool, count=len(keys))
        if np.any(missing_mask):
            missing_idx = idx[missing_mask]
            missing_keys = keys[missing_mask]

            # Unique missing points so repeated corners in the same request are only evaluated once.
            unique_keys, unique_pos = np.unique(missing_keys, return_index=True)
            unique_idx = missing_idx[unique_pos]

            for start in range(0, len(unique_idx), self.batch_size):
                stop = min(len(unique_idx), start + self.batch_size)
                pts = self.indices_to_world(unique_idx[start:stop])
                vals = mandelbulb_inside_points(
                    pts,
                    power=self.power,
                    max_iter=self.max_iter,
                    bailout=self.bailout,
                )
                for k, v in zip(unique_keys[start:stop], vals):
                    self.cache[int(k)] = bool(v)

        return np.fromiter((self.cache[int(k)] for k in keys), dtype=bool, count=len(keys))


def chunked_iter_array(arr: np.ndarray, chunk_size: int):
    for start in range(0, len(arr), chunk_size):
        yield arr[start : start + chunk_size]


def classify_cells(cells: np.ndarray, size_units: int, cfg: OctreeConfig, cache: CornerCache):
    """
    Classify cells as boundary candidates.

    cells are integer origins in fine-grid units. size_units is the cell edge length
    in fine-grid units. Returns a boolean mask for mixed cells and the corner values.
    """
    cells = np.asarray(cells, dtype=np.int64)
    corners = cells[:, None, :] + CORNER_OFFSETS[None, :, :] * int(size_units)
    flat_corners = corners.reshape(-1, 3)
    corner_vals = cache.get(flat_corners).reshape(len(cells), 8)

    any_inside = np.any(corner_vals, axis=1)
    any_outside = np.any(~corner_vals, axis=1)
    mixed = any_inside & any_outside

    # Center probe can catch some cells whose boundary slips between corners.
    # It only affects refinement, not final tetra polygonization directly.
    if cfg.use_center_probe and size_units >= 2:
        center = cells + int(size_units // 2)
        center_vals = cache.get(center)
        all_corners_inside = np.all(corner_vals, axis=1)
        all_corners_outside = np.all(~corner_vals, axis=1)
        mixed |= (all_corners_inside & (~center_vals)) | (all_corners_outside & center_vals)

    return mixed, corner_vals


def initial_base_cells(base_resolution: int, initial_size_units: int) -> np.ndarray:
    coords = np.arange(base_resolution, dtype=np.int64) * int(initial_size_units)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.int64)


def subdivide_cells(cells: np.ndarray, child_size_units: int) -> np.ndarray:
    shifts = CORNER_OFFSETS * int(child_size_units)
    children = cells[:, None, :] + shifts[None, :, :]
    return children.reshape(-1, 3).astype(np.int64)


def build_boundary_leaves(cfg: OctreeConfig, power: int) -> tuple[np.ndarray, int, CornerCache]:
    """Build adaptive boundary leaf cells."""
    fine_cells = int(cfg.base_resolution) * (2 ** int(cfg.max_depth))
    initial_size = 2 ** int(cfg.max_depth)

    print(f"Power: {power}")
    print(f"Base resolution: {cfg.base_resolution}³")
    print(f"Max depth: {cfg.max_depth}")
    print(f"Effective finest grid: {fine_cells}³")
    print(f"Extent: [-{cfg.extent}, {cfg.extent}]")

    cache = CornerCache(
        fine_cells=fine_cells,
        extent=cfg.extent,
        power=power,
        max_iter=cfg.max_iter,
        bailout=cfg.bailout,
        batch_size=cfg.point_batch_size,
    )

    cells = initial_base_cells(cfg.base_resolution, initial_size)
    size = initial_size

    for level in range(cfg.max_depth + 1):
        print(f"\nLevel {level}/{cfg.max_depth}: cell size = {size} fine units, candidate cells = {len(cells):,}")

        kept_parts = []
        pbar = tqdm(total=len(cells), desc=f"Classifying level {level}", unit=" cell")
        for chunk in chunked_iter_array(cells, cfg.cell_chunk_size):
            mixed, _ = classify_cells(chunk, size, cfg, cache)
            if np.any(mixed):
                kept_parts.append(chunk[mixed])
            pbar.update(len(chunk))
        pbar.close()

        cells = np.concatenate(kept_parts) if kept_parts else np.empty((0, 3), dtype=np.int64)
        print(f"Boundary/mixed cells kept: {len(cells):,}")
        print(f"Cached corner/probe samples: {len(cache.cache):,}")

        if level == cfg.max_depth or len(cells) == 0:
            break

        size //= 2
        cells = subdivide_cells(cells, size)

    return cells, size, cache


def tet_intersections(points4: np.ndarray, inside4: np.ndarray):
    """Return triangle vertices generated by a tetrahedron with binary signs."""
    # Scalar values: inside=+1, outside=-1. Iso-level 0, so interpolation is midpoint.
    pts = []
    for a, b in TET_EDGES:
        if inside4[a] != inside4[b]:
            # Binary signs -> midpoint. This is crude but robust.
            pts.append(0.5 * (points4[a] + points4[b]))

    if len(pts) == 3:
        return [pts]
    if len(pts) == 4:
        return [[pts[0], pts[1], pts[2]], [pts[0], pts[2], pts[3]]]
    return []


def polygonize_cells(cells: np.ndarray, size_units: int, cfg: OctreeConfig, cache: CornerCache):
    """Create a triangle mesh from final adaptive boundary cells using marching tetrahedra."""
    print("\nPolygonizing final boundary cells with marching tetrahedra...")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    emitted = 0

    for chunk in tqdm(list(chunked_iter_array(cells, cfg.cell_chunk_size)), desc="Polygonizing chunks", unit=" chunk"):
        mixed, corner_vals = classify_cells(chunk, size_units, cfg, cache)
        if not np.any(mixed):
            continue

        chunk = chunk[mixed]
        corner_vals = corner_vals[mixed]

        for origin, vals in zip(chunk, corner_vals):
            corner_idx = origin[None, :] + CORNER_OFFSETS * int(size_units)
            corner_pos = cache.indices_to_world(corner_idx).astype(np.float64)

            for tet in TETS:
                tet = np.asarray(tet, dtype=np.int64)
                tris = tet_intersections(corner_pos[tet], vals[tet])
                for tri in tris:
                    i0 = len(vertices)
                    vertices.extend([tri[0].tolist(), tri[1].tolist(), tri[2].tolist()])
                    faces.append([i0, i0 + 1, i0 + 2])
                    emitted += 1

    if not vertices or not faces:
        raise RuntimeError("No triangles generated. Try larger base/depth, larger extent, or different bailout.")

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)
    print(f"Raw adaptive mesh before cleanup: {len(vertices_arr):,} vertices, {len(faces_arr):,} triangles")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_arr)
    mesh.triangles = o3d.utility.Vector3iVector(faces_arr)

    cleanup_mesh(mesh, cfg)
    return mesh


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh, cfg: OctreeConfig):
    """In-place-ish cleanup and optional smoothing."""
    fine_cells = int(cfg.base_resolution) * (2 ** int(cfg.max_depth))
    dx = (2.0 * cfg.extent) / float(fine_cells)
    merge_eps = max(0.0, cfg.merge_close_factor) * dx

    if merge_eps > 0:
        try:
            mesh.merge_close_vertices(float(merge_eps))
            print(f"Merged close vertices with eps={merge_eps:.6g}")
        except Exception as exc:
            print(f"merge_close_vertices failed/skipped: {exc}")

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

    if cfg.taubin_iterations > 0:
        print(f"Applying Taubin smoothing: {cfg.taubin_iterations} iterations")
        mesh2 = mesh.filter_smooth_taubin(number_of_iterations=int(cfg.taubin_iterations))
        mesh.vertices = mesh2.vertices
        mesh.triangles = mesh2.triangles
        mesh.compute_vertex_normals()

    print(f"Mesh after cleanup: {len(np.asarray(mesh.vertices)):,} vertices, {len(np.asarray(mesh.triangles)):,} triangles")


def simplify_mesh_progressive(mesh: o3d.geometry.TriangleMesh, target_triangles: int, reduction_factor: float):
    if target_triangles is None:
        return mesh
    if not (0 < reduction_factor < 1):
        raise ValueError("reduction_factor must be between 0 and 1")

    current = len(np.asarray(mesh.triangles))
    target_triangles = min(int(target_triangles), current)

    if current <= target_triangles:
        return mesh

    print(f"Simplifying mesh: {current:,} -> {target_triangles:,} triangles, factor={reduction_factor:.3f}")
    simplified = mesh
    stage = 0

    with tqdm(desc="Simplification", unit=" stage") as pbar:
        while len(np.asarray(simplified.triangles)) > target_triangles:
            before = len(np.asarray(simplified.triangles))
            next_target = max(target_triangles, int(before * reduction_factor))
            t0 = time.perf_counter()
            next_mesh = simplified.simplify_quadric_decimation(next_target)
            next_mesh.remove_degenerate_triangles()
            next_mesh.remove_duplicated_triangles()
            next_mesh.remove_duplicated_vertices()
            next_mesh.remove_unreferenced_vertices()
            next_mesh.remove_non_manifold_edges()
            try:
                next_mesh.orient_triangles()
            except Exception:
                pass
            next_mesh.compute_vertex_normals()
            after = len(np.asarray(next_mesh.triangles))
            stage += 1
            tqdm.write(f"  stage {stage:02d}: {before:,} -> {after:,} triangles in {time.perf_counter() - t0:.1f}s")
            simplified = next_mesh
            pbar.update(1)
            if after >= before:
                print("No further simplification achieved; stopping.")
                break

    return simplified


def plot_mesh_html(mesh: o3d.geometry.TriangleMesh, output_html: Path, html_float_precision: int | None = 6):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if html_float_precision is not None:
        verts = np.round(verts.astype(np.float64, copy=False), int(html_float_precision))
        print(f"Rounded vertices to {html_float_precision} decimals for HTML")

    radius = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
    cmin = float(np.nanmin(radius))
    cmax = float(np.nanmax(radius))
    if np.isclose(cmin, cmax):
        cmin -= 1e-6
        cmax += 1e-6

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
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                showscale=False,
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

    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"Wrote {output_html}")


def run_one_power(cfg: OctreeConfig, power: int):
    t0 = time.perf_counter()
    print("=" * 80)
    print(f"Running adaptive Mandelbulb power={power}")

    leaves, size_units, cache = build_boundary_leaves(cfg, power=power)
    if len(leaves) == 0:
        print(f"No boundary leaves found for power={power}; skipping.")
        return

    mesh = polygonize_cells(leaves, size_units, cfg, cache)

    if cfg.target_triangles is not None:
        mesh = simplify_mesh_progressive(mesh, cfg.target_triangles, cfg.simplification_factor)

    stem = f"{cfg.output_prefix}_p{power}_base{cfg.base_resolution}_d{cfg.max_depth}"
    if cfg.write_ply:
        ply_path = Path(f"{stem}.ply")
        o3d.io.write_triangle_mesh(str(ply_path), mesh)
        print(f"Wrote {ply_path}")

    if cfg.write_html:
        html_path = Path(f"{stem}.html")
        plot_mesh_html(mesh, html_path, html_float_precision=cfg.html_float_precision)

    dt = time.perf_counter() - t0
    print(f"Finished power={power} in {dt / 60:.2f} min")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive octree-ish Mandelbulb mesher")
    parser.add_argument("--powers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--base", type=int, default=64, help="base grid resolution, e.g. 64 or 128")
    parser.add_argument("--depth", type=int, default=4, help="subdivision depth beyond base grid")
    parser.add_argument("--extent", type=float, default=1.4)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--bailout", type=float, default=2.0)
    parser.add_argument("--chunk", type=int, default=200_000, help="cell chunk size")
    parser.add_argument("--point-batch", type=int, default=1_000_000)
    parser.add_argument("--target", type=int, default=250_000, help="target triangles after simplification; <=0 disables")
    parser.add_argument("--simplification-factor", type=float, default=0.75)
    parser.add_argument("--taubin", type=int, default=0)
    parser.add_argument("--prefix", default="mandelbulb_octree")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--no-center-probe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OctreeConfig(
        powers=tuple(args.powers),
        base_resolution=args.base,
        max_depth=args.depth,
        extent=args.extent,
        max_iter=args.max_iter,
        bailout=args.bailout,
        cell_chunk_size=args.chunk,
        point_batch_size=args.point_batch,
        use_center_probe=not args.no_center_probe,
        target_triangles=None if args.target <= 0 else args.target,
        simplification_factor=args.simplification_factor,
        taubin_iterations=args.taubin,
        write_html=not args.no_html,
        write_ply=not args.no_ply,
        output_prefix=args.prefix,
    )

    for power in cfg.powers:
        run_one_power(cfg, power)


if __name__ == "__main__":
    main()
