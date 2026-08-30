#!/usr/bin/env python3
"""
Mandelbrot escape-potential surface using robust contours from mandelbrot_tools.

This is the simplified replacement for the older experimental Gmsh contour script:

  1. Trace inner/outer equipotential contours G(c)=gmin/gmax with RK4 tangent flow
     from mandelbrot_tools.
  2. Feed those two loops to Gmsh as an annular planar region.
  3. Let Gmsh triangulate the annulus.
  4. Evaluate G at mesh vertices.
  5. Lift vertices to z = height_mode(G).
  6. Render with Plotly Mesh3d.

Expected local files:

    mandelbrot_tools.py

The module must provide the RK contour functions added earlier:

    trace_mandelbrot_contour(...)
    mandelbrot_potential(...)

Install dependencies:

    pip install numpy scipy plotly gmsh

On Linux/WSL, if gmsh import fails with a libGL/libGLU error:

    sudo apt install libglu1-mesa
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from mandelbrot_tools import (
        BAILOUT_DEFAULT,
        DERIVATIVE_EPSILON_DEFAULT,
        DERIVATIVE_MAX_ITER_DEFAULT,
        MAX_ITER_DEFAULT,
        POTENTIAL_EPSILON_DEFAULT,
        ROOT_EPSILON_DEFAULT,
        mandelbrot_potential,
        trace_mandelbrot_contour,
    )
except ImportError:
    # Handy if you have not renamed the downloaded helper file yet.
    from mandelbrot_tools_with_contours import (  # type: ignore
        BAILOUT_DEFAULT,
        DERIVATIVE_EPSILON_DEFAULT,
        DERIVATIVE_MAX_ITER_DEFAULT,
        MAX_ITER_DEFAULT,
        POTENTIAL_EPSILON_DEFAULT,
        ROOT_EPSILON_DEFAULT,
        mandelbrot_potential,
        trace_mandelbrot_contour,
    )


HeightMode = Literal["G", "logG", "minusLogG", "invG"]


# -----------------------------------------------------------------------------
# Basic polyline helpers
# -----------------------------------------------------------------------------


def polygon_area(points: np.ndarray) -> float:
    """Signed polygon area. Positive means CCW."""

    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts[::-1].copy() if polygon_area(pts) < 0.0 else pts.copy()


def ensure_cw(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts[::-1].copy() if polygon_area(pts) > 0.0 else pts.copy()


def close_if_needed(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        return np.vstack([pts, pts[0]])
    return pts


def drop_closure_duplicate(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        return pts[:-1].copy()
    return pts.copy()


def dedupe_consecutive(points: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()

    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)

    return np.asarray(out, dtype=float)


def polyline_length_closed(points: np.ndarray) -> float:
    pts = close_if_needed(points)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs * diffs).sum(axis=1))))


def resample_closed_polyline_by_count(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample a closed polyline to exactly n_samples points.

    Returns an open ring: the first point is not repeated at the end.
    """

    n_samples = int(n_samples)
    if n_samples <= 0:
        return drop_closure_duplicate(points)

    pts = close_if_needed(np.asarray(points, dtype=float))
    if len(pts) < 4:
        raise ValueError("Need at least 3 points to resample a closed contour.")

    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs * diffs).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])

    if total <= 0.0:
        raise ValueError("Degenerate contour with zero length.")

    target = np.linspace(0.0, total, n_samples, endpoint=False)
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])
    return np.column_stack([x, y])


def format_g_level(level: float) -> str:
    """Human-friendly G-level for filenames."""

    level = float(level)
    if level == 0.0:
        return "0"

    abs_level = abs(level)
    if 1e-6 <= abs_level < 1e6:
        s = f"{level:.12f}".rstrip("0").rstrip(".")
    else:
        s = f"{level:.12g}"

    return s.replace("+", "")


def save_contour_json(level: float, points: np.ndarray, out_dir: str | Path | None) -> None:
    if out_dir is None:
        return

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    fname = f"{format_g_level(level)}.json"
    out = path / fname

    pts = np.asarray(points, dtype=float)
    data = [[float(x), float(y)] for x, y in pts]
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  saved contour G={level:g}: {out} ({len(pts):,} points)", flush=True)


# -----------------------------------------------------------------------------
# Contours from mandelbrot_tools
# -----------------------------------------------------------------------------


def ds_for_level(level: float, gmin: float, gmax: float, inner_ds: float, outer_ds: float) -> float:
    """Geometric interpolation of contour step size between gmin and gmax."""

    level = float(level)
    gmin = float(gmin)
    gmax = float(gmax)

    if gmin <= 0.0 or gmax <= 0.0 or gmax <= gmin:
        return float(inner_ds)

    t = (math.log(level) - math.log(gmin)) / (math.log(gmax) - math.log(gmin))
    t = max(0.0, min(1.0, t))
    return float(inner_ds) * (float(outer_ds) / float(inner_ds)) ** t


def trace_xy_contour(level: float, args: argparse.Namespace) -> np.ndarray:
    """Trace one Mandelbrot equipotential and return an open xy ring."""

    ds = ds_for_level(level, args.gmin, args.gmax, args.inner_ds, args.outer_ds)
    print(
        f"Tracing G={level:g} with RK4 contour flow: ds≈{ds:g}, "
        f"max_turn_angle={args.max_turn_angle:g} rad",
        flush=True,
    )

    path = trace_mandelbrot_contour(
        target_G=float(level),
        derivative_epsilon=args.derivative_epsilon,
        derivative_max_iter=args.derivative_max_iter,
        escape_epsilon=args.escape_epsilon,
        escape_max_iter=args.escape_max_iter,
        root_epsilon=args.root_epsilon,
        power=2,
        bailout=args.trace_bailout,
        ds=ds,
        max_turn_angle=args.max_turn_angle,
        tol_G=args.root_epsilon,
        max_steps=args.trace_max_steps,
        use_y_symmetry=not args.no_symmetry,
        upper_half=True,
        close_loop=True,
        project_each_step=not args.no_project_each_step,
        max_step_halvings=args.max_step_halvings,
    )

    pts = np.asarray([(x, y) for x, y, _g in path], dtype=float)
    pts = drop_closure_duplicate(pts)
    pts = dedupe_consecutive(pts)

    print(
        f"  contour G={level:g}: {len(pts):,} points, "
        f"length≈{polyline_length_closed(pts):.8g}, area≈{polygon_area(pts):.8g}",
        flush=True,
    )

    return pts


# -----------------------------------------------------------------------------
# G evaluation on mesh vertices
# -----------------------------------------------------------------------------


def compute_mandelbrot_potential_points_vectorized(
    xy: np.ndarray,
    max_iter: int,
    escape_radius: float,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Fast vectorized approximation of G(c) on arbitrary points.

    This uses first escape past a large escape_radius:

        G ≈ 2^{-n} log |z_n|.

    For mesh-height evaluation this is usually good enough, and boundary values
    are clamped afterwards to exact gmin/gmax.
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
        local_indices = np.arange(len(C))

        with np.errstate(over="ignore", invalid="ignore"):
            for n in range(1, int(max_iter) + 1):
                if not np.any(active):
                    break

                Z[active] = Z[active] * Z[active] + C[active]
                abs_active = np.abs(Z[active])
                newly_local = abs_active > escape_radius

                if np.any(newly_local):
                    idx = local_indices[active][newly_local]
                    G[idx] = np.ldexp(np.log(abs_active[newly_local]), -n)

                    tmp = np.zeros_like(active)
                    tmp[active] = newly_local
                    active[tmp] = False

        out[start:end] = G
        print(f"  G eval: {end:,}/{len(xy):,} vertices", flush=True)

    return out


def compute_mandelbrot_potential_points_scalar(
    xy: np.ndarray,
    epsilon: float,
    max_iter: int,
    bailout: float,
) -> np.ndarray:
    """Slower but module-consistent scalar G evaluation."""

    xy = np.asarray(xy, dtype=float)
    out = np.empty(len(xy), dtype=float)

    for i, (x, y) in enumerate(xy):
        out[i] = mandelbrot_potential(
            float(x),
            float(y),
            epsilon=epsilon,
            max_iter=max_iter,
            power=2,
            bailout=bailout,
            use_quadratic_interior_tests=False,
        )
        if (i + 1) % 10_000 == 0 or i + 1 == len(xy):
            print(f"  G eval: {i + 1:,}/{len(xy):,} vertices", flush=True)

    return out


def clamp_boundary_g_values(
    xy: np.ndarray,
    Gv: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    gmin: float,
    gmax: float,
    tol: float,
) -> np.ndarray:
    """Force mesh nodes close to boundary rings to the exact boundary levels."""

    if tol <= 0.0:
        return Gv

    xy = np.asarray(xy, dtype=float)
    out = np.asarray(Gv, dtype=float).copy()

    inner_tree = cKDTree(np.asarray(inner, dtype=float))
    outer_tree = cKDTree(np.asarray(outer, dtype=float))

    d_inner, _ = inner_tree.query(xy, k=1)
    d_outer, _ = outer_tree.query(xy, k=1)

    inner_mask = d_inner <= tol
    outer_mask = d_outer <= tol

    out[inner_mask] = float(gmin)
    out[outer_mask] = float(gmax)

    print(
        f"Boundary clamp: {int(inner_mask.sum()):,} inner nodes, "
        f"{int(outer_mask.sum()):,} outer nodes within tol={tol:g}",
        flush=True,
    )

    return out


# -----------------------------------------------------------------------------
# Gmsh meshing
# -----------------------------------------------------------------------------


@dataclass
class GmshMesh:
    xy: np.ndarray
    tris: np.ndarray


def add_gmsh_loop(gmsh, points: np.ndarray, mesh_size: float) -> int:
    """Add a closed polyline loop to gmsh.model.geo and return curve loop tag."""

    pts = drop_closure_duplicate(np.asarray(points, dtype=float))
    point_tags: list[int] = []

    for x, y in pts:
        point_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(mesh_size)))

    line_tags: list[int] = []
    n = len(point_tags)
    for i in range(n):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n]))

    return gmsh.model.geo.addCurveLoop(line_tags)


def build_gmsh_annulus_mesh(
    outer_ccw: np.ndarray,
    inner_cw: np.ndarray,
    outer_mesh_size: float,
    inner_mesh_size: float,
    mesh_size_min: float,
    mesh_size_max: float,
    algorithm: int,
    optimize: bool,
    verbose: bool,
    save_msh: str | None,
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

        outer_loop = add_gmsh_loop(gmsh, outer_ccw, outer_mesh_size)
        inner_loop = add_gmsh_loop(gmsh, inner_cw, inner_mesh_size)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)

        if optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                gmsh.model.mesh.optimize()

        if save_msh:
            gmsh.write(str(save_msh))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        xy = coords[:, :2]
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tris: list[tuple[int, int, int]] = []

        for elem_type, nodes in zip(elem_types, elem_node_tags):
            # Gmsh element type 2 is a 3-node triangle.
            if int(elem_type) != 2:
                continue

            arr = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            for a, b, c in arr:
                tris.append((tag_to_idx[int(a)], tag_to_idx[int(b)], tag_to_idx[int(c)]))

        if not tris:
            raise RuntimeError("Gmsh produced no linear triangle elements.")

        return GmshMesh(xy=xy, tris=np.asarray(tris, dtype=int))

    finally:
        gmsh.finalize()


# -----------------------------------------------------------------------------
# Plotly rendering
# -----------------------------------------------------------------------------


def transform_g(G: np.ndarray, mode: HeightMode) -> np.ndarray:
    g = np.asarray(G, dtype=float)
    g = np.maximum(g, 1e-300)

    if mode == "G":
        return g
    if mode == "logG":
        return np.log10(g)
    if mode == "minusLogG":
        return -np.log10(g)
    if mode == "invG":
        return 1.0 / g

    raise ValueError(f"Unknown height/color mode: {mode}")


def colorbar_title(mode: HeightMode) -> str:
    if mode == "G":
        return "G(c)"
    if mode == "logG":
        return "log₁₀ G(c)"
    if mode == "minusLogG":
        return "−log₁₀ G(c)"
    if mode == "invG":
        return "1/G(c)"
    return str(mode)


def orient_triangles_up(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tris = np.asarray(tris, dtype=int).copy()
    for idx, (a, b, c) in enumerate(tris):
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        n = np.cross(p1 - p0, p2 - p0)
        if n[2] < 0.0:
            tris[idx] = (a, c, b)
    return tris


def build_plotly_figure(
    xy: np.ndarray,
    tris: np.ndarray,
    Gv: np.ndarray,
    contours: list[tuple[float, np.ndarray]],
    height_mode: HeightMode,
    color_mode: HeightMode,
    z_scale: float,
    invert_z: bool,
    colorscale: str,
    contour_color: str,
    contour_width: float,
    title: str,
    unlit: bool,
    flat_shading: bool,
) -> go.Figure:
    H = float(z_scale) * transform_g(Gv, height_mode)
    if invert_z:
        H = -H

    intensity = transform_g(Gv, color_mode)
    vertices = np.column_stack([xy[:, 0], xy[:, 1], H])
    tris = orient_triangles_up(vertices, tris)

    if unlit:
        lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    else:
        lighting = dict(ambient=0.92, diffuse=0.12, specular=0.0, roughness=1.0, fresnel=0.0)

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
        customdata=Gv,
        colorscale=colorscale,
        flatshading=flat_shading,
        showscale=True,
        colorbar=dict(title=colorbar_title(color_mode)),
        hovertemplate=(
            "Re(c)=%{x:.6f}<br>"
            "Im(c)=%{y:.6f}<br>"
            "G(c)=%{customdata:.6g}<br>"
            "z=%{z:.6f}<extra></extra>"
        ),
        lighting=lighting,
        lightposition=dict(x=80, y=80, z=140),
        name="Mandelbrot potential annulus",
    ))

    for level, pts in contours:
        closed = close_if_needed(pts)
        g_line = np.full(len(closed), float(level))
        z_line = float(z_scale) * transform_g(g_line, height_mode)
        if invert_z:
            z_line = -z_line

        fig.add_trace(go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=z_line,
            mode="lines",
            line=dict(color=contour_color, width=contour_width),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=42, b=0),
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Mandelbrot escape-potential annulus mesh using robust RK contours."
    )

    parser.add_argument("--gmin", type=float, default=1e-3, help="Inner contour G level.")
    parser.add_argument("--gmax", type=float, default=0.25, help="Outer contour G level.")

    parser.add_argument("--inner-ds", type=float, default=0.002, help="Base arclength step for the inner contour.")
    parser.add_argument("--outer-ds", type=float, default=0.02, help="Base arclength step for the outer contour.")
    parser.add_argument("--max-turn-angle", type=float, default=0.04, help="Maximum tangent turn per RK step, in radians.")
    parser.add_argument("--trace-max-steps", type=int, default=300_000)
    parser.add_argument("--max-step-halvings", type=int, default=24)
    parser.add_argument("--no-project-each-step", action="store_true")
    parser.add_argument("--no-symmetry", action="store_true", help="Trace full loops instead of tracing half and mirroring.")

    parser.add_argument("--derivative-epsilon", type=float, default=DERIVATIVE_EPSILON_DEFAULT)
    parser.add_argument("--derivative-max-iter", type=int, default=DERIVATIVE_MAX_ITER_DEFAULT)
    parser.add_argument("--escape-epsilon", type=float, default=POTENTIAL_EPSILON_DEFAULT)
    parser.add_argument("--escape-max-iter", type=int, default=MAX_ITER_DEFAULT)
    parser.add_argument("--root-epsilon", type=float, default=ROOT_EPSILON_DEFAULT)
    parser.add_argument("--trace-bailout", type=float, default=BAILOUT_DEFAULT)

    parser.add_argument("--inner-boundary-points", type=int, default=0,
                        help="If >0, resample inner ring to this many points before Gmsh. Default keeps raw traced points.")
    parser.add_argument("--outer-boundary-points", type=int, default=0,
                        help="If >0, resample outer ring to this many points before Gmsh. Default keeps raw traced points.")

    parser.add_argument("--inner-mesh-size", type=float, default=0.006)
    parser.add_argument("--outer-mesh-size", type=float, default=0.035)
    parser.add_argument("--mesh-size-min", type=float, default=0.003)
    parser.add_argument("--mesh-size-max", type=float, default=0.08)
    parser.add_argument("--gmsh-algorithm", type=int, default=6)
    parser.add_argument("--gmsh-verbose", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--save-msh", default="")

    parser.add_argument("--g-eval", choices=["vectorized", "scalar"], default="vectorized")
    parser.add_argument("--eval-max-iter", type=int, default=5000)
    parser.add_argument("--eval-escape-radius", type=float, default=1e6)
    parser.add_argument("--eval-epsilon", type=float, default=POTENTIAL_EPSILON_DEFAULT)
    parser.add_argument("--eval-batch-size", type=int, default=50_000)
    parser.add_argument("--boundary-clamp-tol", type=float, default=1e-8)

    parser.add_argument("--height-mode", choices=["G", "logG", "minusLogG", "invG"], default="minusLogG")
    parser.add_argument("--color-mode", choices=["G", "logG", "minusLogG", "invG"], default="minusLogG")
    parser.add_argument("--z-scale", type=float, default=0.15)
    parser.add_argument("--invert-z", action="store_true")
    parser.add_argument("--colorscale", default="Viridis")
    parser.add_argument("--unlit", action="store_true")
    parser.add_argument("--flat-shading", action="store_true")

    parser.add_argument("--overlay-contours", type=int, default=0,
                        help="Trace and draw this many log-spaced intermediate contours. 0 draws only inner/outer.")
    parser.add_argument("--contour-color", default="black")
    parser.add_argument("--contour-width", type=float, default=3.0)
    parser.add_argument("--save-contours-dir", default="")

    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "work"
            / "promote"
            / "mandelbrot"
            / "potential_surface_tools_gmsh.html"
        ),
    )
    parser.add_argument("--title", default="Mandelbrot escape-potential surface")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gmin <= 0.0 or args.gmax <= 0.0:
        raise ValueError("gmin and gmax must be positive exterior potential levels.")
    if args.gmin >= args.gmax:
        raise ValueError("Expected gmin < gmax: inner contour should be closer to the boundary.")

    save_contours_dir = Path(args.save_contours_dir) if args.save_contours_dir else None

    inner = trace_xy_contour(args.gmin, args)
    outer = trace_xy_contour(args.gmax, args)

    save_contour_json(args.gmin, inner, save_contours_dir)
    save_contour_json(args.gmax, outer, save_contours_dir)

    if args.inner_boundary_points > 0:
        before = len(inner)
        inner = resample_closed_polyline_by_count(inner, args.inner_boundary_points)
        print(f"Resampled inner boundary: {before:,} -> {len(inner):,} points", flush=True)

    if args.outer_boundary_points > 0:
        before = len(outer)
        outer = resample_closed_polyline_by_count(outer, args.outer_boundary_points)
        print(f"Resampled outer boundary: {before:,} -> {len(outer):,} points", flush=True)

    outer_ccw = ensure_ccw(outer)
    inner_cw = ensure_cw(inner)

    print(
        f"Outer boundary: {len(outer_ccw):,} points, length≈{polyline_length_closed(outer_ccw):.8g}, "
        f"area≈{polygon_area(outer_ccw):.8g}",
        flush=True,
    )
    print(
        f"Inner boundary: {len(inner_cw):,} points, length≈{polyline_length_closed(inner_cw):.8g}, "
        f"area≈{polygon_area(inner_cw):.8g}",
        flush=True,
    )

    print("Meshing annulus with Gmsh...", flush=True)
    mesh = build_gmsh_annulus_mesh(
        outer_ccw=outer_ccw,
        inner_cw=inner_cw,
        outer_mesh_size=args.outer_mesh_size,
        inner_mesh_size=args.inner_mesh_size,
        mesh_size_min=args.mesh_size_min,
        mesh_size_max=args.mesh_size_max,
        algorithm=args.gmsh_algorithm,
        optimize=not args.no_optimize,
        verbose=args.gmsh_verbose,
        save_msh=args.save_msh or None,
    )
    print(f"Gmsh mesh: {len(mesh.xy):,} vertices, {len(mesh.tris):,} triangles", flush=True)

    print(f"Evaluating G on mesh vertices using {args.g_eval} mode...", flush=True)
    if args.g_eval == "vectorized":
        Gv = compute_mandelbrot_potential_points_vectorized(
            mesh.xy,
            max_iter=args.eval_max_iter,
            escape_radius=args.eval_escape_radius,
            batch_size=args.eval_batch_size,
        )
    else:
        Gv = compute_mandelbrot_potential_points_scalar(
            mesh.xy,
            epsilon=args.eval_epsilon,
            max_iter=args.eval_max_iter,
            bailout=args.trace_bailout,
        )

    bad = ~np.isfinite(Gv)
    if np.any(bad):
        print(f"Warning: {int(bad.sum()):,} non-finite G values; setting to gmin.", flush=True)
        Gv[bad] = args.gmin

    Gv = clamp_boundary_g_values(
        mesh.xy,
        Gv,
        inner=inner,
        outer=outer,
        gmin=args.gmin,
        gmax=args.gmax,
        tol=args.boundary_clamp_tol,
    )

    # The mesh is supposed to live in gmin <= G <= gmax. Small leaks can happen
    # from finite iteration, polygonization, or vertices exactly on boundaries.
    Gv = np.clip(Gv, args.gmin, args.gmax)

    print(
        f"Mesh G after clipping: min={float(np.min(Gv)):.6g}, "
        f"median={float(np.median(Gv)):.6g}, max={float(np.max(Gv)):.6g}",
        flush=True,
    )

    overlay: list[tuple[float, np.ndarray]] = [(args.gmin, inner), (args.gmax, outer)]
    if args.overlay_contours > 0:
        levels = np.geomspace(args.gmin, args.gmax, int(args.overlay_contours) + 2)[1:-1]
        for level in levels:
            pts = trace_xy_contour(float(level), args)
            save_contour_json(float(level), pts, save_contours_dir)
            overlay.append((float(level), pts))

    overlay.sort(key=lambda item: item[0])

    fig = build_plotly_figure(
        xy=mesh.xy,
        tris=mesh.tris,
        Gv=Gv,
        contours=overlay,
        height_mode=args.height_mode,
        color_mode=args.color_mode,
        z_scale=args.z_scale,
        invert_z=args.invert_z,
        colorscale=args.colorscale,
        contour_color=args.contour_color,
        contour_width=args.contour_width,
        title=args.title,
        unlit=args.unlit,
        flat_shading=args.flat_shading,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
