import argparse
import base64
import colorsys
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import sympy as sp


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Symbolic center polynomials
# ============================================================

c_sym = sp.Symbol("c")


def proper_divisors(n):
    return [d for d in range(1, n) if n % d == 0]


@lru_cache(None)
def P(n):
    """
    P_n(c) = f_c^n(0), where f_c(z) = z^2 + c.
    """
    z = sp.Integer(0)
    for _ in range(n):
        z = sp.expand(z * z + c_sym)
    return sp.Poly(z, c_sym, domain=sp.ZZ)


@lru_cache(None)
def center_poly(n):
    """
    Exact-period-n center polynomial Psi_n(c).
    """
    poly = P(n)
    for d in proper_divisors(n):
        poly = poly.exquo(center_poly(d))
    return poly


def centers_for_period(n, precision=50):
    roots = sp.nroots(center_poly(n).as_expr(), n=precision, maxsteps=200)
    return [complex(r) for r in roots]


# ============================================================
# Iteration + derivatives
# ============================================================

def iterate_data(z, c, n):
    """
    For F_n(z,c) = f_c^n(z), compute:

      A = F_n
      B = dF_n/dz
      C = dF_n/dc
      D = dB/dz
      E = dB/dc

    Used for Newton's method on:
      F_n(z,c) - z = 0
      dF_n/dz - lambda = 0
    """
    A = z
    B = 1.0 + 0.0j
    C = 0.0 + 0.0j
    D = 0.0 + 0.0j
    E = 0.0 + 0.0j

    for _ in range(n):
        A0, B0, C0, D0, E0 = A, B, C, D, E

        A = A0 * A0 + c

        B = 2 * A0 * B0
        C = 2 * A0 * C0 + 1

        D = 2 * (B0 * B0 + A0 * D0)
        E = 2 * (C0 * B0 + A0 * E0)

    return A, B, C, D, E


def solve_multiplier_point(n, z, c, lam, max_iter=40, tol=1e-12):
    """
    Solve
        f_c^n(z) = z
        (f_c^n)'(z) = lam
    via Newton's method.
    """
    for _ in range(max_iter):
        A, B, C, D, E = iterate_data(z, c, n)

        G1 = A - z
        G2 = B - lam

        if abs(G1) + abs(G2) < tol:
            return z, c, True

        J = np.array([
            [B - 1, C],
            [D,     E],
        ], dtype=complex)

        rhs = np.array([G1, G2], dtype=complex)

        try:
            dz, dc = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            return z, c, False

        z -= dz
        c -= dc

    return z, c, False


def cycle_points(z0, c, n):
    points = []
    z = z0
    for _ in range(n):
        points.append(z)
        z = z * z + c
    return points


def solve_periodic_point_for_c(n, c, z_seed, max_iter=35, tol=1e-11):
    """
    Solve f_c^n(z) = z for a fixed c, using z_seed as the branch seed.

    This is used by the long-edge splitter. A midpoint in the c-plane is not
    allowed to use a simple 3D midpoint height; the periodic point must be
    recomputed on the relevant sheet/branch.
    """
    z = complex(z_seed)

    for _ in range(max_iter):
        A, B, _, _, _ = iterate_data(z, c, n)
        G = A - z
        dG = B - 1

        if abs(G) < tol:
            return z, True

        if abs(dG) <= 1e-15:
            return z, False

        z -= G / dG

    return z, False


# ============================================================
# Component tracing
# ============================================================

def make_radial_points(nr, rmax, alpha=0.0):
    """
    Create radial multiplier samples in [0, rmax].

    alpha = 0 gives uniform spacing:
        f(x) = x

    alpha > 0 clusters points near both r = 0 and r = rmax:
        f(x) = x - alpha/(2*pi) * sin(2*pi*x)

    For alpha in [0, 1], this remains monotonic.
    alpha = 1 makes f'(0) = f'(1) = 0.
    """
    if nr <= 1:
        return np.array([0.0])

    alpha = max(0.0, min(1.0, alpha))

    x = np.linspace(0.0, 1.0, nr)
    f = x - (alpha / (2 * np.pi)) * np.sin(2 * np.pi * x)

    # protect exact endpoints from floating point silliness
    f[0] = 0.0
    f[-1] = 1.0

    return rmax * f


def trace_component(n, center, nr, nt, rmax, radial_bias_alpha=0.0):
    """
    Trace one hyperbolic component in multiplier coordinates using a rectangular grid.
    """
    rs = make_radial_points(nr, rmax, radial_bias_alpha)
    thetas = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)

    Cgrid = np.empty((nr, nt), dtype=complex)
    Zgrid = np.empty((nr, nt), dtype=complex)
    OK = np.zeros((nr, nt), dtype=bool)

    for it, theta in enumerate(thetas):
        z_guess = 0.0 + 0.0j
        c_guess = center

        for ir, r in enumerate(rs):
            # At r = 0, lambda = 0 for every angle.
            # All angular samples collapse to the same center point.
            if ir == 0 and abs(r) < 1e-15:
                Cgrid[ir, it] = center
                Zgrid[ir, it] = 0.0 + 0.0j
                OK[ir, it] = True
                continue

            lam = r * np.exp(1j * theta)
            z_guess, c_guess, ok = solve_multiplier_point(
                n=n,
                z=z_guess,
                c=c_guess,
                lam=lam,
            )

            Cgrid[ir, it] = c_guess
            Zgrid[ir, it] = z_guess
            OK[ir, it] = ok

            if not ok:
                break

    return Cgrid, Zgrid, OK


# ============================================================
# Petal / ring mesh tracing
# ============================================================

def _round_to_multiple(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def angular_count_for_radius(r, rmax, nt_outer, min_nt_ring=6, beta=0.75, multiple=2):
    """
    Choose angular samples for a ring in a disk-like multiplier mesh.

    The center gets one point. Rings farther out get more angular points.
    beta < 1 keeps a little more angular detail near the center.
    """
    if r <= 1e-15 or rmax <= 0:
        return 1

    nt_outer = max(3, int(nt_outer))
    min_nt_ring = min(max(3, int(min_nt_ring)), nt_outer)

    frac = max(0.0, min(1.0, r / rmax))
    if frac >= 1.0 - 1e-12:
        return nt_outer

    n = nt_outer * (frac ** beta)
    n = _round_to_multiple(n, multiple)
    n = max(min_nt_ring, min(nt_outer, n))
    return int(n)


def nearest_ok_point_on_ring(ring, theta):
    """
    Pick a continuation seed from the previous ring.
    """
    if len(ring["theta"]) == 1:
        idx = 0
        if ring["OK"][idx]:
            return ring["Z"][idx], ring["C"][idx]
        return 0.0 + 0.0j, ring.get("center", 0.0 + 0.0j)

    dtheta = np.angle(np.exp(1j * (ring["theta"] - theta)))
    order = np.argsort(np.abs(dtheta))

    for idx in order:
        if ring["OK"][idx]:
            return ring["Z"][idx], ring["C"][idx]

    return 0.0 + 0.0j, ring.get("center", 0.0 + 0.0j)


def trace_component_petal(
    n,
    center,
    nr,
    nt,
    rmax,
    radial_bias_alpha=0.0,
    petal_min_nt=6,
    petal_beta=0.75,
    petal_multiple=2,
):
    """
    Trace one hyperbolic component as adaptive concentric rings.

    Ring 0 has one center point. Outer rings receive increasing angular resolution.
    Neighboring rings are stitched with triangles later.
    """
    rs = make_radial_points(nr, rmax, radial_bias_alpha)
    rings = []

    for ir, r in enumerate(rs):
        nt_ring = angular_count_for_radius(
            r=r,
            rmax=rmax,
            nt_outer=nt,
            min_nt_ring=petal_min_nt,
            beta=petal_beta,
            multiple=petal_multiple,
        )

        if ir == 0 or abs(r) < 1e-15:
            theta = np.array([0.0])
            C = np.array([center], dtype=complex)
            Z = np.array([0.0 + 0.0j], dtype=complex)
            OK = np.array([True], dtype=bool)
        else:
            theta = np.linspace(0.0, 2 * np.pi, nt_ring, endpoint=False)
            C = np.empty(nt_ring, dtype=complex)
            Z = np.empty(nt_ring, dtype=complex)
            OK = np.zeros(nt_ring, dtype=bool)

            prev_ring = rings[-1]

            # Solve all points on this ring. Each point starts from the closest
            # successful point on the previous ring, so Newton continuation is radial/local.
            for it, th in enumerate(theta):
                z_guess, c_guess = nearest_ok_point_on_ring(prev_ring, th)
                lam = r * np.exp(1j * th)

                z, c, ok = solve_multiplier_point(
                    n=n,
                    z=z_guess,
                    c=c_guess,
                    lam=lam,
                )

                C[it] = c
                Z[it] = z
                OK[it] = ok

        rings.append({
            "r": r,
            "theta": theta,
            "C": C,
            "Z": Z,
            "OK": OK,
            "center": center,
        })

    return rings


# ============================================================
# Adaptive sampling
# ============================================================

def estimate_component_radius(n, center, r_probe=0.92, n_steps=5):
    """
    Very rough size estimate by tracing a few multiplier directions
    out to lambda = r_probe * {1, i, -1, -i}.
    """
    directions = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    samples = []

    for theta in directions:
        z_guess = 0.0 + 0.0j
        c_guess = center
        success = True

        for r in np.linspace(0.0, r_probe, n_steps):
            lam = r * np.exp(1j * theta)
            z_guess, c_guess, ok = solve_multiplier_point(
                n=n,
                z=z_guess,
                c=c_guess,
                lam=lam,
            )
            if not ok:
                success = False
                break

        if success:
            samples.append(c_guess)

    if not samples:
        return 0.0

    return max(abs(c - center) for c in samples)


def sample_counts_from_radius(radius, max_radius, min_nr, max_nr, min_nt, max_nt, gamma=0.6):
    """
    Allocate more samples to larger components.
    gamma < 1 softens the scaling.
    """
    if max_radius <= 0:
        return min_nr, min_nt

    frac = max(0.0, min(1.0, radius / max_radius))
    frac = frac ** gamma

    nr = int(round(min_nr + frac * (max_nr - min_nr)))
    nt = int(round(min_nt + frac * (max_nt - min_nt)))

    nr = max(min_nr, min(max_nr, nr))
    nt = max(min_nt, min(max_nt, nt))

    return nr, nt


# ============================================================
# Colors
# ============================================================

def phase_to_rgb(phi):
    """
    Match the plot coloring exactly.

    np.angle(z) returns phi in [-pi, pi].
    This mapping shifts that interval to hue in [0, 1].
    """
    h = ((phi + np.pi) / (2 * np.pi)) % 1.0
    return colorsys.hsv_to_rgb(h, 0.85, 1.0)


def phase_to_hex(phi):
    r, g, b = phase_to_rgb(phi)
    return f"#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}"


def hsv_to_rgb_arrays(h, s=0.85, v=1.0):
    """
    Vectorized HSV-to-RGB matching colorsys.hsv_to_rgb behavior.
    """
    h = h % 1.0
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.zeros_like(h, dtype=float)
    g = np.zeros_like(h, dtype=float)
    b = np.zeros_like(h, dtype=float)

    mask = i == 0
    r[mask], g[mask], b[mask] = v, t[mask], p

    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v, p

    mask = i == 2
    r[mask], g[mask], b[mask] = p, v, t[mask]

    mask = i == 3
    r[mask], g[mask], b[mask] = p, q[mask], v

    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p, v

    mask = i == 5
    r[mask], g[mask], b[mask] = v, p, q[mask]

    return r, g, b


# ============================================================
# Plot helpers
# ============================================================

def build_scene_axis(title, scene_mode, show_spikes, background_rgba):
    if scene_mode == "pure":
        return dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title="",
            showspikes=False,
        )

    if scene_mode == "axes":
        return dict(
            visible=True,
            showbackground=False,
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            title=title,
            showspikes=show_spikes,
        )

    # boxed
    return dict(
        visible=True,
        showbackground=True,
        backgroundcolor=background_rgba,
        showgrid=True,
        gridcolor="rgba(120,130,150,0.35)",
        showline=True,
        linecolor="rgba(60,70,90,0.55)",
        linewidth=2,
        showticklabels=True,
        zeroline=False,
        title=title,
        showspikes=show_spikes,
    )


def load_font(size, bold=False):
    """
    Try to load a clean system font; fall back to Pillow's default.
    """
    from PIL import ImageFont

    candidates = []
    windows_directory = os.environ.get("WINDIR")
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
        if windows_directory:
            candidates.append(Path(windows_directory) / "Fonts" / "arialbd.ttf")
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])
        if windows_directory:
            candidates.append(Path(windows_directory) / "Fonts" / "arial.ttf")

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass

    return ImageFont.load_default()


def make_phase_wheel_png_data_uri(
    size_px=260,
    ring_radius=78,
    ring_width=24,
    label_radius=108,
    center_offset_x=0,
    center_offset_y=0,
):
    """
    Create a compact PNG data URI showing the exact phase-to-color mapping.

    This keeps the fine-tuned label nudges from the SVG version but renders the
    angular color wheel as one small raster image instead of hundreds of SVG segments.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "The phase wheel PNG renderer needs Pillow. Install it with: pip install pillow"
        ) from exc

    cx = size_px / 2 + center_offset_x
    cy = size_px / 2 + center_offset_y

    img = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background card
    draw.rounded_rectangle(
        (2, 2, size_px - 2, size_px - 2),
        radius=20,
        fill=(255, 255, 255, 184),
        outline=(0, 0, 0, 26),
        width=1,
    )

    # Angular-gradient ring
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    dx = xx - cx
    dy = cy - yy  # positive-up convention

    rho = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx)

    inner = ring_radius - ring_width / 2
    outer = ring_radius + ring_width / 2
    mask = (rho >= inner) & (rho <= outer)

    h = ((phi + np.pi) / (2 * np.pi)) % 1.0
    r, g, b = hsv_to_rgb_arrays(h, s=0.85, v=1.0)

    ring = np.zeros((size_px, size_px, 4), dtype=np.uint8)
    ring[..., 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
    ring[..., 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    ring[..., 2] = np.clip(b * 255, 0, 255).astype(np.uint8)
    ring[..., 3] = np.where(mask, 255, 0).astype(np.uint8)

    ring_img = Image.fromarray(ring, mode="RGBA")
    img.alpha_composite(ring_img)

    # Subtle inner/outer outline
    draw.ellipse(
        (cx - inner, cy - inner, cx + inner, cy + inner),
        outline=(0, 0, 0, 50),
        width=1,
    )
    draw.ellipse(
        (cx - outer, cy - outer, cx + outer, cy + outer),
        outline=(0, 0, 0, 50),
        width=1,
    )

    # Center label
    font_center = load_font(24, bold=True)
    font_sub = load_font(18, bold=False)
    font_angle = load_font(16, bold=True)

    draw.text(
        (cx, cy - 6),
        "arg(z)",
        anchor="mm",
        font=font_center,
        fill=(17, 17, 17, 255),
    )
    draw.text(
        (cx, cy + 17),
        "phase",
        anchor="mm",
        font=font_sub,
        fill=(85, 85, 85, 255),
    )

    # Fine-tuned label positions; intentionally preserved.
    labels = [
        (0, "0°", 0, 0),
        (45, "45°", 0, 0),
        (90, "90°", 0, -0.02),
        (135, "135°", 0, 0),
        (180, "±180°", 0.1, 0),
        (-135, "225°", 0.04, 0),
        (-90, "270°", 0, -0.015),
        (-45, "315°", 0.02, 0),
    ]

    for deg, text, fac_x, fac_y in labels:
        phi_label = np.deg2rad(deg)

        x = cx + (1 + fac_x) * label_radius * np.cos(phi_label)
        y = cy - (1 + fac_y) * label_radius * np.sin(phi_label)

        draw.text(
            (x, y),
            text,
            anchor="mm",
            font=font_angle,
            fill=(34, 34, 34, 255),
        )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def add_phase_color_wheel(fig, args):
    """
    Add a 2D overlay color wheel in paper coordinates.
    """

    source = make_phase_wheel_png_data_uri(
        size_px=args.phase_wheel_px,
        ring_radius=args.phase_wheel_ring_radius,
        ring_width=args.phase_wheel_ring_width,
        label_radius=args.phase_wheel_label_radius,
        center_offset_x=args.phase_wheel_center_offset_x,
        center_offset_y=args.phase_wheel_center_offset_y,
    )

    fig.add_layout_image(
        dict(
            source=source,
            xref="paper",
            yref="paper",
            x=args.phase_wheel_x,
            y=args.phase_wheel_y,
            sizex=args.phase_wheel_size,
            sizey=args.phase_wheel_size,
            xanchor="right",
            yanchor="top",
            layer="above",
        )
    )


def build_figure(title, z_title, args):
    fig = go.Figure()

    scene = dict(
        xaxis=build_scene_axis("Re(c)", args.scene_mode, args.show_spikes, args.box_bg),
        yaxis=build_scene_axis("Im(c)", args.scene_mode, args.show_spikes, args.box_bg),
        zaxis=build_scene_axis(z_title, args.scene_mode, args.show_spikes, args.box_bg),
        aspectmode="data",
    )

    fig.update_layout(
        title=dict(
            text=title if not args.hide_title else "",
            x=0.02,
            y=0.955,
            xanchor="left",
            yanchor="top",
            font=dict(size=17),
        ),
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=70 if not args.hide_title else 0),
        showlegend=False,
        hovermode="closest" if args.enable_hover else False,
        dragmode="turntable",
    )

    return fig


def add_mesh(fig, x, y, z, i, j, k, vertex_colors, name, opacity, enable_hover=False):
    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        vertexcolor=vertex_colors,
        opacity=opacity,
        flatshading=False,
        name=name,
        showscale=False,
        hoverinfo="x+y+z" if enable_hover else "skip",
        lighting=dict(
            ambient=0.55,
            diffuse=0.65,
            specular=0.10,
            roughness=0.85,
            fresnel=0.05,
        ),
    ))


class MeshAccumulator:
    """
    Collect many disconnected sheet meshes into one Plotly Mesh3d trace.

    This is the important performance trick: Plotly is much happier with one
    large Mesh3d trace than with hundreds or thousands of tiny traces.
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.i = []
        self.j = []
        self.k = []
        self.vertex_colors = []

    @property
    def vertex_count(self):
        return len(self.x)

    @property
    def triangle_count(self):
        return len(self.i)

    def add(self, x, y, z, i, j, k, vertex_colors):
        if not i:
            return

        offset = len(self.x)

        self.x.extend(float(v) for v in x)
        self.y.extend(float(v) for v in y)
        self.z.extend(float(v) for v in z)
        self.vertex_colors.extend(vertex_colors)

        self.i.extend(offset + int(v) for v in i)
        self.j.extend(offset + int(v) for v in j)
        self.k.extend(offset + int(v) for v in k)

    def add_to_figure(self, fig, name, opacity, enable_hover=False):
        if self.triangle_count == 0:
            return

        add_mesh(
            fig=fig,
            x=self.x,
            y=self.y,
            z=self.z,
            i=self.i,
            j=self.j,
            k=self.k,
            vertex_colors=self.vertex_colors,
            name=name,
            opacity=opacity,
            enable_hover=enable_hover,
        )


class LineAccumulator:
    """
    Collect many disconnected outline polylines into one Scatter3d trace.
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

    @property
    def segment_count(self):
        # Each segment is stored as x0, x1, None.
        return self.x.count(None)

    def add_segment(self, x0, y0, z0, x1, y1, z1):
        self.x.extend([float(x0), float(x1), None])
        self.y.extend([float(y0), float(y1), None])
        self.z.extend([float(z0), float(z1), None])

    def add_to_figure(self, fig, color, width, opacity=1.0, name="component-outlines"):
        if not self.x:
            return

        fig.add_trace(go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode="lines",
            line=dict(
                color=color,
                width=width,
            ),
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
            name=name,
        ))


def add_cyclic_outline_segments(outline_lines, xs, ys, zs, valid):
    """
    Add outline segments along a cyclic boundary, skipping invalid gaps.
    """
    if outline_lines is None:
        return

    n = len(xs)
    if n < 2:
        return

    for idx in range(n):
        idx2 = (idx + 1) % n
        if valid[idx] and valid[idx2]:
            outline_lines.add_segment(
                xs[idx], ys[idx], zs[idx],
                xs[idx2], ys[idx2], zs[idx2],
            )


def add_wireframe_segments(wireframe_lines, xs, ys, zs, I, J, K, z_lift):
    """
    Add all unique triangle edges from one sheet as debug wireframe segments.

    Edges are deduplicated within the sheet so shared triangle edges are drawn once.
    This is intended as a mesh-debug overlay, not as a production visual layer.
    """
    if wireframe_lines is None or not I:
        return

    seen = set()

    for a, b, c in zip(I, J, K):
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue

            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)

            wireframe_lines.add_segment(
                xs[u], ys[u], zs[u] + z_lift,
                xs[v], ys[v], zs[v] + z_lift,
            )


# ============================================================
# Mesh postprocessing: local edge flips
# ============================================================

def _triangle_min_angle(points, tri):
    """
    Minimum interior angle of a triangle in either 2D or 3D coordinate space.
    Returns 0 for degenerate triangles.
    """
    p0 = points[tri[0]]
    p1 = points[tri[1]]
    p2 = points[tri[2]]

    def angle_at(pa, pb, pc):
        u = pb - pa
        v = pc - pa
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-15 or nv <= 1e-15:
            return 0.0
        cosang = float(np.dot(u, v) / (nu * nv))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.arccos(cosang))

    a0 = angle_at(p0, p1, p2)
    a1 = angle_at(p1, p2, p0)
    a2 = angle_at(p2, p0, p1)

    if not np.isfinite(a0 + a1 + a2):
        return 0.0

    return min(a0, a1, a2)


def _triangle_normal(points3d, tri):
    p0 = points3d[tri[0]]
    p1 = points3d[tri[1]]
    p2 = points3d[tri[2]]
    return np.cross(p1 - p0, p2 - p0)


def _orient_triangle_like(tri, reference_normal, points3d):
    """
    Keep candidate triangles oriented roughly like the two triangles they replace.
    This avoids introducing local normal flips purely from postprocessing.
    """
    tri = list(tri)
    ref_norm = float(np.linalg.norm(reference_normal))
    if ref_norm <= 1e-15:
        return tri

    n = _triangle_normal(points3d, tri)
    if float(np.dot(n, reference_normal)) < 0.0:
        tri[1], tri[2] = tri[2], tri[1]
    return tri


def _orient2d(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross_2d(a, b, c, d, eps=1e-14):
    """
    True when the two open line segments cross in the 2D base plane.
    Used as a conservative guard before flipping an edge.
    """
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def improve_triangles_by_edge_flips(
    x,
    y,
    z,
    I,
    J,
    K,
    mode=False,
    passes=2,
    epsilon=1e-7,
    ring_ids=None,
    max_diagonal_ratio=1.25,
):
    """
    Improve a triangle mesh by locally flipping shared edges when the flip improves
    the minimum angle of the two affected triangles.

    mode=False: no-op
    mode="2D": judge triangle quality in the Re(c), Im(c) plane
    mode="3D": judge triangle quality in the actual rendered 3D sheet coordinates

    The topological guard still checks that the old and new diagonals cross in the
    Re(c), Im(c) plane before flipping. This avoids flipping non-convex projected
    quads into invalid local topology.

    If ring_ids is provided, flips are restricted to true annular diagonals:
    the four vertices must occupy exactly two adjacent non-center rings, with
    two vertices on each ring. This preserves ring boundaries and the center fan.

    max_diagonal_ratio prevents flips that improve the minimum angle only by
    creating a much longer diagonal, which can produce large fan/spike artifacts
    near cusps or compressed component tips.
    """
    if mode is False or mode is None or not I:
        return I, J, K

    if mode not in ("2D", "3D"):
        raise ValueError(f"Unknown postprocess mesh mode: {mode!r}")

    passes = max(0, int(passes))
    if passes == 0:
        return I, J, K

    points2d = np.column_stack([
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    ])
    points3d = np.column_stack([
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    ])
    quality_points = points2d if mode == "2D" else points3d

    if ring_ids is not None:
        ring_ids = np.asarray(ring_ids, dtype=int)
        if len(ring_ids) != len(points2d):
            raise ValueError(
                f"ring_ids length {len(ring_ids)} does not match vertex count {len(points2d)}"
            )

    max_diagonal_ratio = float(max_diagonal_ratio)

    triangles = [
        [int(a), int(b), int(c)]
        for a, b, c in zip(I, J, K)
        if int(a) != int(b) and int(b) != int(c) and int(a) != int(c)
    ]

    total_flips = 0

    for _ in range(passes):
        edge_to_tris = {}
        for ti, tri in enumerate(triangles):
            for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (u, v) if u < v else (v, u)
                edge_to_tris.setdefault(key, []).append(ti)

        changed = 0
        touched = set()

        for (u, v), adj in edge_to_tris.items():
            if len(adj) != 2:
                continue

            t0, t1 = adj
            if t0 in touched or t1 in touched:
                continue

            tri0 = triangles[t0]
            tri1 = triangles[t1]

            # The edge map was built at the start of the pass. Skip stale pairs.
            if not (u in tri0 and v in tri0 and u in tri1 and v in tri1):
                continue

            opp0 = [p for p in tri0 if p != u and p != v]
            opp1 = [p for p in tri1 if p != u and p != v]
            if len(opp0) != 1 or len(opp1) != 1:
                continue

            a = opp0[0]
            b = opp1[0]
            if a == b or a in (u, v) or b in (u, v):
                continue

            if ring_ids is not None:
                ru, rv, ra, rb = (int(ring_ids[p]) for p in (u, v, a, b))
                rings_here = sorted({ru, rv, ra, rb})

                # Preserve the center fan and ring boundaries. Only flip true
                # annular diagonals between two adjacent non-center rings.
                if len(rings_here) != 2:
                    continue
                if rings_here[0] == 0:
                    continue
                if rings_here[1] - rings_here[0] != 1:
                    continue
                if [ru, rv, ra, rb].count(rings_here[0]) != 2:
                    continue
                if [ru, rv, ra, rb].count(rings_here[1]) != 2:
                    continue

                # The old and new diagonals should both connect the two rings.
                # If either diagonal lies along a ring, the flip would erase or
                # invent a ring edge and can create fan/spike artifacts near tips.
                if ru == rv:
                    continue
                if ra == rb:
                    continue

            # Conservative 2D topology guard. In a valid convex projected quad,
            # the current diagonal (u,v) and candidate diagonal (a,b) cross.
            if not _segments_cross_2d(points2d[u], points2d[v], points2d[a], points2d[b]):
                continue

            if max_diagonal_ratio > 0:
                old_diag_len = float(np.linalg.norm(quality_points[u] - quality_points[v]))
                new_diag_len = float(np.linalg.norm(quality_points[a] - quality_points[b]))
                if old_diag_len <= 1e-15:
                    continue
                if new_diag_len > max_diagonal_ratio * old_diag_len:
                    continue

            old_quality = min(
                _triangle_min_angle(quality_points, tri0),
                _triangle_min_angle(quality_points, tri1),
            )

            # Candidate flip: replace shared edge (u,v) with (a,b).
            ref_normal = _triangle_normal(points3d, tri0) + _triangle_normal(points3d, tri1)
            cand0 = _orient_triangle_like([a, b, u], ref_normal, points3d)
            cand1 = _orient_triangle_like([b, a, v], ref_normal, points3d)

            new_quality = min(
                _triangle_min_angle(quality_points, cand0),
                _triangle_min_angle(quality_points, cand1),
            )

            if new_quality > old_quality + epsilon:
                triangles[t0] = cand0
                triangles[t1] = cand1
                touched.add(t0)
                touched.add(t1)
                changed += 1

        total_flips += changed
        if changed == 0:
            break

    if total_flips:
        print(f"  postprocess {mode}: flipped {total_flips:,} edges")

    return (
        [tri[0] for tri in triangles],
        [tri[1] for tri in triangles],
        [tri[2] for tri in triangles],
    )


# ============================================================
# Mesh postprocessing: long-edge subdivision
# ============================================================

def _height_and_color_from_sheet_value(zj, height_mode, height_scale):
    if height_mode == "magphase":
        return height_scale * abs(zj), phase_to_hex(np.angle(zj))
    if height_mode == "real":
        return zj.real, "#3b82f6"
    if height_mode == "imag":
        return zj.imag, "#f59e0b"
    raise ValueError(f"Unknown height_mode: {height_mode}")


def _trimmed_middle_mean(values, trim_fraction=0.10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0

    values.sort()
    trim_fraction = max(0.0, min(0.45, float(trim_fraction)))
    n = len(values)
    lo = int(round(trim_fraction * n))
    hi = int(round((1.0 - trim_fraction) * n))

    if hi <= lo:
        lo = 0
        hi = n

    return float(np.mean(values[lo:hi]))


def split_long_edges_by_sheet(
    x,
    y,
    z,
    colors,
    I,
    J,
    K,
    sheet_values,
    n,
    height_mode,
    height_scale,
    enabled=False,
    factor=1.9,
    trim_fraction=0.10,
    passes=1,
    max_new_vertices=200000,
):
    """
    Split unusually long 3D edges in one already-built sheet mesh.

    The threshold is computed per sheet:
      1. collect unique existing triangle edges,
      2. sort by 3D length,
      3. discard the shortest 10% and longest 10% by default,
      4. use factor * mean(middle 80%) as the long-edge threshold.

    For every long edge, a midpoint is inserted in the c-plane. Its rendered
    height is not interpolated. Instead, f_c^n(z)=z is solved again at c_mid,
    using the neighboring sheet values as the Newton seed. This keeps the new
    vertex on the actual periodic branch as much as possible.
    """
    if not enabled or not I:
        return x, y, z, colors, I, J, K, sheet_values

    factor = float(factor)
    if factor <= 0:
        return x, y, z, colors, I, J, K, sheet_values

    passes = max(0, int(passes))
    max_new_vertices = max(0, int(max_new_vertices))

    x = list(x)
    y = list(y)
    z = list(z)
    colors = list(colors)
    I = [int(v) for v in I]
    J = [int(v) for v in J]
    K = [int(v) for v in K]
    sheet_values = list(sheet_values)

    total_added = 0

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def edge_length_3d(u, v):
        dx = x[u] - x[v]
        dy = y[u] - y[v]
        dz = z[u] - z[v]
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    def add_midpoint(u, v, midpoint_cache):
        nonlocal total_added
        key = edge_key(u, v)
        if key in midpoint_cache:
            return midpoint_cache[key]

        if total_added >= max_new_vertices:
            return None

        cu = complex(x[u], y[u])
        cv = complex(x[v], y[v])
        c_mid = 0.5 * (cu + cv)

        zu = sheet_values[u]
        zv = sheet_values[v]
        if zu is None or zv is None:
            return None

        seed = 0.5 * (zu + zv)
        zj_mid, ok = solve_periodic_point_for_c(n=n, c=c_mid, z_seed=seed)

        # If Newton refuses to converge, do not split this edge. Falling back to
        # the literal 3D midpoint would hide the problem but would not improve the
        # actual surface geometry.
        if not ok or not np.isfinite(zj_mid.real + zj_mid.imag):
            return None

        h_mid, color_mid = _height_and_color_from_sheet_value(
            zj=zj_mid,
            height_mode=height_mode,
            height_scale=height_scale,
        )

        idx = len(x)
        x.append(float(c_mid.real))
        y.append(float(c_mid.imag))
        z.append(float(h_mid))
        colors.append(color_mid)
        sheet_values.append(zj_mid)

        midpoint_cache[key] = idx
        total_added += 1
        return idx

    def orient_like_original(new_tri, original_tri):
        points3d = np.column_stack([
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        ])
        ref = _triangle_normal(points3d, original_tri)
        return _orient_triangle_like(new_tri, ref, points3d)

    for _ in range(passes):
        if not I:
            break

        edges = set()
        for a, b, c in zip(I, J, K):
            edges.add(edge_key(a, b))
            edges.add(edge_key(b, c))
            edges.add(edge_key(c, a))

        lengths = [edge_length_3d(u, v) for u, v in edges]
        reference = _trimmed_middle_mean(lengths, trim_fraction=trim_fraction)
        if reference <= 1e-15:
            break

        threshold = factor * reference
        long_edges = {key for key in edges if edge_length_3d(*key) > threshold}
        if not long_edges:
            break

        midpoint_cache = {}
        new_tris = []
        changed = 0

        for a, b, c in zip(I, J, K):
            e_ab = edge_key(a, b) in long_edges
            e_bc = edge_key(b, c) in long_edges
            e_ca = edge_key(c, a) in long_edges
            count = int(e_ab) + int(e_bc) + int(e_ca)
            original = [a, b, c]

            if count == 0:
                new_tris.append(original)
                continue

            if count == 1:
                if e_ab:
                    m_ab = add_midpoint(a, b, midpoint_cache)
                    if m_ab is None:
                        new_tris.append(original)
                    else:
                        candidates = [[a, m_ab, c], [m_ab, b, c]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1
                elif e_bc:
                    m_bc = add_midpoint(b, c, midpoint_cache)
                    if m_bc is None:
                        new_tris.append(original)
                    else:
                        candidates = [[a, b, m_bc], [a, m_bc, c]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1
                else:  # e_ca
                    m_ca = add_midpoint(c, a, midpoint_cache)
                    if m_ca is None:
                        new_tris.append(original)
                    else:
                        candidates = [[a, b, m_ca], [b, c, m_ca]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1

            elif count == 2:
                if e_ab and e_bc:
                    m_ab = add_midpoint(a, b, midpoint_cache)
                    m_bc = add_midpoint(b, c, midpoint_cache)
                    if m_ab is None or m_bc is None:
                        new_tris.append(original)
                    else:
                        candidates = [[m_ab, b, m_bc], [a, m_ab, m_bc], [a, m_bc, c]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1
                elif e_bc and e_ca:
                    m_bc = add_midpoint(b, c, midpoint_cache)
                    m_ca = add_midpoint(c, a, midpoint_cache)
                    if m_bc is None or m_ca is None:
                        new_tris.append(original)
                    else:
                        candidates = [[m_bc, c, m_ca], [a, b, m_bc], [a, m_bc, m_ca]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1
                else:  # e_ca and e_ab
                    m_ca = add_midpoint(c, a, midpoint_cache)
                    m_ab = add_midpoint(a, b, midpoint_cache)
                    if m_ca is None or m_ab is None:
                        new_tris.append(original)
                    else:
                        candidates = [[m_ca, a, m_ab], [m_ab, b, c], [m_ab, c, m_ca]]
                        new_tris.extend(orient_like_original(t, original) for t in candidates)
                        changed += 1

            else:  # count == 3
                m_ab = add_midpoint(a, b, midpoint_cache)
                m_bc = add_midpoint(b, c, midpoint_cache)
                m_ca = add_midpoint(c, a, midpoint_cache)
                if m_ab is None or m_bc is None or m_ca is None:
                    new_tris.append(original)
                else:
                    candidates = [
                        [a, m_ab, m_ca],
                        [m_ab, b, m_bc],
                        [m_ca, m_bc, c],
                        [m_ab, m_bc, m_ca],
                    ]
                    new_tris.extend(orient_like_original(t, original) for t in candidates)
                    changed += 1

        I = [tri[0] for tri in new_tris]
        J = [tri[1] for tri in new_tris]
        K = [tri[2] for tri in new_tris]

        if changed == 0:
            break

    if total_added:
        print(f"  long-edge split: inserted {total_added:,} vertices")

    return x, y, z, colors, I, J, K, sheet_values


# ============================================================
# Rectangular mesh plotting
# ============================================================

def make_faces_from_valid_mask(valid, nt):
    """
    Build triangle connectivity only for cells whose 4 corners are valid.
    """
    nr = valid.shape[0]

    I, J, K = [], [], []

    def idx(ir, it):
        return ir * nt + (it % nt)

    for ir in range(nr - 1):
        for it in range(nt):
            it2 = (it + 1) % nt

            a = valid[ir, it]
            b = valid[ir + 1, it]
            c = valid[ir + 1, it2]
            d = valid[ir, it2]

            if a and b and c and d:
                v0 = idx(ir, it)
                v1 = idx(ir + 1, it)
                v2 = idx(ir + 1, it2)
                v3 = idx(ir, it2)

                I.extend([v0, v0])
                J.extend([v1, v2])
                K.extend([v2, v3])

    return I, J, K


def add_magphase_sheet(fig, Cgrid, Zgrid, OK, n, sheet_index, opacity, name, height_scale):
    nr, nt = Cgrid.shape

    x, y, z_height, colors = [], [], [], []
    valid = np.zeros((nr, nt), dtype=bool)

    for ir in range(nr):
        for it in range(nt):
            c = Cgrid[ir, it]
            z0 = Zgrid[ir, it]

            x.append(c.real)
            y.append(c.imag)

            if OK[ir, it]:
                zs = cycle_points(z0, c, n)
                zj = zs[sheet_index]
                z_height.append(height_scale * abs(zj))
                colors.append(phase_to_hex(np.angle(zj)))
                valid[ir, it] = True
            else:
                z_height.append(0.0)
                colors.append("#000000")

    I, J, K = make_faces_from_valid_mask(valid, nt)
    if I:
        add_mesh(fig, x, y, z_height, I, J, K, colors, name, opacity)


def add_reim_sheet(fig, Cgrid, Zgrid, OK, n, sheet_index, component_name, opacity, mode):
    """
    mode = "real" or "imag"
    """
    nr, nt = Cgrid.shape

    x, y, z_height, colors = [], [], [], []
    valid = np.zeros((nr, nt), dtype=bool)

    if mode == "real":
        base_color = "#3b82f6"  # blue
    else:
        base_color = "#f59e0b"  # orange

    for ir in range(nr):
        for it in range(nt):
            c = Cgrid[ir, it]
            z0 = Zgrid[ir, it]

            x.append(c.real)
            y.append(c.imag)

            if OK[ir, it]:
                zs = cycle_points(z0, c, n)
                zj = zs[sheet_index]

                if mode == "real":
                    z_height.append(zj.real)
                else:
                    z_height.append(zj.imag)

                colors.append(base_color)
                valid[ir, it] = True
            else:
                z_height.append(0.0)
                colors.append(base_color)

    I, J, K = make_faces_from_valid_mask(valid, nt)
    if I:
        add_mesh(
            fig,
            x, y, z_height,
            I, J, K,
            colors,
            f"{component_name}-{mode}",
            opacity
        )


# ============================================================
# Petal mesh plotting
# ============================================================

def triangulate_petal_rings(flat_rings):
    """
    Build triangles between concentric rings with variable angular resolution.
    """
    I, J, K = [], [], []

    def add_tri(a, b, c, valid):
        if valid[a] and valid[b] and valid[c] and a != b and b != c and a != c:
            I.append(a)
            J.append(b)
            K.append(c)

    for ring_idx in range(len(flat_rings) - 1):
        inner = flat_rings[ring_idx]
        outer = flat_rings[ring_idx + 1]

        inner_idx = inner["idx"]
        outer_idx = outer["idx"]
        valid = inner["global_valid"]

        m = len(inner_idx)
        n = len(outer_idx)

        if m == 0 or n == 0:
            continue

        # Center fan.
        if m == 1:
            center_idx = inner_idx[0]
            for j in range(n):
                add_tri(center_idx, outer_idx[j], outer_idx[(j + 1) % n], valid)
            continue

        i = 0
        j = 0

        while i < m or j < n:
            next_inner = (i + 1) / m if i < m else float("inf")
            next_outer = (j + 1) / n if j < n else float("inf")

            ii = i % m
            jj = j % n

            if abs(next_inner - next_outer) < 1e-14:
                # One clean quad sector split into two triangles.
                add_tri(inner_idx[ii], outer_idx[jj], outer_idx[(j + 1) % n], valid)
                add_tri(inner_idx[ii], outer_idx[(j + 1) % n], inner_idx[(i + 1) % m], valid)
                i += 1
                j += 1
            elif next_inner < next_outer:
                # Advance inner boundary.
                add_tri(inner_idx[ii], outer_idx[jj], inner_idx[(i + 1) % m], valid)
                i += 1
            else:
                # Advance outer boundary.
                add_tri(inner_idx[ii], outer_idx[jj], outer_idx[(j + 1) % n], valid)
                j += 1

            if i >= m and j >= n:
                break

    return I, J, K


def flatten_petal_sheet(
    rings,
    n,
    sheet_index,
    height_mode,
    height_scale=1.0,
    postprocess_mesh=False,
    postprocess_passes=2,
    postprocess_epsilon=1e-7,
    postprocess_max_diagonal_ratio=1.25,
    split_long_edges=False,
    long_edge_factor=1.9,
    long_edge_trim=0.10,
    long_edge_passes=1,
    long_edge_max_new_vertices=200000,
):
    """
    Flatten petal rings into Mesh3d vertex arrays for one sheet.

    height_mode:
        "magphase" -> z = height_scale * |z_j|, color = arg(z_j)
        "real"     -> z = Re(z_j), blue
        "imag"     -> z = Im(z_j), orange
    """
    x, y, z_height, colors = [], [], [], []
    sheet_values = []
    global_valid = []
    flat_rings = []
    ring_ids = []

    for ring_index, ring in enumerate(rings):
        idxs = []

        for c, z0, ok in zip(ring["C"], ring["Z"], ring["OK"]):
            idxs.append(len(x))
            x.append(c.real)
            y.append(c.imag)
            ring_ids.append(ring_index)

            if ok:
                zs = cycle_points(z0, c, n)
                zj = zs[sheet_index]

                h_val, color_val = _height_and_color_from_sheet_value(
                    zj=zj,
                    height_mode=height_mode,
                    height_scale=height_scale,
                )
                z_height.append(h_val)
                colors.append(color_val)
                sheet_values.append(zj)

                global_valid.append(True)
            else:
                z_height.append(0.0)
                colors.append("#000000" if height_mode == "magphase" else "#999999")
                sheet_values.append(None)
                global_valid.append(False)

        flat_rings.append({
            "idx": np.array(idxs, dtype=int),
            "global_valid": global_valid,
        })

    I, J, K = triangulate_petal_rings(flat_rings)
    I, J, K = improve_triangles_by_edge_flips(
        x=x,
        y=y,
        z=z_height,
        I=I,
        J=J,
        K=K,
        mode=postprocess_mesh,
        passes=postprocess_passes,
        epsilon=postprocess_epsilon,
        ring_ids=ring_ids,
        max_diagonal_ratio=postprocess_max_diagonal_ratio,
    )

    x, y, z_height, colors, I, J, K, sheet_values = split_long_edges_by_sheet(
        x=x,
        y=y,
        z=z_height,
        colors=colors,
        I=I,
        J=J,
        K=K,
        sheet_values=sheet_values,
        n=n,
        height_mode=height_mode,
        height_scale=height_scale,
        enabled=split_long_edges,
        factor=long_edge_factor,
        trim_fraction=long_edge_trim,
        passes=long_edge_passes,
        max_new_vertices=long_edge_max_new_vertices,
    )

    return x, y, z_height, colors, I, J, K


def add_magphase_sheet_petal(fig, rings, n, sheet_index, opacity, name, height_scale):
    x, y, z_height, colors, I, J, K = flatten_petal_sheet(
        rings=rings,
        n=n,
        sheet_index=sheet_index,
        height_mode="magphase",
        height_scale=height_scale,
    )
    if I:
        add_mesh(fig, x, y, z_height, I, J, K, colors, name, opacity)


def add_reim_sheet_petal(fig, rings, n, sheet_index, component_name, opacity, mode):
    x, y, z_height, colors, I, J, K = flatten_petal_sheet(
        rings=rings,
        n=n,
        sheet_index=sheet_index,
        height_mode=mode,
        height_scale=1.0,
    )
    if I:
        add_mesh(fig, x, y, z_height, I, J, K, colors, f"{component_name}-{mode}", opacity)


# ============================================================
# Center guide helpers
# ============================================================

def add_center_guides(fig, center_data, args):
    """
    center_data is a list of dicts like:
        {
            "x": center.real,
            "y": center.imag,
            "marker_heights": [h0, h1, ..., h_{n-1}],
            "line_top": max(h0, h1, ..., h_{n-1})
        }
    """

    if not center_data:
        return

    # --------------------------------
    # Solid vertical lines
    # --------------------------------
    line_x = []
    line_y = []
    line_z = []

    for item in center_data:
        ztop = item["line_top"]

        # If the local max is exactly 0, the line would be invisible anyway.
        # We still allow it; Plotly just draws a zero-length segment.
        line_x.extend([item["x"], item["x"], None])
        line_y.extend([item["y"], item["y"], None])
        line_z.extend([0.0, ztop, None])

    fig.add_trace(go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(
            color=args.center_line_color,
            width=args.center_line_width,
        ),
        hoverinfo="skip",
        showlegend=False,
        name="center-lines",
    ))

    # --------------------------------
    # Circle markers at center heights
    # --------------------------------
    mx, my, mz = [], [], []

    for item in center_data:
        for h in item["marker_heights"]:
            mx.append(item["x"])
            my.append(item["y"])
            mz.append(h)

    fig.add_trace(go.Scatter3d(
        x=mx,
        y=my,
        z=mz,
        mode="markers",
        marker=dict(
            size=args.center_marker_size,
            color=args.center_marker_fill,
            line=dict(
                color=args.center_marker_line_color,
                width=args.center_marker_line_width,
            ),
            symbol="circle",
        ),
        hoverinfo="skip",
        showlegend=False,
        name="center-markers",
    ))


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate interactive Mandelbrot component-atlas plots."
    )

    # General
    p.add_argument("--max-period", type=int, default=5)
    p.add_argument("--rmax", type=float, default=0.9999)
    p.add_argument("--precision", type=int, default=50)

    # Scene / interaction
    p.add_argument("--scene-mode", choices=["boxed", "axes", "pure"], default="boxed")
    p.add_argument("--show-spikes", action="store_true", help="Enable hover projection lines.")
    p.add_argument("--enable-hover", action="store_true", help="Enable hover mode (can slow rendering).")
    p.add_argument("--hide-title", action="store_true")
    p.add_argument("--box-bg", default="rgba(230,235,245,0.85)")

    # Center guides
    p.add_argument("--show-center-lines", action="store_true")
    p.add_argument("--center-line-color", default="rgba(0,0,0,0.25)")
    p.add_argument("--center-line-width", type=float, default=1.0)
    p.add_argument("--center-marker-size", type=float, default=2.0)
    p.add_argument("--center-marker-fill", default="rgba(255,255,255,0.25)")
    p.add_argument("--center-marker-line-color", default="black")
    p.add_argument("--center-marker-line-width", type=float, default=1.0)

    # Component outlines
    p.add_argument("--draw-outlines", action="store_true",
                   help="Draw outer boundary outlines for each component sheet.")
    p.add_argument("--outline-color", default="rgba(0,0,0,1.0)")
    p.add_argument("--outline-width", type=float, default=2.0)
    p.add_argument("--outline-opacity", type=float, default=1.0)
    p.add_argument("--outline-z-lift", type=float, default=0.0002,
                   help="Small upward offset to reduce z-fighting with the surface.")

    # Wireframe debug overlay
    p.add_argument("--wireframe", action="store_true",
                   help="Draw every triangle edge as a debug wireframe overlay on the mag/phase atlas.")
    p.add_argument("--wireframe-color", default="rgba(0,0,0,1.0)")
    p.add_argument("--wireframe-width", type=float, default=2.0)
    p.add_argument("--wireframe-opacity", type=float, default=1.0)
    p.add_argument("--wireframe-z-lift", type=float, default=0.0004,
                   help="Small upward offset to reduce z-fighting with the surface.")

    # Sampling
    p.add_argument("--adaptive-sampling", action="store_true")
    p.add_argument("--nr", type=int, default=10)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--min-nr", type=int, default=5)
    p.add_argument("--max-nr", type=int, default=14)
    p.add_argument("--min-nt", type=int, default=16)
    p.add_argument("--max-nt", type=int, default=64)
    p.add_argument("--sampling-gamma", type=float, default=1.4)
    p.add_argument("--probe-r", type=float, default=0.99)
    p.add_argument("--radial-bias-alpha", type=float, default=0.0,
        help=(
            "Bias radial samples toward both center and boundary. "
            "0 = uniform, 1 = strongest smooth endpoint clustering."
        ),
    )

    # Petal mesh
    p.add_argument("--mesh-mode", choices=["rectangular", "petal"], default="petal")
    p.add_argument("--petal-min-nt", type=int, default=6)
    p.add_argument("--petal-beta", type=float, default=0.75)
    p.add_argument("--petal-multiple", type=int, default=2)

    # Mesh postprocessing
    p.add_argument("--postprocess-mesh", choices=[False, "2D", "3D"], default=False,
                   help=(
                       "Flip shared triangle edges when doing so improves local triangle quality. "
                       "2D judges quality in the Re(c), Im(c) plane; "
                       "3D judges quality in the rendered sheet geometry."
                   ))
    p.add_argument("--postprocess-passes", type=int, default=4)
    p.add_argument("--postprocess-epsilon", type=float, default=1e-7)
    p.add_argument("--postprocess-max-diagonal-ratio", type=float, default=1.25,
                   help=(
                       "Skip flips whose new diagonal is longer than this multiple of the old diagonal. "
                       "Use <=0 to disable this guard."
                   ))


    # Long-edge subdivision
    p.add_argument("--split-long-edges", action="store_true",
                   help=(
                       "Debug/quality option: split unusually long 3D triangle edges per sheet, "
                       "recomputing the inserted midpoint's periodic point instead of interpolating height."
                   ))
    p.add_argument("--long-edge-factor", type=float, default=1.9,
                   help="Split edges longer than this multiple of the trimmed mean edge length for the sheet.")
    p.add_argument("--long-edge-trim", type=float, default=0.10,
                   help="Fraction trimmed from each end before computing the reference mean edge length.")
    p.add_argument("--long-edge-passes", type=int, default=1,
                   help="How many repeated long-edge subdivision passes to run per sheet.")
    p.add_argument("--long-edge-max-new-vertices", type=int, default=200000,
                   help="Safety cap on new vertices inserted per sheet by long-edge subdivision.")

    # Appearance
    p.add_argument("--opacity", type=float, default=1.0,
                   help="Use 1.0 to avoid transparency sorting artifacts.")
    p.add_argument("--height-scale", type=float, default=0.5)
    p.add_argument("--reim-opacity", type=float, default=1.0)

    # Phase color wheel
    p.add_argument("--show-phase-wheel", action="store_true")
    p.add_argument("--phase-wheel-size", type=float, default=0.30)
    p.add_argument("--phase-wheel-x", type=float, default=0.99)
    p.add_argument("--phase-wheel-y", type=float, default=1.04)
    p.add_argument("--phase-wheel-px", type=int, default=320)
    p.add_argument("--phase-wheel-ring-radius", type=float, default=96)
    p.add_argument("--phase-wheel-ring-width", type=float, default=28)
    p.add_argument("--phase-wheel-label-radius", type=float, default=130)
    p.add_argument("--phase-wheel-center-offset-x", type=float, default=17)
    p.add_argument("--phase-wheel-center-offset-y", type=float, default=0)

    # Outputs
    p.add_argument("--skip-magphase", action="store_true")
    p.add_argument("--skip-reim", action="store_true")
    p.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "work" / "promote" / "mandelbrot"),
    )
    p.add_argument("--magphase-html", default="mandelbrot_component_atlas_magphase.html")
    p.add_argument("--reim-html", default="mandelbrot_component_atlas_reim.html")

    return p.parse_args()


def rectangular_sheet_mesh(
    Cgrid,
    Zgrid,
    OK,
    n,
    sheet_index,
    height_mode,
    height_scale=1.0,
    postprocess_mesh=False,
    postprocess_passes=2,
    postprocess_epsilon=1e-7,
    postprocess_max_diagonal_ratio=1.25,
    split_long_edges=False,
    long_edge_factor=1.9,
    long_edge_trim=0.10,
    long_edge_passes=1,
    long_edge_max_new_vertices=200000,
):
    """
    Convert one rectangular sheet to Mesh3d-compatible arrays without adding a trace.

    height_mode:
        "magphase" -> z = height_scale * |z_j|, color = arg(z_j)
        "real"     -> z = Re(z_j), blue
        "imag"     -> z = Im(z_j), orange
    """
    nr, nt = Cgrid.shape

    x, y, z_height, colors = [], [], [], []
    sheet_values = []
    valid = np.zeros((nr, nt), dtype=bool)

    for ir in range(nr):
        for it in range(nt):
            c = Cgrid[ir, it]
            z0 = Zgrid[ir, it]

            x.append(c.real)
            y.append(c.imag)

            if OK[ir, it]:
                zs = cycle_points(z0, c, n)
                zj = zs[sheet_index]

                h_val, color_val = _height_and_color_from_sheet_value(
                    zj=zj,
                    height_mode=height_mode,
                    height_scale=height_scale,
                )
                z_height.append(h_val)
                colors.append(color_val)
                sheet_values.append(zj)

                valid[ir, it] = True
            else:
                z_height.append(0.0)
                colors.append("#000000" if height_mode == "magphase" else "#999999")
                sheet_values.append(None)

    I, J, K = make_faces_from_valid_mask(valid, nt)
    I, J, K = improve_triangles_by_edge_flips(
        x=x,
        y=y,
        z=z_height,
        I=I,
        J=J,
        K=K,
        mode=postprocess_mesh,
        passes=postprocess_passes,
        epsilon=postprocess_epsilon,
        max_diagonal_ratio=postprocess_max_diagonal_ratio,
    )

    x, y, z_height, colors, I, J, K, sheet_values = split_long_edges_by_sheet(
        x=x,
        y=y,
        z=z_height,
        colors=colors,
        I=I,
        J=J,
        K=K,
        sheet_values=sheet_values,
        n=n,
        height_mode=height_mode,
        height_scale=height_scale,
        enabled=split_long_edges,
        factor=long_edge_factor,
        trim_fraction=long_edge_trim,
        passes=long_edge_passes,
        max_new_vertices=long_edge_max_new_vertices,
    )

    return x, y, z_height, colors, I, J, K


def add_rectangular_sheet_outline(outline_lines, Cgrid, Zgrid, OK, n, sheet_index, height_scale, z_lift):
    """
    Add only the outer boundary of one rectangular mag/phase sheet as line segments.
    """
    if outline_lines is None:
        return

    nr, nt = Cgrid.shape
    if nr == 0 or nt < 2:
        return

    ir = nr - 1
    xs, ys, zs, valid = [], [], [], []

    for it in range(nt):
        c = Cgrid[ir, it]
        z0 = Zgrid[ir, it]

        xs.append(c.real)
        ys.append(c.imag)

        if OK[ir, it]:
            zj = cycle_points(z0, c, n)[sheet_index]
            zs.append(height_scale * abs(zj) + z_lift)
            valid.append(True)
        else:
            zs.append(0.0)
            valid.append(False)

    add_cyclic_outline_segments(outline_lines, xs, ys, zs, valid)


def add_petal_sheet_outline(outline_lines, rings, n, sheet_index, height_scale, z_lift):
    """
    Add only the outer boundary of one petal mag/phase sheet as line segments.
    """
    if outline_lines is None or not rings:
        return

    outer = rings[-1]
    if len(outer["C"]) < 2:
        return

    xs, ys, zs, valid = [], [], [], []

    for c, z0, ok in zip(outer["C"], outer["Z"], outer["OK"]):
        xs.append(c.real)
        ys.append(c.imag)

        if ok:
            zj = cycle_points(z0, c, n)[sheet_index]
            zs.append(height_scale * abs(zj) + z_lift)
            valid.append(True)
        else:
            zs.append(0.0)
            valid.append(False)

    add_cyclic_outline_segments(outline_lines, xs, ys, zs, valid)


def collect_component_sheets_rectangular(mag_mesh, reim_mesh, outline_lines, wireframe_lines, Cgrid, Zgrid, OK, n, args):
    """
    Add all sheets for one rectangular component into merged mesh accumulators.
    """
    if mag_mesh is not None:
        for sheet_index in range(n):
            x, y, z_height, colors, I, J, K = rectangular_sheet_mesh(
                Cgrid=Cgrid,
                Zgrid=Zgrid,
                OK=OK,
                n=n,
                sheet_index=sheet_index,
                height_mode="magphase",
                height_scale=args.height_scale,
                postprocess_mesh=args.postprocess_mesh,
                postprocess_passes=args.postprocess_passes,
                postprocess_epsilon=args.postprocess_epsilon,
                postprocess_max_diagonal_ratio=args.postprocess_max_diagonal_ratio,
                split_long_edges=args.split_long_edges,
                long_edge_factor=args.long_edge_factor,
                long_edge_trim=args.long_edge_trim,
                long_edge_passes=args.long_edge_passes,
                long_edge_max_new_vertices=args.long_edge_max_new_vertices,
            )
            mag_mesh.add(x, y, z_height, I, J, K, colors)

            if args.wireframe:
                add_wireframe_segments(
                    wireframe_lines=wireframe_lines,
                    xs=x,
                    ys=y,
                    zs=z_height,
                    I=I,
                    J=J,
                    K=K,
                    z_lift=args.wireframe_z_lift,
                )

            if args.draw_outlines:
                add_rectangular_sheet_outline(
                    outline_lines=outline_lines,
                    Cgrid=Cgrid,
                    Zgrid=Zgrid,
                    OK=OK,
                    n=n,
                    sheet_index=sheet_index,
                    height_scale=args.height_scale,
                    z_lift=args.outline_z_lift,
                )

    if reim_mesh is not None:
        for sheet_index in range(n):
            for mode in ("real", "imag"):
                x, y, z_height, colors, I, J, K = rectangular_sheet_mesh(
                    Cgrid=Cgrid,
                    Zgrid=Zgrid,
                    OK=OK,
                    n=n,
                    sheet_index=sheet_index,
                    height_mode=mode,
                    height_scale=1.0,
                    postprocess_mesh=args.postprocess_mesh,
                    postprocess_passes=args.postprocess_passes,
                    postprocess_epsilon=args.postprocess_epsilon,
                    postprocess_max_diagonal_ratio=args.postprocess_max_diagonal_ratio,
                split_long_edges=args.split_long_edges,
                long_edge_factor=args.long_edge_factor,
                long_edge_trim=args.long_edge_trim,
                long_edge_passes=args.long_edge_passes,
                long_edge_max_new_vertices=args.long_edge_max_new_vertices,
                )
                reim_mesh.add(x, y, z_height, I, J, K, colors)


def collect_component_sheets_petal(mag_mesh, reim_mesh, outline_lines, wireframe_lines, rings, n, args):
    """
    Add all sheets for one petal component into merged mesh accumulators.
    """
    if mag_mesh is not None:
        for sheet_index in range(n):
            x, y, z_height, colors, I, J, K = flatten_petal_sheet(
                rings=rings,
                n=n,
                sheet_index=sheet_index,
                height_mode="magphase",
                height_scale=args.height_scale,
                postprocess_mesh=args.postprocess_mesh,
                postprocess_passes=args.postprocess_passes,
                postprocess_epsilon=args.postprocess_epsilon,
                postprocess_max_diagonal_ratio=args.postprocess_max_diagonal_ratio,
                split_long_edges=args.split_long_edges,
                long_edge_factor=args.long_edge_factor,
                long_edge_trim=args.long_edge_trim,
                long_edge_passes=args.long_edge_passes,
                long_edge_max_new_vertices=args.long_edge_max_new_vertices,
            )
            mag_mesh.add(x, y, z_height, I, J, K, colors)

            if args.wireframe:
                add_wireframe_segments(
                    wireframe_lines=wireframe_lines,
                    xs=x,
                    ys=y,
                    zs=z_height,
                    I=I,
                    J=J,
                    K=K,
                    z_lift=args.wireframe_z_lift,
                )

            if args.draw_outlines:
                add_petal_sheet_outline(
                    outline_lines=outline_lines,
                    rings=rings,
                    n=n,
                    sheet_index=sheet_index,
                    height_scale=args.height_scale,
                    z_lift=args.outline_z_lift,
                )

    if reim_mesh is not None:
        for sheet_index in range(n):
            for mode in ("real", "imag"):
                x, y, z_height, colors, I, J, K = flatten_petal_sheet(
                    rings=rings,
                    n=n,
                    sheet_index=sheet_index,
                    height_mode=mode,
                    height_scale=1.0,
                    postprocess_mesh=args.postprocess_mesh,
                    postprocess_passes=args.postprocess_passes,
                    postprocess_epsilon=args.postprocess_epsilon,
                    postprocess_max_diagonal_ratio=args.postprocess_max_diagonal_ratio,
                split_long_edges=args.split_long_edges,
                long_edge_factor=args.long_edge_factor,
                long_edge_trim=args.long_edge_trim,
                long_edge_passes=args.long_edge_passes,
                long_edge_max_new_vertices=args.long_edge_max_new_vertices,
                )
                reim_mesh.add(x, y, z_height, I, J, K, colors)

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Collect centers
    # --------------------------------------------------------
    components = []
    for n in range(1, args.max_period + 1):
        centers = centers_for_period(n, precision=args.precision)
        print(f"period {n}: {len(centers)} centers")
        for idx, center in enumerate(centers):
            components.append({
                "n": n,
                "center": center,
                "component_index": idx,
            })

    # --------------------------------------------------------
    # Adaptive sampling estimate
    # --------------------------------------------------------
    if args.adaptive_sampling:
        print("estimating component sizes...")
        radii = []
        for comp in components:
            radius = estimate_component_radius(
                n=comp["n"],
                center=comp["center"],
                r_probe=args.probe_r,
                n_steps=5,
            )
            comp["radius"] = radius
            radii.append(radius)

        max_radius = max(radii) if radii else 0.0

        for comp in components:
            nr, nt = sample_counts_from_radius(
                radius=comp["radius"],
                max_radius=max_radius,
                min_nr=args.min_nr,
                max_nr=args.max_nr,
                min_nt=args.min_nt,
                max_nt=args.max_nt,
                gamma=args.sampling_gamma,
            )
            comp["nr"] = nr
            comp["nt"] = nt
    else:
        for comp in components:
            comp["radius"] = None
            comp["nr"] = args.nr
            comp["nt"] = args.nt

    # --------------------------------------------------------
    # Build figures
    # --------------------------------------------------------
    mag_title = (
        f"Mandelbrot attracting-cycle component atlas. Max period: {args.max_period}"
        "<br><sup>Height corresponds to |z|; color represents arg(z).</sup>"
    )

    fig_mag = None if args.skip_magphase else build_figure(
        mag_title,
        "|z|",
        args
    )

    fig_reim = None if args.skip_reim else build_figure(
        "Mandelbrot attracting-cycle component atlas: Re(z) / Im(z)",
        "Re(z), Im(z)",
        args
    )

    mag_mesh = None if fig_mag is None else MeshAccumulator()
    reim_mesh = None if fig_reim is None else MeshAccumulator()
    outline_lines = None if (fig_mag is None or not args.draw_outlines) else LineAccumulator()
    wireframe_lines = None if (fig_mag is None or not args.wireframe) else LineAccumulator()

    # --------------------------------------------------------
    # Trace components and collect sheets into merged meshes
    # --------------------------------------------------------
    mag_center_data = []

    for comp_num, comp in enumerate(components, start=1):
        n = comp["n"]
        center = comp["center"]
        nr = comp["nr"]
        nt = comp["nt"]

        print(f"[{comp_num}/{len(components)}] period={n}, center={center:.6g}, nr={nr}, nt={nt}, mesh={args.mesh_mode}")

        base_name = f"p{n}_c{comp['component_index']}"

        if args.mesh_mode == "rectangular":
            Cgrid, Zgrid, OK = trace_component(
                n=n,
                center=center,
                nr=nr,
                nt=nt,
                rmax=args.rmax,
                radial_bias_alpha=args.radial_bias_alpha,
            )

            if not OK.any():
                print("  -> skipped (no successful points)")
                continue

            collect_component_sheets_rectangular(mag_mesh, reim_mesh, outline_lines, wireframe_lines, Cgrid, Zgrid, OK, n, args)

        elif args.mesh_mode == "petal":
            rings = trace_component_petal(
                n=n,
                center=center,
                nr=nr,
                nt=nt,
                rmax=args.rmax,
                radial_bias_alpha=args.radial_bias_alpha,
                petal_min_nt=args.petal_min_nt,
                petal_beta=args.petal_beta,
                petal_multiple=args.petal_multiple,
            )

            if not any(np.any(ring["OK"]) for ring in rings):
                print("  -> skipped (no successful points)")
                continue

            collect_component_sheets_petal(mag_mesh, reim_mesh, outline_lines, wireframe_lines, rings, n, args)

        else:
            raise ValueError(f"Unknown mesh mode: {args.mesh_mode}")

        if fig_mag is not None and args.show_center_lines:
            center_cycle = cycle_points(0.0 + 0.0j, center, n)
            marker_heights = [args.height_scale * abs(zj) for zj in center_cycle]

            line_top = max(marker_heights) if marker_heights else 0.0

            mag_center_data.append({
                "x": center.real,
                "y": center.imag,
                "marker_heights": marker_heights,
                "line_top": line_top,
            })

    if fig_mag is not None and mag_mesh is not None:
        print(
            f"merged mag/phase mesh: "
            f"{mag_mesh.vertex_count:,} vertices, {mag_mesh.triangle_count:,} triangles"
        )
        mag_mesh.add_to_figure(
            fig=fig_mag,
            name="merged-magphase-atlas",
            opacity=args.opacity,
            enable_hover=args.enable_hover,
        )

    if fig_reim is not None and reim_mesh is not None:
        print(
            f"merged Re/Im mesh: "
            f"{reim_mesh.vertex_count:,} vertices, {reim_mesh.triangle_count:,} triangles"
        )
        reim_mesh.add_to_figure(
            fig=fig_reim,
            name="merged-reim-atlas",
            opacity=args.reim_opacity,
            enable_hover=args.enable_hover,
        )

    if fig_mag is not None and wireframe_lines is not None:
        print(f"wireframe: {wireframe_lines.segment_count:,} line segments")
        wireframe_lines.add_to_figure(
            fig=fig_mag,
            color=args.wireframe_color,
            width=args.wireframe_width,
            opacity=args.wireframe_opacity,
            name="mesh-wireframe",
        )

    if fig_mag is not None and outline_lines is not None:
        print(f"component outlines: {outline_lines.segment_count:,} line segments")
        outline_lines.add_to_figure(
            fig=fig_mag,
            color=args.outline_color,
            width=args.outline_width,
            opacity=args.outline_opacity,
        )

    if fig_mag is not None and args.show_center_lines:
        add_center_guides(
            fig=fig_mag,
            center_data=mag_center_data,
            args=args,
        )

    if fig_mag is not None and args.show_phase_wheel:
        add_phase_color_wheel(fig_mag, args)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if fig_mag is not None:
        out_path = f"{args.output_root.rstrip('/')}/{args.magphase_html}"
        fig_mag.write_html(
            out_path,
            include_plotlyjs="cdn",
            full_html=True,
        )
        print(f"wrote {out_path}")

    if fig_reim is not None:
        out_path = f"{args.output_root.rstrip('/')}/{args.reim_html}"
        fig_reim.write_html(
            out_path,
            include_plotlyjs="cdn",
            full_html=True,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
