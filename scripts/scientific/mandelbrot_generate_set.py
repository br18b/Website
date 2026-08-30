#!/usr/bin/env python3
"""
Generate Mandelbrot images at increasing sampling resolutions.

All saved images have the SAME final plot size, while the sampling grid changes.

This script writes two families of images:

1) Black/white:
    work/promote/mandelbrot/static/mandelbrot_0064.png
    work/promote/mandelbrot/static/mandelbrot_0128.png
    ...

2) Colored:
    work/promote/mandelbrot/static/mandelbrot_color_0064.png
    work/promote/mandelbrot/static/mandelbrot_color_0128.png
    ...

Coloring:
- OUTSIDE: smooth escape-time coloring
- INSIDE: detected attracting-cycle period coloring
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "work"
    / "promote"
    / "mandelbrot"
    / "static"
)

# Sampling grid sizes for the carousel.
RESOLUTIONS = [64, 128, 256, 512, 1024, 2048]

# Mandelbrot viewing window
X_MIN, X_MAX = -2.1, 0.6
Y_MIN, Y_MAX = -1.35, 1.35

# Escape-time settings
MAX_ITER = 2000

# For the Mandelbrot set, 2 is the mathematically correct bailout radius.
ESCAPE_RADIUS = 2.0

# Final Mandelbrot panel size in pixels
OUTPUT_IMAGE_PX = 2048

# Extra vertical space for title strip
TITLE_HEIGHT_PX = 128
TITLE_FONT_SIZE = 32

# Rendering DPI
DPI = 128

SHOW_AXES = False

SAVE_BW = True
SAVE_COLOR = True


# ---------------------------------------------------------------------------
# Coloring knobs
# ---------------------------------------------------------------------------

# -------- Outside coloring --------
OUTSIDE_CMAP = "turbo"          # try: "turbo", "magma", "plasma", "inferno", "viridis"
OUTSIDE_NEUTRAL_COLOR = "#ffffff"

# Percentile used to normalize escape speeds.
# Lower values make the exterior more colorful.
# Higher values keep most of the far exterior pale/white.
OUTSIDE_NORMALIZE_PERCENTILE = 99.5

# Controls contrast of outside coloring.
# Smaller = more color appears farther from the boundary.
# Larger = colors concentrate closer to the boundary.
OUTSIDE_COLOR_GAMMA = 0.75

# Give even fast-escaping points a tiny bit of color.
# Set to 0.0 if you want far exterior to be pure white.
OUTSIDE_MIN_STRENGTH = 0.05

# Reverse colormap direction for exterior.
OUTSIDE_REVERSE = False


# -------- Inside coloring --------
INSIDE_BURNIN_ITERS = 100

# Periods to detect explicitly.
INSIDE_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8]

# Tolerance for f^p(z) ≈ z.
INSIDE_PERIOD_TOL = 1e-6

INSIDE_PERIOD_COLORS = {
    1: "#0b3c8c",  # fixed point
    2: "#1565c0",  # period-2
    3: "#00897b",  # period-3
    4: "#43a047",  # period-4
    5: "#c0ca33",  # period-5
    6: "#fb8c00",  # period-6
    7: "#e53935",  # period-7
    8: "#8e24aa",  # period-8
}

# Bounded but not classified by tested periods.
INSIDE_OTHER_COLOR = "#111111"


# ---------------------------------------------------------------------------
# Mandelbrot computation
# ---------------------------------------------------------------------------

def mandelbrot_escape_data(
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
    escape_radius: float,
):
    """
    Compute Mandelbrot escape data.

    Returns:
        C             complex parameter grid
        Z             final iterate values
        inside_mask   True where point did not escape by max_iter
        escape_iter   integer escape iteration; max_iter for non-escaped points
        smooth_escape smooth escape-time value; max_iter for non-escaped points
    """
    xs = np.linspace(x_min, x_max, width, dtype=np.float64)
    ys = np.linspace(y_min, y_max, height, dtype=np.float64)

    C = xs[None, :] + 1j * ys[:, None]
    Z = np.zeros_like(C, dtype=np.complex128)

    active = np.ones(C.shape, dtype=bool)

    escape_iter = np.full(C.shape, max_iter, dtype=np.int32)
    smooth_escape = np.full(C.shape, float(max_iter), dtype=np.float64)

    log2 = np.log(2.0)

    for n in range(1, max_iter + 1):
        if not np.any(active):
            break

        with np.errstate(over="ignore", invalid="ignore"):
            Z[active] = Z[active] * Z[active] + C[active]

        mag = np.abs(Z)

        escaped_now = active & np.isfinite(mag) & (mag > escape_radius)

        if np.any(escaped_now):
            absz = mag[escaped_now]

            # Smooth escape-time estimate for z -> z^2 + c.
            # This makes exterior coloring less banded.
            with np.errstate(divide="ignore", invalid="ignore"):
                nu = n + 1.0 - np.log(np.log(absz)) / log2

            # Fallback for rare numerical weirdness.
            nu = np.where(np.isfinite(nu), nu, float(n))

            escape_iter[escaped_now] = n
            smooth_escape[escaped_now] = nu
            active[escaped_now] = False

        bad_now = active & (~np.isfinite(mag))
        if np.any(bad_now):
            escape_iter[bad_now] = n
            smooth_escape[bad_now] = float(n)
            active[bad_now] = False

    inside_mask = active
    return C, Z, inside_mask, escape_iter, smooth_escape


# ---------------------------------------------------------------------------
# Inside classification
# ---------------------------------------------------------------------------

def classify_inside_periods(
    C: np.ndarray,
    Z: np.ndarray,
    inside_mask: np.ndarray,
    period_list: list[int],
    burnin_iters: int,
    tol: float,
    escape_radius: float,
) -> np.ndarray:
    """
    Classify bounded points by detected attracting cycle period.

    Returns:
        0 = bounded but not classified / unresolved
        p = detected period p
    """
    periods = np.zeros(C.shape, dtype=np.int16)

    if not np.any(inside_mask):
        return periods

    z_work = Z.copy()
    remaining = inside_mask.copy()

    # Extra burn-in for interior candidates.
    # If something escapes during this extra test, remove it from classification.
    for _ in range(burnin_iters):
        if not np.any(remaining):
            break

        idx = np.flatnonzero(remaining)
        z = z_work.flat[idx]
        c = C.flat[idx]

        with np.errstate(over="ignore", invalid="ignore"):
            z_next = z * z + c

        ok = np.isfinite(z_next) & (np.abs(z_next) <= escape_radius)

        good_idx = idx[ok]
        bad_idx = idx[~ok]

        z_work.flat[good_idx] = z_next[ok]
        remaining.flat[bad_idx] = False

    for p in sorted(set(period_list)):
        if not np.any(remaining):
            break

        idx = np.flatnonzero(remaining)
        z0 = z_work.flat[idx].copy()
        c0 = C.flat[idx]

        zp = z0.copy()
        ok = np.ones(zp.shape, dtype=bool)

        for _ in range(p):
            with np.errstate(over="ignore", invalid="ignore"):
                zp = zp * zp + c0

            ok &= np.isfinite(zp) & (np.abs(zp) <= escape_radius)

        matched = ok & (np.abs(zp - z0) < tol)

        matched_idx = idx[matched]
        bad_idx = idx[~ok]

        periods.flat[matched_idx] = p
        remaining.flat[matched_idx] = False
        remaining.flat[bad_idx] = False

    return periods


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------

def build_bw_image(inside_mask: np.ndarray) -> np.ndarray:
    """
    Build black/white image.

    inside  -> black
    outside -> white
    """
    return np.where(inside_mask, 0.0, 1.0)


def build_outside_rgb(
    inside_mask: np.ndarray,
    smooth_escape: np.ndarray,
    cmap_name: str,
) -> np.ndarray:
    """
    Color escaped points by escape speed.

    Fast escape -> close to white.
    Slow escape -> stronger color.
    """
    h, w = inside_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)

    outside = ~inside_mask
    if not np.any(outside):
        return rgb

    nu = smooth_escape[outside].astype(np.float64)

    finite = np.isfinite(nu)
    if not np.any(finite):
        return rgb

    nu_finite = nu[finite]

    nu_min = float(np.min(nu_finite))
    nu_hi = float(np.percentile(nu_finite, OUTSIDE_NORMALIZE_PERCENTILE))

    if nu_hi <= nu_min:
        nu_hi = nu_min + 1.0

    t = (nu - nu_min) / (nu_hi - nu_min)
    t = np.clip(t, 0.0, 1.0)

    if OUTSIDE_REVERSE:
        t = 1.0 - t

    cmap = plt.get_cmap(cmap_name)
    base = cmap(t)[:, :3]

    neutral = np.array(to_rgb(OUTSIDE_NEUTRAL_COLOR), dtype=np.float64)

    strength = OUTSIDE_MIN_STRENGTH + (1.0 - OUTSIDE_MIN_STRENGTH) * (t ** OUTSIDE_COLOR_GAMMA)
    mixed = (1.0 - strength[:, None]) * neutral[None, :] + strength[:, None] * base

    rgb[outside] = mixed.astype(np.float32)
    return rgb


def build_color_image(
    C: np.ndarray,
    Z: np.ndarray,
    inside_mask: np.ndarray,
    smooth_escape: np.ndarray,
) -> np.ndarray:
    """
    Build RGB image.

    outside -> smooth escape-time coloring
    inside  -> detected period coloring
    """
    rgb = build_outside_rgb(
        inside_mask=inside_mask,
        smooth_escape=smooth_escape,
        cmap_name=OUTSIDE_CMAP,
    )

    periods = classify_inside_periods(
        C=C,
        Z=Z,
        inside_mask=inside_mask,
        period_list=INSIDE_PERIODS,
        burnin_iters=INSIDE_BURNIN_ITERS,
        tol=INSIDE_PERIOD_TOL,
        escape_radius=ESCAPE_RADIUS,
    )

    for p, color in INSIDE_PERIOD_COLORS.items():
        sel = inside_mask & (periods == p)
        if np.any(sel):
            rgb[sel] = to_rgb(color)

    unresolved = inside_mask & (periods == 0)
    if np.any(unresolved):
        rgb[unresolved] = to_rgb(INSIDE_OTHER_COLOR)

    return rgb


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_image(
    img: np.ndarray,
    path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    title: str | None = None,
    fontsize: int = 16,
) -> None:
    """
    Save a grayscale or RGB image.

    The fractal plot area is exactly OUTPUT_IMAGE_PX x OUTPUT_IMAGE_PX pixels.
    The title strip is added above it.
    """
    total_width_px = OUTPUT_IMAGE_PX
    total_height_px = OUTPUT_IMAGE_PX + TITLE_HEIGHT_PX

    fig = plt.figure(
        figsize=(total_width_px / DPI, total_height_px / DPI),
        dpi=DPI,
    )

    title_frac = TITLE_HEIGHT_PX / total_height_px
    plot_frac = OUTPUT_IMAGE_PX / total_height_px

    ax = fig.add_axes([0, 0, 1, plot_frac])

    if img.ndim == 2:
        ax.imshow(
            img,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
        )

    if SHOW_AXES:
        ax.set_xlabel(r"$\Re(c)$")
        ax.set_ylabel(r"$\Im(c)$")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        fig.text(
            0.5,
            plot_frac + title_frac * 0.5,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in RESOLUTIONS:
        print(
            f"Generating {n} x {n} sampling grid "
            f"-> {OUTPUT_IMAGE_PX} x {OUTPUT_IMAGE_PX} image..."
        )

        C, Z, inside_mask, escape_iter, smooth_escape = mandelbrot_escape_data(
            width=n,
            height=n,
            x_min=X_MIN,
            x_max=X_MAX,
            y_min=Y_MIN,
            y_max=Y_MAX,
            max_iter=MAX_ITER,
            escape_radius=ESCAPE_RADIUS,
        )

        title = f"Sampling grid size: {n} by {n}"

        if SAVE_BW:
            bw_img = build_bw_image(inside_mask)
            bw_path = OUTPUT_DIR / f"mandelbrot_{n:04d}.png"

            save_image(
                bw_img,
                bw_path,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                title=title,
                fontsize=TITLE_FONT_SIZE,
            )

            print(f"Wrote {bw_path}")

        if SAVE_COLOR:
            color_img = build_color_image(
                C=C,
                Z=Z,
                inside_mask=inside_mask,
                smooth_escape=smooth_escape,
            )
            color_path = OUTPUT_DIR / f"mandelbrot_color_{n:04d}.png"

            save_image(
                color_img,
                color_path,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                title=title,
                fontsize=TITLE_FONT_SIZE,
            )

            print(f"Wrote {color_path}")

    print("Done.")


if __name__ == "__main__":
    main()