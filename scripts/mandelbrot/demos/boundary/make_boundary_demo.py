#!/usr/bin/env python3
"""Generate the Mandelbrot almost-boundary iframe demo.

The browser demo is intentionally precomputed: static potential images, hidden
index maps, and decimated contour polylines. JavaScript only does canvas drawing
and pixel lookup.
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.project_root import expand_project_vars
from common.repo_config import RepoConfig, add_config_argument


DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": "$project_root/work/promote/mandelbrot/demos/boundary",
    "overview": {
        "width": 1200,
        "height": 760,
        "xmin": -2.84,
        "xmax": 1.1,
        "ymin": -1.25,
        "ymax": 1.25,
        "max_iter": 1500,
    },
    "potential": {
        "relative_tolerance": 1.0e-5,
        "stable_steps": 2,
        "overflow_radius": 1.0e140,
        "numeric_floor": 1.0e-300,
    },
    "levels": {
        "count": 180,
        "mode": "quantile",
        "min": "auto",
        "max": 0.5,
        "auto_min_support_pixels": 256,
        "auto_min_support_fraction": 5.0e-4,
        "max_samples_per_view": 250000,
    },
    "color": {
        "scheme": "viridis",
        "reverse": False,
        "min": "auto",
        "max": "auto",
        "lower_quantile": 0.002,
        "upper_quantile": 0.995,
        "gamma": 0.62,
    },
    "contour_style": {
        "faint_overview_rgba": "rgba(0,0,0,0.70)",
        "faint_zoom_rgba": "rgba(0,0,0,0.28)",
        "selected_rgba": "rgba(0,0,0,1.0)",
        "faint_width": 1.15,
        "faint_count": 24,
        "selected_overview_width": 2.2,
        "selected_zoom_width": 2.8,
    },
    "window_style": {
        "idle_stroke_rgba": "rgba(255,52,72,0.13)",
        "idle_fill_rgba": "rgba(255,52,72,0.00)",
        "normal_stroke_rgba": "rgba(255,52,72,0.50)",
        "normal_fill_rgba": "rgba(255,52,72,0.035)",
        "hover_stroke_rgba": "rgba(255,52,72,1.00)",
        "hover_fill_rgba": "rgba(255,52,72,0.16)",
        "idle_line_width": 1.0,
        "normal_line_width": 2.0,
        "hover_line_width": 3.0,
        "corner_radius": 0.0,
        "hover_distance_pixels": 38.0,
        "proximity_range_box_widths": 1.0,
        "proximity_falloff": "smoothstep",
        "show_hover_label": True,
        "label_prefix": "click: ",
        "label_font": "700 15px system-ui, -apple-system, Segoe UI, sans-serif",
        "label_background_rgba": "rgba(5,12,26,0.78)",
        "label_text_rgba": "rgba(255,255,255,1.00)",
        "label_padding_x": 8.0,
        "label_height": 25.0,
        "label_corner_radius": 10.0,
        "label_gap": 8.0,
    },
    "windows_global_settings": {
        "width": 1200,
        "height": 760,
        "max_iter": 2000,
    },
    "windows": [
        {
            "id": "seahorse",
            "label": "Seahorse valley",
            "bounds": [[-0.88, -0.02], [-0.58, 0.28]],
            "width": 1200,
            "height": 760,
            "max_iter": 2000,
        },
        {
            "id": "antenna",
            "label": "Antenna tip",
            "bounds": [[-1.95, -0.08], [-1.65, 0.12]],
        },
        {
            "id": "satellite",
            "label": "Satellite bulb cusp",
            "bounds": [[-1.34, -0.16], [-1.06, 0.14]],
        },
        {
            "id": "main_cusp",
            "label": "Main cusp fjord",
            "bounds": [[0.17, -0.20], [0.36, 0.20]],
        },
    ],
}


def _temporary_sibling(path: Path) -> Path:
    """Return a hidden temporary path in the same directory as ``path``.

    Keeping the temporary file beside the destination makes ``os.replace``
    atomic on the destination filesystem. The leading dot also keeps Jekyll
    from treating an in-progress file as a publishable asset.
    """
    return path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def save_png_atomic(
    image: Image.Image,
    path: Path,
    *,
    optimize: bool = True,
) -> None:
    """Write and verify a complete PNG before atomically publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        image.save(temporary, format="PNG", optimize=optimize)

        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"PNG writer produced an empty file: {temporary}")

        with Image.open(temporary) as check:
            check.verify()

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def css_color_to_rgb(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """Best-effort conversion of #rgb/#rrggbb/rgb()/rgba() to preview RGB."""
    text = str(value).strip()

    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in {3, 4}:
            digits = "".join(ch * 2 for ch in digits[:3])
        elif len(digits) in {6, 8}:
            digits = digits[:6]
        else:
            return fallback
        try:
            return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return fallback

    match = re.fullmatch(
        r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)"
        r"(?:\s*,\s*[0-9.]+\s*)?\)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        values = []
        for raw in match.groups():
            try:
                values.append(max(0, min(255, int(round(float(raw))))))
            except ValueError:
                return fallback
        return tuple(values)

    return fallback


def load_or_create_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    print(f"wrote default config: {path}")
    return json.loads(json.dumps(DEFAULT_CONFIG))


def normalized_window(
    raw: Any,
    idx: int,
    overview: dict[str, Any],
    global_settings: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one zoom window, applying global defaults then local overrides."""
    default_width = int(global_settings.get("width", overview["width"]))
    default_height = int(global_settings.get("height", overview["height"]))
    default_max_iter = int(
        global_settings.get("max_iter", overview.get("max_iter", 1000))
    )

    for key, value in {
        "width": default_width,
        "height": default_height,
        "max_iter": default_max_iter,
    }.items():
        if value <= 0:
            raise ValueError(f"windows_global_settings.{key} must be positive")

    if isinstance(raw, dict):
        bounds = raw.get("bounds")
        if bounds is None:
            bounds = [[raw["xmin"], raw["ymin"]], [raw["xmax"], raw["ymax"]]]
        label = raw.get("label", f"region {idx + 1}")
        win_id = raw.get(
            "id",
            re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
            or f"region_{idx + 1}",
        )
        width = int(raw.get("width", default_width))
        height = int(raw.get("height", default_height))
        max_iter = int(raw.get("max_iter", default_max_iter))
    else:
        bounds = raw
        label = f"region {idx + 1}"
        win_id = f"region_{idx + 1}"
        width = default_width
        height = default_height
        max_iter = default_max_iter

    if width <= 0 or height <= 0 or max_iter <= 0:
        raise ValueError(
            f"Window {win_id!r} has invalid width/height/max_iter: "
            f"{width}x{height}, {max_iter}"
        )

    (x0, y0), (x1, y1) = bounds
    xmin, xmax = sorted([float(x0), float(x1)])
    ymin, ymax = sorted([float(y0), float(y1)])

    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"Window {win_id!r} has degenerate bounds: {bounds!r}")

    return {
        "id": str(win_id),
        "label": str(label),
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height,
        "max_iter": max_iter,
    }


def grid_coordinates(view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    W, H = int(view["width"]), int(view["height"])
    xs = np.linspace(view["xmin"], view["xmax"], W, endpoint=False) + (view["xmax"] - view["xmin"]) / (2 * W)
    ys = np.linspace(view["ymax"], view["ymin"], H, endpoint=False) + (view["ymin"] - view["ymax"]) / (2 * H)
    return np.meshgrid(xs, ys)


def potential_grid(view: dict[str, Any], potential_cfg: dict[str, Any]) -> np.ndarray:
    """Compute an approximate escape-potential grid without underflow-driven junk.

    Once |z| > 2, convergence is monitored in log(G):

        log G_n = log(log|z_n|) - n log 2.

    Comparing in log-space avoids the fake 4.94e-324 values produced when
    2**(-n) underflows. Values below ``numeric_floor`` are treated as unresolved
    zero because they cannot be represented or resolved meaningfully by this
    finite pixel grid anyway.
    """
    W, H = int(view["width"]), int(view["height"])
    max_iter = int(view.get("max_iter", 1000))
    rel_tol = float(potential_cfg.get("relative_tolerance", 1.0e-5))
    stable_steps = max(1, int(potential_cfg.get("stable_steps", 2)))
    overflow_radius = float(potential_cfg.get("overflow_radius", 1.0e140))
    numeric_floor = max(float(potential_cfg.get("numeric_floor", 1.0e-300)), 0.0)

    if not (overflow_radius > 2.0 and math.isfinite(overflow_radius)):
        raise ValueError("potential.overflow_radius must be finite and greater than 2")
    if not (0.0 < rel_tol < 1.0):
        raise ValueError("potential.relative_tolerance must lie between 0 and 1")

    overflow2 = overflow_radius * overflow_radius
    ln2 = math.log(2.0)
    log_floor = math.log(numeric_floor) if numeric_floor > 0 else -math.inf
    log_rel_tol = -math.log1p(-rel_tol)

    X, Y = grid_coordinates(view)
    C = (X + 1j * Y).reshape(-1)
    Z = np.zeros_like(C, dtype=np.complex128)
    G = np.zeros(C.shape, dtype=np.float64)
    prev_log_g = np.full(C.shape, np.nan, dtype=np.float64)
    stable = np.zeros(C.shape, dtype=np.int16)
    escaped_once = np.zeros(C.shape, dtype=bool)
    active = np.ones(C.shape, dtype=bool)

    def finish(indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        lg = prev_log_g[indices]
        keep = np.isfinite(lg) & (lg >= log_floor)
        G[indices[keep]] = np.exp(lg[keep])
        G[indices[~keep]] = 0.0
        active[indices] = False

    for n in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        with np.errstate(over="ignore", invalid="ignore"):
            Zi = Z[idx] * Z[idx] + C[idx]
        Z[idx] = Zi

        zr = Zi.real
        zi = Zi.imag
        with np.errstate(over="ignore", invalid="ignore"):
            abs2 = zr * zr + zi * zi
        finite = np.isfinite(abs2)
        bad = ~finite | (abs2 > overflow2)

        if bad.any():
            finish(idx[bad])

        good = finite & ~bad
        if not good.any():
            continue

        good_idx = idx[good]
        good_abs2 = abs2[good]
        escaped = good_abs2 > 4.0
        if not escaped.any():
            continue

        esc_idx = good_idx[escaped]
        log_abs = 0.5 * np.log(good_abs2[escaped])
        log_g_now = np.log(log_abs) - n * ln2

        was_escaped = escaped_once[esc_idx]
        fresh_idx = esc_idx[~was_escaped]
        if fresh_idx.size:
            escaped_once[fresh_idx] = True
            prev_log_g[fresh_idx] = log_g_now[~was_escaped]
            stable[fresh_idx] = 0

        old_idx = esc_idx[was_escaped]
        if old_idx.size:
            old_log = prev_log_g[old_idx]
            now_log = log_g_now[was_escaped]
            delta = np.abs(now_log - old_log)
            converged = np.isfinite(delta) & (delta <= log_rel_tol)
            stable[old_idx[converged]] += 1
            stable[old_idx[~converged]] = 0
            prev_log_g[old_idx] = now_log

            done = stable[old_idx] >= stable_steps
            if done.any():
                finish(old_idx[done])

    finish(np.flatnonzero(active & escaped_once))
    return G.reshape(H, W)

def positive_values(G: np.ndarray) -> np.ndarray:
    return G[np.isfinite(G) & (G > 0)]


def supported_minimum(values: np.ndarray, support_pixels: int) -> float:
    """Smallest value with at least ``support_pixels`` samples beneath it."""
    if values.size == 0:
        return math.nan
    k = min(max(int(support_pixels) - 1, 0), values.size - 1)
    return float(np.partition(values, k)[k])


def choose_levels(grids: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """Choose common levels that are dense where the rendered grids have pixels."""
    count = max(8, int(cfg.get("count", 180)))
    mode = str(cfg.get("mode", "quantile")).strip().lower()
    support_pixels = max(1, int(cfg.get("auto_min_support_pixels", 256)))
    support_fraction = max(0.0, float(cfg.get("auto_min_support_fraction", 5.0e-4)))
    raw_min = cfg.get("min", "auto")
    raw_max = cfg.get("max", "auto")

    per_view = {name: positive_values(G) for name, G in grids.items()}
    per_view = {name: vals for name, vals in per_view.items() if vals.size}
    if not per_view:
        raise RuntimeError("No positive potential values were found.")

    all_values = np.concatenate(list(per_view.values()))
    actual_min = min(
        supported_minimum(v, max(support_pixels, int(math.ceil(v.size * support_fraction))))
        for v in per_view.values()
    )
    actual_max = float(np.max(all_values))

    gmin = actual_min if str(raw_min).strip().lower() == "auto" else float(raw_min)
    gmax = actual_max if str(raw_max).strip().lower() == "auto" else min(float(raw_max), actual_max)

    if not (math.isfinite(gmin) and gmin > 0):
        raise RuntimeError(f"Invalid contour minimum: {gmin!r}")
    if not (math.isfinite(gmax) and gmax > gmin):
        raise RuntimeError(f"Invalid contour range: {gmin!r} .. {gmax!r}")

    clipped_logs = []
    max_samples_per_view = max(10_000, int(cfg.get("max_samples_per_view", 250_000)))
    rng = np.random.default_rng(0)
    for vals in per_view.values():
        vals = vals[(vals >= gmin) & (vals <= gmax)]
        if vals.size > max_samples_per_view:
            vals = vals[rng.choice(vals.size, max_samples_per_view, replace=False)]
        if vals.size:
            clipped_logs.append(np.log(vals))
    if not clipped_logs:
        raise RuntimeError("No potential samples lie inside the requested level range.")
    log_values = np.concatenate(clipped_logs)

    if mode == "geometric":
        levels = np.geomspace(gmin, gmax, count)
    elif mode == "quantile":
        q = np.linspace(0.0, 1.0, count)
        levels = np.exp(np.quantile(log_values, q))
        levels[0] = gmin
        levels[-1] = gmax
        levels = np.unique(levels)
        if levels.size < count:
            levels = np.unique(np.concatenate([levels, np.geomspace(gmin, gmax, count)]))
            pick = np.linspace(0, levels.size - 1, count).round().astype(int)
            levels = levels[pick]
    else:
        raise ValueError("levels.mode must be 'quantile' or 'geometric'")

    return np.asarray(levels, dtype=np.float64)


def resolve_color_range(G: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    vals = positive_values(G)
    if vals.size == 0:
        return 1.0, 2.0

    raw_min = cfg.get("min", "auto")
    raw_max = cfg.get("max", "auto")
    qlo = min(max(float(cfg.get("lower_quantile", 0.002)), 0.0), 1.0)
    qhi = min(max(float(cfg.get("upper_quantile", 0.995)), qlo + 1.0e-9), 1.0)

    gmin = float(np.quantile(vals, qlo)) if str(raw_min).lower() == "auto" else float(raw_min)
    gmax = float(np.quantile(vals, qhi)) if str(raw_max).lower() == "auto" else float(raw_max)
    if not (gmin > 0 and math.isfinite(gmin)):
        gmin = float(vals.min())
    if not (gmax > gmin and math.isfinite(gmax)):
        gmax = float(vals.max())
    return gmin, gmax


def _electric_palette(values: np.ndarray) -> np.ndarray:
    """The original dark-blue -> cyan -> gold -> white potential palette."""
    stops = np.array([
        [0.00,   2,   6,  70],
        [0.30,   0,  44, 155],
        [0.56,   0, 180, 240],
        [0.78, 238, 190,  58],
        [1.00, 255, 255, 240],
    ], dtype=float)

    rgb = np.empty((values.size, 3), dtype=float)
    for k in range(len(stops) - 1):
        m = (values >= stops[k, 0]) & (values <= stops[k + 1, 0])
        if not m.any():
            continue
        u = (values[m] - stops[k, 0]) / (stops[k + 1, 0] - stops[k, 0])
        rgb[m] = (1 - u)[:, None] * stops[k, 1:] + u[:, None] * stops[k + 1, 1:]
    rgb[values <= stops[0, 0]] = stops[0, 1:]
    rgb[values >= stops[-1, 0]] = stops[-1, 1:]
    return rgb


def palette_rgb(values: np.ndarray, *, scheme: str, reverse: bool = False) -> np.ndarray:
    """Map values in [0,1] to RGB using a configured palette.

    ``electric`` is the hand-tuned palette used by the first demo. Any
    Matplotlib colormap name also works, for example ``turbo``, ``viridis``,
    ``plasma``, ``inferno``, ``magma``, or ``cividis``.
    """
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if reverse:
        values = 1.0 - values

    name = str(scheme).strip() or "electric"
    if name.lower() == "electric":
        return _electric_palette(values)

    try:
        cmap = matplotlib.colormaps[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown color.scheme={name!r}. Use 'electric' or a Matplotlib "
            "colormap such as turbo, viridis, plasma, inferno, magma, or cividis."
        ) from exc
    return np.asarray(cmap(values), dtype=float)[:, :3] * 255.0


def colorize(
    G: np.ndarray,
    *,
    gmin: float,
    gmax: float,
    gamma: float = 0.62,
    scheme: str = "electric",
    reverse: bool = False,
) -> np.ndarray:
    H, W = G.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.isfinite(G) & (G > 0)
    if not mask.any():
        return img

    lo = math.log(max(gmin, 1.0e-300))
    hi = math.log(max(gmax, gmin * 1.001))
    t = (np.log(np.maximum(G[mask], max(gmin, 1.0e-300))) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)

    # 0 = far exterior, 1 = close to the boundary.
    palette_position = np.power(1.0 - t, max(float(gamma), 1.0e-6))
    rgb = palette_rgb(palette_position, scheme=scheme, reverse=reverse)
    img[mask] = np.clip(rgb, 0, 255).astype(np.uint8)
    return img


def index_map(G: np.ndarray, levels: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
    """Encode hover information in an RGBA image.

    R/G contain the uint16 contour index. B contains the pixel state:

    * 0: exterior pixel with a selectable contour;
    * 1: Mandelbrot interior / unresolved zero-potential pixel;
    * 2: exterior pixel outside the drawable contour range.
    """
    H, W = G.shape
    idx = np.full((H, W), 65535, dtype=np.uint16)
    exterior = np.isfinite(G) & (G > 0)
    status = np.full((H, W), 1, dtype=np.uint8)
    status[exterior] = 2

    if available is None:
        available = np.ones(levels.shape, dtype=bool)
    available_indices = np.flatnonzero(available)
    if available_indices.size:
        usable_levels = levels[available_indices]
        mask = exterior & (G >= usable_levels[0]) & (G <= usable_levels[-1])
        if mask.any():
            lg = np.log(G[mask])
            ll = np.log(usable_levels)
            j = np.searchsorted(ll, lg)
            j0 = np.clip(j - 1, 0, len(usable_levels) - 1)
            j1 = np.clip(j, 0, len(usable_levels) - 1)
            local = np.where(np.abs(lg - ll[j0]) <= np.abs(lg - ll[j1]), j0, j1)
            idx[mask] = available_indices[local].astype(np.uint16)
            status[mask] = 0

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., 0] = (idx & 255).astype(np.uint8)
    rgba[..., 1] = ((idx >> 8) & 255).astype(np.uint8)
    rgba[..., 2] = status
    rgba[..., 3] = 255
    return rgba

def decimate_segment(seg: np.ndarray, min_dist: float = 1.0) -> list[float] | None:
    if len(seg) < 2:
        return None
    out = [seg[0]]
    last = seg[0]
    md2 = min_dist * min_dist
    for p in seg[1:-1]:
        dx = p[0] - last[0]
        dy = p[1] - last[1]
        if dx * dx + dy * dy >= md2:
            out.append(p)
            last = p
    out.append(seg[-1])
    if len(out) < 2:
        return None
    flat: list[float] = []
    for x, y in out:
        flat.append(round(float(x), 1))
        flat.append(round(float(y), 1))
    return flat if len(flat) >= 4 else None


def contours_for_grid(G: np.ndarray, levels: np.ndarray, min_dist: float = 1.0) -> list[list[list[float]]]:
    H, W = G.shape
    # Keep the interior at zero. NaN-masking it breaks the lowest contours by
    # preventing interpolation against the zero-potential set.
    Z = np.where(np.isfinite(G) & (G > 0), G, 0.0)
    fig = plt.figure(figsize=(1, 1), dpi=10)
    ax = fig.add_subplot(111)
    result: list[list[list[float]]] = [[] for _ in levels]
    try:
        cs = ax.contour(np.arange(W), np.arange(H), Z, levels=levels, linewidths=1)
        for lev_i, segs in enumerate(cs.allsegs):
            polys: list[list[float]] = []
            for seg in segs:
                dec = decimate_segment(seg, min_dist=min_dist)
                if dec:
                    polys.append(dec)
            result[lev_i] = polys
    finally:
        plt.close(fig)
    return result

def resolve_contour_style(config: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULT_CONFIG["contour_style"])
    raw = config.get("contour_style", {})
    if raw is not None:
        if not isinstance(raw, dict):
            raise TypeError("contour_style must be a JSON object")
        defaults.update(raw)

    for key in ["faint_overview_rgba", "faint_zoom_rgba", "selected_rgba"]:
        value = str(defaults[key]).strip()
        if not value:
            raise ValueError(f"contour_style.{key} may not be empty")
        defaults[key] = value

    for key in [
        "faint_width",
        "selected_overview_width",
        "selected_zoom_width",
    ]:
        value = float(defaults[key])
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"contour_style.{key} must be a positive finite number")
        defaults[key] = value

    faint_count = int(defaults.get("faint_count", 24))
    if faint_count <= 0:
        raise ValueError("contour_style.faint_count must be a positive integer")
    defaults["faint_count"] = faint_count

    return defaults


def resolve_window_style(config: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULT_CONFIG["window_style"])
    raw = config.get("window_style", {})
    if raw is not None:
        if not isinstance(raw, dict):
            raise TypeError("window_style must be a JSON object")
        defaults.update(raw)

    color_keys = [
        "idle_stroke_rgba",
        "idle_fill_rgba",
        "normal_stroke_rgba",
        "normal_fill_rgba",
        "hover_stroke_rgba",
        "hover_fill_rgba",
        "label_background_rgba",
        "label_text_rgba",
    ]
    for key in color_keys:
        value = str(defaults[key]).strip()
        if not value:
            raise ValueError(f"window_style.{key} may not be empty")
        defaults[key] = value

    string_keys = ["label_prefix", "label_font", "proximity_falloff"]
    for key in string_keys:
        defaults[key] = str(defaults[key]).strip()

    positive_keys = [
        "idle_line_width",
        "normal_line_width",
        "hover_line_width",
        "label_padding_x",
        "label_height",
    ]

    nonnegative_keys = [
        "corner_radius",
        "hover_distance_pixels",
        "proximity_range_box_widths",
        "label_corner_radius",
        "label_gap",
    ]
    
    for key in positive_keys:
        value = float(defaults[key])
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"window_style.{key} must be positive and finite")
        defaults[key] = value
    for key in nonnegative_keys:
        value = float(defaults[key])
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"window_style.{key} must be non-negative and finite")
        defaults[key] = value

    falloff = defaults["proximity_falloff"].lower()
    if falloff not in {"linear", "smoothstep"}:
        raise ValueError(
            "window_style.proximity_falloff must be 'linear' or 'smoothstep'"
        )
    defaults["proximity_falloff"] = falloff

    defaults["show_hover_label"] = bool(defaults.get("show_hover_label", True))
    return defaults


def render_templates(
    template_dir: Path,
    out_dir: Path,
    contour_style: dict[str, Any],
    window_style: dict[str, Any],
) -> None:
    required = ["demo.css", "demo.js", "index.html"]
    missing = [name for name in required if not (template_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing template files in {template_dir}: {', '.join(missing)}"
        )

    replacements = {
        "__FAINT_OVERVIEW_RGBA__": json.dumps(
            contour_style["faint_overview_rgba"]
        ),
        "__FAINT_ZOOM_RGBA__": json.dumps(contour_style["faint_zoom_rgba"]),
        "__SELECTED_CONTOUR_RGBA__": json.dumps(
            contour_style["selected_rgba"]
        ),
        "__FAINT_CONTOUR_WIDTH__": repr(contour_style["faint_width"]),
        "__FAINT_CONTOUR_COUNT__": repr(contour_style["faint_count"]),
        "__SELECTED_OVERVIEW_WIDTH__": repr(
            contour_style["selected_overview_width"]
        ),
        "__SELECTED_ZOOM_WIDTH__": repr(
            contour_style["selected_zoom_width"]
        ),
        "__WINDOW_IDLE_STROKE_RGBA__": json.dumps(
            window_style["idle_stroke_rgba"]
        ),
        "__WINDOW_IDLE_FILL_RGBA__": json.dumps(
            window_style["idle_fill_rgba"]
        ),
        "__WINDOW_NORMAL_STROKE_RGBA__": json.dumps(
            window_style["normal_stroke_rgba"]
        ),
        "__WINDOW_NORMAL_FILL_RGBA__": json.dumps(
            window_style["normal_fill_rgba"]
        ),
        "__WINDOW_HOVER_STROKE_RGBA__": json.dumps(
            window_style["hover_stroke_rgba"]
        ),
        "__WINDOW_HOVER_FILL_RGBA__": json.dumps(
            window_style["hover_fill_rgba"]
        ),
        "__WINDOW_IDLE_LINE_WIDTH__": repr(
            window_style["idle_line_width"]
        ),
        "__WINDOW_NORMAL_LINE_WIDTH__": repr(
            window_style["normal_line_width"]
        ),
        "__WINDOW_HOVER_LINE_WIDTH__": repr(
            window_style["hover_line_width"]
        ),
        "__WINDOW_CORNER_RADIUS__": repr(window_style["corner_radius"]),
        "__WINDOW_HOVER_DISTANCE_PIXELS__": repr(
            window_style["hover_distance_pixels"]
        ),
        "__WINDOW_PROXIMITY_RANGE_BOX_WIDTHS__": repr(
            window_style["proximity_range_box_widths"]
        ),
        "__WINDOW_PROXIMITY_FALLOFF__": json.dumps(
            window_style["proximity_falloff"]
        ),
        "__WINDOW_SHOW_HOVER_LABEL__": (
            "true" if window_style["show_hover_label"] else "false"
        ),
        "__WINDOW_LABEL_PREFIX__": json.dumps(window_style["label_prefix"]),
        "__WINDOW_LABEL_FONT__": json.dumps(window_style["label_font"]),
        "__WINDOW_LABEL_BACKGROUND_RGBA__": json.dumps(
            window_style["label_background_rgba"]
        ),
        "__WINDOW_LABEL_TEXT_RGBA__": json.dumps(
            window_style["label_text_rgba"]
        ),
        "__WINDOW_LABEL_PADDING_X__": repr(window_style["label_padding_x"]),
        "__WINDOW_LABEL_HEIGHT__": repr(window_style["label_height"]),
        "__WINDOW_LABEL_CORNER_RADIUS__": repr(
            window_style["label_corner_radius"]
        ),
        "__WINDOW_LABEL_GAP__": repr(window_style["label_gap"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    token_pattern = re.compile(r"__[A-Z0-9_]+__")

    for name in required:
        source = template_dir / name
        text = source.read_text(encoding="utf-8")
        for token, replacement in replacements.items():
            text = text.replace(token, replacement)

        unresolved = sorted(set(token_pattern.findall(text)))
        if unresolved:
            raise ValueError(
                f"Unresolved template token(s) in {source}: "
                + ", ".join(unresolved)
            )

        atomic_write_text(out_dir / name, text)


def validate_generated_assets(out_dir: Path, data: dict[str, Any]) -> None:
    """Fail early if data.json points at a missing or unreadable image."""
    refs: list[tuple[str, str]] = []

    overview = data["views"]["overview"]
    refs.append(("overview background", overview["background"]))
    refs.append(("overview hover index map", overview["indexMap"]))

    for win in data["windows"]:
        view = data["views"]["windows"][win["id"]]
        refs.append((f"{win['label']} background", view["background"]))
        refs.append((f"{win['label']} hover index map", view["indexMap"]))

    errors: list[str] = []
    for label, filename in refs:
        path = out_dir / filename
        if not path.is_file():
            errors.append(f"{label}: missing {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"{label}: unreadable {path}: {exc}")

    if errors:
        raise RuntimeError(
            "Generated demo asset validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    print(f"validated generated image assets: {len(refs)}")


def make_preview(
    out_dir: Path,
    data: dict[str, Any],
    window_style: dict[str, Any],
) -> None:
    ov = Image.open(out_dir / data["views"]["overview"]["background"]).convert("RGB")
    W, H = 1100, 720
    preview = Image.new("RGB", (W, H), (7, 16, 31))
    scale = min((W - 40) / ov.width, (H - 80) / ov.height)
    ov2 = ov.resize((int(ov.width * scale), int(ov.height * scale)), Image.LANCZOS)
    ox, oy = (W - ov2.width) // 2, 24
    preview.paste(ov2, (ox, oy))
    d = ImageDraw.Draw(preview)
    # Draw window boxes using the configured normal marker color.
    overview = data["views"]["overview"]
    marker_rgb = css_color_to_rgb(
        window_style["normal_stroke_rgba"],
        fallback=(255, 52, 72),
    )
    for win in data["windows"]:
        x0 = ox + (win["xmin"] - overview["xmin"]) / (overview["xmax"] - overview["xmin"]) * ov2.width
        x1 = ox + (win["xmax"] - overview["xmin"]) / (overview["xmax"] - overview["xmin"]) * ov2.width
        y0 = oy + (overview["ymax"] - win["ymax"]) / (overview["ymax"] - overview["ymin"]) * ov2.height
        y1 = oy + (overview["ymax"] - win["ymin"]) / (overview["ymax"] - overview["ymin"]) * ov2.height
        d.rectangle(
            [x0, y0, x1, y1],
            outline=marker_rgb,
            width=max(1, int(round(window_style["normal_line_width"]))),
        )
    d.text((28, H - 42), "Click a highlighted region, then hover to reveal escape-potential contours.", fill=(245, 248, 255))
    save_png_atomic(preview, out_dir / "preview.png")


def patch_blog_post(project_root: Path, out_dir: Path) -> None:
    post = project_root / "_posts" / "2026-07-08-fractals2.md"
    if not post.exists():
        return

    try:
        rel = out_dir.resolve().relative_to(project_root.resolve()).as_posix().strip("/")
    except ValueError:
        print(f"warning: output directory is outside project root; cannot derive iframe URL: {out_dir}")
        return

    demo_src = f"/{rel}/index.html"
    preview_src = f"/{rel}/preview.png"
    text = post.read_text(encoding="utf-8")
    replacement = f'''{{% include lazy-iframe.html
   label="Interactive demo"
   title="Almost-boundaries: hover over escape-potential contours"
   src="{demo_src}"
   preview_src="{preview_src}"
   preview_alt="Preview of a Mandelbrot escape-potential contour hover demo"
   button_text="Load demo"
   loading_text="Loading contour demo…"
   height="760px"
   mobile_height="520px"
   fit_content=true
   fit_min_height="360"
   fit_max_height="1100"
   note="Precomputed canvas demo. Click a highlighted region, then move the mouse to reveal the nearest escape-potential contour."
   caption="Clicking a highlighted region opens a magnified view. Hovering over the exterior selects the nearest precomputed value of the escape potential $G(c)$ and draws the corresponding contour."
%}}'''
    static_pattern = re.compile(r'<figure style="text-align: center;">\s*<img src="\{\{ \'/\' \| relative_url \}\}fractal/mandelbrot/potential_contours\.png".*?</figure>', re.S)
    iframe_pattern = re.compile(r'\{% include lazy-iframe\.html\s+label="Interactive demo"\s+title="Almost-boundaries: hover over escape-potential contours".*?%\}', re.S)
    if static_pattern.search(text):
        text = static_pattern.sub(replacement, text)
    elif iframe_pattern.search(text):
        text = iframe_pattern.sub(replacement, text)
    else:
        print("warning: did not find Almost-boundaries figure/include to patch")
        return
    atomic_write_text(post, text)
    print(f"patched post: {post}")

def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--template-dir", type=Path, default=SCRIPT_DIR)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--no-patch-post", action="store_true")
    args = ap.parse_args()

    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    project_root = repo.paths.project_root
    config = repo.section("demo.boundary")
    overview = dict(config["overview"])
    potential_cfg = dict(config.get("potential", {}))
    level_cfg = dict(config.get("levels", {}))
    color_cfg = dict(config.get("color", {}))
    contour_style = resolve_contour_style(config)
    window_style = resolve_window_style(config)
    windows_global_settings = dict(config.get("windows_global_settings", {}))
    windows = [
        normalized_window(w, i, overview, windows_global_settings)
        for i, w in enumerate(config.get("windows", []))
    ]
    if not windows:
        raise RuntimeError("No zoom windows configured.")

    output_value = args.output_dir if args.output_dir is not None else config.get("output_dir", DEFAULT_CONFIG["output_dir"])
    out_dir = expand_project_vars(output_value, code_root=repo.paths.code_root, project_root=project_root)
    if not out_dir.is_absolute():
        out_dir = (SCRIPT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"project_root: {project_root}")
    print(f"output:      {out_dir}")
    render_templates(
        args.template_dir,
        out_dir,
        contour_style,
        window_style,
    )
    print(
        "contour style: "
        f"faint overview={contour_style['faint_overview_rgba']}, "
        f"faint zoom={contour_style['faint_zoom_rgba']}, "
        f"selected={contour_style['selected_rgba']}"
    )
    print(
        "window defaults: "
        f"{windows_global_settings.get('width', overview['width'])}x"
        f"{windows_global_settings.get('height', overview['height'])}, "
        f"max_iter={windows_global_settings.get('max_iter', overview.get('max_iter', 1000))}"
    )
    print(
        "window proximity: "
        f"range={window_style['proximity_range_box_widths']} box width(s), "
        f"falloff={window_style['proximity_falloff']}"
    )

    views: dict[str, dict[str, Any]] = {"overview": overview}
    views.update({w["id"]: w for w in windows})

    grids: dict[str, np.ndarray] = {}
    for name, view in views.items():
        print(f"computing potential grid: {name} ({view['width']}x{view['height']}, max_iter={view.get('max_iter')})")
        grids[name] = potential_grid(view, potential_cfg)

    levels = choose_levels(grids, level_cfg)
    print(f"levels: {levels[0]:.6g} -> {levels[-1]:.6g} ({len(levels)}, mode={level_cfg.get('mode', 'quantile')})")

    print("extracting contours")
    view_contours = {
        name: contours_for_grid(G, levels, min_dist=1.0 if views[name]["width"] >= 900 else 0.75)
        for name, G in grids.items()
    }

    asset_version = str(time.time_ns())
    gamma = float(color_cfg.get("gamma", 0.62))
    color_scheme = str(color_cfg.get("scheme", "electric"))
    color_reverse = bool(color_cfg.get("reverse", False))
    print(f"color scheme: {color_scheme} (reverse={color_reverse})")
    for name, G in grids.items():
        cmin, cmax = resolve_color_range(G, color_cfg)
        save_png_atomic(
            Image.fromarray(
                colorize(
                    G,
                    gmin=cmin,
                    gmax=cmax,
                    gamma=gamma,
                    scheme=color_scheme,
                    reverse=color_reverse,
                ),
                "RGB",
            ),
            out_dir / f"{name}.png",
        )
        available = np.asarray(
            [bool(polys) for polys in view_contours[name]],
            dtype=bool,
        )
        save_png_atomic(
            Image.fromarray(index_map(G, levels, available), "RGBA"),
            out_dir / f"{name}_index.png",
        )
        print(f"  {name}: color={cmin:.6g}..{cmax:.6g}, drawable_levels={int(available.sum())}/{len(levels)}")

    contour_records = []
    for i, g in enumerate(levels):
        contour_records.append({
            "G": float(g),
            "overview": view_contours["overview"][i],
            "windows": {w["id"]: view_contours[w["id"]][i] for w in windows},
        })

    views_json = {
        "overview": {k: overview[k] for k in ["width", "height", "xmin", "xmax", "ymin", "ymax"]} | {
            "background": "overview.png",
            "indexMap": "overview_index.png",
        },
        "windows": {},
    }
    for w in windows:
        views_json["windows"][w["id"]] = {k: w[k] for k in ["width", "height", "xmin", "xmax", "ymin", "ymax"]} | {
            "background": f"{w['id']}.png",
            "indexMap": f"{w['id']}_index.png",
        }

    data = {
        "assetVersion": asset_version,
        "levels": [float(x) for x in levels],
        "views": views_json,
        "windows": [{k: w[k] for k in ["id", "label", "xmin", "xmax", "ymin", "ymax"]} for w in windows],
        "contours": contour_records,
    }
    validate_generated_assets(out_dir, data)
    make_preview(out_dir, data, window_style)

    # Publish data.json last. The browser only learns the new assetVersion after
    # every referenced image and template has been atomically replaced.
    atomic_write_text(
        out_dir / "data.json",
        json.dumps(data, separators=(",", ":")),
    )

    atomic_write_text(
        out_dir / "README.md",
        "# Almost-boundary Mandelbrot demo\n\n"
        "Precomputed canvas demo for the Fractals II post.\n\n"
        "Regenerate with:\n\n"
        "```bash\ncd scripts/mandelbrot\n"
        "python3 demos/boundary/make_boundary_demo.py\n"
        "```\n",
    )

    if not args.no_patch_post:
        patch_blog_post(project_root, out_dir)

    print("done")
    for name in ["data.json", "overview.png", "preview.png"]:
        p = out_dir / name
        if p.exists():
            print(f"{name}: {p.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
