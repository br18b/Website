#!/usr/bin/env python3
"""Plot the small derived outputs produced by the C++ contour tools.

Typical workflow:

    ./build.sh
    ./bin/contours
    ./bin/postprocess_contours
    python3 contours/postprocess_contours.py

All programs read the repository-level ``mandelbrot.json`` unless an alternate
JSON configuration is passed with ``--config``.  This plotter never reads the
large raw contour files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.repo_config import RepoConfig, add_config_argument

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def cfg_str(cfg: dict[str, str], key: str, default: str) -> str:
    return str(cfg.get(key, default))


def cfg_bool(cfg: dict[str, str], key: str, default: bool) -> bool:
    raw = cfg.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: str | int | float | None, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def read_scaling_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, float] = {}
            for k, v in row.items():
                converted[k] = safe_float(v)
            rows.append(converted)
    rows = [r for r in rows if math.isfinite(r.get("target_G", math.nan))]
    rows.sort(key=lambda r: r["target_G"], reverse=True)
    return rows


def finite_positive(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x) & (x > 0)]



def cfg_float_optional(cfg: dict[str, str], key: str) -> float | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"auto", "none", "nan"}:
        return None
    try:
        value = float(s)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def cfg_scale(cfg: dict[str, str], key: str, default: str) -> str:
    raw = cfg_str(cfg, key, default).strip().lower()
    aliases = {
        "lin": "linear",
        "linear": "linear",
        "log": "log",
        "symlog": "symlog",
    }
    if raw not in aliases:
        raise SystemExit(f"Invalid {key}={raw!r}; use linear/lin, log, or symlog.")
    return aliases[raw]


def cfg_clean_str(cfg: dict[str, str], key: str, default: str) -> str:
    raw = cfg_str(cfg, key, default).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    return raw.strip()


def set_axis_scale_for_values(ax, axis: str, scale: str, values: np.ndarray, *, symlog_threshold: float = 1.0) -> str:
    finite = values[np.isfinite(values)]
    if scale == "linear":
        return "linear"
    if scale == "symlog":
        if not (symlog_threshold > 0 and math.isfinite(symlog_threshold)):
            symlog_threshold = 1.0
        if axis == "x":
            ax.set_xscale("symlog", linthresh=symlog_threshold)
        else:
            ax.set_yscale("symlog", linthresh=symlog_threshold)
        return "symlog"
    if scale == "log":
        if finite.size and np.all(finite > 0):
            if axis == "x":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")
            return "log"
        print(f"  warning: {axis}-scale=log requested but data include non-positive values; using linear.", flush=True)
        return "linear"
    raise ValueError(scale)



def _plain_log_tick(value: float, _position: int | None = None) -> str:
    """Format useful log ticks as ordinary numbers where practical."""
    if not (math.isfinite(value) and value > 0):
        return ""

    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0 ** exponent)

    for candidate in (1.0, 2.0, 5.0):
        if math.isclose(mantissa, candidate, rel_tol=1.0e-9, abs_tol=1.0e-12):
            mantissa = candidate
            break

    if -3 <= exponent <= 4:
        return f"{value:g}"

    if mantissa == 1.0:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def configure_readable_log_ticks(
    ax,
    axis: str,
    values: np.ndarray,
    cfg: dict[str, str],
) -> None:
    """Use labelled 1/2/5 ticks when a log axis spans only a few decades.

    Matplotlib normally labels only exact powers of ten on logarithmic axes.
    That is excellent over many decades, but an axis spanning roughly 2..20
    may then receive only the label 10. In ``auto`` mode, narrow log ranges use
    labelled 1/2/5 ticks; wide ranges keep the conventional decade labels.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return

    mode = cfg_clean_str(cfg, "postprocess_log_tick_labels", "auto").lower()
    aliases = {
        "auto": "auto",
        "decades": "decades",
        "decade": "decades",
        "powers": "decades",
        "125": "125",
        "1-2-5": "125",
        "dense": "125",
    }
    if mode not in aliases:
        raise SystemExit(
            "postprocess_log_tick_labels must be auto, decades, or 1-2-5."
        )
    mode = aliases[mode]

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span_decades = math.log10(hi / lo) if hi > lo else 0.0
    dense_limit = float(
        cfg_str(cfg, "postprocess_log_dense_max_decades", "2.5")
    )
    if not (math.isfinite(dense_limit) and dense_limit > 0):
        raise SystemExit(
            "postprocess_log_dense_max_decades must be positive and finite."
        )

    use_dense = mode == "125" or (
        mode == "auto" and span_decades <= dense_limit
    )

    axis_obj = ax.xaxis if axis == "x" else ax.yaxis

    if use_dense:
        axis_obj.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=30)
        )
        axis_obj.set_major_formatter(mticker.FuncFormatter(_plain_log_tick))
        axis_obj.set_minor_locator(
            mticker.LogLocator(
                base=10.0,
                subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
                numticks=100,
            )
        )
        axis_obj.set_minor_formatter(mticker.NullFormatter())
    else:
        axis_obj.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=15)
        )
        axis_obj.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        axis_obj.set_minor_locator(
            mticker.LogLocator(
                base=10.0,
                subs=tuple(float(i) for i in range(2, 10)),
                numticks=100,
            )
        )
        axis_obj.set_minor_formatter(mticker.NullFormatter())


def global_line_style(style: str) -> str:
    s = style.strip().lower()
    return {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dash_dotted": "-.",
        "dash-dotted": "-.",
        "dashdot": "-.",
    }.get(s, "-")


def fit_line_kwargs(style: str) -> dict:
    # Example: black,thin,dashed. Unknown tokens are ignored.
    tokens = [t.strip().lower() for t in style.split(",") if t.strip()]
    kwargs: dict = {"color": "black", "linewidth": 1.1, "linestyle": "--"}
    for t in tokens:
        if t in {"black", "gray", "grey", "red", "blue", "green", "orange", "purple"}:
            kwargs["color"] = "gray" if t == "grey" else t
        elif t == "thin":
            kwargs["linewidth"] = 1.0
        elif t == "medium":
            kwargs["linewidth"] = 1.5
        elif t == "thick":
            kwargs["linewidth"] = 2.2
        elif t in {"solid", "dashed", "dotted", "dash_dotted", "dash-dotted", "dashdot"}:
            kwargs["linestyle"] = global_line_style(t)
    return kwargs


def plot_xy(ax, x: np.ndarray, y: np.ndarray, *, label: str | None, style: str, marker: str = "o", linestyle: str = "-") -> None:
    style = style.strip().lower()
    if style == "points":
        ax.plot(x, y, linestyle="None", marker=marker, label=label)
    elif style == "lines":
        ax.plot(x, y, linestyle=linestyle, marker=None, label=label)
    else:  # points_lines
        ax.plot(x, y, linestyle=linestyle, marker=marker, label=label)


def apply_global_axes(
    ax,
    x: np.ndarray,
    y_values: list[np.ndarray],
    *,
    x_scale: str,
    y_scale: str,
    xlabel: str,
    ylabel: str,
    title: str,
    cfg: dict[str, str],
) -> tuple[str, str]:
    all_y = np.concatenate([y[np.isfinite(y)] for y in y_values if np.any(np.isfinite(y))]) if y_values else np.array([])
    x_eff = set_axis_scale_for_values(ax, "x", x_scale, x, symlog_threshold=1.0)
    y_eff = set_axis_scale_for_values(ax, "y", y_scale, all_y, symlog_threshold=1.0)

    if x_eff == "log":
        configure_readable_log_ticks(ax, "x", x, cfg)
    if y_eff == "log":
        configure_readable_log_ticks(ax, "y", all_y, cfg)

    ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    return x_eff, y_eff


def parse_power_fit_equation(equation: str) -> tuple[str, str]:
    """Return (model_key, variable) for the supported small fit language.

    This is intentionally not a full expression parser. It recognizes a few
    explicit model strings and treats c1,c2,c3 as fitted constants, G/rho as
    variables, pi as math.pi, and exp as the exponential function.
    """
    eq = equation.strip().strip('"').strip("'")
    compact = (
        eq.replace(" ", "")
        .lower()
        .replace("**", "^")
        .replace("π", "pi")
    )

    if compact in {"c1+c2*g^c3", "c1+c2*x^c3"}:
        return "power_offset", "G"
    if compact in {"c1+c2*rho^c3", "c1+c2*ρ^c3"}:
        return "power_offset", "rho"

    if compact in {
        "pi*exp(2*g)-c1+c2*g^c3",
        "pi*exp(2*g)-c1+c2*x^c3",
        "π*exp(2*g)-c1+c2*g^c3",
    }:
        return "disk_defect_power", "G"
    if compact in {
        "pi*exp(2*g)-c1+c2*rho^c3",
        "pi*exp(2*g)-c1+c2*ρ^c3",
        "π*exp(2*g)-c1+c2*rho^c3",
    }:
        return "disk_defect_power", "rho"

    raise SystemExit(
        "Unsupported fit equation "
        f"{equation!r}. Supported: c1+c2*G^c3, c1+c2*rho^c3, "
        "pi*exp(2*G)-c1+c2*G^c3, and pi*exp(2*G)-c1+c2*rho^c3."
    )


def select_fit_mask(G: np.ndarray, y: np.ndarray, cfg: dict[str, str], prefix: str) -> np.ndarray:
    # The G axis is usually plotted inverted, so users often think visually:
    #   x_min = auto, x_max = 1e-3
    # means "fit the low-G tail up to 1e-3", i.e. G <= 1e-3.
    #
    # To make this robust, if exactly one bound is numeric, we interpret it as a
    # one-sided low-G tail cutoff: G <= bound. If both bounds are numeric, use the
    # ordinary numeric interval between them.
    xmin = cfg_float_optional(cfg, f"postprocess_{prefix}_fit_x_min")
    xmax = cfg_float_optional(cfg, f"postprocess_{prefix}_fit_x_max")
    mask = np.isfinite(G) & np.isfinite(y) & (G > 0)
    if xmin is not None and xmax is not None:
        lo, hi = sorted((xmin, xmax))
        mask &= (G >= lo) & (G <= hi)
    elif xmin is not None:
        mask &= G <= xmin
    elif xmax is not None:
        mask &= G <= xmax
    return mask


def fit_offset_power_model(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        return None

    def solve_for_alpha(alpha: float) -> tuple[float, float, float]:
        xa = np.power(x, alpha)
        A = np.column_stack([np.ones_like(xa), xa])
        try:
            c1, c2 = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return math.inf, math.nan, math.nan
        resid = y - (c1 + c2 * xa)
        sse = float(np.dot(resid, resid))
        return sse, float(c1), float(c2)

    # Coarse scan over a broad alpha range. Negative powers are useful for length,
    # positive powers for area convergence.
    grid = np.linspace(-3.0, 3.0, 1201)
    vals = [solve_for_alpha(float(a))[0] for a in grid]
    best_i = int(np.nanargmin(vals))
    a0 = float(grid[best_i])

    # Golden-section refinement around the best grid cell.
    step = float(grid[1] - grid[0])
    lo = max(-6.0, a0 - 4.0 * step)
    hi = min(6.0, a0 + 4.0 * step)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = solve_for_alpha(c)[0]
    fd = solve_for_alpha(d)[0]
    for _ in range(80):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = solve_for_alpha(c)[0]
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = solve_for_alpha(d)[0]
    alpha = 0.5 * (lo + hi)
    sse, c1, c2 = solve_for_alpha(alpha)
    rmse = math.sqrt(sse / max(1, x.size))
    ss_tot = float(np.dot(y - float(np.mean(y)), y - float(np.mean(y))))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else math.nan
    return {"c1": c1, "c2": c2, "c3": alpha, "sse": sse, "rmse": rmse, "r2": r2, "n": float(x.size)}


def fit_disk_defect_power_model(G: np.ndarray, y: np.ndarray, variable: str) -> dict[str, float] | None:
    """Fit y = pi*exp(2G) - c1 + c2*x^c3, x=G or rho.

    Equivalently, fit the disk-area defect
        D(G) = pi*exp(2G) - y = c1 - c2*x^c3.
    Here c1 is the extrapolated defect D(0), so the inferred limiting
    Mandelbrot area is pi - c1.
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.expm1(G) if variable == "rho" else G
    disk = math.pi * np.exp(2.0 * G)
    mask = np.isfinite(G) & np.isfinite(y) & np.isfinite(x) & np.isfinite(disk) & (x > 0)
    G = G[mask]
    y = y[mask]
    x = x[mask]
    disk = disk[mask]
    if x.size < 4:
        return None

    target = y - disk

    def solve_for_alpha(alpha: float) -> tuple[float, float, float]:
        xa = np.power(x, alpha)
        # target = -c1 + c2*x^alpha
        A = np.column_stack([-np.ones_like(xa), xa])
        try:
            c1, c2 = np.linalg.lstsq(A, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            return math.inf, math.nan, math.nan
        pred = disk - c1 + c2 * xa
        resid = y - pred
        sse = float(np.dot(resid, resid))
        return sse, float(c1), float(c2)

    # For the area-defect correction c3 should normally be positive, but keep
    # a little negative range so the optimizer can expose a bad/unphysical fit.
    grid = np.linspace(-1.0, 3.0, 1201)
    vals = [solve_for_alpha(float(a))[0] for a in grid]
    best_i = int(np.nanargmin(vals))
    a0 = float(grid[best_i])

    step = float(grid[1] - grid[0])
    lo = max(-2.0, a0 - 4.0 * step)
    hi = min(6.0, a0 + 4.0 * step)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = solve_for_alpha(c)[0]
    fd = solve_for_alpha(d)[0]
    for _ in range(80):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = solve_for_alpha(c)[0]
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = solve_for_alpha(d)[0]
    alpha = 0.5 * (lo + hi)
    sse, c1, c2 = solve_for_alpha(alpha)
    pred = disk - c1 + c2 * np.power(x, alpha)
    rmse = math.sqrt(sse / max(1, x.size))
    ss_tot = float(np.dot(y - float(np.mean(y)), y - float(np.mean(y))))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else math.nan
    area_limit = math.pi - c1
    return {
        "c1": c1,
        "c2": c2,
        "c3": alpha,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "n": float(x.size),
        "area_limit": area_limit,
    }


def evaluate_power_fit(G: np.ndarray, params: dict[str, float], variable: str) -> np.ndarray:
    x = np.expm1(G) if variable == "rho" else G
    return params["c1"] + params["c2"] * np.power(x, params["c3"])


def evaluate_disk_defect_fit(G: np.ndarray, params: dict[str, float], variable: str) -> np.ndarray:
    x = np.expm1(G) if variable == "rho" else G
    return math.pi * np.exp(2.0 * G) - params["c1"] + params["c2"] * np.power(x, params["c3"])


def fmt_sig(x: float, sig: int = 3) -> str:
    """Compact significant-figure formatter for plot labels."""
    x = float(x)
    if not math.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    # Matplotlib mathtext understands ordinary decimals well; for very large/small
    # values use a compact mantissa\times10^{k} form.
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.{sig}g}"
    s = f"{x:.{sig-1}e}"
    mant, exp = s.split("e")
    return rf"{float(mant):.{sig}g}\times10^{{{int(exp)}}}"


def signed_math_term(coef: float, body: str, sig: int = 3) -> str:
    sign = "+" if coef >= 0 else "-"
    return f" {sign} {fmt_sig(abs(coef), sig)}{body}"


def variable_math(variable: str) -> str:
    if variable == "rho":
        return r"(e^G-1)"
    return "G"


def fit_legend_label(prefix: str, equation: str, model: str, variable: str, params: dict[str, float]) -> str:
    """Pretty mathtext legend label with fitted values inserted into the equation."""
    ysym = {"area": "A", "length": "L", "points": "N"}.get(prefix, prefix[:1].upper() or "y")
    var = variable_math(variable)
    c1 = params["c1"]
    c2 = params["c2"]
    c3 = params["c3"]

    if model == "disk_defect_power":
        # A(G)=pi e^{2G}-c1+c2*x^{c3}; A(0+)=pi-c1
        sign_c1 = "-" if c1 >= 0 else "+"
        expr = (
            rf"{ysym}(G)=\pi e^{{2G}} {sign_c1} {fmt_sig(abs(c1))}"
            + signed_math_term(c2, rf"{var}^{{{fmt_sig(c3)}}}")
        )
        limit = rf"{ysym}(G\to0^+)={fmt_sig(params['area_limit'])}"
        return "fit: " + rf"${expr}$" + "\n" + rf"${limit}$"

    # y(G)=c1+c2*x^{c3}
    expr = rf"{ysym}(G)={fmt_sig(c1)}" + signed_math_term(c2, rf"{var}^{{{fmt_sig(c3)}}}")
    return "fit: " + rf"${expr}$"


def maybe_add_fit(
    ax,
    G: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, str],
    prefix: str,
    *,
    fit_rows: list[dict[str, str]],
) -> None:
    if not cfg_bool(cfg, f"postprocess_{prefix}_fit", False):
        return
    equation = cfg_clean_str(cfg, f"postprocess_{prefix}_fit_equation", "c1+c2*G^c3")
    model, variable = parse_power_fit_equation(equation)
    mask = select_fit_mask(G, y, cfg, prefix)
    if int(np.sum(mask)) < 4:
        print(f"  warning: {prefix} fit skipped; only {int(np.sum(mask))} points in fit range.", flush=True)
        return

    if model == "disk_defect_power":
        params = fit_disk_defect_power_model(G[mask], y[mask], variable)
    else:
        xfit_data = np.expm1(G[mask]) if variable == "rho" else G[mask]
        params = fit_offset_power_model(xfit_data, y[mask])
    if params is None:
        print(f"  warning: {prefix} fit skipped; fit failed.", flush=True)
        return

    g_min = float(np.nanmin(G[mask]))
    g_max = float(np.nanmax(G[mask]))
    g_line = np.geomspace(g_min, g_max, 400)
    if model == "disk_defect_power":
        y_line = evaluate_disk_defect_fit(g_line, params, variable)
    else:
        y_line = evaluate_power_fit(g_line, params, variable)
    valid = np.isfinite(y_line)
    if not np.any(valid):
        return

    style = cfg_clean_str(cfg, f"postprocess_{prefix}_fit_line_style", "black,thin,dashed")
    label = fit_legend_label(prefix, equation, model, variable, params)
    ax.plot(g_line[valid], y_line[valid], label=label, **fit_line_kwargs(style))
    fit_rows.append({
        "series": prefix,
        "equation": equation,
        "model": model,
        "variable": variable,
        "c1": f"{params['c1']:.17g}",
        "c2": f"{params['c2']:.17g}",
        "c3": f"{params['c3']:.17g}",
        "area_limit": f"{params.get('area_limit', math.nan):.17g}",
        "rmse": f"{params['rmse']:.17g}",
        "r2": f"{params['r2']:.17g}",
        "n": str(int(params["n"])),
        "G_min": f"{g_min:.17g}",
        "G_max": f"{g_max:.17g}",
    })
    if model == "disk_defect_power":
        print(
            f"{prefix} fit: {equation}, n={int(params['n'])}, "
            f"G=[{g_min:.6g}, {g_max:.6g}], "
            f"area_limit=pi-c1={params['area_limit']:.12g}, "
            f"c1(defect)={params['c1']:.12g}, c2={params['c2']:.12g}, "
            f"c3={params['c3']:.6g}, RMSE={params['rmse']:.6g}, R²={params['r2']:.6g}",
            flush=True,
        )
    else:
        print(
            f"{prefix} fit: {equation}, n={int(params['n'])}, "
            f"G=[{g_min:.6g}, {g_max:.6g}], "
            f"c1={params['c1']:.12g}, c2={params['c2']:.12g}, "
            f"c3={params['c3']:.6g}, RMSE={params['rmse']:.6g}, R²={params['r2']:.6g}",
            flush=True,
        )


def write_fit_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys = ["series", "equation", "model", "variable", "c1", "c2", "c3", "area_limit", "rmse", "r2", "n", "G_min", "G_max"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_global_plots(csv_path: Path, output_dir: Path, cfg: dict[str, str]) -> None:
    rows = read_scaling_csv(csv_path)
    if not rows:
        raise SystemExit(f"No usable rows in CSV: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_bool(cfg, "postprocess_write_legacy_loglog_copies", False):
        for legacy_name in [
            "length_vs_G_loglog.png",
            "area_vs_G_loglog.png",
            "points_vs_G_loglog.png",
            "curvature_quantiles_vs_G_loglog.png",
            "curvature_vs_G_loglog.png",
        ]:
            try:
                (output_dir / legacy_name).unlink(missing_ok=True)
            except Exception:
                pass

    G = np.asarray([r.get("target_G", math.nan) for r in rows], dtype=float)
    length = np.asarray([r.get("length", math.nan) for r in rows], dtype=float)
    area = np.asarray([r.get("area", math.nan) for r in rows], dtype=float)
    points = np.asarray([r.get("points", math.nan) for r in rows], dtype=float)

    plot_style = cfg_clean_str(cfg, "postprocess_global_plot_style", "points_lines").lower()
    if plot_style not in {"points", "lines", "points_lines"}:
        raise SystemExit("postprocess_global_plot_style must be points, lines, or points_lines.")
    line_style = global_line_style(cfg_clean_str(cfg, "postprocess_global_plot_line_style", "solid"))
    fit_rows: list[dict[str, str]] = []

    def plot_single(prefix: str, y: np.ndarray, ylabel: str, title: str, filename: str, fit: bool = False) -> None:
        x_scale = cfg_scale(cfg, f"postprocess_{prefix}_x_scale", "log")
        y_scale = cfg_scale(cfg, f"postprocess_{prefix}_y_scale", "log")
        mask = np.isfinite(G) & np.isfinite(y) & (G > 0)
        if y_scale == "log":
            mask &= y > 0
        if not np.any(mask):
            return
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
        plot_xy(ax, G[mask], y[mask], label=None, style=plot_style, marker="o", linestyle=line_style)
        if fit:
            maybe_add_fit(ax, G, y, cfg, prefix, fit_rows=fit_rows)
        apply_global_axes(
            ax,
            G[mask],
            [y[mask]],
            x_scale=x_scale,
            y_scale=y_scale,
            xlabel="target escape potential G",
            ylabel=ylabel,
            title=title,
            cfg=cfg,
        )
        if fit and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)

    plot_single("length", length, "closed contour length", "Mandelbrot equipotential length vs G", "length_vs_G.png", fit=True)
    plot_single("area", area, "enclosed area", "Mandelbrot equipotential area vs G", "area_vs_G.png", fit=True)
    plot_single("points", points, "traced contour points", "Traced point count vs G", "points_vs_G.png", fit=False)

    # Backward-compatible copies of the old names for blog references/scripts.
    # Disabled by default because the configured axes may not actually be log-log.
    if cfg_bool(cfg, "postprocess_write_legacy_loglog_copies", False):
        for old, new in [
            ("length_vs_G_loglog.png", "length_vs_G.png"),
            ("area_vs_G_loglog.png", "area_vs_G.png"),
            ("points_vs_G_loglog.png", "points_vs_G.png"),
        ]:
            src = output_dir / new
            if src.exists():
                try:
                    (output_dir / old).write_bytes(src.read_bytes())
                except Exception:
                    pass

    p90 = np.asarray([r.get("abs_curvature_p90", math.nan) for r in rows], dtype=float)
    p95 = np.asarray([r.get("abs_curvature_p95", math.nan) for r in rows], dtype=float)
    p99 = np.asarray([r.get("abs_curvature_p99", math.nan) for r in rows], dtype=float)
    x_scale = cfg_scale(cfg, "postprocess_curvature_quantiles_x_scale", "log")
    y_scale = cfg_scale(cfg, "postprocess_curvature_quantiles_y_scale", "log")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    any_line = False
    for y, marker, label in [
        (p90, "o", "weighted p90 |κ|"),
        (p95, "s", "weighted p95 |κ|"),
        (p99, "^", "weighted p99 |κ|"),
    ]:
        mask = np.isfinite(G) & np.isfinite(y) & (G > 0)
        if y_scale == "log":
            mask &= y > 0
        if np.any(mask):
            plot_xy(ax, G[mask], y[mask], label=label, style=plot_style, marker=marker, linestyle=line_style)
            any_line = True
    if any_line:
        apply_global_axes(ax, G[np.isfinite(G) & (G > 0)], [p90, p95, p99], x_scale=x_scale, y_scale=y_scale,
                          xlabel="target escape potential G", ylabel="|κ| quantile",
                          title="Weighted absolute-curvature quantiles vs G", cfg=cfg)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "curvature_quantiles_vs_G.png")
        if cfg_bool(cfg, "postprocess_write_legacy_loglog_copies", False):
            try:
                (output_dir / "curvature_quantiles_vs_G_loglog.png").write_bytes((output_dir / "curvature_quantiles_vs_G.png").read_bytes())
            except Exception:
                pass
    plt.close(fig)

    mean_abs = np.asarray([r.get("mean_abs_curvature", math.nan) for r in rows], dtype=float)
    trimmed = np.asarray([r.get("trimmed_mean_abs_curvature_p99", math.nan) for r in rows], dtype=float)
    rms = np.asarray([r.get("rms_curvature", math.nan) for r in rows], dtype=float)
    x_scale = cfg_scale(cfg, "postprocess_curvature_x_scale", "log")
    y_scale = cfg_scale(cfg, "postprocess_curvature_y_scale", "log")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    any_line = False
    for y, marker, linestyle, label in [
        (mean_abs, "o", line_style, "mean |κ|"),
        (trimmed, ".", "--", "mean min(|κ|, p99)"),
        (rms, "s", ":", "RMS κ"),
    ]:
        mask = np.isfinite(G) & np.isfinite(y) & (G > 0)
        if y_scale == "log":
            mask &= y > 0
        if np.any(mask):
            plot_xy(ax, G[mask], y[mask], label=label, style=plot_style, marker=marker, linestyle=linestyle)
            any_line = True
    if any_line:
        apply_global_axes(ax, G[np.isfinite(G) & (G > 0)], [mean_abs, trimmed, rms], x_scale=x_scale, y_scale=y_scale,
                          xlabel="target escape potential G", ylabel="curvature",
                          title="Weighted curvature vs G", cfg=cfg)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "curvature_vs_G.png")
        if cfg_bool(cfg, "postprocess_write_legacy_loglog_copies", False):
            try:
                (output_dir / "curvature_vs_G_loglog.png").write_bytes((output_dir / "curvature_vs_G.png").read_bytes())
            except Exception:
                pass
    plt.close(fig)

    write_fit_summary(output_dir / "fit_summary.csv", fit_rows)


def merged_hist_plot_config(out_dir: Path, cfg: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    meta_path = out_dir / "hist_data" / "hist_config.json"
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            merged.update({str(k).lower(): str(v) for k, v in raw.items()})
        except Exception:
            pass
    merged.update({str(k).lower(): str(v) for k, v in cfg.items()})
    return merged


def hist_cfg_str(cfg: dict[str, str], key: str, default: str) -> str:
    return str(cfg.get(key, default)).strip().lower()


def hist_cfg_float(cfg: dict[str, str], key: str, default: float) -> float:
    raw = cfg.get(key)
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except Exception:
        return default
    return value if math.isfinite(value) else default


def hist_cfg_bool(cfg: dict[str, str], key: str, default: bool) -> bool:
    raw = cfg.get(key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def hist_cfg_optional_float(cfg: dict[str, str], key: str) -> float | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"auto", "none", "nan"}:
        return None
    try:
        value = float(s)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def apply_optional_axis_ranges(ax, cfg: dict[str, str], kind: str, *, xscale: str, yscale: str) -> None:
    xmin = hist_cfg_optional_float(cfg, f"postprocess_{kind}_hist_plot_x_min")
    xmax = hist_cfg_optional_float(cfg, f"postprocess_{kind}_hist_plot_x_max")
    ymin = hist_cfg_optional_float(cfg, f"postprocess_{kind}_hist_plot_y_min")
    ymax = hist_cfg_optional_float(cfg, f"postprocess_{kind}_hist_plot_y_max")

    if xmin is not None or xmax is not None:
        cur_lo, cur_hi = ax.get_xlim()
        lo = cur_lo if xmin is None else xmin
        hi = cur_hi if xmax is None else xmax
        if xscale == "log":
            if lo > 0 and hi > lo:
                ax.set_xlim(lo, hi)
        elif hi > lo:
            ax.set_xlim(lo, hi)

    if ymin is not None or ymax is not None:
        cur_lo, cur_hi = ax.get_ylim()
        lo = cur_lo if ymin is None else ymin
        hi = cur_hi if ymax is None else ymax
        if yscale == "log":
            if lo > 0 and hi > lo:
                ax.set_ylim(lo, hi)
        elif hi > lo:
            ax.set_ylim(lo, hi)


def plot_hist_data_from_csvs(
    out_dir: Path,
    cfg: dict[str, str],
    *,
    only_index: int | None = None,
    hist_kind: str = "both",
    progress_every: int = 10,
) -> None:
    hdir = out_dir / "hist_data"
    if not hdir.exists():
        print(f"No hist_data folder found: {hdir}")
        return

    plot_cfg = merged_hist_plot_config(out_dir, cfg)

    abs_x = hist_cfg_str(
        plot_cfg,
        "postprocess_abs_hist_x_scale",
        hist_cfg_str(plot_cfg, "postprocess_abs_hist_scale", "log"),
    )
    signed_x = hist_cfg_str(
        plot_cfg,
        "postprocess_signed_hist_x_scale",
        hist_cfg_str(plot_cfg, "postprocess_signed_hist_scale", "symlog"),
    )
    abs_y = hist_cfg_str(plot_cfg, "postprocess_abs_hist_y_scale", "log")
    signed_y = hist_cfg_str(plot_cfg, "postprocess_signed_hist_y_scale", "log")

    valid_scales = {"linear", "log", "symlog"}
    if abs_x not in valid_scales:
        raise SystemExit(f"Invalid postprocess_abs_hist_x_scale={abs_x!r}; use linear, log, or symlog.")
    if signed_x not in valid_scales:
        raise SystemExit(f"Invalid postprocess_signed_hist_x_scale={signed_x!r}; use linear, log, or symlog.")
    if abs_y not in valid_scales:
        raise SystemExit(f"Invalid postprocess_abs_hist_y_scale={abs_y!r}; use linear, log, or symlog.")
    if signed_y not in valid_scales:
        raise SystemExit(f"Invalid postprocess_signed_hist_y_scale={signed_y!r}; use linear, log, or symlog.")

    abs_x_symlog_thr = hist_cfg_float(plot_cfg, "postprocess_abs_hist_x_symlog_threshold", 20.0)
    signed_x_symlog_thr = hist_cfg_float(
        plot_cfg,
        "postprocess_signed_hist_x_symlog_threshold",
        hist_cfg_float(plot_cfg, "postprocess_signed_symlog_threshold", 20.0),
    )
    abs_y_symlog_thr = hist_cfg_float(plot_cfg, "postprocess_abs_hist_y_symlog_threshold", 20.0)
    signed_y_symlog_thr = hist_cfg_float(plot_cfg, "postprocess_signed_hist_y_symlog_threshold", 20.0)

    hist_plot_style = hist_cfg_str(
        plot_cfg,
        "postprocess_hist_plot_style",
        hist_cfg_str(plot_cfg, "hist_plot_style", "bin_midpoints"),
    )
    hist_plot_joined = hist_cfg_bool(
        plot_cfg,
        "postprocess_hist_plot_joined",
        hist_cfg_bool(plot_cfg, "hist_plot_joined", True),
    )
    hist_plot_show_points = hist_cfg_bool(
        plot_cfg,
        "postprocess_hist_plot_show_points",
        hist_cfg_bool(plot_cfg, "hist_plot_show_points", True),
    )

    if hist_plot_style not in {"bins", "bin_midpoints"}:
        raise SystemExit(f"Invalid hist_plot_style={hist_plot_style!r}; use bins or bin_midpoints.")
    if hist_plot_style == "bin_midpoints" and not hist_plot_joined and not hist_plot_show_points:
        raise SystemExit(
            "hist_plot_style=bin_midpoints requires at least one of "
            "hist_plot_joined=true or hist_plot_show_points=true."
        )

    pdir = out_dir / "hist_plots"
    pdir.mkdir(parents=True, exist_ok=True)

    def set_axis_scale(ax, axis: str, scale: str, threshold: float, values: np.ndarray) -> str:
        """Set axis scale if possible. Returns the effective scale."""
        if scale == "linear":
            return "linear"

        if scale == "symlog":
            if threshold <= 0 or not math.isfinite(threshold):
                threshold = 20.0
            if axis == "x":
                ax.set_xscale("symlog", linthresh=threshold)
            else:
                ax.set_yscale("symlog", linthresh=threshold)
            return "symlog"

        # Plain log only works for strictly positive plotted coordinates.
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == values[np.isfinite(values)].size and positive.size > 0:
            if axis == "x":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")
            return "log"

        # Graceful fallback: signed/zero-containing data cannot be shown on log.
        print(f"  warning: {axis}-scale=log requested but data include non-positive values; using linear.", flush=True)
        return "linear"

    def plot_one(path: Path, kind: str) -> bool:
        left: list[float] = []
        right: list[float] = []
        dens: list[float] = []

        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                left.append(safe_float(row.get("left")))
                right.append(safe_float(row.get("right")))
                dens.append(safe_float(row.get("density")))

        if not left or len(left) != len(right) or len(left) != len(dens):
            return False

        left_arr = np.asarray(left, dtype=float)
        right_arr = np.asarray(right, dtype=float)
        y = np.asarray(dens, dtype=float)
        edges = np.asarray(left + [right[-1]], dtype=float)

        if not (np.all(np.isfinite(left_arr)) and np.all(np.isfinite(right_arr)) and np.all(np.isfinite(edges))):
            return False

        xlabel = "|κ|" if kind == "abs" else "signed curvature κ"
        xscale_requested = abs_x if kind == "abs" else signed_x
        yscale_requested = abs_y if kind == "abs" else signed_y
        x_symlog_thr = abs_x_symlog_thr if kind == "abs" else signed_x_symlog_thr
        y_symlog_thr = abs_y_symlog_thr if kind == "abs" else signed_y_symlog_thr

        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)

        y_plot = y.copy()
        if yscale_requested == "log":
            positive = finite_positive(y_plot)
            if positive.size == 0:
                plt.close(fig)
                return False
            y_plot = np.where(y_plot > 0, y_plot, np.nan)

        if hist_plot_style == "bins":
            widths = right_arr - left_arr
            ax.bar(
                left_arr,
                y_plot,
                width=widths,
                align="edge",
                linewidth=0.7,
                edgecolor="black",
                facecolor="#4c78a8",
                alpha=0.25,
            )
            x_values_for_scale = edges
        else:
            centers = 0.5 * (left_arr + right_arr)
            if hist_plot_joined:
                ax.plot(centers, y_plot, linewidth=1.2)
            if hist_plot_show_points:
                ax.plot(centers, y_plot, linestyle="None", marker="o", markersize=2.3)
            x_values_for_scale = centers

        xscale_effective = set_axis_scale(ax, "x", xscale_requested, x_symlog_thr, x_values_for_scale)
        yscale_effective = set_axis_scale(ax, "y", yscale_requested, y_symlog_thr, y_plot)

        # For log y, set a sane positive lower bound before optional user ranges.
        if yscale_effective == "log":
            positive = finite_positive(y_plot)
            if positive.size:
                ax.set_ylim(bottom=float(np.nanmin(positive)) * 0.5)

        apply_optional_axis_ranges(
            ax,
            plot_cfg,
            kind,
            xscale=xscale_effective,
            yscale=yscale_effective,
        )

        stem = path.stem.replace("_abs", "").replace("_signed", "")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("probability density")
        ax.set_title(
            f"{kind} curvature PDF, contour {stem}\n"
            f"x={xscale_effective}, y={yscale_effective}, style={hist_plot_style}"
        )
        ax.grid(True, which="both", alpha=0.35)
        fig.tight_layout()
        fig.savefig(pdir / f"{stem}_curvature_{kind}_hist.png")
        plt.close(fig)
        return True

    if only_index is None:
        abs_paths = sorted(hdir.glob("*_abs.csv"))
        signed_paths = sorted(hdir.glob("*_signed.csv"))
    else:
        stem = f"{only_index:05d}"
        abs_paths = [hdir / f"{stem}_abs.csv"]
        signed_paths = [hdir / f"{stem}_signed.csv"]

    jobs: list[tuple[Path, str]] = []
    if hist_kind in {"both", "abs"}:
        jobs.extend((p, "abs") for p in abs_paths if p.exists())
    if hist_kind in {"both", "signed"}:
        jobs.extend((p, "signed") for p in signed_paths if p.exists())

    total = len(jobs)
    if total == 0:
        print(f"No histogram CSVs to plot in {hdir}")
        return

    print(f"Histogram plots: {total} file(s) -> {pdir}")
    written = 0
    for i, (path, kind) in enumerate(jobs, start=1):
        if progress_every > 0 and (i == 1 or i == total or i % progress_every == 0):
            print(f"  [{i}/{total}] plotting {path.name}", flush=True)
        if plot_one(path, kind):
            written += 1

    print(f"Histogram plots done: {written}/{total} written to {pdir}")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot derived C++ Mandelbrot contour results. This script does not "
            "read raw contours. Run ./bin/postprocess_contours first."
        )
    )
    add_config_argument(parser)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override contours.output_dir from the unified repository config.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Override contour_scaling.csv; default: <output-dir>/contour_scaling.csv.",
    )
    parser.add_argument("--no-global-plots", action="store_true")
    parser.add_argument("--hist-plots", action="store_true", help="Deprecated/no-op.")
    parser.add_argument("--no-hist-plots", action="store_true")
    parser.add_argument("--hist-index", type=int, default=None)
    parser.add_argument("--hist-kind", choices=["both", "abs", "signed"], default="both")
    parser.add_argument("--hist-progress-every", type=int, default=10)
    return parser.parse_args(argv)


def _flat_contour_config(repo: RepoConfig) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in repo.section("contours").items():
        if isinstance(value, bool):
            result[key] = "true" if value else "false"
        elif isinstance(value, list):
            result[key] = ",".join(str(item) for item in value)
        elif value is None:
            result[key] = "auto"
        else:
            result[key] = str(value)
    return result


def main() -> None:
    args = parse_args()
    repo = RepoConfig.load(args.config, start=SCRIPT_DIR)
    cfg = _flat_contour_config(repo)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = repo.paths.code_root / out_dir
        out_dir = out_dir.resolve()
    else:
        out_dir = repo.path("contours.output_dir")
    csv_path = Path(args.csv).expanduser().resolve() if args.csv is not None else out_dir / "contour_scaling.csv"

    if not out_dir.exists():
        raise SystemExit(
            f"Output folder does not exist: {out_dir}\n"
            "Run ./bin/contours and ./bin/postprocess_contours first, or use --output-dir."
        )

    if not csv_path.exists() and not args.no_global_plots:
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run ./bin/postprocess_contours, then rerun this plotter."
        )

    if not args.no_global_plots:
        plots_dir = out_dir / "scaling_plots"
        make_global_plots(csv_path, plots_dir, cfg)
        print(f"Global plots: {plots_dir}")

    if not args.no_hist_plots:
        plot_hist_data_from_csvs(
            out_dir,
            cfg,
            only_index=args.hist_index,
            hist_kind=args.hist_kind,
            progress_every=args.hist_progress_every,
        )


if __name__ == "__main__":
    main()
