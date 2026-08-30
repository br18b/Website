#!/usr/bin/env python3
"""
Generate a small real-quadratic-iteration report for

    x_{n+1} = x_n^2 + C,     x_0 = 0

Outputs:
  1. A Markdown table with selected C values and selected iteration columns.
  2. A density/heatmap plot of x_n as a function of C and n.
  3. A line plot of x_n(C) for selected iterations.

The table is meant to be pasted directly into a Jekyll/Markdown blog post.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# User-facing knobs
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("fractal/2D")

# Table columns: x_1, x_5, x_20, ...
N_ITER_TABLE = [1, 5, 10, 20, 30, 31]

# C values used in the table.
TABLE_C_VALUES = [-2.1, -2.01, -2.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.25, 0.26, 0.3]

# In the Markdown table, values with |x| > 10^N are displayed as infinity.
TABLE_INFINITY_LOG10 = 100

# Density plot range and resolution.
C_MIN = -2.3
C_MAX = 0.55
DC = 0.0025
MAX_ITER_DENSITY = 50

# Line plot iterations.
N_ITER_LINES = [1, 2, 200, 201, 202]

# Values outside this range are visually clipped in the density plot.
# Without clipping, the divergent region dominates the color scale immediately.
DENSITY_VALUE_CLIP = 3.0

# Values outside this range are hidden in the line plot.
LINE_VALUE_CLIP = 3.0

# Once |x_n| is above this, we stop trying to track the raw value for plots.
# For the table, large values are tracked using log10(|x|).
RUNAWAY_THRESHOLD = 1.1e10

# Convergence diagnostics use iterations 20 through 50, as requested.
CONVERGENCE_CHECK_START = 20
CONVERGENCE_CHECK_END = 50

# Divergence classification.
# If |x_n| exceeds RUNAWAY_THRESHOLD by this iteration, call it quick.
DIVERGES_QUICKLY_BY_ITER = 20

# How far to look when deciding slow/quick divergence.
DIVERGENCE_CHECK_END = 80

# Numerical tolerance for recognizing values like -2.0 or 0.25.
EDGE_TOL = 1.0e-12


# ---------------------------------------------------------------------------
# Iteration and formatting helpers
# ---------------------------------------------------------------------------

@dataclass
class IterValue:
    """
    Represents either a normal finite value, or a huge value tracked by log10.

    If log10_abs is not None, the represented value is approximately

        sign * 10^log10_abs

    This avoids Python float overflow for values like 10^85960.
    """

    value: float | None = None
    log10_abs: float | None = None
    sign: int = 1

    @property
    def is_huge(self) -> bool:
        return self.log10_abs is not None


def iterate_values_for_table(c: float, max_n: int) -> list[IterValue]:
    """
    Compute x_0, x_1, ..., x_max_n.

    The normal finite value is used while possible. Once the orbit is huge,
    we switch to log10 tracking:

        log10(x_{n+1}) ~= 2 log10(|x_n|)

    because x_n^2 dominates the + C term.
    """

    values: list[IterValue] = [IterValue(value=0.0)]
    x = 0.0
    log10_abs: float | None = None

    for _ in range(max_n):
        if log10_abs is None:
            x_next = x * x + c

            if not math.isfinite(x_next):
                # Emergency fallback if overflow somehow happened.
                log10_abs = 2.0 * math.log10(abs(x))
                values.append(IterValue(log10_abs=log10_abs, sign=1))
                continue

            if abs(x_next) >= RUNAWAY_THRESHOLD:
                log10_abs = math.log10(abs(x_next))
                sign = 1 if x_next >= 0 else -1
                values.append(IterValue(log10_abs=log10_abs, sign=sign))
            else:
                x = x_next
                values.append(IterValue(value=x))
        else:
            # Once huge, the next value is positive and approximately x^2.
            log10_abs = 2.0 * log10_abs
            values.append(IterValue(log10_abs=log10_abs, sign=1))

    return values


def format_plain_3sig(x: float) -> str:
    """
    Format finite numbers with roughly 3 significant figures.

    Examples:
        0       -> 0
        40.234  -> 40.2
        13.432  -> 13.4
        -5.789  -> -5.79
        0.3881  -> 0.388
        420.1   -> 420

    It intentionally avoids "0.".
    """

    if abs(x) < 1.0e-15:
        return "0"

    s = f"{x:.3g}"

    # Python may emit scientific notation for small values; keep that only
    # for genuinely tiny values. The requested normal range was mostly
    # about values between -999 and 999.
    if "e" in s or "E" in s:
        return s.replace("e", r" \times 10^{").replace("+", "") + "}"

    return s


def format_sci_from_log10(log10_abs: float, sign: int = 1) -> str:
    """
    Return a Markdown/LaTeX-ish scientific notation string with 3 sig figs.

    Example:
        5.5 \times 10^{85960}

    For extremely large exponents the mantissa becomes unreliable because
    normal floats cannot preserve the fractional part of log10_abs. In that
    case, we deliberately fall back to 10^{N}.
    """

    if not math.isfinite(log10_abs):
        return r"\infty"

    exponent = math.floor(log10_abs)

    # Above this, the fractional part of a float log is not trustworthy.
    if abs(log10_abs) > 1.0e12:
        prefix = "-" if sign < 0 else ""
        return rf"{prefix}10^{{{exponent}}}"

    mantissa = 10.0 ** (log10_abs - exponent)

    # Rounding can produce 10.0.
    mantissa_rounded = float(f"{mantissa:.3g}")
    if mantissa_rounded >= 10.0:
        mantissa /= 10.0
        exponent += 1

    mantissa_s = f"{mantissa:.3g}"
    prefix = "-" if sign < 0 else ""
    return rf"{prefix}{mantissa_s} \times 10^{{{exponent}}}"


def format_iter_value(v: IterValue) -> str:
    """
    Format an IterValue as Markdown math:
      - 0 is written as 0
      - moderate numbers are written plainly
      - large-but-readable numbers use scientific notation
      - absurdly large numbers become infinity
    """

    if v.is_huge:
        if (v.log10_abs or 0.0) > TABLE_INFINITY_LOG10:
            return r"$\infty$"

        return "$" + format_sci_from_log10(v.log10_abs or 0.0, v.sign) + "$"

    assert v.value is not None
    x = v.value

    if abs(x) < 1.0e-15:
        return "$0$"

    # Extra safety in case a finite float larger than 10^100 appears.
    if abs(x) > 10.0 ** TABLE_INFINITY_LOG10:
        return r"$\infty$"

    if abs(x) < 999:
        return "$" + format_plain_3sig(x) + "$"

    log10_abs = math.log10(abs(x))

    if log10_abs > TABLE_INFINITY_LOG10:
        return r"$\infty$"

    sign = 1 if x >= 0 else -1
    return "$" + format_sci_from_log10(log10_abs, sign) + "$"


# ---------------------------------------------------------------------------
# Classification helper
# ---------------------------------------------------------------------------

def lower_fixed_point(c: float) -> float | None:
    """
    Lower fixed point of x = x^2 + C:

        x = (1 - sqrt(1 - 4C)) / 2

    It exists only for C <= 1/4.
    """

    disc = 1.0 - 4.0 * c
    if disc < 0:
        return None
    return (1.0 - math.sqrt(disc)) / 2.0


def finite_orbit(c: float, max_n: int) -> np.ndarray:
    """Return x_0, ..., x_max_n as normal floats, stopping at huge values."""
    xs = np.empty(max_n + 1, dtype=float)
    xs[0] = 0.0
    x = 0.0

    for n in range(1, max_n + 1):
        x = x * x + c
        xs[n] = x
        if not math.isfinite(x) or abs(x) > RUNAWAY_THRESHOLD:
            xs[n:] = np.nan
            break

    return xs


def divergence_iteration(c: float, max_n: int = DIVERGENCE_CHECK_END) -> int | None:
    """
    Return the first iteration n where |x_n| exceeds RUNAWAY_THRESHOLD.

    Returns None if it does not exceed the threshold by max_n.
    Uses log tracking after the value gets huge, so it does not overflow.
    """

    x = 0.0
    log10_abs: float | None = None
    threshold_log10 = math.log10(RUNAWAY_THRESHOLD)

    for n in range(1, max_n + 1):
        if log10_abs is None:
            x_next = x * x + c

            if not math.isfinite(x_next):
                return n

            if abs(x_next) >= RUNAWAY_THRESHOLD:
                return n

            x = x_next
        else:
            log10_abs = 2.0 * log10_abs
            if log10_abs >= threshold_log10:
                return n

    return None


def classify_divergence_speed(c: float) -> str:
    """
    Qualitative label for divergent orbits.
    """

    n_escape = divergence_iteration(c)

    if n_escape is None:
        # This can happen very close to the boundary. Since we already know
        # from the C-range that it diverges, call it slow.
        return "diverges slowly"

    if n_escape <= DIVERGES_QUICKLY_BY_ITER:
        return "diverges quickly"

    return "diverges slowly"


def classify_c(c: float) -> str:
    """
    Classify the orbit x_{n+1} = x_n^2 + C, x_0 = 0.

    The broad real facts are:

        bounded:       -2 <= C <= 1/4
        fixed-point convergence interval: -3/4 <= C <= 1/4
        runaway:       C < -2 or C > 1/4

    with special edge behavior at C = -2.
    """

    if c < -2.0 - EDGE_TOL or c > 0.25 + EDGE_TOL:
        return classify_divergence_speed(c)

    if abs(c + 2.0) <= EDGE_TOL:
        return "converges quickly"

    if -0.75 - EDGE_TOL <= c <= 0.25 + EDGE_TOL:
        fixed = lower_fixed_point(c)
        if fixed is None:
            return classify_divergence_speed(c)

        xs = finite_orbit(c, CONVERGENCE_CHECK_END)
        window = xs[CONVERGENCE_CHECK_START:CONVERGENCE_CHECK_END + 1]

        if np.any(~np.isfinite(window)):
            return classify_divergence_speed(c)

        err20 = abs(xs[CONVERGENCE_CHECK_START] - fixed)
        err50 = abs(xs[CONVERGENCE_CHECK_END] - fixed)

        # This is intentionally qualitative, not a theorem prover.
        if err20 < 1.0e-5:
            return "converges quickly"
        if err50 < 1.0e-5:
            return "converges"
        if err50 < err20:
            return "converges slowly"
        return "converges"

    # The remaining bounded real interval is [-2, -3/4).
    # It is bounded but generally not convergent to one value.
    if -2.0 <= c < -0.75:
        if abs(c + 1.0) <= EDGE_TOL:
            return r"bounded, oscillatory"
        return "bounded, oscillatory"

    return "bounded, oscillatory"


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def make_markdown_table() -> str:
    max_n = max(N_ITER_TABLE)

    headers = ["$C$"] + [rf"$x_{{{n}}}$" for n in N_ITER_TABLE] + ["What it is doing"]
    align = ["---:"] * (1 + len(N_ITER_TABLE)) + [":---"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for c in TABLE_C_VALUES:
        values = iterate_values_for_table(c, max_n)

        row = [format_iter_value(IterValue(value=c))]
        row.extend(format_iter_value(values[n]) for n in N_ITER_TABLE)
        row.append(classify_c(c))

        lines.append("| " + " | ".join(row) + " |")

    lines.append("{: .centered-table }")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def compute_plot_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Return:
        c_values
        density_iterations
        heatmap_values[iteration_index, c_index]
        selected_line_values[n] = x_n(c_values)

    Important detail:
        MAX_ITER_DENSITY controls only the density plot height.
        N_ITER_LINES may request larger n values, e.g. [1, 2, 5, 50, 51, 52].
        Therefore we compute the orbit up to the larger of those two needs, while
        only storing heatmap rows up to MAX_ITER_DENSITY.
    """

    c_values = np.arange(C_MIN, C_MAX + 0.5 * DC, DC)

    density_iterations = np.arange(1, MAX_ITER_DENSITY + 1)
    max_line_iter = max(N_ITER_LINES) if N_ITER_LINES else 0
    max_iter_needed = max(MAX_ITER_DENSITY, max_line_iter)

    x = np.zeros_like(c_values, dtype=float)
    escaped = np.zeros_like(c_values, dtype=bool)

    heat_rows: list[np.ndarray] = []
    selected_lines: dict[int, np.ndarray] = {}

    selected_set = set(N_ITER_LINES)

    for n in range(1, max_iter_needed + 1):
        active = ~escaped

        x_next = np.full_like(x, np.nan)
        x_next[active] = x[active] * x[active] + c_values[active]

        newly_escaped = (
            active
            & (
                ~np.isfinite(x_next)
                | (np.abs(x_next) > RUNAWAY_THRESHOLD)
            )
        )
        escaped = escaped | newly_escaped

        if n <= MAX_ITER_DENSITY:
            heat_row = np.clip(x_next, -DENSITY_VALUE_CLIP, DENSITY_VALUE_CLIP)

            # Once escaped, the orbit is visually marked as saturated high.
            heat_row[escaped] = DENSITY_VALUE_CLIP
            heat_rows.append(heat_row)

        if n in selected_set:
            line = x_next.copy()
            line[escaped] = np.nan
            line[np.abs(line) > LINE_VALUE_CLIP] = np.nan
            selected_lines[int(n)] = line

        x = x_next
        x[escaped] = np.nan

    heatmap = np.vstack(heat_rows)
    return c_values, density_iterations, heatmap, selected_lines


def add_reference_lines(ax) -> None:
    """
    Add the important C-values:
        -2      bounded/runaway boundary
        -3/4    fixed-point convergence boundary
         1/4    bounded/runaway boundary
    """

    for x in [-2.0, -0.75, 0.25]:
        ax.axvline(x, ymin=0.0, ymax=0.935, color="black", linestyle="--", linewidth=1)

    ax.text(-2.0, 0.98, "$-2$", transform=ax.get_xaxis_transform(), ha="center", va="top", color="black")
    ax.text(-0.75, 0.98, "$-3/4$", transform=ax.get_xaxis_transform(), ha="center", va="top", color="black")
    ax.text(0.25, 0.98, "$1/4$", transform=ax.get_xaxis_transform(), ha="center", va="top", color="black")


def make_density_plot(c_values: np.ndarray, iterations: np.ndarray, heatmap: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    im = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[c_values[0], c_values[-1], iterations[0], iterations[-1]],
        interpolation="bicubic",
    )

    add_reference_lines(ax)

    ax.set_title(r"Iteration of $x_{n+1}=x_n^2+C$, $x_0=0$")
    ax.set_xlabel(r"constant $C$")
    ax.set_ylabel(r"iteration $n$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$x_n$, clipped for display")

    fig.tight_layout()

    path = OUTPUT_DIR / "real_quadratic_density.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_line_plot(c_values: np.ndarray, selected_lines: dict[int, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for n in N_ITER_LINES:
        y = selected_lines[n]
        ax.plot(c_values, y, linewidth=1.6, label=rf"$x_{{{n}}}$")

    add_reference_lines(ax)

    ax.set_title(r"Selected iterates $x_n(C)$ for $x_{n+1}=x_n^2+C$")
    ax.set_xlabel(r"constant $C$")
    ax.set_ylabel(r"$x_n$")
    ax.set_ylim(-LINE_VALUE_CLIP, LINE_VALUE_CLIP)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    path = OUTPUT_DIR / "real_quadratic_iterates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = make_markdown_table()
    table_path = OUTPUT_DIR / "real_quadratic_table.md"
    table_path.write_text(table + "\n", encoding="utf-8")

    c_values, iterations, heatmap, selected_lines = compute_plot_grid()
    density_path = make_density_plot(c_values, iterations, heatmap)
    line_path = make_line_plot(c_values, selected_lines)

    print()
    print("Markdown table:")
    print()
    print(table)
    print()
    print(f"Wrote: {table_path}")
    print(f"Wrote: {density_path}")
    print(f"Wrote: {line_path}")
    print()


if __name__ == "__main__":
    main()
