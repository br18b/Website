"""
Small Mandelbrot utilities: escape potential, derivative-based geometry,
circle-based sanity checks, and normal-flow integration.

The derivative-based tools are intended for exterior points where the orbit
escapes. They use derivatives of the iteration with respect to the parameter
c = x + i y, not finite differences.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

from scipy.optimize import brentq

POTENTIAL_EPSILON_DEFAULT = 1e-12
DERIVATIVE_EPSILON_DEFAULT = 1e-12
ROOT_EPSILON_DEFAULT = 1e-12
MAX_ITER_DEFAULT = 2000
DERIVATIVE_MAX_ITER_DEFAULT = 5000
BAILOUT_DEFAULT = 2.0

Point = tuple[float, float]
Jet2 = tuple[float, float, float, float, float, float]
# Jet2 = F, Fx, Fy, Fxx, Fxy, Fyy

ScalarFunction = Callable[[float, float], float]
DerivativeFunction = Callable[[float, float], Jet2]
NormalPath = list[tuple[float, float, float]]
ContourPath = list[tuple[float, float, float]]


def _is_finite_number(value: float) -> bool:
    return math.isfinite(value)


def _is_finite_point(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)


def _is_finite_jet(jet: Jet2) -> bool:
    return all(math.isfinite(v) for v in jet)


def _append_path_point(
    path: NormalPath,
    x: float,
    y: float,
    F_val: float,
    tol: float = 1e-14,
) -> None:
    """Append a path point unless it duplicates the previous sample."""

    if not path:
        path.append((x, y, F_val))
        return

    x_prev, y_prev, F_prev = path[-1]

    if (
        abs(x - x_prev) > tol
        or abs(y - y_prev) > tol
        or abs(F_val - F_prev) > tol
    ):
        path.append((x, y, F_val))


def in_main_cardioid_or_period2_bulb(x: float, y: float) -> bool:
    """Fast exact-ish interior tests for the quadratic Mandelbrot set."""

    q = (x - 0.25) * (x - 0.25) + y * y

    if q * (q + x - 0.25) <= 0.25 * y * y:
        return True

    return (x + 1.0) * (x + 1.0) + y * y <= 0.0625


def _scale_for_iter(power: int, n: int) -> float:
    """Return power**(-n), using ldexp for the quadratic case."""

    if power == 2:
        return math.ldexp(1.0, -n)

    exponent = -n * math.log(power)

    if exponent < -745.0:
        return 0.0

    return math.exp(exponent)


def _iterate_power_with_derivatives(
    z: complex,
    dz: complex,
    d2z: complex,
    c: complex,
    power: int,
) -> tuple[complex, complex, complex]:
    """
    Advance z -> z**power + c and its first/second c-derivatives.

    dz  = dz_n/dc
    d2z = d²z_n/dc²

    For power == 2 we special-case the formula to avoid the 0**0 corner.
    """

    if power == 2:
        z_next = z * z + c
        dz_next = 2.0 * z * dz + 1.0
        d2z_next = 2.0 * dz * dz + 2.0 * z * d2z
        return z_next, dz_next, d2z_next

    z_power_minus_1 = z ** (power - 1)
    z_power_minus_2 = z ** (power - 2)

    z_next = z_power_minus_1 * z + c
    dz_next = power * z_power_minus_1 * dz + 1.0
    d2z_next = power * (
        (power - 1) * z_power_minus_2 * dz * dz
        + z_power_minus_1 * d2z
    )

    return z_next, dz_next, d2z_next


def mandelbrot_potential(
    x: float,
    y: float,
    epsilon: float = POTENTIAL_EPSILON_DEFAULT,
    max_iter: int = MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
    use_quadratic_interior_tests: bool = True,
) -> float:
    """
    Approximate the exterior escape potential G(c).

    For power == 2, optional main-cardioid and period-2 bulb tests return 0
    immediately for known interior points.

    Returning 0 after max_iter means unresolved/interior, not a proof of
    interior membership near the boundary.
    """

    if power < 2:
        raise ValueError("power must be >= 2")

    if bailout <= 1.0:
        raise ValueError("bailout must be > 1")

    if use_quadratic_interior_tests and power == 2:
        if in_main_cardioid_or_period2_bulb(x, y):
            return 0.0

    c = complex(x, y)
    z = 0.0 + 0.0j
    g_prev: float | None = None

    for n in range(1, max_iter + 1):
        try:
            z = z ** power + c
        except OverflowError:
            if g_prev is not None:
                return g_prev
            return float("inf")

        absz = abs(z)

        if absz > bailout:
            scale = _scale_for_iter(power, n)
            g = scale * math.log(absz)

            if g_prev is not None:
                if abs(g - g_prev) <= epsilon * max(abs(g), 1e-300):
                    return g

            g_prev = g

    if g_prev is not None:
        warnings.warn(
            f"Potential did not converge to epsilon={epsilon} within "
            f"max_iter={max_iter} at c=({x}, {y}); returning last estimate."
        )
        return g_prev

    return 0.0


def mandelbrot_potential_derivatives(
    x: float,
    y: float,
    epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
) -> Jet2:
    """
    Return G, Gx, Gy, Gxx, Gxy, Gyy for the Mandelbrot escape potential.

    Valid for exterior points where the orbit escapes. Derivatives are obtained
    by differentiating the recurrence, not by finite differences.
    """

    if power < 2:
        raise ValueError("power must be >= 2")

    if bailout <= 1.0:
        raise ValueError("bailout must be > 1")

    c = complex(x, y)

    z = 0.0 + 0.0j
    dz = 0.0 + 0.0j
    d2z = 0.0 + 0.0j

    escaped = False
    prev: Jet2 | None = None

    for n in range(1, max_iter + 1):
        try:
            z, dz, d2z = _iterate_power_with_derivatives(z, dz, d2z, c, power)
        except OverflowError:
            if prev is not None:
                return prev
            raise

        absz = abs(z)

        if absz > bailout:
            escaped = True
            scale = _scale_for_iter(power, n)

            A = dz / z
            B = d2z / z - A * A

            G = scale * math.log(absz)

            Gx = scale * A.real
            Gy = -scale * A.imag

            Gxx = scale * B.real
            Gxy = -scale * B.imag
            Gyy = -scale * B.real

            current: Jet2 = (G, Gx, Gy, Gxx, Gxy, Gyy)

            # Once the orbit has escaped, |z| grows double-exponentially.
            # If we demand too much convergence, z/dz/d2z may overflow before
            # the finite jet changes by less than epsilon. In that case, the
            # last finite jet is much better than returning inf/nan.
            if not _is_finite_jet(current):
                if prev is not None:
                    return prev

                raise FloatingPointError(
                    f"Non-finite derivative jet before any finite estimate at ({x}, {y})."
                )

            if prev is not None:
                err = max(abs(a - b) for a, b in zip(current, prev))
                scale_err = max(max(abs(v) for v in current), 1.0)

                if math.isfinite(err) and err <= epsilon * scale_err:
                    return current

            prev = current

    if escaped and prev is not None:
        warnings.warn(
            f"Derivative potential did not fully converge at ({x}, {y}); "
            f"returning last estimate."
        )
        return prev

    raise ValueError(
        f"Point did not escape within max_iter={max_iter}; "
        f"derivatives of exterior potential unavailable at ({x}, {y})."
    )


def mandelbrot_derivative_function(
    epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
) -> DerivativeFunction:
    """Return a reusable derivative function D(x, y) -> G, Gx, Gy, Gxx, Gxy, Gyy."""

    def D(x: float, y: float) -> Jet2:
        return mandelbrot_potential_derivatives(
            x=x,
            y=y,
            epsilon=epsilon,
            max_iter=max_iter,
            power=power,
            bailout=bailout,
        )

    return D


def find_root_2d(
    F: ScalarFunction,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    val: float = 0.0,
    epsilon: float = ROOT_EPSILON_DEFAULT,
) -> Point:
    """Find where a line segment crosses F(x, y) = val."""

    def f(t: float) -> float:
        x = x1 * (1.0 - t) + x2 * t
        y = y1 * (1.0 - t) + y2 * t
        return F(x, y) - val

    f0 = f(0.0)
    f1 = f(1.0)

    if f0 == 0.0:
        return x1, y1

    if f1 == 0.0:
        return x2, y2

    if f0 * f1 > 0.0:
        raise ValueError(f"Root is not bracketed: F(p1)-val={f0}, F(p2)-val={f1}")

    t0 = brentq(f, 0.0, 1.0, xtol=epsilon, rtol=epsilon)

    return (
        x1 * (1.0 - t0) + x2 * t0,
        y1 * (1.0 - t0) + y2 * t0,
    )


# Backwards-compatible alias for your earlier name.
find_root_2D = find_root_2d


def _circle_point(x: float, y: float, r: float, theta: float) -> Point:
    return x + r * math.cos(theta), y + r * math.sin(theta)


def _find_circle_crossing_intervals(
    F: ScalarFunction,
    x: float,
    y: float,
    r: float,
    F0: float,
    sample_N: int,
    phase: float = 0.12345,
) -> list[tuple[float, float]]:
    """Return angle intervals where F - F0 changes sign around a circle."""

    if sample_N < 3:
        raise ValueError("sample_N must be >= 3")

    def theta_of(n: int) -> float:
        return 2.0 * math.pi * (n + phase) / sample_N

    def ftheta(theta: float) -> float:
        px, py = _circle_point(x, y, r, theta)
        return F(px, py) - F0

    values = [ftheta(theta_of(n)) for n in range(sample_N)]
    intervals: list[tuple[float, float]] = []

    for n in range(sample_N):
        n_next = (n + 1) % sample_N
        v0 = values[n]
        v1 = values[n_next]

        if v0 == 0.0:
            theta = theta_of(n)
            intervals.append((theta, theta))
            continue

        if v0 * v1 < 0.0:
            theta0 = theta_of(n)

            if n == sample_N - 1:
                theta1 = theta_of(0) + 2.0 * math.pi
            else:
                theta1 = theta_of(n + 1)

            intervals.append((theta0, theta1))

    return intervals


def _refine_circle_crossing(
    F: ScalarFunction,
    x: float,
    y: float,
    r: float,
    F0: float,
    theta0: float,
    theta1: float,
    epsilon: float,
) -> Point:
    if theta0 == theta1:
        return _circle_point(x, y, r, theta0)

    def ftheta(theta: float) -> float:
        px, py = _circle_point(x, y, r, theta)
        return F(px, py) - F0

    theta = brentq(ftheta, theta0, theta1, xtol=epsilon, rtol=epsilon)
    return _circle_point(x, y, r, theta)


def _normalize(vx: float, vy: float) -> Point:
    length = math.hypot(vx, vy)

    if length == 0.0:
        raise ValueError("Cannot normalize zero vector.")

    return vx / length, vy / length


def _circumcenter(p0: Point, p1: Point, p2: Point) -> Point | None:
    """Stable circumcenter of three points, or None if numerically degenerate."""

    x0, y0 = p0

    ax = p1[0] - x0
    ay = p1[1] - y0
    bx = p2[0] - x0
    by = p2[1] - y0

    d = 2.0 * (ax * by - ay * bx)

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    local_scale = max(math.sqrt(a2), math.sqrt(b2), 1e-300)

    if abs(d) < 1e-12 * local_scale * local_scale:
        return None

    ux_local = (a2 * by - b2 * ay) / d
    uy_local = (b2 * ax - a2 * bx) / d

    return x0 + ux_local, y0 + uy_local


def _normal_curvature_at_radius(
    F: ScalarFunction,
    x: float,
    y: float,
    F0: float,
    r: float,
    sample_N: int,
    epsilon: float,
    max_sample_doublings: int = 10,
) -> tuple[Point, float, int]:
    """Estimate normal/curvature using one fixed circle radius."""

    p0 = (x, y)
    N = sample_N

    for _ in range(max_sample_doublings + 1):
        intervals = _find_circle_crossing_intervals(
            F=F,
            x=x,
            y=y,
            r=r,
            F0=F0,
            sample_N=N,
        )

        crossings = len(intervals)

        if crossings == 2:
            p1 = _refine_circle_crossing(F, x, y, r, F0, *intervals[0], epsilon)
            p2 = _refine_circle_crossing(F, x, y, r, F0, *intervals[1], epsilon)

            tx, ty = _normalize(p2[0] - p1[0], p2[1] - p1[1])
            nx, ny = -ty, tx

            probe = 0.25 * r
            F_plus = F(x + probe * nx, y + probe * ny)
            F_minus = F(x - probe * nx, y - probe * ny)

            if F_minus > F_plus:
                nx, ny = -nx, -ny

            center = _circumcenter(p0, p1, p2)

            if center is None:
                raise ValueError(f"Curvature fit became numerically degenerate at r={r}")

            cx, cy = center
            radius = math.hypot(cx - x, cy - y)

            if radius == 0.0:
                raise ValueError(f"Curvature radius became zero at r={r}")

            curvature_mag = 1.0 / radius
            to_center_x = (cx - x) / radius
            to_center_y = (cy - y) / radius

            sign = 1.0 if (to_center_x * nx + to_center_y * ny) >= 0.0 else -1.0
            curvature = sign * curvature_mag

            return (nx, ny), curvature, N

        if crossings > 2:
            raise ValueError(
                f"Radius too large / geometry too complicated: r={r}, crossings={crossings}"
            )

        N *= 2

    raise ValueError(
        f"Could not find clean 2-crossing case at r={r}; "
        f"last sample_N={N}, crossings={crossings}"
    )


def find_normal_curvature_circle(
    F: ScalarFunction,
    x: float,
    y: float,
    d0: float = 1e-4,
    sample_N: int = 16,
    epsilon: float = ROOT_EPSILON_DEFAULT,
    max_attempts: int = 30,
    refinement_levels: int = 5,
    normal_tol: float = 1e-4,
    curvature_atol: float = 1e-4,
    curvature_rtol: float = 1e-2,
    require_convergence: bool = False,
) -> tuple[Point, float]:
    """
    Generic circle-based normal/curvature estimate for a level set F = F(x, y).

    This is useful as a sanity check for arbitrary scalar fields, but it is much
    slower and less accurate than derivative-based Mandelbrot geometry.
    """

    if sample_N < 3:
        raise ValueError("sample_N must be >= 3")

    if refinement_levels < 1:
        raise ValueError("refinement_levels must be >= 1")

    F0 = F(x, y)
    r = d0
    N = sample_N
    estimates: list[tuple[float, Point, float]] = []
    attempts = 0

    while attempts < max_attempts and len(estimates) < refinement_levels:
        attempts += 1

        try:
            normal, curvature, N_used = _normal_curvature_at_radius(
                F=F,
                x=x,
                y=y,
                F0=F0,
                r=r,
                sample_N=N,
                epsilon=epsilon,
            )
        except ValueError:
            r *= 0.5
            continue

        N = max(sample_N, N_used)
        nx, ny = normal

        if estimates:
            _, prev_normal, prev_curvature = estimates[-1]
            pnx, pny = prev_normal

            if nx * pnx + ny * pny < 0.0:
                nx, ny = -nx, -ny
                curvature = -curvature

            normal_err = max(abs(nx - pnx), abs(ny - pny))
            curvature_err = abs(curvature - prev_curvature)
            curvature_scale = max(abs(curvature), abs(prev_curvature), 1.0)
            curvature_ok = curvature_err <= curvature_atol + curvature_rtol * curvature_scale
            normal_ok = normal_err <= normal_tol

            estimates.append((r, (nx, ny), curvature))

            if normal_ok and curvature_ok:
                return (nx, ny), curvature
        else:
            estimates.append((r, (nx, ny), curvature))

        r *= 0.5

    if estimates:
        r_last, normal_last, curvature_last = estimates[-1]
        msg = (
            f"Normal/curvature did not converge within refinement_levels={refinement_levels}. "
            f"Returning last estimate at r={r_last}. "
            f"normal={normal_last}, curvature={curvature_last}"
        )

        if require_convergence:
            raise RuntimeError(msg)

        warnings.warn(msg)
        return normal_last, curvature_last

    raise RuntimeError(
        f"Could not find any clean 2-crossing neighborhood near ({x}, {y}); "
        f"last radius={r}, last sample_N={N}"
    )


# Backwards-compatible alias for your earlier name.
find_normal_curvature = find_normal_curvature_circle


def normal_curvature_from_derivatives(
    Fx: float,
    Fy: float,
    Fxx: float,
    Fxy: float,
    Fyy: float,
) -> tuple[Point, float]:
    """
    Return uphill normal and signed level-set curvature from first/second derivatives.

    Sign convention: for F = x² + y² at radius R, curvature = -1/R when the
    normal points outward/uphill.
    """

    grad2 = Fx * Fx + Fy * Fy

    if grad2 == 0.0:
        raise ValueError("Gradient is zero; level-set normal/curvature undefined.")

    grad = math.sqrt(grad2)
    nx = Fx / grad
    ny = Fy / grad

    curvature = -(
        Fxx * Fy * Fy
        - 2.0 * Fxy * Fx * Fy
        + Fyy * Fx * Fx
    ) / (grad2 ** 1.5)

    return (nx, ny), curvature


def mandelbrot_normal_curvature(
    x: float,
    y: float,
    epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
) -> tuple[Point, float]:
    """Derivative-based normal/curvature for the Mandelbrot escape potential."""

    _, Gx, Gy, Gxx, Gxy, Gyy = mandelbrot_potential_derivatives(
        x=x,
        y=y,
        epsilon=epsilon,
        max_iter=max_iter,
        power=power,
        bailout=bailout,
    )

    return normal_curvature_from_derivatives(Gx, Gy, Gxx, Gxy, Gyy)


def find_normal_curvature_derivative_based(
    F_unused: ScalarFunction,
    x: float,
    y: float,
    epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
) -> tuple[Point, float]:
    """Backwards-compatible wrapper around mandelbrot_normal_curvature."""

    del F_unused
    return mandelbrot_normal_curvature(
        x=x,
        y=y,
        epsilon=epsilon,
        max_iter=max_iter,
        power=power,
    )


def charge_density_from_gradient(Fx: float, Fy: float) -> float:
    """2D conductor-style charge density sigma = |grad F| / (2*pi)."""

    return math.hypot(Fx, Fy) / (2.0 * math.pi)


def mandelbrot_charge_density(
    x: float,
    y: float,
    epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
) -> float:
    """Approximate exterior charge density |grad G|/(2*pi) at an exterior point."""

    _, Gx, Gy, _, _, _ = mandelbrot_potential_derivatives(
        x=x,
        y=y,
        epsilon=epsilon,
        max_iter=max_iter,
        power=power,
        bailout=bailout,
    )

    return charge_density_from_gradient(Gx, Gy)


def _curvature_from_jet(Fx: float, Fy: float, Fxx: float, Fxy: float, Fyy: float) -> float:
    return normal_curvature_from_derivatives(Fx, Fy, Fxx, Fxy, Fyy)[1]


def _project_to_level(
    D: DerivativeFunction,
    x: float,
    y: float,
    F_target: float,
    tol: float = ROOT_EPSILON_DEFAULT,
    max_iter: int = 10,
    max_backtracks: int = 20,
) -> tuple[float, float, float]:
    """
    Damped Newton projection onto F(x, y) = F_target along the local gradient.

    The undamped Newton step can be too aggressive near tiny-G Mandelbrot
    filaments/fjords and may jump into an unresolved/interior point. This
    version backtracks until the residual improves and all derivative values
    stay finite.
    """

    if not _is_finite_point(x, y):
        raise FloatingPointError(f"Cannot project non-finite point ({x}, {y}).")

    jet = D(x, y)

    if not _is_finite_jet(jet):
        raise FloatingPointError(f"Cannot project from non-finite derivative jet at ({x}, {y}).")

    F_val, Fx, Fy, _, _, _ = jet

    for _ in range(max_iter):
        err = F_val - F_target

        if abs(err) <= tol:
            return x, y, F_val

        grad2 = Fx * Fx + Fy * Fy

        if not math.isfinite(grad2) or grad2 <= 0.0:
            raise ValueError(f"Cannot project to level: invalid gradient at ({x}, {y}).")

        step_x = err * Fx / grad2
        step_y = err * Fy / grad2

        accepted = False
        best: tuple[float, float, Jet2] | None = None
        best_err = abs(err)

        alpha = 1.0

        for _bt in range(max_backtracks + 1):
            x_try = x - alpha * step_x
            y_try = y - alpha * step_y

            if not _is_finite_point(x_try, y_try):
                alpha *= 0.5
                continue

            try:
                jet_try = D(x_try, y_try)
            except Exception:
                alpha *= 0.5
                continue

            if not _is_finite_jet(jet_try):
                alpha *= 0.5
                continue

            F_try, Fx_try, Fy_try, _, _, _ = jet_try
            err_try = abs(F_try - F_target)

            if err_try < best_err:
                best = (x_try, y_try, jet_try)
                best_err = err_try

            # Accept clear improvement. This does not need to be perfect; it
            # just needs to keep projection from detonating.
            if err_try <= 0.5 * abs(err) or err_try <= tol:
                x, y = x_try, y_try
                F_val, Fx, Fy = F_try, Fx_try, Fy_try
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            if best is None:
                raise RuntimeError(
                    f"Projection failed near ({x}, {y}); all damped Newton trials failed."
                )

            x, y, jet = best
            F_val, Fx, Fy, _, _, _ = jet

            # If even the best candidate barely improves, stop fighting it and
            # let the outer adaptive RK step try a smaller step.
            if best_err >= 0.9 * abs(err):
                raise RuntimeError(
                    f"Projection stagnated near ({x}, {y}); residual {best_err}."
                )

    return x, y, F_val


def follow_normals(
    D: DerivativeFunction,
    x: float,
    y: float,
    F_target: float,
    dF_step: float | None = None,
    max_ds: float | None = None,
    max_turn_angle: float | None = 0.1,
    tol_F: float = ROOT_EPSILON_DEFAULT,
    max_steps: int = 10_000,
    return_path: bool = False,
    project_each_step: bool = True,
    max_step_halvings: int = 24,
) -> Point | NormalPath:
    """
    Follow normal/gradient-flow lines from F(x, y) to F_target.

    The ODE is integrated using F itself as the independent variable:

        dr/dF = grad(F) / |grad(F)|²

    This automatically moves uphill if F_target > F(x, y), and downhill if
    F_target < F(x, y). RK4 is used in F-space, optionally followed by a damped
    Newton projection onto the expected intermediate level.

    The step is adaptive in a simple defensive sense: if an RK trial or
    projection lands in an unresolved/interior/non-finite point, the step in
    F-space is halved and retried.
    """

    if not _is_finite_point(x, y):
        raise ValueError(f"Starting point must be finite, got ({x}, {y}).")

    jet0 = D(x, y)

    if not _is_finite_jet(jet0):
        raise ValueError(f"Starting derivative jet is non-finite at ({x}, {y}).")

    F0, *_ = jet0

    if abs(F_target - F0) <= tol_F:
        if return_path:
            return [(x, y, F0)]
        return x, y

    total_dF = F_target - F0

    if dF_step is None:
        dF_step = abs(total_dF) / 100.0

    if dF_step <= 0.0:
        raise ValueError("dF_step must be positive.")

    path: NormalPath = [(x, y, F0)]

    def rhs(a: float, b: float) -> tuple[float, float, float, float, float]:
        if not _is_finite_point(a, b):
            raise FloatingPointError(f"Cannot evaluate RHS at non-finite point ({a}, {b}).")

        F_val, Fx, Fy, Fxx, Fxy, Fyy = D(a, b)

        if not all(math.isfinite(v) for v in (F_val, Fx, Fy, Fxx, Fxy, Fyy)):
            raise FloatingPointError(f"Non-finite derivative jet at ({a}, {b}).")

        grad2 = Fx * Fx + Fy * Fy

        if not math.isfinite(grad2) or grad2 <= 0.0:
            raise ValueError(f"Invalid/zero gradient at ({a}, {b}).")

        grad = math.sqrt(grad2)
        vx = Fx / grad2
        vy = Fy / grad2

        if not all(math.isfinite(v) for v in (vx, vy, grad)):
            raise FloatingPointError(f"Non-finite normal-flow RHS at ({a}, {b}).")

        curvature = _curvature_from_jet(Fx, Fy, Fxx, Fxy, Fyy)

        return vx, vy, F_val, grad, curvature

    for _step_index in range(max_steps):
        F_cur, *_ = D(x, y)
        remaining = F_target - F_cur

        if abs(remaining) <= tol_F:
            if project_each_step:
                x, y, F_cur = _project_to_level(D, x, y, F_target, tol=tol_F)

            if return_path:
                _append_path_point(path, x, y, F_cur)
                return path

            return x, y

        h_base = math.copysign(min(abs(dF_step), abs(remaining)), remaining)

        _, _, _, grad, curvature = rhs(x, y)
        predicted_ds = abs(h_base) / grad

        if max_ds is not None and predicted_ds > max_ds:
            h_base *= max_ds / predicted_ds
            predicted_ds = max_ds

        if max_turn_angle is not None and curvature != 0.0 and math.isfinite(curvature):
            ds_limit = max_turn_angle / max(abs(curvature), 1e-300)

            if predicted_ds > ds_limit:
                h_base *= ds_limit / predicted_ds

        last_error: Exception | None = None
        accepted = False

        for halvings in range(max_step_halvings + 1):
            h = h_base * (0.5 ** halvings)
            expected_F = F_cur + h

            try:
                k1x, k1y, _, _, _ = rhs(x, y)
                k2x, k2y, _, _, _ = rhs(x + 0.5 * h * k1x, y + 0.5 * h * k1y)
                k3x, k3y, _, _, _ = rhs(x + 0.5 * h * k2x, y + 0.5 * h * k2y)
                k4x, k4y, _, _, _ = rhs(x + h * k3x, y + h * k3y)

                x_trial = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                y_trial = y + (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

                if not _is_finite_point(x_trial, y_trial):
                    raise FloatingPointError(
                        f"RK trial produced non-finite point ({x_trial}, {y_trial})."
                    )

                if project_each_step:
                    x_trial, y_trial, F_trial = _project_to_level(
                        D,
                        x_trial,
                        y_trial,
                        expected_F,
                        tol=tol_F,
                    )
                else:
                    jet_trial = D(x_trial, y_trial)

                    if not _is_finite_jet(jet_trial):
                        raise FloatingPointError(
                            f"RK trial produced non-finite jet at ({x_trial}, {y_trial})."
                        )

                    F_trial, *_ = jet_trial

                x, y = x_trial, y_trial
                accepted = True

                if return_path:
                    _append_path_point(path, x, y, F_trial)

                break

            except Exception as exc:
                last_error = exc
                continue

        if not accepted:
            raise RuntimeError(
                f"Normal following failed near ({x}, {y}), F={F_cur}, "
                f"target={F_target}; even {max_step_halvings} step halvings failed."
            ) from last_error

    raise RuntimeError(
        f"follow_normals did not reach F_target={F_target} within max_steps={max_steps}."
    )


def _contour_tangent_from_jet(
    Fx: float,
    Fy: float,
    direction: int = 1,
) -> Point:
    """Return a unit tangent vector to the level set, rotated from grad(F)."""

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    grad = math.hypot(Fx, Fy)

    if not math.isfinite(grad) or grad <= 0.0:
        raise ValueError("Cannot compute contour tangent: invalid/zero gradient.")

    # +1 is the +90 degree rotation of the uphill normal.
    return direction * (-Fy / grad), direction * (Fx / grad)


def _contour_rhs(
    D: DerivativeFunction,
    x: float,
    y: float,
    direction: int,
) -> tuple[float, float, float, float]:
    """
    Unit-speed tangent RHS for tracing F = constant.

    Returns:
        tx, ty, F_val, curvature
    """

    if not _is_finite_point(x, y):
        raise FloatingPointError(f"Cannot evaluate contour RHS at non-finite point ({x}, {y}).")

    F_val, Fx, Fy, Fxx, Fxy, Fyy = D(x, y)

    if not _is_finite_jet((F_val, Fx, Fy, Fxx, Fxy, Fyy)):
        raise FloatingPointError(f"Non-finite derivative jet at ({x}, {y}).")

    tx, ty = _contour_tangent_from_jet(Fx, Fy, direction=direction)
    curvature = _curvature_from_jet(Fx, Fy, Fxx, Fxy, Fyy)

    if not all(math.isfinite(v) for v in (tx, ty, F_val, curvature)):
        raise FloatingPointError(f"Non-finite contour RHS at ({x}, {y}).")

    return tx, ty, F_val, curvature


def _rk4_contour_step(
    D: DerivativeFunction,
    x: float,
    y: float,
    F_target: float,
    ds: float,
    direction: int,
    project_each_step: bool,
    tol_F: float,
) -> tuple[float, float, float]:
    """Take one RK4 step along the contour F = F_target."""

    k1x, k1y, _, _ = _contour_rhs(D, x, y, direction)
    k2x, k2y, _, _ = _contour_rhs(
        D,
        x + 0.5 * ds * k1x,
        y + 0.5 * ds * k1y,
        direction,
    )
    k3x, k3y, _, _ = _contour_rhs(
        D,
        x + 0.5 * ds * k2x,
        y + 0.5 * ds * k2y,
        direction,
    )
    k4x, k4y, _, _ = _contour_rhs(
        D,
        x + ds * k3x,
        y + ds * k3y,
        direction,
    )

    x_new = x + (ds / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (ds / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if not _is_finite_point(x_new, y_new):
        raise FloatingPointError(f"RK contour step produced non-finite point ({x_new}, {y_new}).")

    if project_each_step:
        return _project_to_level(D, x_new, y_new, F_target, tol=tol_F)

    jet_new = D(x_new, y_new)

    if not _is_finite_jet(jet_new):
        raise FloatingPointError(f"RK contour step produced non-finite jet at ({x_new}, {y_new}).")

    F_new, *_ = jet_new
    return x_new, y_new, F_new


def _local_contour_step_size(
    D: DerivativeFunction,
    x: float,
    y: float,
    ds: float,
    direction: int,
    max_turn_angle: float | None,
) -> float:
    """Limit contour step by local curvature so one step does not turn too much."""

    if ds <= 0.0:
        raise ValueError("ds must be positive.")

    if max_turn_angle is None:
        return ds

    if max_turn_angle <= 0.0:
        raise ValueError("max_turn_angle must be positive or None.")

    _, _, _, curvature = _contour_rhs(D, x, y, direction)

    if curvature == 0.0 or not math.isfinite(curvature):
        return ds

    ds_limit = max_turn_angle / max(abs(curvature), 1e-300)
    return min(ds, ds_limit)


def _segment_cross_x_at_y0(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float | None:
    """Return x where the segment crosses y=0, or None if it does not."""

    dy = y1 - y0

    if dy == 0.0:
        return None

    t = -y0 / dy

    if 0.0 <= t <= 1.0:
        return x0 + t * (x1 - x0)

    return None


def _orient2d(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect_strict(
    a0: Point,
    a1: Point,
    b0: Point,
    b1: Point,
    tol: float = 1e-14,
) -> bool:
    """Strict intersection test for two closed segments, excluding shared endpoints."""

    (ax0, ay0), (ax1, ay1) = a0, a1
    (bx0, by0), (bx1, by1) = b0, b1

    # Quick bbox reject.
    if max(min(ax0, ax1), min(bx0, bx1)) > min(max(ax0, ax1), max(bx0, bx1)) + tol:
        return False
    if max(min(ay0, ay1), min(by0, by1)) > min(max(ay0, ay1), max(by0, by1)) + tol:
        return False

    # Ignore intersections that are just shared endpoints.
    shared = (
        (abs(ax0 - bx0) <= tol and abs(ay0 - by0) <= tol)
        or (abs(ax0 - bx1) <= tol and abs(ay0 - by1) <= tol)
        or (abs(ax1 - bx0) <= tol and abs(ay1 - by0) <= tol)
        or (abs(ax1 - bx1) <= tol and abs(ay1 - by1) <= tol)
    )
    if shared:
        return False

    o1 = _orient2d(ax0, ay0, ax1, ay1, bx0, by0)
    o2 = _orient2d(ax0, ay0, ax1, ay1, bx1, by1)
    o3 = _orient2d(bx0, by0, bx1, by1, ax0, ay0)
    o4 = _orient2d(bx0, by0, bx1, by1, ax1, ay1)

    return (o1 * o2 < -tol) and (o3 * o4 < -tol)


def _candidate_segment_intersects_recent(
    path: ContourPath,
    x_new: float,
    y_new: float,
    window: int = 2048,
) -> bool:
    """Check whether the new segment intersects a recent part of the contour."""

    if len(path) < 4:
        return False

    a0 = (path[-1][0], path[-1][1])
    a1 = (x_new, y_new)

    start = max(0, len(path) - 1 - window)
    # Exclude the most recent two segments, which are adjacent to the candidate.
    end = len(path) - 3

    for i in range(start, end):
        b0 = (path[i][0], path[i][1])
        b1 = (path[i + 1][0], path[i + 1][1])
        if _segments_intersect_strict(a0, a1, b0, b1):
            return True

    return False


def _turn_angle_between_path_segments(
    path: ContourPath,
    x_new: float,
    y_new: float,
) -> float:
    """
    Return the actual turn angle in radians made by the last accepted segment
    and a candidate new segment. Returns 0 for the first usable segment.
    """

    if len(path) < 2:
        return 0.0

    x0, y0, _ = path[-2]
    x1, y1, _ = path[-1]

    v0x = x1 - x0
    v0y = y1 - y0
    v1x = x_new - x1
    v1y = y_new - y1

    n0 = math.hypot(v0x, v0y)
    n1 = math.hypot(v1x, v1y)

    if n0 <= 0.0 or n1 <= 0.0:
        return 0.0

    dot = (v0x * v1x + v0y * v1y) / (n0 * n1)
    dot = max(-1.0, min(1.0, dot))

    return math.acos(dot)



def _half_contour_progress_angle(
    x: float,
    y: float,
    x_left: float,
    x_right: float,
    upper_half: bool,
) -> float:
    """
    Angular progress coordinate for a symmetric half-contour.

    We measure angle around the midpoint of the two real-axis intersections:
        right endpoint -> 0
        left endpoint  -> pi

    For the lower half, y is flipped so the convention is identical.
    """

    cx = 0.5 * (x_left + x_right)
    yy = y if upper_half else -y
    theta = math.atan2(yy, x - cx)

    if theta < 0.0:
        theta = 0.0

    if theta > math.pi:
        theta = math.pi

    return theta


def _half_contour_progress_fraction_from_path(
    path: ContourPath,
    x_left: float,
    x_right: float,
    upper_half: bool,
    recent_fraction: float = 0.10,
    recent_min_points: int = 2000,
) -> float:
    """
    Estimate progress through the current accepted half-contour.

    Important subtlety:
        The contour can temporarily reach an angular coordinate very close to pi
        near a pinch/fjord before the whole local bulb geometry has been traced.
        Therefore "maximum angle ever seen" can wildly overstate progress.

    Instead, use the maximum angular coordinate only over a recent suffix of the
    accepted path. Old angular peaks age out. If rollback trims the path,
    progress naturally moves backward too.
    """

    if not path:
        return 0.0

    n = len(path)

    if recent_fraction <= 0.0:
        recent_path = path
    else:
        recent_count = max(int(recent_min_points), int(math.ceil(float(recent_fraction) * n)))
        recent_count = max(1, min(n, recent_count))
        recent_path = path[-recent_count:]

    max_theta = 0.0

    for x, y, _ in recent_path:
        theta = _half_contour_progress_angle(
            x=x,
            y=y,
            x_left=x_left,
            x_right=x_right,
            upper_half=upper_half,
        )
        max_theta = max(max_theta, theta)

    return max(0.0, min(1.0, max_theta / math.pi))


def _render_progress_bar(fraction: float, width: int = 32) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    width = max(4, int(width))

    filled = int(round(width * fraction))
    filled = max(0, min(width, filled))

    return "[" + "#" * filled + "-" * (width - filled) + f"] {100.0 * fraction:6.2f}%"


def _print_contour_progress(
    *,
    fraction: float,
    target_G: float,
    points: int,
    ds_nominal: float,
    rollbacks: int,
    width: int = 32,
    force_newline: bool = False,
) -> None:
    bar = _render_progress_bar(fraction, width=width)
    ending = "\n" if force_newline else "\r"

    print(
        f"    contour G={target_G:g} {bar} "
        f"points={points:,} ds≈{ds_nominal:.3g} rollbacks={rollbacks}",
        end=ending,
        flush=True,
    )


def trace_contour_rk4(
    D: DerivativeFunction,
    x: float,
    y: float,
    F_target: float | None = None,
    ds: float = 1e-3,
    direction: int = 1,
    max_turn_angle: float | None = 0.08,
    tol_F: float = ROOT_EPSILON_DEFAULT,
    max_steps: int = 100_000,
    close_loop: bool = True,
    close_tol: float | None = None,
    min_steps_before_close: int = 50,
    project_each_step: bool = True,
    max_step_halvings: int = 24,
) -> ContourPath:
    """
    Trace one connected level-set contour using RK4 tangent integration.

    The tangent ODE is:

        dr/ds = rotate90(grad F) / |grad F|

    Analytically this keeps F constant, because grad(F) dot dr/ds = 0.
    Numerically, every step can be projected back to F = F_target with the
    damped Newton projector used by normal following.

    Step-size safety:
        * local curvature limits ds via abs(curvature) * ds <= max_turn_angle
        * failed RK/projection trials halve ds and retry
        * optional closure detection stops when the path returns near the start
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    if ds <= 0.0:
        raise ValueError("ds must be positive")

    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    if min_steps_before_close < 1:
        raise ValueError("min_steps_before_close must be >= 1")

    if close_tol is None:
        close_tol = 2.5 * ds

    if close_tol <= 0.0:
        raise ValueError("close_tol must be positive")

    jet0 = D(x, y)

    if not _is_finite_jet(jet0):
        raise ValueError(f"Starting derivative jet is non-finite at ({x}, {y}).")

    F0, *_ = jet0

    if F_target is None:
        F_target = F0
    elif abs(F0 - F_target) > tol_F:
        x, y, F0 = _project_to_level(D, x, y, F_target, tol=tol_F)

    x_start, y_start = x, y
    path: ContourPath = [(x, y, F_target)]

    for step_index in range(1, max_steps + 1):
        ds_base = _local_contour_step_size(
            D=D,
            x=x,
            y=y,
            ds=ds,
            direction=direction,
            max_turn_angle=max_turn_angle,
        )

        accepted = False
        last_error: Exception | None = None

        for halvings in range(max_step_halvings + 1):
            ds_try = ds_base * (0.5 ** halvings)

            try:
                x_new, y_new, F_new = _rk4_contour_step(
                    D=D,
                    x=x,
                    y=y,
                    F_target=F_target,
                    ds=ds_try,
                    direction=direction,
                    project_each_step=project_each_step,
                    tol_F=tol_F,
                )
            except Exception as exc:
                last_error = exc
                continue

            if _candidate_segment_intersects_recent(path, x_new, y_new):
                last_error = RuntimeError(
                    f"Candidate contour segment self-intersects near ({x_new}, {y_new})."
                )
                continue

            x, y = x_new, y_new
            _append_path_point(path, x, y, F_new)
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                f"Contour tracing failed near ({x}, {y}), F_target={F_target}; "
                f"even {max_step_halvings} step halvings failed."
            ) from last_error

        if close_loop and step_index >= min_steps_before_close:
            dist_to_start = math.hypot(x - x_start, y - y_start)

            if dist_to_start <= close_tol:
                if close_loop:
                    _append_path_point(path, x_start, y_start, F_target, tol=close_tol * 0.1)
                return path

    raise RuntimeError(
        f"trace_contour_rk4 did not close/reach stop condition within max_steps={max_steps}."
    )


def _mandelbrot_potential_for_root(
    target_G: float,
    escape_epsilon: float,
    escape_max_iter: int,
    power: int,
    bailout: float,
) -> ScalarFunction:
    """Build a scalar potential function for real-axis root searches."""

    del target_G

    def F(x: float, y: float) -> float:
        return mandelbrot_potential(
            x=x,
            y=y,
            epsilon=escape_epsilon,
            max_iter=escape_max_iter,
            power=power,
            bailout=bailout,
        )

    return F


def find_mandelbrot_real_axis_intersections(
    target_G: float,
    escape_epsilon: float = POTENTIAL_EPSILON_DEFAULT,
    root_epsilon: float = ROOT_EPSILON_DEFAULT,
    escape_max_iter: int = MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
    left_inside_x: float = -2.0,
    right_inside_x: float = 0.25,
    initial_extent: float = 10.0,
    max_expansions: int = 20,
) -> tuple[Point, Point]:
    """
    Find the two real-axis intersections of G(c) = target_G.

    Returns:
        (left_point, right_point)

    For the quadratic Mandelbrot set and target_G > 0, the exterior
    equipotential intersects the real axis once to the left of -2 and once to
    the right of 1/4.
    """

    if target_G <= 0.0:
        raise ValueError("target_G must be positive for exterior contours.")

    if power != 2:
        raise ValueError("real-axis intersection helper is currently intended for power == 2")

    F = _mandelbrot_potential_for_root(
        target_G=target_G,
        escape_epsilon=escape_epsilon,
        escape_max_iter=escape_max_iter,
        power=power,
        bailout=bailout,
    )

    x_left = -abs(initial_extent)
    x_right = abs(initial_extent)

    for _ in range(max_expansions + 1):
        if F(x_left, 0.0) > target_G:
            break
        x_left *= 2.0
    else:
        raise RuntimeError("Could not bracket left real-axis intersection.")

    for _ in range(max_expansions + 1):
        if F(x_right, 0.0) > target_G:
            break
        x_right *= 2.0
    else:
        raise RuntimeError("Could not bracket right real-axis intersection.")

    left = find_root_2d(
        F,
        x_left,
        0.0,
        left_inside_x,
        0.0,
        val=target_G,
        epsilon=root_epsilon,
    )

    right = find_root_2d(
        F,
        right_inside_x,
        0.0,
        x_right,
        0.0,
        val=target_G,
        epsilon=root_epsilon,
    )

    return left, right



def mirror_contour_y(
    half: ContourPath,
    close_loop: bool = True,
) -> ContourPath:
    """
    Mirror a y-symmetric half-contour across the real axis.

    Input is expected to run from the right real-axis intersection to the left
    real-axis intersection through one half-plane. Endpoints are forced to
    y=0. The mirrored interior points are appended in reverse order so the full
    loop returns to the starting point.
    """

    if len(half) < 2:
        raise ValueError("Need at least two points to mirror a half-contour.")

    clean: ContourPath = [(float(x), float(y), float(g)) for x, y, g in half]

    x0, _y0, g0 = clean[0]
    x1, _y1, g1 = clean[-1]
    clean[0] = (x0, 0.0, g0)
    clean[-1] = (x1, 0.0, g1)

    mirrored: ContourPath = [
        (float(x), -float(y), float(g))
        for x, y, g in reversed(clean[1:-1])
    ]

    full: ContourPath = clean + mirrored

    if close_loop:
        x_start, y_start, g_start = full[0]
        _append_path_point(full, x_start, y_start, g_start)

    return full

def trace_mandelbrot_contour_half_symmetric(
    target_G: float,
    derivative_epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    derivative_max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    escape_epsilon: float = POTENTIAL_EPSILON_DEFAULT,
    escape_max_iter: int = MAX_ITER_DEFAULT,
    root_epsilon: float = ROOT_EPSILON_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
    ds: float = 1e-3,
    upper_half: bool = True,
    max_turn_angle: float | None = 0.08,
    tol_G: float = ROOT_EPSILON_DEFAULT,
    max_steps: int = 100_000,
    min_steps_before_crossing: int = 10,
    project_each_step: bool = True,
    max_step_halvings: int = 32,
    adaptive_ds: bool = True,
    ds_min: float = 1e-10,
    ds_growth: float = 1.015,
    ds_shrink: float = 0.5,
    grow_after_successes: int = 12,
    rollback_points: int = 64,
    rollback_shrink: float = 0.35,
    max_rollbacks: int = 200,
    use_step_doubling: bool = True,
    step_error_factor: float = 0.025,
    max_actual_turn_angle: float | None = None,
    self_intersection_window: int = 4096,
    seam_tol_factor: float = 6.0,
    seam_tol_min: float = 1e-8,
    progress: bool = False,
    progress_every: int = 500,
    progress_width: int = 32,
    progress_recent_fraction: float = 0.10,
    progress_recent_min_points: int = 2000,
) -> ContourPath:
    """
    Trace one y-symmetric half of a quadratic Mandelbrot equipotential.

    This is the robust/adaptive version.

    Adaptive behavior:
        * local curvature limits the proposed step,
        * RK step-doubling rejects steps whose one-step and two-half-step
          endpoints disagree too much,
        * actual polyline turn angle can reject unexpectedly sharp kinks,
        * recent self-intersections reject tiny inverted loops,
        * crossing the symmetry axis away from the known left endpoint is
          rejected instead of prematurely closing the contour,
        * repeated failures roll back to an earlier safe point, shrink ds, and
          continue,
        * long clean stretches gently grow ds back toward the requested value.
    """

    if target_G <= 0.0:
        raise ValueError("target_G must be positive for exterior contours.")

    if power != 2:
        raise ValueError("y-symmetric contour tracing is currently intended for power == 2")

    if ds <= 0.0:
        raise ValueError("ds must be positive")

    if ds_min <= 0.0:
        raise ValueError("ds_min must be positive")

    if ds_growth <= 1.0:
        raise ValueError("ds_growth must be > 1")

    if not (0.0 < ds_shrink < 1.0):
        raise ValueError("ds_shrink must be between 0 and 1")

    if not (0.0 < rollback_shrink < 1.0):
        raise ValueError("rollback_shrink must be between 0 and 1")

    if step_error_factor <= 0.0:
        raise ValueError("step_error_factor must be positive")

    if max_actual_turn_angle is None and max_turn_angle is not None:
        # Allow a little more actual polyline turn than the local differential
        # curvature budget. If this triggers repeatedly, the step is too large.
        max_actual_turn_angle = 2.5 * max_turn_angle

    left, right = find_mandelbrot_real_axis_intersections(
        target_G=target_G,
        escape_epsilon=escape_epsilon,
        root_epsilon=root_epsilon,
        escape_max_iter=escape_max_iter,
        power=power,
        bailout=bailout,
    )

    D = mandelbrot_derivative_function(
        epsilon=derivative_epsilon,
        max_iter=derivative_max_iter,
        power=power,
        bailout=bailout,
    )

    x, y = right
    x_left, _ = left
    x_right, _ = right

    direction = 1 if upper_half else -1
    half_plane_sign = 1.0 if upper_half else -1.0

    path: ContourPath = [(x, y, target_G)]

    progress_every = max(1, int(progress_every))
    last_progress_print_step = 0

    if progress:
        _print_contour_progress(
            fraction=0.0,
            target_G=target_G,
            points=len(path),
            ds_nominal=float(ds),
            rollbacks=0,
            width=progress_width,
        )

    ds_goal = float(ds)
    ds_nominal = float(ds)
    successes_since_shrink = 0
    rollbacks = 0
    logical_step = 0

    def take_step_once(x0: float, y0: float, step: float) -> tuple[float, float, float]:
        """One robust candidate step; optionally use step-doubling error control."""

        x_full, y_full, F_full = _rk4_contour_step(
            D=D,
            x=x0,
            y=y0,
            F_target=target_G,
            ds=step,
            direction=direction,
            project_each_step=project_each_step,
            tol_F=tol_G,
        )

        if not use_step_doubling:
            return x_full, y_full, F_full

        x_half, y_half, _ = _rk4_contour_step(
            D=D,
            x=x0,
            y=y0,
            F_target=target_G,
            ds=0.5 * step,
            direction=direction,
            project_each_step=project_each_step,
            tol_F=tol_G,
        )

        x_two, y_two, F_two = _rk4_contour_step(
            D=D,
            x=x_half,
            y=y_half,
            F_target=target_G,
            ds=0.5 * step,
            direction=direction,
            project_each_step=project_each_step,
            tol_F=tol_G,
        )

        err = math.hypot(x_two - x_full, y_two - y_full)

        if not math.isfinite(err):
            raise FloatingPointError("Non-finite RK step-doubling error.")

        if err > step_error_factor * max(step, 1e-300):
            raise RuntimeError(
                f"RK step-doubling error too large: {err:g} > "
                f"{step_error_factor:g} * {step:g}"
            )

        # Use the two-half-step result; it is normally more accurate.
        return x_two, y_two, F_two

    while logical_step < max_steps:
        logical_step += 1

        ds_base = _local_contour_step_size(
            D=D,
            x=x,
            y=y,
            ds=ds_nominal,
            direction=direction,
            max_turn_angle=max_turn_angle,
        )

        accepted = False
        last_error: Exception | None = None
        accepted_halvings = 0
        accepted_ds = ds_base

        for halvings in range(max_step_halvings + 1):
            ds_try = ds_base * (0.5 ** halvings)

            if ds_try < ds_min:
                last_error = RuntimeError(
                    f"Candidate ds={ds_try:g} below ds_min={ds_min:g}."
                )
                break

            try:
                x_new, y_new, G_new = take_step_once(x, y, ds_try)

                if not _is_finite_point(x_new, y_new):
                    raise FloatingPointError(f"Non-finite candidate point ({x_new}, {y_new}).")

                # Actual geometric turn check. This catches occasional local
                # kinks that the start-point curvature did not predict.
                if max_actual_turn_angle is not None and len(path) >= 2:
                    actual_turn = _turn_angle_between_path_segments(path, x_new, y_new)
                    if actual_turn > max_actual_turn_angle:
                        raise RuntimeError(
                            f"Actual contour turn too large: {actual_turn:g} rad > "
                            f"{max_actual_turn_angle:g} rad"
                        )

                seam_tol = max(float(seam_tol_factor) * ds_try, float(seam_tol_min))
                x_cross = None
                crossed_axis = False

                if logical_step >= min_steps_before_crossing:
                    crossed_axis = (
                        (half_plane_sign * y > 0.0)
                        and (half_plane_sign * y_new <= 0.0)
                    )
                    if crossed_axis:
                        x_cross = _segment_cross_x_at_y0(x, y, x_new, y_new)

                near_left_endpoint = (
                    math.hypot(x_new - x_left, y_new) <= seam_tol
                    or (x_cross is not None and abs(x_cross - x_left) <= seam_tol)
                )

                if crossed_axis:
                    if near_left_endpoint:
                        _append_path_point(path, x_left, 0.0, target_G)
                        return path

                    raise RuntimeError(
                        f"Contour half crossed y=0 away from the left endpoint: "
                        f"x_cross={x_cross}, x_left={x_left}, tol={seam_tol}."
                    )

                # Do not accept points that leak into the wrong half-plane.
                if half_plane_sign * y_new < -seam_tol_min:
                    raise RuntimeError(
                        f"Candidate point left the requested half-plane: ({x_new}, {y_new})."
                    )

                if _candidate_segment_intersects_recent(
                    path,
                    x_new,
                    y_new,
                    window=int(self_intersection_window),
                ):
                    raise RuntimeError(
                        f"Candidate half-contour segment self-intersects near ({x_new}, {y_new})."
                    )

                if near_left_endpoint and logical_step >= min_steps_before_crossing:
                    _append_path_point(path, x_left, 0.0, target_G)
                    if progress:
                        _print_contour_progress(
                            fraction=1.0,
                            target_G=target_G,
                            points=len(path),
                            ds_nominal=ds_nominal,
                            rollbacks=rollbacks,
                            width=progress_width,
                            force_newline=True,
                        )
                    return path

                x, y = x_new, y_new
                _append_path_point(path, x, y, G_new)

                accepted = True
                accepted_halvings = halvings
                accepted_ds = ds_try
                break

            except Exception as exc:
                last_error = exc
                continue

        if accepted:
            if adaptive_ds:
                if accepted_halvings > 0:
                    # The requested local step was too optimistic. Keep the
                    # global nominal step near what actually worked.
                    ds_nominal = max(ds_min, min(ds_nominal, accepted_ds / max(ds_shrink, 1e-300)))
                    ds_nominal *= ds_shrink
                    ds_nominal = max(ds_min, ds_nominal)
                    successes_since_shrink = 0
                else:
                    successes_since_shrink += 1
                    if successes_since_shrink >= int(grow_after_successes):
                        ds_nominal = min(ds_goal, ds_nominal * ds_growth)
                        successes_since_shrink = 0
            if progress and (
                logical_step - last_progress_print_step >= progress_every
                or len(path) <= 2
            ):
                fraction = _half_contour_progress_fraction_from_path(
                    path=path,
                    x_left=x_left,
                    x_right=x_right,
                    upper_half=upper_half,
                    recent_fraction=progress_recent_fraction,
                    recent_min_points=progress_recent_min_points,
                )
                _print_contour_progress(
                    fraction=fraction,
                    target_G=target_G,
                    points=len(path),
                    ds_nominal=ds_nominal,
                    rollbacks=rollbacks,
                    width=progress_width,
                )
                last_progress_print_step = logical_step

            continue

        # If even local step-halving cannot move forward, assume the last few
        # accepted points led us into a numerically bad local configuration.
        # Roll back to a comfortable older point, shrink the nominal step, and
        # try again. Early in the contour there may not be rollback_points
        # samples yet; in that case roll back to the start, or simply stay at
        # the current point, shrink ds, and retry. Do not fail just because the
        # trouble happened early near the right cusp.
        if adaptive_ds and rollbacks < max_rollbacks:
            rollbacks += 1

            if len(path) > 1:
                keep = max(1, len(path) - int(rollback_points))
                path = path[:keep]
                x, y, _ = path[-1]

            ds_nominal = max(ds_min, min(ds_goal, ds_nominal * rollback_shrink))
            successes_since_shrink = 0

            # If the only problem is that the step-halving ladder hit ds_min,
            # continuing with a smaller nominal ds is still useful as long as
            # ds_min itself is not too high. If ds_min is too high, the caller
            # should lower trace_ds_min / ds_min.

            if progress:
                fraction = _half_contour_progress_fraction_from_path(
                    path=path,
                    x_left=x_left,
                    x_right=x_right,
                    upper_half=upper_half,
                    recent_fraction=progress_recent_fraction,
                    recent_min_points=progress_recent_min_points,
                )
                _print_contour_progress(
                    fraction=fraction,
                    target_G=target_G,
                    points=len(path),
                    ds_nominal=ds_nominal,
                    rollbacks=rollbacks,
                    width=progress_width,
                )
                last_progress_print_step = logical_step
            continue

        raise RuntimeError(
            f"Symmetric contour-half tracing failed near ({x}, {y}), target_G={target_G}; "
            f"even {max_step_halvings} step halvings failed after {rollbacks} rollback(s). "
            f"Last error: {last_error}"
        ) from last_error

    raise RuntimeError(
        f"trace_mandelbrot_contour_half_symmetric did not reach the opposite real-axis "
        f"intersection within max_steps={max_steps}."
    )


def trace_mandelbrot_contour(
    target_G: float,
    derivative_epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    derivative_max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    escape_epsilon: float = POTENTIAL_EPSILON_DEFAULT,
    escape_max_iter: int = MAX_ITER_DEFAULT,
    root_epsilon: float = ROOT_EPSILON_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
    ds: float = 1e-3,
    direction: int = 1,
    max_turn_angle: float | None = 0.08,
    tol_G: float = ROOT_EPSILON_DEFAULT,
    max_steps: int = 100_000,
    use_y_symmetry: bool = True,
    upper_half: bool = True,
    close_loop: bool = True,
    project_each_step: bool = True,
    max_step_halvings: int = 32,
    adaptive_ds: bool = True,
    ds_min: float = 1e-10,
    ds_growth: float = 1.015,
    ds_shrink: float = 0.5,
    grow_after_successes: int = 12,
    rollback_points: int = 64,
    rollback_shrink: float = 0.35,
    max_rollbacks: int = 200,
    use_step_doubling: bool = True,
    step_error_factor: float = 0.025,
    max_actual_turn_angle: float | None = None,
    self_intersection_window: int = 4096,
    seam_tol_factor: float = 6.0,
    seam_tol_min: float = 1e-8,
    progress: bool = False,
    progress_every: int = 500,
    progress_width: int = 32,
    progress_recent_fraction: float = 0.10,
    progress_recent_min_points: int = 2000,
) -> ContourPath:
    """
    Trace a Mandelbrot equipotential contour G(c) = target_G.

    By default this uses the y-symmetry G(x,y)=G(x,-y): it root-finds the two
    real-axis intersections, traces one half of the contour with RK4 tangent
    integration, then mirrors that half. This is usually cleaner than trying to
    detect closure of the entire loop numerically.

    Set use_y_symmetry=False to trace a full loop directly from the right
    real-axis intersection using generic closure detection.
    """

    if target_G <= 0.0:
        raise ValueError("target_G must be positive for exterior contours.")

    if use_y_symmetry:
        half = trace_mandelbrot_contour_half_symmetric(
            target_G=target_G,
            derivative_epsilon=derivative_epsilon,
            derivative_max_iter=derivative_max_iter,
            escape_epsilon=escape_epsilon,
            escape_max_iter=escape_max_iter,
            root_epsilon=root_epsilon,
            power=power,
            bailout=bailout,
            ds=ds,
            upper_half=upper_half,
            max_turn_angle=max_turn_angle,
            tol_G=tol_G,
            max_steps=max_steps,
            project_each_step=project_each_step,
            max_step_halvings=max_step_halvings,
            adaptive_ds=adaptive_ds,
            ds_min=ds_min,
            ds_growth=ds_growth,
            ds_shrink=ds_shrink,
            grow_after_successes=grow_after_successes,
            rollback_points=rollback_points,
            rollback_shrink=rollback_shrink,
            max_rollbacks=max_rollbacks,
            use_step_doubling=use_step_doubling,
            step_error_factor=step_error_factor,
            max_actual_turn_angle=max_actual_turn_angle,
            self_intersection_window=self_intersection_window,
            seam_tol_factor=seam_tol_factor,
            seam_tol_min=seam_tol_min,
            progress=progress,
            progress_every=progress_every,
            progress_width=progress_width,
            progress_recent_fraction=progress_recent_fraction,
            progress_recent_min_points=progress_recent_min_points,
        )

        return mirror_contour_y(half, close_loop=close_loop)

    left, right = find_mandelbrot_real_axis_intersections(
        target_G=target_G,
        escape_epsilon=escape_epsilon,
        root_epsilon=root_epsilon,
        escape_max_iter=escape_max_iter,
        power=power,
        bailout=bailout,
    )

    del left
    x0, y0 = right

    D = mandelbrot_derivative_function(
        epsilon=derivative_epsilon,
        max_iter=derivative_max_iter,
        power=power,
        bailout=bailout,
    )

    return trace_contour_rk4(
        D=D,
        x=x0,
        y=y0,
        F_target=target_G,
        ds=ds,
        direction=direction,
        max_turn_angle=max_turn_angle,
        tol_F=tol_G,
        max_steps=max_steps,
        close_loop=close_loop,
        project_each_step=project_each_step,
        max_step_halvings=max_step_halvings,
    )


def follow_mandelbrot_normals(
    x: float,
    y: float,
    target_G: float,
    derivative_epsilon: float = DERIVATIVE_EPSILON_DEFAULT,
    derivative_max_iter: int = DERIVATIVE_MAX_ITER_DEFAULT,
    power: int = 2,
    bailout: float = BAILOUT_DEFAULT,
    dG_step: float | None = None,
    max_ds: float | None = None,
    max_turn_angle: float | None = 0.1,
    tol_G: float = ROOT_EPSILON_DEFAULT,
    max_steps: int = 10_000,
    return_path: bool = False,
    project_each_step: bool = True,
    max_step_halvings: int = 24,
) -> Point | NormalPath:
    """Convenience wrapper for following Mandelbrot escape-potential normals."""

    D = mandelbrot_derivative_function(
        epsilon=derivative_epsilon,
        max_iter=derivative_max_iter,
        power=power,
        bailout=bailout,
    )

    return follow_normals(
        D=D,
        x=x,
        y=y,
        F_target=target_G,
        dF_step=dG_step,
        max_ds=max_ds,
        max_turn_angle=max_turn_angle,
        tol_F=tol_G,
        max_steps=max_steps,
        return_path=return_path,
        project_each_step=project_each_step,
        max_step_halvings=max_step_halvings,
    )
