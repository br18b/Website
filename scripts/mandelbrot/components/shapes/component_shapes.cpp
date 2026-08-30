#include "component_shapes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>

namespace mandelbrot::shapes {
namespace {

using Real = long double;
using Complex = std::complex<Real>;
constexpr Real kPi = 3.141592653589793238462643383279502884L;

CatalogueReal catalogue_real(Real value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << std::scientific << value;
    return catalogue::Catalogue::parse_decimal(out.str());
}

template <std::size_t N>
bool solve_linear(
    std::array<std::array<Real, N>, N> matrix,
    std::array<Real, N> rhs,
    std::array<Real, N>& solution) {
    for (std::size_t column = 0; column < N; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < N; ++row) {
            if (std::abs(matrix[row][column])
                > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1e-30L) return false;
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        const Real inverse = 1 / matrix[column][column];
        for (std::size_t j = column; j < N; ++j) {
            matrix[column][j] *= inverse;
        }
        rhs[column] *= inverse;
        for (std::size_t row = 0; row < N; ++row) {
            if (row == column) continue;
            const Real factor = matrix[row][column];
            for (std::size_t j = column; j < N; ++j) {
                matrix[row][j] -= factor * matrix[column][j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    solution = rhs;
    return true;
}

bool solve3(
    std::array<std::array<Real, 3>, 3> matrix,
    std::array<Real, 3> rhs,
    std::array<Real, 3>& solution) {
    return solve_linear<3>(matrix, rhs, solution);
}

Complex rotate(Complex value, Real angle) {
    const Real cosine = std::cos(angle);
    const Real sine = std::sin(angle);
    return {
        cosine * value.real() - sine * value.imag(),
        sine * value.real() + cosine * value.imag(),
    };
}

Real normalize_angle(Real angle) {
    angle = std::fmod(angle + kPi, 2 * kPi);
    if (angle < 0) angle += 2 * kPi;
    return angle - kPi;
}

struct CardioidCandidate {
    bool valid = false;
    int cusp_index = -1;
    int direction = 1;
    Real center_x = 0;
    Real center_y = 0;
    Real size = 0;
    Real angle = 0;
    Real xi = 0;
    Real phase_offset = 0;
    Real rms = std::numeric_limits<Real>::infinity();
    Real max_error = std::numeric_limits<Real>::infinity();
    Real score = std::numeric_limits<Real>::infinity();
};

struct CardioidBasis {
    std::vector<Complex> base;
    std::vector<Complex> slant;
};

CardioidBasis make_cardioid_basis(
    std::size_t count,
    std::size_t cusp,
    int direction,
    Real phase_offset = 0) {
    CardioidBasis result;
    result.base.resize(count);
    result.slant.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t relative = (index + count - cusp) % count;
        const Real phi = static_cast<Real>(direction) * 2 * kPi
            * static_cast<Real>(relative) / static_cast<Real>(count)
            + phase_offset;
        const Real sine = std::sin(phi);
        const Complex unit{std::cos(phi), sine};
        const Complex unit2 = unit * unit;
        result.base[index] = unit - Real(0.5L) * unit2;
        result.slant[index] = -sine * result.base[index];
    }
    return result;
}

void evaluate_cardioid_candidate(
    CardioidCandidate& candidate,
    const std::vector<Complex>& points,
    const CardioidBasis& basis) {
    if (!(candidate.size > 0) || !std::isfinite(candidate.size)
        || !std::isfinite(candidate.center_x)
        || !std::isfinite(candidate.center_y)
        || !std::isfinite(candidate.angle)
        || !std::isfinite(candidate.xi)) {
        candidate.valid = false;
        return;
    }

    const Complex center{candidate.center_x, candidate.center_y};
    const Real cosine = std::cos(candidate.angle);
    const Real sine = std::sin(candidate.angle);
    Real squared_error = 0;
    Real maximum_error = 0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const Complex canonical = basis.base[index]
            + candidate.xi * basis.slant[index];
        const Complex rotated{
            cosine * canonical.real() - sine * canonical.imag(),
            sine * canonical.real() + cosine * canonical.imag(),
        };
        const Complex model = center + candidate.size * rotated;
        const Real error = std::abs(points[index] - model);
        squared_error += error * error;
        maximum_error = std::max(maximum_error, error);
    }
    candidate.rms = std::sqrt(
        squared_error / static_cast<Real>(points.size())) / candidate.size;
    candidate.max_error = maximum_error / candidate.size;
    candidate.score = candidate.rms + Real(0.05L) * candidate.max_error;
    candidate.valid = std::isfinite(candidate.rms)
        && std::isfinite(candidate.max_error);
}

CardioidCandidate linear_cardioid_fit(
    const std::vector<Complex>& points,
    const CardioidBasis& basis,
    int cusp_index,
    int direction,
    Real phase_offset,
    Real angle,
    Real xi_limit,
    Real fallback_size) {
    CardioidCandidate candidate;
    candidate.cusp_index = cusp_index;
    candidate.direction = direction;
    candidate.phase_offset = phase_offset;
    candidate.angle = normalize_angle(angle);
    candidate.size = fallback_size;

    const std::size_t count = points.size();
    Complex point_mean{0, 0};
    Complex base_mean{0, 0};
    Complex slant_mean{0, 0};
    std::vector<Complex> rotated_base(count);
    std::vector<Complex> rotated_slant(count);
    for (std::size_t index = 0; index < count; ++index) {
        rotated_base[index] = rotate(basis.base[index], candidate.angle);
        rotated_slant[index] = rotate(basis.slant[index], candidate.angle);
        point_mean += points[index];
        base_mean += rotated_base[index];
        slant_mean += rotated_slant[index];
    }
    const Real inverse_count = 1 / static_cast<Real>(count);
    point_mean *= inverse_count;
    base_mean *= inverse_count;
    slant_mean *= inverse_count;

    Real aa = 0;
    Real ab = 0;
    Real bb = 0;
    Real aq = 0;
    Real bq = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const Complex a = rotated_base[index] - base_mean;
        const Complex b = rotated_slant[index] - slant_mean;
        const Complex q = points[index] - point_mean;
        aa += std::norm(a);
        ab += std::real(std::conj(a) * b);
        bb += std::norm(b);
        aq += std::real(std::conj(a) * q);
        bq += std::real(std::conj(b) * q);
    }

    const Real determinant = aa * bb - ab * ab;
    Real size = fallback_size;
    Real size_times_xi = 0;
    if (std::abs(determinant) > 1e-28L) {
        size = (aq * bb - bq * ab) / determinant;
        size_times_xi = (bq * aa - aq * ab) / determinant;
    }
    if (!(size > 0) || !std::isfinite(size)) {
        candidate.valid = false;
        return candidate;
    }

    Real xi = size_times_xi / size;
    xi = std::clamp(xi, -xi_limit, xi_limit);

    // Once xi is clamped, refit the common positive size exactly.
    Complex model_mean = base_mean + xi * slant_mean;
    Real denominator = 0;
    Real numerator = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const Complex model = rotated_base[index]
            + xi * rotated_slant[index] - model_mean;
        const Complex q = points[index] - point_mean;
        denominator += std::norm(model);
        numerator += std::real(std::conj(model) * q);
    }
    if (denominator > 1e-28L) size = numerator / denominator;
    if (!(size > 0) || !std::isfinite(size)) {
        candidate.valid = false;
        return candidate;
    }

    const Complex center = point_mean - size * model_mean;
    candidate.center_x = center.real();
    candidate.center_y = center.imag();
    candidate.size = size;
    candidate.xi = xi;
    evaluate_cardioid_candidate(candidate, points, basis);
    return candidate;
}

CardioidCandidate refine_cardioid_angle(
    CardioidCandidate candidate,
    const std::vector<Complex>& points,
    const CardioidBasis& basis,
    const CardioidFitOptions& options,
    Real xi_limit) {
    Real step = options.initial_angle_step.convert_to<Real>();
    step = std::max(step, 2 * kPi / static_cast<Real>(points.size()));
    const int iterations = std::max(0, options.angle_iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CardioidCandidate lower = linear_cardioid_fit(
            points, basis, candidate.cusp_index, candidate.direction,
            candidate.phase_offset, candidate.angle - step,
            xi_limit, candidate.size);
        CardioidCandidate upper = linear_cardioid_fit(
            points, basis, candidate.cusp_index, candidate.direction,
            candidate.phase_offset, candidate.angle + step,
            xi_limit, candidate.size);
        const CardioidCandidate* best = &candidate;
        if (lower.valid && lower.score < best->score) best = &lower;
        if (upper.valid && upper.score < best->score) best = &upper;
        if (best == &candidate) {
            step *= Real(0.5L);
        } else {
            candidate = *best;
        }
        if (step < 1e-10L) break;
    }
    return candidate;
}

Real cardioid_sse(
    const std::vector<Complex>& points,
    const CardioidBasis& basis,
    const std::array<Real, 5>& parameters) {
    const Complex center{parameters[0], parameters[1]};
    const Real size = std::exp(parameters[2]);
    const Real angle = parameters[3];
    const Real xi = parameters[4];
    Real sse = 0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const Complex canonical = basis.base[index]
            + xi * basis.slant[index];
        const Complex residual = center
            + size * rotate(canonical, angle) - points[index];
        sse += std::norm(residual);
    }
    return sse;
}

CardioidCandidate polish_cardioid_joint(
    CardioidCandidate candidate,
    const std::vector<Complex>& points,
    const CardioidBasis& basis,
    const CardioidFitOptions& options,
    Real xi_limit) {
    if (!candidate.valid || !(candidate.size > 0)) return candidate;
    std::array<Real, 5> parameters{{
        candidate.center_x,
        candidate.center_y,
        std::log(candidate.size),
        candidate.angle,
        candidate.xi,
    }};
    Real current_sse = cardioid_sse(points, basis, parameters);
    Real damping = 1e-6L;

    for (int iteration = 0; iteration < std::max(0, options.max_iterations);
         ++iteration) {
        std::array<std::array<Real, 5>, 5> normal{};
        std::array<Real, 5> rhs{};
        const Complex center{parameters[0], parameters[1]};
        const Real size = std::exp(parameters[2]);
        const Real angle = parameters[3];
        const Real xi = parameters[4];

        for (std::size_t index = 0; index < points.size(); ++index) {
            const Complex canonical = basis.base[index]
                + xi * basis.slant[index];
            const Complex shape = size * rotate(canonical, angle);
            const Complex xi_derivative = size
                * rotate(basis.slant[index], angle);
            const Complex angle_derivative{-shape.imag(), shape.real()};
            const Complex residual = center + shape - points[index];

            const std::array<Complex, 5> jacobian{{
                Complex{1, 0},
                Complex{0, 1},
                shape,
                angle_derivative,
                xi_derivative,
            }};
            for (std::size_t a = 0; a < 5; ++a) {
                rhs[a] -= std::real(std::conj(jacobian[a]) * residual);
                for (std::size_t b = 0; b < 5; ++b) {
                    normal[a][b] += std::real(
                        std::conj(jacobian[a]) * jacobian[b]);
                }
            }
        }
        for (std::size_t diagonal = 0; diagonal < 5; ++diagonal) {
            normal[diagonal][diagonal] += damping
                * std::max<Real>(1, normal[diagonal][diagonal]);
        }

        std::array<Real, 5> delta{};
        if (!solve_linear<5>(normal, rhs, delta)) break;
        std::array<Real, 5> trial = parameters;
        for (std::size_t index = 0; index < 5; ++index) {
            trial[index] += delta[index];
        }
        trial[3] = normalize_angle(trial[3]);
        trial[4] = std::clamp(trial[4], -xi_limit, xi_limit);
        // Prevent a single ill-conditioned step from changing the size by
        // many orders of magnitude.
        trial[2] = std::clamp(
            trial[2], parameters[2] - Real(1.0L), parameters[2] + Real(1.0L));

        const Real trial_sse = cardioid_sse(points, basis, trial);
        if (std::isfinite(trial_sse) && trial_sse < current_sse) {
            parameters = trial;
            const Real improvement = current_sse - trial_sse;
            current_sse = trial_sse;
            damping = std::max<Real>(1e-14L, damping * Real(0.3L));
            Real step_norm = 0;
            for (const Real value : delta) step_norm += value * value;
            if (std::sqrt(step_norm) < 2e-13L
                || improvement < 1e-24L * std::max<Real>(1, current_sse)) {
                break;
            }
        } else {
            damping = std::min<Real>(1e12L, damping * Real(10));
        }
    }

    candidate.center_x = parameters[0];
    candidate.center_y = parameters[1];
    candidate.size = std::exp(parameters[2]);
    candidate.angle = normalize_angle(parameters[3]);
    candidate.xi = std::clamp(parameters[4], -xi_limit, xi_limit);
    evaluate_cardioid_candidate(candidate, points, basis);
    return candidate;
}


Real cardioid_sse_with_phase(
    const std::vector<Complex>& points,
    int cusp_index,
    int direction,
    const std::array<Real, 6>& parameters) {
    const Complex center{parameters[0], parameters[1]};
    const Real size = std::exp(parameters[2]);
    const Real angle = parameters[3];
    const Real xi = parameters[4];
    const Real phase_offset = parameters[5];
    const Real cosine = std::cos(angle);
    const Real sine = std::sin(angle);
    const std::size_t count = points.size();
    Real sse = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t relative = (
            index + count - static_cast<std::size_t>(cusp_index)) % count;
        const Real phi = static_cast<Real>(direction) * 2 * kPi
            * static_cast<Real>(relative) / static_cast<Real>(count)
            + phase_offset;
        const Real sin_phi = std::sin(phi);
        const Complex unit{std::cos(phi), sin_phi};
        const Complex base = unit - Real(0.5L) * unit * unit;
        const Complex canonical = base * (1 - xi * sin_phi);
        const Complex rotated{
            cosine * canonical.real() - sine * canonical.imag(),
            sine * canonical.real() + cosine * canonical.imag(),
        };
        const Complex residual = center + size * rotated - points[index];
        sse += std::norm(residual);
    }
    return sse;
}

CardioidCandidate polish_cardioid_joint_with_phase(
    CardioidCandidate candidate,
    const std::vector<Complex>& points,
    const CardioidFitOptions& options,
    Real xi_limit) {
    if (!candidate.valid || !(candidate.size > 0) || points.empty()) {
        return candidate;
    }
    const Real sample_step = 2 * kPi / static_cast<Real>(points.size());
    std::array<Real, 6> parameters{{
        candidate.center_x,
        candidate.center_y,
        std::log(candidate.size),
        candidate.angle,
        candidate.xi,
        candidate.phase_offset,
    }};
    Real current_sse = cardioid_sse_with_phase(
        points, candidate.cusp_index, candidate.direction, parameters);
    Real damping = 1e-6L;

    for (int iteration = 0; iteration < std::max(0, options.max_iterations);
         ++iteration) {
        std::array<std::array<Real, 6>, 6> normal{};
        std::array<Real, 6> rhs{};
        const Complex center{parameters[0], parameters[1]};
        const Real size = std::exp(parameters[2]);
        const Real angle = parameters[3];
        const Real xi = parameters[4];
        const Real phase_offset = parameters[5];
        const Real cosine = std::cos(angle);
        const Real sine = std::sin(angle);
        const std::size_t count = points.size();

        auto rotate_fast = [&](Complex value) {
            return Complex{
                cosine * value.real() - sine * value.imag(),
                sine * value.real() + cosine * value.imag(),
            };
        };

        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t relative = (
                index + count - static_cast<std::size_t>(candidate.cusp_index))
                % count;
            const Real phi = static_cast<Real>(candidate.direction) * 2 * kPi
                * static_cast<Real>(relative) / static_cast<Real>(count)
                + phase_offset;
            const Real sin_phi = std::sin(phi);
            const Real cos_phi = std::cos(phi);
            const Complex unit{cos_phi, sin_phi};
            const Complex unit2 = unit * unit;
            const Complex base = unit - Real(0.5L) * unit2;
            const Complex dbase{
                -(sin_phi - std::sin(2 * phi)),
                cos_phi - std::cos(2 * phi),
            };
            const Complex slant = -sin_phi * base;
            const Complex dslant = -cos_phi * base - sin_phi * dbase;
            const Complex canonical = base + xi * slant;
            const Complex dcanonical = dbase + xi * dslant;
            const Complex shape = size * rotate_fast(canonical);
            const Complex xi_derivative = size * rotate_fast(slant);
            const Complex phase_derivative = size * rotate_fast(dcanonical);
            const Complex angle_derivative{-shape.imag(), shape.real()};
            const Complex residual = center + shape - points[index];

            const std::array<Complex, 6> jacobian{{
                Complex{1, 0},
                Complex{0, 1},
                shape,
                angle_derivative,
                xi_derivative,
                phase_derivative,
            }};
            for (std::size_t a = 0; a < 6; ++a) {
                rhs[a] -= std::real(std::conj(jacobian[a]) * residual);
                for (std::size_t b = 0; b < 6; ++b) {
                    normal[a][b] += std::real(
                        std::conj(jacobian[a]) * jacobian[b]);
                }
            }
        }
        for (std::size_t diagonal = 0; diagonal < 6; ++diagonal) {
            normal[diagonal][diagonal] += damping
                * std::max<Real>(1, normal[diagonal][diagonal]);
        }

        std::array<Real, 6> delta{};
        if (!solve_linear<6>(normal, rhs, delta)) break;
        std::array<Real, 6> trial = parameters;
        for (std::size_t index = 0; index < 6; ++index) {
            trial[index] += delta[index];
        }
        trial[3] = normalize_angle(trial[3]);
        trial[4] = std::clamp(trial[4], -xi_limit, xi_limit);
        trial[5] = std::clamp(trial[5], -sample_step, sample_step);
        trial[2] = std::clamp(
            trial[2], parameters[2] - Real(1), parameters[2] + Real(1));

        const Real trial_sse = cardioid_sse_with_phase(
            points, candidate.cusp_index, candidate.direction, trial);
        if (std::isfinite(trial_sse) && trial_sse < current_sse) {
            parameters = trial;
            const Real improvement = current_sse - trial_sse;
            current_sse = trial_sse;
            damping = std::max<Real>(1e-14L, damping * Real(0.3L));
            Real step_norm = 0;
            for (const Real value : delta) step_norm += value * value;
            if (std::sqrt(step_norm) < 2e-13L
                || improvement < 1e-24L * std::max<Real>(1, current_sse)) {
                break;
            }
        } else {
            damping = std::min<Real>(1e12L, damping * Real(10));
        }
    }

    candidate.center_x = parameters[0];
    candidate.center_y = parameters[1];
    candidate.size = std::exp(parameters[2]);
    candidate.angle = normalize_angle(parameters[3]);
    candidate.xi = std::clamp(parameters[4], -xi_limit, xi_limit);
    candidate.phase_offset = std::clamp(
        parameters[5], -sample_step, sample_step);
    const CardioidBasis basis = make_cardioid_basis(
        points.size(),
        static_cast<std::size_t>(candidate.cusp_index),
        candidate.direction,
        candidate.phase_offset);
    evaluate_cardioid_candidate(candidate, points, basis);
    return candidate;
}


Real candidate_acceptance_score(
    const CardioidCandidate& candidate,
    const CardioidFitOptions& options) {
    if (!candidate.valid) return std::numeric_limits<Real>::infinity();
    const Real rms_tolerance = options.rms_tolerance.convert_to<Real>();
    const Real max_tolerance = options.max_error_tolerance.convert_to<Real>();
    if (!(rms_tolerance > 0) || !(max_tolerance > 0)) {
        return std::numeric_limits<Real>::infinity();
    }
    const Real rms_ratio = candidate.rms / rms_tolerance;
    const Real max_ratio = candidate.max_error / max_tolerance;
    // The maximum ratio is the actual acceptance boundary.  The small smooth
    // tie-breaker prevents the stochastic search from wandering along a flat
    // max() ridge when two candidates fail by the same limiting ratio.
    return std::max(rms_ratio, max_ratio)
        + Real(0.02L) * (rms_ratio + max_ratio);
}

bool candidate_confident(
    const CardioidCandidate& candidate,
    const CardioidFitOptions& options,
    Real xi_limit) {
    return candidate.valid
        && candidate.rms <= options.rms_tolerance.convert_to<Real>()
        && candidate.max_error
            <= options.max_error_tolerance.convert_to<Real>()
        && std::abs(candidate.xi) <= xi_limit;
}

std::uint64_t stable_seed(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash ? hash : 0x9e3779b97f4a7c15ULL;
}

class ShakeRng {
public:
    explicit ShakeRng(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next_u64() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    Real uniform() {
        // Open interval (0,1), using the high 53 random bits.
        constexpr Real inverse = Real(1.0L / 9007199254740992.0L);
        return (static_cast<Real>((next_u64() >> 11) + 0.5L)) * inverse;
    }

    Real normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const Real radius = std::sqrt(-2 * std::log(uniform()));
        const Real angle = 2 * kPi * uniform();
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    std::uint64_t state_;
    bool has_spare_ = false;
    Real spare_ = 0;
};

int wrapped_index(int index, int count) {
    index %= count;
    if (index < 0) index += count;
    return index;
}

void retain_elite(
    std::vector<CardioidCandidate>& elite,
    const CardioidCandidate& candidate,
    const CardioidFitOptions& options,
    std::size_t keep) {
    if (!candidate.valid || keep == 0) return;
    elite.push_back(candidate);
    std::sort(
        elite.begin(), elite.end(),
        [&](const CardioidCandidate& left, const CardioidCandidate& right) {
            return candidate_acceptance_score(left, options)
                < candidate_acceptance_score(right, options);
        });
    if (elite.size() > keep) elite.resize(keep);
}

CardioidCandidate refine_cardioid_phase(
    CardioidCandidate candidate,
    const std::vector<Complex>& points,
    const CardioidFitOptions& options,
    Real xi_limit) {
    if (!candidate.valid || points.empty()) return candidate;
    const Real sample_step = 2 * kPi / static_cast<Real>(points.size());
    Real step = Real(0.5L) * sample_step;
    for (int iteration = 0; iteration < 12; ++iteration) {
        CardioidCandidate best = candidate;
        Real best_score = candidate_acceptance_score(best, options);
        for (const Real sign : {Real(-1), Real(1)}) {
            const Real phase = std::clamp(
                candidate.phase_offset + sign * step,
                -sample_step,
                sample_step);
            const CardioidBasis basis = make_cardioid_basis(
                points.size(),
                static_cast<std::size_t>(candidate.cusp_index),
                candidate.direction,
                phase);
            CardioidCandidate trial = linear_cardioid_fit(
                points,
                basis,
                candidate.cusp_index,
                candidate.direction,
                phase,
                candidate.angle,
                xi_limit,
                candidate.size);
            const Real score = candidate_acceptance_score(trial, options);
            if (trial.valid && score < best_score) {
                best = trial;
                best_score = score;
            }
        }
        if (best.cusp_index == candidate.cusp_index
            && best.direction == candidate.direction
            && best.phase_offset == candidate.phase_offset) {
            step *= Real(0.5L);
        } else {
            candidate = best;
        }
        if (step < 1e-10L) break;
    }
    return candidate;
}

CardioidCandidate randomized_cardioid_fallback(
    const CardioidCandidate& deterministic,
    const std::vector<Complex>& points,
    const CardioidFitOptions& options,
    Real xi_limit,
    const std::string& component_id,
    int& trials_run) {
    trials_run = 0;
    if (!deterministic.valid || points.empty()
        || !options.randomized_fallback
        || options.shake_trials <= 0) {
        return deterministic;
    }

    const int point_count = static_cast<int>(points.size());
    const Real sample_step = 2 * kPi / static_cast<Real>(point_count);
    const Real angle_sigma = std::max<Real>(
        0, options.shake_angle_sigma.convert_to<Real>());
    const Real phase_sigma = std::max<Real>(
        0, options.shake_phase_sigma.convert_to<Real>()) * sample_step;
    const Real initial_temperature = std::max<Real>(
        1e-12L, options.shake_temperature.convert_to<Real>());
    const Real final_temperature = std::clamp(
        options.shake_final_temperature.convert_to<Real>(),
        Real(1e-12L), initial_temperature);
    const int cusp_jitter = std::max(0, options.shake_cusp_jitter);
    const std::size_t keep = static_cast<std::size_t>(
        std::max(1, options.shake_keep));

    ShakeRng rng(stable_seed(component_id));
    CardioidCandidate current = deterministic;
    CardioidCandidate best = deterministic;
    Real current_energy = candidate_acceptance_score(current, options);
    Real best_energy = current_energy;
    std::vector<CardioidCandidate> elite;
    elite.reserve(keep + 1);
    retain_elite(elite, best, options, keep);

    for (int trial_index = 0; trial_index < options.shake_trials;
         ++trial_index) {
        ++trials_run;
        const Real fraction = options.shake_trials > 1
            ? static_cast<Real>(trial_index)
                / static_cast<Real>(options.shake_trials - 1)
            : 1;
        const Real temperature = initial_temperature * std::pow(
            final_temperature / initial_temperature, fraction);
        const Real spread = Real(0.18L) + Real(0.82L) * (1 - fraction);

        // Periodic restart from the best state prevents one unlucky accepted
        // excursion from consuming the rest of the bounded trial budget.
        if (trial_index > 0 && trial_index % 96 == 0) {
            current = best;
            current_energy = best_energy;
        }

        int cusp = current.cusp_index;
        int direction = current.direction;
        if (cusp_jitter > 0 && rng.uniform() < Real(0.25L)) {
            const int width = 2 * cusp_jitter + 1;
            const int offset = static_cast<int>(rng.uniform() * width)
                - cusp_jitter;
            cusp = wrapped_index(cusp + offset, point_count);
        }
        if (rng.uniform() < Real(0.03L)) direction = -direction;

        const Real phase = std::clamp(
            current.phase_offset + rng.normal() * phase_sigma * spread,
            -sample_step,
            sample_step);
        const Real angle = normalize_angle(
            current.angle + rng.normal() * angle_sigma * spread);
        const CardioidBasis basis = make_cardioid_basis(
            points.size(), static_cast<std::size_t>(cusp), direction, phase);
        CardioidCandidate proposal = linear_cardioid_fit(
            points,
            basis,
            cusp,
            direction,
            phase,
            angle,
            xi_limit,
            current.size);
        if (!proposal.valid) continue;

        const Real proposal_energy = candidate_acceptance_score(
            proposal, options);
        const Real energy_change = proposal_energy - current_energy;
        const bool accept = energy_change <= 0
            || rng.uniform() < std::exp(std::max<Real>(
                Real(-80), -energy_change / temperature));
        if (accept) {
            current = proposal;
            current_energy = proposal_energy;
        }
        if (proposal_energy < best_energy) {
            best = proposal;
            best_energy = proposal_energy;
            retain_elite(elite, best, options, keep);
        }
    }

    // Local deterministic polishing of only the strongest stochastic states
    // is much cheaper than polishing every random proposal and makes the final
    // result independent of whether the last accepted MCMC state was useful.
    retain_elite(elite, current, options, keep);
    for (CardioidCandidate candidate : elite) {
        CardioidBasis basis = make_cardioid_basis(
            points.size(),
            static_cast<std::size_t>(candidate.cusp_index),
            candidate.direction,
            candidate.phase_offset);
        candidate = refine_cardioid_angle(
            candidate, points, basis, options, xi_limit);
        candidate = refine_cardioid_phase(
            candidate, points, options, xi_limit);
        candidate = polish_cardioid_joint_with_phase(
            candidate, points, options, xi_limit);

        const Real score = candidate_acceptance_score(candidate, options);
        if (candidate.valid && score < best_energy) {
            best = candidate;
            best_energy = score;
        }
        if (candidate_confident(best, options, xi_limit)) break;
    }
    return best;
}

} // namespace

bool CircleFitAnalysis::confident(const CircleFitOptions& options) const {
    return converged
        && fit.center_centered.has_value()
        && fit.radius.has_value()
        && fit.max_error.has_value()
        && fit.rms <= options.rms_tolerance
        && *fit.max_error <= options.max_error_tolerance;
}

CircleFitAnalysis fit_circle(
    const ComponentRecord& component,
    const CircleFitOptions& options) {
    CircleFitAnalysis result;
    const auto& polygon = component.geometry.polygon;
    if (polygon.size() < 3) return result;

    Real scale = 0;
    std::vector<Complex> points;
    points.reserve(polygon.size());
    for (const auto& point : polygon) {
        const Real x = point.re.convert_to<Real>();
        const Real y = point.im.convert_to<Real>();
        if (!std::isfinite(x) || !std::isfinite(y)) return result;
        points.emplace_back(x, y);
        scale = std::max(scale, std::hypot(x, y));
    }
    if (!(scale > 0) || !std::isfinite(scale)) return result;
    for (auto& point : points) point /= scale;

    Real sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    Real bx = 0, by = 0, b1 = 0;
    const Real count = static_cast<Real>(points.size());
    for (const auto point : points) {
        const Real x = point.real();
        const Real y = point.imag();
        const Real q = x * x + y * y;
        sx += 2 * x;
        sy += 2 * y;
        sxx += 4 * x * x;
        syy += 4 * y * y;
        sxy += 4 * x * y;
        bx += 2 * x * q;
        by += 2 * y * q;
        b1 += q;
    }

    std::array<std::array<Real, 3>, 3> matrix{{
        {{sxx, sxy, sx}},
        {{sxy, syy, sy}},
        {{sx, sy, count}},
    }};
    std::array<Real, 3> rhs{{bx, by, b1}};
    std::array<Real, 3> solution{};
    Real center_x = 0;
    Real center_y = 0;
    Real radius = 0;
    if (solve3(matrix, rhs, solution)) {
        center_x = solution[0];
        center_y = solution[1];
        const Real radius_squared = solution[2]
            + center_x * center_x + center_y * center_y;
        if (radius_squared > 0 && std::isfinite(radius_squared)) {
            radius = std::sqrt(radius_squared);
        }
    }
    if (!(radius > 0)) {
        for (const auto point : points) {
            center_x += point.real();
            center_y += point.imag();
        }
        center_x /= count;
        center_y /= count;
        for (const auto point : points) {
            radius += std::abs(point - Complex(center_x, center_y));
        }
        radius /= count;
    }
    if (!(radius > 0) || !std::isfinite(radius)) return result;

    for (int iteration = 0; iteration < std::max(1, options.max_iterations);
         ++iteration) {
        std::array<std::array<Real, 3>, 3> normal{};
        std::array<Real, 3> update_rhs{};
        for (const auto point : points) {
            const Real dx = point.real() - center_x;
            const Real dy = point.imag() - center_y;
            const Real distance = std::hypot(dx, dy);
            if (distance < 1e-30L) continue;
            const std::array<Real, 3> jacobian{{
                -dx / distance,
                -dy / distance,
                -1,
            }};
            const Real residual = distance - radius;
            for (int a = 0; a < 3; ++a) {
                update_rhs[a] += -jacobian[a] * residual;
                for (int b = 0; b < 3; ++b) {
                    normal[a][b] += jacobian[a] * jacobian[b];
                }
            }
        }
        std::array<Real, 3> delta{};
        if (!solve3(normal, update_rhs, delta)) break;
        center_x += delta[0];
        center_y += delta[1];
        radius += delta[2];
        if (!(radius > 0)) radius = 1e-30L;
        if (std::hypot(delta[0], delta[1]) + std::abs(delta[2])
            <= 4e-18L * std::max<Real>(1, radius)) {
            break;
        }
    }

    radius = 0;
    for (const auto point : points) {
        radius += std::abs(point - Complex(center_x, center_y));
    }
    radius /= count;
    if (!(radius > 0) || !std::isfinite(radius)) return result;

    Real squared_error = 0;
    Real maximum_error = 0;
    for (const auto point : points) {
        const Real error = std::abs(
            std::abs(point - Complex(center_x, center_y)) - radius);
        squared_error += error * error;
        maximum_error = std::max(maximum_error, error);
    }
    const Real rms_relative = std::sqrt(squared_error / count) / radius;
    const Real max_relative = maximum_error / radius;
    if (!std::isfinite(rms_relative) || !std::isfinite(max_relative)) {
        return result;
    }

    result.converged = true;
    result.fit.center_centered = ComplexValue{
        catalogue_real(center_x * scale),
        catalogue_real(center_y * scale),
    };
    result.fit.radius = catalogue_real(radius * scale);
    result.fit.rms = catalogue_real(rms_relative);
    result.fit.max_error = catalogue_real(max_relative);
    return result;
}

ClassificationRecord classify_disk_quick(
    const ComponentRecord& component,
    const CircleFitOptions& options) {
    ClassificationRecord classification;
    const CircleFitAnalysis analysis = fit_circle(component, options);
    if (!analysis.confident(options)) return classification;

    classification.shape_class = "disk";
    const CatalogueReal rms_ratio = analysis.fit.rms / options.rms_tolerance;
    const CatalogueReal max_ratio = *analysis.fit.max_error
        / options.max_error_tolerance;
    const CatalogueReal score = std::max(rms_ratio, max_ratio);
    classification.shape_confidence = std::max(
        CatalogueReal("0.5"),
        CatalogueReal(1) - score / 2);
    classification.circle_fit = analysis.fit;
    return classification;
}

bool CardioidFitAnalysis::confident(
    const CardioidFitOptions& options) const {
    return converged
        && fit.center_centered.has_value()
        && fit.size.has_value()
        && fit.max_error.has_value()
        && fit.rms <= options.rms_tolerance
        && *fit.max_error <= options.max_error_tolerance
        && boost::multiprecision::abs(fit.xi) <= options.xi_limit;
}

CardioidFitAnalysis fit_cardioid_slanted(
    const ComponentRecord& component,
    const CardioidFitOptions& options) {
    CardioidFitAnalysis result;
    const auto& polygon = component.geometry.polygon;
    if (polygon.size() < 8) return result;

    std::vector<Complex> points;
    points.reserve(polygon.size());
    Real scale = 0;
    for (const auto& point : polygon) {
        const Real x = point.re.convert_to<Real>();
        const Real y = point.im.convert_to<Real>();
        if (!std::isfinite(x) || !std::isfinite(y)) return result;
        points.emplace_back(x, y);
        scale = std::max(scale, std::hypot(x, y));
    }
    if (!(scale > 0) || !std::isfinite(scale)) return result;

    // Some external polygons may repeat the first point at the end.  The
    // canonical tracer does not, but dropping an exact closure keeps the phase
    // grid uniform for imported records.
    if (points.size() > 8
        && std::abs(points.front() - points.back())
            <= 1e-18L * std::max<Real>(1, scale)) {
        points.pop_back();
    }
    if (points.size() < 8) return result;
    for (auto& point : points) point /= scale;

    std::vector<std::pair<Real, std::size_t>> cusp_candidates;
    cusp_candidates.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        cusp_candidates.push_back({std::abs(points[index]), index});
    }
    const std::size_t candidate_count = std::min<std::size_t>(
        points.size(), static_cast<std::size_t>(
            std::max(1, options.cusp_candidates)));
    std::partial_sort(
        cusp_candidates.begin(),
        cusp_candidates.begin() + candidate_count,
        cusp_candidates.end());

    const Real xi_limit = std::clamp(
        options.xi_limit.convert_to<Real>(), Real(0), Real(0.5L));
    // Rank every cusp/orientation hypothesis with the cheap linear fit first.
    // Only the strongest few receive angle refinement, and only the final
    // winner receives the full joint nonlinear polish.  This matters when the
    // classifier is applied to tens of thousands of polygons.
    std::vector<CardioidCandidate> ranked;
    ranked.reserve(2 * candidate_count);
    for (std::size_t candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index) {
        const std::size_t cusp = cusp_candidates[candidate_index].second;
        Real cusp_distance = std::abs(points[cusp]);
        Real angle = 0;
        if (cusp_distance > 1e-20L) {
            angle = std::arg(points[cusp]);
        } else {
            const auto farthest = std::max_element(
                points.begin(), points.end(),
                [](Complex left, Complex right) {
                    return std::abs(left) < std::abs(right);
                });
            angle = normalize_angle(std::arg(-*farthest));
            cusp_distance = std::max<Real>(1e-6L, std::abs(*farthest) / 3);
        }
        const Real initial_size = std::max<Real>(1e-12L, 2 * cusp_distance);
        for (const int direction : {1, -1}) {
            const CardioidBasis basis = make_cardioid_basis(
                points.size(), cusp, direction);
            CardioidCandidate candidate = linear_cardioid_fit(
                points, basis, static_cast<int>(cusp), direction,
                0, angle, xi_limit, initial_size);
            if (candidate.valid) ranked.push_back(candidate);
        }
    }
    if (ranked.empty()) return result;
    std::sort(
        ranked.begin(), ranked.end(),
        [](const CardioidCandidate& left, const CardioidCandidate& right) {
            return left.score < right.score;
        });

    CardioidCandidate best;
    const std::size_t refine_count = std::min<std::size_t>(4, ranked.size());
    for (std::size_t index = 0; index < refine_count; ++index) {
        CardioidCandidate candidate = ranked[index];
        const CardioidBasis basis = make_cardioid_basis(
            points.size(), static_cast<std::size_t>(candidate.cusp_index),
            candidate.direction);
        candidate = refine_cardioid_angle(
            candidate, points, basis, options, xi_limit);
        if (candidate.valid && candidate.score < best.score) best = candidate;
    }
    if (!best.valid) return result;
    CardioidBasis best_basis = make_cardioid_basis(
        points.size(), static_cast<std::size_t>(best.cusp_index),
        best.direction, best.phase_offset);
    best = polish_cardioid_joint(
        best, points, best_basis, options, xi_limit);
    if (!best.valid) return result;

    const bool deterministic_confident = candidate_confident(
        best, options, xi_limit);
    if (!deterministic_confident
        && options.randomized_fallback
        && options.shake_trials > 0) {
        result.used_randomized_fallback = true;
        int trials_run = 0;
        CardioidCandidate shaken = randomized_cardioid_fallback(
            best,
            points,
            options,
            xi_limit,
            component.id,
            trials_run);
        result.randomized_trials = trials_run;
        if (shaken.valid
            && candidate_acceptance_score(shaken, options)
                < candidate_acceptance_score(best, options)) {
            best = shaken;
        }
        result.rescued_by_randomized_fallback = candidate_confident(
            best, options, xi_limit);
    }

    result.converged = true;
    result.cusp_index = best.cusp_index;
    result.direction = best.direction;
    result.phase_offset = catalogue_real(best.phase_offset);
    result.fit.center_centered = ComplexValue{
        catalogue_real(best.center_x * scale),
        catalogue_real(best.center_y * scale),
    };
    result.fit.size = catalogue_real(best.size * scale);
    result.fit.angle = catalogue_real(best.angle);
    result.fit.xi = catalogue_real(best.xi);
    result.fit.rms = catalogue_real(best.rms);
    result.fit.max_error = catalogue_real(best.max_error);
    return result;
}

ComplexValue cardioid_point(
    const CardioidFitRecord& fit,
    const CatalogueReal& phi) {
    using boost::multiprecision::cos;
    using boost::multiprecision::sin;
    if (!fit.center_centered || !fit.size) {
        throw std::invalid_argument(
            "Cardioid evaluation requires centre and size");
    }
    const CatalogueReal factor = 1 - fit.xi * sin(phi);
    const CatalogueReal x = *fit.size
        * (cos(phi) - CatalogueReal("0.5") * cos(2 * phi)) * factor;
    const CatalogueReal y = *fit.size
        * (sin(phi) - CatalogueReal("0.5") * sin(2 * phi)) * factor;
    const CatalogueReal ca = cos(fit.angle);
    const CatalogueReal sa = sin(fit.angle);
    return ComplexValue{
        fit.center_centered->re + ca * x - sa * y,
        fit.center_centered->im + sa * x + ca * y,
    };
}

} // namespace mandelbrot::shapes
