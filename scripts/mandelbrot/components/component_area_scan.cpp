#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "common/repo_config.hpp"
#include "components/catalogue/component_catalogue.hpp"
#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using Real = long double;
using Complex = std::complex<Real>;
using Clock = std::chrono::steady_clock;

constexpr Real PI = 3.141592653589793238462643383279502884L;
constexpr Real GOLDEN_ANGLE = 2.399963229728653322231555506633613853L;
constexpr int MAX_REAL_ROOT_GRID_POWER = 30;

// -----------------------------------------------------------------------------
// Small utilities
// -----------------------------------------------------------------------------

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        result.push_back(trim(item));
    }
    return result;
}

std::string format_duration(Clock::duration duration) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    std::ostringstream out;
    if (seconds < 60) {
        out << seconds << "s";
    } else if (seconds < 3600) {
        out << (seconds / 60) << ':' << std::setw(2) << std::setfill('0') << (seconds % 60);
    } else {
        out << (seconds / 3600) << ':' << std::setw(2) << std::setfill('0')
            << ((seconds / 60) % 60) << ':' << std::setw(2) << std::setfill('0')
            << (seconds % 60);
    }
    return out.str();
}

std::string real_string(Real value, int precision = std::numeric_limits<Real>::max_digits10) {
    std::ostringstream out;
    out << std::setprecision(precision) << std::scientific << value;
    return out.str();
}

std::string complex_string(
    const Complex& value,
    int precision = std::numeric_limits<Real>::max_digits10
) {
    std::ostringstream out;
    out << std::setprecision(precision) << std::scientific
        << value.real();
    if (!std::signbit(value.imag())) out << '+';
    out << value.imag() << 'i';
    return out.str();
}

bool finite(Real value) {
    return std::isfinite(value);
}

bool finite(const Complex& value) {
    return finite(value.real()) && finite(value.imag());
}

Real safe_abs(const Complex& value) {
    const Real result = std::abs(value);
    return finite(result) ? result : std::numeric_limits<Real>::infinity();
}

Real median(std::vector<Real> values) {
    if (values.empty()) {
        return std::numeric_limits<Real>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2;
}

Real quantile(std::vector<Real> values, Real q) {
    if (values.empty()) {
        return std::numeric_limits<Real>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const Real position = q * static_cast<Real>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(position));
    const auto hi = static_cast<std::size_t>(std::ceil(position));
    if (lo == hi) {
        return values[lo];
    }
    const Real fraction = position - static_cast<Real>(lo);
    return values[lo] * (1 - fraction) + values[hi] * fraction;
}

void atomic_replace(const fs::path& temporary, const fs::path& target) {
    std::error_code error;
    fs::rename(temporary, target, error);
    if (!error) {
        return;
    }
    fs::remove(target, error);
    error.clear();
    fs::rename(temporary, target, error);
    if (error) {
        throw std::runtime_error("Could not replace " + target.string() + ": " + error.message());
    }
}

fs::path executable_parent_or_cwd(const char* argv0) {
    try {
        fs::path path(argv0 ? argv0 : "");
        if (!path.empty() && path.has_parent_path()) {
            if (path.is_relative()) {
                path = fs::absolute(path);
            }
            return fs::weakly_canonical(path.parent_path());
        }
    } catch (...) {
    }
    return fs::current_path();
}

fs::path find_project_root_from(fs::path start) {
    try {
        start = fs::weakly_canonical(start);
    } catch (...) {
        start = fs::absolute(start);
    }

    fs::path current = start;
    while (true) {
        if (fs::exists(current / ".git") || fs::exists(current / ".root")) {
            return current;
        }
        if (!current.has_parent_path() || current == current.parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    return start;
}

void replace_all_inplace(
    std::string& value,
    const std::string& from,
    const std::string& to
) {
    if (from.empty()) {
        return;
    }
    std::size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

fs::path canonical_or_absolute(fs::path path) {
    try {
        return fs::weakly_canonical(path);
    } catch (...) {
        return fs::absolute(path);
    }
}

fs::path expand_path_tokens(
    std::string value,
    const fs::path& code_root,
    const fs::path& project_root,
    const fs::path& data_root
) {
    replace_all_inplace(value, "${data_root}", data_root.string());
    replace_all_inplace(value, "$data_root", data_root.string());
    replace_all_inplace(value, "${code_root}", code_root.string());
    replace_all_inplace(value, "$code_root", code_root.string());
    replace_all_inplace(value, "${project_root}", project_root.string());
    replace_all_inplace(value, "$project_root", project_root.string());

    fs::path path(value);
    if (path.is_relative()) {
        path = code_root / path;
    }
    return canonical_or_absolute(path);
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct Config {
    int period = 10;
    int period_start = 1;
    std::vector<Real> radii{0.9L, 0.99L, 0.999L, 0.9999L, 0.99999L};

    // "auto" means the directory containing the executable and the nearest
    // parent containing .git or .root, respectively.
    std::string code_root = "auto";
    std::string project_root = "auto";
    fs::path output_dir = "$data_root/component_catalogue/exports";
    fs::path catalogue_root = "$data_root/component_catalogue";
    fs::path run_dir;
    std::string run_name = "default";
    bool resume = false;
    bool verify_catalogue = false;
    bool use_conjugate_symmetry = true;
    bool compute_areas = true;

    // Center isolation workers. 0 means hardware_concurrency().
    unsigned threads = 0;

    // Area tracing workers. 0 inherits the resolved center-worker count.
    // Components are parallel; radii and angles remain sequential inside one
    // component so radial continuation stays on the trusted branch.
    unsigned area_threads = 0;
    // Completed component rows are written to immutable atomic batch files by
    // a dedicated writer thread. The canonical CSV and summaries are rebuilt
    // only once, after the area stage finishes successfully.
    int area_checkpoint_components = 250;
    int area_checkpoint_seconds = 20;

    // Interactive progress UI for root solving and parallel area tracing.
    // progress_style: bars | lines | none
    // progress_screen: alternate | normal
    bool progress = true;
    std::string progress_style = "bars";
    std::string progress_screen = "alternate";
    int progress_bar_width = 30;
    int progress_refresh_ms = 250;

    Real root_tolerance = 1.0e-13L;
    int root_max_iterations = 1200;
    Real root_max_step = 0.25L;
    Real root_bound = 2.05L;
    int root_checkpoint_every = 10;
    int root_progress_every = 5;
    bool reset_root_checkpoint = false;

    // Optional uniform-grid acceleration for the Aberth repulsion sum.
    // Nearby cells are evaluated exactly. Sufficiently distant cells are
    // represented by their centroid and, optionally, their second complex
    // moment. root_grid_y=0 chooses it from the root-cloud aspect ratio.
    bool root_grid_enabled = false;
    int root_grid_x = 16;
    int root_grid_y = 0;
    int root_grid_near_cells = 1;
    Real root_grid_opening_angle = 0.35L;
    int root_grid_multipole_order = 2;  // 0 or 2
    int root_grid_far_min_roots = 4;

    // Adaptive Barnes-Hut quadtree used by the large conjugate-symmetric solve.
    // Nearby leaves are summed exactly; sufficiently distant nodes use a
    // centroid plus second-moment multipole approximation.
    bool root_tree_enabled = true;
    int root_tree_leaf_size = 48;
    int root_tree_max_depth = 32;
    Real root_tree_theta_initial = 0.65L;
    Real root_tree_theta_final = 0.24L;
    Real root_tree_tighten_start_delta = 1.0e-3L;
    Real root_tree_tighten_end_delta = 1.0e-9L;
    int root_tree_multipole_order = 2;  // 0 or 2

    // Use the preceding period's certified centers as a density scaffold for
    // the next upper-half root cloud. The remaining seeds retain the robust
    // low-discrepancy ellipse initialization.
    bool root_warm_start_previous_period = true;
    Real root_warm_start_fraction = 0.95L;
    Real root_warm_start_jitter = 0.18L;

    // Hybrid global/local solve. Roots that are well separated relative to
    // their exact-period Newton correction are polished early and frozen.
    // Aberth-Ehrlich then continues only on unresolved clusters, while frozen
    // roots remain in the repulsion field.
    bool root_early_polish_enabled = true;
    int root_isolation_check_every = 25;
    Real root_early_polish_start_delta = 1.0e-3L;
    Real root_early_polish_ratio = 0.03L;
    int root_early_polish_steps = 8;
    Real root_early_polish_tolerance = 5.0e-14L;
    Real root_cluster_phase_fraction = 0.08L;
    Real root_cluster_theta = 0.18L;
    int root_exact_active_limit = 2048;
    // Once the unresolved set enters the cluster phase, use a much gentler
    // step cap. This prevents difficult roots from bouncing indefinitely at
    // the global root_max_step limit while the tail is being isolated.
    Real root_cluster_max_step = 0.02L;

    // Exact all-pairs iterations are gradually introduced after the maximum
    // correction drops below root_grid_exact_start_delta. Zero means
    // sqrt(root_tolerance). The cadence transitions logarithmically from
    // root_grid_exact_start_every to root_grid_exact_steps_every. Setting
    // root_grid_exact_steps_every=1 recovers exact O(N^2) Aberth on every step.
    Real root_grid_exact_start_delta = 0;
    int root_grid_exact_start_every = 100;
    int root_grid_exact_steps_every = 10;

    // Solve the real roots first, then move only upper-half-plane roots while
    // representing their lower-half-plane conjugates implicitly.
    bool root_half_plane_symmetry = true;
    Real root_upper_axis_step_fraction = 0.5L;

    // Certified real-axis prepass for raw F_n(c)=f_c^n(0) on [-2,-5/4].
    // A nested quadratic-dyadic sign grid concentrates samples near c=-2,
    // where the real roots cluster, and is refined until the theorem-backed
    // raw count is reached. Near the end, derivative sign changes recover
    // close root pairs around extrema without doubling the whole grid again.
    int real_root_initial_grid_power = 12;
    int real_root_samples_per_expected = 4;
    int real_root_critical_recovery_power = 20;
    int real_root_max_grid_power = 28;
    int real_root_polish_iterations = 100;
    Real real_root_bracket_tolerance = 1.0e-18L;

    int center_polish_iterations = 80;
    Real center_residual_tolerance = 1.0e-12L;
    Real center_duplicate_tolerance = 1.0e-10L;

    int theta_start = 64;
    int theta_max = 4096;
    Real area_rtol = 1.0e-8L;
    Real area_atol = 1.0e-13L;

    int newton_max_iterations = 50;
    Real newton_tolerance = 1.0e-14L;
    int continuation_max_depth = 14;
    Real continuation_max_step = 0.05L;
    Real branch_jump_factor = 16.0L;
    Real exact_period_tolerance = 1.0e-10L;

    // Selective arbitrary-precision continuation. The fast long-double path is
    // retained for ordinary components. A whole component is retraced in MP
    // when the estimated c-displacement has fewer than mp_guard_digits decimal
    // digits above the local long-double ulp, or whenever the long-double trace
    // fails. Boost.Multiprecision is header-only; no extra linker flags needed.
    bool mp_fallback = true;
    bool mp_proactive = true;
    int mp_guard_digits = 8;
    int mp_extra_digits = 28;
    int mp_min_dps = 50;
    int mp_max_dps = 200;
    int mp_newton_max_iterations = 80;
    int mp_continuation_max_depth = 24;
    Real mp_continuation_max_step = 0.02L;

    // Optional disposable browser export. Canonical geometry is always stored
    // without the demo cutoff; the export is a catalogue query performed later.
    bool export_atlas_geometry = false;
    fs::path atlas_geometry_file =
        "$data_root/component_catalogue/exports/atlas_components.json";
    Real demo_min_area = 1.0e-10L;
    Real atlas_area_rho = 0.99999L;
    Real atlas_polygon_rho = 0.9995L;
    int atlas_polygon_points = 192;
    // Completed canonical geometry is upserted by a dedicated SQLite writer.
    // Either threshold triggers a transaction, bounding interruption loss while
    // leaving the tracing workers free to continue with the dynamic job queue.
    int geometry_checkpoint_components = 250;
    int geometry_checkpoint_seconds = 20;
};

void resolve_config_paths(Config& config, const char* argv0) {
    fs::path code_root;
    if (lower(trim(config.code_root)) == "auto") {
        code_root = executable_parent_or_cwd(argv0);
    } else {
        code_root = canonical_or_absolute(fs::path(config.code_root));
    }

    fs::path project_root;
    if (lower(trim(config.project_root)) == "auto") {
        project_root = find_project_root_from(code_root);
    } else {
        std::string raw_project_root = config.project_root;
        replace_all_inplace(raw_project_root, "${code_root}", code_root.string());
        replace_all_inplace(raw_project_root, "$code_root", code_root.string());

        fs::path configured_project_root(raw_project_root);
        if (configured_project_root.is_relative()) {
            configured_project_root = code_root / configured_project_root;
        }
        project_root = canonical_or_absolute(configured_project_root);
    }

    config.code_root = code_root.string();
    config.project_root = project_root.string();
    const fs::path data_root = mandelbrot::repo::resolve_data_root({}, project_root);
    config.output_dir = expand_path_tokens(
        config.output_dir.string(),
        code_root,
        project_root,
        data_root
    );
    config.atlas_geometry_file = expand_path_tokens(
        config.atlas_geometry_file.string(),
        code_root,
        project_root,
        data_root
    );
    config.catalogue_root = expand_path_tokens(
        config.catalogue_root.string(), code_root, project_root, data_root
    );
    if (config.run_dir.empty()) config.run_dir = config.catalogue_root / "runs/area_scan/default";
    else {
        config.run_dir = expand_path_tokens(
            config.run_dir.string(), code_root, project_root, data_root
        );
    }
}

Config read_repository_config(const fs::path& path, const char* argv0) {
    const auto repo = mandelbrot::repo::RepoConfig::load(path, executable_parent_or_cwd(argv0));
    Config config;
    config.code_root = repo.code_root().string();
    config.project_root = repo.project_root().string();
    config.catalogue_root = repo.path("paths.catalogue_root");
    config.output_dir = config.catalogue_root / "exports";
    config.run_name = repo.string("component_area_scan.run_name", "default");
    config.run_dir = config.catalogue_root / "runs/area_scan" / config.run_name;
    config.atlas_geometry_file = config.output_dir /
        repo.string("component_area_scan.atlas_geometry_export", "atlas_components.json");
    config.threads = repo.threads();
    config.area_threads = config.threads;

    auto integer = [&](const char* key, int fallback) {
        return repo.integer(std::string("component_area_scan.") + key, fallback);
    };
    auto number = [&](const char* key, Real fallback) {
        return static_cast<Real>(repo.number(std::string("component_area_scan.") + key, fallback));
    };
    auto boolean = [&](const char* key, bool fallback) {
        return repo.boolean(std::string("component_area_scan.") + key, fallback);
    };
    auto string = [&](const char* key, const std::string& fallback) {
        return repo.string(std::string("component_area_scan.") + key, fallback);
    };

    config.period = integer("period", config.period);
    config.period_start = integer("period_start", config.period_start);
    if (repo.find("component_area_scan.radii")) {
        config.radii.clear();
        for (const auto value : repo.number_array("component_area_scan.radii")) {
            config.radii.push_back(static_cast<Real>(value));
        }
    }
    config.resume = boolean("resume", config.resume);
    config.use_conjugate_symmetry = boolean("use_conjugate_symmetry", config.use_conjugate_symmetry);
    config.compute_areas = boolean("compute_areas", config.compute_areas);
    config.area_checkpoint_components = integer("area_checkpoint_components", config.area_checkpoint_components);
    config.area_checkpoint_seconds = integer("area_checkpoint_seconds", config.area_checkpoint_seconds);
    config.progress = boolean("progress", repo.boolean("runtime.progress.enabled", config.progress));
    config.progress_style = string("progress_style", repo.string("runtime.progress.style", config.progress_style));
    config.progress_screen = string("progress_screen", repo.string("runtime.progress.screen", config.progress_screen));
    config.progress_bar_width = integer("progress_bar_width", repo.integer("runtime.progress.bar_width", config.progress_bar_width));
    config.progress_refresh_ms = integer("progress_refresh_ms", repo.integer("runtime.progress.refresh_ms", config.progress_refresh_ms));
    config.root_tolerance = number("root_tolerance", config.root_tolerance);
    config.root_max_iterations = integer("root_max_iterations", config.root_max_iterations);
    config.root_max_step = number("root_max_step", config.root_max_step);
    config.root_bound = number("root_bound", config.root_bound);
    config.root_checkpoint_every = integer("root_checkpoint_every", config.root_checkpoint_every);
    config.root_progress_every = integer("root_progress_every", config.root_progress_every);
    config.reset_root_checkpoint = boolean("reset_root_checkpoint", config.reset_root_checkpoint);
    config.root_grid_enabled = boolean("root_grid_enabled", config.root_grid_enabled);
    config.root_grid_x = integer("root_grid_x", config.root_grid_x);
    config.root_grid_y = integer("root_grid_y", config.root_grid_y);
    config.root_grid_near_cells = integer("root_grid_near_cells", config.root_grid_near_cells);
    config.root_grid_opening_angle = number(
        "root_grid_opening_angle", config.root_grid_opening_angle);
    config.root_grid_multipole_order = integer(
        "root_grid_multipole_order", config.root_grid_multipole_order);
    config.root_grid_far_min_roots = integer(
        "root_grid_far_min_roots", config.root_grid_far_min_roots);
    config.root_tree_enabled = boolean(
        "root_tree_enabled", config.root_tree_enabled);
    config.root_tree_leaf_size = integer(
        "root_tree_leaf_size", config.root_tree_leaf_size);
    config.root_tree_max_depth = integer(
        "root_tree_max_depth", config.root_tree_max_depth);
    config.root_tree_theta_initial = number(
        "root_tree_theta_initial", config.root_tree_theta_initial);
    config.root_tree_theta_final = number(
        "root_tree_theta_final", config.root_tree_theta_final);
    config.root_tree_tighten_start_delta = number(
        "root_tree_tighten_start_delta", config.root_tree_tighten_start_delta);
    config.root_tree_tighten_end_delta = number(
        "root_tree_tighten_end_delta", config.root_tree_tighten_end_delta);
    config.root_tree_multipole_order = integer(
        "root_tree_multipole_order", config.root_tree_multipole_order);
    config.root_warm_start_previous_period = boolean(
        "root_warm_start_previous_period", config.root_warm_start_previous_period);
    config.root_warm_start_fraction = number(
        "root_warm_start_fraction", config.root_warm_start_fraction);
    config.root_warm_start_jitter = number(
        "root_warm_start_jitter", config.root_warm_start_jitter);
    config.root_early_polish_enabled = boolean(
        "root_early_polish_enabled", config.root_early_polish_enabled);
    config.root_isolation_check_every = integer(
        "root_isolation_check_every", config.root_isolation_check_every);
    config.root_early_polish_start_delta = number(
        "root_early_polish_start_delta", config.root_early_polish_start_delta);
    config.root_early_polish_ratio = number(
        "root_early_polish_ratio", config.root_early_polish_ratio);
    config.root_early_polish_steps = integer(
        "root_early_polish_steps", config.root_early_polish_steps);
    config.root_early_polish_tolerance = number(
        "root_early_polish_tolerance", config.root_early_polish_tolerance);
    config.root_cluster_phase_fraction = number(
        "root_cluster_phase_fraction", config.root_cluster_phase_fraction);
    config.root_cluster_theta = number(
        "root_cluster_theta", config.root_cluster_theta);
    config.root_exact_active_limit = integer(
        "root_exact_active_limit", config.root_exact_active_limit);
    config.root_cluster_max_step = number(
        "root_cluster_max_step", config.root_cluster_max_step);
    config.root_grid_exact_start_delta = number(
        "root_grid_exact_start_delta", config.root_grid_exact_start_delta);
    config.root_grid_exact_start_every = integer(
        "root_grid_exact_start_every", config.root_grid_exact_start_every);
    config.root_grid_exact_steps_every = integer(
        "root_grid_exact_steps_every", config.root_grid_exact_steps_every);
    config.root_half_plane_symmetry = boolean(
        "root_half_plane_symmetry", config.root_half_plane_symmetry);
    config.root_upper_axis_step_fraction = number(
        "root_upper_axis_step_fraction", config.root_upper_axis_step_fraction);
    config.real_root_initial_grid_power = integer(
        "real_root_initial_grid_power", config.real_root_initial_grid_power);
    config.real_root_samples_per_expected = integer(
        "real_root_samples_per_expected", config.real_root_samples_per_expected);
    config.real_root_critical_recovery_power = integer(
        "real_root_critical_recovery_power", config.real_root_critical_recovery_power);
    config.real_root_max_grid_power = integer(
        "real_root_max_grid_power", config.real_root_max_grid_power);
    config.real_root_polish_iterations = integer(
        "real_root_polish_iterations", config.real_root_polish_iterations);
    config.real_root_bracket_tolerance = number(
        "real_root_bracket_tolerance", config.real_root_bracket_tolerance);
    config.center_polish_iterations = integer("center_polish_iterations", config.center_polish_iterations);
    config.center_residual_tolerance = number("center_residual_tolerance", config.center_residual_tolerance);
    config.center_duplicate_tolerance = number("center_duplicate_tolerance", config.center_duplicate_tolerance);
    config.theta_start = integer("theta_start", config.theta_start);
    config.theta_max = integer("theta_max", config.theta_max);
    config.area_rtol = number("area_rtol", config.area_rtol);
    config.area_atol = number("area_atol", config.area_atol);
    config.newton_max_iterations = integer("newton_max_iterations", config.newton_max_iterations);
    config.newton_tolerance = number("newton_tolerance", config.newton_tolerance);
    config.continuation_max_depth = integer("continuation_max_depth", config.continuation_max_depth);
    config.continuation_max_step = number("continuation_max_step", config.continuation_max_step);
    config.branch_jump_factor = number("branch_jump_factor", config.branch_jump_factor);
    config.exact_period_tolerance = number("exact_period_tolerance", config.exact_period_tolerance);
    config.mp_fallback = boolean("mp_fallback", config.mp_fallback);
    config.mp_proactive = boolean("mp_proactive", config.mp_proactive);
    config.mp_guard_digits = integer("mp_guard_digits", config.mp_guard_digits);
    config.mp_extra_digits = integer("mp_extra_digits", config.mp_extra_digits);
    config.mp_min_dps = integer("mp_min_dps", config.mp_min_dps);
    config.mp_max_dps = integer("mp_max_dps", config.mp_max_dps);
    config.mp_newton_max_iterations = integer("mp_newton_max_iterations", config.mp_newton_max_iterations);
    config.mp_continuation_max_depth = integer("mp_continuation_max_depth", config.mp_continuation_max_depth);
    config.mp_continuation_max_step = number("mp_continuation_max_step", config.mp_continuation_max_step);
    config.export_atlas_geometry = boolean("export_atlas_geometry", config.export_atlas_geometry);
    config.demo_min_area = static_cast<Real>(
        repo.number("demo.atlas.components.min_area", config.demo_min_area));
    config.atlas_area_rho = number("atlas_area_rho", config.atlas_area_rho);
    config.atlas_polygon_rho = number("atlas_polygon_rho", config.atlas_polygon_rho);
    config.atlas_polygon_points = integer("atlas_polygon_points", config.atlas_polygon_points);
    config.geometry_checkpoint_components = integer(
        "geometry_checkpoint_components",
        config.geometry_checkpoint_components);
    config.geometry_checkpoint_seconds = integer(
        "geometry_checkpoint_seconds", config.geometry_checkpoint_seconds);
    if (config.area_checkpoint_components < 1) {
        throw std::runtime_error(
            "area_checkpoint_components must be at least 1");
    }
    if (config.area_checkpoint_seconds < 1
        || config.area_checkpoint_seconds > 600) {
        throw std::runtime_error(
            "area_checkpoint_seconds must lie in [1,600] so completed "
            "area work is never held uncommitted for more than ten minutes");
    }
    if (config.geometry_checkpoint_components < 1) {
        throw std::runtime_error(
            "geometry_checkpoint_components must be at least 1");
    }
    if (config.geometry_checkpoint_seconds < 1
        || config.geometry_checkpoint_seconds > 600) {
        throw std::runtime_error(
            "geometry_checkpoint_seconds must lie in [1,600] so completed "
            "geometry work is never held uncommitted for more than ten minutes");
    }
    resolve_config_paths(config, argv0);
    return config;
}

// -----------------------------------------------------------------------------
// Parallel area progress UI
// -----------------------------------------------------------------------------

static std::mutex g_print_mutex;

struct AreaProgressSlot {
    bool active = false;
    int period = 0;
    int representative_job_index = 0;
    int representative_job_count = 0;
    int component_index = -1;
    int radius_index = 0;   // 1-based while active
    int radius_count = 0;
    Real rho = 0;
    Real fraction = 0;
    int theta_points = 0;
    Real area = std::numeric_limits<Real>::quiet_NaN();
    std::string status;
};

static std::mutex g_area_progress_mutex;
static std::vector<AreaProgressSlot> g_area_progress_slots;
static std::atomic<bool> g_area_monitor_stop{false};
static std::atomic<bool> g_alt_screen_active{false};
static volatile std::sig_atomic_t g_alt_screen_active_signal = 0;

bool stderr_is_terminal() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(STDERR_FILENO) != 0;
#else
    return true;
#endif
}

bool stdout_is_terminal() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(STDOUT_FILENO) != 0;
#else
    return true;
#endif
}

void leave_alternate_screen() {
    if (!g_alt_screen_active.exchange(false)) return;
    g_alt_screen_active_signal = 0;
    std::cerr << "\033[?25h\033[?1049l";
    std::cerr.flush();
}

void enter_alternate_screen() {
    if (g_alt_screen_active.exchange(true)) return;
    g_alt_screen_active_signal = 1;
    std::cerr << "\033[?1049h\033[?25l\033[H\033[2J";
    std::cerr.flush();
}

class AlternateScreenGuard {
public:
    explicit AlternateScreenGuard(bool enabled)
        : active_(enabled) {
        if (active_) enter_alternate_screen();
    }

    AlternateScreenGuard(const AlternateScreenGuard&) = delete;
    AlternateScreenGuard& operator=(const AlternateScreenGuard&) = delete;

    ~AlternateScreenGuard() {
        close();
    }

    bool active() const {
        return active_;
    }

    void close() {
        if (!active_) return;
        leave_alternate_screen();
        active_ = false;
    }

private:
    bool active_ = false;
};

#if defined(__unix__) || defined(__APPLE__)
void progress_signal_cleanup_handler(int signal_number) {
    if (g_alt_screen_active_signal) {
        const char sequence[] = "\033[?25h\033[?1049l";
        (void)!write(STDERR_FILENO, sequence, sizeof(sequence) - 1);
    }
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}
#endif

void install_terminal_cleanup_handlers() {
    std::atexit(leave_alternate_screen);
#if defined(__unix__) || defined(__APPLE__)
    std::signal(SIGINT, progress_signal_cleanup_handler);
    std::signal(SIGTERM, progress_signal_cleanup_handler);
#endif
}

int terminal_width_columns() {
#if defined(__unix__) || defined(__APPLE__)
    struct winsize size{};
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
        return static_cast<int>(size.ws_col);
    }
#endif
    if (const char* columns = std::getenv("COLUMNS")) {
        try {
            const int parsed = std::stoi(columns);
            if (parsed > 0) return parsed;
        } catch (...) {
        }
    }
    return 120;
}

std::string fit_to_width(std::string value, int width) {
    if (width <= 0) return "";
    if (static_cast<int>(value.size()) <= width) return value;
    if (width <= 3) return value.substr(0, static_cast<std::size_t>(width));
    return value.substr(0, static_cast<std::size_t>(width - 3)) + "...";
}

std::string progress_bar(Real fraction, int width) {
    width = std::max(1, width);
    fraction = std::clamp(fraction, static_cast<Real>(0), static_cast<Real>(1));
    int filled = static_cast<int>(std::llround(fraction * static_cast<Real>(width)));
    filled = std::clamp(filled, 0, width);
    return std::string(static_cast<std::size_t>(filled), '#')
         + std::string(static_cast<std::size_t>(width - filled), '-');
}

std::string compact_real(Real value, int precision = 4) {
    std::ostringstream output;
    output << std::setprecision(precision) << std::defaultfloat
           << static_cast<long double>(value);
    return output.str();
}

std::string percent_string(Real fraction) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(1)
           << static_cast<double>(100 * std::clamp(
                  fraction, static_cast<Real>(0), static_cast<Real>(1)))
           << '%';
    return output.str();
}

struct FreezeProgressSample {
    int iteration = 0;
    std::size_t total_frozen = 0;
};

struct FreezeProjection {
    bool ready = false;
    std::size_t interval_count = 0;
    Real latest_rate = 0;
    Real smoothed_rate = 0;
    Real acceleration = 0;
    Real remaining_iterations = 0;
    std::string model = "warming up";
};

struct RootProgressDashboardState {
    int period = 0;
    int start_iteration = 0;
    int iteration = 0;
    int maximum_iterations = 0;
    int last_checkpoint_iteration = 0;
    int checkpoint_every = 0;
    std::size_t active_roots = 0;
    std::size_t total_roots = 0;
    std::size_t fixed_real_roots = 0;
    std::size_t full_root_count = 0;
    std::size_t total_frozen = 0;
    int latest_polish_iteration = 0;
    std::size_t latest_polish_moved = 0;
    std::size_t latest_polish_frozen = 0;
    FreezeProjection freeze_projection;
    Real maximum_correction = std::numeric_limits<Real>::infinity();
    Real q99_correction = std::numeric_limits<Real>::infinity();
    Real median_correction = std::numeric_limits<Real>::infinity();
    Real initial_maximum_correction = std::numeric_limits<Real>::infinity();
    Real target_correction = 0;
    Real step_cap = 0;
    bool cluster_phase = false;
    bool exact_step = false;
    bool converged = false;
    std::string interaction_mode = "initializing";
    std::optional<std::pair<std::size_t, Complex>> worst_root;
    Clock::time_point started = Clock::now();
};

std::string dashboard_real(Real value, int precision = 4) {
    return finite(value) ? real_string(value, precision) : "--";
}

Real logarithmic_correction_progress(
    Real initial,
    Real current,
    Real target
) {
    if (!(target > 0) || !finite(current)) return 0;
    if (current <= target) return 1;
    if (!finite(initial) || !(initial > target)) return 0;
    const Real denominator = std::log10(initial / target);
    if (!(denominator > 0)) return 0;
    const Real progress =
        std::log10(initial / std::max(current, target)) / denominator;
    return std::clamp(progress, static_cast<Real>(0), static_cast<Real>(1));
}

FreezeProjection estimate_freeze_projection(
    const std::vector<FreezeProgressSample>& history,
    std::size_t remaining_active
) {
    struct RateSample {
        Real midpoint = 0;
        Real rate = 0;
    };

    FreezeProjection result;
    if (remaining_active == 0) {
        result.ready = true;
        result.remaining_iterations = 0;
        result.model = "complete";
        return result;
    }

    std::vector<RateSample> rates;
    rates.reserve(history.size());
    for (std::size_t i = 1; i < history.size(); ++i) {
        const int delta_iterations =
            history[i].iteration - history[i - 1].iteration;
        if (delta_iterations <= 0
            || history[i].total_frozen
                < history[i - 1].total_frozen) {
            continue;
        }
        const std::size_t delta_frozen =
            history[i].total_frozen - history[i - 1].total_frozen;
        rates.push_back({
            static_cast<Real>(
                history[i].iteration + history[i - 1].iteration) / 2,
            static_cast<Real>(delta_frozen)
                / static_cast<Real>(delta_iterations),
        });
    }

    constexpr std::size_t MAX_RATE_INTERVALS = 16;
    if (rates.size() > MAX_RATE_INTERVALS) {
        rates.erase(
            rates.begin(),
            rates.begin()
                + static_cast<std::vector<RateSample>::difference_type>(
                    rates.size() - MAX_RATE_INTERVALS));
    }
    result.interval_count = rates.size();
    if (!rates.empty()) result.latest_rate = rates.back().rate;

    constexpr std::size_t MIN_RATE_INTERVALS = 6;
    if (rates.size() < MIN_RATE_INTERVALS) {
        result.model = "warming up "
            + std::to_string(rates.size()) + '/'
            + std::to_string(MIN_RATE_INTERVALS) + " intervals";
        return result;
    }

    const std::size_t recent_count =
        std::min<std::size_t>(6, rates.size());
    std::vector<Real> recent_rates;
    recent_rates.reserve(recent_count);
    for (std::size_t i = rates.size() - recent_count;
         i < rates.size();
         ++i) {
        recent_rates.push_back(rates[i].rate);
    }
    result.smoothed_rate = median(std::move(recent_rates));
    if (!(result.smoothed_rate > 0)
        || !finite(result.smoothed_rate)) {
        result.model = "recent median freeze rate is zero";
        return result;
    }

    const std::size_t block =
        std::min<std::size_t>(6, rates.size() / 2);
    std::vector<Real> older_rates;
    std::vector<Real> newer_rates;
    older_rates.reserve(block);
    newer_rates.reserve(block);
    Real older_midpoint = 0;
    Real newer_midpoint = 0;
    const std::size_t older_begin = rates.size() - 2 * block;
    const std::size_t newer_begin = rates.size() - block;
    for (std::size_t i = 0; i < block; ++i) {
        const auto& older = rates[older_begin + i];
        const auto& newer = rates[newer_begin + i];
        older_rates.push_back(older.rate);
        newer_rates.push_back(newer.rate);
        older_midpoint += older.midpoint;
        newer_midpoint += newer.midpoint;
    }
    older_midpoint /= static_cast<Real>(block);
    newer_midpoint /= static_cast<Real>(block);
    const Real midpoint_delta = newer_midpoint - older_midpoint;
    if (midpoint_delta > 0) {
        result.acceleration =
            (median(std::move(newer_rates))
             - median(std::move(older_rates)))
            / midpoint_delta;
    }

    const Real remaining = static_cast<Real>(remaining_active);
    const Real constant_rate_projection =
        remaining / result.smoothed_rate;
    result.remaining_iterations = constant_rate_projection;
    result.model = "median freeze rate";

    // Apply the smoothed second difference only over a horizon no longer than
    // the observed rate history, then continue at the projected rate. A raw
    // quadratic extrapolation all the way to completion is far too sensitive:
    // even a small local acceleration becomes absurd over tens of thousands
    // of iterations.
    const Real acceleration = result.acceleration;
    const Real observed_span =
        rates.back().midpoint - rates.front().midpoint;
    const Real trend_horizon = std::min(
        constant_rate_projection,
        std::max<Real>(1, observed_span));
    const Real projected_rate_change =
        std::abs(acceleration) * trend_horizon;
    if (finite(acceleration)
        && projected_rate_change
            >= 0.05L * result.smoothed_rate) {
        const Real projected_rate =
            result.smoothed_rate + acceleration * trend_horizon;
        const Real projected_frozen =
            result.smoothed_rate * trend_horizon
            + 0.5L * acceleration
                * trend_horizon * trend_horizon;
        Real bounded_projection =
            std::numeric_limits<Real>::infinity();
        if (projected_frozen >= remaining) {
            const Real discriminant =
                result.smoothed_rate * result.smoothed_rate
                + 2 * acceleration * remaining;
            if (discriminant > 0) {
                bounded_projection =
                    2 * remaining
                    / (result.smoothed_rate
                       + std::sqrt(discriminant));
            }
        } else if (projected_rate
                       >= 0.25L * result.smoothed_rate
                   && projected_rate
                       <= 4.0L * result.smoothed_rate
                   && projected_frozen >= 0) {
            bounded_projection =
                trend_horizon
                + (remaining - projected_frozen) / projected_rate;
        }

        if (finite(bounded_projection)
            && bounded_projection > 0
            && bounded_projection
                >= 0.25L * constant_rate_projection
            && bounded_projection
                <= 4.0L * constant_rate_projection) {
            result.remaining_iterations = bounded_projection;
            result.model = "median rate + bounded acceleration";
        } else {
            result.model = "median rate (acceleration unstable)";
        }
    }

    result.ready = finite(result.remaining_iterations)
        && result.remaining_iterations >= 0;
    return result;
}

std::string dashboard_progress_bar(
    const Config& config,
    const std::string& label,
    Real fraction,
    const std::string& suffix,
    int width
) {
    const std::string percentage = percent_string(fraction);
    const int fixed_width = static_cast<int>(label.size())
        + 4 + static_cast<int>(percentage.size())
        + 1 + static_cast<int>(suffix.size());
    const int bar_width = std::max(
        6,
        std::min(
            config.progress_bar_width,
            std::max(6, width - fixed_width)));
    return fit_to_width(
        label + " [" + progress_bar(fraction, bar_width) + "] "
            + percentage + " " + suffix,
        width);
}

void render_root_progress_dashboard(
    const Config& config,
    const RootProgressDashboardState& state
) {
    const int width = std::max(20, terminal_width_columns() - 1);
    const std::size_t resolved = state.total_roots >= state.active_roots
        ? state.total_roots - state.active_roots
        : 0;
    const Real resolved_fraction = state.total_roots > 0
        ? static_cast<Real>(resolved)
            / static_cast<Real>(state.total_roots)
        : 1;
    const Real correction_fraction = logarithmic_correction_progress(
        state.initial_maximum_correction,
        state.maximum_correction,
        state.target_correction);
    const auto elapsed = Clock::now() - state.started;
    const long double elapsed_seconds =
        std::chrono::duration<long double>(elapsed).count();
    const long double iteration_rate = elapsed_seconds > 0
        ? static_cast<long double>(
              std::max(0, state.iteration - state.start_iteration))
            / elapsed_seconds
        : 0;
    Real projected_fraction = 0;
    if (state.converged) {
        projected_fraction = 1;
    } else if (state.freeze_projection.ready) {
        const Real projected_finish =
            static_cast<Real>(state.iteration)
            + state.freeze_projection.remaining_iterations;
        if (projected_finish > 0) {
            projected_fraction =
                static_cast<Real>(state.iteration) / projected_finish;
        }
    }

    std::ostringstream title;
    title << "Mandelbrot upper-half root solver | period " << state.period
          << " | " << (state.converged ? "CONVERGED" : "running");

    std::ostringstream projection_suffix;
    if (state.freeze_projection.ready) {
        const auto projected_finish = static_cast<long long>(std::llround(
            static_cast<Real>(state.iteration)
            + state.freeze_projection.remaining_iterations));
        projection_suffix << "finish~iteration " << projected_finish;
        if (iteration_rate > 0) {
            const long double eta_seconds =
                static_cast<long double>(
                    state.freeze_projection.remaining_iterations)
                / iteration_rate;
            const auto eta = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<long double>(eta_seconds));
            projection_suffix << " | ETA~" << format_duration(eta);
        } else {
            projection_suffix << " | ETA=--";
        }
    } else {
        projection_suffix << state.freeze_projection.model;
    }

    std::ostringstream correction_suffix;
    correction_suffix << "max=" << dashboard_real(state.maximum_correction)
                      << " target=" << dashboard_real(
                             state.target_correction, 3);

    std::ostringstream iteration_line;
    iteration_line << "iteration: " << state.iteration << '/'
                   << state.maximum_iterations
                   << " (safety ceiling) | elapsed=" << format_duration(elapsed)
                   << " | rate=" << std::fixed << std::setprecision(2)
                   << iteration_rate << "/s";

    std::ostringstream corrections_line;
    corrections_line << "corrections: max="
                     << dashboard_real(state.maximum_correction)
                     << " | q99=" << dashboard_real(state.q99_correction)
                     << " | median="
                     << dashboard_real(state.median_correction);

    std::ostringstream active_line;
    active_line << "dynamic pool: active=" << state.active_roots << '/'
                << state.total_roots
                << " | frozen=" << state.total_frozen
                << " | actual resolved="
                << percent_string(resolved_fraction);

    std::ostringstream cloud_line;
    cloud_line << "full cloud: upper representatives=" << state.total_roots
               << " | implicit conjugates=" << state.total_roots
               << " | fixed real=" << state.fixed_real_roots
               << " | total=" << state.full_root_count;

    std::ostringstream solver_line;
    solver_line << "solver: phase="
                << (state.cluster_phase ? "cluster-AE" : "global-AE")
                << " | interaction=" << state.interaction_mode
                << " | exact step=" << (state.exact_step ? "yes" : "no")
                << " | step cap=" << dashboard_real(state.step_cap, 3);

    std::ostringstream polish_line;
    if (state.latest_polish_iteration > 0) {
        polish_line << "isolation sweep @iteration "
                    << state.latest_polish_iteration
                    << ": moved=" << state.latest_polish_moved
                    << " | newly frozen="
                    << state.latest_polish_frozen;
    } else {
        polish_line << "isolation sweep: not run yet";
    }

    std::ostringstream freeze_rate_line;
    freeze_rate_line
        << "freeze rate: latest="
        << dashboard_real(state.freeze_projection.latest_rate)
        << "/iteration | median="
        << dashboard_real(state.freeze_projection.smoothed_rate)
        << "/iteration | intervals="
        << state.freeze_projection.interval_count;

    std::ostringstream freeze_projection_line;
    freeze_projection_line
        << "freeze acceleration="
        << dashboard_real(state.freeze_projection.acceleration)
        << "/iteration^2 | model="
        << state.freeze_projection.model;
    if (state.freeze_projection.ready) {
        freeze_projection_line
            << " | remaining~"
            << static_cast<long long>(std::llround(
                   state.freeze_projection.remaining_iterations))
            << " iterations";
    }

    std::ostringstream worst_line;
    worst_line << "largest correction: ";
    if (state.worst_root) {
        worst_line << '#' << state.worst_root->first << " at "
                   << complex_string(state.worst_root->second, 6);
    } else {
        worst_line << "--";
    }

    std::ostringstream checkpoint_line;
    checkpoint_line << "checkpoint: last iteration "
                    << state.last_checkpoint_iteration;
    if (state.checkpoint_every > 0) {
        const int next = (
            state.iteration / state.checkpoint_every + 1)
            * state.checkpoint_every;
        checkpoint_line << " | next at " << next
                        << " | cadence=" << state.checkpoint_every;
    } else {
        checkpoint_line << " | disabled";
    }

    std::lock_guard<std::mutex> lock(g_print_mutex);
    std::cerr << "\033[H";
    auto line = [&](const std::string& text) {
        std::cerr << "\r\033[2K" << fit_to_width(text, width) << '\n';
    };
    line(title.str());
    line("");
    line(dashboard_progress_bar(
        config,
        "projected",
        projected_fraction,
        projection_suffix.str(),
        width));
    line(dashboard_progress_bar(
        config,
        "log correction",
        correction_fraction,
        correction_suffix.str(),
        width));
    line("");
    line(iteration_line.str());
    line(corrections_line.str());
    line(cloud_line.str());
    line(active_line.str());
    line(solver_line.str());
    line(polish_line.str());
    line(freeze_rate_line.str());
    line(freeze_projection_line.str());
    line(worst_line.str());
    line(checkpoint_line.str());
    line("Frozen roots remain in the cloud and are independently validated later. Ctrl-C restores the terminal.");
    std::cerr << "\033[J";
    std::cerr.flush();
}

void render_inline_progress(
    const Config& config,
    const std::string& label,
    std::size_t current,
    std::size_t total,
    Clock::time_point started,
    const std::string& detail = {},
    bool final = false,
    bool show_eta = true
) {
    if (!config.progress || config.progress_style != "bars") return;

    const Real fraction = total > 0
        ? static_cast<Real>(current) / static_cast<Real>(total)
        : 1;
    const auto elapsed = Clock::now() - started;

    std::ostringstream suffix;
    suffix << current << '/' << total;
    if (!detail.empty()) suffix << " | " << detail;
    suffix << " | elapsed " << format_duration(elapsed);

    if (show_eta && current > 0 && current < total) {
        const long double elapsed_seconds =
            std::chrono::duration<long double>(elapsed).count();
        if (elapsed_seconds > 0) {
            const long double remaining_seconds = elapsed_seconds
                * static_cast<long double>(total - current)
                / static_cast<long double>(current);
            const auto eta = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<long double>(remaining_seconds));
            suffix << " | eta " << format_duration(eta);
        }
    } else if (show_eta && current >= total && total > 0) {
        suffix << " | eta 0s";
    }

    const int safe_columns = std::max(20, terminal_width_columns() - 1);
    const std::string percentage = percent_string(fraction);
    const std::string suffix_text = suffix.str();
    const int fixed_width = static_cast<int>(label.size())
        + 4 + static_cast<int>(percentage.size())
        + 1 + static_cast<int>(suffix_text.size());
    const int bar_width = std::max(
        6,
        std::min(config.progress_bar_width,
                 std::max(6, safe_columns - fixed_width)));

    std::string line = label + " [" + progress_bar(fraction, bar_width)
        + "] " + percentage + " " + suffix_text;
    line = fit_to_width(std::move(line), safe_columns);

    std::lock_guard<std::mutex> lock(g_print_mutex);
    if (stdout_is_terminal()) {
        std::cout << "\r\033[2K" << line;
        if (final) std::cout << '\n';
        std::cout.flush();
    } else if (final) {
        std::cout << line << '\n';
    }
}

mandelbrot::catalogue::AreaScanProgressCallback make_inline_progress_callback(
    const Config& config,
    std::string label,
    Clock::time_point started,
    std::string detail = {},
    bool show_eta = true
) {
    return [&config,
            label = std::move(label),
            started,
            detail = std::move(detail),
            show_eta](std::size_t current, std::size_t total) {
        const bool final = total == 0 || current >= total;
        render_inline_progress(
            config,
            label,
            current,
            total,
            started,
            detail,
            final,
            show_eta);
    };
}

std::string render_area_progress_line(
    const Config& config,
    int thread_id,
    const AreaProgressSlot& slot,
    int terminal_columns
) {
    const int width = std::max(20, terminal_columns - 1);

    std::ostringstream worker;
    worker << 'T' << std::setw(2) << std::setfill('0') << thread_id
           << std::setfill(' ');

    if (!slot.active) {
        return fit_to_width(worker.str() + " idle", width);
    }

    std::ostringstream identity;
    const int job_width = std::max(
        1,
        static_cast<int>(std::to_string(
            std::max(1, slot.representative_job_count)).size()));
    identity << " p" << slot.period
             << " job" << std::setw(job_width) << std::setfill('0')
             << slot.representative_job_index
             << '/' << std::setw(job_width) << slot.representative_job_count
             << " c" << std::setw(5) << slot.component_index
             << std::setfill(' ');

    const std::string prefix = worker.str() + identity.str() + ' ';
    const std::string percentage = percent_string(slot.fraction);

    std::vector<std::string> suffixes;
    {
        std::ostringstream rich;
        rich << "rho=" << compact_real(slot.rho, 7)
             << " r=" << slot.radius_index << '/' << slot.radius_count;
        if (slot.theta_points > 0) rich << " N=" << slot.theta_points;
        if (finite(slot.area)) rich << " A=" << compact_real(slot.area, 5);
        if (!slot.status.empty()) rich << ' ' << slot.status;
        suffixes.push_back(rich.str());
    }
    {
        std::ostringstream medium;
        medium << "rho=" << compact_real(slot.rho, 5)
               << " r=" << slot.radius_index << '/' << slot.radius_count;
        if (slot.theta_points > 0) medium << " N=" << slot.theta_points;
        if (!slot.status.empty()) medium << ' ' << slot.status;
        suffixes.push_back(medium.str());
    }
    suffixes.push_back(slot.status);
    suffixes.push_back("");

    for (const std::string& suffix : suffixes) {
        int fixed = static_cast<int>(prefix.size())
                  + 2 + 1 + static_cast<int>(percentage.size());
        if (!suffix.empty()) fixed += 1 + static_cast<int>(suffix.size());
        const int available = width - fixed;
        const int bar_width = std::min(config.progress_bar_width, available);
        if (bar_width >= 6) {
            std::string line = prefix + '[' + progress_bar(slot.fraction, bar_width)
                             + "] " + percentage;
            if (!suffix.empty()) line += ' ' + suffix;
            return fit_to_width(line, width);
        }
    }

    return fit_to_width(
        worker.str() + identity.str() + ' ' + percentage + ' ' + slot.status,
        width);
}

void update_area_progress(
    int thread_id,
    int period,
    int representative_job_index,
    int representative_job_count,
    int component_index,
    int radius_index,
    int radius_count,
    Real rho,
    Real fraction,
    int theta_points,
    Real area,
    const std::string& status
) {
    if (thread_id < 0) return;
    std::lock_guard<std::mutex> lock(g_area_progress_mutex);
    if (static_cast<std::size_t>(thread_id) >= g_area_progress_slots.size()) return;

    auto& slot = g_area_progress_slots[static_cast<std::size_t>(thread_id)];
    slot.active = true;
    slot.period = period;
    slot.representative_job_index = representative_job_index;
    slot.representative_job_count = representative_job_count;
    slot.component_index = component_index;
    slot.radius_index = radius_index;
    slot.radius_count = radius_count;
    slot.rho = rho;
    // Callers report whole-job progress. Do not retain a previous 100% from a
    // completed sub-stage; retries and later radii must be allowed to reset the
    // displayed fraction instead of appearing permanently finished.
    slot.fraction = fraction;
    slot.theta_points = theta_points;
    slot.area = area;
    slot.status = status;
}

void clear_area_progress(int thread_id) {
    std::lock_guard<std::mutex> lock(g_area_progress_mutex);
    if (thread_id >= 0
        && static_cast<std::size_t>(thread_id) < g_area_progress_slots.size()) {
        g_area_progress_slots[static_cast<std::size_t>(thread_id)] = AreaProgressSlot{};
    }
}

void monitor_area_progress(
    const Config& config,
    int thread_count,
    int total_jobs,
    const std::atomic<int>& completed_jobs,
    const std::atomic<int>& failed_jobs,
    const std::atomic<int>& row_count,
    const std::atomic<int>& checkpointed_jobs,
    const std::atomic<int>& checkpoint_batches,
    Clock::time_point started,
    std::vector<std::string> header_lines
) {
    if (!config.progress || config.progress_style != "bars") return;

    const bool alternate = config.progress_screen == "alternate"
                        && stderr_is_terminal();
    bool first = true;
    const int lines = static_cast<int>(header_lines.size()) + thread_count + 1;

    while (!g_area_monitor_stop.load()) {
        if (alternate) {
            // Full repaint means lines wrapped before a terminal resize are
            // erased instead of lingering on screen.
            std::cerr << "\033[H";
        } else if (!first) {
            std::cerr << "\033[" << lines << 'A';
        }
        first = false;

        std::vector<AreaProgressSlot> slots;
        {
            std::lock_guard<std::mutex> lock(g_area_progress_mutex);
            slots = g_area_progress_slots;
        }

        const int columns = terminal_width_columns();
        const int safe_columns = std::max(20, columns - 1);

        for (const auto& line : header_lines) {
            std::cerr << "\r\033[2K" << fit_to_width(line, safe_columns) << '\n';
        }
        for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
            std::cerr << "\r\033[2K"
                      << render_area_progress_line(
                             config,
                             thread_id,
                             slots[static_cast<std::size_t>(thread_id)],
                             columns)
                      << '\n';
        }

        const auto now = Clock::now();
        const auto elapsed = now - started;
        const int completed = completed_jobs.load();

        std::ostringstream footer;
        footer << "done=" << completed << '/' << total_jobs
               << " failed=" << failed_jobs.load()
               << " rows=" << row_count.load()
               << " checkpoint=" << checkpointed_jobs.load()
               << " batches=" << checkpoint_batches.load()
               << " elapsed=" << format_duration(elapsed);

        // Estimate remaining wall time from the average completed-job rate.
        // Suppress the estimate during the very early startup phase, where a
        // handful of unusually cheap or expensive components would make it
        // jump around wildly.
        if (completed >= std::min(50, total_jobs) && completed < total_jobs) {
            const long double elapsed_seconds =
                std::chrono::duration<long double>(elapsed).count();
            if (elapsed_seconds > 0) {
                const long double seconds_per_job =
                    elapsed_seconds / static_cast<long double>(completed);
                const long double eta_seconds = seconds_per_job
                    * static_cast<long double>(total_jobs - completed);
                const auto eta = std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<long double>(eta_seconds));
                footer << " eta=" << format_duration(eta);
            }
        } else if (completed >= total_jobs && total_jobs > 0) {
            footer << " eta=0s";
        } else {
            footer << " eta=--";
        }

        std::cerr << "\r\033[2K" << fit_to_width(footer.str(), safe_columns) << '\n';

        if (alternate) std::cerr << "\033[J";
        std::cerr.flush();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config.progress_refresh_ms));
    }
}

// -----------------------------------------------------------------------------
// Integer/combinatorial helpers
// -----------------------------------------------------------------------------

constexpr int PRECOMPUTED_PERIOD_MAX = 20;

int mobius_uncached(int n) {
    if (n < 1) {
        throw std::invalid_argument("mobius requires n >= 1");
    }
    if (n == 1) return 1;
    int prime_count = 0;
    for (int p = 2; p <= n / p; ++p) {
        if (n % p != 0) continue;
        n /= p;
        ++prime_count;
        if (n % p == 0) return 0;
        while (n % p == 0) n /= p;
    }
    if (n > 1) ++prime_count;
    return (prime_count % 2 == 0) ? 1 : -1;
}

std::vector<int> divisors_uncached(int n) {
    if (n < 1) {
        throw std::invalid_argument("divisors requires n >= 1");
    }
    std::vector<int> result;
    for (int d = 1; d <= n / d; ++d) {
        if (n % d != 0) continue;
        result.push_back(d);
        if (d != n / d) result.push_back(n / d);
    }
    std::sort(result.begin(), result.end());
    return result;
}

struct ExactPeriodTerm {
    int iterate = 0;
    int mobius = 0;
};

struct PeriodArithmetic {
    int period = 0;
    int mobius = 0;
    std::vector<int> divisors;
    std::vector<int> proper_divisors;
    std::vector<ExactPeriodTerm> exact_terms;
};

PeriodArithmetic make_period_arithmetic(int n) {
    PeriodArithmetic result;
    result.period = n;
    result.mobius = mobius_uncached(n);
    result.divisors = divisors_uncached(n);
    result.proper_divisors = result.divisors;
    if (!result.proper_divisors.empty()
        && result.proper_divisors.back() == n) {
        result.proper_divisors.pop_back();
    }
    result.exact_terms.reserve(result.divisors.size());
    for (int d : result.divisors) {
        const int coefficient = mobius_uncached(n / d);
        if (coefficient != 0) {
            result.exact_terms.push_back({d, coefficient});
        }
    }
    return result;
}

struct IntegerArithmeticCache {
    std::array<int, PRECOMPUTED_PERIOD_MAX + 1> mobius{};
    std::array<PeriodArithmetic, PRECOMPUTED_PERIOD_MAX + 1> periods{};

    IntegerArithmeticCache() {
        for (int n = 1; n <= PRECOMPUTED_PERIOD_MAX; ++n) {
            periods[static_cast<std::size_t>(n)] = make_period_arithmetic(n);
            mobius[static_cast<std::size_t>(n)] =
                periods[static_cast<std::size_t>(n)].mobius;
        }
    }
};

const IntegerArithmeticCache& integer_arithmetic_cache() {
    // Constructed once, before the expensive root iterations. All period <= 20
    // lookups thereafter are array accesses with no factorization or allocation.
    static const IntegerArithmeticCache cache;
    return cache;
}

const PeriodArithmetic& period_arithmetic(int n) {
    if (n < 1) {
        throw std::invalid_argument("period arithmetic requires n >= 1");
    }
    if (n <= PRECOMPUTED_PERIOD_MAX) {
        return integer_arithmetic_cache().periods[static_cast<std::size_t>(n)];
    }

    // The scanner is currently intended for periods through 20, but retaining
    // a stable fallback keeps the helper correct if a larger period is queried.
    static std::mutex fallback_mutex;
    static std::map<int, PeriodArithmetic> fallback;
    std::lock_guard<std::mutex> lock(fallback_mutex);
    auto [iterator, inserted] = fallback.try_emplace(n);
    if (inserted) iterator->second = make_period_arithmetic(n);
    return iterator->second;
}

int mobius(int n) {
    if (n >= 1 && n <= PRECOMPUTED_PERIOD_MAX) {
        return integer_arithmetic_cache().mobius[static_cast<std::size_t>(n)];
    }
    return mobius_uncached(n);
}

const std::vector<int>& divisors(int n) {
    return period_arithmetic(n).divisors;
}

const std::vector<int>& proper_divisors(int n) {
    return period_arithmetic(n).proper_divisors;
}

// Moebius inversion
// let's have functions g and f defined on natural numbers, such that
// g(n) = sum_(d: d | n) f(d), where d | n means "d divides n". Examples:
// g(1) = f(1)
// g(2) = f(1) + f(2)
// g(3) = f(1) + f(3)
// g(4) = f(1) + f(2) + f(4)
// In a sense, this is an infinite set of equations
// We can easily solve for f:
// f(1) = g(1)
// f(2) = g(2) - g(1)
// f(3) = g(3) - g(1)
// f(4) = g(4) - g(1) - (g(2) - g(1)) = g(4) - g(2)
// f(6) = g(6) - g(1) - (g(2) - g(1)) - (g(3) - g(1)) = g(6) - g(3) - g(2) + g(1)
// We can see that
// f (n) = sum_(d: d | n) M_n (d) g(d)
// Substitute for g:
// f (n) = sum_(d: d | n) M_n (d) sum_(e: e | d) f(e)
// Reverse the sums
// f (n) = sum_(e: e | n) f(e) sum_(d: d | n, e | d) M_n (d)
// Since this is true for any n, we have to demand
// sum_(d: d | n, e | d) M_n (d) = 1 if e = n, 0 otherwise.
// The only function that satisfies this condition for any e, n is M_n (d) = mu(n/d), the Moebius function.

// Counting centers

boost::multiprecision::cpp_int center_count_big(int n) {
    using boost::multiprecision::cpp_int;
    cpp_int degree = 0;
    for (const auto& term : period_arithmetic(n).exact_terms) {
        degree += term.mobius * (cpp_int(1) << (term.iterate - 1));
    }
    if (degree <= 0) throw std::runtime_error("Invalid exact-period center count.");
    return degree;
}

std::size_t center_count(int n) {
    const auto degree = center_count_big(n);
    const auto maximum = boost::multiprecision::cpp_int(
        std::numeric_limits<std::size_t>::max());
    if (degree > maximum) {
        throw std::runtime_error(
            "Exact-period center count does not fit in size_t; this operation "
            "requires explicitly materializing all centers.");
    }
    return degree.convert_to<std::size_t>();
}


boost::multiprecision::cpp_int real_exact_center_count_big(int n) {
    using boost::multiprecision::cpp_int;
    if (n < 1) {
        throw std::invalid_argument("real center count requires n >= 1");
    }

    cpp_int numerator = 0;
    for (int d : divisors(n)) {
        if ((d & 1) == 0) continue;
        const int coefficient = mobius(d);
        if (coefficient == 0) continue;
        numerator += coefficient * (cpp_int(1) << (n / d));
    }

    const cpp_int denominator = 2 * n;
    if (numerator <= 0 || numerator % denominator != 0) {
        throw std::runtime_error("Invalid exact-period real-center count.");
    }
    return numerator / denominator;
}

std::size_t real_exact_center_count(int n) {
    const auto count = real_exact_center_count_big(n);
    const auto maximum = boost::multiprecision::cpp_int(
        std::numeric_limits<std::size_t>::max());
    if (count > maximum) {
        throw std::runtime_error(
            "Exact-period real-center count does not fit in size_t.");
    }
    return count.convert_to<std::size_t>();
}

boost::multiprecision::cpp_int real_raw_left_count_big(int n) {
    using boost::multiprecision::cpp_int;
    cpp_int count = 0;
    for (int d : divisors(n)) {
        // Periods 1 and 2 have the known centers 0 and -1, outside the search
        // interval [-2,-5/4]. Every real center of period >2 lies inside it.
        if (d > 2) count += real_exact_center_count_big(d);
    }
    return count;
}

std::size_t real_raw_left_count(int n) {
    const auto count = real_raw_left_count_big(n);
    const auto maximum = boost::multiprecision::cpp_int(
        std::numeric_limits<std::size_t>::max());
    if (count > maximum) {
        throw std::runtime_error("Raw real-root count does not fit in size_t.");
    }
    return count.convert_to<std::size_t>();
}

// -----------------------------------------------------------------------------
// Critical orbit and exact-period logarithmic derivative
// -----------------------------------------------------------------------------

struct OrbitValue {
    Complex value{};
    Complex derivative{};
};

OrbitValue critical_orbit(Complex c, int n) {
    Complex z{0, 0};
    Complex derivative{0, 0};
    for (int k = 0; k < n; ++k) {
        derivative = 2.0L * z * derivative + Complex{1, 0};
        z = z * z + c;
    }
    return {z, derivative};
}

bool exact_period_log_derivative(
    Complex c,
    const PeriodArithmetic& arithmetic,
    Complex& result
) {
    const auto& terms = arithmetic.exact_terms;

    Complex z{0, 0};
    Complex derivative{0, 0};
    bool reciprocal_mode = false;
    Complex reciprocal{0, 0};
    Complex logarithmic_derivative{0, 0};

    result = Complex{0, 0};
    std::size_t term_index = 0;
    for (int k = 1; k <= arithmetic.period; ++k) {
        if (!reciprocal_mode) {
            derivative = 2.0L * z * derivative + Complex{1, 0};
            z = z * z + c;
            const Real magnitude = safe_abs(z);
            if (!finite(z) || magnitude > 1.0e1000L) {
                if (!finite(z) || magnitude == 0) return false;
                reciprocal_mode = true;
                reciprocal = Complex{1, 0} / z;
                logarithmic_derivative = derivative / z;
            }
        } else {
            const Complex u2 = reciprocal * reciprocal;
            const Complex denominator = Complex{1, 0} + c * u2;
            if (safe_abs(denominator) < 1.0e-40L || !finite(denominator)) {
                return false;
            }
            logarithmic_derivative =
                (2.0L * logarithmic_derivative + u2) / denominator;
            reciprocal = u2 / denominator;
        }

        if (term_index < terms.size() && terms[term_index].iterate == k) {
            Complex ratio;
            if (reciprocal_mode) {
                ratio = logarithmic_derivative;
            } else {
                if (safe_abs(z) < 1.0e-40L) return false;
                ratio = derivative / z;
            }
            if (!finite(ratio)) return false;
            result += static_cast<Real>(terms[term_index].mobius) * ratio;
            ++term_index;
        }
    }

    return term_index == terms.size()
        && finite(result)
        && safe_abs(result) > 1.0e-40L;
}

std::optional<Complex> exact_period_newton_correction(
    Complex c,
    const PeriodArithmetic& arithmetic
) {
    Complex logarithmic_derivative;
    if (!exact_period_log_derivative(c, arithmetic, logarithmic_derivative)) {
        return std::nullopt;
    }
    const Complex correction = Complex{1, 0} / logarithmic_derivative;
    if (!finite(correction)) return std::nullopt;
    return correction;
}

int detected_center_period(Complex c, int n, Real tolerance) {
    const auto& divs = divisors(n);
    std::size_t divisor_index = 0;
    Complex z{0, 0};
    for (int k = 1; k <= n; ++k) {
        z = z * z + c;
        if (divisor_index < divs.size() && divs[divisor_index] == k) {
            if (safe_abs(z) <= tolerance) return k;
            ++divisor_index;
        }
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Simultaneous center root solver (damped Aberth-Ehrlich)
// -----------------------------------------------------------------------------

struct RootCheckpoint {
    int iteration = 0;
    std::vector<Real> real_roots;
    std::vector<Complex> roots;
};

void write_root_checkpoint_full(
    const fs::path& path,
    int period,
    int iteration,
    const std::vector<Complex>& roots,
    const Config& config
) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        throw std::runtime_error("Could not write checkpoint: " + temporary.string());
    }
    output << "version 1\n";
    output << "period " << period << '\n';
    output << "iteration " << iteration << '\n';
    output << "count " << roots.size() << '\n';
    output << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << std::scientific;

    (void)config;
    for (const auto& root : roots) {
        output << root.real() << ' ' << root.imag() << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Could not finish checkpoint: " + temporary.string());
    }
    atomic_replace(temporary, path);

}

void write_root_checkpoint_symmetric(
    const fs::path& path,
    int period,
    int iteration,
    const std::vector<Real>& real_roots,
    const std::vector<Complex>& upper_roots,
    const Config& config
) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        throw std::runtime_error("Could not write checkpoint: " + temporary.string());
    }
    output << "version 2\n";
    output << "representation upper-half-conjugate\n";
    output << "period " << period << '\n';
    output << "iteration " << iteration << '\n';
    output << "real_count " << real_roots.size() << '\n';
    output << "upper_count " << upper_roots.size() << '\n';
    output << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << std::scientific;

    (void)config;
    for (Real root : real_roots) {
        output << root << '\n';
    }
    for (const auto& root : upper_roots) {
        output << root.real() << ' ' << root.imag() << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Could not finish checkpoint: " + temporary.string());
    }
    atomic_replace(temporary, path);

}

std::optional<RootCheckpoint> read_root_checkpoint_full(
    const fs::path& path,
    int period,
    std::size_t expected_count,
    const Config& config
) {
    std::ifstream input(path);
    if (!input) return std::nullopt;

    std::string key;
    int version = 0;
    int stored_period = 0;
    int iteration = 0;
    std::size_t count = 0;
    input >> key >> version;
    input >> key >> stored_period;
    input >> key >> iteration;
    input >> key >> count;
    if (!input || version != 1 || stored_period != period
        || count != expected_count) {
        return std::nullopt;
    }

    std::vector<Complex> roots;
    roots.reserve(count);
    const auto started = Clock::now();
    if (count > 0) {
        render_inline_progress(
            config, "  loading root checkpoint", 0, count, started);
    }
    for (std::size_t i = 0; i < count; ++i) {
        Real re = 0;
        Real im = 0;
        input >> re >> im;
        if (!input) return std::nullopt;
        roots.emplace_back(re, im);
        const std::size_t done = i + 1;
        if (done == count || done % 2048 == 0) {
            render_inline_progress(
                config,
                "  loading root checkpoint",
                done,
                count,
                started,
                {},
                done == count,
                true);
        }
    }
    return RootCheckpoint{iteration, {}, std::move(roots)};
}

std::optional<RootCheckpoint> read_root_checkpoint_symmetric(
    const fs::path& path,
    int period,
    std::size_t expected_real_count,
    std::size_t expected_upper_count,
    const Config& config
) {
    std::ifstream input(path);
    if (!input) return std::nullopt;

    std::string key;
    std::string representation;
    int version = 0;
    int stored_period = 0;
    int iteration = 0;
    std::size_t real_count = 0;
    std::size_t upper_count = 0;
    input >> key >> version;
    input >> key >> representation;
    input >> key >> stored_period;
    input >> key >> iteration;
    input >> key >> real_count;
    input >> key >> upper_count;
    if (!input || version != 2
        || representation != "upper-half-conjugate"
        || stored_period != period
        || real_count != expected_real_count
        || upper_count != expected_upper_count) {
        return std::nullopt;
    }

    const std::size_t total = real_count + upper_count;
    const auto started = Clock::now();
    if (total > 0) {
        render_inline_progress(
            config, "  loading root checkpoint", 0, total, started);
    }
    std::size_t loaded = 0;
    std::vector<Real> real_roots(real_count);
    for (Real& root : real_roots) {
        input >> root;
        if (!input) return std::nullopt;
        ++loaded;
        if (loaded == total || loaded % 2048 == 0) {
            render_inline_progress(
                config,
                "  loading root checkpoint",
                loaded,
                total,
                started,
                {},
                loaded == total,
                true);
        }
    }

    std::vector<Complex> upper_roots;
    upper_roots.reserve(upper_count);
    for (std::size_t i = 0; i < upper_count; ++i) {
        Real re = 0;
        Real im = 0;
        input >> re >> im;
        if (!input || !(im > 0)) return std::nullopt;
        upper_roots.emplace_back(re, im);
        ++loaded;
        if (loaded == total || loaded % 2048 == 0) {
            render_inline_progress(
                config,
                "  loading root checkpoint",
                loaded,
                total,
                started,
                {},
                loaded == total,
                true);
        }
    }
    return RootCheckpoint{
        iteration,
        std::move(real_roots),
        std::move(upper_roots),
    };
}

std::vector<Complex> initial_root_cloud(
    std::size_t count,
    const Config& config
) {
    std::vector<Complex> roots;
    roots.reserve(count);
    const Real center_x = -0.75L;
    const Real axis_x = 1.35L;
    const Real axis_y = 1.20L;
    const auto started = Clock::now();
    if (count > 0) {
        render_inline_progress(
            config, "  initializing root cloud", 0, count, started);
    }
    for (std::size_t i = 0; i < count; ++i) {
        const Real radial = std::sqrt(
            (static_cast<Real>(i) + 0.5L) / static_cast<Real>(count));
        const Real angle = GOLDEN_ANGLE * static_cast<Real>(i);
        roots.emplace_back(
            center_x + axis_x * radial * std::cos(angle),
            axis_y * radial * std::sin(angle));
        const std::size_t done = i + 1;
        if (done == count || done % 2048 == 0) {
            render_inline_progress(
                config,
                "  initializing root cloud",
                done,
                count,
                started,
                {},
                done == count,
                true);
        }
    }
    return roots;
}

std::vector<Complex> initial_upper_root_cloud(
    std::size_t count,
    const Config& config
) {
    std::vector<Complex> roots;
    roots.reserve(count);
    const Real center_x = -0.75L;
    const Real axis_x = 1.35L;
    const Real axis_y = 1.20L;
    const Real golden_fraction = 0.618033988749894848204586834365638118L;
    const auto started = Clock::now();
    if (count > 0) {
        render_inline_progress(
            config, "  initializing upper roots", 0, count, started);
    }
    for (std::size_t i = 0; i < count; ++i) {
        const Real radial = std::sqrt(
            (static_cast<Real>(i) + 0.5L) / static_cast<Real>(count));
        const Real fraction = std::fmod(
            (static_cast<Real>(i) + 0.5L) * golden_fraction,
            static_cast<Real>(1));
        const Real angle = PI * fraction;
        Real imaginary = axis_y * radial * std::sin(angle);
        imaginary = std::max<Real>(imaginary, 1.0e-8L);
        roots.emplace_back(
            center_x + axis_x * radial * std::cos(angle),
            imaginary);
        const std::size_t done = i + 1;
        if (done == count || done % 2048 == 0) {
            render_inline_progress(
                config,
                "  initializing upper roots",
                done,
                count,
                started,
                {},
                done == count,
                true);
        }
    }
    return roots;
}

template <typename Function>
void parallel_for(std::size_t count, unsigned threads, Function function) {
    if (count == 0) return;
    threads = std::max(
        1u,
        std::min<unsigned>(threads, static_cast<unsigned>(count)));
    if (threads == 1 || count < 64) {
        function(0, count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threads);
    const std::size_t chunk = (count + threads - 1) / threads;
    for (unsigned thread = 0; thread < threads; ++thread) {
        const std::size_t begin = thread * chunk;
        const std::size_t end = std::min(count, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([=, &function]() { function(begin, end); });
    }
    for (auto& worker : workers) worker.join();
}

// Dynamically claim small batches for independent work whose cost varies by
// item. Unlike parallel_for's fixed contiguous partitions, a worker that
// finishes an inexpensive batch immediately helps with the remaining queue.
// Keeping the claim size configurable avoids an atomic operation per item for
// fine-grained workloads, while expensive component traces can use a claim of
// one to eliminate long straggler tails.
template <typename Function>
void parallel_for_dynamic(
    std::size_t count,
    unsigned threads,
    std::size_t claim_size,
    Function function
) {
    if (count == 0) return;
    threads = std::max(
        1u,
        std::min<unsigned>(threads, static_cast<unsigned>(count)));
    claim_size = std::max<std::size_t>(1, claim_size);
    if (threads == 1) {
        for (std::size_t i = 0; i < count; ++i) function(i);
        return;
    }

    std::atomic<std::size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&] {
            while (true) {
                const std::size_t begin = next.fetch_add(
                    claim_size, std::memory_order_relaxed);
                if (begin >= count) break;
                const std::size_t end = std::min(count, begin + claim_size);
                for (std::size_t i = begin; i < end; ++i) function(i);
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

template <typename Function>
void parallel_for_progress(
    std::size_t count,
    unsigned threads,
    const Config& config,
    const std::string& label,
    const std::string& detail,
    Function function
) {
    if (count == 0) return;
    if (!config.progress || config.progress_style != "bars") {
        parallel_for(count, threads, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) function(i);
        });
        return;
    }

    const auto started = Clock::now();
    render_inline_progress(config, label, 0, count, started, detail);
    std::atomic<std::size_t> completed{0};
    const std::size_t report_stride = std::max<std::size_t>(
        256,
        count / 200);
    std::atomic<std::size_t> next_report{report_stride};

    parallel_for(count, threads, [&](std::size_t begin, std::size_t end) {
        std::size_t local_completed = 0;
        for (std::size_t i = begin; i < end; ++i) {
            function(i);
            ++local_completed;
            if (local_completed < 256 && i + 1 != end) continue;
            const std::size_t done =
                completed.fetch_add(local_completed) + local_completed;
            local_completed = 0;
            std::size_t report = next_report.load();
            while (done >= report
                   && !next_report.compare_exchange_weak(
                       report, report + report_stride)) {
            }
            if (done >= report || done == count) {
                render_inline_progress(
                    config,
                    label,
                    std::min(done, count),
                    count,
                    started,
                    detail,
                    done >= count,
                    true);
            }
        }
    });
}

template <typename Function>
void parallel_root_iteration(
    std::size_t count,
    unsigned threads,
    Function function
) {
    if (count == 0) return;
    parallel_for(count, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) function(i);
    });
}

// -----------------------------------------------------------------------------
// Certified real-axis prepass
// -----------------------------------------------------------------------------

struct RealOrbitSecond {
    Real value = 0;
    Real derivative = 0;
    Real second_derivative = 0;
};

RealOrbitSecond critical_orbit_real_second(Real c, int n) {
    Real z = 0;
    Real derivative = 0;
    Real second_derivative = 0;
    for (int k = 0; k < n; ++k) {
        second_derivative = 2 * (
            derivative * derivative + z * second_derivative);
        derivative = 2 * z * derivative + 1;
        z = z * z + c;
    }
    return {z, derivative, second_derivative};
}

template <unsigned Digits10>
int critical_orbit_real_sign_mp(Real c, int n) {
    using Backend = boost::multiprecision::cpp_dec_float<Digits10>;
    using MpReal = boost::multiprecision::number<Backend>;
    const MpReal parameter(real_string(c));
    MpReal z = 0;
    for (int k = 0; k < n; ++k) z = z * z + parameter;
    return z > 0 ? 1 : z < 0 ? -1 : 0;
}

template <unsigned Digits10>
int critical_orbit_real_derivative_sign_mp(Real c, int n) {
    using Backend = boost::multiprecision::cpp_dec_float<Digits10>;
    using MpReal = boost::multiprecision::number<Backend>;
    const MpReal parameter(real_string(c));
    MpReal z = 0;
    MpReal derivative = 0;
    for (int k = 0; k < n; ++k) {
        derivative = 2 * z * derivative + 1;
        z = z * z + parameter;
    }
    return derivative > 0 ? 1 : derivative < 0 ? -1 : 0;
}

template <unsigned Digits10>
int critical_orbit_real_second_derivative_sign_mp(Real c, int n) {
    using Backend = boost::multiprecision::cpp_dec_float<Digits10>;
    using MpReal = boost::multiprecision::number<Backend>;
    const MpReal parameter(real_string(c));
    MpReal z = 0;
    MpReal derivative = 0;
    MpReal second_derivative = 0;
    for (int k = 0; k < n; ++k) {
        second_derivative = 2 * (
            derivative * derivative + z * second_derivative);
        derivative = 2 * z * derivative + 1;
        z = z * z + parameter;
    }
    return second_derivative > 0 ? 1 : second_derivative < 0 ? -1 : 0;
}

int critical_orbit_real_sign(Real c, int n) {
    Real z = 0;
    for (int k = 0; k < n; ++k) {
        z = z * z + c;
        if (!finite(z) || z > 1.0e1000L) return 1;
    }
    if (std::abs(z) > 1.0e-14L) return z > 0 ? 1 : -1;
    return critical_orbit_real_sign_mp<200>(c, n);
}

int critical_orbit_real_derivative_sign(Real c, int n) {
    const auto orbit = critical_orbit_real_second(c, n);
    if (finite(orbit.derivative) && std::abs(orbit.derivative) > 1.0e-13L) {
        return orbit.derivative > 0 ? 1 : -1;
    }
    return critical_orbit_real_derivative_sign_mp<200>(c, n);
}

int critical_orbit_real_second_derivative_sign(Real c, int n) {
    const auto orbit = critical_orbit_real_second(c, n);
    if (finite(orbit.second_derivative)
        && std::abs(orbit.second_derivative) > 1.0e-13L) {
        return orbit.second_derivative > 0 ? 1 : -1;
    }
    return critical_orbit_real_second_derivative_sign_mp<200>(c, n);
}

int exact_dyadic_orbit_sign(
    int n,
    boost::multiprecision::cpp_int parameter_numerator,
    std::size_t denominator_power
) {
    using boost::multiprecision::cpp_int;
    while (denominator_power > 0
           && parameter_numerator % 2 == 0) {
        parameter_numerator /= 2;
        --denominator_power;
    }
    if (parameter_numerator == 0) return 0;

    const cpp_int c_numerator = parameter_numerator;
    cpp_int numerator = parameter_numerator;
    std::size_t exponent = denominator_power;
    for (int k = 1; k < n; ++k) {
        const std::size_t shift = 2 * exponent - denominator_power;
        numerator = numerator * numerator
            + c_numerator * (cpp_int(1) << shift);
        exponent *= 2;
    }
    return numerator > 0 ? 1 : numerator < 0 ? -1 : 0;
}

std::size_t real_grid_denominator_power(int power) {
    return static_cast<std::size_t>(2 * power + 2);
}

boost::multiprecision::cpp_int real_grid_numerator(
    std::size_t index,
    int power
) {
    using boost::multiprecision::cpp_int;
    // Let u = index / 2^power and use
    //
    //     c(u) = -2 + (3/4) u^2.
    //
    // Thus c has the exact dyadic representation
    //
    //     [-2 * 2^(2*power+2) + 3*index^2] / 2^(2*power+2).
    //
    // The quadratic coordinate is nested under power refinement and places
    // substantially more samples in the root-dense chaotic region near -2.
    const auto denominator_power = real_grid_denominator_power(power);
    const cpp_int j = index;
    return -cpp_int(2) * (cpp_int(1) << denominator_power)
        + cpp_int(3) * j * j;
}

Real real_grid_coordinate(std::size_t index, int power) {
    const Real u = std::ldexp(static_cast<Real>(index), -power);
    return -2.0L + 0.75L * u * u;
}

int critical_orbit_dyadic_sign(
    std::size_t index,
    int power,
    int period
) {
    const Real c = real_grid_coordinate(index, power);
    Real z = 0;
    for (int k = 0; k < period; ++k) {
        z = z * z + c;
        if (!finite(z) || z > 1.0e1000L) return 1;
    }
    if (std::abs(z) > 1.0e-13L) return z > 0 ? 1 : -1;

    using MpReal = boost::multiprecision::number<
        boost::multiprecision::cpp_dec_float<200>>;
    const auto numerator = real_grid_numerator(index, power);
    const auto denominator_power = real_grid_denominator_power(power);
    const MpReal denominator = MpReal(
        boost::multiprecision::cpp_int(1) << denominator_power);
    const MpReal parameter = MpReal(numerator) / denominator;
    MpReal mp_z = 0;
    for (int k = 0; k < period; ++k) {
        mp_z = mp_z * mp_z + parameter;
    }
    if (abs(mp_z) > pow(MpReal(10), -170)) {
        return mp_z > 0 ? 1 : -1;
    }
    return exact_dyadic_orbit_sign(
        period, numerator, denominator_power);
}

struct RealRootBracket {
    Real left = 0;
    Real right = 0;
    int left_sign = 0;
    int right_sign = 0;
};

std::optional<Real> polish_real_critical_bracket(
    Real left,
    Real right,
    int left_sign,
    int right_sign,
    int period,
    const Config& config
) {
    if (left_sign == 0 || right_sign == 0
        || left_sign == right_sign) {
        return std::nullopt;
    }

    Real x = (left + right) / 2;
    for (int iteration = 0;
         iteration < config.real_root_polish_iterations;
         ++iteration) {
        const auto orbit = critical_orbit_real_second(x, period);
        Real candidate = (left + right) / 2;
        if (finite(orbit.derivative)
            && finite(orbit.second_derivative)
            && std::abs(orbit.second_derivative) > 1.0e-30L) {
            const Real newton = x
                - orbit.derivative / orbit.second_derivative;
            if (finite(newton) && newton > left && newton < right) {
                candidate = newton;
            }
        }

        int sign = critical_orbit_real_derivative_sign(candidate, period);
        if (sign == 0) return candidate;
        if (sign == left_sign) {
            left = candidate;
            left_sign = sign;
        } else {
            right = candidate;
            right_sign = sign;
        }
        x = candidate;
        if (right - left <= config.real_root_bracket_tolerance) break;
    }
    return (left + right) / 2;
}

std::optional<std::pair<Real, int>> polish_real_critical_bracket_mp(
    Real left,
    Real right,
    int left_sign,
    int right_sign,
    int period
) {
    using MpReal = boost::multiprecision::number<
        boost::multiprecision::cpp_dec_float<200>>;
    if (left_sign == 0 || right_sign == 0 || left_sign == right_sign) {
        return std::nullopt;
    }

    MpReal a(real_string(left));
    MpReal b(real_string(right));
    int sign_a = left_sign;
    MpReal x = (a + b) / 2;
    MpReal function_value = 0;
    for (int iteration = 0; iteration < 260; ++iteration) {
        MpReal z = 0;
        MpReal derivative = 0;
        MpReal second_derivative = 0;
        for (int k = 0; k < period; ++k) {
            second_derivative = 2 * (
                derivative * derivative + z * second_derivative);
            derivative = 2 * z * derivative + 1;
            z = z * z + x;
        }
        function_value = z;
        MpReal candidate = (a + b) / 2;
        if (second_derivative != 0) {
            const MpReal newton = x - derivative / second_derivative;
            if (newton > a && newton < b) candidate = newton;
        }

        MpReal candidate_z = 0;
        MpReal candidate_derivative = 0;
        for (int k = 0; k < period; ++k) {
            candidate_derivative = 2 * candidate_z
                * candidate_derivative + 1;
            candidate_z = candidate_z * candidate_z + candidate;
        }
        const int sign = candidate_derivative > 0
            ? 1 : candidate_derivative < 0 ? -1 : 0;
        if (sign == 0) {
            x = candidate;
            function_value = candidate_z;
            break;
        }
        if (sign == sign_a) {
            a = candidate;
            sign_a = sign;
        } else {
            b = candidate;
        }
        x = candidate;
        function_value = candidate_z;
        if (b - a < pow(MpReal(10), -80)) break;
    }

    // Evaluate F_n at the high-precision critical point itself.
    MpReal z = 0;
    for (int k = 0; k < period; ++k) z = z * z + x;
    const int function_sign = z > 0 ? 1 : z < 0 ? -1 : 0;
    if (function_sign == 0) return std::nullopt;

    Real point = x.convert_to<Real>();
    auto usable = [&](Real candidate) {
        return candidate > left && candidate < right
            && critical_orbit_real_sign_mp<200>(candidate, period)
                == function_sign;
    };
    if (usable(point)) return std::pair<Real, int>{point, function_sign};

    // The exact extremum can round to the wrong side of a very narrow lobe.
    // Search neighboring representable long doubles before declaring that the
    // pair cannot be represented by this Real type.
    Real below = point;
    Real above = point;
    for (int step = 0; step < 64; ++step) {
        below = std::nextafter(below, left);
        if (usable(below)) return std::pair<Real, int>{below, function_sign};
        above = std::nextafter(above, right);
        if (usable(above)) return std::pair<Real, int>{above, function_sign};
    }
    return std::nullopt;
}

// Finds an extremum of F'_n (a zero of F''_n) and returns both its location
// and the sign of F'_n there. This is the second-level recovery needed when a
// dyadic interval contains two zeros of F'_n: the derivative then has the same
// sign at both endpoints, so the ordinary critical-point scan cannot see them.
std::optional<std::pair<Real, int>>
polish_real_second_critical_bracket_mp(
    Real left,
    Real right,
    int left_sign,
    int right_sign,
    int period
) {
    using MpReal = boost::multiprecision::number<
        boost::multiprecision::cpp_dec_float<200>>;
    if (left_sign == 0 || right_sign == 0 || left_sign == right_sign) {
        return std::nullopt;
    }

    MpReal a(real_string(left));
    MpReal b(real_string(right));
    int sign_a = left_sign;
    MpReal x = (a + b) / 2;
    MpReal derivative_value = 0;
    for (int iteration = 0; iteration < 260; ++iteration) {
        const MpReal candidate = (a + b) / 2;
        MpReal z = 0;
        MpReal derivative = 0;
        MpReal second_derivative = 0;
        for (int k = 0; k < period; ++k) {
            second_derivative = 2 * (
                derivative * derivative + z * second_derivative);
            derivative = 2 * z * derivative + 1;
            z = z * z + candidate;
        }
        const int sign = second_derivative > 0
            ? 1 : second_derivative < 0 ? -1 : 0;
        x = candidate;
        derivative_value = derivative;
        if (sign == 0) break;
        if (sign == sign_a) {
            a = candidate;
            sign_a = sign;
        } else {
            b = candidate;
        }
        if (b - a < pow(MpReal(10), -80)) break;
    }

    // Re-evaluate F'_n at the high-precision zero of F''_n.
    MpReal z = 0;
    derivative_value = 0;
    for (int k = 0; k < period; ++k) {
        derivative_value = 2 * z * derivative_value + 1;
        z = z * z + x;
    }
    const int derivative_sign = derivative_value > 0
        ? 1 : derivative_value < 0 ? -1 : 0;
    if (derivative_sign == 0) return std::nullopt;

    Real point = x.convert_to<Real>();
    auto usable = [&](Real candidate) {
        return candidate > left && candidate < right
            && critical_orbit_real_derivative_sign_mp<200>(
                   candidate, period) == derivative_sign;
    };
    if (usable(point)) {
        return std::pair<Real, int>{point, derivative_sign};
    }

    Real below = point;
    Real above = point;
    for (int step = 0; step < 64; ++step) {
        below = std::nextafter(below, left);
        if (usable(below)) {
            return std::pair<Real, int>{below, derivative_sign};
        }
        above = std::nextafter(above, right);
        if (usable(above)) {
            return std::pair<Real, int>{above, derivative_sign};
        }
    }
    return std::nullopt;
}

Real polish_real_root_bracket(
    RealRootBracket bracket,
    int period,
    const Config& config
) {
    if (bracket.left_sign == 0) return bracket.left;
    if (bracket.right_sign == 0) return bracket.right;
    if (bracket.left_sign == bracket.right_sign) {
        throw std::runtime_error("Real-root bracket does not change sign.");
    }

    Real x = (bracket.left + bracket.right) / 2;
    for (int iteration = 0;
         iteration < config.real_root_polish_iterations;
         ++iteration) {
        const auto orbit = critical_orbit_real_second(x, period);
        Real candidate = (bracket.left + bracket.right) / 2;
        if (finite(orbit.value)
            && finite(orbit.derivative)
            && std::abs(orbit.derivative) > 1.0e-30L) {
            const Real newton = x - orbit.value / orbit.derivative;
            if (finite(newton)
                && newton > bracket.left
                && newton < bracket.right) {
                candidate = newton;
            }
        }

        const int sign = critical_orbit_real_sign(candidate, period);
        if (sign == 0) return candidate;
        if (sign == bracket.left_sign) {
            bracket.left = candidate;
            bracket.left_sign = sign;
        } else {
            bracket.right = candidate;
            bracket.right_sign = sign;
        }
        x = candidate;
        if (bracket.right - bracket.left
            <= config.real_root_bracket_tolerance) {
            break;
        }
    }
    return (bracket.left + bracket.right) / 2;
}

struct RealRootPrepassResult {
    std::vector<Real> raw_roots;
    std::vector<Real> exact_roots;
    int grid_power = 0;
    std::size_t critical_pairs_recovered = 0;
};

Real lower_period_distance_score(Real root, int period) {
    Real score = std::numeric_limits<Real>::infinity();
    for (int d : proper_divisors(period)) {
        if (d <= 2) continue;
        const auto orbit = critical_orbit_real_second(root, d);
        if (!finite(orbit.value) || !finite(orbit.derivative)
            || std::abs(orbit.derivative) <= 1.0e-30L) {
            continue;
        }
        score = std::min(score, std::abs(orbit.value / orbit.derivative));
    }
    return score;
}

RealRootPrepassResult find_real_exact_period_roots(
    int period,
    const Config& config
) {
    if (period == 1) return {{0}, {0}, 0, 0};
    if (period == 2) return {{-1}, {-1}, 0, 0};

    const std::size_t expected_raw = real_raw_left_count(period);
    const std::size_t expected_exact = real_exact_center_count(period);
    if (expected_raw < expected_exact) {
        throw std::runtime_error("Real-root count arithmetic is inconsistent.");
    }

    std::cout << "  real-axis prepass: raw F_" << period
              << " on [-2,-5/4], expect " << expected_raw
              << " raw root(s), " << expected_exact
              << " exact-period root(s)\n"
              << "  real-axis coordinate: c=-2+(3/4)u^2"
              << " (quadratic dyadic density near -2)\n";

    int power = std::max(1, config.real_root_initial_grid_power);
    const long double requested_samples = std::max<long double>(
        1,
        static_cast<long double>(expected_raw)
            * config.real_root_samples_per_expected);
    while (power < config.real_root_max_grid_power
           && std::ldexp(1.0L, power) < requested_samples) {
        ++power;
    }

    std::vector<std::int8_t> signs;
    std::vector<RealRootBracket> final_brackets;
    std::vector<Real> exact_grid_roots;
    std::size_t recovered_pairs = 0;
    int final_power = power;
    const auto started = Clock::now();

    for (; power <= config.real_root_max_grid_power; ++power) {
        const std::size_t segments = std::size_t(1) << power;
        const std::string grid_label =
            "  real-axis grid 2^" + std::to_string(power);
        if (signs.empty()) {
            const auto allocation_started = Clock::now();
            render_inline_progress(
                config, grid_label, 0, 1, allocation_started, "allocating sign grid");
            signs.resize(segments + 1);
            render_inline_progress(
                config, grid_label, 1, 1, allocation_started,
                "allocated " + std::to_string(segments + 1) + " signs",
                true, false);
            parallel_for_progress(
                segments + 1,
                config.threads,
                config,
                grid_label,
                "orbit signs",
                [&](std::size_t j) {
                    signs[j] = static_cast<std::int8_t>(
                        critical_orbit_dyadic_sign(j, power, period));
                });
        } else {
            const auto allocation_started = Clock::now();
            render_inline_progress(
                config, grid_label, 0, 1, allocation_started, "allocating refined grid");
            std::vector<std::int8_t> refined(segments + 1);
            render_inline_progress(
                config, grid_label, 1, 1, allocation_started,
                "allocated " + std::to_string(segments + 1) + " signs",
                true, false);
            const std::size_t old_segments = segments / 2;
            parallel_for_progress(
                old_segments + 1,
                config.threads,
                config,
                grid_label,
                "copying previous signs",
                [&](std::size_t j) { refined[2 * j] = signs[j]; });
            parallel_for_progress(
                old_segments,
                config.threads,
                config,
                grid_label,
                "new midpoint signs",
                [&](std::size_t j) {
                    const std::size_t index = 2 * j + 1;
                    refined[index] = static_cast<std::int8_t>(
                        critical_orbit_dyadic_sign(index, power, period));
                });
            signs.swap(refined);
        }

        std::vector<RealRootBracket> brackets;
        std::vector<Real> zero_roots;
        brackets.reserve(expected_raw);
        zero_roots.reserve(4);
        const auto bracket_started = Clock::now();
        render_inline_progress(
            config,
            "  real-axis brackets",
            0,
            segments + 1,
            bracket_started,
            "grid=2^" + std::to_string(power));
        for (std::size_t j = 0; j <= segments; ++j) {
            if (signs[j] == 0) {
                zero_roots.push_back(real_grid_coordinate(j, power));
            }
            if (j == segments || signs[j] == 0 || signs[j + 1] == 0) {
                continue;
            }
            if (signs[j] != signs[j + 1]) {
                brackets.push_back({
                    real_grid_coordinate(j, power),
                    real_grid_coordinate(j + 1, power),
                    signs[j],
                    signs[j + 1],
                });
            }
            const std::size_t done = j + 1;
            if (done == segments + 1 || done % 8192 == 0) {
                render_inline_progress(
                    config,
                    "  real-axis brackets",
                    done,
                    segments + 1,
                    bracket_started,
                    "found="
                        + std::to_string(brackets.size() + zero_roots.size()),
                    done == segments + 1,
                    true);
            }
        }
        render_inline_progress(
            config,
            "  real-axis brackets",
            segments + 1,
            segments + 1,
            bracket_started,
            "found=" + std::to_string(brackets.size() + zero_roots.size()),
            true,
            true);

        std::size_t found = brackets.size() + zero_roots.size();
        if (found > expected_raw) {
            throw std::runtime_error(
                "Real-axis scan found more roots than the theoretical count.");
        }

        recovered_pairs = 0;
        const std::size_t missing = expected_raw - found;
        const std::size_t recovery_limit = std::max<std::size_t>(
            64,
            std::max<std::size_t>(1, expected_raw / 100));
        if (missing > 0
            && missing <= recovery_limit
            && power >= config.real_root_critical_recovery_power) {
            std::vector<std::int8_t> derivative_signs(segments + 1);
            parallel_for_progress(
                segments + 1,
                config.threads,
                config,
                "  critical recovery",
                "derivative signs",
                [&](std::size_t j) {
                    derivative_signs[j] = static_cast<std::int8_t>(
                        critical_orbit_real_derivative_sign(
                            real_grid_coordinate(j, power), period));
                });

            struct CriticalCandidate {
                std::size_t interval = 0;
                std::optional<Real> point;
            };
            std::vector<CriticalCandidate> candidates;
            const auto candidate_started = Clock::now();
            render_inline_progress(
                config, "  critical candidates", 0, segments, candidate_started);
            for (std::size_t j = 0; j < segments; ++j) {
                if (signs[j] != 0 && signs[j + 1] != 0
                    && signs[j] == signs[j + 1]
                    && derivative_signs[j] != 0
                    && derivative_signs[j + 1] != 0
                    && derivative_signs[j] != derivative_signs[j + 1]) {
                    candidates.push_back({j, std::nullopt});
                }
                const std::size_t done = j + 1;
                if (done == segments || done % 8192 == 0) {
                    render_inline_progress(
                        config,
                        "  critical candidates",
                        done,
                        segments,
                        candidate_started,
                        "found=" + std::to_string(candidates.size()),
                        done == segments,
                        true);
                }
            }

            parallel_for_progress(
                candidates.size(),
                config.threads,
                config,
                "  critical recovery",
                "polishing extrema",
                [&](std::size_t k) {
                    const std::size_t j = candidates[k].interval;
                    candidates[k].point = polish_real_critical_bracket(
                        real_grid_coordinate(j, power),
                        real_grid_coordinate(j + 1, power),
                        derivative_signs[j],
                        derivative_signs[j + 1],
                        period,
                        config);
                });

            std::vector<RealRootBracket> recovered;
            recovered.reserve(missing);
            std::vector<std::pair<Real, std::size_t>> mp_fallbacks;
            mp_fallbacks.reserve(candidates.size());

            auto append_recovered_pair = [&](std::size_t j,
                                             Real splitting_point,
                                             int critical_sign) {
                const int endpoint_sign = signs[j];
                const Real left = real_grid_coordinate(j, power);
                const Real right = real_grid_coordinate(j + 1, power);
                if (critical_sign == 0 || critical_sign == endpoint_sign
                    || !(splitting_point > left
                         && splitting_point < right)) {
                    return false;
                }
                recovered.push_back({
                    left, splitting_point, endpoint_sign, critical_sign});
                recovered.push_back({
                    splitting_point, right, critical_sign, endpoint_sign});
                return true;
            };

            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size();
                 ++candidate_index) {
                const auto& candidate = candidates[candidate_index];
                if (!candidate.point) continue;
                const std::size_t j = candidate.interval;
                const Real critical = *candidate.point;
                const int critical_sign =
                    critical_orbit_real_sign_mp<200>(critical, period);
                if (append_recovered_pair(j, critical, critical_sign)) {
                    continue;
                }
                const auto orbit = critical_orbit_real_second(
                    critical, period);
                const Real magnitude = finite(orbit.value)
                    ? std::abs(orbit.value)
                    : std::numeric_limits<Real>::infinity();
                mp_fallbacks.emplace_back(magnitude, candidate_index);
            }

            // A very narrow lobe can be missed because the long-double
            // critical point rounds just outside it. Refine only the most
            // promising extrema at 200 digits; hidden root pairs necessarily
            // live at extrema with unusually small |F_n|.
            if (found + recovered.size() < expected_raw) {
                const auto rank_started = Clock::now();
                render_inline_progress(
                    config, "  ranking critical fallbacks", 0, 1, rank_started,
                    std::to_string(mp_fallbacks.size()) + " extrema");
                std::sort(mp_fallbacks.begin(), mp_fallbacks.end(),
                    [](const auto& a, const auto& b) {
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second;
                    });
                render_inline_progress(
                    config, "  ranking critical fallbacks", 1, 1, rank_started,
                    std::to_string(mp_fallbacks.size()) + " extrema", true, false);
                // A global 2^(power+1) refinement is deliberately forbidden
                // above the configured ceiling: at power 30 it would require
                // more than two billion sign entries.  At the ceiling, exhaust
                // the already isolated critical-point candidates instead.
                // They are ranked by |F_n|, and each MP check is local and
                // constant-memory.  Earlier powers retain the small cap because
                // another cheap dyadic refinement is still available.
                const bool hard_grid_ceiling =
                    power == MAX_REAL_ROOT_GRID_POWER;
                const std::size_t mp_limit = std::min<std::size_t>(
                    mp_fallbacks.size(),
                    hard_grid_ceiling
                        ? mp_fallbacks.size()
                        : (missing <= 16
                            ? std::size_t(1024)
                            : std::max<std::size_t>(64, 16 * missing)));
                const std::string mp_label = hard_grid_ceiling
                    ? "  MP ceiling recovery"
                    : "  MP critical fallbacks";
                const auto mp_started = Clock::now();
                if (mp_limit > 0) {
                    render_inline_progress(
                        config, mp_label, 0, mp_limit, mp_started);
                }
                for (std::size_t rank = 0; rank < mp_limit; ++rank) {
                    const auto& candidate =
                        candidates[mp_fallbacks[rank].second];
                    const std::size_t j = candidate.interval;
                    const Real left = real_grid_coordinate(j, power);
                    const Real right = real_grid_coordinate(j + 1, power);
                    auto mp_critical = polish_real_critical_bracket_mp(
                        left,
                        right,
                        derivative_signs[j],
                        derivative_signs[j + 1],
                        period);
                    if (mp_critical) {
                        append_recovered_pair(
                            j, mp_critical->first, mp_critical->second);
                    }
                    const std::size_t done = rank + 1;
                    render_inline_progress(
                        config,
                        mp_label,
                        done,
                        mp_limit,
                        mp_started,
                        "recovered=" + std::to_string(recovered.size() / 2),
                        done == mp_limit
                            || found + recovered.size() >= expected_raw,
                        true);
                    if (found + recovered.size() >= expected_raw) break;
                }
            }

            if (power == MAX_REAL_ROOT_GRID_POWER
                && found + recovered.size() < expected_raw) {
                // One interval can contain two zeros of F'_n. In that case the
                // derivative signs at its endpoints agree and the first-level
                // scan above cannot see either critical point. Reuse the
                // derivative-sign byte array to retain both sign(F') and
                // sign(F'') without allocating another 2^30-element vector.
                parallel_for_progress(
                    segments + 1,
                    config.threads,
                    config,
                    "  nested critical recovery",
                    "second-derivative signs",
                    [&](std::size_t j) {
                        const int first_sign = derivative_signs[j];
                        const int second_sign =
                            critical_orbit_real_second_derivative_sign(
                                real_grid_coordinate(j, power), period);
                        derivative_signs[j] = static_cast<std::int8_t>(
                            (first_sign + 1) * 3 + (second_sign + 1));
                    });

                auto first_derivative_sign = [&](std::size_t j) {
                    return static_cast<int>(derivative_signs[j]) / 3 - 1;
                };
                auto second_derivative_sign = [&](std::size_t j) {
                    return static_cast<int>(derivative_signs[j]) % 3 - 1;
                };

                std::vector<std::size_t> nested_candidates;
                const auto nested_scan_started = Clock::now();
                render_inline_progress(
                    config,
                    "  nested critical candidates",
                    0,
                    segments,
                    nested_scan_started);
                for (std::size_t j = 0; j < segments; ++j) {
                    if (signs[j] != 0 && signs[j + 1] != 0
                        && signs[j] == signs[j + 1]
                        && first_derivative_sign(j) != 0
                        && first_derivative_sign(j + 1) != 0
                        && first_derivative_sign(j)
                            == first_derivative_sign(j + 1)
                        && second_derivative_sign(j) != 0
                        && second_derivative_sign(j + 1) != 0
                        && second_derivative_sign(j)
                            != second_derivative_sign(j + 1)) {
                        nested_candidates.push_back(j);
                    }
                    const std::size_t done = j + 1;
                    if (done == segments || done % 8192 == 0) {
                        render_inline_progress(
                            config,
                            "  nested critical candidates",
                            done,
                            segments,
                            nested_scan_started,
                            "found=" + std::to_string(
                                nested_candidates.size()),
                            done == segments,
                            true);
                    }
                }

                const auto nested_mp_started = Clock::now();
                if (!nested_candidates.empty()) {
                    render_inline_progress(
                        config,
                        "  MP nested critical recovery",
                        0,
                        nested_candidates.size(),
                        nested_mp_started);
                }
                for (std::size_t candidate_index = 0;
                     candidate_index < nested_candidates.size();
                     ++candidate_index) {
                    const std::size_t j =
                        nested_candidates[candidate_index];
                    const Real left = real_grid_coordinate(j, power);
                    const Real right = real_grid_coordinate(j + 1, power);
                    const int endpoint_derivative_sign =
                        first_derivative_sign(j);
                    const auto derivative_extremum =
                        polish_real_second_critical_bracket_mp(
                            left,
                            right,
                            second_derivative_sign(j),
                            second_derivative_sign(j + 1),
                            period);

                    if (derivative_extremum
                        && derivative_extremum->second
                            != endpoint_derivative_sign) {
                        const Real split = derivative_extremum->first;
                        const int split_derivative_sign =
                            derivative_extremum->second;
                        const auto first_critical =
                            polish_real_critical_bracket_mp(
                                left,
                                split,
                                endpoint_derivative_sign,
                                split_derivative_sign,
                                period);
                        const auto second_critical =
                            polish_real_critical_bracket_mp(
                                split,
                                right,
                                split_derivative_sign,
                                endpoint_derivative_sign,
                                period);
                        if (first_critical && second_critical
                            && first_critical->first
                                < second_critical->first) {
                            std::vector<RealRootBracket> nested_brackets;
                            auto append_sign_change = [&](
                                Real a,
                                int sign_a,
                                Real b,
                                int sign_b
                            ) {
                                if (a < b && sign_a != 0 && sign_b != 0
                                    && sign_a != sign_b) {
                                    nested_brackets.push_back({
                                        a, b, sign_a, sign_b});
                                }
                            };
                            append_sign_change(
                                left,
                                signs[j],
                                first_critical->first,
                                first_critical->second);
                            append_sign_change(
                                first_critical->first,
                                first_critical->second,
                                second_critical->first,
                                second_critical->second);
                            append_sign_change(
                                second_critical->first,
                                second_critical->second,
                                right,
                                signs[j + 1]);
                            if (nested_brackets.size() == 2
                                && found + recovered.size()
                                    + nested_brackets.size()
                                    <= expected_raw) {
                                recovered.insert(
                                    recovered.end(),
                                    nested_brackets.begin(),
                                    nested_brackets.end());
                            }
                        }
                    }

                    const std::size_t done = candidate_index + 1;
                    render_inline_progress(
                        config,
                        "  MP nested critical recovery",
                        done,
                        nested_candidates.size(),
                        nested_mp_started,
                        "recovered="
                            + std::to_string(recovered.size() / 2),
                        done == nested_candidates.size()
                            || found + recovered.size() >= expected_raw,
                        true);
                    if (found + recovered.size() >= expected_raw) break;
                }
            }

            std::cout << "      critical recovery: " << candidates.size()
                      << " derivative bracket(s), "
                      << recovered.size() / 2 << " root pair(s) confirmed\n";
            recovered_pairs = recovered.size() / 2;
            if (found + recovered.size() == expected_raw) {
                brackets.insert(
                    brackets.end(), recovered.begin(), recovered.end());
                found += recovered.size();
            }
        }

        const std::size_t certified_found = found == expected_raw
            ? found : found + 2 * recovered_pairs;
        std::cout << "    grid 2^" << power << ": " << certified_found << '/'
                  << expected_raw << " root bracket(s)";
        if (recovered_pairs > 0) {
            std::cout << ", recovered " << recovered_pairs
                      << " close pair(s) through critical points";
        }
        std::cout << " | elapsed "
                  << format_duration(Clock::now() - started) << '\n';

        if (found == expected_raw) {
            std::sort(zero_roots.begin(), zero_roots.end());
            exact_grid_roots = std::move(zero_roots);
            final_brackets = std::move(brackets);
            std::sort(final_brackets.begin(), final_brackets.end(),
                [](const RealRootBracket& a, const RealRootBracket& b) {
                    return a.left < b.left;
                });
            final_power = power;
            break;
        }

        if (power == config.real_root_max_grid_power) {
            std::ostringstream message;
            message << "Real-axis scan certified " << certified_found << " of "
                    << expected_raw << " required raw roots at grid power "
                    << power << ". ";
            if (power < MAX_REAL_ROOT_GRID_POWER) {
                message << "Increase real_root_max_grid_power up to "
                        << MAX_REAL_ROOT_GRID_POWER << '.';
            } else {
                message << "The maximum supported global grid has already "
                        << "been reached, and exhaustive targeted "
                        << "multiprecision critical-point recovery did not "
                        << "find the remaining "
                        << (expected_raw - certified_found)
                        << " root(s).";
            }
            throw std::runtime_error(message.str());
        }
    }

    std::vector<Real> raw_roots = exact_grid_roots;
    raw_roots.reserve(expected_raw);
    const auto polish_started = Clock::now();
    if (!final_brackets.empty()) {
        render_inline_progress(
            config,
            "  polishing real roots",
            0,
            final_brackets.size(),
            polish_started);
    }
    for (std::size_t i = 0; i < final_brackets.size(); ++i) {
        raw_roots.push_back(polish_real_root_bracket(
            final_brackets[i], period, config));
        const std::size_t done = i + 1;
        if (done == final_brackets.size() || done % 128 == 0) {
            render_inline_progress(
                config,
                "  polishing real roots",
                done,
                final_brackets.size(),
                polish_started,
                {},
                done == final_brackets.size(),
                true);
        }
    }
    std::sort(raw_roots.begin(), raw_roots.end());

    std::vector<Real> unique_raw;
    unique_raw.reserve(raw_roots.size());
    const Real dedup_tolerance = std::max<Real>(
        8 * std::numeric_limits<Real>::epsilon(),
        config.real_root_bracket_tolerance / 4);
    for (Real root : raw_roots) {
        if (!unique_raw.empty()
            && std::abs(root - unique_raw.back()) <= dedup_tolerance) {
            continue;
        }
        unique_raw.push_back(root);
    }
    if (unique_raw.size() != expected_raw) {
        std::ostringstream message;
        message << "Real-axis polishing produced " << unique_raw.size()
                << " distinct raw roots; expected " << expected_raw << '.';
        throw std::runtime_error(message.str());
    }

    const std::size_t lower_count = expected_raw - expected_exact;
    std::vector<bool> lower(unique_raw.size(), false);
    Real largest_lower_score = 0;
    Real smallest_exact_score = std::numeric_limits<Real>::infinity();
    if (lower_count > 0) {
        std::vector<std::pair<Real, std::size_t>> scores;
        scores.reserve(unique_raw.size());
        const auto score_started = Clock::now();
        render_inline_progress(
            config,
            "  classifying real roots",
            0,
            unique_raw.size(),
            score_started);
        for (std::size_t i = 0; i < unique_raw.size(); ++i) {
            scores.emplace_back(
                lower_period_distance_score(unique_raw[i], period), i);
            const std::size_t done = i + 1;
            if (done == unique_raw.size() || done % 128 == 0) {
                render_inline_progress(
                    config,
                    "  classifying real roots",
                    done,
                    unique_raw.size(),
                    score_started,
                    {},
                    done == unique_raw.size(),
                    true);
            }
        }
        std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
        for (std::size_t i = 0; i < lower_count; ++i) {
            if (!finite(scores[i].first)) {
                throw std::runtime_error(
                    "Could not identify lower-period real roots.");
            }
            lower[scores[i].second] = true;
            largest_lower_score = std::max(
                largest_lower_score, scores[i].first);
        }
        if (lower_count < scores.size()) {
            smallest_exact_score = scores[lower_count].first;
        }
    }

    std::vector<Real> exact_roots;
    exact_roots.reserve(expected_exact);
    for (std::size_t i = 0; i < unique_raw.size(); ++i) {
        if (!lower[i]) exact_roots.push_back(unique_raw[i]);
    }
    if (exact_roots.size() != expected_exact) {
        throw std::runtime_error(
            "Exact-period real-root selection did not match R(n).");
    }

    std::cout << "  real-axis prepass complete: " << unique_raw.size()
              << " raw F_" << period << " roots, "
              << exact_roots.size() << " exact-period real roots";
    if (lower_count > 0) {
        std::cout << ", lower-period score boundary "
                  << real_string(largest_lower_score, 3) << " / "
                  << real_string(smallest_exact_score, 3);
    }
    std::cout << '\n';

    return {
        std::move(unique_raw),
        std::move(exact_roots),
        final_power,
        recovered_pairs,
    };
}

// -----------------------------------------------------------------------------
// Uniform interaction grids
// -----------------------------------------------------------------------------

inline Complex inverse_difference(const Complex& difference) {
    const Real denominator = difference.real() * difference.real()
        + difference.imag() * difference.imag();
    if (!(denominator > 1.0e-48L) || !finite(denominator)) {
        return Complex{0, 0};
    }
    return Complex{
        difference.real() / denominator,
        -difference.imag() / denominator,
    };
}

struct RootGridCell {
    std::vector<std::size_t> roots;
    Complex centroid{0, 0};
    Complex second_moment{0, 0};
    Real radius = 0;
};

struct RootInteractionGrid {
    int nx = 1;
    int ny = 1;
    Real xmin = 0;
    Real ymin = 0;
    Real dx = 1;
    Real dy = 1;
    std::vector<RootGridCell> cells;
    std::vector<int> root_cell_x;
    std::vector<int> root_cell_y;

    std::size_t cell_index(int x, int y) const {
        return static_cast<std::size_t>(y * nx + x);
    }
};

struct RootBounds {
    Real xmin = 0;
    Real xmax = 0;
    Real ymin = 0;
    Real ymax = 0;
};

RootBounds root_bounds(const std::vector<Complex>& roots) {
    if (roots.empty()) return {};
    RootBounds bounds{
        roots.front().real(), roots.front().real(),
        roots.front().imag(), roots.front().imag(),
    };
    for (const auto& root : roots) {
        bounds.xmin = std::min(bounds.xmin, root.real());
        bounds.xmax = std::max(bounds.xmax, root.real());
        bounds.ymin = std::min(bounds.ymin, root.imag());
        bounds.ymax = std::max(bounds.ymax, root.imag());
    }
    return bounds;
}

int automatic_root_grid_y(const std::vector<Complex>& roots, int grid_x) {
    const auto bounds = root_bounds(roots);
    const Real width = std::max<Real>(bounds.xmax - bounds.xmin, 1.0e-18L);
    const Real height = std::max<Real>(bounds.ymax - bounds.ymin, 1.0e-18L);
    const Real estimate = static_cast<Real>(grid_x) * height / width;
    return std::clamp(
        static_cast<int>(std::llround(estimate)),
        1,
        std::max(1, 4 * grid_x));
}

RootInteractionGrid build_root_interaction_grid(
    const std::vector<Complex>& roots,
    int nx,
    int ny
) {
    RootInteractionGrid grid;
    grid.nx = std::max(1, nx);
    grid.ny = std::max(1, ny);
    grid.cells.resize(static_cast<std::size_t>(grid.nx * grid.ny));
    grid.root_cell_x.resize(roots.size());
    grid.root_cell_y.resize(roots.size());

    auto bounds = root_bounds(roots);
    Real width = bounds.xmax - bounds.xmin;
    Real height = bounds.ymax - bounds.ymin;
    if (!(width > 0)) width = 1;
    if (!(height > 0)) height = 1;
    const Real pad_x = std::max<Real>(
        1.0e-18L,
        64 * std::numeric_limits<Real>::epsilon() * width);
    const Real pad_y = std::max<Real>(
        1.0e-18L,
        64 * std::numeric_limits<Real>::epsilon() * height);
    grid.xmin = bounds.xmin - pad_x;
    grid.ymin = bounds.ymin - pad_y;
    width += 2 * pad_x;
    height += 2 * pad_y;
    grid.dx = width / static_cast<Real>(grid.nx);
    grid.dy = height / static_cast<Real>(grid.ny);

    auto coordinate = [](Real value, Real minimum, Real step, int count) {
        int index = static_cast<int>(std::floor((value - minimum) / step));
        return std::clamp(index, 0, count - 1);
    };

    for (std::size_t i = 0; i < roots.size(); ++i) {
        const int x = coordinate(roots[i].real(), grid.xmin, grid.dx, grid.nx);
        const int y = coordinate(roots[i].imag(), grid.ymin, grid.dy, grid.ny);
        grid.root_cell_x[i] = x;
        grid.root_cell_y[i] = y;
        grid.cells[grid.cell_index(x, y)].roots.push_back(i);
    }

    for (auto& cell : grid.cells) {
        if (cell.roots.empty()) continue;
        Complex sum{0, 0};
        for (std::size_t index : cell.roots) sum += roots[index];
        cell.centroid = sum / static_cast<Real>(cell.roots.size());
        for (std::size_t index : cell.roots) {
            const Complex offset = roots[index] - cell.centroid;
            cell.second_moment += offset * offset;
            cell.radius = std::max(cell.radius, safe_abs(offset));
        }
    }
    return grid;
}

struct RealGridCell {
    std::vector<std::size_t> roots;
    Real centroid = 0;
    Real second_moment = 0;
    Real radius = 0;
};

struct RealInteractionGrid {
    int nx = 1;
    Real xmin = -2;
    Real dx = 0.75L;
    std::vector<RealGridCell> cells;
};

RealInteractionGrid build_real_interaction_grid(
    const std::vector<Real>& roots,
    int nx
) {
    RealInteractionGrid grid;
    grid.nx = std::max(1, nx);
    grid.xmin = -2.0L;
    grid.dx = 0.75L / static_cast<Real>(grid.nx);
    grid.cells.resize(static_cast<std::size_t>(grid.nx));
    for (std::size_t i = 0; i < roots.size(); ++i) {
        int cell = static_cast<int>(std::floor(
            (roots[i] - grid.xmin) / grid.dx));
        cell = std::clamp(cell, 0, grid.nx - 1);
        grid.cells[static_cast<std::size_t>(cell)].roots.push_back(i);
    }
    for (auto& cell : grid.cells) {
        if (cell.roots.empty()) continue;
        for (std::size_t index : cell.roots) cell.centroid += roots[index];
        cell.centroid /= static_cast<Real>(cell.roots.size());
        for (std::size_t index : cell.roots) {
            const Real offset = roots[index] - cell.centroid;
            cell.second_moment += offset * offset;
            cell.radius = std::max(cell.radius, std::abs(offset));
        }
    }
    return grid;
}

Complex cell_multipole(
    Complex z,
    std::size_t count,
    Complex centroid,
    Complex second_moment,
    int order
) {
    const Complex separation = z - centroid;
    Complex result = static_cast<Real>(count) / separation;
    if (order >= 2) {
        const Complex squared = separation * separation;
        result += second_moment / (squared * separation);
    }
    return result;
}

Complex exact_root_repulsion(
    std::size_t target,
    Complex z,
    const std::vector<Complex>& roots
) {
    Complex result{0, 0};
    for (std::size_t j = 0; j < roots.size(); ++j) {
        if (j == target) continue;
        result += inverse_difference(z - roots[j]);
    }
    return result;
}

Complex grid_root_repulsion(
    std::size_t target,
    Complex z,
    const std::vector<Complex>& roots,
    const RootInteractionGrid& grid,
    const Config& config
) {
    Complex result{0, 0};
    const int target_x = grid.root_cell_x[target];
    const int target_y = grid.root_cell_y[target];
    for (int y = 0; y < grid.ny; ++y) {
        for (int x = 0; x < grid.nx; ++x) {
            const auto& cell = grid.cells[grid.cell_index(x, y)];
            if (cell.roots.empty()) continue;
            const int cell_distance = std::max(
                std::abs(x - target_x),
                std::abs(y - target_y));
            const Real distance = safe_abs(z - cell.centroid);
            const bool exact = cell_distance <= config.root_grid_near_cells
                || static_cast<int>(cell.roots.size())
                    < config.root_grid_far_min_roots
                || !(distance > 1.0e-24L)
                || cell.radius / distance > config.root_grid_opening_angle;
            if (exact) {
                for (std::size_t j : cell.roots) {
                    if (j != target) {
                        result += inverse_difference(z - roots[j]);
                    }
                }
            } else {
                result += cell_multipole(
                    z,
                    cell.roots.size(),
                    cell.centroid,
                    cell.second_moment,
                    config.root_grid_multipole_order);
            }
        }
    }
    return result;
}

Complex exact_symmetric_repulsion(
    std::size_t target,
    Complex z,
    const std::vector<Complex>& upper_roots,
    const std::vector<Real>& real_roots
) {
    Complex result{0, 0};
    for (Real root : real_roots) {
        result += inverse_difference(z - Complex{root, 0});
    }
    for (std::size_t j = 0; j < upper_roots.size(); ++j) {
        if (j != target) {
            result += inverse_difference(z - upper_roots[j]);
        }
        // Includes j==target: a root always feels its own distinct conjugate.
        result += inverse_difference(z - std::conj(upper_roots[j]));
    }
    return result;
}

Complex grid_real_repulsion(
    Complex z,
    const std::vector<Real>& real_roots,
    const RealInteractionGrid& grid,
    const Config& config
) {
    Complex result{0, 0};
    int target_x = static_cast<int>(std::floor(
        (z.real() - grid.xmin) / grid.dx));
    target_x = std::clamp(target_x, 0, grid.nx - 1);
    for (int x = 0; x < grid.nx; ++x) {
        const auto& cell = grid.cells[static_cast<std::size_t>(x)];
        if (cell.roots.empty()) continue;
        const Complex centroid{cell.centroid, 0};
        const Real distance = safe_abs(z - centroid);
        const bool exact = std::abs(x - target_x)
                <= config.root_grid_near_cells
            || static_cast<int>(cell.roots.size())
                < config.root_grid_far_min_roots
            || !(distance > 1.0e-24L)
            || cell.radius / distance > config.root_grid_opening_angle;
        if (exact) {
            for (std::size_t index : cell.roots) {
                result += inverse_difference(
                    z - Complex{real_roots[index], 0});
            }
        } else {
            result += cell_multipole(
                z,
                cell.roots.size(),
                centroid,
                Complex{cell.second_moment, 0},
                config.root_grid_multipole_order);
        }
    }
    return result;
}

Complex grid_symmetric_repulsion(
    std::size_t target,
    Complex z,
    const std::vector<Complex>& upper_roots,
    const std::vector<Real>& real_roots,
    const RootInteractionGrid& upper_grid,
    const RealInteractionGrid& real_grid,
    const Config& config
) {
    Complex result = grid_real_repulsion(
        z, real_roots, real_grid, config);
    const int target_x = upper_grid.root_cell_x[target];
    const int target_y = upper_grid.root_cell_y[target];

    for (int y = 0; y < upper_grid.ny; ++y) {
        for (int x = 0; x < upper_grid.nx; ++x) {
            const auto& cell =
                upper_grid.cells[upper_grid.cell_index(x, y)];
            if (cell.roots.empty()) continue;

            const int cell_distance = std::max(
                std::abs(x - target_x),
                std::abs(y - target_y));
            const Complex upper_separation = z - cell.centroid;
            const Complex mirror_centroid = std::conj(cell.centroid);
            const Complex mirror_separation = z - mirror_centroid;
            const Real upper_distance = safe_abs(upper_separation);
            const Real mirror_distance = safe_abs(mirror_separation);
            const bool too_small = static_cast<int>(cell.roots.size())
                < config.root_grid_far_min_roots;
            const bool contains_target = x == target_x && y == target_y;
            const bool upper_exact = cell_distance
                    <= config.root_grid_near_cells
                || too_small
                || !(upper_distance > 1.0e-24L)
                || cell.radius / upper_distance
                    > config.root_grid_opening_angle;
            const bool mirror_exact = contains_target
                || too_small
                || !(mirror_distance > 1.0e-24L)
                || cell.radius / mirror_distance
                    > config.root_grid_opening_angle;

            if (!upper_exact && !mirror_exact) {
                result += cell_multipole(
                    z,
                    cell.roots.size(),
                    cell.centroid,
                    cell.second_moment,
                    config.root_grid_multipole_order);
                result += cell_multipole(
                    z,
                    cell.roots.size(),
                    mirror_centroid,
                    std::conj(cell.second_moment),
                    config.root_grid_multipole_order);
                continue;
            }

            if (upper_exact) {
                for (std::size_t j : cell.roots) {
                    if (j != target) {
                        result += inverse_difference(z - upper_roots[j]);
                    }
                }
            } else {
                result += cell_multipole(
                    z,
                    cell.roots.size(),
                    cell.centroid,
                    cell.second_moment,
                    config.root_grid_multipole_order);
            }

            if (mirror_exact) {
                for (std::size_t j : cell.roots) {
                    result += inverse_difference(
                        z - std::conj(upper_roots[j]));
                }
            } else {
                result += cell_multipole(
                    z,
                    cell.roots.size(),
                    mirror_centroid,
                    std::conj(cell.second_moment),
                    config.root_grid_multipole_order);
            }
        }
    }
    return result;
}


// -----------------------------------------------------------------------------
// Adaptive Barnes-Hut interaction tree
// -----------------------------------------------------------------------------

struct BarnesHutNode {
    std::size_t begin = 0;
    std::size_t end = 0;
    Real center_x = 0;
    Real center_y = 0;
    Real half_size = 0;
    Complex centroid{0, 0};
    Complex second_moment{0, 0};
    Real radius = 0;
    std::array<int, 4> children{{-1, -1, -1, -1}};

    std::size_t count() const { return end - begin; }
    bool leaf() const {
        return children[0] < 0 && children[1] < 0
            && children[2] < 0 && children[3] < 0;
    }
};

class BarnesHutTree {
public:
    BarnesHutTree() = default;

    BarnesHutTree(
        const std::vector<Complex>& points,
        int leaf_size,
        int maximum_depth,
        int multipole_order
    ) {
        rebuild(points, leaf_size, maximum_depth, multipole_order);
    }

    void rebuild(
        const std::vector<Complex>& points,
        int leaf_size,
        int maximum_depth,
        int multipole_order
    ) {
        points_ = &points;
        leaf_size_ = std::max(1, leaf_size);
        maximum_depth_ = std::max(1, maximum_depth);
        multipole_order_ = multipole_order;
        order_.resize(points.size());
        scratch_.resize(points.size());
        std::iota(order_.begin(), order_.end(), 0);
        nodes_.clear();
        if (points.empty()) return;

        RootBounds bounds = root_bounds(points);
        const Real width = std::max<Real>(
            bounds.xmax - bounds.xmin, 1.0e-18L);
        const Real height = std::max<Real>(
            bounds.ymax - bounds.ymin, 1.0e-18L);
        const Real half = 0.5L * std::max(width, height)
            * (1 + 256 * std::numeric_limits<Real>::epsilon())
            + 1.0e-18L;
        const Real center_x = 0.5L * (bounds.xmin + bounds.xmax);
        const Real center_y = 0.5L * (bounds.ymin + bounds.ymax);
        nodes_.reserve(std::max<std::size_t>(
            1, 2 * points.size() / static_cast<std::size_t>(leaf_size_) + 8));
        build_node(0, points.size(), center_x, center_y, half, 0);
    }

    bool empty() const { return nodes_.empty(); }

    Complex repulsion(
        std::size_t target,
        Complex z,
        Real opening_angle,
        bool mirrored = false
    ) const {
        if (nodes_.empty()) return Complex{0, 0};
        return accumulate_repulsion(
            0, target, z, opening_angle, mirrored);
    }

    Real nearest_distance(std::size_t target, Complex z) const {
        if (nodes_.empty()) {
            return std::numeric_limits<Real>::infinity();
        }
        Real best = std::numeric_limits<Real>::infinity();
        nearest_recursive(0, target, z, best);
        return best;
    }

    std::size_t node_count() const { return nodes_.size(); }

private:
    const std::vector<Complex>* points_ = nullptr;
    int leaf_size_ = 48;
    int maximum_depth_ = 32;
    int multipole_order_ = 2;
    std::vector<std::size_t> order_;
    std::vector<std::size_t> scratch_;
    std::vector<BarnesHutNode> nodes_;

    int quadrant(Complex value, Real center_x, Real center_y) const {
        const int right = value.real() >= center_x ? 1 : 0;
        const int upper = value.imag() >= center_y ? 1 : 0;
        return right + 2 * upper;
    }

    int build_node(
        std::size_t begin,
        std::size_t end,
        Real center_x,
        Real center_y,
        Real half_size,
        int depth
    ) {
        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back(BarnesHutNode{});
        BarnesHutNode& created = nodes_.back();
        created.begin = begin;
        created.end = end;
        created.center_x = center_x;
        created.center_y = center_y;
        created.half_size = half_size;

        Complex sum{0, 0};
        for (std::size_t position = begin; position < end; ++position) {
            sum += (*points_)[order_[position]];
        }
        if (end > begin) {
            created.centroid = sum / static_cast<Real>(end - begin);
        }
        for (std::size_t position = begin; position < end; ++position) {
            const Complex offset =
                (*points_)[order_[position]] - created.centroid;
            created.second_moment += offset * offset;
            created.radius = std::max(created.radius, safe_abs(offset));
        }

        const std::size_t count = end - begin;
        if (count <= static_cast<std::size_t>(leaf_size_)
            || depth >= maximum_depth_
            || !(half_size > 1.0e-24L)) {
            return node_index;
        }

        std::array<std::size_t, 4> counts{{0, 0, 0, 0}};
        for (std::size_t position = begin; position < end; ++position) {
            ++counts[static_cast<std::size_t>(quadrant(
                (*points_)[order_[position]], center_x, center_y))];
        }

        int nonempty = 0;
        for (std::size_t count_in_quadrant : counts) {
            if (count_in_quadrant > 0) ++nonempty;
        }
        if (nonempty <= 1 && half_size <= 1.0e-16L) {
            return node_index;
        }

        std::array<std::size_t, 4> offsets{};
        offsets[0] = begin;
        for (int q = 1; q < 4; ++q) {
            offsets[static_cast<std::size_t>(q)] =
                offsets[static_cast<std::size_t>(q - 1)]
                + counts[static_cast<std::size_t>(q - 1)];
        }
        auto cursors = offsets;
        for (std::size_t position = begin; position < end; ++position) {
            const std::size_t index = order_[position];
            const int q = quadrant((*points_)[index], center_x, center_y);
            scratch_[cursors[static_cast<std::size_t>(q)]++] = index;
        }
        std::copy(
            scratch_.begin() + static_cast<std::ptrdiff_t>(begin),
            scratch_.begin() + static_cast<std::ptrdiff_t>(end),
            order_.begin() + static_cast<std::ptrdiff_t>(begin));

        const Real child_half = 0.5L * half_size;
        std::array<int, 4> children{{-1, -1, -1, -1}};
        for (int q = 0; q < 4; ++q) {
            const std::size_t child_begin =
                offsets[static_cast<std::size_t>(q)];
            const std::size_t child_end =
                child_begin + counts[static_cast<std::size_t>(q)];
            if (child_begin == child_end) continue;
            const Real child_x = center_x
                + (q & 1 ? child_half : -child_half);
            const Real child_y = center_y
                + (q & 2 ? child_half : -child_half);
            children[static_cast<std::size_t>(q)] = build_node(
                child_begin,
                child_end,
                child_x,
                child_y,
                child_half,
                depth + 1);
        }
        nodes_[static_cast<std::size_t>(node_index)].children = children;
        return node_index;
    }

    bool node_contains_target(
        const BarnesHutNode& node,
        Complex target_value
    ) const {
        const Real slack = 16 * std::numeric_limits<Real>::epsilon()
            * std::max<Real>(1, node.half_size);
        return std::abs(target_value.real() - node.center_x)
                    <= node.half_size + slack
            && std::abs(target_value.imag() - node.center_y)
                    <= node.half_size + slack;
    }

    Complex node_multipole(
        const BarnesHutNode& node,
        Complex z,
        bool mirrored
    ) const {
        const Complex centroid = mirrored
            ? std::conj(node.centroid)
            : node.centroid;
        const Complex second_moment = mirrored
            ? std::conj(node.second_moment)
            : node.second_moment;
        return cell_multipole(
            z,
            node.count(),
            centroid,
            second_moment,
            multipole_order_);
    }

    Complex accumulate_repulsion(
        int node_index,
        std::size_t target,
        Complex z,
        Real opening_angle,
        bool mirrored
    ) const {
        const BarnesHutNode& node =
            nodes_[static_cast<std::size_t>(node_index)];
        const Complex centroid = mirrored
            ? std::conj(node.centroid)
            : node.centroid;
        const Real distance = safe_abs(z - centroid);
        const Complex target_value =
            target < points_->size() ? (*points_)[target] : z;
        const bool contains_target = !mirrored
            && target < points_->size()
            && node_contains_target(node, target_value);

        if (!node.leaf()
            && !contains_target
            && distance > 1.0e-24L
            && node.radius / distance <= opening_angle) {
            return node_multipole(node, z, mirrored);
        }

        if (node.leaf()) {
            Complex result{0, 0};
            for (std::size_t position = node.begin;
                 position < node.end;
                 ++position) {
                const std::size_t index = order_[position];
                if (!mirrored && index == target) continue;
                const Complex source = mirrored
                    ? std::conj((*points_)[index])
                    : (*points_)[index];
                result += inverse_difference(z - source);
            }
            return result;
        }

        Complex result{0, 0};
        for (int child : node.children) {
            if (child >= 0) {
                result += accumulate_repulsion(
                    child, target, z, opening_angle, mirrored);
            }
        }
        return result;
    }

    Real box_distance(const BarnesHutNode& node, Complex z) const {
        const Real dx = std::max<Real>(
            0, std::abs(z.real() - node.center_x) - node.half_size);
        const Real dy = std::max<Real>(
            0, std::abs(z.imag() - node.center_y) - node.half_size);
        return std::hypot(dx, dy);
    }

    void nearest_recursive(
        int node_index,
        std::size_t target,
        Complex z,
        Real& best
    ) const {
        const BarnesHutNode& node =
            nodes_[static_cast<std::size_t>(node_index)];
        if (box_distance(node, z) >= best) return;

        if (node.leaf()) {
            for (std::size_t position = node.begin;
                 position < node.end;
                 ++position) {
                const std::size_t index = order_[position];
                if (index == target) continue;
                best = std::min(best, safe_abs(z - (*points_)[index]));
            }
            return;
        }

        std::array<std::pair<Real, int>, 4> ordered_children{{
            {std::numeric_limits<Real>::infinity(), -1},
            {std::numeric_limits<Real>::infinity(), -1},
            {std::numeric_limits<Real>::infinity(), -1},
            {std::numeric_limits<Real>::infinity(), -1},
        }};
        int count = 0;
        for (int child : node.children) {
            if (child < 0) continue;
            ordered_children[static_cast<std::size_t>(count++)] = {
                box_distance(
                    nodes_[static_cast<std::size_t>(child)], z),
                child,
            };
        }
        // At most four children. A hand-written insertion sort avoids a GCC 13
        // false-positive -Warray-bounds warning inside libstdc++'s std::sort.
        for (int i = 1; i < count; ++i) {
            const auto value =
                ordered_children[static_cast<std::size_t>(i)];

            int j = i;
            while (
                j > 0
                && ordered_children[static_cast<std::size_t>(j - 1)].first
                    > value.first
            ) {
                ordered_children[static_cast<std::size_t>(j)] =
                    ordered_children[static_cast<std::size_t>(j - 1)];
                --j;
            }

            ordered_children[static_cast<std::size_t>(j)] = value;
        }
        for (int i = 0; i < count; ++i) {
            if (ordered_children[static_cast<std::size_t>(i)].first >= best) {
                break;
            }
            nearest_recursive(
                ordered_children[static_cast<std::size_t>(i)].second,
                target,
                z,
                best);
        }
    }
};

Real dynamic_tree_opening_angle(
    Real correction_metric,
    Real active_fraction,
    const Config& config
) {
    Real progress = 0;
    const Real start = config.root_tree_tighten_start_delta;
    const Real finish = config.root_tree_tighten_end_delta;
    if (finite(correction_metric) && correction_metric > 0
        && start > finish && finish > 0) {
        progress = std::log10(start / std::max(correction_metric, finish))
            / std::log10(start / finish);
        progress = std::clamp(progress, static_cast<Real>(0), static_cast<Real>(1));
    }
    Real theta = config.root_tree_theta_initial
        + progress * (
            config.root_tree_theta_final - config.root_tree_theta_initial);
    if (active_fraction <= config.root_cluster_phase_fraction) {
        theta = std::min(theta, config.root_cluster_theta);
    }
    return std::clamp(theta, static_cast<Real>(0.05), static_cast<Real>(0.95));
}

Complex tree_symmetric_repulsion(
    std::size_t target,
    Complex z,
    const BarnesHutTree& upper_tree,
    const BarnesHutTree& real_tree,
    Real opening_angle
) {
    return upper_tree.repulsion(target, z, opening_angle, false)
         + upper_tree.repulsion(target, z, opening_angle, true)
         + real_tree.repulsion(
             std::numeric_limits<std::size_t>::max(),
             z,
             opening_angle,
             false);
}

std::vector<Complex> warm_initial_upper_root_cloud(
    std::size_t count,
    const Config& config,
    const std::vector<Complex>& previous_centers
) {
    std::vector<Complex> previous_upper;
    previous_upper.reserve(previous_centers.size() / 2 + 1);
    for (Complex center : previous_centers) {
        if (center.imag() > 1.0e-15L && finite(center)) {
            previous_upper.push_back(center);
        }
    }
    if (!config.root_warm_start_previous_period
        || previous_upper.empty()
        || count == 0) {
        return initial_upper_root_cloud(count, config);
    }

    BarnesHutTree previous_tree(
        previous_upper,
        config.root_tree_leaf_size,
        config.root_tree_max_depth,
        config.root_tree_multipole_order);

    const std::size_t warm_count = std::min(
        count,
        static_cast<std::size_t>(std::llround(
            config.root_warm_start_fraction
            * static_cast<Real>(count))));
    std::vector<Complex> roots;
    roots.reserve(count);
    const auto started = Clock::now();
    render_inline_progress(
        config,
        "  warm-starting upper roots",
        0,
        count,
        started,
        "previous period density scaffold");

    for (std::size_t i = 0; i < warm_count; ++i) {
        const std::size_t base_index = i % previous_upper.size();
        const std::size_t generation = i / previous_upper.size();
        const Complex base = previous_upper[base_index];
        Real spacing = previous_tree.nearest_distance(base_index, base);
        if (!finite(spacing) || !(spacing > 0)) spacing = 0.02L;
        spacing = std::clamp(spacing, 1.0e-8L, 0.12L);

        Real angle = GOLDEN_ANGLE
            * static_cast<Real>(base_index + 1)
            + static_cast<Real>(generation) * PI;
        if (generation >= 2) {
            angle += GOLDEN_ANGLE * static_cast<Real>(generation);
        }
        const Real magnitude = std::clamp(
            config.root_warm_start_jitter * spacing,
            2.0e-9L,
            0.03L);
        Complex seed = base + magnitude * Complex{
            std::cos(angle), std::sin(angle)};
        if (!(seed.imag() > 0)) {
            seed = Complex{
                seed.real(),
                std::max<Real>(
                    1.0e-8L,
                    base.imag() * 0.35L + magnitude * 0.25L)};
        }
        const Real radius = safe_abs(seed);
        if (radius > config.root_bound) {
            seed *= config.root_bound / radius;
            seed = Complex{seed.real(), std::max<Real>(seed.imag(), 1.0e-8L)};
        }
        roots.push_back(seed);
        if ((i + 1) % 2048 == 0 || i + 1 == count) {
            render_inline_progress(
                config,
                "  warm-starting upper roots",
                i + 1,
                count,
                started,
                "previous period density scaffold",
                i + 1 == count,
                true);
        }
    }

    if (roots.size() < count) {
        const std::size_t remainder = count - roots.size();
        auto generic = initial_upper_root_cloud(remainder, config);
        roots.insert(roots.end(), generic.begin(), generic.end());
    }
    render_inline_progress(
        config,
        "  warm-starting upper roots",
        count,
        count,
        started,
        std::to_string(warm_count) + " warm + "
            + std::to_string(count - warm_count) + " generic",
        true,
        true);
    return roots;
}

struct RootCorrectionSummary {
    Real maximum = 0;
    Real q99 = 0;
    Real median = 0;
    std::size_t maximum_index = 0;
};

RootCorrectionSummary summarize_active_corrections(
    const std::vector<Real>& corrections,
    const std::vector<std::size_t>& active_indices
) {
    RootCorrectionSummary result;
    if (active_indices.empty()) return result;
    std::vector<Real> values;
    values.reserve(active_indices.size());
    result.maximum = -1;
    for (std::size_t index : active_indices) {
        const Real value = corrections[index];
        values.push_back(value);
        if (value > result.maximum) {
            result.maximum = value;
            result.maximum_index = index;
        }
    }
    auto select = [&](Real q) {
        const std::size_t position = static_cast<std::size_t>(std::llround(
            q * static_cast<Real>(values.size() - 1)));
        std::vector<Real> copy = values;
        std::nth_element(
            copy.begin(),
            copy.begin() + static_cast<std::ptrdiff_t>(position),
            copy.end());
        return copy[position];
    };
    result.q99 = select(0.99L);
    result.median = select(0.5L);
    return result;
}

struct EarlyPolishSweep {
    std::size_t moved = 0;
    std::size_t frozen = 0;
};

EarlyPolishSweep early_polish_isolated_roots(
    int period,
    const PeriodArithmetic& arithmetic,
    std::vector<Complex>& roots,
    std::vector<unsigned char>& active,
    const BarnesHutTree& tree,
    const BarnesHutTree& real_tree,
    const Config& config
) {
    (void)period;
    EarlyPolishSweep result;
    if (!config.root_early_polish_enabled) return result;

    std::vector<std::size_t> active_indices;
    active_indices.reserve(roots.size());
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (active[i]) active_indices.push_back(i);
    }
    std::vector<Complex> candidates(roots.size());
    std::vector<unsigned char> status(roots.size(), 0);
    std::vector<Real> ratios(
        roots.size(), std::numeric_limits<Real>::infinity());

    // Candidates may return immediately or take as many as eight Newton
    // steps plus spatial-neighbour queries. Claim modest batches dynamically
    // so dense difficult regions cannot strand otherwise idle workers.
    parallel_for_dynamic(
        active_indices.size(),
        config.threads,
        32,
        [&](std::size_t position) {
            const std::size_t i = active_indices[position];
            const Complex original = roots[i];
            auto correction =
                exact_period_newton_correction(original, arithmetic);
            if (!correction) return;
            const Real correction_size = safe_abs(*correction);
            if (!(correction_size <= config.root_early_polish_start_delta)) {
                return;
            }

            const Real nearest_upper =
                tree.nearest_distance(i, original);
            const Real nearest_mirror = 2 * original.imag();
            const Real nearest_real = real_tree.nearest_distance(
                std::numeric_limits<std::size_t>::max(), original);
            const Real nearest = std::min({
                nearest_upper, nearest_mirror, nearest_real});
            // Use the same distinctness boundary as final center validation.
            // The former 16x multiplier permanently stranded valid high-period
            // roots whose spacing was between 1x and 16x the duplicate
            // tolerance. Basin safety is already enforced independently by the
            // Newton-correction/nearest-neighbour ratio below.
            const Real minimum_separation = std::max<Real>(
                config.center_duplicate_tolerance, 1.0e-14L);
            if (!finite(nearest) || !(nearest > minimum_separation)) return;
            const Real ratio = correction_size / nearest;
            ratios[i] = ratio;
            if (!(ratio <= config.root_early_polish_ratio)) return;

            Complex c = original;
            Real travelled = 0;
            bool improved = false;
            for (int step = 0; step < config.root_early_polish_steps; ++step) {
                auto local = exact_period_newton_correction(c, arithmetic);
                if (!local) break;
                Real size = safe_abs(*local);
                if (!finite(size)) break;
                if (size <= config.root_early_polish_tolerance) {
                    improved = true;
                    break;
                }
                const Real maximum_step = 0.20L * nearest;
                if (size > maximum_step) {
                    *local *= maximum_step / size;
                    size = maximum_step;
                }
                if (travelled + size > 0.35L * nearest) break;
                const Complex candidate = c - *local;
                if (!(candidate.imag() > 0) || !finite(candidate)) break;
                c = candidate;
                travelled += size;
                improved = true;
            }
            if (!improved) return;

            auto final_correction =
                exact_period_newton_correction(c, arithmetic);
            if (!final_correction) return;
            const Real final_size = safe_abs(*final_correction);
            if (!finite(final_size)) return;

            candidates[i] = c;
            status[i] = final_size <= config.root_early_polish_tolerance
                     && final_size / nearest
                            <= 0.25L * config.root_early_polish_ratio
                ? 2
                : 1;
        });

    for (std::size_t i : active_indices) {
        if (status[i] == 0) continue;
        roots[i] = candidates[i];
        ++result.moved;
        if (status[i] == 2) {
            active[i] = 0;
            ++result.frozen;
        }
    }
    return result;
}

Real resolved_exact_start_delta(const Config& config) {
    if (config.root_grid_exact_start_delta > 0) {
        return config.root_grid_exact_start_delta;
    }
    return std::sqrt(config.root_tolerance);
}

int scheduled_exact_interval(Real maximum, const Config& config) {
    if (!(config.root_grid_enabled || config.root_tree_enabled)
        || config.root_grid_exact_steps_every <= 1) {
        return 1;
    }
    const Real start = resolved_exact_start_delta(config);
    if (!(maximum <= start)) return 0;
    const int initial_every = std::max(
        config.root_grid_exact_steps_every,
        config.root_grid_exact_start_every);
    const int final_every = config.root_grid_exact_steps_every;
    if (initial_every == final_every) return final_every;

    const Real denominator = std::log10(start / config.root_tolerance);
    Real progress = denominator > 0
        ? std::log10(start / std::max(maximum, config.root_tolerance))
            / denominator
        : 1;
    progress = std::clamp(progress, static_cast<Real>(0), static_cast<Real>(1));
    const Real log_initial = std::log(static_cast<Real>(initial_every));
    const Real log_final = std::log(static_cast<Real>(final_every));
    return std::max(
        final_every,
        static_cast<int>(std::llround(std::exp(
            log_initial + progress * (log_final - log_initial)))));
}

// Once only a small unresolved cluster remains, its exact cadence must no
// longer depend on q99. For tiny populations q99 collapses to the maximum, so
// one clipped or oscillating root can otherwise prevent exact mode forever.
int small_cluster_exact_interval(
    std::size_t active_count,
    const Config& config
) {
    if (active_count <= 64) return 1;
    if (active_count <= 256) return 2;
    if (active_count <= 1024) return 5;
    return std::max(1, config.root_grid_exact_steps_every);
}

void validate_root_grid_config(const Config& config) {
    if (config.root_grid_x < 1) {
        throw std::runtime_error("root_grid_x must be at least 1");
    }
    if (config.root_grid_y < 0) {
        throw std::runtime_error(
            "root_grid_y must be zero (auto) or positive");
    }
    if (config.root_grid_near_cells < 0) {
        throw std::runtime_error(
            "root_grid_near_cells must be non-negative");
    }
    if (!(config.root_grid_opening_angle > 0)
        || !(config.root_grid_opening_angle < 1)) {
        throw std::runtime_error(
            "root_grid_opening_angle must lie strictly between 0 and 1");
    }
    if (config.root_grid_multipole_order != 0
        && config.root_grid_multipole_order != 2) {
        throw std::runtime_error(
            "root_grid_multipole_order must be 0 or 2");
    }
    if (config.root_grid_far_min_roots < 1) {
        throw std::runtime_error(
            "root_grid_far_min_roots must be at least 1");
    }
    if (config.root_tree_leaf_size < 2) {
        throw std::runtime_error(
            "root_tree_leaf_size must be at least 2");
    }
    if (config.root_tree_max_depth < 1 || config.root_tree_max_depth > 64) {
        throw std::runtime_error(
            "root_tree_max_depth must lie in [1,64]");
    }
    if (!(config.root_tree_theta_initial > 0)
        || !(config.root_tree_theta_initial < 1)
        || !(config.root_tree_theta_final > 0)
        || !(config.root_tree_theta_final < 1)) {
        throw std::runtime_error(
            "root-tree opening angles must lie strictly between 0 and 1");
    }
    if (config.root_tree_theta_initial < config.root_tree_theta_final) {
        throw std::runtime_error(
            "root_tree_theta_initial must be >= root_tree_theta_final");
    }
    if (!(config.root_tree_tighten_start_delta > 0)
        || !(config.root_tree_tighten_end_delta > 0)
        || config.root_tree_tighten_start_delta
            <= config.root_tree_tighten_end_delta) {
        throw std::runtime_error(
            "root-tree tightening deltas must be positive and decreasing");
    }
    if (config.root_tree_multipole_order != 0
        && config.root_tree_multipole_order != 2) {
        throw std::runtime_error(
            "root_tree_multipole_order must be 0 or 2");
    }
    if (!(config.root_warm_start_fraction >= 0)
        || !(config.root_warm_start_fraction <= 1)
        || !(config.root_warm_start_jitter > 0)) {
        throw std::runtime_error(
            "invalid warm-start fraction or jitter");
    }
    if (config.root_isolation_check_every < 1
        || config.root_early_polish_steps < 1
        || !(config.root_early_polish_start_delta > 0)
        || !(config.root_early_polish_ratio > 0)
        || !(config.root_early_polish_ratio < 0.5L)
        || !(config.root_early_polish_tolerance > 0)) {
        throw std::runtime_error(
            "invalid early-polish configuration");
    }
    if (!(config.root_cluster_phase_fraction > 0)
        || !(config.root_cluster_phase_fraction < 1)
        || !(config.root_cluster_theta > 0)
        || !(config.root_cluster_theta < 1)
        || config.root_exact_active_limit < 0
        || !(config.root_cluster_max_step > 0)
        || config.root_cluster_max_step > config.root_max_step) {
        throw std::runtime_error(
            "invalid unresolved-cluster configuration: root_cluster_max_step "
            "must be positive and no larger than root_max_step");
    }
    if (!(config.root_tolerance > 0)) {
        throw std::runtime_error("root_tolerance must be positive");
    }
    if (config.root_progress_every < 1) {
        throw std::runtime_error("root_progress_every must be at least 1");
    }
    if (config.root_grid_exact_start_delta < 0) {
        throw std::runtime_error(
            "root_grid_exact_start_delta cannot be negative");
    }
    if (config.root_grid_exact_start_delta > 0
        && config.root_grid_exact_start_delta <= config.root_tolerance) {
        throw std::runtime_error(
            "root_grid_exact_start_delta must exceed root_tolerance, or be 0 for auto");
    }
    if (config.root_grid_exact_start_every < 1
        || config.root_grid_exact_steps_every < 1) {
        throw std::runtime_error(
            "root-grid exact-step cadences must be at least 1");
    }
    if (config.root_grid_exact_start_every
        < config.root_grid_exact_steps_every) {
        throw std::runtime_error(
            "root_grid_exact_start_every must be >= root_grid_exact_steps_every");
    }
    if (!(config.root_upper_axis_step_fraction > 0)
        || !(config.root_upper_axis_step_fraction < 1)) {
        throw std::runtime_error(
            "root_upper_axis_step_fraction must lie strictly between 0 and 1");
    }
    if (config.real_root_initial_grid_power < 1
        || config.real_root_initial_grid_power > MAX_REAL_ROOT_GRID_POWER) {
        throw std::runtime_error(
            "real_root_initial_grid_power must lie in [1,30]");
    }
    if (config.real_root_samples_per_expected < 1) {
        throw std::runtime_error(
            "real_root_samples_per_expected must be at least 1");
    }
    if (config.real_root_critical_recovery_power < 1
        || config.real_root_critical_recovery_power
            > MAX_REAL_ROOT_GRID_POWER) {
        throw std::runtime_error(
            "real_root_critical_recovery_power must lie in [1,30]");
    }
    if (config.real_root_max_grid_power
        < config.real_root_initial_grid_power
        || config.real_root_max_grid_power > MAX_REAL_ROOT_GRID_POWER) {
        throw std::runtime_error(
            "real_root_max_grid_power must be >= the initial power and <=30");
    }
    if (config.real_root_polish_iterations < 1
        || !(config.real_root_bracket_tolerance > 0)) {
        throw std::runtime_error(
            "Invalid real-root polishing configuration");
    }
}

struct RootSolveResult {
    std::vector<Complex> roots;
    int iterations = 0;
    Real max_correction = std::numeric_limits<Real>::infinity();
    bool globally_converged = false;
};

RootSolveResult solve_center_roots_full(
    int period,
    const Config& config,
    const fs::path& checkpoint_path
) {
    const auto& arithmetic = period_arithmetic(period);
    const std::size_t count = center_count(period);
    std::vector<Complex> roots;
    int start_iteration = 0;
    bool resumed_checkpoint = false;

    if (config.resume) {
        if (auto checkpoint = read_root_checkpoint_full(
                checkpoint_path, period, count, config)) {
            roots = std::move(checkpoint->roots);
            start_iteration = checkpoint->iteration;
            resumed_checkpoint = true;
            std::cout << "  resuming full root cloud at iteration "
                      << start_iteration << " from " << checkpoint_path << '\n';
        }
    }
    if (roots.empty()) roots = initial_root_cloud(count, config);

    const bool grid_active = config.root_grid_enabled
        && config.root_grid_exact_steps_every != 1;
    const int resolved_grid_y = config.root_grid_y > 0
        ? config.root_grid_y
        : automatic_root_grid_y(roots, config.root_grid_x);
    const Real exact_start_delta = resolved_exact_start_delta(config);
    std::vector<Complex> next(count);
    std::vector<Real> corrections(
        count, std::numeric_limits<Real>::infinity());
    const auto started = Clock::now();
    Real maximum = std::numeric_limits<Real>::infinity();
    bool converged = false;
    int completed_iteration = start_iteration;
    int last_exact_iteration = start_iteration;

    std::cout << "  root solver: full-plane damped Aberth-Ehrlich, long double, "
              << config.threads << " thread(s), target Δ="
              << real_string(config.root_tolerance, 3) << '\n';
    if (grid_active) {
        std::cout << "  repulsion: uniform grid " << config.root_grid_x << 'x'
                  << resolved_grid_y << ", exact checks below Δ="
                  << real_string(exact_start_delta, 3) << '\n';
    }

    for (int iteration = start_iteration + 1;
         iteration <= config.root_max_iterations;
         ++iteration) {
        const auto old = roots;
        const int exact_interval = scheduled_exact_interval(maximum, config);
        const bool exact_step = !grid_active
            || (resumed_checkpoint && iteration == start_iteration + 1)
            || maximum <= config.root_tolerance
            || (exact_interval > 0
                && iteration - last_exact_iteration >= exact_interval);
        std::optional<RootInteractionGrid> grid;
        if (!exact_step) {
            grid.emplace(build_root_interaction_grid(
                old, config.root_grid_x, resolved_grid_y));
        }

        parallel_root_iteration(
            count,
            config.threads,
            [&](std::size_t i) {
                    const Complex zi = old[i];
                    auto newton = exact_period_newton_correction(zi, arithmetic);
                    if (!newton) {
                        const Real angle = GOLDEN_ANGLE
                            * static_cast<Real>(i + iteration);
                        const Complex perturb{
                            1.0e-8L * std::cos(angle),
                            1.0e-8L * std::sin(angle)};
                        next[i] = zi + perturb;
                        corrections[i] = safe_abs(perturb);
                        return;
                    }
                    const Complex repulsion = exact_step
                        ? exact_root_repulsion(i, zi, old)
                        : grid_root_repulsion(i, zi, old, *grid, config);
                    const Complex denominator =
                        Complex{1, 0} - (*newton) * repulsion;
                    Complex delta = safe_abs(denominator) > 1.0e-24L
                        ? (*newton) / denominator
                        : *newton;
                    Real delta_size = safe_abs(delta);
                    if (!finite(delta_size)) {
                        delta = *newton;
                        delta_size = safe_abs(delta);
                    }
                    if (delta_size > config.root_max_step) {
                        delta *= config.root_max_step / delta_size;
                        delta_size = config.root_max_step;
                    }
                    Complex candidate = zi - delta;
                    const Real radius = safe_abs(candidate);
                    if (radius > config.root_bound) {
                        candidate *= config.root_bound / radius;
                    }
                    next[i] = candidate;
                    corrections[i] = delta_size;
            });

        roots.swap(next);
        maximum = *std::max_element(corrections.begin(), corrections.end());
        completed_iteration = iteration;
        if (exact_step) last_exact_iteration = iteration;
        if (config.root_checkpoint_every > 0
            && iteration % config.root_checkpoint_every == 0) {
            write_root_checkpoint_full(
                checkpoint_path, period, iteration, roots, config);
        }
        if (iteration % config.root_progress_every == 0
            || (exact_step && maximum <= config.root_tolerance)) {
            std::ostringstream detail;
            detail << "current Δ=" << std::setprecision(5) << std::scientific
                   << maximum;
            if (config.progress && config.progress_style == "bars") {
                render_inline_progress(
                    config,
                    "  roots",
                    static_cast<std::size_t>(iteration),
                    static_cast<std::size_t>(config.root_max_iterations),
                    started,
                    detail.str(),
                    false,
                    false);
            } else if (config.progress_style == "lines") {
                std::cout << "  roots " << std::setw(5) << iteration << '/'
                          << config.root_max_iterations << " | "
                          << detail.str() << " | elapsed "
                          << format_duration(Clock::now() - started) << '\n';
            }
        }
        if (exact_step && maximum <= config.root_tolerance) {
            converged = true;
            break;
        }
    }
    if (config.progress && config.progress_style == "bars") {
        std::ostringstream detail;
        detail << "current Δ=" << std::setprecision(5) << std::scientific
               << maximum << " | "
               << (converged ? "converged" : "iteration limit");
        render_inline_progress(
            config,
            "  roots",
            static_cast<std::size_t>(completed_iteration),
            static_cast<std::size_t>(config.root_max_iterations),
            started,
            detail.str(),
            true,
            false);
    }
    write_root_checkpoint_full(
        checkpoint_path, period, completed_iteration, roots, config);
    return {std::move(roots), completed_iteration, maximum, converged};
}

std::vector<Complex> expand_symmetric_root_cloud(
    const std::vector<Real>& real_roots,
    const std::vector<Complex>& upper_roots,
    const Config& config
) {
    const std::size_t total = real_roots.size() + 2 * upper_roots.size();
    std::vector<Complex> result;
    result.reserve(total);
    const auto started = Clock::now();
    if (total > 0) {
        render_inline_progress(
            config, "  expanding conjugate roots", 0, total, started);
    }
    std::size_t done = 0;
    auto append = [&](Complex root) {
        result.push_back(root);
        ++done;
        if (done == total || done % 2048 == 0) {
            render_inline_progress(
                config,
                "  expanding conjugate roots",
                done,
                total,
                started,
                {},
                done == total,
                true);
        }
    };
    for (Real root : real_roots) append(Complex{root, 0});
    for (const auto& root : upper_roots) append(root);
    for (const auto& root : upper_roots) append(std::conj(root));
    return result;
}

RootSolveResult solve_center_roots_symmetric(
    int period,
    const Config& config,
    const fs::path& checkpoint_path,
    const std::vector<Complex>& warm_start_centers
) {
    const auto& arithmetic = period_arithmetic(period);
    const std::size_t total_count = center_count(period);
    const std::size_t expected_real_count = real_exact_center_count(period);
    if (total_count < expected_real_count
        || (total_count - expected_real_count) % 2 != 0) {
        throw std::runtime_error(
            "Total and real root counts are incompatible with conjugation.");
    }
    const std::size_t upper_count =
        (total_count - expected_real_count) / 2;

    std::vector<Real> real_roots;
    std::vector<Complex> upper_roots;
    int start_iteration = 0;
    bool resumed_checkpoint = false;
    if (config.resume) {
        if (auto checkpoint = read_root_checkpoint_symmetric(
                checkpoint_path,
                period,
                expected_real_count,
                upper_count,
                config)) {
            real_roots = std::move(checkpoint->real_roots);
            upper_roots = std::move(checkpoint->roots);
            start_iteration = checkpoint->iteration;
            resumed_checkpoint = true;
            std::cout << "  resuming upper-half root cloud at iteration "
                      << start_iteration << " from " << checkpoint_path << '\n';
        }
    }

    if (real_roots.empty()) {
        real_roots = find_real_exact_period_roots(period, config).exact_roots;
    }
    if (real_roots.size() != expected_real_count) {
        throw std::runtime_error(
            "Real-axis prepass did not produce R(n) roots.");
    }
    if (upper_roots.empty()) {
        upper_roots = warm_initial_upper_root_cloud(
            upper_count, config, warm_start_centers);
    }

    std::cout << "  dynamic roots: " << upper_count
              << " upper-half representative(s); "
              << real_roots.size() << " fixed real root(s); "
              << upper_count << " implicit conjugate(s)\n";

    const bool tree_active = config.root_tree_enabled;
    const bool legacy_grid_active = !tree_active
        && config.root_grid_enabled
        && config.root_grid_exact_steps_every != 1;
    const int resolved_grid_y = config.root_grid_y > 0
        ? config.root_grid_y
        : automatic_root_grid_y(upper_roots, config.root_grid_x);
    const Real exact_start_delta = resolved_exact_start_delta(config);
    const RealInteractionGrid legacy_real_grid = build_real_interaction_grid(
        real_roots, config.root_grid_x);

    std::vector<Complex> real_points;
    real_points.reserve(real_roots.size());
    for (Real root : real_roots) real_points.emplace_back(root, 0);
    const BarnesHutTree real_tree(
        real_points,
        std::max(8, config.root_tree_leaf_size),
        config.root_tree_max_depth,
        config.root_tree_multipole_order);

    std::vector<Complex> next(upper_count);
    std::vector<Real> corrections(upper_count, 0);
    std::vector<unsigned char> active(upper_count, 1);
    const auto started = Clock::now();
    Real maximum = std::numeric_limits<Real>::infinity();
    Real schedule_metric = std::numeric_limits<Real>::infinity();
    bool converged = false;
    int completed_iteration = start_iteration;
    int last_exact_iteration = start_iteration;
    int last_checkpoint_iteration = start_iteration;
    std::size_t total_frozen = 0;
    std::size_t remaining_active = upper_count;

    std::cout << "  root solver: hybrid conjugate-symmetric Aberth-Ehrlich + "
              << "early exact-period Newton, long double, " << config.threads
              << " thread(s), target Δ="
              << real_string(config.root_tolerance, 3) << '\n';
    if (tree_active) {
        std::cout << "  repulsion: adaptive Barnes-Hut quadtree, leaf<="
                  << config.root_tree_leaf_size
                  << ", depth<=" << config.root_tree_max_depth
                  << ", order " << config.root_tree_multipole_order
                  << ", theta " << compact_real(
                         config.root_tree_theta_initial, 3)
                  << " -> " << compact_real(
                         config.root_tree_theta_final, 3) << '\n'
                  << "  basin isolation: every "
                  << config.root_isolation_check_every
                  << " iteration(s), |Newton|/nearest <= "
                  << compact_real(config.root_early_polish_ratio, 3)
                  << ", cluster phase below "
                  << percent_string(config.root_cluster_phase_fraction)
                  << " active\n"
                  << "  exact active-root checks: begin below Δ="
                  << real_string(exact_start_delta, 3)
                  << ", cadence " << config.root_grid_exact_start_every
                  << " -> " << config.root_grid_exact_steps_every
                  << ", only when active <= "
                  << config.root_exact_active_limit << '\n'
                  << "  cluster safeguards: step cap="
                  << compact_real(config.root_cluster_max_step, 4)
                  << " below "
                  << percent_string(config.root_cluster_phase_fraction)
                  << " active; exact cadence "
                  << "<=64:1, <=256:2, <=1024:5, <=limit:"
                  << config.root_grid_exact_steps_every
                  << '\n';
    } else if (legacy_grid_active) {
        std::cout << "  repulsion: upper grid " << config.root_grid_x << 'x'
                  << resolved_grid_y
                  << " with separately tested mirror cells, order "
                  << config.root_grid_multipole_order
                  << "; fixed real roots use " << config.root_grid_x
                  << " x-bins\n";
    } else {
        std::cout << "  repulsion: exact conjugate-symmetric all-pairs every iteration\n";
    }

    const bool root_dashboard =
        config.progress
        && config.progress_style == "bars"
        && config.progress_screen == "alternate"
        && stderr_is_terminal();
    std::cout.flush();
    AlternateScreenGuard root_screen(root_dashboard);

    RootProgressDashboardState dashboard_state;
    dashboard_state.period = period;
    dashboard_state.start_iteration = start_iteration;
    dashboard_state.iteration = start_iteration;
    dashboard_state.maximum_iterations = config.root_max_iterations;
    dashboard_state.last_checkpoint_iteration =
        last_checkpoint_iteration;
    dashboard_state.checkpoint_every = config.root_checkpoint_every;
    dashboard_state.active_roots = upper_count;
    dashboard_state.total_roots = upper_count;
    dashboard_state.fixed_real_roots = real_roots.size();
    dashboard_state.full_root_count = total_count;
    dashboard_state.target_correction = config.root_tolerance;
    dashboard_state.step_cap = config.root_max_step;
    dashboard_state.started = started;
    std::vector<FreezeProgressSample> freeze_history;
    freeze_history.reserve(17);

    // Active/frozen flags are deliberately derived state rather than part of
    // the checkpoint format. Rebuild them from the saved roots before taking
    // another Aberth step. In particular, do not let a resumed high-q99 cloud
    // take one global 0.2 step with every previously frozen root reactivated.
    if (resumed_checkpoint && config.root_early_polish_enabled) {
        BarnesHutTree resume_tree(
            upper_roots,
            config.root_tree_leaf_size,
            config.root_tree_max_depth,
            config.root_tree_multipole_order);
        const EarlyPolishSweep resume_sweep =
            early_polish_isolated_roots(
                period,
                arithmetic,
                upper_roots,
                active,
                resume_tree,
                real_tree,
                config);
        total_frozen = resume_sweep.frozen;
        remaining_active = upper_count - total_frozen;
        freeze_history.push_back({start_iteration, total_frozen});
        dashboard_state.active_roots = remaining_active;
        dashboard_state.total_frozen = total_frozen;
        dashboard_state.latest_polish_iteration = start_iteration;
        dashboard_state.latest_polish_moved = resume_sweep.moved;
        dashboard_state.latest_polish_frozen = resume_sweep.frozen;
        dashboard_state.cluster_phase =
            static_cast<Real>(remaining_active)
                / static_cast<Real>(
                    std::max<std::size_t>(1, upper_count))
            <= config.root_cluster_phase_fraction;
        dashboard_state.step_cap = dashboard_state.cluster_phase
            ? std::min(
                  config.root_max_step, config.root_cluster_max_step)
            : config.root_max_step;
    }
    if (root_dashboard) {
        render_root_progress_dashboard(config, dashboard_state);
    }

    for (int iteration = start_iteration + 1;
         iteration <= config.root_max_iterations;
         ++iteration) {
        std::vector<std::size_t> active_indices;
        active_indices.reserve(upper_count - total_frozen);
        for (std::size_t i = 0; i < upper_count; ++i) {
            if (active[i]) active_indices.push_back(i);
        }
        if (active_indices.empty()) {
            converged = true;
            completed_iteration = iteration - 1;
            maximum = 0;
            break;
        }

        const Real active_fraction =
            static_cast<Real>(active_indices.size())
            / static_cast<Real>(std::max<std::size_t>(1, upper_count));
        const bool cluster_phase =
            active_fraction <= config.root_cluster_phase_fraction;
        const Real opening_angle = dynamic_tree_opening_angle(
            schedule_metric, active_fraction, config);
        const bool small_cluster =
            config.root_exact_active_limit > 0
            && active_indices.size()
                <= static_cast<std::size_t>(
                    config.root_exact_active_limit);
        int exact_interval =
            scheduled_exact_interval(schedule_metric, config);
        if (small_cluster) {
            // Do not let a single q99/max outlier suppress exact repulsion.
            exact_interval = small_cluster_exact_interval(
                active_indices.size(), config);
        }
        const bool exact_allowed = !tree_active || small_cluster;
        const bool exact_step = (!tree_active && !legacy_grid_active)
            || (exact_allowed
                && (schedule_metric <= config.root_tolerance
                    || (exact_interval > 0
                        && iteration - last_exact_iteration
                            >= exact_interval)));
        const Real iteration_max_step = cluster_phase
            ? std::min(config.root_max_step, config.root_cluster_max_step)
            : config.root_max_step;

        const auto old = upper_roots;
        next = old;
        std::fill(corrections.begin(), corrections.end(), 0);

        std::optional<BarnesHutTree> upper_tree;
        std::optional<RootInteractionGrid> legacy_grid;
        if (tree_active) {
            upper_tree.emplace(
                old,
                config.root_tree_leaf_size,
                config.root_tree_max_depth,
                config.root_tree_multipole_order);
        } else if (legacy_grid_active && !exact_step) {
            legacy_grid.emplace(build_root_interaction_grid(
                old, config.root_grid_x, resolved_grid_y));
        }

        parallel_root_iteration(
            active_indices.size(),
            config.threads,
            [&](std::size_t active_position) {
                const std::size_t i = active_indices[active_position];
                const Complex zi = old[i];
                auto newton = exact_period_newton_correction(zi, arithmetic);
                if (!newton) {
                    const Real angle = GOLDEN_ANGLE
                        * static_cast<Real>(i + iteration);
                    Complex perturb{
                        1.0e-8L * std::cos(angle),
                        std::abs(1.0e-8L * std::sin(angle))};
                    next[i] = zi + perturb;
                    corrections[i] = safe_abs(perturb);
                    return;
                }

                Complex repulsion{0, 0};
                if (exact_step) {
                    repulsion = exact_symmetric_repulsion(
                        i, zi, old, real_roots);
                } else if (tree_active) {
                    repulsion = tree_symmetric_repulsion(
                        i, zi, *upper_tree, real_tree, opening_angle);
                } else {
                    repulsion = grid_symmetric_repulsion(
                        i,
                        zi,
                        old,
                        real_roots,
                        *legacy_grid,
                        legacy_real_grid,
                        config);
                }

                const Complex denominator =
                    Complex{1, 0} - (*newton) * repulsion;
                Complex delta = safe_abs(denominator) > 1.0e-24L
                    ? (*newton) / denominator
                    : *newton;
                Real delta_size = safe_abs(delta);
                if (!finite(delta_size)) {
                    delta = *newton;
                    delta_size = safe_abs(delta);
                }
                if (delta_size > iteration_max_step) {
                    delta *= iteration_max_step / delta_size;
                    delta_size = iteration_max_step;
                }

                // Keep the stored representative in the upper half-plane.
                if (delta.imag() > 0) {
                    const Real maximum_downward =
                        config.root_upper_axis_step_fraction * zi.imag();
                    if (delta.imag() > maximum_downward) {
                        const Real scale =
                            maximum_downward / delta.imag();
                        delta *= scale;
                        delta_size = safe_abs(delta);
                    }
                }
                Complex candidate = zi - delta;
                if (!(candidate.imag() > 0)) {
                    candidate = Complex{
                        candidate.real(),
                        std::max<Real>(
                            zi.imag()
                                * (1 - config.root_upper_axis_step_fraction),
                            1.0e-20L)};
                    delta_size = safe_abs(zi - candidate);
                }
                const Real radius = safe_abs(candidate);
                if (radius > config.root_bound) {
                    candidate *= config.root_bound / radius;
                    if (!(candidate.imag() > 0)) {
                        candidate = Complex{
                            candidate.real(), 1.0e-20L};
                    }
                    delta_size = safe_abs(zi - candidate);
                }
                next[i] = candidate;
                corrections[i] = delta_size;
            });

        upper_roots.swap(next);
        RootCorrectionSummary summary =
            summarize_active_corrections(corrections, active_indices);
        maximum = summary.maximum;
        schedule_metric = summary.q99;
        completed_iteration = iteration;
        if (exact_step) last_exact_iteration = iteration;

        EarlyPolishSweep sweep;
        const bool isolation_cadence_due =
            iteration % config.root_isolation_check_every == 0
            || small_cluster;
        const bool polish_due = config.root_early_polish_enabled
            && isolation_cadence_due
            && (cluster_phase
                || schedule_metric
                    <= config.root_early_polish_start_delta);
        if (polish_due) {
            BarnesHutTree polish_tree(
                upper_roots,
                config.root_tree_leaf_size,
                config.root_tree_max_depth,
                config.root_tree_multipole_order);
            sweep = early_polish_isolated_roots(
                period,
                arithmetic,
                upper_roots,
                active,
                polish_tree,
                real_tree,
                config);
            total_frozen += sweep.frozen;
        }

        remaining_active = 0;
        for (unsigned char state : active) {
            if (state) ++remaining_active;
        }
        if (remaining_active == 0) {
            converged = true;
            maximum = 0;
        }

        if (polish_due
            && (freeze_history.empty()
                || iteration - freeze_history.back().iteration
                    >= config.root_isolation_check_every
                || converged)) {
            freeze_history.push_back({iteration, total_frozen});
            constexpr std::size_t MAX_FREEZE_SAMPLES = 17;
            if (freeze_history.size() > MAX_FREEZE_SAMPLES) {
                freeze_history.erase(freeze_history.begin());
            }
            dashboard_state.freeze_projection =
                estimate_freeze_projection(
                    freeze_history, remaining_active);
        }

        if (config.root_checkpoint_every > 0
            && iteration % config.root_checkpoint_every == 0) {
            write_root_checkpoint_symmetric(
                checkpoint_path,
                period,
                iteration,
                real_roots,
                upper_roots,
                config);
            last_checkpoint_iteration = iteration;
        }

        if (iteration % config.root_progress_every == 0
            || converged) {
            const Real remaining_fraction =
                static_cast<Real>(remaining_active)
                / static_cast<Real>(std::max<std::size_t>(1, upper_count));
            const bool cluster_phase_now =
                remaining_fraction <= config.root_cluster_phase_fraction;

            std::ostringstream interaction_mode;
            if (exact_step) {
                interaction_mode << "exact all-pairs";
            } else if (tree_active) {
                interaction_mode << "Barnes-Hut(theta="
                                 << std::setprecision(3)
                                 << std::defaultfloat << opening_angle << ')';
            } else {
                interaction_mode << "interaction grid";
            }

            std::optional<std::pair<std::size_t, Complex>> worst_root;
            if (!active_indices.empty()
                && summary.maximum_index < old.size()) {
                worst_root = std::make_pair(
                    summary.maximum_index,
                    old[summary.maximum_index]);
            }

            std::ostringstream detail;
            detail << "delta_max=" << std::setprecision(4) << std::scientific
                   << summary.maximum
                   << " q99=" << summary.q99
                   << " med=" << summary.median
                   << " active=" << remaining_active << '/' << upper_count
                   << " phase="
                   << (cluster_phase_now ? "cluster-AE" : "global-AE")
                   << " step<=" << compact_real(iteration_max_step, 3)
                   << " mode=" << interaction_mode.str();
            if (sweep.moved > 0) {
                detail << " polish=" << sweep.moved
                       << " freeze=" << sweep.frozen;
            }
            if (worst_root) {
                detail << " max#" << summary.maximum_index
                       << '@' << compact_real(
                              old[summary.maximum_index].real(), 5)
                       << (old[summary.maximum_index].imag() >= 0 ? "+" : "")
                       << compact_real(
                              old[summary.maximum_index].imag(), 5)
                       << 'i';
            }

            if (!finite(dashboard_state.initial_maximum_correction)
                && finite(summary.maximum)
                && summary.maximum > config.root_tolerance) {
                dashboard_state.initial_maximum_correction =
                    summary.maximum;
            }
            dashboard_state.iteration = iteration;
            dashboard_state.last_checkpoint_iteration =
                last_checkpoint_iteration;
            dashboard_state.active_roots = remaining_active;
            dashboard_state.total_frozen = total_frozen;
            if (polish_due) {
                dashboard_state.latest_polish_iteration = iteration;
                dashboard_state.latest_polish_moved = sweep.moved;
                dashboard_state.latest_polish_frozen = sweep.frozen;
            }
            dashboard_state.maximum_correction =
                converged ? 0 : summary.maximum;
            dashboard_state.q99_correction = summary.q99;
            dashboard_state.median_correction = summary.median;
            dashboard_state.step_cap = iteration_max_step;
            dashboard_state.cluster_phase = cluster_phase_now;
            dashboard_state.exact_step = exact_step;
            dashboard_state.converged = converged;
            dashboard_state.interaction_mode =
                interaction_mode.str();
            dashboard_state.worst_root = worst_root;

            if (root_dashboard) {
                render_root_progress_dashboard(
                    config, dashboard_state);
            } else if (config.progress
                       && config.progress_style == "bars") {
                render_inline_progress(
                    config,
                    "  upper roots",
                    static_cast<std::size_t>(iteration),
                    static_cast<std::size_t>(config.root_max_iterations),
                    started,
                    detail.str(),
                    converged,
                    false);
            } else if (config.progress_style == "lines") {
                std::cout << "  upper roots " << std::setw(7) << iteration
                          << '/' << config.root_max_iterations << " | "
                          << detail.str() << " | elapsed "
                          << format_duration(Clock::now() - started) << '\n';
            }
        }

        if (converged) break;
    }

    if (root_dashboard) {
        dashboard_state.iteration = completed_iteration;
        dashboard_state.last_checkpoint_iteration =
            last_checkpoint_iteration;
        dashboard_state.active_roots = remaining_active;
        dashboard_state.total_frozen = total_frozen;
        dashboard_state.maximum_correction = maximum;
        dashboard_state.converged = converged;
        if (!converged) {
            dashboard_state.interaction_mode +=
                " | iteration limit reached";
            render_root_progress_dashboard(
                config, dashboard_state);
        }
        root_screen.close();
        std::cout << "  upper-root solver "
                  << (converged ? "converged" : "reached iteration limit")
                  << ": iteration=" << completed_iteration
                  << " active=" << remaining_active << '/' << upper_count
                  << " frozen=" << total_frozen
                  << " delta_max=" << real_string(maximum, 5)
                  << " elapsed=" << format_duration(Clock::now() - started)
                  << '\n';
    } else if (config.progress
               && config.progress_style == "bars"
               && !converged) {
        std::ostringstream detail;
        detail << "current delta=" << std::setprecision(5) << std::scientific
               << maximum << " | iteration limit";
        render_inline_progress(
            config,
            "  upper roots",
            static_cast<std::size_t>(completed_iteration),
            static_cast<std::size_t>(config.root_max_iterations),
            started,
            detail.str(),
            true,
            false);
    }
    write_root_checkpoint_symmetric(
        checkpoint_path,
        period,
        completed_iteration,
        real_roots,
        upper_roots,
        config);
    if (converged) {
        std::cout << "  hybrid root isolation complete: "
                  << total_frozen
                  << " upper-half root(s) Newton-polished and frozen; "
                  << "the full cloud will now be independently validated\n";
    }
    return {
        expand_symmetric_root_cloud(real_roots, upper_roots, config),
        completed_iteration,
        maximum,
        converged,
    };
}

RootSolveResult solve_center_roots(
    int period,
    const Config& config,
    const fs::path& checkpoint_path,
    const std::vector<Complex>& warm_start_centers
) {
    validate_root_grid_config(config);
    if (config.reset_root_checkpoint) {
        std::error_code error;
        fs::remove(checkpoint_path, error);
    }
    if (config.root_half_plane_symmetry && period > 2) {
        return solve_center_roots_symmetric(
            period, config, checkpoint_path, warm_start_centers);
    }
    return solve_center_roots_full(period, config, checkpoint_path);
}


template <unsigned Digits10>
using CenterMpReal = boost::multiprecision::number<
    boost::multiprecision::cpp_dec_float<Digits10>>;

template <typename MP>
using CenterMpComplex = std::complex<MP>;

template <typename MP>
bool center_mp_finite(const MP& value) {
    return boost::multiprecision::isfinite(value);
}

template <typename MP>
bool center_mp_finite(const CenterMpComplex<MP>& value) {
    return center_mp_finite(value.real()) && center_mp_finite(value.imag());
}

template <typename MP>
MP center_mp_abs(const CenterMpComplex<MP>& value) {
    return std::abs(value);
}

template <typename MP>
MP center_mp_pow10(int exponent) {
    using boost::multiprecision::pow;
    return pow(MP(10), exponent);
}

template <typename MP>
CenterMpComplex<MP> center_mp_from_complex(const Complex& value) {
    return {
        MP(real_string(value.real())),
        MP(real_string(value.imag())),
    };
}

template <typename MP>
Real center_real_from_mp(const MP& value) {
    try {
        return value.template convert_to<Real>();
    } catch (...) {
        return value < 0
            ? -std::numeric_limits<Real>::infinity()
            : std::numeric_limits<Real>::infinity();
    }
}

template <typename MP>
std::pair<CenterMpComplex<MP>, CenterMpComplex<MP>>
center_mp_critical_orbit(CenterMpComplex<MP> c, int period) {
    CenterMpComplex<MP> z{MP(0), MP(0)};
    CenterMpComplex<MP> derivative{MP(0), MP(0)};
    for (int k = 0; k < period; ++k) {
        derivative = MP(2) * z * derivative
            + CenterMpComplex<MP>{MP(1), MP(0)};
        z = z * z + c;
    }
    return {z, derivative};
}

struct CenterMpRescueDiagnostics {
    bool attempted = false;
    bool success = false;
    int dps = 0;
    int iterations_completed = 0;
    std::string stop_reason = "not attempted";
    Complex rounded_root{};
    Real mp_residual = std::numeric_limits<Real>::infinity();
    Real rounded_residual = std::numeric_limits<Real>::infinity();
    int detected_period = 0;
    int nearest_proper_divisor = 0;
    Real nearest_proper_divisor_residual =
        std::numeric_limits<Real>::infinity();
    Real validation_tolerance = 0;
};

template <typename MP>
int center_mp_detected_period(
    CenterMpComplex<MP> c,
    int period,
    const MP& tolerance,
    MP& period_residual,
    int& nearest_proper_divisor,
    MP& nearest_proper_divisor_residual
) {
    const auto& divs = divisors(period);
    std::size_t divisor_index = 0;
    CenterMpComplex<MP> z{MP(0), MP(0)};
    bool have_proper_divisor = false;
    int detected_period = 0;
    nearest_proper_divisor = 0;
    nearest_proper_divisor_residual = MP(0);

    for (int k = 1; k <= period; ++k) {
        z = z * z + c;
        if (divisor_index >= divs.size() || divs[divisor_index] != k) {
            continue;
        }

        const MP magnitude = center_mp_abs(z);
        if (k < period) {
            if (!have_proper_divisor
                || magnitude < nearest_proper_divisor_residual) {
                have_proper_divisor = true;
                nearest_proper_divisor = k;
                nearest_proper_divisor_residual = magnitude;
            }
            if (magnitude <= tolerance && detected_period == 0) {
                detected_period = k;
            }
        } else {
            period_residual = magnitude;
            if (magnitude <= tolerance && detected_period == 0) {
                detected_period = period;
            }
        }
        ++divisor_index;
    }

    return detected_period;
}

template <unsigned Digits10>
CenterMpRescueDiagnostics rescue_center_multiprecision_impl(
    Complex seed,
    int period,
    const Config& config
) {
    using MP = CenterMpReal<Digits10>;
    using MPC = CenterMpComplex<MP>;

    CenterMpRescueDiagnostics diagnostics;
    diagnostics.attempted = true;
    diagnostics.dps = static_cast<int>(Digits10);

    // Leave a generous guard band below the backend's nominal precision.  At
    // 200 decimal digits this asks for a 1e-160 orbit residual, far beyond what
    // long double can demonstrate near the steep real roots close to c=-2.
    constexpr int guard_digits = 40;
    constexpr int certified_digits =
        Digits10 > guard_digits + 20
            ? static_cast<int>(Digits10) - guard_digits
            : static_cast<int>(Digits10) / 2;
    const MP tolerance = center_mp_pow10<MP>(-certified_digits);
    diagnostics.validation_tolerance = center_real_from_mp(tolerance);

    MPC c = center_mp_from_complex<MP>(seed);
    const int maximum_iterations = std::max({
        100,
        config.center_polish_iterations,
        config.mp_newton_max_iterations,
    });

    MP final_residual = std::numeric_limits<MP>::infinity();
    std::string stop_reason = "multiprecision iteration limit";
    int completed_iterations = 0;

    for (int iteration = 0; iteration <= maximum_iterations; ++iteration) {
        auto [value, derivative] = center_mp_critical_orbit(c, period);
        completed_iterations = iteration;
        if (!center_mp_finite(value) || !center_mp_finite(derivative)) {
            stop_reason = "non-finite multiprecision orbit";
            break;
        }

        final_residual = center_mp_abs(value);
        if (final_residual <= tolerance) {
            stop_reason = "multiprecision residual target reached";
            break;
        }
        if (iteration == maximum_iterations) {
            stop_reason = "multiprecision iteration limit exhausted";
            break;
        }

        const MP derivative_magnitude = center_mp_abs(derivative);
        if (derivative_magnitude == 0) {
            stop_reason = "multiprecision orbit derivative is zero";
            break;
        }

        MPC correction = value / derivative;
        if (!center_mp_finite(correction)) {
            stop_reason = "non-finite multiprecision Newton correction";
            break;
        }
        const MP correction_size = center_mp_abs(correction);
        if (correction_size > MP("0.25")) {
            correction *= MP("0.25") / correction_size;
        }

        const MPC next = c - correction;
        if (!center_mp_finite(next)) {
            stop_reason = "non-finite multiprecision polished iterate";
            break;
        }
        if (next == c) {
            stop_reason = "multiprecision iterate stagnated";
            break;
        }
        c = next;
    }

    diagnostics.iterations_completed = completed_iterations;
    diagnostics.stop_reason = stop_reason;

    MP certified_period_residual = std::numeric_limits<MP>::infinity();
    int nearest_proper_divisor = 0;
    MP nearest_proper_residual = MP(0);
    const int detected_period = center_mp_detected_period(
        c,
        period,
        tolerance,
        certified_period_residual,
        nearest_proper_divisor,
        nearest_proper_residual);

    diagnostics.detected_period = detected_period;
    diagnostics.nearest_proper_divisor = nearest_proper_divisor;
    if (nearest_proper_divisor > 0) {
        diagnostics.nearest_proper_divisor_residual =
            center_real_from_mp(nearest_proper_residual);
    }
    diagnostics.mp_residual = center_real_from_mp(certified_period_residual);

    const Complex rounded{
        center_real_from_mp(c.real()),
        center_real_from_mp(c.imag()),
    };
    diagnostics.rounded_root = rounded;
    diagnostics.rounded_residual =
        safe_abs(critical_orbit(rounded, period).value);

    if (detected_period != period) {
        if (detected_period > 0 && detected_period < period) {
            diagnostics.stop_reason =
                "multiprecision validation found proper divisor period "
                + std::to_string(detected_period);
        } else {
            diagnostics.stop_reason =
                "multiprecision period residual did not reach certification target";
        }
        return diagnostics;
    }
    if (!finite(rounded)) {
        diagnostics.stop_reason =
            "certified multiprecision root did not convert to finite long double";
        return diagnostics;
    }

    diagnostics.success = true;
    diagnostics.stop_reason =
        "certified exact period in Boost.Multiprecision";
    return diagnostics;
}

CenterMpRescueDiagnostics rescue_center_multiprecision(
    Complex seed,
    int period,
    const Config& config
) {
    // The rescue is deliberately selective: only roots rejected by the normal
    // long-double validator pay for a 200-decimal-digit Newton polish.
    return rescue_center_multiprecision_impl<200>(seed, period, config);
}

struct CenterPolishDiagnostics {
    Complex final_point{};
    Real final_derivative_magnitude = std::numeric_limits<Real>::infinity();
    int iterations_completed = 0;
    std::string stop_reason = "not started";
};

std::optional<Complex> polish_center(
    Complex seed,
    int period,
    const Config& config,
    int& iterations,
    Real& residual,
    CenterPolishDiagnostics* diagnostics = nullptr
) {
    Complex c = seed;
    std::string stop_reason = "iteration limit";
    int completed_iterations = 0;

    auto finish_diagnostics = [&](const OrbitValue& orbit) {
        if (!diagnostics) return;
        diagnostics->final_point = c;
        diagnostics->final_derivative_magnitude = safe_abs(orbit.derivative);
        diagnostics->iterations_completed = completed_iterations;
        diagnostics->stop_reason = stop_reason;
    };

    for (int iteration = 0; iteration <= config.center_polish_iterations; ++iteration) {
        const auto orbit = critical_orbit(c, period);
        residual = safe_abs(orbit.value);
        completed_iterations = iteration;
        if (residual <= config.center_residual_tolerance) {
            iterations = iteration;
            stop_reason = "converged to center_residual_tolerance";
            finish_diagnostics(orbit);
            return c;
        }
        if (iteration == config.center_polish_iterations) {
            stop_reason = "center_polish_iterations exhausted";
            break;
        }
        const Real derivative_magnitude = safe_abs(orbit.derivative);
        if (derivative_magnitude < 1.0e-30L) {
            stop_reason = "orbit derivative too small";
            break;
        }
        Complex correction = orbit.value / orbit.derivative;
        if (!finite(correction)) {
            stop_reason = "non-finite Newton correction";
            break;
        }
        Real size = safe_abs(correction);
        if (size > 0.25L) correction *= 0.25L / size;
        c -= correction;
        if (!finite(c)) {
            stop_reason = "non-finite polished iterate";
            break;
        }
    }

    // Preserve the original public result: any path that reaches the relaxed
    // acceptance check reports the configured maximum iteration count.
    iterations = config.center_polish_iterations;
    const auto final_orbit = critical_orbit(c, period);
    residual = safe_abs(final_orbit.value);
    if (finite(c) && residual <= 100 * config.center_residual_tolerance) {
        stop_reason = "accepted by relaxed 100x residual limit";
        finish_diagnostics(final_orbit);
        return c;
    }
    finish_diagnostics(final_orbit);
    return std::nullopt;
}

struct CenterRow {
    int period = 0;
    int component_index = 0;
    int expected_period_count = 0;
    Complex center{};
    Real residual = 0;
    int detected_period = 0;
    int conjugate_index = 0;
    int newton_iterations = 0;
    std::string refinement_method = "cpp-long-double";
    int refinement_dps = std::numeric_limits<Real>::digits10;
};


std::vector<CenterRow> analytic_center_rows(int period) {
    if (period != 1 && period != 2) {
        throw std::invalid_argument(
            "Analytic center rows exist only for periods 1 and 2.");
    }
    const Real center = period == 1 ? 0.0L : -1.0L;
    return {CenterRow{
        period,
        0,
        1,
        Complex{center, 0},
        0,
        period,
        0,
        0,
        "analytic",
        std::numeric_limits<Real>::digits10,
    }};
}

std::vector<CenterRow> refine_root_cloud(int period, const RootSolveResult& solve,
                                         const Config& config) {
    struct Candidate {
        Complex root{};
        int iterations = 0;
        Real residual = 0;
        std::size_t seed_index = 0;
        std::string refinement_method = "cpp-long-double";
        int refinement_dps = std::numeric_limits<Real>::digits10;
    };

    struct ValidationFailure {
        std::size_t seed_index = 0;
        std::string source;
        Complex seed{};
        Real seed_residual = std::numeric_limits<Real>::infinity();
        bool polished = false;
        Complex polished_root{};
        int iterations = 0;
        Real residual = std::numeric_limits<Real>::infinity();
        int detected_period = 0;
        CenterPolishDiagnostics polish;
        CenterMpRescueDiagnostics mp_rescue;
    };

    struct MpRescueExample {
        std::size_t seed_index = 0;
        std::string source;
        Complex rounded_root{};
        Real mp_residual = std::numeric_limits<Real>::infinity();
        Real rounded_residual = std::numeric_limits<Real>::infinity();
        int iterations = 0;
        int dps = 0;
    };

    struct DuplicateExample {
        std::size_t first_seed_index = 0;
        std::size_t second_seed_index = 0;
        Complex first_root{};
        Complex second_root{};
        Real distance = std::numeric_limits<Real>::infinity();
    };

    constexpr std::size_t MAX_DIAGNOSTIC_EXAMPLES = 16;
    const std::size_t expected = center_count(period);
    const std::size_t real_seed_count =
        config.root_half_plane_symmetry && period > 2
            ? real_exact_center_count(period)
            : 0;
    const std::size_t upper_seed_count =
        config.root_half_plane_symmetry && period > 2
        && expected >= real_seed_count
        && (expected - real_seed_count) % 2 == 0
            ? (expected - real_seed_count) / 2
            : 0;

    auto seed_source = [&](std::size_t index) {
        std::ostringstream source;
        if (!(config.root_half_plane_symmetry && period > 2)
            || real_seed_count + 2 * upper_seed_count != expected) {
            source << "full-cloud #" << index;
        } else if (index < real_seed_count) {
            source << "fixed-real #" << index;
        } else if (index < real_seed_count + upper_seed_count) {
            source << "upper-half #" << (index - real_seed_count);
        } else {
            source << "conjugate #"
                   << (index - real_seed_count - upper_seed_count);
        }
        return source.str();
    };

    std::vector<Candidate> candidates;
    candidates.reserve(expected);
    std::vector<ValidationFailure> failure_examples;
    failure_examples.reserve(MAX_DIAGNOSTIC_EXAMPLES);
    std::vector<MpRescueExample> mp_rescue_examples;
    mp_rescue_examples.reserve(MAX_DIAGNOSTIC_EXAMPLES);
    std::size_t long_double_rejections = 0;
    std::size_t mp_rescue_attempts = 0;
    std::size_t mp_rescue_successes = 0;
    std::size_t mp_rescue_failures = 0;
    std::size_t polish_failures = 0;
    std::size_t wrong_period_failures = 0;

    const auto started = Clock::now();
    std::cout << "  polishing and validating " << expected
              << " roots independently...\n";
    for (std::size_t i = 0; i < solve.roots.size(); ++i) {
        const Complex seed = solve.roots[i];
        const Real seed_residual = safe_abs(critical_orbit(seed, period).value);
        int iterations = 0;
        Real residual = 0;
        CenterPolishDiagnostics polish_diagnostics;
        auto root = polish_center(
            seed,
            period,
            config,
            iterations,
            residual,
            &polish_diagnostics);

        int detected_period = 0;
        if (root) {
            detected_period = detected_center_period(
                *root,
                period,
                config.exact_period_tolerance);
        }

        if (root && detected_period == period) {
            candidates.push_back(Candidate{
                *root,
                iterations,
                residual,
                i,
                "cpp-long-double",
                std::numeric_limits<Real>::digits10,
            });
        } else {
            ++long_double_rejections;
            ++mp_rescue_attempts;

            const Complex rescue_seed = root
                ? *root
                : (finite(polish_diagnostics.final_point)
                    ? polish_diagnostics.final_point
                    : seed);
            CenterMpRescueDiagnostics mp_rescue =
                rescue_center_multiprecision(
                    rescue_seed,
                    period,
                    config);

            if (mp_rescue.success) {
                ++mp_rescue_successes;
                candidates.push_back(Candidate{
                    mp_rescue.rounded_root,
                    iterations + mp_rescue.iterations_completed,
                    mp_rescue.rounded_residual,
                    i,
                    "cpp-mp-" + std::to_string(mp_rescue.dps),
                    mp_rescue.dps,
                });
                if (mp_rescue_examples.size() < MAX_DIAGNOSTIC_EXAMPLES) {
                    mp_rescue_examples.push_back(MpRescueExample{
                        i,
                        seed_source(i),
                        mp_rescue.rounded_root,
                        mp_rescue.mp_residual,
                        mp_rescue.rounded_residual,
                        mp_rescue.iterations_completed,
                        mp_rescue.dps,
                    });
                }
            } else {
                ++mp_rescue_failures;
                if (!root) ++polish_failures;
                else ++wrong_period_failures;
                if (failure_examples.size() < MAX_DIAGNOSTIC_EXAMPLES) {
                    failure_examples.push_back(ValidationFailure{
                        i,
                        seed_source(i),
                        seed,
                        seed_residual,
                        root.has_value(),
                        root.value_or(polish_diagnostics.final_point),
                        iterations,
                        residual,
                        detected_period,
                        polish_diagnostics,
                        mp_rescue,
                    });
                }
            }
        }

        if ((i + 1) % 100 == 0 || i + 1 == solve.roots.size()) {
            std::ostringstream detail;
            detail << "valid=" << candidates.size()
                   << " mp-rescued=" << mp_rescue_successes
                   << " rejected=" << (polish_failures + wrong_period_failures);
            if (config.progress && config.progress_style == "bars") {
                render_inline_progress(
                    config,
                    "    polishing",
                    i + 1,
                    solve.roots.size(),
                    started,
                    detail.str(),
                    i + 1 == solve.roots.size(),
                    true);
            } else if (config.progress_style == "lines") {
                std::cout << "    seeds " << (i + 1) << '/'
                          << solve.roots.size() << " | " << detail.str()
                          << " | elapsed "
                          << format_duration(Clock::now() - started) << '\n';
            }
        }
    }

    if (mp_rescue_attempts > 0) {
        std::cout << "    multiprecision center rescue: attempted="
                  << mp_rescue_attempts
                  << " rescued=" << mp_rescue_successes
                  << " failed=" << mp_rescue_failures
                  << " (Boost cpp_dec_float_200)\n";
        for (const auto& rescue : mp_rescue_examples) {
            std::cout << "      rescued seed " << rescue.seed_index
                      << " (" << rescue.source << ") with cpp-mp-"
                      << rescue.dps
                      << ": MP |F_" << period << "(c)|="
                      << real_string(rescue.mp_residual, 8)
                      << ", stored-long-double |F_" << period << "(c)|="
                      << real_string(rescue.rounded_residual, 8)
                      << ", MP iterations=" << rescue.iterations
                      << '\n';
        }
        if (mp_rescue_successes > mp_rescue_examples.size()) {
            std::cout << "      ... "
                      << (mp_rescue_successes - mp_rescue_examples.size())
                      << " additional rescued seed(s) omitted.\n";
        }
    }

    // Sort first, then deduplicate adjacent roots. This is O(N log N), unlike
    // repeatedly scanning the entire accepted list, which becomes painful past
    // period 10.
    const auto ordering_started = Clock::now();
    render_inline_progress(
        config, "    ordering centers", 0, 2, ordering_started, "sorting roots");
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.root.real() != b.root.real()) return a.root.real() < b.root.real();
        return a.root.imag() < b.root.imag();
    });

    std::vector<Candidate> unique;
    unique.reserve(candidates.size());
    std::vector<DuplicateExample> duplicate_examples;
    duplicate_examples.reserve(MAX_DIAGNOSTIC_EXAMPLES);
    std::size_t duplicates_removed = 0;
    std::size_t close_certified_real_pairs_preserved = 0;
    for (const auto& candidate : candidates) {
        if (!unique.empty()) {
            const Real distance = safe_abs(candidate.root - unique.back().root);
            const bool both_certified_real_seeds =
                unique.back().seed_index < real_seed_count
                && candidate.seed_index < real_seed_count;
            if (distance <= config.center_duplicate_tolerance
                && both_certified_real_seeds) {
                // The real-axis prepass supplies one independently bracketed,
                // exact-period seed per real root. Near c=-2 their genuine
                // spacing is already below the generic cloud-dedup tolerance
                // at period 21 and shrinks further with period. Preserve those
                // certified identities; blanket geometric deduplication is for
                // collisions in the dynamically solved complex cloud.
                ++close_certified_real_pairs_preserved;
            } else if (distance <= config.center_duplicate_tolerance) {
                ++duplicates_removed;
                if (duplicate_examples.size() < MAX_DIAGNOSTIC_EXAMPLES) {
                    duplicate_examples.push_back(DuplicateExample{
                        unique.back().seed_index,
                        candidate.seed_index,
                        unique.back().root,
                        candidate.root,
                        distance,
                    });
                }
                if (candidate.residual < unique.back().residual) {
                    unique.back() = candidate;
                }
                continue;
            }
        }
        unique.push_back(candidate);
    }
    render_inline_progress(
        config, "    ordering centers", 1, 2, ordering_started,
        "deduplicated=" + std::to_string(unique.size())
            + (close_certified_real_pairs_preserved > 0
                ? " preserved-real-neighbours="
                    + std::to_string(close_certified_real_pairs_preserved)
                : std::string()));

    if (unique.size() != expected) {
        std::cerr << "\n  root validation diagnostics:\n"
                  << "    expected roots:            " << expected << '\n'
                  << "    solver seeds supplied:     " << solve.roots.size() << '\n'
                  << "    accepted before dedup:     " << candidates.size() << '\n'
                  << "    long-double rejections:    " << long_double_rejections << '\n'
                  << "    MP rescue attempts:        " << mp_rescue_attempts << '\n'
                  << "    MP rescue successes:       " << mp_rescue_successes << '\n'
                  << "    MP rescue failures:        " << mp_rescue_failures << '\n'
                  << "    final polishing failures:  " << polish_failures << '\n'
                  << "    final wrong-period rejects: "
                  << wrong_period_failures << '\n'
                  << "    duplicates removed:        " << duplicates_removed << '\n'
                  << "    unique validated roots:    " << unique.size() << '\n'
                  << "    center residual tolerance: "
                  << real_string(config.center_residual_tolerance, 6) << '\n'
                  << "    exact-period tolerance:    "
                  << real_string(config.exact_period_tolerance, 6) << '\n'
                  << "    duplicate tolerance:       "
                  << real_string(config.center_duplicate_tolerance, 6) << '\n';

        for (const auto& failure : failure_examples) {
            const Complex comparison_point = failure.polished
                ? failure.polished_root
                : (finite(failure.polish.final_point)
                    ? failure.polish.final_point
                    : failure.seed);
            const Candidate* nearest = nullptr;
            Real nearest_distance = std::numeric_limits<Real>::infinity();
            for (const auto& candidate : unique) {
                const Real distance = safe_abs(candidate.root - comparison_point);
                if (distance < nearest_distance) {
                    nearest_distance = distance;
                    nearest = &candidate;
                }
            }

            std::cerr << "\n    rejected seed " << failure.seed_index
                      << " (" << failure.source << "):\n"
                      << "      seed c:              "
                      << complex_string(failure.seed) << '\n'
                      << "      seed |F_" << period << "(c)|:       "
                      << real_string(failure.seed_residual, 8) << '\n'
                      << "      polish result:       "
                      << (failure.polished ? "returned a root" : "failed")
                      << '\n'
                      << "      polish stop reason:  "
                      << failure.polish.stop_reason << '\n'
                      << "      polish iterations:   "
                      << failure.polish.iterations_completed << " completed"
                      << " (reported=" << failure.iterations << ")\n"
                      << "      final c:             "
                      << complex_string(failure.polish.final_point) << '\n'
                      << "      final |F_" << period << "(c)|:      "
                      << real_string(failure.residual, 8) << '\n'
                      << "      final |dF_" << period << "/dc|:     "
                      << real_string(
                             failure.polish.final_derivative_magnitude, 8)
                      << '\n'
                      << "      MP rescue attempted: "
                      << (failure.mp_rescue.attempted ? "yes" : "no") << '\n';
            if (failure.mp_rescue.attempted) {
                std::cerr
                    << "      MP rescue result:    "
                    << (failure.mp_rescue.success ? "succeeded" : "failed")
                    << '\n'
                    << "      MP precision:        "
                    << failure.mp_rescue.dps << " decimal digits\n"
                    << "      MP stop reason:      "
                    << failure.mp_rescue.stop_reason << '\n'
                    << "      MP iterations:       "
                    << failure.mp_rescue.iterations_completed << '\n'
                    << "      MP detected period:  "
                    << failure.mp_rescue.detected_period << '\n'
                    << "      MP |F_" << period << "(c)|:       "
                    << real_string(failure.mp_rescue.mp_residual, 8) << '\n'
                    << "      rounded c:           "
                    << complex_string(failure.mp_rescue.rounded_root) << '\n'
                    << "      rounded |F_" << period << "(c)|:  "
                    << real_string(failure.mp_rescue.rounded_residual, 8)
                    << '\n'
                    << "      MP target:           "
                    << real_string(
                           failure.mp_rescue.validation_tolerance, 8)
                    << '\n';
                if (failure.mp_rescue.nearest_proper_divisor > 0) {
                    std::cerr
                        << "      closest divisor:     "
                        << failure.mp_rescue.nearest_proper_divisor
                        << " with |F_d(c)|="
                        << real_string(
                               failure.mp_rescue
                                   .nearest_proper_divisor_residual,
                               8)
                        << '\n';
                }
            }
            if (failure.polished) {
                std::cerr << "      detected period:     "
                          << failure.detected_period << '\n';
            }
            if (nearest) {
                std::cerr << "      nearest accepted:    seed "
                          << nearest->seed_index << " ("
                          << seed_source(nearest->seed_index) << ")\n"
                          << "      nearest root c:      "
                          << complex_string(nearest->root) << '\n'
                          << "      nearest distance:    "
                          << real_string(nearest_distance, 8) << '\n';
            }
        }
        const std::size_t rejected_total = polish_failures + wrong_period_failures;
        if (rejected_total > failure_examples.size()) {
            std::cerr << "\n    ... "
                      << (rejected_total - failure_examples.size())
                      << " additional rejected seed(s) omitted.\n";
        }

        for (const auto& duplicate : duplicate_examples) {
            std::cerr << "\n    duplicate accepted roots:\n"
                      << "      seed " << duplicate.first_seed_index
                      << " (" << seed_source(duplicate.first_seed_index) << ") c="
                      << complex_string(duplicate.first_root) << '\n'
                      << "      seed " << duplicate.second_seed_index
                      << " (" << seed_source(duplicate.second_seed_index) << ") c="
                      << complex_string(duplicate.second_root) << '\n'
                      << "      separation: "
                      << real_string(duplicate.distance, 8) << '\n';
        }
        if (duplicates_removed > duplicate_examples.size()) {
            std::cerr << "\n    ... "
                      << (duplicates_removed - duplicate_examples.size())
                      << " additional duplicate pair(s) omitted.\n";
        }
        std::cerr.flush();

        std::ostringstream message;
        message << "Global solve yielded " << unique.size()
                << " unique validated period-" << period
                << " centers; expected " << expected << ". "
                << "Long-double validation rejected "
                << long_double_rejections << " seed(s); Boost.Multiprecision "
                << "rescued " << mp_rescue_successes << " of "
                << mp_rescue_attempts << " attempt(s), leaving "
                << (polish_failures + wrong_period_failures)
                << " final rejection(s) (polish failures=" << polish_failures
                << ", wrong period=" << wrong_period_failures
                << "). Deduplication removed " << duplicates_removed
                << ". The checkpoint was kept; see the diagnostics above.";
        if (polish_failures > 0) {
            message << " For polishing failures, inspect both the long-double and "
                    << "multiprecision stop reasons before changing tolerances.";
        }
        if (wrong_period_failures > 0) {
            message << " For wrong-period rejections, inspect detected_period and consider "
                    << "reducing exact_period_tolerance if a proper divisor is being "
                    << "accepted too loosely.";
        }
        if (duplicates_removed > 0) {
            message << " For duplicate removals, inspect the reported separations before "
                    << "changing center_duplicate_tolerance.";
        }
        throw std::runtime_error(message.str());
    }

    std::vector<std::size_t> order(expected);
    std::iota(order.begin(), order.end(), 0);
    const Real ordering_scale = 1.0e14L;
    auto rounded14 = [&](Real value) {
        return std::round(value * ordering_scale) / ordering_scale;
    };
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const Real ar = rounded14(unique[a].root.real());
        const Real br = rounded14(unique[b].root.real());
        if (ar != br) return ar < br;
        const Real ai = rounded14(unique[a].root.imag());
        const Real bi = rounded14(unique[b].root.imag());
        if (ai != bi) return ai < bi;
        if (unique[a].root.real() != unique[b].root.real()) {
            return unique[a].root.real() < unique[b].root.real();
        }
        return unique[a].root.imag() < unique[b].root.imag();
    });
    render_inline_progress(
        config, "    ordering centers", 2, 2, ordering_started,
        "stable component indexes", true, true);

    std::vector<CenterRow> rows;
    rows.reserve(expected);
    const auto row_pack_started = Clock::now();
    if (expected > 0) {
        render_inline_progress(
            config, "    packing center rows", 0, expected, row_pack_started);
    }
    for (std::size_t index = 0; index < expected; ++index) {
        const auto& source = unique[order[index]];
        rows.push_back(CenterRow{
            period,
            static_cast<int>(index),
            static_cast<int>(expected),
            source.root,
            source.residual,
            period,
            0,
            source.iterations,
            source.refinement_method,
            source.refinement_dps,
        });
        const std::size_t done = index + 1;
        if (done == expected || done % 2048 == 0) {
            render_inline_progress(
                config,
                "    packing center rows",
                done,
                expected,
                row_pack_started,
                {},
                done == expected,
                true);
        }
    }

    // Match conjugates through a spatial index instead of the former O(N^2)
    // all-pairs nearest-neighbour scan. At period 16 that old loop performed
    // more than a billion distance evaluations after the polishing bar ended.
    const Real conjugate_cell_size = std::max<Real>(
        4 * config.center_duplicate_tolerance, 1.0e-14L);
    auto conjugate_cell = [&](const Complex& value) {
        return std::pair<std::int64_t, std::int64_t>{
            static_cast<std::int64_t>(std::floor(
                value.real() / conjugate_cell_size)),
            static_cast<std::int64_t>(std::floor(
                value.imag() / conjugate_cell_size))};
    };
    struct ConjugateCellHash {
        std::size_t operator()(
            const std::pair<std::int64_t, std::int64_t>& key
        ) const noexcept {
            std::size_t seed = std::hash<std::int64_t>{}(key.first);
            seed ^= std::hash<std::int64_t>{}(key.second)
                + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    std::unordered_map<
        std::pair<std::int64_t, std::int64_t>,
        std::vector<std::size_t>,
        ConjugateCellHash> conjugate_index;
    conjugate_index.reserve(rows.size() * 2 + 1);
    const auto conjugate_index_started = Clock::now();
    if (!rows.empty()) {
        render_inline_progress(
            config, "    indexing conjugates", 0, rows.size(),
            conjugate_index_started);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        conjugate_index[conjugate_cell(rows[i].center)].push_back(i);
        const std::size_t done = i + 1;
        if (done == rows.size() || done % 2048 == 0) {
            render_inline_progress(
                config,
                "    indexing conjugates",
                done,
                rows.size(),
                conjugate_index_started,
                {},
                done == rows.size(),
                true);
        }
    }

    const auto conjugate_started = Clock::now();
    if (!rows.empty()) {
        render_inline_progress(
            config, "    matching conjugates", 0, rows.size(), conjugate_started);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Complex target = std::conj(rows[i].center);
        const auto base = conjugate_cell(target);
        std::size_t best = rows.size();
        Real best_distance = std::numeric_limits<Real>::infinity();
        for (std::int64_t dx = -1; dx <= 1; ++dx) {
            for (std::int64_t dy = -1; dy <= 1; ++dy) {
                const auto found = conjugate_index.find(
                    {base.first + dx, base.second + dy});
                if (found == conjugate_index.end()) continue;
                for (const std::size_t candidate : found->second) {
                    const Real distance = safe_abs(
                        rows[candidate].center - target);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = candidate;
                    }
                }
            }
        }
        if (best == rows.size()
            || best_distance > 8 * config.center_duplicate_tolerance) {
            throw std::runtime_error(
                "Could not match the conjugate of period-"
                + std::to_string(period) + " center "
                + std::to_string(i) + "; nearest indexed distance="
                + real_string(best_distance, 6));
        }
        rows[i].conjugate_index = static_cast<int>(best);
        const std::size_t done = i + 1;
        if (done == rows.size() || done % 1024 == 0) {
            render_inline_progress(
                config,
                "    matching conjugates",
                done,
                rows.size(),
                conjugate_started,
                {},
                done == rows.size(),
                true);
        }
    }

    return rows;
}

// -----------------------------------------------------------------------------
// Typed scanner cache bridge
// -----------------------------------------------------------------------------

using AreaScanStore = mandelbrot::catalogue::AreaScanStore;
using AreaScanCenterRecord = mandelbrot::catalogue::AreaScanCenterRecord;
using AreaMeasurementRecord = mandelbrot::catalogue::AreaMeasurementRecord;
using AreaPeriodSummaryRecord = mandelbrot::catalogue::AreaPeriodSummaryRecord;
using CatalogueReal = mandelbrot::catalogue::CatalogueReal;
using CatalogueComplex = mandelbrot::catalogue::ComplexValue;

CatalogueReal catalogue_real(Real value) {
    return mandelbrot::catalogue::Catalogue::parse_decimal(real_string(value));
}

Real scanner_real(const CatalogueReal& value) {
    return value.convert_to<Real>();
}

AreaScanCenterRecord center_to_record(const CenterRow& row) {
    AreaScanCenterRecord record;
    record.period = row.period;
    record.component_index = row.component_index;
    record.expected_period_count = row.expected_period_count;
    record.center = {catalogue_real(row.center.real()), catalogue_real(row.center.imag())};
    record.center_residual = catalogue_real(row.residual);
    record.detected_exact_period = row.detected_period;
    record.conjugate_index = row.conjugate_index;
    record.center_newton_iterations = row.newton_iterations;
    record.center_refinement_method = row.refinement_method;
    record.center_refinement_dps = row.refinement_dps;
    return record;
}

CenterRow center_from_record(const AreaScanCenterRecord& record) {
    CenterRow row;
    row.period = record.period;
    row.component_index = record.component_index;
    row.expected_period_count = record.expected_period_count;
    row.center = {scanner_real(record.center.re), scanner_real(record.center.im)};
    row.residual = scanner_real(record.center_residual);
    row.detected_period = record.detected_exact_period;
    row.conjugate_index = record.conjugate_index;
    row.newton_iterations = record.center_newton_iterations;
    row.refinement_method = record.center_refinement_method;
    row.refinement_dps = record.center_refinement_dps;
    return row;
}

std::vector<CenterRow> load_center_period(
    const AreaScanStore& store,
    const Config& config,
    int period
) {
    const auto load_started = Clock::now();
    const auto stored = store.load_centers(
        period,
        make_inline_progress_callback(
            config,
            "  loading centers p" + std::to_string(period),
            load_started,
            {},
            true));
    const std::size_t total = stored.size();

    std::vector<CenterRow> result;
    result.reserve(total);
    const auto convert_started = Clock::now();
    if (total > 0) {
        render_inline_progress(
            config,
            "  decoding centers p" + std::to_string(period),
            0,
            total,
            convert_started);
    }
    for (std::size_t index = 0; index < stored.size(); ++index) {
        result.push_back(center_from_record(stored[index]));
        const std::size_t converted = index + 1;
        if (converted == total || converted % 2048 == 0) {
            render_inline_progress(
                config,
                "  decoding centers p" + std::to_string(period),
                converted,
                total,
                convert_started,
                {},
                converted == total,
                true);
        }
    }
    return result;
}

void write_center_period(
    const AreaScanStore& store,
    int period,
    const std::vector<CenterRow>& centers,
    const Config& config
) {
    const std::size_t total = centers.size();
    std::vector<AreaScanCenterRecord> records;
    records.reserve(total);
    const auto pack_started = Clock::now();
    if (total > 0) {
        render_inline_progress(
            config, "  packing centers", 0, total, pack_started);
    }
    for (std::size_t index = 0; index < centers.size(); ++index) {
        records.push_back(center_to_record(centers[index]));
        const std::size_t packed = index + 1;
        if (packed == total || packed % 2048 == 0) {
            render_inline_progress(
                config,
                "  packing centers",
                packed,
                total,
                pack_started,
                {},
                packed == total,
                true);
        }
    }
    const auto write_started = Clock::now();
    store.save_centers(
        period,
        records,
        make_inline_progress_callback(
            config, "  writing centers", write_started, {}, true));
}

bool complete_center_period(
    const std::vector<CenterRow>& rows,
    int period,
    const Config& config
) {
    const std::size_t expected = center_count(period);
    if (rows.size() != expected) return false;
    const auto started = Clock::now();
    if (!rows.empty()) {
        render_inline_progress(
            config,
            "  checking centers p" + std::to_string(period),
            0,
            rows.size(),
            started);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].period != period || rows[i].component_index != static_cast<int>(i)
            || rows[i].detected_period != period) {
            render_inline_progress(
                config,
                "  checking centers p" + std::to_string(period),
                i + 1,
                rows.size(),
                started,
                "invalid cache",
                true,
                false);
            return false;
        }
        const std::size_t done = i + 1;
        if (done == rows.size() || done % 4096 == 0) {
            render_inline_progress(
                config,
                "  checking centers p" + std::to_string(period),
                done,
                rows.size(),
                started,
                {},
                done == rows.size(),
                true);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Multiplier continuation
// -----------------------------------------------------------------------------

struct OrbitData {
    Complex A{}, B{}, C{}, D{}, E{};
};

OrbitData iterate_data(Complex z, Complex c, int n) {
    Complex A = z;
    Complex B{1, 0};
    Complex C{0, 0};
    Complex D{0, 0};
    Complex E{0, 0};
    for (int i = 0; i < n; ++i) {
        const Complex A0 = A, B0 = B, C0 = C, D0 = D, E0 = E;
        A = A0 * A0 + c;
        B = 2.0L * A0 * B0;
        C = 2.0L * A0 * C0 + Complex{1, 0};
        D = 2.0L * (B0 * B0 + A0 * D0);
        E = 2.0L * (C0 * B0 + A0 * E0);
    }
    return {A, B, C, D, E};
}

struct MultiplierState {
    Complex lambda{};
    Complex z{};
    Complex c{};
    Complex z_lambda{};
    Complex c_lambda{};
    Real residual = 0;
    int newton_iterations = 0;
};

struct ContinuationStats {
    long long solve_calls = 0;
    long long failed_attempts = 0;
    long long newton_iterations = 0;
    int max_subdivision_depth = 0;
    Real max_residual = 0;
    long long rejected_branch_candidates = 0;
    long long cyclic_seed_attempts = 0;
    long long cyclic_recoveries = 0;
};

std::optional<MultiplierState> state_from_solution(int n, Complex lambda, Complex z,
                                                   Complex c, int iterations) {
    const auto data = iterate_data(z, c, n);
    const Real residual = safe_abs(data.A - z) + safe_abs(data.B - lambda);
    const Complex determinant = (data.B - Complex{1, 0}) * data.E - data.C * data.D;
    if (!finite(determinant) || safe_abs(determinant) <= 1.0e-36L) return std::nullopt;
    const Complex z_lambda = -data.C / determinant;
    const Complex c_lambda = (data.B - Complex{1, 0}) / determinant;
    if (!finite(z_lambda) || !finite(c_lambda)) return std::nullopt;
    return MultiplierState{lambda, z, c, z_lambda, c_lambda, residual, iterations};
}

std::optional<MultiplierState> solve_multiplier_point(int n, Complex z_guess, Complex c_guess,
                                                      Complex lambda, const Config& config) {
    Complex z = z_guess;
    Complex c = c_guess;
    for (int iteration = 0; iteration <= config.newton_max_iterations; ++iteration) {
        const auto data = iterate_data(z, c, n);
        const Complex G1 = data.A - z;
        const Complex G2 = data.B - lambda;
        const Real residual = safe_abs(G1) + safe_abs(G2);
        if (residual <= config.newton_tolerance) {
            return state_from_solution(n, lambda, z, c, iteration);
        }
        if (iteration == config.newton_max_iterations) break;

        const Complex determinant = (data.B - Complex{1, 0}) * data.E - data.C * data.D;
        if (!finite(determinant) || safe_abs(determinant) <= 1.0e-36L) return std::nullopt;
        const Complex dz = (G1 * data.E - data.C * G2) / determinant;
        const Complex dc = ((data.B - Complex{1, 0}) * G2 - data.D * G1) / determinant;
        const Real relative_step = std::max(
            safe_abs(dz) / std::max<Real>(1, safe_abs(z)),
            safe_abs(dc) / std::max<Real>(1, safe_abs(c)));
        if (relative_step <= 128 * std::numeric_limits<Real>::epsilon()
            && residual <= 64 * config.newton_tolerance) {
            return state_from_solution(n, lambda, z, c, iteration);
        }
        z -= dz;
        c -= dc;
        if (!finite(z) || !finite(c)) return std::nullopt;
    }
    return std::nullopt;
}

bool periodic_point_has_exact_period(int n, Complex z, Complex c, Real tolerance) {
    const Real scale = std::max<Real>(1, safe_abs(z));
    for (int d : proper_divisors(n)) {
        Complex value = z;
        for (int k = 0; k < d; ++k) value = value * value + c;
        if (safe_abs(value - z) <= tolerance * scale) return false;
    }
    return true;
}

std::vector<MultiplierState> cyclic_marked_states(int n, const MultiplierState& state) {
    std::vector<MultiplierState> result{state};
    Complex z = state.z;
    for (int i = 1; i < n; ++i) {
        z = z * z + state.c;
        if (auto marked = state_from_solution(n, state.lambda, z, state.c, 0)) {
            result.push_back(*marked);
        }
    }
    return result;
}

Real predictor_score(const MultiplierState& start, Complex target,
                     const MultiplierState& candidate, const Config& config) {
    const Complex delta = target - start.lambda;
    const Complex predicted_z = start.z + start.z_lambda * delta;
    const Complex predicted_c = start.c + start.c_lambda * delta;
    const Real eps = std::numeric_limits<Real>::epsilon();
    const Real c_scale = std::max({safe_abs(start.c_lambda * delta),
                                   1024 * eps * std::max<Real>(1, safe_abs(start.c)),
                                   32 * config.newton_tolerance});
    const Real z_scale = std::max({safe_abs(start.z_lambda * delta),
                                   1024 * eps * std::max<Real>(1, safe_abs(start.z)),
                                   32 * config.newton_tolerance});
    return std::max(safe_abs(candidate.c - predicted_c) / c_scale,
                    safe_abs(candidate.z - predicted_z) / z_scale);
}

std::optional<MultiplierState> solve_local_branch(int n, const MultiplierState& start,
                                                  Complex target, const Config& config,
                                                  ContinuationStats& stats) {
    auto attempt = [&](const MultiplierState& seed) -> std::optional<MultiplierState> {
        const Complex delta = target - seed.lambda;
        const Complex z_guess = seed.z + seed.z_lambda * delta;
        const Complex c_guess = seed.c + seed.c_lambda * delta;
        ++stats.solve_calls;
        auto result = solve_multiplier_point(n, z_guess, c_guess, target, config);
        if (!result) {
            ++stats.failed_attempts;
            return std::nullopt;
        }
        stats.newton_iterations += result->newton_iterations;
        stats.max_residual = std::max(stats.max_residual, result->residual);
        if (!periodic_point_has_exact_period(n, result->z, result->c,
                                             config.exact_period_tolerance)
            || predictor_score(seed, target, *result, config) > config.branch_jump_factor) {
            ++stats.rejected_branch_candidates;
            return std::nullopt;
        }
        return result;
    };

    if (auto result = attempt(start)) return result;
    const auto seeds = cyclic_marked_states(n, start);
    for (std::size_t i = 1; i < seeds.size(); ++i) {
        ++stats.cyclic_seed_attempts;
        if (auto result = attempt(seeds[i])) {
            ++stats.cyclic_recoveries;
            return result;
        }
    }
    return std::nullopt;
}

std::optional<MultiplierState> continue_to_lambda(int n, const MultiplierState& start,
                                                  Complex target, const Config& config,
                                                  ContinuationStats& stats, int depth = 0) {
    stats.max_subdivision_depth = std::max(stats.max_subdivision_depth, depth);
    const Complex delta = target - start.lambda;
    if (safe_abs(delta) > config.continuation_max_step) {
        if (depth >= config.continuation_max_depth) return std::nullopt;
        const Complex middle = (start.lambda + target) / 2.0L;
        auto first = continue_to_lambda(n, start, middle, config, stats, depth + 1);
        if (!first) return std::nullopt;
        return continue_to_lambda(n, *first, target, config, stats, depth + 1);
    }

    if (auto result = solve_local_branch(n, start, target, config, stats)) return result;
    if (depth >= config.continuation_max_depth) return std::nullopt;
    const Complex middle = (start.lambda + target) / 2.0L;
    auto first = continue_to_lambda(n, start, middle, config, stats, depth + 1);
    if (!first) return std::nullopt;
    return continue_to_lambda(n, *first, target, config, stats, depth + 1);
}

struct RingTrace {
    Real rho = 0;
    std::vector<Complex> lambda;
    std::vector<Complex> c;
    std::vector<Complex> z;
    std::vector<Complex> c_lambda;
    std::vector<Real> residual;
    Real closure_error = 0;
    Real marked_z_closure_error = 0;
    ContinuationStats stats;
};


std::optional<MultiplierState> ring_sample_state(int n, const RingTrace& trace,
                                                 std::size_t index) {
    return state_from_solution(n, trace.lambda[index], trace.z[index], trace.c[index], 0);
}

using RingProgressCallback =
    std::function<void(int theta_points, int completed_points, int total_points,
                       const std::string& phase)>;

std::optional<RingTrace> trace_ring(int n, const MultiplierState& center_state, Real rho,
                                    int theta_points, const Config& config,
                                    const RingTrace* seed_trace,
                                    const RingProgressCallback& progress = {}) {
    RingTrace trace;
    trace.rho = rho;
    trace.lambda.resize(theta_points);
    trace.c.resize(theta_points);
    trace.z.resize(theta_points);
    trace.c_lambda.resize(theta_points);
    trace.residual.resize(theta_points);

    std::vector<MultiplierState> scaffold;
    if (seed_trace && seed_trace->rho < rho && !seed_trace->lambda.empty()) {
        scaffold.reserve(theta_points);
        const int progress_stride = std::max(1, theta_points / 16);
        for (int i = 0; i < theta_points; ++i) {
            if (progress
                && (i == 0 || i + 1 == theta_points || (i + 1) % progress_stride == 0)) {
                progress(theta_points, i + 1, theta_points, "seeding");
            }
            const Real angle = 2 * PI * static_cast<Real>(i) / theta_points;
            const Complex inner_target = seed_trace->rho * Complex{std::cos(angle), std::sin(angle)};
            const Real fractional = static_cast<Real>(i) * seed_trace->lambda.size() / theta_points;
            const std::size_t nearest = static_cast<std::size_t>(std::llround(fractional))
                                        % seed_trace->lambda.size();
            auto source = ring_sample_state(n, *seed_trace, nearest);
            if (!source) return std::nullopt;
            if (safe_abs(source->lambda - inner_target) > 128 * std::numeric_limits<Real>::epsilon()) {
                auto moved = continue_to_lambda(n, *source, inner_target, config, trace.stats);
                if (!moved) return std::nullopt;
                scaffold.push_back(*moved);
            } else {
                scaffold.push_back(*source);
            }
        }
    }

    std::optional<MultiplierState> current;
    const int trace_progress_stride = std::max(1, theta_points / 16);
    for (int i = 0; i < theta_points; ++i) {
        if (progress
            && (i == 0 || i + 1 == theta_points
                || (i + 1) % trace_progress_stride == 0)) {
            progress(theta_points, i + 1, theta_points, "tracing");
        }
        const Real angle = 2 * PI * static_cast<Real>(i) / theta_points;
        const Complex target = rho * Complex{std::cos(angle), std::sin(angle)};
        trace.lambda[i] = target;

        std::optional<MultiplierState> radial;
        std::optional<MultiplierState> angular;
        if (!scaffold.empty()) {
            radial = continue_to_lambda(n, scaffold[i], target, config, trace.stats);
        }
        if (current) {
            angular = continue_to_lambda(n, *current, target, config, trace.stats);
        } else if (scaffold.empty()) {
            angular = continue_to_lambda(n, center_state, target, config, trace.stats);
        }

        auto chosen = radial ? radial : angular;
        if (!chosen) return std::nullopt;
        if (radial && angular) {
            const Real radial_scale = safe_abs(radial->c - scaffold[i].c);
            const Real angular_scale = current ? safe_abs(angular->c - current->c) : 0;
            const Real local_scale = std::max({radial_scale, angular_scale,
                2048 * std::numeric_limits<Real>::epsilon()
                    * std::max<Real>(1, safe_abs(chosen->c))});
            if (safe_abs(radial->c - angular->c) > config.branch_jump_factor * local_scale) {
                ++trace.stats.rejected_branch_candidates;
                chosen = radial;
            }
        }

        trace.c[i] = chosen->c;
        trace.z[i] = chosen->z;
        trace.c_lambda[i] = chosen->c_lambda;
        trace.residual[i] = chosen->residual;
        current = chosen;
    }

    if (progress) progress(theta_points, theta_points, theta_points, "closing");
    auto closure = continue_to_lambda(n, *current, trace.lambda.front(), config, trace.stats);
    if (!closure) return std::nullopt;
    trace.closure_error = safe_abs(closure->c - trace.c.front());
    trace.marked_z_closure_error = safe_abs(closure->z - trace.z.front());
    return trace;
}

Real polygon_area(const std::vector<Complex>& values) {
    Real sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const Complex product = std::conj(values[i]) * values[(i + 1) % values.size()];
        sum += product.imag();
    }
    return 0.5L * sum;
}

Real derivative_area(const std::vector<Complex>& offsets,
                     const std::vector<Complex>& lambdas,
                     const std::vector<Complex>& c_lambda) {
    Real sum = 0;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        const Complex dc_dtheta = Complex{0, 1} * lambdas[i] * c_lambda[i];
        sum += 0.5L * (std::conj(offsets[i]) * dc_dtheta).imag();
    }
    return (2 * PI / static_cast<Real>(offsets.size())) * sum;
}

struct Measurement {
    int period = 0;
    int component_index = 0;
    int conjugate_index = 0;
    int symmetry_source_component_index = 0;
    Complex center{};
    Real rho = 0;
    int theta_points = 0;
    Real area_polygon = 0;
    Real area_derivative = 0;
    Real area_estimate = 0;
    Real method_spread = 0;
    Real resolution_delta = 0;
    Real error_estimate = 0;
    Real closure_error = 0;
    Real marked_z_closure_error = 0;
    Real max_residual = 0;
    ContinuationStats stats;
    Real seed_rho = std::numeric_limits<Real>::quiet_NaN();
    Real component_scale = 0;
    int mp_solve_calls = 0;
    int mp_recoveries = 0;
    int max_mp_dps = 0;
    bool converged = false;
    std::string failure_reason;
};

std::optional<MultiplierState> center_state(int period, Complex center, const Config& config) {
    auto result = solve_multiplier_point(period, Complex{0, 0}, center, Complex{0, 0}, config);
    if (result) return result;
    return state_from_solution(period, Complex{0, 0}, Complex{0, 0}, center, 0);
}

using MeasurementProgressCallback = RingProgressCallback;

Measurement measure_component(int period, int component_index, int conjugate_index,
                              Complex center, Real rho, const Config& config,
                              const RingTrace* seed_trace, RingTrace& output_trace,
                              const MeasurementProgressCallback& progress = {}) {
    Measurement measurement;
    measurement.period = period;
    measurement.component_index = component_index;
    measurement.conjugate_index = conjugate_index;
    measurement.symmetry_source_component_index = component_index;
    measurement.center = center;
    measurement.rho = rho;
    measurement.seed_rho = seed_trace ? seed_trace->rho : std::numeric_limits<Real>::quiet_NaN();

    auto state = center_state(period, center, config);
    if (!state) {
        measurement.failure_reason = "could not construct center continuation state";
        return measurement;
    }

    int theta = 1;
    while (theta < config.theta_start) theta <<= 1;
    const int theta_max = std::max(theta, config.theta_max);

    while (theta <= theta_max) {
        if (progress) progress(theta, 0, theta, "starting");
        auto trace = trace_ring(
            period, *state, rho, theta, config, seed_trace, progress);
        if (!trace) {
            if (progress) progress(theta, theta, theta, "retrying");
            if (theta == theta_max) {
                measurement.failure_reason = "ring tracing failed";
                return measurement;
            }
            theta <<= 1;
            continue;
        }

        std::vector<Complex> offsets(theta);
        for (int i = 0; i < theta; ++i) offsets[i] = trace->c[i] - center;
        const Real polygon = std::abs(polygon_area(offsets));
        const Real derivative = std::abs(derivative_area(offsets, trace->lambda, trace->c_lambda));
        // The derivative line integral is spectrally accurate for the smooth
        // periodic multiplier parameterization. The inscribed-polygon area is
        // retained as a diagnostic, but it converges only as O(N^-2) and must
        // not prevent otherwise excellent traces from being accepted.
        const Real estimate = derivative;
        const Real method_spread = std::abs(polygon - derivative);

        Real resolution = std::numeric_limits<Real>::infinity();
        if (theta >= 16 && theta % 2 == 0) {
            std::vector<Complex> coarse_offsets;
            std::vector<Complex> coarse_lambda;
            std::vector<Complex> coarse_c_lambda;
            coarse_offsets.reserve(theta / 2);
            coarse_lambda.reserve(theta / 2);
            coarse_c_lambda.reserve(theta / 2);
            for (int i = 0; i < theta; i += 2) {
                coarse_offsets.push_back(offsets[i]);
                coarse_lambda.push_back(trace->lambda[i]);
                coarse_c_lambda.push_back(trace->c_lambda[i]);
            }
            const Real coarse_derivative = std::abs(
                derivative_area(coarse_offsets, coarse_lambda, coarse_c_lambda));
            resolution = std::abs(estimate - coarse_derivative);
        }

        const Real requested = config.area_atol + config.area_rtol * std::max(estimate, config.area_atol);
        Real component_scale = 0;
        for (const auto& offset : offsets) component_scale = std::max(component_scale, safe_abs(offset));
        const bool closure_ok = trace->closure_error <= std::max({
            1.0e-14L, 50 * config.newton_tolerance, 1.0e-8L * component_scale});
        const bool resolution_ok = finite(resolution) && resolution <= requested;

        measurement.theta_points = theta;
        measurement.area_polygon = polygon;
        measurement.area_derivative = derivative;
        measurement.area_estimate = estimate;
        measurement.method_spread = method_spread;
        measurement.resolution_delta = finite(resolution) ? resolution : method_spread;
        measurement.error_estimate = measurement.resolution_delta;
        measurement.closure_error = trace->closure_error;
        measurement.marked_z_closure_error = trace->marked_z_closure_error;
        measurement.stats = trace->stats;
        measurement.component_scale = component_scale;
        measurement.max_residual = trace->stats.max_residual;
        for (Real value : trace->residual) measurement.max_residual = std::max(measurement.max_residual, value);
        measurement.converged = closure_ok && resolution_ok;
        output_trace = std::move(*trace);

        if (measurement.converged || theta == theta_max) return measurement;
        theta <<= 1;
    }
    measurement.failure_reason = "no usable ring trace";
    return measurement;
}


// -----------------------------------------------------------------------------
// Adaptive arbitrary-precision component continuation
// -----------------------------------------------------------------------------

struct PrecisionPlan {
    bool use_mp = false;
    int requested_dps = 0;
    int actual_dps = 0;
    Real estimated_scale = 0;
    Real ulp_scale = 0;
    Real digits_above_ulp = std::numeric_limits<Real>::infinity();
    std::string reason;
};

int mp_precision_tier(int requested) {
    if (requested <= 50) return 50;
    if (requested <= 80) return 80;
    if (requested <= 120) return 120;
    if (requested <= 160) return 160;
    return 200;
}

PrecisionPlan assess_component_precision(int period, Complex center, const Config& config) {
    PrecisionPlan plan;
    if (!config.mp_fallback) return plan;

    auto state = center_state(period, center, config);
    if (!state) {
        plan.use_mp = true;
        plan.requested_dps = config.mp_min_dps;
        plan.actual_dps = mp_precision_tier(plan.requested_dps);
        plan.reason = "long-double center state failed";
        return plan;
    }

    const Real max_rho = config.radii.empty() ? 1 : config.radii.back();
    plan.estimated_scale = safe_abs(state->c_lambda) * max_rho;
    plan.ulp_scale = std::numeric_limits<Real>::epsilon()
                   * std::max<Real>(1, safe_abs(center));

    if (plan.estimated_scale > 0 && plan.ulp_scale > 0) {
        plan.digits_above_ulp = std::log10(plan.estimated_scale / plan.ulp_scale);
    } else {
        plan.digits_above_ulp = -std::numeric_limits<Real>::infinity();
    }

    const Real coordinate_scale = std::max<Real>(1, safe_abs(center));
    int lost_decimal_digits = config.mp_max_dps;
    if (plan.estimated_scale > 0 && finite(plan.estimated_scale)) {
        lost_decimal_digits = static_cast<int>(std::ceil(std::max<Real>(
            0, -std::log10(plan.estimated_scale / coordinate_scale))));
    }
    plan.requested_dps = std::clamp(
        std::max(config.mp_min_dps, lost_decimal_digits + config.mp_extra_digits),
        config.mp_min_dps,
        config.mp_max_dps);
    plan.actual_dps = mp_precision_tier(plan.requested_dps);

    if (config.mp_proactive
        && plan.digits_above_ulp < static_cast<Real>(config.mp_guard_digits)) {
        plan.use_mp = true;
        std::ostringstream reason;
        reason << "estimated component scale " << real_string(plan.estimated_scale, 4)
               << " is only " << std::fixed << std::setprecision(1)
               << plan.digits_above_ulp
               << " decimal digits above the local long-double ulp";
        plan.reason = reason.str();
    }
    return plan;
}

bool long_double_measurement_needs_mp(
    const Measurement& measurement,
    Complex center,
    const Config& config
) {
    if (!config.mp_fallback) return false;
    if (!measurement.converged) return true;
    if (!config.mp_proactive) return false;

    const Real ulp = std::numeric_limits<Real>::epsilon()
                   * std::max<Real>(1, safe_abs(center));
    if (!(measurement.component_scale > 0) || !(ulp > 0)) return true;
    const Real digits = std::log10(measurement.component_scale / ulp);
    return digits < static_cast<Real>(config.mp_guard_digits);
}

template <unsigned Digits10>
using MpReal = boost::multiprecision::number<
    boost::multiprecision::cpp_dec_float<Digits10>>;

template <typename MP>
using MpComplex = std::complex<MP>;

template <typename MP>
bool mp_finite(const MP& value) {
    return boost::multiprecision::isfinite(value);
}

template <typename MP>
bool mp_finite(const MpComplex<MP>& value) {
    return mp_finite(value.real()) && mp_finite(value.imag());
}

template <typename MP>
MP mp_abs(const MpComplex<MP>& value) {
    return std::abs(value);
}

template <typename MP>
MP mp_pow10(int exponent) {
    using boost::multiprecision::pow;
    return pow(MP(10), exponent);
}

template <typename MP>
MP mp_from_real(Real value) {
    return MP(real_string(value));
}

template <typename MP>
Real real_from_mp(const MP& value) {
    try {
        return value.template convert_to<Real>();
    } catch (...) {
        return value < 0 ? -std::numeric_limits<Real>::infinity()
                         : std::numeric_limits<Real>::infinity();
    }
}

template <typename MP>
struct MpOrbitData {
    MpComplex<MP> A{}, B{}, C{}, D{}, E{};
};

template <typename MP>
MpOrbitData<MP> mp_iterate_data(MpComplex<MP> z, MpComplex<MP> c, int n) {
    MpComplex<MP> A = z;
    MpComplex<MP> B{MP(1), MP(0)};
    MpComplex<MP> C{MP(0), MP(0)};
    MpComplex<MP> D{MP(0), MP(0)};
    MpComplex<MP> E{MP(0), MP(0)};
    for (int i = 0; i < n; ++i) {
        const auto A0 = A, B0 = B, C0 = C, D0 = D, E0 = E;
        A = A0 * A0 + c;
        B = MP(2) * A0 * B0;
        C = MP(2) * A0 * C0 + MpComplex<MP>{MP(1), MP(0)};
        D = MP(2) * (B0 * B0 + A0 * D0);
        E = MP(2) * (C0 * B0 + A0 * E0);
    }
    return {A, B, C, D, E};
}

template <typename MP>
std::pair<MpComplex<MP>, MpComplex<MP>> mp_critical_orbit(
    MpComplex<MP> c, int n
) {
    MpComplex<MP> z{MP(0), MP(0)};
    MpComplex<MP> derivative{MP(0), MP(0)};
    for (int i = 0; i < n; ++i) {
        derivative = MP(2) * z * derivative + MpComplex<MP>{MP(1), MP(0)};
        z = z * z + c;
    }
    return {z, derivative};
}

template <typename MP>
MpComplex<MP> mp_refine_center(
    MpComplex<MP> center, int period, const MP& tolerance, int max_iterations
) {
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        auto [value, derivative] = mp_critical_orbit(center, period);
        if (mp_abs(value) <= tolerance) return center;
        if (mp_abs(derivative) == 0) break;
        center -= value / derivative;
    }
    auto [value, derivative] = mp_critical_orbit(center, period);
    (void)derivative;
    if (mp_abs(value) > MP(100) * tolerance) {
        throw std::runtime_error("multiprecision center refinement failed");
    }
    return center;
}

template <typename MP>
struct MpState {
    MpComplex<MP> lambda{}, z{}, c{}, z_lambda{}, c_lambda{};
    MP residual = 0;
    int newton_iterations = 0;
};

template <typename MP>
std::optional<MpState<MP>> mp_state_from_solution(
    int n,
    MpComplex<MP> lambda,
    MpComplex<MP> z,
    MpComplex<MP> c,
    int iterations,
    const MP& determinant_floor
) {
    const auto data = mp_iterate_data(z, c, n);
    const MP residual = mp_abs(data.A - z) + mp_abs(data.B - lambda);
    const auto determinant = (data.B - MpComplex<MP>{MP(1), MP(0)}) * data.E
                           - data.C * data.D;
    if (!mp_finite(determinant) || mp_abs(determinant) <= determinant_floor) {
        return std::nullopt;
    }
    const auto z_lambda = -data.C / determinant;
    const auto c_lambda = (data.B - MpComplex<MP>{MP(1), MP(0)}) / determinant;
    if (!mp_finite(z_lambda) || !mp_finite(c_lambda)) return std::nullopt;
    return MpState<MP>{lambda, z, c, z_lambda, c_lambda, residual, iterations};
}

template <typename MP>
bool mp_periodic_point_has_exact_period(
    int n, MpComplex<MP> z, MpComplex<MP> c, const MP& tolerance
) {
    const MP scale = std::max(MP(1), mp_abs(z));
    for (int d : proper_divisors(n)) {
        auto value = z;
        for (int k = 0; k < d; ++k) value = value * value + c;
        if (mp_abs(value - z) <= tolerance * scale) return false;
    }
    return true;
}

template <typename MP>
std::optional<MpState<MP>> mp_solve_multiplier_point(
    int n,
    MpComplex<MP> z_guess,
    MpComplex<MP> c_guess,
    MpComplex<MP> lambda,
    const MP& tolerance,
    const MP& determinant_floor,
    int max_iterations,
    ContinuationStats& stats
) {
    auto z = z_guess;
    auto c = c_guess;
    ++stats.solve_calls;
    for (int iteration = 0; iteration <= max_iterations; ++iteration) {
        const auto data = mp_iterate_data(z, c, n);
        const auto G1 = data.A - z;
        const auto G2 = data.B - lambda;
        const MP residual = mp_abs(G1) + mp_abs(G2);
        if (residual <= tolerance) {
            auto state = mp_state_from_solution(
                n, lambda, z, c, iteration, determinant_floor);
            if (state) {
                stats.newton_iterations += iteration;
                stats.max_residual = std::max(
                    stats.max_residual,
                    real_from_mp(state->residual));
            }
            return state;
        }
        if (iteration == max_iterations) break;
        const auto determinant = (data.B - MpComplex<MP>{MP(1), MP(0)}) * data.E
                               - data.C * data.D;
        if (!mp_finite(determinant) || mp_abs(determinant) <= determinant_floor) break;
        const auto dz = (G1 * data.E - data.C * G2) / determinant;
        const auto dc = ((data.B - MpComplex<MP>{MP(1), MP(0)}) * G2
                       - data.D * G1) / determinant;
        z -= dz;
        c -= dc;
        if (!mp_finite(z) || !mp_finite(c)) break;
    }
    ++stats.failed_attempts;
    return std::nullopt;
}

template <typename MP>
MP mp_predictor_score(
    const MpState<MP>& start,
    MpComplex<MP> target,
    const MpState<MP>& candidate,
    const MP& tolerance
) {
    const auto delta = target - start.lambda;
    const auto predicted_z = start.z + start.z_lambda * delta;
    const auto predicted_c = start.c + start.c_lambda * delta;
    const MP tiny = MP(32) * tolerance;
    const MP c_scale = std::max(mp_abs(start.c_lambda * delta), tiny);
    const MP z_scale = std::max(mp_abs(start.z_lambda * delta), tiny);
    return std::max(
        mp_abs(candidate.c - predicted_c) / c_scale,
        mp_abs(candidate.z - predicted_z) / z_scale);
}

template <typename MP>
std::vector<MpState<MP>> mp_cyclic_marked_states(
    int n,
    const MpState<MP>& state,
    const MP& determinant_floor
) {
    std::vector<MpState<MP>> result{state};
    auto z = state.z;
    for (int i = 1; i < n; ++i) {
        z = z * z + state.c;
        auto marked = mp_state_from_solution(
            n, state.lambda, z, state.c, 0, determinant_floor);
        if (marked) result.push_back(*marked);
    }
    return result;
}

template <typename MP>
std::optional<MpState<MP>> mp_solve_local_branch(
    int n,
    const MpState<MP>& start,
    MpComplex<MP> target,
    const MP& tolerance,
    const MP& exact_period_tolerance,
    const MP& determinant_floor,
    int max_iterations,
    Real branch_jump_factor,
    ContinuationStats& stats
) {
    auto attempt = [&](const MpState<MP>& seed) -> std::optional<MpState<MP>> {
        const auto delta = target - seed.lambda;
        const auto z_guess = seed.z + seed.z_lambda * delta;
        const auto c_guess = seed.c + seed.c_lambda * delta;
        auto result = mp_solve_multiplier_point(
            n, z_guess, c_guess, target, tolerance, determinant_floor,
            max_iterations, stats);
        if (!result) return std::nullopt;
        if (!mp_periodic_point_has_exact_period(
                n, result->z, result->c, exact_period_tolerance)
            || mp_predictor_score(seed, target, *result, tolerance)
               > MP(branch_jump_factor)) {
            ++stats.rejected_branch_candidates;
            return std::nullopt;
        }
        return result;
    };

    if (auto result = attempt(start)) return result;
    const auto seeds = mp_cyclic_marked_states(n, start, determinant_floor);
    for (std::size_t i = 1; i < seeds.size(); ++i) {
        ++stats.cyclic_seed_attempts;
        if (auto result = attempt(seeds[i])) {
            ++stats.cyclic_recoveries;
            return result;
        }
    }
    return std::nullopt;
}

template <typename MP>
std::optional<MpState<MP>> mp_continue_to_lambda(
    int n,
    const MpState<MP>& start,
    MpComplex<MP> target,
    const MP& tolerance,
    const MP& exact_period_tolerance,
    const MP& determinant_floor,
    int max_iterations,
    int max_depth,
    const MP& max_step,
    Real branch_jump_factor,
    ContinuationStats& stats,
    int depth = 0
) {
    stats.max_subdivision_depth = std::max(stats.max_subdivision_depth, depth);
    const auto delta = target - start.lambda;
    if (mp_abs(delta) > max_step) {
        if (depth >= max_depth) return std::nullopt;
        const auto middle = (start.lambda + target) / MP(2);
        auto first = mp_continue_to_lambda(
            n, start, middle, tolerance, exact_period_tolerance,
            determinant_floor, max_iterations, max_depth, max_step,
            branch_jump_factor, stats, depth + 1);
        if (!first) return std::nullopt;
        return mp_continue_to_lambda(
            n, *first, target, tolerance, exact_period_tolerance,
            determinant_floor, max_iterations, max_depth, max_step,
            branch_jump_factor, stats, depth + 1);
    }

    if (auto result = mp_solve_local_branch(
            n, start, target, tolerance, exact_period_tolerance,
            determinant_floor, max_iterations, branch_jump_factor, stats)) {
        return result;
    }
    if (depth >= max_depth) return std::nullopt;
    const auto middle = (start.lambda + target) / MP(2);
    auto first = mp_continue_to_lambda(
        n, start, middle, tolerance, exact_period_tolerance,
        determinant_floor, max_iterations, max_depth, max_step,
        branch_jump_factor, stats, depth + 1);
    if (!first) return std::nullopt;
    return mp_continue_to_lambda(
        n, *first, target, tolerance, exact_period_tolerance,
        determinant_floor, max_iterations, max_depth, max_step,
        branch_jump_factor, stats, depth + 1);
}

template <typename MP>
struct MpRing {
    MP rho = 0;
    std::vector<MpComplex<MP>> lambda, c, z, c_lambda;
    std::vector<MP> residual;
    MP closure_error = 0;
    MP marked_z_closure_error = 0;
    ContinuationStats stats;
};

template <typename MP>
std::optional<MpState<MP>> mp_ring_sample_state(
    int n,
    const MpRing<MP>& ring,
    std::size_t index,
    const MP& determinant_floor
) {
    return mp_state_from_solution(
        n, ring.lambda[index], ring.z[index], ring.c[index], 0,
        determinant_floor);
}

template <typename MP>
std::optional<MpRing<MP>> mp_trace_ring(
    int n,
    const MpState<MP>& center_state,
    const MP& rho,
    int theta_points,
    const Config& config,
    const MP& tolerance,
    const MP& exact_period_tolerance,
    const MP& determinant_floor,
    const MpRing<MP>* seed_ring,
    const MeasurementProgressCallback& progress
) {
    MpRing<MP> ring;
    ring.rho = rho;
    ring.lambda.resize(theta_points);
    ring.c.resize(theta_points);
    ring.z.resize(theta_points);
    ring.c_lambda.resize(theta_points);
    ring.residual.resize(theta_points);

    const MP pi = boost::math::constants::pi<MP>();
    const MP max_step = mp_from_real<MP>(config.mp_continuation_max_step);
    std::vector<MpState<MP>> scaffold;

    if (seed_ring && seed_ring->rho < rho && !seed_ring->lambda.empty()) {
        scaffold.reserve(theta_points);
        for (int i = 0; i < theta_points; ++i) {
            if (progress && (i == 0 || i + 1 == theta_points
                || (i + 1) % std::max(1, theta_points / 16) == 0)) {
                progress(theta_points, i + 1, theta_points, "MP seeding");
            }
            const MP angle = MP(2) * pi * MP(i) / MP(theta_points);
            using boost::multiprecision::cos;
            using boost::multiprecision::sin;
            const MpComplex<MP> inner_target{
                seed_ring->rho * cos(angle), seed_ring->rho * sin(angle)};
            const Real fractional = static_cast<Real>(i)
                                  * seed_ring->lambda.size() / theta_points;
            const std::size_t nearest = static_cast<std::size_t>(std::llround(fractional))
                                      % seed_ring->lambda.size();
            auto source = mp_ring_sample_state(n, *seed_ring, nearest, determinant_floor);
            if (!source) return std::nullopt;
            auto moved = mp_continue_to_lambda(
                n, *source, inner_target, tolerance, exact_period_tolerance,
                determinant_floor, config.mp_newton_max_iterations,
                config.mp_continuation_max_depth, max_step,
                config.branch_jump_factor, ring.stats);
            if (!moved) return std::nullopt;
            scaffold.push_back(*moved);
        }
    }

    std::optional<MpState<MP>> current;
    for (int i = 0; i < theta_points; ++i) {
        if (progress && (i == 0 || i + 1 == theta_points
            || (i + 1) % std::max(1, theta_points / 16) == 0)) {
            progress(theta_points, i + 1, theta_points, "MP tracing");
        }
        const MP angle = MP(2) * pi * MP(i) / MP(theta_points);
        using boost::multiprecision::cos;
        using boost::multiprecision::sin;
        const MpComplex<MP> target{rho * cos(angle), rho * sin(angle)};
        ring.lambda[i] = target;

        std::optional<MpState<MP>> radial;
        std::optional<MpState<MP>> angular;
        if (!scaffold.empty()) {
            radial = mp_continue_to_lambda(
                n, scaffold[i], target, tolerance, exact_period_tolerance,
                determinant_floor, config.mp_newton_max_iterations,
                config.mp_continuation_max_depth, max_step,
                config.branch_jump_factor, ring.stats);
        }
        if (current) {
            angular = mp_continue_to_lambda(
                n, *current, target, tolerance, exact_period_tolerance,
                determinant_floor, config.mp_newton_max_iterations,
                config.mp_continuation_max_depth, max_step,
                config.branch_jump_factor, ring.stats);
        } else if (scaffold.empty()) {
            angular = mp_continue_to_lambda(
                n, center_state, target, tolerance, exact_period_tolerance,
                determinant_floor, config.mp_newton_max_iterations,
                config.mp_continuation_max_depth, max_step,
                config.branch_jump_factor, ring.stats);
        }

        auto chosen = radial ? radial : angular;
        if (!chosen) return std::nullopt;
        if (radial && angular) {
            const MP radial_scale = mp_abs(radial->c - scaffold[i].c);
            const MP angular_scale = current ? mp_abs(angular->c - current->c) : MP(0);
            const MP local_scale = std::max({
                radial_scale, angular_scale, MP(32) * tolerance});
            if (mp_abs(radial->c - angular->c)
                > MP(config.branch_jump_factor) * local_scale) {
                ++ring.stats.rejected_branch_candidates;
                chosen = radial;
            }
        }

        ring.c[i] = chosen->c;
        ring.z[i] = chosen->z;
        ring.c_lambda[i] = chosen->c_lambda;
        ring.residual[i] = chosen->residual;
        current = chosen;
    }

    if (progress) progress(theta_points, theta_points, theta_points, "MP closing");
    auto closure = mp_continue_to_lambda(
        n, *current, ring.lambda.front(), tolerance, exact_period_tolerance,
        determinant_floor, config.mp_newton_max_iterations,
        config.mp_continuation_max_depth, max_step,
        config.branch_jump_factor, ring.stats);
    if (!closure) return std::nullopt;
    ring.closure_error = mp_abs(closure->c - ring.c.front());
    ring.marked_z_closure_error = mp_abs(closure->z - ring.z.front());
    return ring;
}

template <typename MP>
MP mp_polygon_area(const std::vector<MpComplex<MP>>& values) {
    MP sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        sum += (std::conj(values[i]) * values[(i + 1) % values.size()]).imag();
    }
    return sum / MP(2);
}

template <typename MP>
MP mp_derivative_area(
    const std::vector<MpComplex<MP>>& offsets,
    const std::vector<MpComplex<MP>>& lambdas,
    const std::vector<MpComplex<MP>>& c_lambda
) {
    MP sum = 0;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        const MpComplex<MP> dc_dtheta = MpComplex<MP>{MP(0), MP(1)}
                                         * lambdas[i] * c_lambda[i];
        sum += (std::conj(offsets[i]) * dc_dtheta).imag() / MP(2);
    }
    return MP(2) * boost::math::constants::pi<MP>()
         * sum / MP(offsets.size());
}

template <unsigned Digits10>
std::vector<Measurement> measure_component_mp_impl(
    int period,
    int component_index,
    int conjugate_index,
    Complex center,
    const Config& config,
    const MeasurementProgressCallback& progress
) {
    using MP = MpReal<Digits10>;
    using MPC = MpComplex<MP>;

    const MP tolerance = mp_pow10<MP>(-static_cast<int>(Digits10 - 15));
    const MP exact_tolerance = mp_pow10<MP>(-static_cast<int>(Digits10 / 2));
    const MP determinant_floor = mp_pow10<MP>(-static_cast<int>(Digits10 - 10));
    const MP area_floor = mp_pow10<MP>(-static_cast<int>(Digits10 - 10));

    MPC mp_center{
        MP(real_string(center.real())),
        MP(real_string(center.imag()))};
    mp_center = mp_refine_center(
        mp_center, period, tolerance, config.mp_newton_max_iterations);

    auto center_state = mp_state_from_solution(
        period, MPC{MP(0), MP(0)}, MPC{MP(0), MP(0)}, mp_center, 0,
        determinant_floor);
    if (!center_state) {
        throw std::runtime_error("could not construct multiprecision center state");
    }

    std::vector<Measurement> result;
    result.reserve(config.radii.size());
    std::optional<MpRing<MP>> seed_ring;

    for (std::size_t radius_index = 0; radius_index < config.radii.size(); ++radius_index) {
        const Real rho_real = config.radii[radius_index];
        const MP rho = mp_from_real<MP>(rho_real);
        int theta = 1;
        while (theta < config.theta_start) theta <<= 1;
        const int theta_max = std::max(theta, config.theta_max);
        Measurement measurement;
        measurement.period = period;
        measurement.component_index = component_index;
        measurement.conjugate_index = conjugate_index;
        measurement.symmetry_source_component_index = component_index;
        measurement.center = center;
        measurement.rho = rho_real;
        measurement.seed_rho = seed_ring
            ? real_from_mp(seed_ring->rho)
            : std::numeric_limits<Real>::quiet_NaN();
        measurement.max_mp_dps = Digits10;

        while (theta <= theta_max) {
            if (progress) progress(theta, 0, theta, "MP starting");
            auto ring = mp_trace_ring(
                period, *center_state, rho, theta, config, tolerance,
                exact_tolerance, determinant_floor,
                seed_ring ? &*seed_ring : nullptr, progress);
            if (!ring) {
                if (theta == theta_max) {
                    measurement.failure_reason = "multiprecision ring tracing failed";
                    break;
                }
                theta <<= 1;
                continue;
            }

            std::vector<MPC> offsets(theta);
            MP component_scale = 0;
            for (int i = 0; i < theta; ++i) {
                offsets[i] = ring->c[i] - mp_center;
                component_scale = std::max(component_scale, mp_abs(offsets[i]));
            }
            const MP polygon = abs(mp_polygon_area(offsets));
            const MP derivative = abs(mp_derivative_area(
                offsets, ring->lambda, ring->c_lambda));

            std::vector<MPC> coarse_offsets, coarse_lambda, coarse_c_lambda;
            coarse_offsets.reserve(theta / 2);
            coarse_lambda.reserve(theta / 2);
            coarse_c_lambda.reserve(theta / 2);
            for (int i = 0; i < theta; i += 2) {
                coarse_offsets.push_back(offsets[i]);
                coarse_lambda.push_back(ring->lambda[i]);
                coarse_c_lambda.push_back(ring->c_lambda[i]);
            }
            const MP coarse = abs(mp_derivative_area(
                coarse_offsets, coarse_lambda, coarse_c_lambda));
            const MP resolution = abs(derivative - coarse);
            const MP requested = std::max(
                MP(config.area_rtol) * derivative,
                area_floor);
            const MP closure_requested = std::max(
                MP(100) * tolerance,
                MP("1e-10") * component_scale);

            measurement.theta_points = theta;
            measurement.area_polygon = real_from_mp(polygon);
            measurement.area_derivative = real_from_mp(derivative);
            measurement.area_estimate = real_from_mp(derivative);
            measurement.method_spread = real_from_mp(MP(abs(polygon - derivative)));
            measurement.resolution_delta = real_from_mp(resolution);
            measurement.error_estimate = measurement.resolution_delta;
            measurement.closure_error = real_from_mp(ring->closure_error);
            measurement.marked_z_closure_error = real_from_mp(
                ring->marked_z_closure_error);
            measurement.component_scale = real_from_mp(component_scale);
            measurement.stats = ring->stats;
            measurement.max_residual = ring->stats.max_residual;
            for (const auto& value : ring->residual) {
                measurement.max_residual = std::max(
                    measurement.max_residual, real_from_mp(value));
            }
            measurement.mp_solve_calls = static_cast<int>(std::min<long long>(
                ring->stats.solve_calls, std::numeric_limits<int>::max()));
            measurement.mp_recoveries = 1;
            measurement.converged = ring->closure_error <= closure_requested
                                  && ring->marked_z_closure_error <= closure_requested
                                  && resolution <= requested;

            if (measurement.converged || theta == theta_max) {
                if (!measurement.converged) {
                    measurement.failure_reason =
                        "multiprecision ring did not satisfy closure/resolution tolerance";
                } else {
                    seed_ring = std::move(*ring);
                }
                break;
            }
            theta <<= 1;
        }

        result.push_back(measurement);
        if (!measurement.converged) {
            // Later radii depend on this trusted inner ring. Return explicit
            // failures for them instead of pretending independent traces are comparable.
            for (std::size_t j = radius_index + 1; j < config.radii.size(); ++j) {
                Measurement skipped = measurement;
                skipped.rho = config.radii[j];
                skipped.theta_points = 0;
                skipped.area_polygon = 0;
                skipped.area_derivative = 0;
                skipped.area_estimate = 0;
                skipped.failure_reason = "skipped after earlier multiprecision radius failed";
                result.push_back(skipped);
            }
            break;
        }
    }
    return result;
}

std::vector<Measurement> measure_component_mp(
    int period,
    int component_index,
    int conjugate_index,
    Complex center,
    const Config& config,
    int requested_dps,
    const MeasurementProgressCallback& progress
) {
    const int tier = mp_precision_tier(requested_dps);
    if (tier > config.mp_max_dps) {
        throw std::runtime_error(
            "required multiprecision exceeds mp_max_dps="
            + std::to_string(config.mp_max_dps));
    }
    if (tier <= 50) {
        return measure_component_mp_impl<50>(
            period, component_index, conjugate_index, center, config, progress);
    }
    if (tier <= 80) {
        return measure_component_mp_impl<80>(
            period, component_index, conjugate_index, center, config, progress);
    }
    if (tier <= 120) {
        return measure_component_mp_impl<120>(
            period, component_index, conjugate_index, center, config, progress);
    }
    if (tier <= 160) {
        return measure_component_mp_impl<160>(
            period, component_index, conjugate_index, center, config, progress);
    }
    return measure_component_mp_impl<200>(
        period, component_index, conjugate_index, center, config, progress);
}

AreaMeasurementRecord measurement_to_record(const Measurement& value) {
    AreaMeasurementRecord record;
    record.period = value.period;
    record.component_index = value.component_index;
    record.conjugate_index = value.conjugate_index;
    record.symmetry_source_component_index = value.symmetry_source_component_index;
    record.center = {catalogue_real(value.center.real()), catalogue_real(value.center.imag())};
    record.rho = catalogue_real(value.rho);
    record.theta_points = value.theta_points;
    record.area_polygon = catalogue_real(value.area_polygon);
    record.area_derivative = catalogue_real(value.area_derivative);
    record.area_estimate = catalogue_real(value.area_estimate);
    record.method_spread = catalogue_real(value.method_spread);
    record.spectral_spread = catalogue_real(value.method_spread);
    record.resolution_delta = catalogue_real(value.resolution_delta);
    record.error_estimate = catalogue_real(value.error_estimate);
    record.closure_error = catalogue_real(value.closure_error);
    record.marked_z_closure_error = catalogue_real(value.marked_z_closure_error);
    record.max_residual = catalogue_real(value.max_residual);
    record.solve_calls = value.stats.solve_calls;
    record.failed_attempts = value.stats.failed_attempts;
    record.newton_iterations = value.stats.newton_iterations;
    record.max_subdivision_depth = value.stats.max_subdivision_depth;
    record.rejected_branch_candidates = value.stats.rejected_branch_candidates;
    record.cyclic_seed_attempts = value.stats.cyclic_seed_attempts;
    record.cyclic_recoveries = value.stats.cyclic_recoveries;
    record.mp_solve_calls = value.mp_solve_calls;
    record.mp_recoveries = value.mp_recoveries;
    record.max_mp_dps = value.max_mp_dps;
    if (finite(value.seed_rho)) record.seed_rho = catalogue_real(value.seed_rho);
    record.converged = value.converged;
    record.failure_reason = value.failure_reason;
    return record;
}

std::string measurement_key(int period, int component, Real rho) {
    return std::to_string(period) + ":" + std::to_string(component) + ":" + real_string(rho, 18);
}

Real canonical_measurement_rho(Real rho, const Config& config) {
    if (config.radii.empty() || !finite(rho)) return rho;
    Real nearest = config.radii.front();
    Real distance = std::abs(rho - nearest);
    for (Real candidate : config.radii) {
        const Real candidate_distance = std::abs(rho - candidate);
        if (candidate_distance < distance) {
            nearest = candidate;
            distance = candidate_distance;
        }
    }
    const Real tolerance = std::max(
        1.0e-14L,
        1024.0L * std::numeric_limits<Real>::epsilon()
            * std::max<Real>(1.0L, std::abs(nearest)));
    return distance <= tolerance ? nearest : rho;
}

Real measurement_relative_error(const AreaMeasurementRecord& row) {
    const Real area = std::abs(scanner_real(row.area_estimate));
    const Real error = std::abs(scanner_real(row.error_estimate));
    if (finite(area) && area > 0 && finite(error)) return error / area;
    return std::numeric_limits<Real>::infinity();
}

bool usable_measurement_row(const AreaMeasurementRecord& row) {
    const Real area = scanner_real(row.area_estimate);
    return row.converged && finite(area) && area > 0;
}

bool complete_area_period(
    const std::vector<AreaPeriodSummaryRecord>& summaries,
    const std::vector<CenterRow>& centers,
    int period,
    const Config& config
) {
    // A summary is published only after the corresponding measurement rows
    // have been committed.  It is therefore the cheap resume marker.  If a
    // crash lands between those two commits, this check returns false and the
    // area stage performs its normal row-level recovery without recomputing
    // measurements that are already present.
    std::unordered_map<std::string, const AreaPeriodSummaryRecord*> rows;
    rows.reserve(config.radii.size());
    for (const auto& row : summaries) {
        if (row.period == period) {
            const Real rho = canonical_measurement_rho(scanner_real(row.rho), config);
            rows[real_string(rho, 18)] = &row;
        }
    }

    const std::size_t expected_components = centers.size();
    const std::size_t total_checks = config.radii.size();
    const std::string check_label =
        "  checking area summary p" + std::to_string(period);
    const auto check_started = Clock::now();
    std::size_t checked = 0;
    if (total_checks > 0) {
        render_inline_progress(
            config, check_label, 0, total_checks, check_started);
    }
    for (Real rho : config.radii) {
        ++checked;
        const auto found = rows.find(real_string(rho, 18));
        const bool complete = found != rows.end()
            && found->second->period_complete
            && found->second->expected_components
                == static_cast<int>(expected_components)
            && found->second->completed_components
                == static_cast<int>(expected_components)
            && found->second->converged_components
                == static_cast<int>(expected_components)
            && found->second->missing_or_unconverged_components == 0;
        if (!complete) {
            render_inline_progress(
                config,
                check_label,
                checked,
                total_checks,
                check_started,
                "incomplete",
                true,
                false);
            return false;
        }
        if (checked == total_checks || checked % 4096 == 0) {
            render_inline_progress(
                config,
                check_label,
                checked,
                total_checks,
                check_started,
                {},
                checked == total_checks,
                true);
        }
    }
    return true;
}

bool prefer_measurement_row(
    const AreaMeasurementRecord& candidate,
    const AreaMeasurementRecord& current
) {
    const bool candidate_usable = usable_measurement_row(candidate);
    const bool current_usable = usable_measurement_row(current);
    if (candidate_usable != current_usable) return candidate_usable;
    const Real candidate_error = measurement_relative_error(candidate);
    const Real current_error = measurement_relative_error(current);
    if (candidate_error != current_error) return candidate_error < current_error;
    if (candidate.theta_points != current.theta_points) {
        return candidate.theta_points > current.theta_points;
    }
    return true;
}

std::size_t canonicalize_measurement_rows(
    std::vector<AreaMeasurementRecord>& measurements,
    const Config& config,
    const std::string& label = "  canonicalizing measurements"
) {
    const std::size_t original_size = measurements.size();
    std::unordered_map<std::string, AreaMeasurementRecord> unique;
    unique.reserve(measurements.size());

    const auto canonicalize_started = Clock::now();
    if (!measurements.empty()) {
        render_inline_progress(
            config, label, 0, measurements.size(), canonicalize_started);
    }
    std::size_t processed = 0;
    for (auto row : measurements) {
        const Real rho = canonical_measurement_rho(scanner_real(row.rho), config);
        row.rho = catalogue_real(rho);
        const std::string key = measurement_key(row.period, row.component_index, rho);
        const auto found = unique.find(key);
        if (found == unique.end()) unique.emplace(key, std::move(row));
        else if (prefer_measurement_row(row, found->second)) found->second = std::move(row);
        ++processed;
        if (processed == original_size || processed % 4096 == 0) {
            render_inline_progress(
                config,
                label,
                processed,
                original_size,
                canonicalize_started,
                "unique=" + std::to_string(unique.size()),
                processed == original_size,
                true);
        }
    }

    measurements.clear();
    measurements.reserve(unique.size());
    const auto pack_started = Clock::now();
    const std::size_t unique_size = unique.size();
    std::size_t packed = 0;
    if (unique_size > 0) {
        render_inline_progress(
            config, "  packing measurements", 0, unique_size, pack_started);
    }
    for (auto& [_, row] : unique) {
        measurements.push_back(std::move(row));
        ++packed;
        if (packed == unique_size || packed % 4096 == 0) {
            render_inline_progress(
                config,
                "  packing measurements",
                packed,
                unique_size,
                pack_started,
                {},
                packed == unique_size,
                true);
        }
    }
    return original_size - measurements.size();
}

struct AreaSummaryBucket {
    std::vector<Real> areas;
    Real error_sum = 0;
    int completed = 0;
    int converged = 0;
};

Real quantile_sorted(const std::vector<Real>& values, Real q) {
    if (values.empty()) return std::numeric_limits<Real>::quiet_NaN();
    const Real position = q * static_cast<Real>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(position));
    const auto hi = static_cast<std::size_t>(std::ceil(position));
    if (lo == hi) return values[lo];
    const Real fraction = position - static_cast<Real>(lo);
    return values[lo] * (1 - fraction) + values[hi] * fraction;
}

std::optional<std::size_t> configured_rho_index(Real rho, const Config& config) {
    if (config.radii.empty() || !finite(rho)) return std::nullopt;
    std::size_t best = 0;
    Real distance = std::abs(rho - config.radii.front());
    for (std::size_t i = 1; i < config.radii.size(); ++i) {
        const Real candidate = std::abs(rho - config.radii[i]);
        if (candidate < distance) {
            distance = candidate;
            best = i;
        }
    }
    const Real tolerance = std::max(
        1.0e-14L,
        1024.0L * std::numeric_limits<Real>::epsilon()
            * std::max<Real>(1.0L, std::abs(config.radii[best])));
    return distance <= tolerance ? std::optional<std::size_t>(best) : std::nullopt;
}

std::vector<AreaPeriodSummaryRecord> build_period_summaries(
    const std::vector<AreaMeasurementRecord>& measurements,
    const std::vector<AreaPeriodSummaryRecord>& existing_summaries,
    const Config& config
) {
    const int period = config.period;
    std::vector<AreaSummaryBucket> buckets(config.radii.size());

    const auto aggregate_started = Clock::now();
    if (!measurements.empty()) {
        render_inline_progress(
            config,
            "  aggregating summary",
            0,
            measurements.size(),
            aggregate_started);
    }
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        const auto& row = measurements[i];
        if (row.period == period) {
            const auto rho_index = configured_rho_index(scanner_real(row.rho), config);
            if (rho_index) {
                auto& bucket = buckets[*rho_index];
                ++bucket.completed;
                if (row.converged) {
                    const Real area = scanner_real(row.area_estimate);
                    if (finite(area)) {
                        bucket.areas.push_back(area);
                        ++bucket.converged;
                        const Real error = scanner_real(row.error_estimate);
                        if (finite(error)) bucket.error_sum += error;
                    }
                }
            }
        }
        const std::size_t done = i + 1;
        if (done == measurements.size() || done % 4096 == 0) {
            render_inline_progress(
                config,
                "  aggregating summary",
                done,
                measurements.size(),
                aggregate_started,
                {},
                done == measurements.size(),
                true);
        }
    }

    std::vector<AreaPeriodSummaryRecord> summary;
    summary.reserve(config.radii.size());
    std::vector<Real> previous_cumulative(config.radii.size(), 0);
    std::vector<bool> previous_complete(config.radii.size(), period == 1);
    if (period > 1) {
        for (const auto& row : existing_summaries) {
            if (row.period != period - 1) continue;
            const auto rho_index = configured_rho_index(scanner_real(row.rho), config);
            if (!rho_index) continue;
            previous_cumulative[*rho_index] = scanner_real(row.cumulative_area);
            previous_complete[*rho_index] = row.cumulative_complete_through_period;
        }
    }

    const std::size_t total_buckets = config.radii.size();
    const auto summarize_started = Clock::now();
    std::size_t summarized = 0;
    if (total_buckets > 0) {
        render_inline_progress(
            config, "  building summary", 0, total_buckets, summarize_started);
    }

    std::optional<Real> previous_rho_area;
    for (std::size_t rho_index = 0; rho_index < config.radii.size(); ++rho_index) {
        const Real rho = config.radii[rho_index];
        auto& bucket = buckets[rho_index];
        std::sort(bucket.areas.begin(), bucket.areas.end());
        const int expected = static_cast<int>(center_count(period));
        const bool complete = bucket.converged == expected;
        const Real period_area = std::accumulate(
            bucket.areas.begin(), bucket.areas.end(), 0.0L);

        AreaPeriodSummaryRecord row;
        row.period = period;
        row.rho = catalogue_real(rho);
        row.expected_components = expected;
        row.completed_components = bucket.completed;
        row.converged_components = bucket.converged;
        row.missing_or_unconverged_components = expected - bucket.converged;
        row.period_complete = complete;
        if (!bucket.areas.empty()) {
            row.min_area = catalogue_real(bucket.areas.front());
            row.p10_area = catalogue_real(quantile_sorted(bucket.areas, 0.1L));
            row.median_area = catalogue_real(quantile_sorted(bucket.areas, 0.5L));
            row.mean_area = catalogue_real(period_area / bucket.areas.size());
            row.p90_area = catalogue_real(quantile_sorted(bucket.areas, 0.9L));
            row.max_area = catalogue_real(bucket.areas.back());
        }
        row.period_area = catalogue_real(period_area);
        row.cumulative_area = catalogue_real(
            previous_cumulative[rho_index] + period_area);
        row.cumulative_complete_through_period =
            previous_complete[rho_index] && complete;
        row.summed_error_estimate = catalogue_real(bucket.error_sum);
        if (previous_rho_area) {
            row.radial_increment_from_previous_rho = catalogue_real(
                period_area - *previous_rho_area);
        }
        previous_rho_area = period_area;
        summary.push_back(std::move(row));

        ++summarized;
        if (summarized == total_buckets || summarized % 8 == 0) {
            render_inline_progress(
                config,
                "  building summary",
                summarized,
                total_buckets,
                summarize_started,
                {},
                summarized == total_buckets,
                true);
        }
    }
    return summary;
}

void write_summary(
    const AreaScanStore& store,
    const std::vector<AreaMeasurementRecord>& measurements,
    const Config& config
) {
    const auto existing_summaries = store.load_summaries();
    auto summaries = build_period_summaries(
        measurements, existing_summaries, config);
    const auto write_started = Clock::now();
    store.save_summaries(
        config.period,
        summaries,
        make_inline_progress_callback(
            config, "  writing summary", write_started, {}, true));
}

// -----------------------------------------------------------------------------
// Parallel component-area stage
// -----------------------------------------------------------------------------

struct AreaJob {
    int period = 0;
    CenterRow center;
    int representative_job_index = 0;
    int representative_job_count = 0;
    PrecisionPlan precision;
};

void sort_measurement_rows(std::vector<AreaMeasurementRecord>& rows) {
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        if (a.period != b.period) return a.period < b.period;
        if (a.component_index != b.component_index) {
            return a.component_index < b.component_index;
        }
        return a.rho < b.rho;
    });
}

fs::path area_checkpoint_directory(const AreaScanStore& store) {
    return store.run_directory() / "area_checkpoints";
}

std::optional<std::uint64_t> area_checkpoint_batch_number(
    const fs::path& path
) {
    const std::string name = path.filename().string();
    constexpr const char* prefix = "batch_";
    constexpr const char* suffix = ".csv";
    if (name.rfind(prefix, 0) != 0
        || name.size() <= std::char_traits<char>::length(prefix)
                            + std::char_traits<char>::length(suffix)
        || name.substr(name.size() - std::char_traits<char>::length(suffix))
               != suffix) {
        return std::nullopt;
    }
    const std::string digits = name.substr(
        std::char_traits<char>::length(prefix),
        name.size() - std::char_traits<char>::length(prefix)
            - std::char_traits<char>::length(suffix));
    if (digits.empty()
        || !std::all_of(digits.begin(), digits.end(), [](unsigned char ch) {
               return std::isdigit(ch) != 0;
           })) {
        return std::nullopt;
    }
    try {
        return static_cast<std::uint64_t>(std::stoull(digits));
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<fs::path> list_area_checkpoint_batches(
    const AreaScanStore& store
) {
    std::vector<std::pair<std::uint64_t, fs::path>> numbered;
    const fs::path directory = area_checkpoint_directory(store);
    std::error_code error;
    if (!fs::is_directory(directory, error)) return {};
    for (const auto& entry : fs::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file()) continue;
        if (const auto number = area_checkpoint_batch_number(entry.path())) {
            numbered.emplace_back(*number, entry.path());
        }
    }
    std::sort(numbered.begin(), numbered.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<fs::path> result;
    result.reserve(numbered.size());
    for (auto& [_, path] : numbered) result.push_back(std::move(path));
    return result;
}

std::size_t remove_stale_area_checkpoint_temporaries(
    const AreaScanStore& store
) {
    const fs::path directory = area_checkpoint_directory(store);
    std::error_code error;
    if (!fs::is_directory(directory, error)) return 0;
    std::size_t removed = 0;
    for (const auto& entry : fs::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("batch_", 0) != 0
            || name.find(".tmp") == std::string::npos) {
            continue;
        }
        std::error_code remove_error;
        if (fs::remove(entry.path(), remove_error) && !remove_error) ++removed;
    }
    return removed;
}

void remove_area_checkpoint_batches(
    const AreaScanStore& store,
    const Config& config
) {
    const auto batches = list_area_checkpoint_batches(store);
    const auto started = Clock::now();
    if (!batches.empty()) {
        render_inline_progress(
            config, "  removing checkpoints", 0, batches.size(), started);
    }
    for (std::size_t i = 0; i < batches.size(); ++i) {
        std::error_code error;
        fs::remove(batches[i], error);
        if (error) {
            throw std::runtime_error(
                "Could not remove area checkpoint " + batches[i].string()
                + ": " + error.message());
        }
        render_inline_progress(
            config,
            "  removing checkpoints",
            i + 1,
            batches.size(),
            started,
            {},
            i + 1 == batches.size(),
            true);
    }
}

class AreaCheckpointWriter {
public:
    AreaCheckpointWriter(
        const Config& config,
        const AreaScanStore& store,
        std::atomic<int>& checkpointed_jobs,
        std::atomic<int>& checkpoint_batches
    )
        : config_(config),
          store_(store),
          checkpointed_jobs_(checkpointed_jobs),
          checkpoint_batches_(checkpoint_batches),
          directory_(area_checkpoint_directory(store)) {
        fs::create_directories(directory_);
        for (const auto& path : list_area_checkpoint_batches(store_)) {
            if (const auto number = area_checkpoint_batch_number(path)) {
                next_batch_number_ = std::max(next_batch_number_, *number + 1);
            }
        }
        last_flush_ = Clock::now();
        thread_ = std::thread([this] { run(); });
    }

    AreaCheckpointWriter(const AreaCheckpointWriter&) = delete;
    AreaCheckpointWriter& operator=(const AreaCheckpointWriter&) = delete;

    ~AreaCheckpointWriter() {
        if (thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_requested_ = true;
                flush_requested_ = true;
            }
            condition_.notify_one();
            thread_.join();
        }
    }

    void enqueue(
        std::vector<AreaMeasurementRecord> rows,
        bool durable_job
    ) {
        if (rows.empty()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_rows_.reserve(pending_rows_.size() + rows.size());
            for (auto& row : rows) pending_rows_.push_back(std::move(row));
            ++pending_trigger_jobs_;
            if (durable_job) ++pending_durable_jobs_;
        }
        condition_.notify_one();
    }

    bool failed() const noexcept { return failed_.load(); }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
            flush_requested_ = true;
        }
        condition_.notify_one();
        if (thread_.joinable()) thread_.join();
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    const Config& config_;
    const AreaScanStore& store_;
    std::atomic<int>& checkpointed_jobs_;
    std::atomic<int>& checkpoint_batches_;
    fs::path directory_;
    std::uint64_t next_batch_number_ = 1;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<AreaMeasurementRecord> pending_rows_;
    int pending_trigger_jobs_ = 0;
    int pending_durable_jobs_ = 0;
    bool stop_requested_ = false;
    bool flush_requested_ = false;
    Clock::time_point last_flush_{};
    std::thread thread_;
    std::atomic<bool> failed_{false};
    std::exception_ptr failure_;

    fs::path next_batch_path() {
        std::ostringstream name;
        name << "batch_" << std::setw(8) << std::setfill('0')
             << next_batch_number_++ << ".csv";
        return directory_ / name.str();
    }

    bool threshold_reached() const {
        return config_.area_checkpoint_components > 0
            && pending_trigger_jobs_ >= config_.area_checkpoint_components;
    }

    void run() noexcept {
        try {
            while (true) {
                std::vector<AreaMeasurementRecord> rows;
                int durable_jobs = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    const bool timed_flush_enabled =
                        config_.area_checkpoint_seconds > 0;
                    if (timed_flush_enabled) {
                        const auto deadline = last_flush_
                            + std::chrono::seconds(
                                  config_.area_checkpoint_seconds);
                        condition_.wait_until(lock, deadline, [&] {
                            return stop_requested_ || flush_requested_
                                || threshold_reached();
                        });
                    } else {
                        condition_.wait(lock, [&] {
                            return stop_requested_ || flush_requested_
                                || threshold_reached();
                        });
                    }

                    const bool time_reached = timed_flush_enabled
                        && Clock::now() - last_flush_
                            >= std::chrono::seconds(
                                   config_.area_checkpoint_seconds);
                    const bool should_flush = !pending_rows_.empty()
                        && (stop_requested_ || flush_requested_
                            || threshold_reached() || time_reached);
                    if (!should_flush) {
                        if (stop_requested_) break;
                        continue;
                    }

                    rows.swap(pending_rows_);
                    durable_jobs = pending_durable_jobs_;
                    pending_trigger_jobs_ = 0;
                    pending_durable_jobs_ = 0;
                    flush_requested_ = false;
                }

                sort_measurement_rows(rows);
                store_.save_measurements_to(next_batch_path(), rows);
                checkpointed_jobs_.fetch_add(durable_jobs);
                checkpoint_batches_.fetch_add(1);
                last_flush_ = Clock::now();

                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_requested_ && pending_rows_.empty()) break;
            }
        } catch (...) {
            failure_ = std::current_exception();
            failed_.store(true);
        }
    }
};

void consolidate_area_measurements(
    const AreaScanStore& store,
    std::vector<AreaMeasurementRecord>& measurements,
    const Config& config
) {
    const auto sort_started = Clock::now();
    render_inline_progress(
        config,
        "  sorting final rows",
        0,
        1,
        sort_started,
        std::to_string(measurements.size()) + " rows",
        false,
        false);
    sort_measurement_rows(measurements);
    render_inline_progress(
        config,
        "  sorting final rows",
        1,
        1,
        sort_started,
        std::to_string(measurements.size()) + " rows",
        true,
        false);

    const auto write_started = Clock::now();
    store.save_measurements(
        config.period,
        measurements,
        make_inline_progress_callback(
            config, "  writing final rows", write_started, {}, true));
    write_summary(store, measurements, config);

    // Checkpoint batches are deleted only after both canonical files have
    // completed their atomic replacement. If either write throws, every
    // committed batch remains available for the next resume.
    remove_area_checkpoint_batches(store, config);
}

int run_parallel_area_stage(
    const Config& config,
    const std::map<int, std::vector<CenterRow>>& centers,
    const AreaScanStore& store,
    std::vector<AreaMeasurementRecord>& measurements
) {
    const std::size_t removed_duplicates =
        canonicalize_measurement_rows(measurements, config);
    if (removed_duplicates > 0) {
        std::cout << "  canonicalized area measurements: removed "
                  << removed_duplicates
                  << " duplicate measurement record(s)\n";
    }

    std::unordered_map<std::string, AreaMeasurementRecord> original_rows;
    std::unordered_map<std::string, std::size_t> measurement_index;

    original_rows.reserve(measurements.size());
    measurement_index.reserve(measurements.size());
    const auto index_started = Clock::now();
    if (!measurements.empty()) {
        render_inline_progress(
            config, "  indexing measurements", 0, measurements.size(), index_started);
    }
    for (std::size_t index = 0; index < measurements.size(); ++index) {
        const auto& measurement = measurements[index];
        const Real rho = scanner_real(measurement.rho);
        const std::string key = measurement_key(
            measurement.period, measurement.component_index, rho);
        original_rows[key] = measurement;
        measurement_index[key] = index;
        const std::size_t done = index + 1;
        if (done == measurements.size() || done % 4096 == 0) {
            render_inline_progress(
                config,
                "  indexing measurements",
                done,
                measurements.size(),
                index_started,
                {},
                done == measurements.size(),
                true);
        }
    }

    std::vector<AreaJob> jobs;
    int cached_representatives = 0;
    int total_representatives = 0;
    std::size_t center_candidates = 0;
    for (int period = config.period_start; period <= config.period; ++period) {
        center_candidates += centers.at(period).size();
    }
    const auto planning_started = Clock::now();
    std::size_t planned_centers = 0;
    if (center_candidates > 0) {
        render_inline_progress(
            config, "  planning area jobs", 0, center_candidates, planning_started);
    }

    for (int period = config.period_start; period <= config.period; ++period) {
        const auto& period_centers = centers.at(period);
        for (const auto& center : period_centers) {
            ++planned_centers;
            if (config.use_conjugate_symmetry
                && center.component_index > center.conjugate_index) {
                if (planned_centers == center_candidates
                    || planned_centers % 1024 == 0) {
                    render_inline_progress(
                        config,
                        "  planning area jobs",
                        planned_centers,
                        center_candidates,
                        planning_started,
                        "jobs=" + std::to_string(jobs.size())
                            + " cached=" + std::to_string(cached_representatives),
                        planned_centers == center_candidates,
                        true);
                }
                continue;
            }
            ++total_representatives;
            PrecisionPlan precision = assess_component_precision(
                period, center.center, config);

            bool needs_work = !config.resume;
            if (config.resume) {
                for (Real rho : config.radii) {
                    const auto existing = original_rows.find(
                        measurement_key(period, center.component_index, rho));
                    bool cached_ok = existing != original_rows.end()
                                  && existing->second.converged;
                    if (cached_ok && config.mp_proactive && !precision.use_mp) {
                        try {
                            const Real cached_area = std::abs(
                                scanner_real(existing->second.area_estimate));
                            const Real inferred_scale = cached_area > 0
                                ? std::sqrt(cached_area / PI)
                                : 0;
                            const Real ulp = std::numeric_limits<Real>::epsilon()
                                           * std::max<Real>(1, safe_abs(center.center));
                            const Real digits = inferred_scale > 0 && ulp > 0
                                ? std::log10(inferred_scale / ulp)
                                : -std::numeric_limits<Real>::infinity();
                            if (digits < static_cast<Real>(config.mp_guard_digits)) {
                                const Real coordinate_scale = std::max<Real>(
                                    1, safe_abs(center.center));
                                const int lost_digits = inferred_scale > 0
                                    ? static_cast<int>(std::ceil(std::max<Real>(
                                        0, -std::log10(inferred_scale / coordinate_scale))))
                                    : config.mp_max_dps;
                                precision.use_mp = true;
                                precision.estimated_scale = inferred_scale;
                                precision.ulp_scale = ulp;
                                precision.digits_above_ulp = digits;
                                precision.requested_dps = std::clamp(
                                    std::max(config.mp_min_dps,
                                             lost_digits + config.mp_extra_digits),
                                    config.mp_min_dps,
                                    config.mp_max_dps);
                                precision.actual_dps = mp_precision_tier(
                                    precision.requested_dps);
                                precision.reason =
                                    "cached area implies a component scale too close to long-double ulp";
                            }
                        } catch (...) {
                        }
                    }
                    if (cached_ok && precision.use_mp) {
                        const int cached_dps = existing->second.max_mp_dps;
                        cached_ok = cached_dps >= precision.actual_dps;
                    }
                    if (!cached_ok) {
                        needs_work = true;
                        break;
                    }
                }
            }

            if (needs_work) {
                jobs.push_back(AreaJob{
                    period,
                    center,
                    0,
                    0,
                    precision,
                });
            } else {
                ++cached_representatives;
            }

            if (planned_centers == center_candidates
                || planned_centers % 1024 == 0) {
                render_inline_progress(
                    config,
                    "  planning area jobs",
                    planned_centers,
                    center_candidates,
                    planning_started,
                    "jobs=" + std::to_string(jobs.size())
                        + " cached=" + std::to_string(cached_representatives),
                    planned_centers == center_candidates,
                    true);
            }
        }
    }

    for (std::size_t i = 0; i < jobs.size(); ++i) {
        jobs[i].representative_job_index = static_cast<int>(i + 1);
        jobs[i].representative_job_count = static_cast<int>(jobs.size());
    }

    if (jobs.empty()) {
        std::cout << "\nArea stage: every representative component is already cached.\n";
        const bool staged_rows_pending =
            !list_area_checkpoint_batches(store).empty();
        if (staged_rows_pending) {
            std::cout << "  recovered checkpoint batches still need final "
                         "consolidation\n";
            consolidate_area_measurements(store, measurements, config);
        } else if (!store.has_summaries(config.period)) {
            write_summary(store, measurements, config);
        }
        return 0;
    }

    const int thread_count = static_cast<int>(std::max<unsigned>(
        1,
        std::min<unsigned>(
            config.area_threads,
            static_cast<unsigned>(jobs.size()))));

    std::cout << "\nArea stage: " << jobs.size()
              << " representative component job(s), "
              << cached_representatives << " cached, "
              << thread_count << " worker thread(s)\n";
    std::cout.flush();

    std::atomic<int> next_job{0};
    std::atomic<int> completed_jobs{0};
    std::atomic<int> failed_jobs{0};
    std::atomic<int> row_count{static_cast<int>(measurements.size())};
    std::atomic<int> checkpointed_jobs{0};
    std::atomic<int> checkpoint_batches{0};

    std::mutex measurement_mutex;
    std::mutex failure_mutex;
    std::vector<std::string> failure_messages;

    auto merge_rows = [&](const std::vector<std::pair<std::string, AreaMeasurementRecord>>& updates) {
        std::lock_guard<std::mutex> lock(measurement_mutex);
        for (const auto& [key, row] : updates) {
            const auto existing = measurement_index.find(key);
            if (existing != measurement_index.end()) {
                measurements[existing->second] = row;
            } else {
                measurement_index[key] = measurements.size();
                measurements.push_back(row);
            }
        }
        row_count.store(static_cast<int>(measurements.size()));
    };

    AreaCheckpointWriter checkpoint_writer(
        config, store, checkpointed_jobs, checkpoint_batches);

    const int proactive_mp_jobs = static_cast<int>(std::count_if(
        jobs.begin(), jobs.end(), [](const AreaJob& job) {
            return job.precision.use_mp;
        }));

    const auto area_started = Clock::now();
    g_area_progress_slots.assign(
        static_cast<std::size_t>(thread_count),
        AreaProgressSlot{});
    g_area_monitor_stop.store(false);

    std::vector<std::string> dashboard_header{
        "Mandelbrot component-area tracing",
        "components run in parallel; rho and theta stay sequential within each component",
        "jobs=" + std::to_string(jobs.size())
            + " cached=" + std::to_string(cached_representatives)
            + " representatives=" + std::to_string(total_representatives)
            + " radii=" + std::to_string(config.radii.size())
            + " threads=" + std::to_string(thread_count)
            + " proactive_MP=" + std::to_string(proactive_mp_jobs),
    };

    const bool alternate_screen =
        config.progress
        && config.progress_style == "bars"
        && config.progress_screen == "alternate"
        && stderr_is_terminal();

    if (alternate_screen) enter_alternate_screen();

    std::thread monitor;
    if (config.progress && config.progress_style == "bars") {
        monitor = std::thread(
            monitor_area_progress,
            std::cref(config),
            thread_count,
            static_cast<int>(jobs.size()),
            std::cref(completed_jobs),
            std::cref(failed_jobs),
            std::cref(row_count),
            std::cref(checkpointed_jobs),
            std::cref(checkpoint_batches),
            area_started,
            dashboard_header);
    }

    auto worker = [&](int thread_id) {
        while (true) {
            if (checkpoint_writer.failed()) break;
            const int job_number = next_job.fetch_add(1);
            if (job_number >= static_cast<int>(jobs.size())) break;

            const AreaJob& job = jobs[static_cast<std::size_t>(job_number)];
            const CenterRow& center = job.center;
            const int radius_count = static_cast<int>(config.radii.size());
            bool job_failed = false;
            std::vector<std::pair<std::string, AreaMeasurementRecord>> local_updates;
            std::optional<RingTrace> seed_trace;
            std::optional<Real> cached_seed_rho;
            int cached_seed_theta = config.theta_start;

            auto append_measurement = [&](const Measurement& measurement) {
                const std::string measurement_key_value = measurement_key(
                    measurement.period, measurement.component_index, measurement.rho);
                AreaMeasurementRecord record =
                    measurement_to_record(measurement);
                local_updates.emplace_back(measurement_key_value, record);
                std::vector<AreaMeasurementRecord> checkpoint_rows;
                checkpoint_rows.push_back(record);

                if (config.use_conjugate_symmetry
                    && center.conjugate_index != center.component_index) {
                    Measurement conjugate = measurement;
                    conjugate.component_index = center.conjugate_index;
                    conjugate.conjugate_index = center.component_index;
                    conjugate.symmetry_source_component_index = center.component_index;
                    conjugate.center = std::conj(center.center);
                    AreaMeasurementRecord conjugate_record =
                        measurement_to_record(conjugate);
                    local_updates.emplace_back(
                        measurement_key(
                            conjugate.period,
                            conjugate.component_index,
                            conjugate.rho),
                        conjugate_record);
                    checkpoint_rows.push_back(std::move(conjugate_record));
                }
                // Make each completed radius eligible for the timed durable
                // checkpoint immediately. A long component job therefore
                // does not hold all of its already-finished radii in memory.
                checkpoint_writer.enqueue(
                    std::move(checkpoint_rows),
                    false);
            };

            auto run_mp_component = [&](int requested_dps, const std::string& reason) {
                local_updates.clear();
                update_area_progress(
                    thread_id,
                    job.period,
                    job.representative_job_index,
                    job.representative_job_count,
                    center.component_index,
                    1,
                    radius_count,
                    config.radii.front(), 0, 0,
                    std::numeric_limits<Real>::quiet_NaN(),
                    "MP" + std::to_string(mp_precision_tier(requested_dps))
                        + " " + reason);

                MeasurementProgressCallback mp_progress =
                    [&](int theta, int completed, int total, const std::string& phase) {
                        const Real local = total > 0
                            ? static_cast<Real>(completed) / static_cast<Real>(total)
                            : 0;
                        update_area_progress(
                            thread_id,
                            job.period,
                            job.representative_job_index,
                            job.representative_job_count,
                            center.component_index,
                            1,
                            radius_count,
                            config.radii.front(), local, theta,
                            std::numeric_limits<Real>::quiet_NaN(),
                            "MP" + std::to_string(mp_precision_tier(requested_dps))
                                + " " + phase);
                    };

                auto measurements_mp = measure_component_mp(
                    job.period, center.component_index, center.conjugate_index,
                    center.center, config, requested_dps, mp_progress);
                job_failed = false;
                for (const auto& measurement : measurements_mp) {
                    append_measurement(measurement);
                    if (!measurement.converged) job_failed = true;
                }
            };

            update_area_progress(
                thread_id,
                job.period,
                job.representative_job_index,
                job.representative_job_count,
                center.component_index,
                1,
                radius_count,
                config.radii.front(),
                0,
                0,
                std::numeric_limits<Real>::quiet_NaN(),
                "starting");

            try {
                if (job.precision.use_mp) {
                    run_mp_component(
                        job.precision.requested_dps,
                        "proactive: " + job.precision.reason);
                } else {
                bool switch_to_mp = false;
                std::string mp_reason;
                for (int radius_index = 0;
                     radius_index < radius_count;
                     ++radius_index) {
                    const Real rho =
                        config.radii[static_cast<std::size_t>(radius_index)];
                    const std::string key = measurement_key(
                        job.period,
                        center.component_index,
                        rho);

                    if (config.resume) {
                        const auto existing = original_rows.find(key);
                        if (existing != original_rows.end()
                            && existing->second.converged) {
                            cached_seed_rho = rho;
                            cached_seed_theta = existing->second.theta_points > 0
                                ? existing->second.theta_points
                                : config.theta_start;

                            update_area_progress(
                                thread_id,
                                job.period,
                                job.representative_job_index,
                                job.representative_job_count,
                                center.component_index,
                                radius_index + 1,
                                radius_count,
                                rho,
                                static_cast<Real>(radius_index + 1)
                                    / static_cast<Real>(radius_count),
                                cached_seed_theta,
                                scanner_real(existing->second.area_estimate),
                                "cached");
                            continue;
                        }
                    }

                    const Real base_fraction =
                        static_cast<Real>(radius_index)
                        / static_cast<Real>(radius_count);

                    if (!seed_trace
                        && cached_seed_rho
                        && *cached_seed_rho < rho) {
                        update_area_progress(
                            thread_id,
                            job.period,
                            job.representative_job_index,
                            job.representative_job_count,
                            center.component_index,
                            radius_index + 1,
                            radius_count,
                            rho,
                            base_fraction,
                            cached_seed_theta,
                            std::numeric_limits<Real>::quiet_NaN(),
                            "rebuilding seed");

                        if (auto state = center_state(
                                job.period,
                                center.center,
                                config)) {
                            const int rebuild_theta =
                                std::max(8, cached_seed_theta);

                            RingProgressCallback seed_progress =
                                [&](int theta, int completed, int total,
                                    const std::string& phase) {
                                    Real local = total > 0
                                        ? static_cast<Real>(completed)
                                          / static_cast<Real>(total)
                                        : 0;
                                    if (phase == "seeding") local *= 0.20L;
                                    else if (phase == "tracing") {
                                        local = 0.20L + 0.70L * local;
                                    } else {
                                        local = 0.95L;
                                    }
                                    update_area_progress(
                                        thread_id,
                                        job.period,
                                        job.representative_job_index,
                                        job.representative_job_count,
                                        center.component_index,
                                        radius_index + 1,
                                        radius_count,
                                        rho,
                                        base_fraction
                                            + 0.15L * local
                                              / static_cast<Real>(radius_count),
                                        theta,
                                        std::numeric_limits<Real>::quiet_NaN(),
                                        "seed " + phase);
                                };

                            auto rebuilt = trace_ring(
                                job.period,
                                *state,
                                *cached_seed_rho,
                                rebuild_theta,
                                config,
                                nullptr,
                                seed_progress);
                            if (rebuilt) seed_trace = std::move(*rebuilt);
                        }
                    }

                    const auto radius_started = Clock::now();
                    RingTrace trace;

                    MeasurementProgressCallback ring_progress =
                        [&](int theta, int completed, int total,
                            const std::string& phase) {
                            Real local = total > 0
                                ? static_cast<Real>(completed)
                                  / static_cast<Real>(total)
                                : 0;
                            if (phase == "seeding") local *= 0.20L;
                            else if (phase == "tracing") {
                                local = 0.20L + 0.70L * local;
                            } else if (phase == "closing") {
                                local = 0.95L;
                            } else if (phase == "retrying") {
                                local = 0.98L;
                            } else {
                                local = 0;
                            }

                            update_area_progress(
                                thread_id,
                                job.period,
                                job.representative_job_index,
                                job.representative_job_count,
                                center.component_index,
                                radius_index + 1,
                                radius_count,
                                rho,
                                (static_cast<Real>(radius_index) + local)
                                    / static_cast<Real>(radius_count),
                                theta,
                                std::numeric_limits<Real>::quiet_NaN(),
                                phase);
                        };

                    Measurement measurement = measure_component(
                        job.period,
                        center.component_index,
                        center.conjugate_index,
                        center.center,
                        rho,
                        config,
                        seed_trace ? &*seed_trace : nullptr,
                        trace,
                        ring_progress);

                    append_measurement(measurement);

                    if (long_double_measurement_needs_mp(
                            measurement, center.center, config)) {
                        switch_to_mp = true;
                        mp_reason = measurement.converged
                            ? "measured scale approached long-double ulp"
                            : "long-double trace failed";
                        break;
                    }

                    if (measurement.converged) {
                        seed_trace = std::move(trace);
                    } else {
                        job_failed = true;
                    }

                    update_area_progress(
                        thread_id,
                        job.period,
                        job.representative_job_index,
                        job.representative_job_count,
                        center.component_index,
                        radius_index + 1,
                        radius_count,
                        rho,
                        static_cast<Real>(radius_index + 1)
                            / static_cast<Real>(radius_count),
                        measurement.theta_points,
                        measurement.area_estimate,
                        measurement.converged
                            ? "done " + format_duration(
                                  Clock::now() - radius_started)
                            : "unconverged");
                }
                if (switch_to_mp) {
                    PrecisionPlan fallback_plan = assess_component_precision(
                        job.period, center.center, config);
                    int requested_dps = fallback_plan.requested_dps > 0
                        ? fallback_plan.requested_dps
                        : config.mp_min_dps;
                    run_mp_component(requested_dps, mp_reason);
                }
                }
            } catch (const std::exception& error) {
                job_failed = true;
                std::lock_guard<std::mutex> lock(failure_mutex);
                std::ostringstream message;
                message << "period=" << job.period
                        << " component=" << center.component_index
                        << ": " << error.what();
                failure_messages.push_back(message.str());
            }

            merge_rows(local_updates);
            std::vector<AreaMeasurementRecord> checkpoint_rows;
            checkpoint_rows.reserve(local_updates.size());
            for (auto& [_, row] : local_updates) {
                checkpoint_rows.push_back(std::move(row));
            }
            checkpoint_writer.enqueue(
                std::move(checkpoint_rows),
                !job_failed);
            if (job_failed) ++failed_jobs;
            const int completed = ++completed_jobs;

            if (config.progress_style == "lines") {
                std::lock_guard<std::mutex> lock(g_print_mutex);
                std::cerr << "period=" << job.period
                          << " job=" << job.representative_job_index
                          << '/' << job.representative_job_count
                          << " component=" << center.component_index
                          << " done=" << completed << '/' << jobs.size()
                          << " failed=" << (job_failed ? "true" : "false")
                          << '\n';
            } else if (config.progress_style == "bars") {
                clear_area_progress(thread_id);
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
        workers.emplace_back(worker, thread_id);
    }
    for (auto& worker_thread : workers) worker_thread.join();

    g_area_monitor_stop.store(true);
    if (monitor.joinable()) monitor.join();
    if (alternate_screen) leave_alternate_screen();

    const auto flush_started = Clock::now();
    render_inline_progress(
        config, "  flushing area checkpoints", 0, 1, flush_started,
        "waiting for dedicated writer");
    checkpoint_writer.finish();
    render_inline_progress(
        config, "  flushing area checkpoints", 1, 1, flush_started,
        "all batches durable", true, false);
    consolidate_area_measurements(store, measurements, config);

    std::cout << "Area stage complete: done=" << completed_jobs.load()
              << '/' << jobs.size()
              << " failed=" << failed_jobs.load()
              << " elapsed=" << format_duration(Clock::now() - area_started)
              << '\n';

    if (!failure_messages.empty()) {
        std::cout << "Area worker exceptions:\n";
        for (const auto& message : failure_messages) {
            std::cout << "  " << message << '\n';
        }
    }

    return failed_jobs.load();
}

// -----------------------------------------------------------------------------
// Optional component-atlas geometry export
// -----------------------------------------------------------------------------

struct AtlasAreaRecord {
    Real area = 0;
    Real error = 0;
    Real rho = 0;
};

struct AtlasGeometryRecord {
    int period = 0;
    int index = 0;
    Complex center{};
    Real polygon_rho = 0;
    Real polygon_area = 0;
    Real area = 0;
    Real area_error = 0;
    Real area_rho = 0;
    int theta_points = 0;
    std::vector<Complex> points;
};

std::map<std::pair<int, int>, AtlasAreaRecord> select_atlas_area_rows(
    const std::vector<AreaMeasurementRecord>& rows,
    const Config& config
) {
    std::map<std::pair<int, int>, AtlasAreaRecord> chosen;
    const auto started = Clock::now();
    if (!rows.empty()) {
        render_inline_progress(
            config, "  selecting geometry areas", 0, rows.size(), started);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (row.converged
            && row.period >= std::max(1, config.period_start)
            && row.period <= config.period) {
            const Real rho = scanner_real(row.rho);
            const Real area = scanner_real(row.area_estimate);
            const Real error = scanner_real(row.error_estimate);
            if (finite(rho) && finite(area) && finite(error)
                && area >= 0 && error >= 0) {
                const auto key = std::make_pair(row.period, row.component_index);
                const auto found = chosen.find(key);
                if (found == chosen.end()
                    || std::abs(rho - config.atlas_area_rho)
                        < std::abs(found->second.rho - config.atlas_area_rho)) {
                    chosen[key] = AtlasAreaRecord{area, error, rho};
                }
            }
        }
        const std::size_t done = i + 1;
        if (done == rows.size() || done % 4096 == 0) {
            render_inline_progress(
                config,
                "  selecting geometry areas",
                done,
                rows.size(),
                started,
                "selected=" + std::to_string(chosen.size()),
                done == rows.size(),
                true);
        }
    }
    return chosen;
}

std::array<Real, 4> atlas_bbox(const std::vector<Complex>& points) {
    if (points.empty()) {
        return {0, 0, 0, 0};
    }
    Real xmin = points.front().real();
    Real xmax = xmin;
    Real ymin = points.front().imag();
    Real ymax = ymin;
    for (const auto& point : points) {
        xmin = std::min(xmin, point.real());
        xmax = std::max(xmax, point.real());
        ymin = std::min(ymin, point.imag());
        ymax = std::max(ymax, point.imag());
    }
    return {xmin, xmax, ymin, ymax};
}

std::vector<Complex> atlas_ccw_points(std::vector<Complex> points) {
    if (polygon_area(points) < 0) {
        std::reverse(points.begin(), points.end());
    }
    return points;
}

std::vector<Complex> atlas_conjugate_points(const std::vector<Complex>& source) {
    std::vector<Complex> result;
    result.reserve(source.size());
    for (auto it = source.rbegin(); it != source.rend(); ++it) {
        result.push_back(std::conj(*it));
    }
    return result;
}

std::vector<Real> atlas_trace_stages(const Config& config) {
    std::vector<Real> stages;
    for (Real candidate : {0.9L, 0.99L, 0.999L}) {
        if (candidate < config.atlas_polygon_rho) stages.push_back(candidate);
    }
    stages.push_back(config.atlas_polygon_rho);
    return stages;
}

RingTrace decimate_ring_trace(
    const RingTrace& source,
    int multiplier,
    int target_points
) {
    if (multiplier == 1) return source;
    RingTrace decimated;
    decimated.rho = source.rho;
    decimated.closure_error = source.closure_error;
    decimated.marked_z_closure_error = source.marked_z_closure_error;
    decimated.stats = source.stats;
    decimated.lambda.reserve(target_points);
    decimated.c.reserve(target_points);
    decimated.z.reserve(target_points);
    decimated.c_lambda.reserve(target_points);
    decimated.residual.reserve(target_points);
    for (int i = 0; i < target_points; ++i) {
        const std::size_t index = static_cast<std::size_t>(i * multiplier);
        decimated.lambda.push_back(source.lambda[index]);
        decimated.c.push_back(source.c[index]);
        decimated.z.push_back(source.z[index]);
        decimated.c_lambda.push_back(source.c_lambda[index]);
        decimated.residual.push_back(source.residual[index]);
    }
    return decimated;
}

std::optional<RingTrace> trace_atlas_polygon_long_double(
    const CenterRow& center,
    const Config& config
) {
    auto state = center_state(center.period, center.center, config);
    if (!state) return std::nullopt;
    const auto stages = atlas_trace_stages(config);

    for (int multiplier : {1, 2, 4}) {
        const int theta = config.atlas_polygon_points * multiplier;
        std::optional<RingTrace> seed;
        bool ok = true;
        for (Real rho : stages) {
            auto traced = trace_ring(
                center.period,
                *state,
                rho,
                theta,
                config,
                seed ? &*seed : nullptr);
            if (!traced) {
                ok = false;
                break;
            }
            seed = std::move(*traced);
        }
        if (ok && seed) {
            return decimate_ring_trace(
                *seed, multiplier, config.atlas_polygon_points);
        }
    }
    return std::nullopt;
}

template <unsigned Digits10>
std::optional<RingTrace> trace_atlas_polygon_mp_impl(
    const CenterRow& center,
    const Config& config
) {
    using MP = MpReal<Digits10>;
    using MPC = MpComplex<MP>;

    const MP tolerance = mp_pow10<MP>(-static_cast<int>(Digits10 - 15));
    const MP exact_tolerance = mp_pow10<MP>(-static_cast<int>(Digits10 / 2));
    const MP determinant_floor = mp_pow10<MP>(-static_cast<int>(Digits10 - 10));

    MPC mp_center{
        MP(real_string(center.center.real())),
        MP(real_string(center.center.imag()))};
    mp_center = mp_refine_center(
        mp_center,
        center.period,
        tolerance,
        config.mp_newton_max_iterations);

    auto state = mp_state_from_solution(
        center.period,
        MPC{MP(0), MP(0)},
        MPC{MP(0), MP(0)},
        mp_center,
        0,
        determinant_floor);
    if (!state) return std::nullopt;

    const auto stages = atlas_trace_stages(config);
    for (int multiplier : {1, 2, 4}) {
        const int theta = config.atlas_polygon_points * multiplier;
        std::optional<MpRing<MP>> seed;
        bool ok = true;
        for (Real rho_real : stages) {
            const MP rho = mp_from_real<MP>(rho_real);
            auto traced = mp_trace_ring(
                center.period,
                *state,
                rho,
                theta,
                config,
                tolerance,
                exact_tolerance,
                determinant_floor,
                seed ? &*seed : nullptr,
                {});
            if (!traced) {
                ok = false;
                break;
            }
            seed = std::move(*traced);
        }
        if (!ok || !seed) continue;

        RingTrace converted;
        converted.rho = real_from_mp(seed->rho);
        converted.closure_error = real_from_mp(seed->closure_error);
        converted.marked_z_closure_error = real_from_mp(
            seed->marked_z_closure_error);
        converted.stats = seed->stats;
        converted.lambda.reserve(config.atlas_polygon_points);
        converted.c.reserve(config.atlas_polygon_points);
        converted.z.reserve(config.atlas_polygon_points);
        converted.c_lambda.reserve(config.atlas_polygon_points);
        converted.residual.reserve(config.atlas_polygon_points);
        for (int i = 0; i < config.atlas_polygon_points; ++i) {
            const std::size_t index = static_cast<std::size_t>(i * multiplier);
            const auto& lambda = seed->lambda[index];
            const auto& c = seed->c[index];
            const auto& z = seed->z[index];
            const auto& c_lambda = seed->c_lambda[index];
            converted.lambda.emplace_back(
                real_from_mp(lambda.real()), real_from_mp(lambda.imag()));
            converted.c.emplace_back(
                real_from_mp(c.real()), real_from_mp(c.imag()));
            converted.z.emplace_back(
                real_from_mp(z.real()), real_from_mp(z.imag()));
            converted.c_lambda.emplace_back(
                real_from_mp(c_lambda.real()), real_from_mp(c_lambda.imag()));
            converted.residual.push_back(real_from_mp(seed->residual[index]));
        }
        return converted;
    }
    return std::nullopt;
}

std::optional<RingTrace> trace_atlas_polygon_mp(
    const CenterRow& center,
    const Config& config,
    int requested_dps,
    int& used_dps,
    std::string& diagnostics
) {
    const int first_tier = mp_precision_tier(
        std::max(config.mp_min_dps, requested_dps));
    std::vector<std::string> attempts;

    auto try_tier = [&](int tier, auto function) -> std::optional<RingTrace> {
        if (tier < first_tier || tier > config.mp_max_dps) return std::nullopt;
        attempts.push_back(std::to_string(tier));
        try {
            auto traced = function();
            if (traced) {
                used_dps = tier;
                diagnostics = "MP" + std::to_string(tier);
                return traced;
            }
        } catch (const std::exception& error) {
            diagnostics = "MP" + std::to_string(tier) + ": " + error.what();
        }
        return std::nullopt;
    };

    if (auto result = try_tier(50, [&] {
            return trace_atlas_polygon_mp_impl<50>(center, config);
        })) return result;
    if (auto result = try_tier(80, [&] {
            return trace_atlas_polygon_mp_impl<80>(center, config);
        })) return result;
    if (auto result = try_tier(120, [&] {
            return trace_atlas_polygon_mp_impl<120>(center, config);
        })) return result;
    if (auto result = try_tier(160, [&] {
            return trace_atlas_polygon_mp_impl<160>(center, config);
        })) return result;
    if (auto result = try_tier(200, [&] {
            return trace_atlas_polygon_mp_impl<200>(center, config);
        })) return result;

    std::ostringstream message;
    message << "MP tiers attempted";
    if (attempts.empty()) {
        message << ": none (mp_max_dps=" << config.mp_max_dps << ')';
    } else {
        message << ": ";
        for (std::size_t i = 0; i < attempts.size(); ++i) {
            if (i) message << ", ";
            message << attempts[i];
        }
    }
    if (!diagnostics.empty()) message << "; last error: " << diagnostics;
    diagnostics = message.str();
    return std::nullopt;
}

std::string legacy_exact_component_id(int period, int index) {
    std::ostringstream out;
    out << "exact-p" << std::setw(6) << std::setfill('0') << period
        << "-c" << std::setw(8) << std::setfill('0') << index;
    return out.str();
}

std::string exact_component_id(int period, int index) {
    return mandelbrot::catalogue::Catalogue::stable_id(
        "exact-period:" + std::to_string(period) + ":"
        + std::to_string(index));
}

std::optional<int> exact_component_index_from_legacy_id(
    const std::string& id,
    int expected_period
) {
    const std::string prefix = "exact-p";
    const auto component_marker = id.find("-c");
    if (id.rfind(prefix, 0) != 0 || component_marker == std::string::npos) {
        return std::nullopt;
    }
    try {
        const int period = std::stoi(
            id.substr(prefix.size(), component_marker - prefix.size()));
        if (period != expected_period) return std::nullopt;
        return std::stoi(id.substr(component_marker + 2));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> exact_component_index_from_record(
    const mandelbrot::catalogue::ComponentRecord& component
) {
    if (const auto identity =
            mandelbrot::catalogue::Catalogue::exact_period_index(component)) {
        return identity->component_index;
    }
    return exact_component_index_from_legacy_id(component.id, component.period);
}

bool is_canonical_exact_representative(
    const CenterRow& center,
    Real tolerance
) {
    if (center.center.imag() > tolerance) return true;
    if (center.center.imag() < -tolerance) return false;
    return center.component_index <= center.conjugate_index;
}

std::optional<mandelbrot::catalogue::ComponentRecord>
load_or_migrate_exact_component(
    mandelbrot::catalogue::Catalogue& catalogue,
    const CenterRow& center,
    const Config& config,
    bool* catalogue_changed = nullptr
) {
    using namespace mandelbrot::catalogue;
    const std::string canonical_id = exact_component_id(
        center.period, center.component_index);
    std::vector<std::string> candidates{canonical_id};
    candidates.push_back(legacy_exact_component_id(
        center.period, center.component_index));
    if (center.conjugate_index != center.component_index) {
        candidates.push_back(legacy_exact_component_id(
            center.period, center.conjugate_index));
    }

    for (const auto& candidate_id : candidates) {
        if (!catalogue.component_exists(candidate_id)) continue;
        ComponentRecord component = catalogue.load_component(candidate_id);
        bool changed = candidate_id != canonical_id;

        if (component.id != canonical_id) {
            component.id = canonical_id;
            changed = true;
        }
        const auto aliases_before = component.provenance.aliases;
        Catalogue::set_exact_period_index(
            component, center.period, center.component_index);
        if (component.provenance.aliases != aliases_before) changed = true;
        const std::string legacy_alias = "legacy-id:" + candidate_id;
        if (candidate_id != canonical_id
            && std::find(component.provenance.aliases.begin(),
                         component.provenance.aliases.end(),
                         legacy_alias)
                   == component.provenance.aliases.end()) {
            component.provenance.aliases.push_back(legacy_alias);
            changed = true;
        }

        Real canonical_imag = center.center.imag();
        if (std::abs(canonical_imag) <= config.center_duplicate_tolerance) {
            canonical_imag = 0;
        }
        const ComplexValue canonical_center{
            CatalogueReal(real_string(center.center.real())),
            CatalogueReal(real_string(canonical_imag))};
        if (component.center.re != canonical_center.re
            || component.center.im != canonical_center.im) {
            component.center = canonical_center;
            changed = true;
        }

        if (changed) {
            catalogue.save_component(component, false);
            if (catalogue_changed) *catalogue_changed = true;
        }
        for (const auto& obsolete_id : candidates) {
            if (obsolete_id != canonical_id
                && catalogue.component_exists(obsolete_id)) {
                catalogue.delete_component(obsolete_id, false);
                if (catalogue_changed) *catalogue_changed = true;
            }
        }
        return component;
    }
    return std::nullopt;
}

bool catalogue_has_current_atlas_geometry(
    mandelbrot::catalogue::Catalogue& catalogue,
    const CenterRow& center,
    Real area,
    const Config& config,
    bool* catalogue_changed = nullptr
) {
    using mandelbrot::catalogue::CatalogueReal;
    try {
        const auto loaded = load_or_migrate_exact_component(
            catalogue, center, config, catalogue_changed);
        if (!loaded) return false;
        const auto& component = *loaded;
        if (!component.quality.center_validated
            || !component.quality.exact_period_validated
            || !component.quality.polygon_converged
            || component.geometry.polygon.size()
                   != static_cast<std::size_t>(config.atlas_polygon_points)) {
            return false;
        }
        const Real stored_rho = component.geometry.polygon_rho.convert_to<Real>();
        const Real stored_area = component.geometry.area_estimate.convert_to<Real>();
        const Real rho_tolerance = 64 * std::numeric_limits<Real>::epsilon()
            * std::max<Real>(1, std::abs(config.atlas_polygon_rho));
        const Real area_tolerance = std::max(
            config.area_atol,
            config.area_rtol * std::max(std::abs(area), std::abs(stored_area)));
        const Real stored_area_rho = component.geometry.area_rho.convert_to<Real>();
        return std::abs(stored_rho - config.atlas_polygon_rho) <= rho_tolerance
            && std::abs(stored_area - area) <= area_tolerance
            && std::abs(stored_area_rho - config.atlas_area_rho)
                   <= rho_tolerance;
    } catch (...) {
        return false;
    }
}

bool catalogue_period_has_current_exact_geometry(
    mandelbrot::catalogue::Catalogue& catalogue,
    const std::vector<CenterRow>& period_centers,
    const Config& config,
    bool& trusted_legacy_index
) {
    using mandelbrot::catalogue::CatalogueReal;
    trusted_legacy_index = false;
    if (!config.resume || config.verify_catalogue
        || !catalogue.period_exists(config.period)) {
        return false;
    }

    try {
        const auto index = catalogue.load_period(config.period);
        std::vector<std::string> expected_ids;
        std::size_t expected_with_symmetry = 0;
        const auto started = Clock::now();
        if (!period_centers.empty()) {
            render_inline_progress(
                config,
                "  validating period index",
                0,
                period_centers.size(),
                started);
        }
        for (std::size_t i = 0; i < period_centers.size(); ++i) {
            const auto& center = period_centers[i];
            if (!is_canonical_exact_representative(
                    center, config.center_duplicate_tolerance)) {
                const std::size_t done = i + 1;
                if (done == period_centers.size() || done % 2048 == 0) {
                    render_inline_progress(
                        config,
                        "  validating period index",
                        done,
                        period_centers.size(),
                        started,
                        "representatives=" + std::to_string(expected_ids.size()),
                        done == period_centers.size(),
                        true);
                }
                continue;
            }
            expected_ids.push_back(exact_component_id(
                center.period, center.component_index));
            expected_with_symmetry +=
                center.conjugate_index == center.component_index ? 1 : 2;
            const std::size_t done = i + 1;
            if (done == period_centers.size() || done % 2048 == 0) {
                render_inline_progress(
                    config,
                    "  validating period index",
                    done,
                    period_centers.size(),
                    started,
                    "representatives=" + std::to_string(expected_ids.size()),
                    done == period_centers.size(),
                    true);
            }
        }
        const auto compare_started = Clock::now();
        render_inline_progress(
            config, "  comparing period index", 0, 2, compare_started,
            std::to_string(expected_ids.size()) + " component IDs");
        std::sort(expected_ids.begin(), expected_ids.end());

        auto stored_ids = index.component_ids;
        std::sort(stored_ids.begin(), stored_ids.end());
        render_inline_progress(
            config, "  comparing period index", 1, 2, compare_started,
            std::to_string(expected_ids.size()) + " component IDs");
        const bool index_matches = index.catalogue_complete
            && index.known_representative_count == expected_ids.size()
            && index.known_component_count_with_symmetry == expected_with_symmetry
            && stored_ids == expected_ids;
        render_inline_progress(
            config, "  comparing period index", 2, 2, compare_started,
            index_matches ? "match" : "mismatch", true, false);
        if (!index_matches) {
            return false;
        }

        if (!index.exact_geometry_complete) {
            // v2 period indexes were written only after all component files
            // had been atomically saved. Treat that completed index as the
            // transaction marker and upgrade it to v3 without reopening every
            // large polygon JSON. --verify-catalogue opts into the deep path.
            trusted_legacy_index = true;
            return true;
        }

        const CatalogueReal rho_tolerance("1e-18");
        return boost::multiprecision::abs(
                   index.polygon_rho
                   - CatalogueReal(real_string(config.atlas_polygon_rho)))
                   <= rho_tolerance
            && boost::multiprecision::abs(
                   index.area_rho
                   - CatalogueReal(real_string(config.atlas_area_rho)))
                   <= rho_tolerance
            && index.polygon_points
                   == static_cast<std::size_t>(config.atlas_polygon_points);
    } catch (...) {
        return false;
    }
}

const CenterRow* find_center_row(
    const std::map<int, std::vector<CenterRow>>& centers,
    int period,
    int component_index
) {
    const auto period_it = centers.find(period);
    if (period_it == centers.end()) return nullptr;
    for (const auto& center : period_it->second) {
        if (center.component_index == component_index) return &center;
    }
    return nullptr;
}


struct CatalogueWriteStats {
    std::size_t written_representatives = 0;
    std::size_t created_representatives = 0;
    std::size_t created_components_with_symmetry = 0;
};

CatalogueWriteStats save_atlas_records_to_catalogue(
    const std::vector<AtlasGeometryRecord>& records,
    const Config& config,
    mandelbrot::catalogue::Catalogue& catalogue,
    bool show_progress
) {
    using namespace mandelbrot::catalogue;
    CatalogueWriteStats stats;
    const std::size_t representative_total = static_cast<std::size_t>(
        std::count_if(records.begin(), records.end(), [&](const auto& record) {
            return record.center.imag() >= -config.center_duplicate_tolerance;
        }));
    const auto write_started = Clock::now();
    std::size_t representative_done = 0;
    const std::size_t progress_stride = std::max<std::size_t>(
        1, representative_total / 200);
    constexpr std::size_t catalogue_batch_size = 1000;
    std::vector<ComponentRecord> pending_components;
    pending_components.reserve(catalogue_batch_size);
    auto flush_components = [&] {
        if (pending_components.empty()) return;
        catalogue.save_components(pending_components, false);
        pending_components.clear();
    };
    if (show_progress && representative_total > 0
        && config.progress && config.progress_style == "bars") {
        render_inline_progress(
            config,
            "  catalogue write",
            0,
            representative_total,
            write_started,
            "SQLite component rows",
            false,
            true);
    }
    for (const auto& record : records) {
        if (record.center.imag() < -config.center_duplicate_tolerance) continue;

        const std::string id = exact_component_id(record.period, record.index);
        const bool existed = catalogue.component_exists(id);
        ComponentRecord component;
        if (existed) {
            component = catalogue.load_component(id);
        } else {
            component.id = id;
            component.period = record.period;
        }

        component.id = id;
        component.period = record.period;
        const Real center_imag =
            std::abs(record.center.imag()) <= config.center_duplicate_tolerance
                ? 0
                : record.center.imag();
        component.center = {
            CatalogueReal(real_string(record.center.real())),
            CatalogueReal(real_string(center_imag))};
        component.numeric.working_precision_digits =
            std::numeric_limits<Real>::max_digits10;
        component.numeric.validated_digits = std::max(
            0, std::numeric_limits<Real>::digits10 - 2);
        component.geometry.polygon_rho =
            CatalogueReal(real_string(record.polygon_rho));
        component.geometry.polygon_area =
            CatalogueReal(real_string(record.polygon_area));
        component.geometry.area_estimate =
            CatalogueReal(real_string(record.area));
        component.geometry.area_error =
            CatalogueReal(real_string(record.area_error));
        component.geometry.area_rho =
            CatalogueReal(real_string(record.area_rho));
        component.geometry.characteristic_size =
            boost::multiprecision::sqrt(
                component.geometry.area_estimate / CatalogueReal(PI));
        const auto bounds = atlas_bbox(record.points);
        component.geometry.bbox_centered = {
            CatalogueReal(real_string(bounds[0] - record.center.real())),
            CatalogueReal(real_string(bounds[1] - record.center.real())),
            CatalogueReal(real_string(bounds[2] - center_imag)),
            CatalogueReal(real_string(bounds[3] - center_imag))};
        component.geometry.polygon.clear();
        component.geometry.polygon.reserve(record.points.size());
        for (const auto& point : record.points) {
            component.geometry.polygon.push_back({
                CatalogueReal(real_string(
                    point.real() - record.center.real())),
                CatalogueReal(real_string(point.imag() - center_imag))});
        }
        if (component.classification.shape_class.empty()) {
            component.classification.shape_class = "unknown";
        }
        component.provenance.method = "exact-period-area-scan";
        component.provenance.run_id = config.run_dir.filename().string();
        if (component.provenance.discovered_at.empty()) {
            component.provenance.discovered_at = utc_timestamp();
        }
        Catalogue::set_exact_period_index(
            component, record.period, record.index);
        component.quality.center_validated = true;
        component.quality.exact_period_validated = true;
        component.quality.polygon_converged = true;
        // The exact-period scanner has no scientific area cutoff. Every exact
        // component it solves belongs in the canonical catalogue. The
        // demo_min_area threshold is applied only to the disposable demo export.
        component.quality.area_above_cutoff = true;
        pending_components.push_back(std::move(component));
        if (pending_components.size() >= catalogue_batch_size) {
            flush_components();
        }
        ++stats.written_representatives;
        if (!existed) {
            ++stats.created_representatives;
            stats.created_components_with_symmetry +=
                center_imag == 0 ? 1 : 2;
        }
        ++representative_done;
        if (representative_done == representative_total
            || representative_done % progress_stride == 0) {
            std::ostringstream detail;
            detail << "SQLite component rows"
                   << " | created=" << stats.created_representatives;
            if (show_progress
                && config.progress && config.progress_style == "bars") {
                render_inline_progress(
                    config,
                    "  catalogue write",
                    representative_done,
                    representative_total,
                    write_started,
                    detail.str(),
                    representative_done == representative_total,
                    true);
            } else if (show_progress && config.progress_style == "lines"
                       && (representative_done == representative_total
                           || representative_done % std::max<std::size_t>(
                               1, representative_total / 20) == 0)) {
                std::cout << "  catalogue write " << representative_done
                          << '/' << representative_total << " | "
                          << detail.str() << " | elapsed "
                          << format_duration(Clock::now() - write_started)
                          << '\n';
            }
        }
    }
    flush_components();
    return stats;
}

void merge_catalogue_write_stats(
    CatalogueWriteStats& total,
    const CatalogueWriteStats& batch
) {
    total.written_representatives += batch.written_representatives;
    total.created_representatives += batch.created_representatives;
    total.created_components_with_symmetry +=
        batch.created_components_with_symmetry;
}

// Geometry traces are expensive and arrive at an uneven rate. Keep SQLite I/O
// off the tracing workers: they hand completed representative/conjugate groups
// to this single writer, which upserts atomic transactions by count or time.
// A crash can therefore lose only the pending queue; every completed transaction
// is detected by the normal geometry-cache scan on the next resume.
class GeometryCheckpointWriter {
public:
    GeometryCheckpointWriter(
        const Config& config,
        std::atomic<std::size_t>& checkpointed_jobs,
        std::atomic<std::size_t>& checkpoint_batches
    )
        : config_(config),
          checkpointed_jobs_(checkpointed_jobs),
          checkpoint_batches_(checkpoint_batches) {
        // Start only after every member (including failure state and stats) is
        // fully constructed; the writer may run immediately on another core.
        last_flush_ = Clock::now();
        thread_ = std::thread([this] { run(); });
    }

    GeometryCheckpointWriter(const GeometryCheckpointWriter&) = delete;
    GeometryCheckpointWriter& operator=(const GeometryCheckpointWriter&) = delete;

    ~GeometryCheckpointWriter() {
        if (thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_requested_ = true;
                flush_requested_ = true;
            }
            condition_.notify_one();
            thread_.join();
        }
    }

    void enqueue(std::vector<AtlasGeometryRecord> records) {
        if (records.empty()) return;
        if (failed_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Geometry checkpoint writer has failed");
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_records_.reserve(
                pending_records_.size() + records.size());
            for (auto& record : records) {
                pending_records_.push_back(std::move(record));
            }
            ++pending_jobs_;
        }
        condition_.notify_one();
    }

    bool failed() const noexcept {
        return failed_.load(std::memory_order_acquire);
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
            flush_requested_ = true;
        }
        condition_.notify_one();
        if (thread_.joinable()) thread_.join();
        if (failure_) std::rethrow_exception(failure_);
    }

    CatalogueWriteStats stats() const noexcept { return stats_; }

private:
    const Config& config_;
    std::atomic<std::size_t>& checkpointed_jobs_;
    std::atomic<std::size_t>& checkpoint_batches_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<AtlasGeometryRecord> pending_records_;
    std::size_t pending_jobs_ = 0;
    bool stop_requested_ = false;
    bool flush_requested_ = false;
    Clock::time_point last_flush_{};
    std::thread thread_;
    std::atomic<bool> failed_{false};
    std::exception_ptr failure_;
    CatalogueWriteStats stats_;

    bool threshold_reached() const {
        return pending_jobs_
            >= static_cast<std::size_t>(
                   config_.geometry_checkpoint_components);
    }

    void run() noexcept {
        try {
            mandelbrot::catalogue::Catalogue catalogue(
                config_.catalogue_root);
            catalogue.ensure_layout();
            while (true) {
                std::vector<AtlasGeometryRecord> records;
                std::size_t jobs = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    const auto interval = std::chrono::seconds(
                        config_.geometry_checkpoint_seconds);
                    condition_.wait_until(
                        lock,
                        last_flush_ + interval,
                        [&] {
                            return stop_requested_ || flush_requested_
                                || threshold_reached();
                        });

                    const bool time_reached =
                        Clock::now() - last_flush_ >= interval;
                    const bool should_flush = !pending_records_.empty()
                        && (stop_requested_ || flush_requested_
                            || threshold_reached() || time_reached);
                    if (!should_flush) {
                        if (stop_requested_) break;
                        continue;
                    }

                    records.swap(pending_records_);
                    jobs = pending_jobs_;
                    pending_jobs_ = 0;
                    flush_requested_ = false;
                }

                const CatalogueWriteStats batch =
                    save_atlas_records_to_catalogue(
                        records, config_, catalogue, false);
                merge_catalogue_write_stats(stats_, batch);
                checkpointed_jobs_.fetch_add(
                    jobs, std::memory_order_relaxed);
                checkpoint_batches_.fetch_add(
                    1, std::memory_order_relaxed);
                last_flush_ = Clock::now();

                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_requested_ && pending_records_.empty()) break;
            }
        } catch (...) {
            failure_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    }
};

std::string big_integer_string(const boost::multiprecision::cpp_int& value) {
    std::ostringstream output;
    output << value;
    return output.str();
}

void finalize_exact_catalogue_period(
    const Config& config,
    const std::vector<CenterRow>& period_centers,
    const std::map<std::pair<int, int>, AtlasAreaRecord>& areas,
    const CatalogueWriteStats& writes,
    bool catalogue_changed
) {
    using namespace mandelbrot::catalogue;
    Catalogue catalogue(config.catalogue_root);
    catalogue.ensure_layout();

    Manifest manifest = catalogue.load_manifest();
    if (catalogue_changed) {
        ++manifest.catalogue_revision;
        manifest.updated_at = mandelbrot::catalogue::utc_timestamp();
    }

    PeriodRecord period;
    period.period = config.period;
    period.theoretical_component_count = big_integer_string(
        center_count_big(config.period));
    period.area_cutoff = 0;
    period.exact_geometry_complete = true;
    period.polygon_rho = CatalogueReal(real_string(config.atlas_polygon_rho));
    period.area_rho = CatalogueReal(real_string(config.atlas_area_rho));
    period.polygon_points = static_cast<std::size_t>(config.atlas_polygon_points);
    period.generated_from_catalogue_revision = manifest.catalogue_revision;

    const auto index_started = Clock::now();
    if (!period_centers.empty()) {
        render_inline_progress(
            config,
            "  building period index",
            0,
            period_centers.size(),
            index_started);
    }
    for (std::size_t i = 0; i < period_centers.size(); ++i) {
        const auto& center = period_centers[i];
        if (!is_canonical_exact_representative(
                center, config.center_duplicate_tolerance)) {
        } else {
            const auto area_it = areas.find({config.period, center.component_index});
            if (area_it == areas.end()) {
                throw std::runtime_error(
                    "Cannot finalize canonical period "
                    + std::to_string(config.period)
                    + ": missing area row for representative component "
                    + std::to_string(center.component_index) + '.');
            }

            const std::string id = exact_component_id(
                config.period, center.component_index);
            period.component_ids.push_back(id);
            ++period.known_representative_count;
            const std::size_t multiplicity =
                center.conjugate_index == center.component_index ? 1 : 2;
            period.known_component_count_with_symmetry += multiplicity;
            period.known_area += CatalogueReal(real_string(area_it->second.area))
                * multiplicity;
            period.known_area_error += CatalogueReal(
                real_string(area_it->second.error)) * multiplicity;
        }
        const std::size_t done = i + 1;
        if (done == period_centers.size() || done % 2048 == 0) {
            render_inline_progress(
                config,
                "  building period index",
                done,
                period_centers.size(),
                index_started,
                "representatives="
                    + std::to_string(period.known_representative_count),
                done == period_centers.size(),
                true);
        }
    }

    const auto id_sort_started = Clock::now();
    render_inline_progress(
        config, "  sorting period IDs", 0, 1, id_sort_started,
        std::to_string(period.component_ids.size()) + " component IDs");
    std::sort(period.component_ids.begin(), period.component_ids.end());
    render_inline_progress(
        config, "  sorting period IDs", 1, 1, id_sort_started,
        std::to_string(period.component_ids.size()) + " component IDs",
        true, false);
    const std::size_t expected = center_count(config.period);
    period.catalogue_complete =
        period.known_component_count_with_symmetry == expected;
    const auto period_write_started = Clock::now();
    render_inline_progress(
        config, "  writing period index", 0, 1, period_write_started,
        std::to_string(period.component_ids.size()) + " component IDs");
    catalogue.save_period(period);
    render_inline_progress(
        config, "  writing period index", 1, 1, period_write_started,
        std::to_string(period.component_ids.size()) + " component IDs",
        true, false);

    if (!period.catalogue_complete) {
        throw std::runtime_error(
            "Canonical period " + std::to_string(config.period)
            + " contains "
            + std::to_string(period.known_component_count_with_symmetry)
            + " component(s) with symmetry; expected "
            + std::to_string(expected) + '.');
    }

    // Manifest counts are rebuilt from tiny period indexes, never by reopening
    // every polygon JSON. This remains O(number of periods), not O(components).
    manifest.component_count_stored = 0;
    manifest.component_count_with_symmetry = 0;
    manifest.exact_through_period = 0;
    const std::vector<int> period_numbers = catalogue.list_periods();
    const auto manifest_started = Clock::now();
    if (!period_numbers.empty()) {
        render_inline_progress(
            config, "  rebuilding manifest", 0, period_numbers.size(),
            manifest_started);
    }
    for (std::size_t i = 0; i < period_numbers.size(); ++i) {
        const int indexed_period = period_numbers[i];
        try {
            const PeriodRecord indexed = catalogue.load_period(indexed_period);
            manifest.component_count_stored += indexed.known_representative_count;
            manifest.component_count_with_symmetry +=
                indexed.known_component_count_with_symmetry;
        } catch (...) {
        }
        render_inline_progress(
            config,
            "  rebuilding manifest",
            i + 1,
            period_numbers.size(),
            manifest_started,
            {},
            i + 1 == period_numbers.size(),
            true);
    }
    const std::set<int> indexed_periods(period_numbers.begin(), period_numbers.end());
    for (int candidate = 1; candidate <= config.period; ++candidate) {
        if (!indexed_periods.contains(candidate)) break;
        try {
            const PeriodRecord indexed = catalogue.load_period(candidate);
            if (!indexed.catalogue_complete
                || !indexed.exact_geometry_complete) {
                break;
            }
            manifest.exact_through_period = candidate;
        } catch (...) {
            break;
        }
    }
    manifest.minimum_area = 0;
    if (catalogue_changed
        || manifest.exact_through_period >= config.period
        || writes.created_representatives > 0) {
        manifest.updated_at = mandelbrot::catalogue::utc_timestamp();
    }
    const auto manifest_write_started = Clock::now();
    render_inline_progress(
        config, "  writing manifest", 0, 1, manifest_write_started);
    catalogue.save_manifest(manifest);
    render_inline_progress(
        config, "  writing manifest", 1, 1, manifest_write_started,
        {}, true, false);

    std::cout << "Canonical period " << config.period << ": "
              << period.known_representative_count
              << " stored representative UUID file(s), "
              << period.known_component_count_with_symmetry
              << " component(s) with conjugate symmetry expanded\n";
}


bool export_atlas_geometry(
    const Config& config,
    const std::map<int, std::vector<CenterRow>>& centers,
    const AreaScanStore& store,
    const std::vector<AreaMeasurementRecord>& measurements
) {
    using mandelbrot::catalogue::Catalogue;

    Catalogue catalogue(config.catalogue_root);
    catalogue.ensure_layout();

    struct AtlasJob {
        CenterRow center;
        Real area = 0;
        Real area_error = 0;
        PrecisionPlan precision;
    };
    std::vector<AtlasJob> jobs;
    std::size_t selected_components = 0;
    std::size_t selected_representatives = 0;
    std::size_t cached_representatives = 0;
    std::size_t missing_area_rows = 0;
    bool catalogue_changed = false;
    bool trusted_legacy_period_index = false;
    bool period_index_cache_hit = false;

    const auto target_centers_it = centers.find(config.period);
    if (config.period_start == config.period
        && target_centers_it != centers.end()) {
        period_index_cache_hit = catalogue_period_has_current_exact_geometry(
            catalogue,
            target_centers_it->second,
            config,
            trusted_legacy_period_index);
    }
    if (period_index_cache_hit && !trusted_legacy_period_index) {
        std::cout << "period " << config.period
                  << ": canonical geometry already complete; "
                     "using cached period index\n";
        return false;
    }

    std::vector<AreaMeasurementRecord> geometry_measurements;
    const std::vector<AreaMeasurementRecord>* area_source = &measurements;
    if (measurements.empty()) {
        const auto load_started = Clock::now();
        geometry_measurements = store.load_measurements(
            config.period,
            catalogue_real(config.atlas_area_rho),
            make_inline_progress_callback(
                config,
                "  loading geometry areas p" + std::to_string(config.period),
                load_started,
                {},
                true));
        const std::size_t expected_rows = target_centers_it == centers.end()
            ? 0
            : target_centers_it->second.size();
        if (geometry_measurements.size() < expected_rows) {
            // Older imports may have used a different textual spelling for the
            // same rho. Fall back to this period only; never load run history.
            const auto fallback_started = Clock::now();
            geometry_measurements = store.load_measurements(
                config.period,
                make_inline_progress_callback(
                    config,
                    "  loading area fallback p" + std::to_string(config.period),
                    fallback_started,
                    {},
                    true));
        }
        area_source = &geometry_measurements;
    }
    const auto areas = select_atlas_area_rows(*area_source, config);

    std::size_t geometry_candidates = 0;
    for (int period = std::max(1, config.period_start);
         period <= config.period;
         ++period) {
        if (const auto it = centers.find(period); it != centers.end()) {
            geometry_candidates += it->second.size();
        }
    }
    const auto preflight_started = Clock::now();
    std::size_t geometry_checked = 0;
    if (geometry_candidates > 0) {
        render_inline_progress(
            config,
            "  scanning geometry cache",
            0,
            geometry_candidates,
            preflight_started);
    }

    for (int period = std::max(1, config.period_start);
         period <= config.period;
         ++period) {
        const auto centers_it = centers.find(period);
        if (centers_it == centers.end()) continue;
        for (const auto& center : centers_it->second) {
            ++geometry_checked;
            const auto area_it = areas.find({period, center.component_index});
            if (area_it == areas.end()) {
                ++missing_area_rows;
            } else {
                ++selected_components;
                if (is_canonical_exact_representative(
                        center, config.center_duplicate_tolerance)) {
                    ++selected_representatives;
                    bool cached = period_index_cache_hit;
                    if (!cached && config.resume) {
                        cached = catalogue_has_current_atlas_geometry(
                            catalogue,
                            center,
                            area_it->second.area,
                            config,
                            &catalogue_changed);
                    }
                    if (cached) {
                        ++cached_representatives;
                        // A partial checkpoint from an interrupted run is a
                        // real catalogue change even though this invocation
                        // did not write it. Finalization must still advance
                        // the manifest revision after all geometry is present.
                        if (!period_index_cache_hit) catalogue_changed = true;
                    } else {
                        jobs.push_back(AtlasJob{
                            center,
                            area_it->second.area,
                            area_it->second.error,
                            assess_component_precision(
                                period, center.center, config)});
                    }
                }
            }

            if (geometry_checked == geometry_candidates
                || geometry_checked % 64 == 0) {
                render_inline_progress(
                    config,
                    "  scanning geometry cache",
                    geometry_checked,
                    geometry_candidates,
                    preflight_started,
                    "cached=" + std::to_string(cached_representatives)
                        + " jobs=" + std::to_string(jobs.size()),
                    geometry_checked == geometry_candidates,
                    true);
            }
        }
    }

    if (missing_area_rows > 0) {
        throw std::runtime_error(
            "Atlas geometry export found " + std::to_string(missing_area_rows)
            + " component(s) without a converged area row near atlas_area_rho.");
    }

    const std::size_t proactive_mp_jobs = static_cast<std::size_t>(
        std::count_if(jobs.begin(), jobs.end(), [](const AtlasJob& job) {
            return job.precision.use_mp;
        }));

    std::cout << "\nCanonical exact geometry for period range "
              << config.period_start << ".." << config.period << ": selected "
              << selected_components << " component(s), "
              << selected_representatives << " representative(s), "
              << cached_representatives << " cached, "
              << jobs.size() << " trace job(s), "
              << proactive_mp_jobs << " proactive MP, "
              << "no area cutoff, rho="
              << real_string(config.atlas_polygon_rho, 6)
              << ", points=" << config.atlas_polygon_points << '\n';
    std::cout << "  geometry checkpoints: atomic SQLite batches every "
              << config.geometry_checkpoint_components << " job(s) or "
              << config.geometry_checkpoint_seconds << "s\n";
    if (period_index_cache_hit) {
        std::cout << "  period index cache hit: trusted "
                  << cached_representatives
                  << " representative component file(s) without reopening "
                     "their polygon JSON";
        if (trusted_legacy_period_index) {
            std::cout << " (upgrading legacy period index)";
        }
        std::cout << "\n";
    } else if (config.verify_catalogue) {
        std::cout << "  deep catalogue verification enabled\n";
    }

    std::atomic<std::size_t> completed{0};
    std::atomic<std::size_t> failed{0};
    std::atomic<std::size_t> mp_successes{0};
    std::atomic<std::size_t> checkpointed_jobs{0};
    std::atomic<std::size_t> checkpoint_batches{0};
    std::atomic<std::size_t> produced_records{0};
    std::mutex errors_mutex;
    std::vector<std::string> errors;
    const auto started = Clock::now();
    GeometryCheckpointWriter checkpoint_writer(
        config, checkpointed_jobs, checkpoint_batches);

    if (!jobs.empty()) {
        if (config.progress && config.progress_style == "bars") {
            render_inline_progress(
                config,
                "  atlas geometry",
                0,
                jobs.size(),
                started,
                "failed=0",
                false,
                true);
        }
        // Geometry traces have highly variable cost: some finish in long
        // double, while others need multiprecision tiers and denser retries.
        // Claim one job at a time so every free worker helps until only the
        // final in-flight traces remain.
        parallel_for_dynamic(
            jobs.size(),
            config.area_threads,
            1,
            [&](std::size_t job_index) {
                if (checkpoint_writer.failed()) return;
                const auto& job = jobs[job_index];
                try {
                    std::optional<RingTrace> trace;
                    int used_mp_dps = 0;
                    std::string mp_diagnostics;

                    if (job.precision.use_mp) {
                        trace = trace_atlas_polygon_mp(
                            job.center,
                            config,
                            job.precision.requested_dps,
                            used_mp_dps,
                            mp_diagnostics);
                    } else {
                        trace = trace_atlas_polygon_long_double(
                            job.center, config);
                        if (!trace && config.mp_fallback) {
                            trace = trace_atlas_polygon_mp(
                                job.center,
                                config,
                                job.precision.requested_dps,
                                used_mp_dps,
                                mp_diagnostics);
                        }
                    }

                    if (!trace) {
                        std::ostringstream reason;
                        reason << "ring tracing failed";
                        if (config.mp_fallback) {
                            reason << " after long-double/MP fallback";
                            if (!mp_diagnostics.empty()) {
                                reason << " (" << mp_diagnostics << ')';
                            }
                        }
                        throw std::runtime_error(reason.str());
                    }
                    if (used_mp_dps > 0) ++mp_successes;
                    auto points = atlas_ccw_points(std::move(trace->c));
                    AtlasGeometryRecord original;
                    original.period = job.center.period;
                    original.index = job.center.component_index;
                    original.center = job.center.center;
                    original.polygon_rho = config.atlas_polygon_rho;
                    original.polygon_area = std::abs(polygon_area(points));
                    original.area = job.area;
                    original.area_error = job.area_error;
                    original.area_rho = config.atlas_area_rho;
                    original.theta_points = config.atlas_polygon_points;
                    original.points = points;
                    std::vector<AtlasGeometryRecord> records;
                    records.reserve(2);
                    records.push_back(std::move(original));

                    if (job.center.conjugate_index
                        != job.center.component_index) {
                        AtlasGeometryRecord conjugate;
                        conjugate.period = job.center.period;
                        conjugate.index = job.center.conjugate_index;
                        conjugate.center = std::conj(job.center.center);
                        conjugate.polygon_rho = config.atlas_polygon_rho;
                        conjugate.points = atlas_conjugate_points(points);
                        conjugate.polygon_area = std::abs(
                            polygon_area(conjugate.points));
                        conjugate.area = job.area;
                        conjugate.area_error = job.area_error;
                        conjugate.area_rho = config.atlas_area_rho;
                        conjugate.theta_points = config.atlas_polygon_points;
                        records.push_back(std::move(conjugate));
                    }
                    const std::size_t record_count = records.size();
                    checkpoint_writer.enqueue(std::move(records));
                    produced_records.fetch_add(
                        record_count, std::memory_order_relaxed);
                } catch (const std::exception& error) {
                    ++failed;
                    std::lock_guard<std::mutex> lock(errors_mutex);
                    std::ostringstream detail;
                    detail << "period=" << job.center.period
                           << " component=" << job.center.component_index
                           << " center=(" << real_string(job.center.center.real(), 12)
                           << ',' << real_string(job.center.center.imag(), 12)
                           << ") area=" << real_string(job.area, 6)
                           << ": " << error.what();
                    errors.push_back(detail.str());
                }

                const std::size_t done = ++completed;
                if (done == jobs.size() || done % 25 == 0) {
                    std::ostringstream detail;
                    detail << "failed=" << failed.load()
                           << " durable=" << checkpointed_jobs.load()
                           << " batches=" << checkpoint_batches.load();
                    if (config.progress && config.progress_style == "bars") {
                        render_inline_progress(
                            config,
                            "  atlas geometry",
                            done,
                            jobs.size(),
                            started,
                            detail.str(),
                            done == jobs.size(),
                            true);
                    } else if (config.progress_style == "lines") {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cout << "  atlas geometry " << done << '/'
                                  << jobs.size() << ' ' << detail.str()
                                  << " elapsed="
                                  << format_duration(Clock::now() - started)
                                  << '\n';
                    }
                }
            });
    }

    std::size_t expected_new_records = 0;
    for (const auto& job : jobs) {
        expected_new_records +=
            job.center.conjugate_index == job.center.component_index ? 1 : 2;
    }

    const auto flush_started = Clock::now();
    render_inline_progress(
        config,
        "  flushing geometry checkpoints",
        0,
        1,
        flush_started,
        "pending SQLite batch");
    checkpoint_writer.finish();
    render_inline_progress(
        config,
        "  flushing geometry checkpoints",
        1,
        1,
        flush_started,
        std::to_string(checkpointed_jobs.load()) + " durable job(s)",
        true,
        false);

    const CatalogueWriteStats write_stats = checkpoint_writer.stats();
    if (write_stats.written_representatives > 0) {
        catalogue_changed = true;
        std::cout << "Canonical geometry checkpoints wrote "
                  << write_stats.written_representatives
                  << " representative SQLite component row(s) in "
                  << checkpoint_batches.load() << " atomic batch(es)";
        if (failed.load() > 0) {
            std::cout << "; a rerun will reuse them";
        }
        std::cout << '\n';
    } else if (failed.load() == 0) {
        std::cout << "Canonical catalogue already contains the requested "
                  << "period geometry.\n";
    }

    if (mp_successes.load() > 0) {
        std::cout << "  multiprecision geometry recoveries: "
                  << mp_successes.load() << '\n';
    }

    if (failed.load() > 0) {
        std::ostringstream message;
        message << "Canonical geometry tracing failed for " << failed.load()
                << " representative job(s). "
                << write_stats.written_representatives
                << " successful representative(s) were checkpointed; rerun "
                << "to process only the remaining failures.";
        for (std::size_t i = 0;
             i < std::min<std::size_t>(errors.size(), 8);
             ++i) {
            message << "\n  " << errors[i];
        }
        throw std::runtime_error(message.str());
    }

    if (checkpointed_jobs.load() != jobs.size()) {
        throw std::runtime_error(
            "Geometry checkpoint writer committed "
            + std::to_string(checkpointed_jobs.load())
            + " representative job(s); expected "
            + std::to_string(jobs.size()) + '.');
    }

    if (produced_records.load() != expected_new_records) {
        throw std::runtime_error(
            "Atlas geometry symmetry expansion produced "
            + std::to_string(produced_records.load())
            + " new record(s); expected "
            + std::to_string(expected_new_records) + '.');
    }

    const auto period_centers_it = centers.find(config.period);
    if (period_centers_it == centers.end()) {
        throw std::runtime_error(
            "Cannot finalize canonical period without its center rows.");
    }
    const auto finalize_started = Clock::now();
    std::cout << "  finalizing period index and catalogue manifest...\n";
    finalize_exact_catalogue_period(
        config,
        period_centers_it->second,
        areas,
        write_stats,
        catalogue_changed);
    std::cout << "  catalogue finalization complete in "
              << format_duration(Clock::now() - finalize_started) << '\n';
    return catalogue_changed;
}

std::string demo_atlas_state_signature(
    const Config& config,
    std::uint64_t catalogue_revision
) {
    std::ostringstream output;
    output << "version=1\n"
           << "catalogue_revision=" << catalogue_revision << '\n'
           << "max_period=" << config.period << '\n'
           << "polygon_rho=" << real_string(config.atlas_polygon_rho) << '\n'
           << "polygon_points=" << config.atlas_polygon_points << '\n'
           << "min_area=" << real_string(config.demo_min_area) << '\n';
    return output.str();
}

std::string read_small_text_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) return {};
    return {std::istreambuf_iterator<char>(input), {}};
}

void write_small_text_file_atomic(
    const fs::path& path,
    const std::string& contents
) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        throw std::runtime_error("Could not write " + temporary.string());
    }
    output << contents;
    output.close();
    atomic_replace(temporary, path);
}

void update_demo_atlas_export(
    const Config& config,
    const std::map<int, std::vector<CenterRow>>& centers,
    bool catalogue_changed
) {
    (void)centers;
    using namespace mandelbrot::catalogue;
    Catalogue catalogue(config.catalogue_root);
    const Manifest manifest = catalogue.load_manifest();
    const fs::path state_path = config.run_dir / "demo_atlas.state";
    const std::string signature = demo_atlas_state_signature(
        config, manifest.catalogue_revision);
    if (!catalogue_changed
        && fs::is_regular_file(config.atlas_geometry_file)
        && read_small_text_file(state_path) == signature) {
        std::cout << "Demo atlas export already current.\n";
        return;
    }

    ComponentExportOptions options;
    options.format = "mandelbrot-atlas-geometry-v3";
    options.complete = false;
    options.coordinate_digits = std::numeric_limits<Real>::max_digits10;
    options.query.max_period = config.period;
    options.query.min_area = CatalogueReal(config.demo_min_area);
    options.query.require_polygon = true;
    options.query.require_polygon_converged = true;
    const auto export_started = Clock::now();
    const auto selected = catalogue.write_component_export(
        config.atlas_geometry_file,
        options,
        make_inline_progress_callback(
            config, "  selecting demo geometry", export_started, {}, true),
        make_inline_progress_callback(
            config, "  writing demo geometry", export_started, {}, true));
    write_small_text_file_atomic(state_path, signature);
    std::cout << "  demo atlas export complete in "
              << format_duration(Clock::now() - export_started) << '\n';
    std::cout << "Demo atlas selection: " << selected
              << " canonical component polygon(s) with area >= "
              << real_string(config.demo_min_area, 4) << '\n'
              << "Atlas geometry written by catalogue module: "
              << config.atlas_geometry_file << '\n';
}

std::vector<AreaMeasurementRecord> load_area_measurements_with_checkpoints(
    const AreaScanStore& store,
    const Config& config,
    int period
) {
    const auto load_started = Clock::now();
    auto measurements = store.load_measurements(
        period,
        make_inline_progress_callback(
            config,
            "  loading measurements p" + std::to_string(period),
            load_started,
            {},
            true));

    const std::size_t removed_temporaries =
        remove_stale_area_checkpoint_temporaries(store);
    if (removed_temporaries > 0) {
        std::cout << "  removed " << removed_temporaries
                  << " incomplete area checkpoint temporary file(s)\n";
    }

    const auto batches = list_area_checkpoint_batches(store);
    if (batches.empty()) return measurements;

    const std::size_t canonical_rows = measurements.size();
    std::size_t staged_rows = 0;
    const auto recovery_started = Clock::now();
    render_inline_progress(
        config,
        "  recovering checkpoints",
        0,
        batches.size(),
        recovery_started,
        {},
        false,
        true);
    for (std::size_t i = 0; i < batches.size(); ++i) {
        auto rows = store.load_measurements_from(batches[i]);
        measurements.reserve(measurements.size() + rows.size());
        for (auto& row : rows) {
            if (row.period != period) continue;
            measurements.push_back(std::move(row));
            ++staged_rows;
        }
        render_inline_progress(
            config,
            "  recovering checkpoints",
            i + 1,
            batches.size(),
            recovery_started,
            "rows=" + std::to_string(staged_rows),
            i + 1 == batches.size(),
            true);
    }

    const std::size_t removed_duplicates = canonicalize_measurement_rows(
        measurements,
        config,
        "  merging recovered rows");
    std::cout << "  recovered " << staged_rows << " staged row(s) from "
              << batches.size() << " atomic checkpoint batch(es); "
              << measurements.size() << " canonical row(s) available";
    if (removed_duplicates > 0) {
        std::cout << ", " << removed_duplicates << " duplicate/older row(s) removed";
    }
    std::cout << " (base file had " << canonical_rows << ")\n";
    return measurements;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    install_terminal_cleanup_handlers();
    try {
        const std::string usage =
            "Usage: component_area_scan [--config PATH] [--verify-catalogue]";
        const auto cli = mandelbrot::repo::parse_common_cli(argc, argv, usage);
        if (cli.help) { std::cout << usage << '\n'; return 0; }
        Config config = read_repository_config(cli.config, argv[0]);
        for (const auto& argument : cli.remaining) {
            if (argument == "--verify-catalogue") {
                config.verify_catalogue = true;
            } else {
                throw std::runtime_error(
                    "Unknown option: " + argument + "\n" + usage);
            }
        }
        fs::create_directories(config.output_dir);
        fs::create_directories(config.run_dir);
        mandelbrot::catalogue::Catalogue catalogue(config.catalogue_root);
        catalogue.ensure_layout();

        const AreaScanStore store = catalogue.area_scan_store(config.run_name);
        const fs::path catalogue_database = catalogue.database_path();

        std::cout << "Mandelbrot hyperbolic-component area scanner (C++20)\n"
                  << "  periods:      " << config.period_start << ".." << config.period << '\n'
                  << "  code root:    " << config.code_root << '\n'
                  << "  project root: " << config.project_root << '\n'
                  << "  catalogue:    " << config.catalogue_root << '\n'
                  << "  exports:      " << config.output_dir << '\n'
                  << "  root threads: " << config.threads << '\n'
                  << "  root symmetry: "
                  << (config.root_half_plane_symmetry
                        ? "real-axis prepass + upper-half conjugate cloud"
                        : "full-plane cloud") << '\n'
                  << "  root accelerator: "
                  << (config.root_tree_enabled
                        ? "Barnes-Hut + early Newton + unresolved-cluster AE"
                        : (config.root_grid_enabled
                            ? "uniform interaction grid"
                            : "exact all-pairs")) << '\n'
                  << "  area threads: " << config.area_threads << '\n'
                  << "  area checkpoints: atomic batches every "
                  << config.area_checkpoint_components << " job(s) or "
                  << config.area_checkpoint_seconds << "s\n"
                  << "  MP fallback:  " << (config.mp_fallback ? "enabled" : "disabled")
                  << ", guard=" << config.mp_guard_digits << " digits" << '\n'
                  << "  geometry:      " << (config.export_atlas_geometry ? "canonical + demo export" : "disabled")
                  << (config.export_atlas_geometry
                        ? ", demo cutoff=" + real_string(config.demo_min_area, 3)
                        : std::string()) << '\n'
                  << "  geometry checkpoints: atomic SQLite batches every "
                  << config.geometry_checkpoint_components << " job(s) or "
                  << config.geometry_checkpoint_seconds << "s\n"
                  << "  real type: long double (" << std::numeric_limits<Real>::digits10
                  << " decimal digits)\n"
                  << "  catalogue resume: "
                  << (config.verify_catalogue
                        ? "deep verification"
                        : "period-index fast path")
                  << "\n\n";

        std::map<int, std::vector<CenterRow>> centers;
        std::vector<AreaMeasurementRecord> measurements;
        auto summaries = store.load_summaries();
        int area_failures = 0;
        int completed_through_period = 0;
        bool catalogue_changed_any = false;

        if (!config.compute_areas) {
            std::cout << "Center solving will run period by period; "
                      << "compute_areas=false, so existing area measurements "
                      << "will be reused.\n";
        }

        for (int period = std::max(1, config.period_start); period <= config.period; ++period) {
            // Keep only the current period and, while solving, its immediate
            // predecessor. All older periods remain queryable in SQLite.
            for (auto it = centers.begin(); it != centers.end();) {
                if (it->first < period - 1) it = centers.erase(it);
                else ++it;
            }
            if (!centers.contains(period)) {
                centers.emplace(period, load_center_period(store, config, period));
            }
            auto centers_it = centers.find(period);
            const bool centers_cached =
                config.resume
                && centers_it != centers.end()
                && complete_center_period(centers_it->second, period, config);

            if (centers_cached) {
                std::cout << "period " << period << ": reusing "
                          << centers_it->second.size()
                          << " cached center(s)\n";
            } else {
                const std::size_t expected = center_count(period);
                std::cout << "\nperiod " << period << ": solving "
                          << expected
                          << " exact-period center(s) without expanding the "
                          << "polynomial...\n";
                const fs::path checkpoint = store.root_checkpoint_path(period);
                std::vector<CenterRow> rows;
                if (period <= 2) {
                    rows = analytic_center_rows(period);
                    std::cout << "  analytic center: c="
                              << (period == 1 ? "0" : "-1") << '\n';
                } else {
                    std::vector<Complex> warm_start_centers;
                    if (config.root_warm_start_previous_period && period > 1) {
                        if (!centers.contains(period - 1)) {
                            centers.emplace(
                                period - 1,
                                load_center_period(store, config, period - 1));
                        }
                        const auto previous = centers.find(period - 1);
                        if (previous != centers.end()
                            && complete_center_period(
                                previous->second, period - 1, config)) {
                            warm_start_centers.reserve(previous->second.size());
                            for (const auto& row : previous->second) {
                                warm_start_centers.push_back(row.center);
                            }
                            std::cout << "  warm-start scaffold: "
                                      << warm_start_centers.size()
                                      << " certified period-" << (period - 1)
                                      << " center(s)\n";
                        }
                    }
                    auto solve = solve_center_roots(
                        period, config, checkpoint, warm_start_centers);
                    rows = refine_root_cloud(period, solve, config);
                }
                centers[period] = std::move(rows);
                write_center_period(store, period, centers[period], config);
                std::error_code error;
                fs::remove(checkpoint, error);
                std::cout << "  centers committed: "
                          << catalogue_database << '\n';
                centers_it = centers.find(period);
            }

            // The predecessor is needed only to seed this period's root solve.
            // Release it before the much larger area/geometry stages begin.
            if (period > 1) centers.erase(period - 1);

            if (period < config.period_start) {
                completed_through_period = period;
                continue;
            }

            Config period_config = config;
            period_config.period_start = period;
            period_config.period = period;

            if (config.compute_areas) {
                const bool areas_cached = complete_area_period(
                    summaries,
                    centers_it->second,
                    period,
                    period_config);
                if (areas_cached) {
                    std::cout << "period " << period
                              << ": area measurements already complete; "
                              << "using cached period data\n";
                } else if (centers_cached) {
                    std::cout << "period " << period
                              << ": found cached centers but incomplete area "
                              << "measurements; computing them now\n";
                } else {
                    std::cout << "period " << period
                              << ": centers complete; computing component "
                              << "areas\n";
                }

                if (!areas_cached) {
                    measurements = load_area_measurements_with_checkpoints(
                        store, period_config, period);
                    const int period_failures = run_parallel_area_stage(
                        period_config,
                        centers,
                        store,
                        measurements);
                    area_failures += period_failures;
                    summaries = store.load_summaries();
                    if (period_failures != 0) {
                        std::cout << "period " << period << ": "
                                  << period_failures
                                  << " area job(s) failed; stopping before period "
                                  << (period + 1) << " so this period can resume "
                                  << "cleanly\n";
                        break;
                    }
                }
            }

            if (config.export_atlas_geometry) {
                catalogue_changed_any = export_atlas_geometry(
                    period_config,
                    centers,
                    store,
                    measurements) || catalogue_changed_any;
            }

            // Area rows are potentially much larger than the center cloud and
            // are never needed by the next period's root warm start.
            measurements.clear();
            measurements.shrink_to_fit();

            completed_through_period = period;
            std::cout << "period " << period
                      << ": centers, areas, and requested geometry are "
                      << "complete\n\n";
        }

        if (config.export_atlas_geometry
            && completed_through_period >= config.period_start) {
            Config demo_config = config;
            demo_config.period = completed_through_period;
            update_demo_atlas_export(
                demo_config,
                centers,
                catalogue_changed_any);
        }

        std::cout << "\nDone.\n"
                  << "  catalogue and scanner state: "
                  << catalogue_database << '\n';
        if (config.export_atlas_geometry) {
            std::cout << "  atlas:      " << config.atlas_geometry_file << '\n';
        }
        std::cout << "  completed through period: "
                  << completed_through_period << '\n'
                  << "  area failures: " << area_failures << '\n';
        return area_failures == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        leave_alternate_screen();
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
