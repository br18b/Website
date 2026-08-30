#include "common/repo_config.hpp"
#include "components/catalogue/component_catalogue.hpp"
#include "components/shapes/component_shapes.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <ctime>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using mandelbrot::catalogue::Catalogue;
using mandelbrot::catalogue::CatalogueReal;
using mandelbrot::catalogue::ClassificationRecord;
using mandelbrot::catalogue::ClassificationUpdate;
using mandelbrot::catalogue::ComponentRecord;
using mandelbrot::repo::RepoConfig;
using mandelbrot::shapes::CardioidFitAnalysis;
using mandelbrot::shapes::CardioidFitOptions;
using mandelbrot::shapes::CircleFitAnalysis;
using mandelbrot::shapes::CircleFitOptions;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    fs::path config;
    fs::path catalogue_root;
    int min_period = 1;
    int max_period = std::numeric_limits<int>::max();
    CatalogueReal min_area = 0;
    unsigned threads = 0;
    bool force = false;
    bool dry_run = false;
    bool progress = true;
    std::string progress_style = "bars";
    int progress_bar_width = 32;
    int progress_refresh_ms = 500;
    bool export_failed_cardioids = true;
    fs::path failed_cardioid_export_dir;
    CircleFitOptions circle;
    CardioidFitOptions cardioid;
};

constexpr const char* kUsage = R"(Usage: classify_component_shapes [options]

Classify catalogue polygons as disks or translated, rotated, slanted
cardioids.  The disk fit runs first; only disk failures enter the cardioid fit.

Options:
  --config PATH             Alternate unified repository config
  --catalogue-root PATH     Override paths.catalogue_root
  --min-period N            First period to inspect (default: 1)
  --max-period N            Last period to inspect (default: no limit)
  --min-area VALUE          Ignore smaller canonical representatives
  --circle-rms VALUE        Relative RMS acceptance threshold
  --circle-max VALUE        Relative maximum-error threshold
  --cardioid-rms VALUE      Relative cardioid RMS acceptance threshold
  --cardioid-max VALUE      Relative cardioid maximum-error threshold
  --cardioid-cusps N        Candidate cusp points nearest the component centre
  --cardioid-iterations N   Joint cardioid polish iterations
  --cardioid-shake-trials N Randomized fallback proposals for failed fits
  --no-cardioid-shake       Disable the randomized failed-fit fallback
  --threads N               Worker threads
  --force                   Refit records that are already classified
  --dry-run                 Measure fits without writing the catalogue
  --failed-export-dir PATH  Failed-fit diagnostics directory
  --no-failed-export        Do not write failed-cardioid diagnostics
  --no-progress             Disable updating progress bars
  -h, --help                Show this help
)";

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::string format_duration(Clock::duration duration) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        duration).count();
    std::ostringstream output;
    if (seconds < 60) {
        output << seconds << 's';
    } else if (seconds < 3600) {
        output << seconds / 60 << ':' << std::setw(2) << std::setfill('0')
               << seconds % 60;
    } else {
        output << seconds / 3600 << ':' << std::setw(2) << std::setfill('0')
               << (seconds / 60) % 60 << ':' << std::setw(2)
               << seconds % 60;
    }
    return output.str();
}

bool stdout_is_terminal() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(STDOUT_FILENO) != 0;
#else
    return true;
#endif
}

int terminal_width_columns() {
#if defined(__unix__) || defined(__APPLE__)
    struct winsize size{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
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

std::string fit_to_width(std::string text, int width) {
    if (width <= 0) return {};
    if (static_cast<int>(text.size()) <= width) return text;
    if (width <= 3) return text.substr(0, static_cast<std::size_t>(width));
    return text.substr(0, static_cast<std::size_t>(width - 3)) + "...";
}

std::string progress_bar(long double fraction, int width) {
    width = std::max(1, width);
    fraction = std::clamp(fraction, 0.0L, 1.0L);
    const int filled = std::clamp(
        static_cast<int>(std::llround(fraction * width)), 0, width);
    return std::string(static_cast<std::size_t>(filled), '#')
         + std::string(static_cast<std::size_t>(width - filled), '-');
}

class InlineProgress {
public:
    InlineProgress(
        const Options& options,
        std::string label,
        std::size_t total)
        : enabled_(options.progress && options.progress_style == "bars"),
          terminal_(stdout_is_terminal()),
          label_(std::move(label)),
          total_(total),
          bar_width_(std::max(6, options.progress_bar_width)),
          refresh_(std::chrono::milliseconds(
              std::max(50, options.progress_refresh_ms))),
          started_(Clock::now()),
          last_render_(started_ - refresh_) {}

    void update(
        std::size_t current,
        const std::string& detail = {},
        bool final = false,
        bool force = false) {
        if (finished_) return;
        const auto now = Clock::now();
        if (!final && !force && now - last_render_ < refresh_) return;
        last_render_ = now;

        if (!enabled_) {
            if (final) {
                std::cout << "  " << label_ << ": " << current << '/'
                          << total_;
                if (!detail.empty()) std::cout << " | " << detail;
                std::cout << " | elapsed "
                          << format_duration(now - started_) << '\n';
                finished_ = true;
            }
            return;
        }

        const long double fraction = total_ > 0
            ? static_cast<long double>(current)
                / static_cast<long double>(total_)
            : 1.0L;
        std::ostringstream suffix;
        suffix << current << '/' << total_;
        if (!detail.empty()) suffix << " | " << detail;
        suffix << " | elapsed " << format_duration(now - started_);
        if (current > 0 && current < total_) {
            const long double elapsed_seconds =
                std::chrono::duration<long double>(now - started_).count();
            const long double eta_seconds = elapsed_seconds
                * static_cast<long double>(total_ - current)
                / static_cast<long double>(current);
            suffix << " | eta " << format_duration(
                std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<long double>(eta_seconds)));
        }

        std::ostringstream percentage;
        percentage << std::fixed << std::setprecision(1)
                   << static_cast<double>(100 * std::clamp(
                          fraction, 0.0L, 1.0L)) << '%';
        const int width = std::max(20, terminal_width_columns() - 1);
        const int fixed = 2 + static_cast<int>(label_.size())
            + 4 + static_cast<int>(percentage.str().size())
            + 1 + static_cast<int>(suffix.str().size());
        const int actual_bar_width = std::max(
            6, std::min(bar_width_, std::max(6, width - fixed)));
        std::string line = "  " + label_ + " ["
            + progress_bar(fraction, actual_bar_width) + "] "
            + percentage.str() + ' ' + suffix.str();
        line = fit_to_width(std::move(line), width);

        if (terminal_) {
            std::cout << "\r\033[2K" << line;
            if (final) std::cout << '\n';
            std::cout.flush();
        } else if (final) {
            std::cout << line << '\n';
        }
        if (final) finished_ = true;
    }

private:
    bool enabled_ = true;
    bool terminal_ = true;
    bool finished_ = false;
    std::string label_;
    std::size_t total_ = 0;
    int bar_width_ = 32;
    std::chrono::milliseconds refresh_{500};
    Clock::time_point started_;
    Clock::time_point last_render_;
};

Options parse_options(int argc, char** argv) {
    Options options;
    bool min_period_explicit = false;
    bool max_period_explicit = false;
    bool min_area_explicit = false;
    bool circle_rms_explicit = false;
    bool circle_max_explicit = false;
    bool cardioid_rms_explicit = false;
    bool cardioid_max_explicit = false;
    bool cardioid_cusps_explicit = false;
    bool cardioid_iterations_explicit = false;
    bool cardioid_shake_trials_explicit = false;
    bool cardioid_shake_enabled_explicit = false;
    bool threads_explicit = false;
    bool progress_explicit = false;
    bool failed_export_dir_explicit = false;
    bool failed_export_explicit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (++i >= argc) throw std::runtime_error(
                std::string(name) + " requires a value\n" + kUsage);
            return argv[i];
        };
        if (argument == "--config") options.config = value("--config");
        else if (argument == "--catalogue-root") {
            options.catalogue_root = value("--catalogue-root");
        } else if (argument == "--min-period") {
            options.min_period = std::stoi(value("--min-period"));
            min_period_explicit = true;
        } else if (argument == "--max-period") {
            options.max_period = std::stoi(value("--max-period"));
            max_period_explicit = true;
        } else if (argument == "--min-area") {
            options.min_area = Catalogue::parse_decimal(value("--min-area"));
            min_area_explicit = true;
        } else if (argument == "--circle-rms") {
            options.circle.rms_tolerance = Catalogue::parse_decimal(
                value("--circle-rms"));
            circle_rms_explicit = true;
        } else if (argument == "--circle-max") {
            options.circle.max_error_tolerance = Catalogue::parse_decimal(
                value("--circle-max"));
            circle_max_explicit = true;
        } else if (argument == "--cardioid-rms") {
            options.cardioid.rms_tolerance = Catalogue::parse_decimal(
                value("--cardioid-rms"));
            cardioid_rms_explicit = true;
        } else if (argument == "--cardioid-max") {
            options.cardioid.max_error_tolerance = Catalogue::parse_decimal(
                value("--cardioid-max"));
            cardioid_max_explicit = true;
        } else if (argument == "--cardioid-cusps") {
            options.cardioid.cusp_candidates = std::stoi(
                value("--cardioid-cusps"));
            cardioid_cusps_explicit = true;
        } else if (argument == "--cardioid-iterations") {
            options.cardioid.max_iterations = std::stoi(
                value("--cardioid-iterations"));
            cardioid_iterations_explicit = true;
        } else if (argument == "--cardioid-shake-trials") {
            options.cardioid.shake_trials = std::stoi(
                value("--cardioid-shake-trials"));
            options.cardioid.randomized_fallback =
                options.cardioid.shake_trials > 0;
            cardioid_shake_trials_explicit = true;
            cardioid_shake_enabled_explicit = true;
        } else if (argument == "--no-cardioid-shake") {
            options.cardioid.randomized_fallback = false;
            cardioid_shake_enabled_explicit = true;
        } else if (argument == "--threads") {
            options.threads = static_cast<unsigned>(
                std::stoul(value("--threads")));
            threads_explicit = true;
        } else if (argument == "--force") {
            options.force = true;
        } else if (argument == "--dry-run") {
            options.dry_run = true;
        } else if (argument == "--failed-export-dir") {
            options.failed_cardioid_export_dir = value("--failed-export-dir");
            failed_export_dir_explicit = true;
        } else if (argument == "--no-failed-export") {
            options.export_failed_cardioids = false;
            failed_export_explicit = true;
        } else if (argument == "--no-progress") {
            options.progress = false;
            progress_explicit = true;
        } else if (argument == "-h" || argument == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else {
            throw std::runtime_error(
                "Unknown option: " + argument + "\n" + kUsage);
        }
    }

    const RepoConfig config = RepoConfig::load(
        options.config,
        fs::path(argv[0] ? argv[0] : "."));
    if (options.catalogue_root.empty()) {
        options.catalogue_root = config.path("paths.catalogue_root");
    } else {
        options.catalogue_root = fs::absolute(options.catalogue_root);
    }
    const std::string prefix = "component_shape_classifier.";
    // Period selection is intentionally command-line only. Post-processing
    // operates on the complete accumulated catalogue by default and must not
    // inherit the current component-area scan window from mandelbrot.json.
    // --min-period and --max-period remain available for deliberate subsets.
    (void)min_period_explicit;
    (void)max_period_explicit;
    if (!min_area_explicit) {
        options.min_area = CatalogueReal(config.number(
            prefix + "min_area", options.min_area.convert_to<long double>()));
    }
    if (!circle_rms_explicit) {
        options.circle.rms_tolerance = CatalogueReal(config.number(
            prefix + "circle_fit_rms_tolerance",
            options.circle.rms_tolerance.convert_to<long double>()));
    }
    if (!circle_max_explicit) {
        options.circle.max_error_tolerance = CatalogueReal(config.number(
            prefix + "circle_fit_max_tolerance",
            options.circle.max_error_tolerance.convert_to<long double>()));
    }
    options.circle.max_iterations = config.integer(
        prefix + "circle_fit_iterations", options.circle.max_iterations);
    if (!cardioid_rms_explicit) {
        options.cardioid.rms_tolerance = CatalogueReal(config.number(
            prefix + "cardioid_fit_rms_tolerance",
            options.cardioid.rms_tolerance.convert_to<long double>()));
    }
    if (!cardioid_max_explicit) {
        options.cardioid.max_error_tolerance = CatalogueReal(config.number(
            prefix + "cardioid_fit_max_tolerance",
            options.cardioid.max_error_tolerance.convert_to<long double>()));
    }
    if (!cardioid_cusps_explicit) {
        options.cardioid.cusp_candidates = config.integer(
            prefix + "cardioid_cusp_candidates",
            options.cardioid.cusp_candidates);
    }
    if (!cardioid_iterations_explicit) {
        options.cardioid.max_iterations = config.integer(
            prefix + "cardioid_fit_iterations",
            options.cardioid.max_iterations);
    }
    options.cardioid.angle_iterations = config.integer(
        prefix + "cardioid_angle_iterations",
        options.cardioid.angle_iterations);
    options.cardioid.initial_angle_step = CatalogueReal(config.number(
        prefix + "cardioid_initial_angle_step",
        options.cardioid.initial_angle_step.convert_to<long double>()));
    options.cardioid.xi_limit = CatalogueReal(config.number(
        prefix + "cardioid_xi_limit",
        options.cardioid.xi_limit.convert_to<long double>()));
    if (!cardioid_shake_enabled_explicit) {
        options.cardioid.randomized_fallback = config.boolean(
            prefix + "cardioid_shake_enabled",
            options.cardioid.randomized_fallback);
    }
    if (!cardioid_shake_trials_explicit) {
        options.cardioid.shake_trials = config.integer(
            prefix + "cardioid_shake_trials",
            options.cardioid.shake_trials);
    }
    options.cardioid.shake_keep = config.integer(
        prefix + "cardioid_shake_keep",
        options.cardioid.shake_keep);
    options.cardioid.shake_cusp_jitter = config.integer(
        prefix + "cardioid_shake_cusp_jitter",
        options.cardioid.shake_cusp_jitter);
    options.cardioid.shake_angle_sigma = CatalogueReal(config.number(
        prefix + "cardioid_shake_angle_sigma",
        options.cardioid.shake_angle_sigma.convert_to<long double>()));
    options.cardioid.shake_phase_sigma = CatalogueReal(config.number(
        prefix + "cardioid_shake_phase_sigma",
        options.cardioid.shake_phase_sigma.convert_to<long double>()));
    options.cardioid.shake_temperature = CatalogueReal(config.number(
        prefix + "cardioid_shake_temperature",
        options.cardioid.shake_temperature.convert_to<long double>()));
    options.cardioid.shake_final_temperature = CatalogueReal(config.number(
        prefix + "cardioid_shake_final_temperature",
        options.cardioid.shake_final_temperature.convert_to<long double>()));
    if (!threads_explicit || options.threads == 0) {
        options.threads = config.threads();
    }
    if (!progress_explicit) {
        options.progress = config.boolean(
            prefix + "progress",
            config.boolean("runtime.progress.enabled", options.progress));
    }
    options.progress_style = config.string(
        prefix + "progress_style",
        config.string("runtime.progress.style", options.progress_style));
    options.progress_bar_width = config.integer(
        prefix + "progress_bar_width",
        config.integer("runtime.progress.bar_width", options.progress_bar_width));
    options.progress_refresh_ms = config.integer(
        prefix + "progress_refresh_ms",
        config.integer("runtime.progress.refresh_ms", options.progress_refresh_ms));
    if (!failed_export_explicit) {
        options.export_failed_cardioids = config.boolean(
            prefix + "export_failed_cardioids",
            options.export_failed_cardioids);
    }
    if (!failed_export_dir_explicit) {
        options.failed_cardioid_export_dir = options.catalogue_root / "exports"
            / config.string(
                prefix + "failed_cardioid_export_dir",
                "shape_debug/failed_cardioids");
    } else if (options.failed_cardioid_export_dir.is_relative()) {
        options.failed_cardioid_export_dir = fs::absolute(
            options.failed_cardioid_export_dir);
    }

    if (options.min_period < 1 || options.max_period < options.min_period) {
        throw std::runtime_error("Require 1 <= min_period <= max_period");
    }
    if (options.min_area < 0
        || options.circle.rms_tolerance <= 0
        || options.circle.max_error_tolerance <= 0
        || options.cardioid.rms_tolerance <= 0
        || options.cardioid.max_error_tolerance <= 0
        || options.cardioid.cusp_candidates < 1
        || options.cardioid.max_iterations < 0
        || options.cardioid.angle_iterations < 0
        || options.cardioid.initial_angle_step <= 0
        || options.cardioid.xi_limit < 0
        || options.cardioid.xi_limit > CatalogueReal("0.5")
        || options.cardioid.shake_trials < 0
        || options.cardioid.shake_keep < 1
        || options.cardioid.shake_cusp_jitter < 0
        || options.cardioid.shake_angle_sigma < 0
        || options.cardioid.shake_phase_sigma < 0
        || options.cardioid.shake_temperature <= 0
        || options.cardioid.shake_final_temperature <= 0
        || options.cardioid.shake_final_temperature
            > options.cardioid.shake_temperature) {
        throw std::runtime_error(
            "Shape-fit thresholds and iteration counts are invalid");
    }
    options.threads = std::max(1u, options.threads);
    return options;
}

bool component_is_eligible(
    const ComponentRecord& component,
    const Options& options) {
    if (component.period < options.min_period
        || component.period > options.max_period) {
        return false;
    }
    if (component.geometry.area_estimate < options.min_area) return false;
    if (component.geometry.polygon.empty()) return false;
    if (!component.quality.polygon_converged) return false;
    return options.force
        || component.classification.shape_class == "unknown";
}

ClassificationRecord classification_from_circle_fit(
    const CircleFitAnalysis& analysis,
    const CircleFitOptions& options) {
    ClassificationRecord classification;
    classification.shape_class = "disk";
    const CatalogueReal rms_ratio = analysis.fit.rms / options.rms_tolerance;
    const CatalogueReal max_ratio = *analysis.fit.max_error
        / options.max_error_tolerance;
    const CatalogueReal score = std::max(rms_ratio, max_ratio);
    classification.shape_confidence = std::max(
        CatalogueReal("0.5"), CatalogueReal(1) - score / 2);
    classification.circle_fit = analysis.fit;
    return classification;
}


ClassificationRecord classification_from_cardioid_fit(
    const CircleFitAnalysis& circle,
    const CardioidFitAnalysis& analysis,
    const CardioidFitOptions& options) {
    ClassificationRecord classification;
    classification.shape_class = "cardioid";
    const CatalogueReal rms_ratio = analysis.fit.rms / options.rms_tolerance;
    const CatalogueReal max_ratio = *analysis.fit.max_error
        / options.max_error_tolerance;
    const CatalogueReal score = std::max(rms_ratio, max_ratio);
    classification.shape_confidence = std::max(
        CatalogueReal("0.5"), CatalogueReal(1) - score / 2);
    if (circle.converged) classification.circle_fit = circle.fit;
    classification.cardioid_fit = analysis.fit;
    return classification;
}

std::vector<std::string> discover_component_ids(
    const Catalogue& catalogue,
    const Options& options) {
    const auto all_periods = catalogue.list_periods();
    std::vector<int> periods;
    for (const int period : all_periods) {
        if (period >= options.min_period && period <= options.max_period) {
            periods.push_back(period);
        }
    }

    std::vector<std::string> ids;
    InlineProgress progress(options, "reading period indexes", periods.size());
    progress.update(0, "candidate records=0", false, true);
    for (std::size_t index = 0; index < periods.size(); ++index) {
        try {
            const auto period = catalogue.load_period(periods[index]);
            ids.insert(
                ids.end(), period.component_ids.begin(), period.component_ids.end());
        } catch (...) {
            // Preserve the catalogue query behavior: a missing or stale period
            // index does not abort the whole classification run.
        }
        progress.update(
            index + 1,
            "candidate records=" + std::to_string(ids.size()),
            index + 1 == periods.size());
    }
    if (periods.empty()) progress.update(0, "no period indexes", true, true);

    if (ids.empty()) {
        std::cout << "  period indexes yielded no records; scanning component files...\n";
        ids = catalogue.list_component_ids();
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

struct FailedCardioidDiagnostic {
    std::string component_id;
    int period = 0;
    mandelbrot::catalogue::ComplexValue center;
    CatalogueReal area = 0;
    CatalogueReal polygon_rho = 0;
    std::vector<mandelbrot::catalogue::ComplexValue> polygon;
    CircleFitAnalysis circle;
    CardioidFitAnalysis cardioid;
};

CatalogueReal rejected_score(
    const FailedCardioidDiagnostic& diagnostic,
    const Options& options) {
    if (!diagnostic.cardioid.converged
        || !diagnostic.cardioid.fit.max_error) {
        return std::numeric_limits<CatalogueReal>::infinity();
    }
    return std::max(
        diagnostic.cardioid.fit.rms / options.cardioid.rms_tolerance,
        *diagnostic.cardioid.fit.max_error
            / options.cardioid.max_error_tolerance);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(ch)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    return output.str();
}

void atomic_replace_file(const fs::path& temporary, const fs::path& target) {
    std::error_code error;
    fs::rename(temporary, target, error);
    if (!error) return;
    fs::remove(target, error);
    error.clear();
    fs::rename(temporary, target, error);
    if (error) {
        throw std::runtime_error(
            "Could not replace " + target.string() + ": " + error.message());
    }
}

void write_number(std::ostream& output, const CatalogueReal& value) {
    output << Catalogue::decimal_string(value, 24);
}

void write_optional_number(
    std::ostream& output,
    const std::optional<CatalogueReal>& value) {
    if (value) write_number(output, *value);
    else output << "null";
}

void write_complex(
    std::ostream& output,
    const mandelbrot::catalogue::ComplexValue& value) {
    output << '[';
    write_number(output, value.re);
    output << ',';
    write_number(output, value.im);
    output << ']';
}

void write_failed_cardioid_diagnostics(
    const Options& options,
    std::vector<FailedCardioidDiagnostic> diagnostics) {
    if (!options.export_failed_cardioids) return;
    fs::create_directories(options.failed_cardioid_export_dir);
    std::sort(
        diagnostics.begin(), diagnostics.end(),
        [&](const auto& left, const auto& right) {
            const CatalogueReal left_score = rejected_score(left, options);
            const CatalogueReal right_score = rejected_score(right, options);
            if (left_score != right_score) return left_score < right_score;
            if (left.period != right.period) return left.period < right.period;
            return left.component_id < right.component_id;
        });

    const fs::path ndjson = options.failed_cardioid_export_dir
        / "failed_cardioids.ndjson";
    const fs::path summary = options.failed_cardioid_export_dir
        / "summary.csv";
    const fs::path ndjson_tmp = ndjson.string() + ".tmp";
    const fs::path summary_tmp = summary.string() + ".tmp";

    InlineProgress progress(
        options, "writing failed-fit diagnostics", diagnostics.size());
    progress.update(0, "records=0", false, true);

    std::ofstream json_output(ndjson_tmp, std::ios::binary | std::ios::trunc);
    std::ofstream csv_output(summary_tmp, std::ios::binary | std::ios::trunc);
    if (!json_output || !csv_output) {
        throw std::runtime_error(
            "Could not create failed-cardioid diagnostic exports");
    }
    csv_output
        << "rank,component_id,period,area,polygon_points,circle_rms,circle_max,"
           "cardioid_rms,cardioid_max,score,size,angle,xi,cusp_index,direction,"
           "phase_offset,randomized_trials\n";

    for (std::size_t index = 0; index < diagnostics.size(); ++index) {
        const auto& item = diagnostics[index];
        const auto& fit = item.cardioid.fit;
        const CatalogueReal score = rejected_score(item, options);

        json_output << "{\"component_id\":\""
                    << json_escape(item.component_id)
                    << "\",\"period\":" << item.period
                    << ",\"center\":";
        write_complex(json_output, item.center);
        json_output << ",\"area\":";
        write_number(json_output, item.area);
        json_output << ",\"polygon_rho\":";
        write_number(json_output, item.polygon_rho);
        json_output << ",\"score\":";
        write_number(json_output, score);
        json_output << ",\"thresholds\":{\"rms\":";
        write_number(json_output, options.cardioid.rms_tolerance);
        json_output << ",\"max\":";
        write_number(json_output, options.cardioid.max_error_tolerance);
        json_output << "},\"circle_fit\":{\"converged\":"
                    << (item.circle.converged ? "true" : "false")
                    << ",\"rms\":";
        write_number(json_output, item.circle.fit.rms);
        json_output << ",\"max_error\":";
        write_optional_number(json_output, item.circle.fit.max_error);
        json_output << "},\"cardioid_fit\":{\"converged\":"
                    << (item.cardioid.converged ? "true" : "false")
                    << ",\"cusp_index\":" << item.cardioid.cusp_index
                    << ",\"direction\":" << item.cardioid.direction
                    << ",\"phase_offset\":";
        write_number(json_output, item.cardioid.phase_offset);
        json_output << ",\"randomized_fallback\":"
                    << (item.cardioid.used_randomized_fallback
                            ? "true" : "false")
                    << ",\"randomized_trials\":"
                    << item.cardioid.randomized_trials
                    << ",\"center_centered\":";
        if (fit.center_centered) write_complex(json_output, *fit.center_centered);
        else json_output << "null";
        json_output << ",\"size\":";
        write_optional_number(json_output, fit.size);
        json_output << ",\"angle\":";
        write_number(json_output, fit.angle);
        json_output << ",\"xi\":";
        write_number(json_output, fit.xi);
        json_output << ",\"rms\":";
        write_number(json_output, fit.rms);
        json_output << ",\"max_error\":";
        write_optional_number(json_output, fit.max_error);
        json_output << "},\"polygon_centered\":[";
        for (std::size_t point = 0; point < item.polygon.size(); ++point) {
            if (point) json_output << ',';
            write_complex(json_output, item.polygon[point]);
        }
        json_output << "]}\n";

        csv_output << index + 1 << ',' << item.component_id << ','
                   << item.period << ',';
        write_number(csv_output, item.area);
        csv_output << ',' << item.polygon.size() << ',';
        write_number(csv_output, item.circle.fit.rms);
        csv_output << ',';
        write_optional_number(csv_output, item.circle.fit.max_error);
        csv_output << ',';
        write_number(csv_output, fit.rms);
        csv_output << ',';
        write_optional_number(csv_output, fit.max_error);
        csv_output << ',';
        write_number(csv_output, score);
        csv_output << ',';
        write_optional_number(csv_output, fit.size);
        csv_output << ',';
        write_number(csv_output, fit.angle);
        csv_output << ',';
        write_number(csv_output, fit.xi);
        csv_output << ',' << item.cardioid.cusp_index << ','
                   << item.cardioid.direction << ',';
        write_number(csv_output, item.cardioid.phase_offset);
        csv_output << ',' << item.cardioid.randomized_trials << '\n';

        progress.update(
            index + 1,
            "records=" + std::to_string(index + 1),
            index + 1 == diagnostics.size());
    }
    if (diagnostics.empty()) {
        progress.update(0, "records=0", true, true);
    }
    json_output.close();
    csv_output.close();
    if (!json_output || !csv_output) {
        throw std::runtime_error(
            "Failed while writing failed-cardioid diagnostics");
    }
    atomic_replace_file(ndjson_tmp, ndjson);
    atomic_replace_file(summary_tmp, summary);
    std::cout << "  failed-fit diagnostics: "
              << options.failed_cardioid_export_dir << '\n';
}

struct WorkResult {
    bool eligible = false;
    CircleFitAnalysis circle;
    CardioidFitAnalysis cardioid;
    std::optional<ClassificationUpdate> update;
    std::optional<FailedCardioidDiagnostic> failed_cardioid;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Catalogue catalogue(options.catalogue_root);
        catalogue.ensure_layout();

        // Print the banner before any catalogue-wide scan. The previous
        // version loaded every polygon before producing its first byte of
        // output, which made a healthy run look frozen on large catalogues.
        std::cout << "Mandelbrot component-shape classifier\n"
                  << "  catalogue: " << catalogue.root() << '\n'
                  << "  periods:   ";
        if (options.max_period == std::numeric_limits<int>::max()) {
            std::cout << options.min_period << "..all catalogue periods\n";
        } else {
            std::cout << options.min_period << ".." << options.max_period << '\n';
        }
        std::cout
                  << "  min area:  "
                  << Catalogue::decimal_string(options.min_area, 8) << '\n'
                  << "  disk fit:  RMS <= "
                  << Catalogue::decimal_string(
                         options.circle.rms_tolerance, 8)
                  << ", max <= "
                  << Catalogue::decimal_string(
                         options.circle.max_error_tolerance, 8) << '\n'
                  << "  cardioid:  RMS <= "
                  << Catalogue::decimal_string(
                         options.cardioid.rms_tolerance, 8)
                  << ", max <= "
                  << Catalogue::decimal_string(
                         options.cardioid.max_error_tolerance, 8)
                  << ", |xi| <= "
                  << Catalogue::decimal_string(options.cardioid.xi_limit, 4)
                  << '\n'
                  << "  cusp fit:  " << options.cardioid.cusp_candidates
                  << " candidates, " << options.cardioid.max_iterations
                  << " joint iteration(s)\n"
                  << "  final shake: "
                  << (options.cardioid.randomized_fallback
                          ? std::to_string(options.cardioid.shake_trials)
                              + " reproducible Metropolis proposal(s)"
                          : std::string("disabled"))
                  << '\n'
                  << "  threads:   " << options.threads << '\n'
                  << "  mode:      "
                  << (options.force ? "refit all eligible polygons"
                                    : "classify unknown polygons only")
                  << (options.dry_run ? " (dry run)" : "") << '\n'
                  << "  failed fit export: "
                  << (options.export_failed_cardioids
                          ? options.failed_cardioid_export_dir.string()
                          : std::string("disabled"))
                  << '\n';

        const std::vector<std::string> component_ids =
            discover_component_ids(catalogue, options);
        std::cout << "  catalogue candidates: " << component_ids.size() << '\n';

        std::vector<WorkResult> results(component_ids.size());
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> completed{0};
        std::atomic<std::size_t> eligible_count{0};
        std::atomic<std::size_t> circle_converged_count{0};
        std::atomic<std::size_t> disk_count{0};
        std::atomic<std::size_t> cardioid_attempt_count{0};
        std::atomic<std::size_t> cardioid_converged_count{0};
        std::atomic<std::size_t> cardioid_count{0};
        std::atomic<std::size_t> shake_attempt_count{0};
        std::atomic<std::size_t> shake_rescue_count{0};
        std::atomic<bool> abort_workers{false};
        std::exception_ptr worker_error;
        std::mutex error_mutex;

        InlineProgress fit_progress(
            options, "loading polygons and fitting", component_ids.size());
        fit_progress.update(
            0, "eligible=0 disks=0 cardioids=0 shake=0/0", false, true);
        std::atomic<bool> monitor_stop{false};
        std::thread monitor;
        if (options.progress && options.progress_style == "bars") {
            monitor = std::thread([&] {
                while (!monitor_stop.load()) {
                    std::ostringstream detail;
                    detail << "eligible=" << eligible_count.load()
                           << " disks=" << disk_count.load()
                           << " cardioid=" << cardioid_count.load()
                           << '/' << cardioid_attempt_count.load()
                           << " shake=" << shake_rescue_count.load()
                           << '/' << shake_attempt_count.load();
                    fit_progress.update(completed.load(), detail.str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                        std::max(50, options.progress_refresh_ms)));
                }
            });
        }

        const unsigned worker_count = std::min<unsigned>(
            options.threads,
            static_cast<unsigned>(
                std::max<std::size_t>(1, component_ids.size())));
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&] {
                while (!abort_workers.load()) {
                    const std::size_t job = next.fetch_add(1);
                    if (job >= component_ids.size()) break;
                    try {
                        WorkResult result;
                        const std::string& component_id = component_ids[job];
                        if (catalogue.component_exists(component_id)) {
                            ComponentRecord component =
                                catalogue.load_component(component_id);
                            if (component_is_eligible(component, options)) {
                                result.eligible = true;
                                eligible_count.fetch_add(1);
                                result.circle = mandelbrot::shapes::fit_circle(
                                    component, options.circle);
                                if (result.circle.converged) {
                                    circle_converged_count.fetch_add(1);
                                }
                                if (result.circle.confident(options.circle)) {
                                    result.update = ClassificationUpdate{
                                        component.id,
                                        classification_from_circle_fit(
                                            result.circle, options.circle),
                                    };
                                    disk_count.fetch_add(1);
                                } else {
                                    cardioid_attempt_count.fetch_add(1);
                                    result.cardioid =
                                        mandelbrot::shapes::fit_cardioid_slanted(
                                            component, options.cardioid);
                                    if (result.cardioid.converged) {
                                        cardioid_converged_count.fetch_add(1);
                                    }
                                    if (result.cardioid.used_randomized_fallback) {
                                        shake_attempt_count.fetch_add(1);
                                    }
                                    if (result.cardioid
                                            .rescued_by_randomized_fallback) {
                                        shake_rescue_count.fetch_add(1);
                                    }
                                    if (result.cardioid.confident(options.cardioid)) {
                                        result.update = ClassificationUpdate{
                                            component.id,
                                            classification_from_cardioid_fit(
                                                result.circle,
                                                result.cardioid,
                                                options.cardioid),
                                        };
                                        cardioid_count.fetch_add(1);
                                    } else if (options.export_failed_cardioids
                                               && result.cardioid.converged) {
                                        FailedCardioidDiagnostic diagnostic;
                                        diagnostic.component_id = component.id;
                                        diagnostic.period = component.period;
                                        diagnostic.center = component.center;
                                        diagnostic.area = component.geometry.area_estimate;
                                        diagnostic.polygon_rho = component.geometry.polygon_rho;
                                        diagnostic.polygon = std::move(
                                            component.geometry.polygon);
                                        diagnostic.circle = result.circle;
                                        diagnostic.cardioid = result.cardioid;
                                        result.failed_cardioid = std::move(diagnostic);
                                    }
                                }
                            }
                        }
                        results[job] = std::move(result);
                    } catch (...) {
                        {
                            std::lock_guard lock(error_mutex);
                            if (!worker_error) {
                                worker_error = std::current_exception();
                            }
                        }
                        abort_workers.store(true);
                    }
                    completed.fetch_add(1);
                }
            });
        }
        for (auto& worker : workers) worker.join();
        monitor_stop.store(true);
        if (monitor.joinable()) monitor.join();

        {
            std::ostringstream detail;
            detail << "eligible=" << eligible_count.load()
                   << " disks=" << disk_count.load()
                   << " cardioid=" << cardioid_count.load()
                   << '/' << cardioid_attempt_count.load()
                   << " shake=" << shake_rescue_count.load()
                   << '/' << shake_attempt_count.load();
            fit_progress.update(
                completed.load(), detail.str(), true, true);
        }
        if (worker_error) std::rethrow_exception(worker_error);

        std::vector<ClassificationUpdate> updates;
        updates.reserve(disk_count.load() + cardioid_count.load());
        std::size_t eligible = 0;
        std::size_t circle_converged = 0;
        std::size_t cardioid_attempted = 0;
        std::size_t cardioid_converged = 0;
        std::size_t disks = 0;
        std::size_t cardioids = 0;
        std::size_t shake_attempted = 0;
        std::size_t shake_rescued = 0;
        CatalogueReal worst_disk_rms = 0;
        CatalogueReal worst_disk_max = 0;
        CatalogueReal worst_cardioid_rms = 0;
        CatalogueReal worst_cardioid_max = 0;
        std::optional<CatalogueReal> best_rejected_cardioid_rms;
        CatalogueReal minimum_xi = CatalogueReal("0.5");
        CatalogueReal maximum_xi = CatalogueReal("-0.5");
        std::vector<FailedCardioidDiagnostic> failed_diagnostics;
        failed_diagnostics.reserve(
            cardioid_attempt_count.load() - cardioid_count.load());
        for (auto& result : results) {
            if (!result.eligible) continue;
            ++eligible;
            if (result.circle.converged) ++circle_converged;
            const bool is_disk = result.update
                && result.update->classification.shape_class == "disk";
            const bool is_cardioid = result.update
                && result.update->classification.shape_class == "cardioid";
            if (!is_disk) {
                ++cardioid_attempted;
                if (result.cardioid.converged) ++cardioid_converged;
                if (result.cardioid.used_randomized_fallback) {
                    ++shake_attempted;
                }
                if (result.cardioid.rescued_by_randomized_fallback) {
                    ++shake_rescued;
                }
            }
            if (is_disk) {
                ++disks;
                worst_disk_rms = std::max(
                    worst_disk_rms, result.circle.fit.rms);
                worst_disk_max = std::max(
                    worst_disk_max, *result.circle.fit.max_error);
                updates.push_back(*result.update);
            } else if (is_cardioid) {
                ++cardioids;
                worst_cardioid_rms = std::max(
                    worst_cardioid_rms, result.cardioid.fit.rms);
                worst_cardioid_max = std::max(
                    worst_cardioid_max, *result.cardioid.fit.max_error);
                minimum_xi = std::min(minimum_xi, result.cardioid.fit.xi);
                maximum_xi = std::max(maximum_xi, result.cardioid.fit.xi);
                updates.push_back(*result.update);
            } else if (result.cardioid.converged) {
                if (!best_rejected_cardioid_rms
                    || result.cardioid.fit.rms
                        < *best_rejected_cardioid_rms) {
                    best_rejected_cardioid_rms = result.cardioid.fit.rms;
                }
                if (result.failed_cardioid) {
                    failed_diagnostics.push_back(
                        std::move(*result.failed_cardioid));
                }
            }
        }

        const std::size_t failed_diagnostic_count = failed_diagnostics.size();
        write_failed_cardioid_diagnostics(
            options, std::move(failed_diagnostics));

        std::size_t changed = 0;
        if (!options.dry_run && !updates.empty()) {
            // Each component is an independent atomically replaced JSON file.
            // Write distinct files concurrently, then bump the shared manifest
            // exactly once after every worker has committed successfully.
            InlineProgress write_progress(
                options, "writing classifications", updates.size() + 1);
            write_progress.update(
                0, "changed=0 | atomic component files", false, true);

            std::atomic<std::size_t> write_next{0};
            std::atomic<std::size_t> write_completed{0};
            std::atomic<std::size_t> write_changed{0};
            std::atomic<bool> abort_writes{false};
            std::exception_ptr write_error;
            std::mutex write_error_mutex;
            std::atomic<bool> write_monitor_stop{false};
            std::thread write_monitor;
            if (options.progress && options.progress_style == "bars") {
                write_monitor = std::thread([&] {
                    while (!write_monitor_stop.load()) {
                        write_progress.update(
                            write_completed.load(),
                            "changed=" + std::to_string(write_changed.load())
                                + " | atomic component files");
                        std::this_thread::sleep_for(std::chrono::milliseconds(
                            std::max(50, options.progress_refresh_ms)));
                    }
                });
            }

            const unsigned write_worker_count = std::min<unsigned>(
                options.threads,
                static_cast<unsigned>(
                    std::max<std::size_t>(1, updates.size())));
            std::vector<std::thread> write_workers;
            write_workers.reserve(write_worker_count);
            for (unsigned worker = 0; worker < write_worker_count; ++worker) {
                write_workers.emplace_back([&] {
                    while (!abort_writes.load()) {
                        const std::size_t index = write_next.fetch_add(1);
                        if (index >= updates.size()) break;
                        try {
                            if (catalogue.update_component_classification(
                                    updates[index].component_id,
                                    updates[index].classification,
                                    false)) {
                                write_changed.fetch_add(1);
                            }
                        } catch (...) {
                            {
                                std::lock_guard lock(write_error_mutex);
                                if (!write_error) {
                                    write_error = std::current_exception();
                                }
                            }
                            abort_writes.store(true);
                        }
                        write_completed.fetch_add(1);
                    }
                });
            }
            for (auto& worker : write_workers) worker.join();
            write_monitor_stop.store(true);
            if (write_monitor.joinable()) write_monitor.join();
            if (write_error) std::rethrow_exception(write_error);

            changed = write_changed.load();
            write_progress.update(
                write_completed.load(),
                "changed=" + std::to_string(changed)
                    + " | committing manifest",
                false,
                true);
            if (changed) {
                auto manifest = catalogue.load_manifest();
                ++manifest.catalogue_revision;
                manifest.updated_at = utc_timestamp();
                catalogue.save_manifest(manifest);
            }
            write_progress.update(
                updates.size() + 1,
                "changed=" + std::to_string(changed)
                    + " | manifest committed",
                true,
                true);
        }

        std::cout << "\nClassification summary\n"
                  << "  catalogue records scanned: " << component_ids.size() << '\n'
                  << "  eligible polygon records:  " << eligible << '\n'
                  << "  circle fits converged:      " << circle_converged << '\n'
                  << "  classified as disk:        " << disks << '\n'
                  << "  cardioid fits attempted:   " << cardioid_attempted << '\n'
                  << "  cardioid fits converged:   " << cardioid_converged << '\n'
                  << "  randomized shake attempted:" << ' ' << shake_attempted << '\n'
                  << "  rescued by final shake:    " << shake_rescued << '\n'
                  << "  classified as cardioid:    " << cardioids << '\n'
                  << "  left unknown:              "
                  << (eligible - disks - cardioids) << '\n';
        if (options.export_failed_cardioids) {
            std::cout << "  failed fits exported:      "
                      << failed_diagnostic_count << '\n';
        }
        std::cout << "  catalogue files changed:   " << changed
                  << (options.dry_run ? " (dry run)" : "") << '\n';
        if (disks) {
            std::cout << "  worst disk RMS:            "
                      << Catalogue::decimal_string(worst_disk_rms, 8) << '\n'
                      << "  worst disk max:            "
                      << Catalogue::decimal_string(worst_disk_max, 8) << '\n';
        }
        if (cardioids) {
            std::cout << "  worst cardioid RMS:        "
                      << Catalogue::decimal_string(worst_cardioid_rms, 8) << '\n'
                      << "  worst cardioid max:        "
                      << Catalogue::decimal_string(worst_cardioid_max, 8) << '\n'
                      << "  accepted xi range:         ["
                      << Catalogue::decimal_string(minimum_xi, 6) << ", "
                      << Catalogue::decimal_string(maximum_xi, 6) << "]\n";
        }
        if (best_rejected_cardioid_rms) {
            std::cout << "  best rejected cardioid RMS: "
                      << Catalogue::decimal_string(
                             *best_rejected_cardioid_rms, 8) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "classify_component_shapes: " << error.what() << '\n';
        return 1;
    }
}
