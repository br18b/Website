#include "component_catalogue.hpp"
#include "common/repo_config.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

using mandelbrot::catalogue::Catalogue;
using mandelbrot::catalogue::CatalogueReal;
using mandelbrot::repo::RepoConfig;
namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path config;
    fs::path catalogue_root;
    std::string command;
    std::string component_id;
    CatalogueReal area_cutoff = 0;
    int exact_through_period = 0;
    int min_period = 1;
    int max_period = std::numeric_limits<int>::max();
    fs::path output;
    bool help = false;
};

constexpr const char* kUsage = R"(Usage: catalogue_tool [options]

Options:
  --config PATH                 Alternate unified repository config
  --catalogue-root PATH         Override paths.catalogue_root
  --command COMMAND             init, list, show, verify, rebuild-indexes, or export-shape-index
  --component-id UUID           Component for --command show
  --area-cutoff VALUE           Cutoff for --command rebuild-indexes
  --exact-through-period N      Exhaustive period limit for manifest metadata
  --min-period N                First period for export-shape-index
  --max-period N                Last period for export-shape-index
  --output PATH                 Output CSV for export-shape-index
  -h, --help                    Show this help

Examples:
  catalogue_tool --command init
  catalogue_tool --command list
  catalogue_tool --command show --component-id COMPONENT_ID
  catalogue_tool --command verify
  catalogue_tool --command rebuild-indexes --area-cutoff 1e-10 --exact-through-period 14
  catalogue_tool --command export-shape-index --min-period 1 --max-period 16
)";

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string(name) + " requires a value\n" + kUsage);
            return argv[i];
        };
        if (arg == "--config") options.config = require_value("--config");
        else if (arg == "--catalogue-root") options.catalogue_root = require_value("--catalogue-root");
        else if (arg == "--command") options.command = require_value("--command");
        else if (arg == "--component-id") options.component_id = require_value("--component-id");
        else if (arg == "--area-cutoff") options.area_cutoff = Catalogue::parse_decimal(require_value("--area-cutoff"));
        else if (arg == "--exact-through-period") options.exact_through_period = std::stoi(require_value("--exact-through-period"));
        else if (arg == "--min-period") options.min_period = std::stoi(require_value("--min-period"));
        else if (arg == "--max-period") options.max_period = std::stoi(require_value("--max-period"));
        else if (arg == "--output") options.output = require_value("--output");
        else if (arg == "-h" || arg == "--help") options.help = true;
        else throw std::runtime_error("Unknown option: " + arg + "\n" + kUsage);
    }
    if (options.min_period < 1 || options.max_period < options.min_period) {
        throw std::runtime_error("Require 1 <= min-period <= max-period");
    }
    return options;
}

void atomic_replace(const fs::path& temporary, const fs::path& target) {
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

void render_progress(
    std::size_t current,
    std::size_t total,
    std::chrono::steady_clock::time_point started,
    std::size_t rows,
    bool final = false
) {
    bool terminal = true;
#if defined(__unix__) || defined(__APPLE__)
    terminal = isatty(STDERR_FILENO) != 0;
#endif
    if (!terminal && !final) return;

    constexpr int width = 30;
    const long double fraction = total > 0
        ? static_cast<long double>(current) / static_cast<long double>(total)
        : 1.0L;
    const int filled = std::clamp(
        static_cast<int>(fraction * width + 0.5L), 0, width);
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started).count();

    if (terminal) std::cerr << "\r\033[2K";
    std::cerr << "  shape index ["
              << std::string(static_cast<std::size_t>(filled), '#')
              << std::string(static_cast<std::size_t>(width - filled), '-')
              << "] " << std::fixed << std::setprecision(1)
              << static_cast<double>(100.0L * fraction) << "% "
              << current << '/' << total
              << " | rows=" << rows
              << " | elapsed=" << elapsed << 's';
    if (final) std::cerr << '\n';
    std::cerr.flush();
}

struct LightweightShapeRecord {
    int period = 0;
    int component_index = -1;
    std::string shape_class = "unknown";
};

std::optional<LightweightShapeRecord> load_shape_record(
    const Catalogue& catalogue,
    const std::string& component_id
) {
    const auto component = catalogue.load_component(component_id);
    const auto exact = Catalogue::exact_period_index(component);
    if (!exact) return std::nullopt;
    LightweightShapeRecord result;
    result.period = exact->period;
    result.component_index = exact->component_index;
    result.shape_class = component.classification.shape_class;
    return result;
}

void export_shape_index(
    const Catalogue& catalogue,
    int min_period,
    int max_period,
    fs::path output
) {
    if (output.empty()) {
        output = catalogue.exports_path() / "component_shape_classes.csv";
    } else if (output.is_relative()) {
        output = fs::absolute(output);
    }
    fs::create_directories(output.parent_path());

    std::vector<std::string> ids;
    for (const int period : catalogue.list_periods()) {
        if (period < min_period || period > max_period) continue;
        try {
            const auto index = catalogue.load_period(period);
            ids.insert(ids.end(), index.component_ids.begin(), index.component_ids.end());
        } catch (...) {
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    const auto manifest = catalogue.load_manifest();
    const fs::path temporary = output.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Could not write shape index: " + temporary.string());
    }
    stream << "schema,catalogue_revision,index_min_period,index_max_period,period,component_index,shape_class\n";

    const auto started = std::chrono::steady_clock::now();
    std::size_t written = 0;
    render_progress(0, ids.size(), started, written);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        try {
            const auto indexed = load_shape_record(catalogue, ids[i]);
            if (indexed
                && indexed->period >= min_period
                && indexed->period <= max_period) {
                stream << "mandelbrot-shape-index-v1,"
                       << manifest.catalogue_revision << ','
                       << min_period << ',' << max_period << ','
                       << indexed->period << ','
                       << indexed->component_index << ','
                       << indexed->shape_class << '\n';
                ++written;
            }
        } catch (...) {
        }
        if ((i + 1) % 64 == 0 || i + 1 == ids.size()) {
            render_progress(i + 1, ids.size(), started, written, i + 1 == ids.size());
        }
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed while writing shape index: " + temporary.string());
    }
    stream.close();
    atomic_replace(temporary, output);
    std::cout << "Shape-class index written: " << output
              << " (" << written << " row(s), catalogue revision "
              << manifest.catalogue_revision << ")\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            std::cout << kUsage;
            return 0;
        }
        if (options.command.empty()) throw std::runtime_error("--command is required\n" + std::string(kUsage));

        const RepoConfig config = RepoConfig::load(options.config, argv[0]);
        const fs::path catalogue_root = options.catalogue_root.empty()
            ? config.path("paths.catalogue_root")
            : fs::absolute(options.catalogue_root);
        Catalogue catalogue(catalogue_root);

        if (options.command == "init") {
            catalogue.ensure_layout();
            std::cout << "Initialized catalogue: " << catalogue.root() << '\n';
        } else if (options.command == "list") {
            for (const auto& id : catalogue.list_component_ids()) std::cout << id << '\n';
        } else if (options.command == "show") {
            if (options.component_id.empty()) throw std::runtime_error("--component-id is required for --command show");
            const auto component = catalogue.load_component(options.component_id);
            std::cout << "id: " << component.id << '\n'
                      << "period: " << component.period << '\n'
                      << "center: " << Catalogue::decimal_string(component.center.re) << ' '
                      << Catalogue::decimal_string(component.center.im) << '\n'
                      << "area: " << Catalogue::decimal_string(component.geometry.area_estimate) << '\n'
                      << "working digits: " << component.numeric.working_precision_digits << '\n'
                      << "validated digits: " << component.numeric.validated_digits << '\n'
                      << "shape: " << component.classification.shape_class << '\n';
        } else if (options.command == "verify") {
            catalogue.verify_integrity();
            std::cout << "SQLite catalogue integrity: OK\n"
                      << "database: " << catalogue.database_path() << '\n'
                      << "components: " << catalogue.list_component_ids().size()
                      << '\n';
        } else if (options.command == "rebuild-indexes") {
            catalogue.rebuild_period_indexes(options.area_cutoff);
            catalogue.rebuild_hierarchy_indexes(options.area_cutoff);
            catalogue.rebuild_manifest(options.exact_through_period, options.area_cutoff);
            std::cout << "Rebuilt period, hierarchy, and manifest indexes in "
                      << catalogue.root() << ".\n";
        } else if (options.command == "export-shape-index") {
            export_shape_index(
                catalogue,
                options.min_period,
                options.max_period,
                options.output);
        } else {
            throw std::runtime_error("Unknown command: " + options.command);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "catalogue_tool: " << error.what() << '\n';
        return 1;
    }
}
