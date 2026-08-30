#define main component_area_scan_embedded_main
#include "component_area_scan.cpp"
#undef main

#include <array>
#include <deque>
#include <memory>
#include <regex>
#include <unordered_set>
#include <boost/multiprecision/cpp_int.hpp>

namespace quadtree_hunter {

static volatile std::sig_atomic_t g_stop_requested = 0;

void quadtree_stop_handler(int) {
    g_stop_requested = 1;
}


bool stdout_is_terminal() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(STDOUT_FILENO) != 0;
#else
    return true;
#endif
}

std::string human_count(std::uint64_t value, int decimals = 1) {
    static constexpr std::array<const char*, 5> suffixes{{"", "k", "M", "G", "T"}};
    long double scaled = static_cast<long double>(value);
    std::size_t suffix = 0;
    while (scaled >= 1000.0L && suffix + 1 < suffixes.size()) {
        scaled /= 1000.0L;
        ++suffix;
    }
    std::ostringstream out;
    if (suffix == 0) {
        out << value;
    } else {
        int shown = decimals;
        if (scaled >= 100.0L) shown = 0;
        else if (scaled >= 10.0L) shown = std::min(shown, 1);
        out << std::fixed << std::setprecision(shown) << static_cast<double>(scaled)
            << suffixes[suffix];
    }
    return out.str();
}

class DepthDashboard {
public:
    DepthDashboard(int depth, int max_depth, std::size_t total_batches,
                   std::size_t chunk_cells, Clock::time_point started,
                   std::uint64_t depth_cells_before_chunk,
                   std::uint64_t depth_cells_total,
                   std::size_t chunk_index,
                   std::size_t chunk_count)
        : depth_(depth), max_depth_(max_depth),
          total_batches_(std::max<std::size_t>(1, total_batches)),
          chunk_cells_(chunk_cells), started_(started),
          depth_cells_before_chunk_(depth_cells_before_chunk),
          depth_cells_total_(std::max<std::uint64_t>(1, depth_cells_total)),
          chunk_index_(std::max<std::size_t>(1, chunk_index)),
          chunk_count_(std::max<std::size_t>(1, chunk_count)),
          terminal_(stdout_is_terminal()) {}

    void update(std::size_t batch_index, long double batch_fraction,
                std::size_t cells_done, const std::string& stage,
                std::size_t stage_done, std::size_t stage_total,
                std::uint64_t accepted) {
        render(batch_index, batch_fraction, cells_done, stage,
               stage_done, stage_total, accepted, false);
    }

    void finish(std::uint64_t accepted) {
        render(total_batches_ - 1, 1.0L, chunk_cells_, "complete",
               0, 0, accepted, true);
    }

private:
    static std::string compact_count(std::uint64_t value) {
        static constexpr const char* suffixes[] = {"", "k", "M", "G", "T"};
        long double scaled = static_cast<long double>(value);
        std::size_t suffix = 0;
        while (scaled >= 1000.0L && suffix + 1 < std::size(suffixes)) {
            scaled /= 1000.0L;
            ++suffix;
        }
        std::ostringstream out;
        if (suffix == 0) {
            out << value;
        } else if (scaled >= 100.0L) {
            out << std::fixed << std::setprecision(0) << static_cast<double>(scaled);
        } else if (scaled >= 10.0L) {
            out << std::fixed << std::setprecision(1) << static_cast<double>(scaled);
        } else {
            out << std::fixed << std::setprecision(2) << static_cast<double>(scaled);
        }
        out << suffixes[suffix];
        return out.str();
    }

    static std::string compact_duration(long double seconds) {
        if (!(seconds >= 0) || !std::isfinite(static_cast<double>(seconds))) return "--";
        const auto total = static_cast<std::uint64_t>(std::llround(seconds));
        const std::uint64_t days = total / 86400;
        const std::uint64_t hours = (total % 86400) / 3600;
        const std::uint64_t minutes = (total % 3600) / 60;
        const std::uint64_t secs = total % 60;
        std::ostringstream out;
        if (days > 0) out << days << 'd' << hours << 'h';
        else if (hours > 0) out << hours << 'h' << minutes << 'm';
        else if (minutes > 0) out << minutes << 'm' << secs << 's';
        else out << secs << 's';
        return out.str();
    }

    void render(std::size_t batch_index, long double batch_fraction,
                [[maybe_unused]] std::size_t cells_done, const std::string& stage,
                std::size_t stage_done, std::size_t stage_total,
                std::uint64_t accepted, bool force) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        const auto now = Clock::now();

        // Worker threads can report progress extremely frequently.  A terminal
        // dashboard only needs a few updates per second and otherwise wastes
        // CPU while producing unreadable wrapped output.
        if (terminal_ && !force && rendered_
            && now - last_render_ < std::chrono::milliseconds(500)) {
            return;
        }

        batch_fraction = std::clamp(batch_fraction, 0.0L, 1.0L);
        const long double chunk_fraction = std::clamp(
            (static_cast<long double>(batch_index) + batch_fraction)
                / static_cast<long double>(total_batches_),
            0.0L, 1.0L);
        const std::uint64_t completed_in_chunk = std::min<std::uint64_t>(
            static_cast<std::uint64_t>(std::llround(
                chunk_fraction * static_cast<long double>(chunk_cells_))),
            chunk_cells_);
        const std::uint64_t depth_done = std::min<std::uint64_t>(
            depth_cells_before_chunk_ + completed_in_chunk, depth_cells_total_);
        const long double depth_fraction = std::clamp(
            static_cast<long double>(depth_done)
                / static_cast<long double>(depth_cells_total_),
            0.0L, 1.0L);
        const std::uint64_t remaining = depth_cells_total_ - depth_done;

        const int percent = static_cast<int>(std::floor(depth_fraction * 100.0L));
        if (!terminal_) {
            if (!force && percent < next_log_percent_ && depth_fraction < 1.0L) return;
            next_log_percent_ = std::min(100, ((percent / 10) + 1) * 10);
        }

        const auto elapsed_duration = now - started_;
        const long double elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<long double>>(
                elapsed_duration).count();
        std::string eta = "--";
        if (completed_in_chunk > 0 && remaining > 0 && elapsed_seconds > 0) {
            const long double rate = static_cast<long double>(completed_in_chunk)
                                   / elapsed_seconds;
            eta = compact_duration(static_cast<long double>(remaining) / rate);
        } else if (remaining == 0) {
            eta = "0s";
        }

        constexpr int bar_width = 42;
        const int filled = std::clamp(
            static_cast<int>(std::llround(depth_fraction * bar_width)),
            0, bar_width);
        std::ostringstream bar;
        bar << '['
            << std::string(static_cast<std::size_t>(filled), '#')
            << std::string(static_cast<std::size_t>(bar_width - filled), '-')
            << "] " << std::fixed << std::setprecision(1)
            << static_cast<double>(100.0L * depth_fraction) << '%';

        const std::size_t shown_batch = std::min(
            batch_index + (batch_fraction >= 1.0L ? 1u : 0u), total_batches_);
        std::vector<std::string> lines;
        lines.reserve(9);
        lines.push_back("Mandelbrot adaptive quadtree hunter");
        {
            std::ostringstream line;
            line << "Depth " << depth_ << " / " << max_depth_
                 << "    Chunk " << chunk_index_ << " / " << chunk_count_
                 << "    Batch " << shown_batch << " / " << total_batches_;
            lines.push_back(line.str());
        }
        lines.push_back(bar.str());
        lines.push_back("Chunk       " + compact_count(completed_in_chunk)
                        + " / " + compact_count(chunk_cells_) + " cells");
        lines.push_back("Remaining   " + compact_count(remaining) + " cells");
        lines.push_back("Accepted    " + compact_count(accepted) + " this chunk");
        lines.push_back("Elapsed     " + compact_duration(elapsed_seconds)
                        + "    ETA " + eta);
        {
            std::ostringstream line;
            line << "Stage       " << stage;
            if (stage_total > 0) {
                line << ' ' << compact_count(stage_done)
                     << " / " << compact_count(stage_total);
            }
            lines.push_back(line.str());
        }
        lines.push_back("Ctrl+C finishes the current batch, checkpoints, and exits.");

        std::lock_guard<std::mutex> output_lock(g_print_mutex);
        if (terminal_) {
            if (rendered_) std::cout << "\033[" << rendered_lines_ << 'A';
            for (const auto& line : lines) {
                std::cout << "\r\033[2K" << line << '\n';
            }
            std::cout.flush();
            rendered_ = true;
            rendered_lines_ = lines.size();
            last_render_ = now;
        } else {
            std::cout << "depth " << depth_ << '/' << max_depth_
                      << " " << std::fixed << std::setprecision(1)
                      << static_cast<double>(100.0L * depth_fraction) << '%'
                      << " chunk " << chunk_index_ << '/' << chunk_count_
                      << " batch " << shown_batch << '/' << total_batches_
                      << " left " << compact_count(remaining)
                      << " accepted " << accepted
                      << " stage " << stage
                      << " elapsed " << compact_duration(elapsed_seconds)
                      << " eta " << eta << '\n';
        }
    }

    int depth_;
    int max_depth_;
    std::size_t total_batches_;
    std::size_t chunk_cells_;
    Clock::time_point started_;
    std::uint64_t depth_cells_before_chunk_;
    std::uint64_t depth_cells_total_;
    std::size_t chunk_index_;
    std::size_t chunk_count_;
    bool terminal_;
    bool rendered_ = false;
    std::size_t rendered_lines_ = 0;
    int next_log_percent_ = 0;
    Clock::time_point last_render_{};
    std::mutex state_mutex_;
};

boost::multiprecision::cpp_int quadtree_center_count_big(int n) {
    boost::multiprecision::cpp_int degree = 0;
    for (int d : divisors(n)) {
        const int mu = mobius(n / d);
        if (mu == 0) continue;
        boost::multiprecision::cpp_int term = boost::multiprecision::cpp_int(1);
        term <<= (d - 1);
        degree += mu * term;
    }
    if (degree <= 0) {
        throw std::runtime_error("Invalid exact-period center count.");
    }
    return degree;
}

struct QuadtreeConfig {
    int base_max_period = 14;
    int max_period = 50;
    int base_resolution = 128;
    int base_resolution_y = 0;
    int max_depth = 10;
    std::uint64_t max_cells = 4000000;
    std::size_t cell_batch_size = 20000;
    unsigned threads = 0;
    bool use_conjugate_symmetry = true;
    bool symmetry_active = false;

    Real xmin = -2.25L;
    Real xmax = 0.85L;
    Real ymin = -1.18L;
    Real ymax = 1.18L;

    int orbit_iterations = 1200;
    Real escape_radius = 64.0L;
    Real cycle_probe_tolerance = 1.0e-5L;
    Real cycle_refine_tolerance = 1.0e-13L;
    int period_candidates = 8;
    Real attracting_margin = 1.0e-10L;
    Real distance_prune_factor = 1.5L;
    Real inside_prune_factor = 1.15L;
    Real interior_margin_safety = 0.25L;

    Real min_area = 1.0e-10L;
    Real area_rho = 0.99999L;
    Real polygon_rho = 0.9995L;
    int polygon_points = 192;
    int coordinate_digits = 11;
    Real center_duplicate_tolerance = 1.0e-9L;
    Real screening_lower_bound_ratio = 1.0e-8L;

    int spatial_bins_x = 256;
    int spatial_bins_y = 192;

    fs::path catalogue_root;
    fs::path run_dir;
    std::string output_export = "atlas_components_quadtree.json";
    fs::path accepted_checkpoint_file;
    fs::path frontier_checkpoint_file;
};

struct Node {
    std::string id;
    int period = 0;
    Complex center{};
    Complex sample{};
    int discovery_depth = 0;
    Real area = 0;
    Real area_rho = 0;
    Real polygon_rho = 0;
    Real polygon_area = 0;
    std::vector<Complex> points;
};

struct GeometryResult {
    Real area = 0;
    Real polygon_area = 0;
    std::vector<Complex> points;
};

struct Cell {
    int depth = 0;
    std::uint32_t ix = 0;
    std::uint32_t iy = 0;
};

struct RefineCell {
    Cell cell;
    Real priority = 0;
};

QuadtreeConfig load_config(const fs::path& config_path, const Config& numerical,
                           const char* argv0) {
    const auto repo = mandelbrot::repo::RepoConfig::load(config_path, executable_parent_or_cwd(argv0));
    QuadtreeConfig cfg;
    cfg.catalogue_root = repo.path("paths.catalogue_root");
    mandelbrot::catalogue::Catalogue catalogue(cfg.catalogue_root);
    const std::string run_name = repo.string("component_quadtree_hunter.run_name", "default");
    cfg.run_dir = catalogue.run_path("quadtree_hunter", run_name);
    cfg.threads = repo.threads();
    cfg.base_max_period = repo.integer("component_area_scan.period", cfg.base_max_period);
    cfg.polygon_rho = static_cast<Real>(repo.number("component_area_scan.atlas_polygon_rho", cfg.polygon_rho));
    cfg.polygon_points = repo.integer("component_area_scan.atlas_polygon_points", cfg.polygon_points);
    cfg.xmin = static_cast<Real>(repo.number("demo.atlas.view.xmin", cfg.xmin));
    cfg.xmax = static_cast<Real>(repo.number("demo.atlas.view.xmax", cfg.xmax));
    cfg.ymin = static_cast<Real>(repo.number("demo.atlas.view.ymin", cfg.ymin));
    cfg.ymax = static_cast<Real>(repo.number("demo.atlas.view.ymax", cfg.ymax));
    const std::string prefix = "component_quadtree_hunter.";
    cfg.min_area = static_cast<Real>(repo.number(prefix + "min_area", cfg.min_area));
    cfg.xmin = static_cast<Real>(repo.number(prefix + "xmin", cfg.xmin));
    cfg.xmax = static_cast<Real>(repo.number(prefix + "xmax", cfg.xmax));
    cfg.ymin = static_cast<Real>(repo.number(prefix + "ymin", cfg.ymin));
    cfg.ymax = static_cast<Real>(repo.number(prefix + "ymax", cfg.ymax));
    cfg.max_period = repo.integer(prefix + "max_period", cfg.max_period);
    cfg.base_resolution = repo.integer(prefix + "base_resolution", cfg.base_resolution);
    cfg.base_resolution_y = repo.integer(prefix + "base_resolution_y", cfg.base_resolution_y);
    cfg.max_depth = repo.integer(prefix + "max_depth", cfg.max_depth);
    cfg.max_cells = repo.u64(prefix + "max_cells", cfg.max_cells);
    cfg.cell_batch_size = static_cast<std::size_t>(repo.u64(prefix + "cell_batch_size", cfg.cell_batch_size));
    cfg.use_conjugate_symmetry = repo.boolean(prefix + "use_conjugate_symmetry", cfg.use_conjugate_symmetry);
    cfg.orbit_iterations = repo.integer(prefix + "orbit_iterations", cfg.orbit_iterations);
    cfg.escape_radius = static_cast<Real>(repo.number(prefix + "escape_radius", cfg.escape_radius));
    cfg.cycle_probe_tolerance = static_cast<Real>(repo.number(prefix + "cycle_probe_tolerance", cfg.cycle_probe_tolerance));
    cfg.cycle_refine_tolerance = static_cast<Real>(repo.number(prefix + "cycle_refine_tolerance", cfg.cycle_refine_tolerance));
    cfg.period_candidates = repo.integer(prefix + "period_candidates", cfg.period_candidates);
    cfg.attracting_margin = static_cast<Real>(repo.number(prefix + "attracting_margin", cfg.attracting_margin));
    cfg.distance_prune_factor = static_cast<Real>(repo.number(prefix + "distance_prune_factor", cfg.distance_prune_factor));
    cfg.inside_prune_factor = static_cast<Real>(repo.number(prefix + "inside_prune_factor", cfg.inside_prune_factor));
    cfg.interior_margin_safety = static_cast<Real>(repo.number(prefix + "interior_margin_safety", cfg.interior_margin_safety));
    cfg.area_rho = static_cast<Real>(repo.number(prefix + "area_rho", cfg.area_rho));
    cfg.polygon_rho = static_cast<Real>(repo.number(prefix + "polygon_rho", cfg.polygon_rho));
    cfg.polygon_points = repo.integer(prefix + "polygon_points", cfg.polygon_points);
    cfg.center_duplicate_tolerance = static_cast<Real>(repo.number(prefix + "center_duplicate_tolerance", cfg.center_duplicate_tolerance));
    cfg.screening_lower_bound_ratio = static_cast<Real>(repo.number(prefix + "screening_lower_bound_ratio", cfg.screening_lower_bound_ratio));
    cfg.spatial_bins_x = repo.integer(prefix + "spatial_bins_x", cfg.spatial_bins_x);
    cfg.spatial_bins_y = repo.integer(prefix + "spatial_bins_y", cfg.spatial_bins_y);
    cfg.coordinate_digits = repo.integer(prefix + "coordinate_digits", cfg.coordinate_digits);
    cfg.output_export = repo.string(prefix + "output_export", cfg.output_export);
    cfg.accepted_checkpoint_file = catalogue.run_path("quadtree_hunter", run_name, "accepted.tsv");
    cfg.frontier_checkpoint_file = catalogue.run_path("quadtree_hunter", run_name, "frontier.tsv");
    fs::create_directories(cfg.run_dir);
    cfg.base_resolution = std::max(4, cfg.base_resolution);
    const Real symmetry_scale = std::max<Real>(1, std::max(std::abs(cfg.ymin), std::abs(cfg.ymax)));
    if (cfg.use_conjugate_symmetry && cfg.ymin < 0 && cfg.ymax > 0
        && std::abs(cfg.ymax + cfg.ymin) <= 1.0e-12L * symmetry_scale) {
        cfg.ymin = 0;
        cfg.symmetry_active = true;
    }
    if (cfg.base_resolution_y <= 0) {
        const Real aspect = (cfg.ymax - cfg.ymin) / (cfg.xmax - cfg.xmin);
        cfg.base_resolution_y = std::max(4, static_cast<int>(std::llround(cfg.base_resolution * aspect)));
    }
    cfg.max_depth = std::clamp(cfg.max_depth, 0, 20);
    cfg.max_period = std::max(1, cfg.max_period);
    cfg.orbit_iterations = std::max(cfg.max_period + 8, cfg.orbit_iterations);
    cfg.period_candidates = std::max(1, cfg.period_candidates);
    cfg.polygon_points = std::max(32, cfg.polygon_points);
    cfg.coordinate_digits = std::clamp(cfg.coordinate_digits, 8, 18);
    cfg.cell_batch_size = std::max<std::size_t>(100, cfg.cell_batch_size);
    cfg.spatial_bins_x = std::max(8, cfg.spatial_bins_x);
    cfg.spatial_bins_y = std::max(8, cfg.spatial_bins_y);
    if (!(cfg.xmin < cfg.xmax && cfg.ymin < cfg.ymax)) {
        throw std::runtime_error("Invalid quadtree search bounds.");
    }
    if (!(cfg.min_area > 0 && cfg.area_rho > 0 && cfg.area_rho < 1
          && cfg.polygon_rho > 0 && cfg.polygon_rho < 1 && cfg.escape_radius > 2)) {
        throw std::runtime_error("Invalid component_quadtree_hunter numerical configuration.");
    }
    (void)numerical;
    return cfg;
}

// -----------------------------------------------------------------------------
// Component geometry and output
// -----------------------------------------------------------------------------

std::optional<GeometryResult> trace_geometry(Complex center, int period,
                                             const QuadtreeConfig& cfg,
                                             const Config& numerical) {
    auto state = center_state(period, center, numerical);
    if (!state) return std::nullopt;

    std::vector<Real> stages{0.9L, 0.99L, 0.999L, cfg.polygon_rho, cfg.area_rho};
    std::sort(stages.begin(), stages.end());
    stages.erase(std::unique(stages.begin(), stages.end(), [](Real a, Real b) {
        return std::abs(a - b) < 1.0e-18L;
    }), stages.end());

    for (int multiplier : {1, 2, 4}) {
        const int theta = cfg.polygon_points * multiplier;
        std::optional<RingTrace> seed;
        std::optional<RingTrace> polygon_ring;
        std::optional<RingTrace> area_ring;
        bool ok = true;
        for (Real rho : stages) {
            auto ring = trace_ring(period, *state, rho, theta, numerical,
                                   seed ? &*seed : nullptr);
            if (!ring) { ok = false; break; }
            if (std::abs(rho - cfg.polygon_rho) < 1.0e-18L) polygon_ring = *ring;
            if (std::abs(rho - cfg.area_rho) < 1.0e-18L) area_ring = *ring;
            seed = std::move(*ring);
        }
        if (!ok || !polygon_ring || !area_ring) continue;

        std::vector<Complex> offsets(theta);
        for (int i = 0; i < theta; ++i) offsets[i] = area_ring->c[i] - center;
        const Real area = std::abs(derivative_area(offsets, area_ring->lambda,
                                                   area_ring->c_lambda));
        std::vector<Complex> coarse_offsets, coarse_lambda, coarse_derivative;
        for (int i = 0; i < theta; i += 2) {
            coarse_offsets.push_back(offsets[i]);
            coarse_lambda.push_back(area_ring->lambda[i]);
            coarse_derivative.push_back(area_ring->c_lambda[i]);
        }
        const Real coarse = std::abs(derivative_area(
            coarse_offsets, coarse_lambda, coarse_derivative));
        const Real error = std::abs(area - coarse);
        const Real requested = std::max<Real>(
            1.0e-14L, 1.0e-5L * std::max(area, cfg.min_area));
        if (multiplier < 4
            && (error > requested || (area > 0.5L * cfg.min_area
                                      && area < 2 * cfg.min_area))) {
            continue;
        }

        std::vector<Complex> points;
        points.reserve(cfg.polygon_points);
        for (int i = 0; i < cfg.polygon_points; ++i) {
            points.push_back(polygon_ring->c[static_cast<std::size_t>(i * multiplier)]);
        }
        if (polygon_area(points) < 0) std::reverse(points.begin(), points.end());
        return GeometryResult{area, std::abs(polygon_area(points)), std::move(points)};
    }
    return std::nullopt;
}

std::array<Real, 4> bbox(const std::vector<Complex>& points) {
    if (points.empty()) return {0, 0, 0, 0};
    Real xmin = points.front().real(), xmax = xmin;
    Real ymin = points.front().imag(), ymax = ymin;
    for (const auto& point : points) {
        xmin = std::min(xmin, point.real());
        xmax = std::max(xmax, point.real());
        ymin = std::min(ymin, point.imag());
        ymax = std::max(ymax, point.imag());
    }
    return {xmin, xmax, ymin, ymax};
}

std::string encode_points(const std::vector<Complex>& points) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Real>::max_digits10) << std::scientific;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i) out << ';';
        out << points[i].real() << ':' << points[i].imag();
    }
    return out.str();
}

std::vector<Complex> decode_points(const std::string& text) {
    std::vector<Complex> result;
    for (const auto& pair : split(text, ';')) {
        const auto colon = pair.find(':');
        if (colon == std::string::npos) continue;
        try {
            result.emplace_back(std::stold(pair.substr(0, colon)),
                                std::stold(pair.substr(colon + 1)));
        } catch (...) {}
    }
    return result;
}

mandelbrot::catalogue::CatalogueReal catalogue_real(Real value) {
    return mandelbrot::catalogue::Catalogue::parse_decimal(real_string(value));
}

mandelbrot::catalogue::ComponentRecord node_to_component(const Node& node) {
    using namespace mandelbrot::catalogue;
    ComponentRecord component;
    component.id = node.id;
    component.period = node.period;
    component.center = {
        catalogue_real(node.center.real()), catalogue_real(node.center.imag())};
    component.numeric.working_precision_digits = std::numeric_limits<Real>::max_digits10;
    component.numeric.validated_digits = std::numeric_limits<Real>::digits10;
    component.geometry.coordinate_frame = "centered";
    component.geometry.polygon_rho = catalogue_real(node.polygon_rho);
    component.geometry.area_rho = catalogue_real(node.area_rho);
    component.geometry.polygon_area = catalogue_real(node.polygon_area);
    component.geometry.area_estimate = catalogue_real(node.area);
    component.geometry.area_error = 0;
    component.geometry.characteristic_size = catalogue_real(
        std::sqrt(std::max<Real>(node.area, 0) / PI));
    component.geometry.polygon.reserve(node.points.size());
    std::array<Real, 4> bounds{
        std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity()};
    for (const auto& point : node.points) {
        const Real x = point.real() - node.center.real();
        const Real y = point.imag() - node.center.imag();
        component.geometry.polygon.push_back({catalogue_real(x), catalogue_real(y)});
        bounds[0] = std::min(bounds[0], x); bounds[1] = std::max(bounds[1], x);
        bounds[2] = std::min(bounds[2], y); bounds[3] = std::max(bounds[3], y);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        component.geometry.bbox_centered[i] = catalogue_real(bounds[i]);
    }
    component.classification.shape_class = "unknown";
    component.classification.shape_confidence = 0;
    component.hierarchy.hierarchy_root = component.id;
    component.hierarchy.generation = 0;
    component.provenance.method = "quadtree-hunter";
    component.provenance.run_id = "default";
    component.provenance.discovered_at = utc_timestamp();
    component.provenance.aliases = {
        "quadtree-depth:" + std::to_string(node.discovery_depth),
        "quadtree-sample:" + real_string(node.sample.real()) + ":"
            + real_string(node.sample.imag())};
    component.quality.center_validated = true;
    component.quality.exact_period_validated = true;
    component.quality.polygon_converged = true;
    component.quality.area_above_cutoff = true;
    return component;
}

bool save_nodes_to_catalogue(
    mandelbrot::catalogue::Catalogue& catalogue,
    std::vector<Node>& nodes,
    const QuadtreeConfig& cfg
) {
    using namespace mandelbrot::catalogue;
    if (nodes.empty()) return false;
    std::vector<ComponentRecord> records;
    records.reserve(nodes.size());
    for (const auto& node : nodes) records.push_back(node_to_component(node));
    UpsertOptions upsert;
    upsert.center_tolerance = CatalogueReal(cfg.center_duplicate_tolerance);
    upsert.bump_revision = true;
    const auto results = catalogue.upsert_components(std::move(records), upsert);
    std::set<int> changed_periods;
    bool changed = false;
    for (std::size_t i = 0; i < results.size(); ++i) {
        nodes[i].id = results[i].component.id;
        nodes[i].center = {
            results[i].component.center.re.convert_to<Real>(),
            results[i].component.center.im.convert_to<Real>()};
        if (results[i].inserted || results[i].updated) {
            changed_periods.insert(results[i].component.period);
            changed = true;
        }
    }
    if (!changed_periods.empty()) {
        catalogue.rebuild_period_indexes(
            std::vector<int>(changed_periods.begin(), changed_periods.end()),
            CatalogueReal(cfg.min_area));
        catalogue.rebuild_manifest_from_period_indexes(
            cfg.base_max_period, CatalogueReal(cfg.min_area));
    }
    return changed;
}

void write_quadtree_export(
    mandelbrot::catalogue::Catalogue& catalogue,
    const QuadtreeConfig& cfg
) {
    using namespace mandelbrot::catalogue;
    ComponentExportOptions options;
    options.complete = false;
    options.coordinate_digits = cfg.coordinate_digits;
    options.query.min_area = CatalogueReal(cfg.min_area);
    options.query.require_polygon = true;
    options.query.require_polygon_converged = true;
    options.query.provenance_method = "quadtree-hunter";
    catalogue.write_component_export(catalogue.export_path(cfg.output_export), options);
}

void ensure_accepted_checkpoint(const fs::path& path) {
    if (fs::exists(path)) return;
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "id\tperiod\tcenter_re\tcenter_im\tsample_re\tsample_im\tdepth\tarea\tarea_rho\tpolygon_rho\tpolygon_area\tpoints\n";
}

void append_accepted_checkpoint(std::ostream& out, const Node& node) {
    out << node.id << '\t' << node.period
        << '\t' << real_string(node.center.real()) << '\t' << real_string(node.center.imag())
        << '\t' << real_string(node.sample.real()) << '\t' << real_string(node.sample.imag())
        << '\t' << node.discovery_depth
        << '\t' << real_string(node.area) << '\t' << real_string(node.area_rho)
        << '\t' << real_string(node.polygon_rho) << '\t' << real_string(node.polygon_area)
        << '\t' << encode_points(node.points) << '\n';
}

std::vector<Node> load_accepted_checkpoint(const fs::path& path) {
    std::vector<Node> result;
    std::ifstream in(path);
    if (!in) return result;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const auto fields = split(line, '\t');
        if (fields.size() < 12) continue;
        try {
            Node node;
            node.id = fields[0];
            node.period = std::stoi(fields[1]);
            node.center = {std::stold(fields[2]), std::stold(fields[3])};
            node.sample = {std::stold(fields[4]), std::stold(fields[5])};
            node.discovery_depth = std::stoi(fields[6]);
            node.area = std::stold(fields[7]);
            node.area_rho = std::stold(fields[8]);
            node.polygon_rho = std::stold(fields[9]);
            node.polygon_area = std::stold(fields[10]);
            node.points = decode_points(fields[11]);
            if (!node.points.empty()) result.push_back(std::move(node));
        } catch (...) {}
    }
    return result;
}

// -----------------------------------------------------------------------------
// Known component polygons and a small spatial index
// -----------------------------------------------------------------------------

struct KnownPolygon {
    std::string id;
    int period = 0;
    Complex center{};
    Real area = 0;
    std::array<Real, 4> bounds{};
    std::vector<Complex> points;
};

KnownPolygon polygon_from_component(
    const mandelbrot::catalogue::ComponentRecord& component
) {
    KnownPolygon polygon;
    polygon.id = component.id;
    polygon.period = component.period;
    polygon.center = {
        component.center.re.convert_to<Real>(),
        component.center.im.convert_to<Real>()};
    polygon.area = component.geometry.area_estimate.convert_to<Real>();
    polygon.points.reserve(component.geometry.polygon.size());
    for (const auto& offset : component.geometry.polygon) {
        polygon.points.emplace_back(
            polygon.center.real() + offset.re.convert_to<Real>(),
            polygon.center.imag() + offset.im.convert_to<Real>());
    }
    polygon.bounds = bbox(polygon.points);
    return polygon;
}

std::vector<KnownPolygon> load_known_geometry(
    const mandelbrot::catalogue::CatalogueSnapshot& snapshot
) {
    std::vector<KnownPolygon> result;
    result.reserve(snapshot.components.size());
    for (const auto& component : snapshot.components) {
        if (component.geometry.polygon.size() >= 3) {
            result.push_back(polygon_from_component(component));
        }
    }
    return result;
}

bool point_in_polygon(Complex point, const KnownPolygon& polygon) {
    const Real x = point.real(), y = point.imag();
    if (x < polygon.bounds[0] || x > polygon.bounds[1]
        || y < polygon.bounds[2] || y > polygon.bounds[3]) return false;
    bool inside = false;
    const auto& values = polygon.points;
    for (std::size_t i = 0, j = values.size() - 1; i < values.size(); j = i++) {
        const Real xi = values[i].real(), yi = values[i].imag();
        const Real xj = values[j].real(), yj = values[j].imag();
        const bool crosses = ((yi > y) != (yj > y));
        if (!crosses) continue;
        const Real denominator = yj - yi;
        if (std::abs(denominator) < 1.0e-30L) continue;
        const Real crossing_x = (xj - xi) * (y - yi) / denominator + xi;
        if (x < crossing_x) inside = !inside;
    }
    return inside;
}

Real segment_distance(Complex point, Complex a, Complex b) {
    const Complex edge = b - a;
    const Real norm = std::norm(edge);
    if (norm <= 1.0e-36L) return safe_abs(point - a);
    const Real t = std::clamp(((point - a) * std::conj(edge)).real() / norm,
                              static_cast<Real>(0), static_cast<Real>(1));
    return safe_abs(point - (a + t * edge));
}

Real polygon_edge_distance(Complex point, const KnownPolygon& polygon) {
    Real result = std::numeric_limits<Real>::infinity();
    for (std::size_t i = 0; i < polygon.points.size(); ++i) {
        result = std::min(result, segment_distance(
            point, polygon.points[i], polygon.points[(i + 1) % polygon.points.size()]));
    }
    return result;
}

struct PolygonHit {
    std::string id;
    int period = 0;
    Complex center{};
    Real margin = 0;
};

class PolygonIndex {
public:
    PolygonIndex(const QuadtreeConfig& cfg, const std::vector<KnownPolygon>& polygons)
        : xmin_(cfg.xmin), xmax_(cfg.xmax), ymin_(cfg.ymin), ymax_(cfg.ymax),
          nx_(cfg.spatial_bins_x), ny_(cfg.spatial_bins_y), polygons_(&polygons),
          bins_(static_cast<std::size_t>(nx_) * ny_) {
        const Real dx = (xmax_ - xmin_) / nx_;
        const Real dy = (ymax_ - ymin_) / ny_;
        for (std::size_t index = 0; index < polygons.size(); ++index) {
            const auto& bounds = polygons[index].bounds;
            if (bounds[1] < xmin_ || bounds[0] > xmax_
                || bounds[3] < ymin_ || bounds[2] > ymax_) continue;
            const int x0 = std::clamp(static_cast<int>(std::floor((bounds[0] - xmin_) / dx)), 0, nx_ - 1);
            const int x1 = std::clamp(static_cast<int>(std::floor((bounds[1] - xmin_) / dx)), 0, nx_ - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor((bounds[2] - ymin_) / dy)), 0, ny_ - 1);
            const int y1 = std::clamp(static_cast<int>(std::floor((bounds[3] - ymin_) / dy)), 0, ny_ - 1);
            for (int iy = y0; iy <= y1; ++iy) {
                for (int ix = x0; ix <= x1; ++ix) {
                    bins_[static_cast<std::size_t>(iy) * nx_ + ix].push_back(index);
                }
            }
        }
    }

    std::optional<PolygonHit> query(Complex point) const {
        if (point.real() < xmin_ || point.real() > xmax_
            || point.imag() < ymin_ || point.imag() > ymax_) return std::nullopt;
        const int ix = std::clamp(static_cast<int>((point.real() - xmin_) / (xmax_ - xmin_) * nx_), 0, nx_ - 1);
        const int iy = std::clamp(static_cast<int>((point.imag() - ymin_) / (ymax_ - ymin_) * ny_), 0, ny_ - 1);
        std::optional<PolygonHit> best;
        for (std::size_t polygon_index : bins_[static_cast<std::size_t>(iy) * nx_ + ix]) {
            const auto& polygon = (*polygons_)[polygon_index];
            if (!point_in_polygon(point, polygon)) continue;
            const Real margin = polygon_edge_distance(point, polygon);
            if (!best || margin > best->margin) {
                best = PolygonHit{polygon.id, polygon.period, polygon.center, margin};
            }
        }
        return best;
    }

private:
    Real xmin_, xmax_, ymin_, ymax_;
    int nx_, ny_;
    const std::vector<KnownPolygon>* polygons_;
    std::vector<std::vector<std::size_t>> bins_;
};

KnownPolygon polygon_from_node(const Node& node) {
    KnownPolygon polygon;
    polygon.id = node.id;
    polygon.period = node.period;
    polygon.center = node.center;
    polygon.area = node.area;
    polygon.points = node.points;
    polygon.bounds = bbox(polygon.points);
    return polygon;
}

// -----------------------------------------------------------------------------
// Interior detection and continuation to the component center
// -----------------------------------------------------------------------------

enum class SampleKind { Escaped, Component, Unresolved };

struct SampleResult {
    SampleKind kind = SampleKind::Unresolved;
    int period = 0;
    Complex center{};
    Real distance = 0;
    Real interior_margin = 0;
    Real lambda_abs = 0;
};

std::optional<MultiplierState> refine_cycle_at_parameter(
    Complex c, Complex z_guess, int period,
    const QuadtreeConfig& cfg, const Config& numerical
) {
    Complex z = z_guess;
    for (int iteration = 0; iteration < 80; ++iteration) {
        const auto data = iterate_data(z, c, period);
        const Complex residual = data.A - z;
        const Real scale = std::max<Real>(1, safe_abs(z));
        if (safe_abs(residual) <= cfg.cycle_refine_tolerance * scale) {
            if (!periodic_point_has_exact_period(
                    period, z, c, numerical.exact_period_tolerance)) return std::nullopt;
            const Real lambda_abs = safe_abs(data.B);
            if (!(lambda_abs < 1 - cfg.attracting_margin)) return std::nullopt;
            return state_from_solution(period, data.B, z, c, iteration);
        }
        const Complex derivative = data.B - Complex{1, 0};
        if (!finite(derivative) || safe_abs(derivative) < 1.0e-24L) return std::nullopt;
        Complex correction = residual / derivative;
        const Real correction_abs = safe_abs(correction);
        if (correction_abs > 1) correction /= correction_abs;
        z -= correction;
        if (!finite(z)) return std::nullopt;
    }
    return std::nullopt;
}

SampleResult classify_sample(Complex c, const QuadtreeConfig& cfg,
                             const Config& numerical, const PolygonIndex& index) {
    if (auto hit = index.query(c)) {
        return SampleResult{SampleKind::Component, hit->period, hit->center,
                            0, hit->margin, 0};
    }

    const int ring_size = cfg.max_period + 1;
    std::vector<Complex> ring(static_cast<std::size_t>(ring_size));
    Complex z{0, 0};
    Complex derivative{0, 0};
    int completed = 0;
    const Real escape_squared = cfg.escape_radius * cfg.escape_radius;
    for (int iteration = 1; iteration <= cfg.orbit_iterations; ++iteration) {
        derivative = 2.0L * z * derivative + Complex{1, 0};
        z = z * z + c;
        ring[static_cast<std::size_t>(iteration % ring_size)] = z;
        completed = iteration;
        if (!finite(z) || std::norm(z) > escape_squared) {
            const Real abs_z = safe_abs(z);
            const Real abs_derivative = safe_abs(derivative);
            Real distance = 0;
            if (finite(abs_z) && finite(abs_derivative)
                && abs_z > 1 && abs_derivative > 1.0e-30L) {
                distance = abs_z * std::log(abs_z) / abs_derivative;
            }
            if (!finite(distance) || distance < 0) distance = 0;
            return SampleResult{SampleKind::Escaped, 0, {}, distance, 0, 0};
        }
    }

    struct PeriodCandidate { Real score; int period; };
    std::vector<PeriodCandidate> candidates;
    candidates.reserve(cfg.max_period);
    const Complex current = ring[static_cast<std::size_t>(completed % ring_size)];
    const Real scale = std::max<Real>(1, safe_abs(current));
    for (int period = 1; period <= cfg.max_period && period < completed; ++period) {
        const Complex previous = ring[static_cast<std::size_t>((completed - period) % ring_size)];
        candidates.push_back({safe_abs(current - previous) / scale, period});
    }
    // Once the orbit has settled, every multiple of the exact period also has
    // a small return. Prefer the smallest plausible periods rather than merely
    // the numerically smallest return; otherwise a fixed point can be hidden by
    // p=17, p=31, ... roundoff ties and never reach the candidate list.
    std::vector<PeriodCandidate> plausible;
    for (const auto& candidate : candidates) {
        if (candidate.score <= cfg.cycle_probe_tolerance) plausible.push_back(candidate);
    }
    if (!plausible.empty()) {
        std::sort(plausible.begin(), plausible.end(), [](const auto& a, const auto& b) {
            return a.period < b.period;
        });
    } else {
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score < b.score;
            return a.period < b.period;
        });
        plausible = candidates;
    }

    const int attempts = std::min<int>(cfg.period_candidates, plausible.size());
    for (int i = 0; i < attempts; ++i) {
        const auto candidate = plausible[static_cast<std::size_t>(i)];
        if (candidate.score > cfg.cycle_probe_tolerance) break;
        auto state = refine_cycle_at_parameter(c, current, candidate.period, cfg, numerical);
        if (!state) continue;
        ContinuationStats stats;
        auto centered = continue_to_lambda(
            candidate.period, *state, Complex{0, 0}, numerical, stats);
        if (!centered) continue;
        if (detected_center_period(centered->c, candidate.period,
                                   numerical.exact_period_tolerance) != candidate.period) {
            continue;
        }
        const Real lambda_abs = safe_abs(state->lambda);
        const Real margin = cfg.interior_margin_safety
            * std::max<Real>(0, 1 - lambda_abs) * safe_abs(state->c_lambda);
        return SampleResult{SampleKind::Component, candidate.period, centered->c,
                            0, margin, lambda_abs};
    }
    return SampleResult{};
}

bool near_center(Complex center, int period, const std::vector<KnownPolygon>& polygons,
                 Real tolerance) {
    for (const auto& polygon : polygons) {
        if (polygon.period == period && safe_abs(center - polygon.center) <= tolerance) return true;
    }
    return false;
}

bool near_center(Complex center, int period,
                 const std::vector<std::pair<int, Complex>>& centers,
                 Real tolerance) {
    for (const auto& [known_period, known_center] : centers) {
        if (known_period == period && safe_abs(center - known_center) <= tolerance) return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Frontier checkpoint
// -----------------------------------------------------------------------------

std::string signature(const QuadtreeConfig& cfg) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << cfg.base_resolution << ',' << cfg.base_resolution_y << ',' << cfg.max_depth << ','
        << cfg.max_period << ',' << cfg.symmetry_active << ','
        << cfg.xmin << ',' << cfg.xmax << ',' << cfg.ymin << ','
        << cfg.ymax << ',' << cfg.orbit_iterations << ',' << cfg.min_area;
    return out.str();
}

fs::path deferred_directory(const QuadtreeConfig& cfg) {
    return fs::path(cfg.frontier_checkpoint_file.string() + ".deferred");
}

fs::path deferred_file(const QuadtreeConfig& cfg, int depth) {
    return deferred_directory(cfg) / ("depth_" + std::to_string(depth) + ".bin");
}

void clear_deferred_frontiers(const QuadtreeConfig& cfg) {
    std::error_code error;
    fs::remove_all(deferred_directory(cfg), error);
}

void append_deferred_cells(const QuadtreeConfig& cfg, int depth,
                           const std::vector<Cell>& cells) {
    if (cells.empty()) return;
    fs::create_directories(deferred_directory(cfg));
    std::ofstream out(deferred_file(cfg, depth), std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("Could not append deferred frontier for depth "
                                      + std::to_string(depth));
    for (const auto& cell : cells) {
        out.write(reinterpret_cast<const char*>(&cell.ix), sizeof(cell.ix));
        out.write(reinterpret_cast<const char*>(&cell.iy), sizeof(cell.iy));
    }
    if (!out) throw std::runtime_error("Could not write deferred frontier for depth "
                                      + std::to_string(depth));
}

std::uint64_t deferred_record_count(const QuadtreeConfig& cfg, int depth) {
    std::error_code error;
    const auto bytes = fs::file_size(deferred_file(cfg, depth), error);
    if (error) return 0;
    return bytes / (sizeof(std::uint32_t) * 2);
}

std::vector<Cell> load_deferred_chunk(const QuadtreeConfig& cfg, int depth,
                                      std::uint64_t& offset_records) {
    std::vector<Cell> cells;
    const auto path = deferred_file(cfg, depth);
    std::ifstream in(path, std::ios::binary);
    if (!in) return cells;
    in.seekg(static_cast<std::streamoff>(offset_records * sizeof(std::uint32_t) * 2));
    if (!in) return cells;
    const std::uint64_t total = deferred_record_count(cfg, depth);
    if (offset_records >= total) return cells;
    const std::uint64_t available = total - offset_records;
    const std::uint64_t wanted = std::min<std::uint64_t>(cfg.max_cells, available);
    cells.reserve(static_cast<std::size_t>(wanted));
    for (std::uint64_t i = 0; i < wanted; ++i) {
        std::uint32_t ix = 0, iy = 0;
        in.read(reinterpret_cast<char*>(&ix), sizeof(ix));
        in.read(reinterpret_cast<char*>(&iy), sizeof(iy));
        if (!in) break;
        cells.push_back(Cell{depth, ix, iy});
    }
    offset_records += cells.size();
    return cells;
}

void write_frontier(const fs::path& path, const QuadtreeConfig& cfg,
                    std::uint64_t processed, const std::vector<Cell>& cells,
                    const std::vector<std::uint64_t>& deferred_offsets) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary);
    if (!out) throw std::runtime_error("Could not write " + temporary.string());
    out << "# version=2\n";
    out << "# signature=" << signature(cfg) << '\n';
    out << "# processed=" << processed << '\n';
    out << "# offsets=";
    for (std::size_t i = 0; i < deferred_offsets.size(); ++i) {
        if (i) out << ',';
        out << deferred_offsets[i];
    }
    out << '\n';
    out << "depth\tix\tiy\n";
    for (const auto& cell : cells) {
        out << cell.depth << '\t' << cell.ix << '\t' << cell.iy << '\n';
    }
    out.close();
    atomic_replace(temporary, path);
}

bool load_frontier(const fs::path& path, const QuadtreeConfig& cfg,
                   std::uint64_t& processed, std::vector<Cell>& cells,
                   std::vector<std::uint64_t>& deferred_offsets) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line) || line != "# version=2") return false;
    if (!std::getline(in, line) || line != "# signature=" + signature(cfg)) return false;
    if (!std::getline(in, line) || line.rfind("# processed=", 0) != 0) return false;
    try { processed = std::stoull(line.substr(12)); } catch (...) { return false; }
    if (!std::getline(in, line) || line.rfind("# offsets=", 0) != 0) return false;
    deferred_offsets.assign(static_cast<std::size_t>(cfg.max_depth + 1), 0);
    const auto fields = split(line.substr(10), ',');
    for (std::size_t i = 0; i < fields.size() && i < deferred_offsets.size(); ++i) {
        try { deferred_offsets[i] = std::stoull(fields[i]); } catch (...) { return false; }
    }
    std::getline(in, line);
    while (std::getline(in, line)) {
        const auto values = split(line, '\t');
        if (values.size() < 3) continue;
        try {
            cells.push_back(Cell{std::stoi(values[0]),
                                 static_cast<std::uint32_t>(std::stoul(values[1])),
                                 static_cast<std::uint32_t>(std::stoul(values[2]))});
        } catch (...) {}
    }
    return true;
}

Complex lattice_point(std::uint32_t gx, std::uint32_t gy,
                      std::uint64_t nx2, std::uint64_t ny2,
                      const QuadtreeConfig& cfg) {
    const Real x = cfg.xmin + (cfg.xmax - cfg.xmin)
        * static_cast<Real>(gx) / static_cast<Real>(nx2);
    const Real y = cfg.ymin + (cfg.ymax - cfg.ymin)
        * static_cast<Real>(gy) / static_cast<Real>(ny2);
    return {x, y};
}

std::uint64_t sample_key(std::uint32_t gx, std::uint32_t gy) {
    return (static_cast<std::uint64_t>(gx) << 32) | gy;
}

struct BatchStats {
    std::uint64_t samples = 0;
    std::uint64_t escaped = 0;
    std::uint64_t component = 0;
    std::uint64_t unresolved = 0;
    std::uint64_t pruned_far = 0;
    std::uint64_t pruned_inside = 0;
    std::uint64_t terminal = 0;
    std::uint64_t trace_failed = 0;
    std::uint64_t below_cutoff = 0;
    std::uint64_t accepted = 0;
};

Real cell_diagonal(const Cell& cell, const QuadtreeConfig& cfg) {
    const std::uint64_t nx = static_cast<std::uint64_t>(cfg.base_resolution) << cell.depth;
    const std::uint64_t ny = static_cast<std::uint64_t>(cfg.base_resolution_y) << cell.depth;
    const Real dx = (cfg.xmax - cfg.xmin) / static_cast<Real>(nx);
    const Real dy = (cfg.ymax - cfg.ymin) / static_cast<Real>(ny);
    return std::hypot(dx, dy);
}

bool same_component(const std::array<const SampleResult*, 9>& samples,
                    Real tolerance) {
    if (samples[0]->kind != SampleKind::Component) return false;
    const int period = samples[0]->period;
    const Complex center = samples[0]->center;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        if (samples[i]->kind != SampleKind::Component || samples[i]->period != period
            || safe_abs(samples[i]->center - center) > tolerance) return false;
    }
    return true;
}

Real refinement_priority(const std::array<const SampleResult*, 9>& samples,
                         Real diagonal, bool same) {
    int unresolved = 0, component = 0, escaped = 0;
    Real min_distance = std::numeric_limits<Real>::infinity();
    for (const auto* sample : samples) {
        if (sample->kind == SampleKind::Unresolved) ++unresolved;
        else if (sample->kind == SampleKind::Component) ++component;
        else {
            ++escaped;
            min_distance = std::min(min_distance, sample->distance);
        }
    }
    Real priority = 0;
    priority += 1000 * unresolved;
    if (!same && component > 0) priority += 500;
    if (component > 0 && escaped > 0) priority += 500;
    if (finite(min_distance) && diagonal > 0) {
        priority += 100 / (1 + min_distance / diagonal);
    }
    return priority;
}

// -----------------------------------------------------------------------------
// Main search
// -----------------------------------------------------------------------------

int run(int argc, char** argv) {
    const std::string usage="Usage: component_quadtree_hunter [--config PATH] [--reset|--restart-frontier]";
    const auto cli=mandelbrot::repo::parse_common_cli(argc,argv,usage);
    if(cli.help){std::cout<<usage<<'\n';return 0;}
    bool reset=false, restart_frontier=false;
    for(const auto& option:cli.remaining){
        if(option=="--reset") reset=true;
        else if(option=="--restart-frontier") restart_frontier=true;
        else throw std::runtime_error("Unknown option: "+option+"\n"+usage);
    }
    if(reset&&restart_frontier) throw std::runtime_error("Use only one of --reset and --restart-frontier");
    Config numerical=read_repository_config(cli.config,argv[0]);
    QuadtreeConfig cfg=load_config(cli.config,numerical,argv[0]);
    numerical.compute_areas = false;
    numerical.newton_max_iterations = std::max(numerical.newton_max_iterations, 80);
    numerical.continuation_max_depth = std::max(numerical.continuation_max_depth, 28);
    numerical.continuation_max_step = std::min<Real>(numerical.continuation_max_step, 0.025L);

    std::signal(SIGINT, quadtree_stop_handler);
    std::signal(SIGTERM, quadtree_stop_handler);

    mandelbrot::catalogue::Catalogue catalogue(cfg.catalogue_root);
    catalogue.ensure_layout();

    if (reset) {
        std::error_code error;
        fs::remove(catalogue.export_path(cfg.output_export), error);
        fs::remove(cfg.accepted_checkpoint_file, error);
        fs::remove(cfg.frontier_checkpoint_file, error);
        clear_deferred_frontiers(cfg);
    } else if (restart_frontier) {
        std::error_code error;
        fs::remove(cfg.frontier_checkpoint_file, error);
        clear_deferred_frontiers(cfg);
    }
    ensure_accepted_checkpoint(cfg.accepted_checkpoint_file);
    std::vector<Node> accepted = load_accepted_checkpoint(cfg.accepted_checkpoint_file);

    mandelbrot::catalogue::ComponentQuery known_query;
    known_query.require_polygon = true;
    known_query.require_polygon_converged = true;
    auto known_snapshot = catalogue.load_snapshot(known_query);
    std::vector<KnownPolygon> polygons = load_known_geometry(known_snapshot);
    if (polygons.empty()) {
        throw std::runtime_error(
            "The canonical catalogue contains no polygon geometry. "
            "Run component_area_scan before component_quadtree_hunter.");
    }
    for (auto& node : accepted) {
        const auto existing = known_snapshot.find_near_center(
            node.period,
            {catalogue_real(node.center.real()), catalogue_real(node.center.imag())},
            mandelbrot::catalogue::CatalogueReal(cfg.center_duplicate_tolerance));
        if (existing) node.id = existing->id;
        else if (node.id.empty() || node.id.rfind("q", 0) == 0) {
            node.id = mandelbrot::catalogue::Catalogue::stable_id(
                "quadtree:" + std::to_string(node.period) + ":"
                + real_string(node.center.real()) + ":"
                + real_string(std::abs(node.center.imag())));
        }
        polygons.push_back(polygon_from_node(node));
    }
    if (!accepted.empty()) save_nodes_to_catalogue(catalogue, accepted, cfg);
    std::vector<std::pair<int, Complex>> measured_centers;
    measured_centers.reserve(polygons.size() + 1024);
    for (const auto& polygon : polygons) {
        measured_centers.push_back({polygon.period, polygon.center});
    }

    std::uint64_t processed_cells = 0;
    std::vector<Cell> frontier;
    std::vector<std::uint64_t> deferred_offsets(
        static_cast<std::size_t>(cfg.max_depth + 1), 0);
    const bool resumed_frontier = !reset
        && load_frontier(cfg.frontier_checkpoint_file, cfg, processed_cells, frontier,
                         deferred_offsets);
    if (!resumed_frontier) {
        frontier.reserve(static_cast<std::size_t>(cfg.base_resolution) * cfg.base_resolution_y);
        for (int iy = 0; iy < cfg.base_resolution_y; ++iy) {
            for (int ix = 0; ix < cfg.base_resolution; ++ix) {
                frontier.push_back(Cell{0, static_cast<std::uint32_t>(ix),
                                        static_cast<std::uint32_t>(iy)});
            }
        }
        processed_cells = 0;
        write_frontier(cfg.frontier_checkpoint_file, cfg, processed_cells, frontier, deferred_offsets);
    }

    std::cout << "Mandelbrot adaptive quadtree component hunter\n"
              << "  bounds: [" << static_cast<double>(cfg.xmin) << ", "
              << static_cast<double>(cfg.xmax) << "] x ["
              << static_cast<double>(cfg.ymin) << ", " << static_cast<double>(cfg.ymax) << "]\n"
              << "  base grid: " << cfg.base_resolution << 'x' << cfg.base_resolution_y << '\n'
              << "  conjugate symmetry: "
              << (cfg.symmetry_active ? "active (searching Im(c) >= 0)" : "off") << '\n'
              << "  maximum depth: " << cfg.max_depth << " (effective x resolution "
              << (static_cast<std::uint64_t>(cfg.base_resolution) << cfg.max_depth) << ")\n"
              << "  period ceiling: " << cfg.max_period << '\n'
              << "  area cutoff: " << real_string(cfg.min_area, 4) << '\n'
              << "  active frontier chunk: " << cfg.max_cells << " cell(s)" << '\n'
              << "  known polygons: " << polygons.size() - accepted.size() << '\n'
              << "  resumed accepted: " << accepted.size() << '\n'
              << "  resumed frontier: " << (resumed_frontier ? frontier.size() : 0) << '\n'
              << "  output: " << catalogue.export_path(cfg.output_export) << '\n'
              << "  threads: " << cfg.threads << "\n\n";

    const auto search_started = Clock::now();
    std::uint64_t total_samples = 0;

    while (!frontier.empty()) {
        const int depth = frontier.front().depth;
        const std::uint64_t deferred_total = deferred_record_count(cfg, depth);
        const std::uint64_t loaded_offset = deferred_offsets[static_cast<std::size_t>(depth)];
        const std::uint64_t depth_cells_before_chunk = deferred_total > 0
            ? loaded_offset - std::min<std::uint64_t>(loaded_offset, frontier.size())
            : 0;
        const std::uint64_t depth_cells_total = deferred_total > 0
            ? deferred_total
            : static_cast<std::uint64_t>(frontier.size());
        const std::size_t chunk_index = static_cast<std::size_t>(
            depth_cells_before_chunk / std::max<std::uint64_t>(1, cfg.max_cells)) + 1;
        const std::size_t chunk_count = static_cast<std::size_t>(
            (depth_cells_total + cfg.max_cells - 1) / cfg.max_cells);
        const std::uint64_t queue_remaining = depth_cells_total - depth_cells_before_chunk;

        std::cout << "depth " << depth << ": " << human_count(frontier.size()) << " cells"
                  << "  chunk " << chunk_index << '/' << chunk_count
                  << "  left " << human_count(queue_remaining)
                  << "  processed " << human_count(processed_cells)
                  << "  elapsed " << format_duration(Clock::now() - search_started) << '\n';

        const auto depth_started = Clock::now();
        const std::size_t total_batches =
            (frontier.size() + cfg.cell_batch_size - 1) / cfg.cell_batch_size;
        DepthDashboard dashboard(
            depth, cfg.max_depth, total_batches, frontier.size(), depth_started,
            depth_cells_before_chunk, depth_cells_total, chunk_index, chunk_count);

        BatchStats level_stats;
        auto polygon_index = std::make_unique<PolygonIndex>(cfg, polygons);

        for (std::size_t batch_begin = 0; batch_begin < frontier.size();
             batch_begin += cfg.cell_batch_size) {
            const std::size_t batch_end = std::min(
                frontier.size(), batch_begin + cfg.cell_batch_size);
            const std::size_t batch_count = batch_end - batch_begin;
            if (batch_count == 0) break;
            const std::size_t batch_index = batch_begin / cfg.cell_batch_size;
            const std::uint64_t nx = static_cast<std::uint64_t>(cfg.base_resolution) << depth;
            const std::uint64_t ny = static_cast<std::uint64_t>(cfg.base_resolution_y) << depth;
            const std::uint64_t nx2 = 2 * nx;
            const std::uint64_t ny2 = 2 * ny;

            std::unordered_map<std::uint64_t, std::size_t> key_to_index;
            key_to_index.reserve(batch_count * 5);
            std::vector<std::pair<std::uint32_t, std::uint32_t>> coordinates;
            std::vector<std::array<std::size_t, 9>> references(batch_count);
            constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 9> offsets{{
                {0,0}, {2,0}, {2,2}, {0,2}, {1,0}, {2,1}, {1,2}, {0,1}, {1,1}
            }};
            for (std::size_t local = 0; local < batch_count; ++local) {
                const auto& cell = frontier[batch_begin + local];
                for (std::size_t probe = 0; probe < offsets.size(); ++probe) {
                    const std::uint32_t gx = 2 * cell.ix + offsets[probe].first;
                    const std::uint32_t gy = 2 * cell.iy + offsets[probe].second;
                    const std::uint64_t key = sample_key(gx, gy);
                    auto [it, inserted] = key_to_index.emplace(key, coordinates.size());
                    if (inserted) coordinates.push_back({gx, gy});
                    references[local][probe] = it->second;
                }
            }

            std::vector<SampleResult> samples(coordinates.size());
            std::atomic<std::size_t> sample_done{0};
            dashboard.update(
                batch_index, 0.0L, batch_begin, "samples", 0,
                coordinates.size(), level_stats.accepted);
            parallel_for(coordinates.size(), cfg.threads, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    const auto [gx, gy] = coordinates[i];
                    samples[i] = classify_sample(
                        lattice_point(gx, gy, nx2, ny2, cfg), cfg, numerical, *polygon_index);
                    const auto done = ++sample_done;
                    if (done % 4000 == 0 || done == coordinates.size()) {
                        const long double stage_fraction = coordinates.empty()
                            ? 1.0L
                            : static_cast<long double>(done)
                                / static_cast<long double>(coordinates.size());
                        dashboard.update(
                            batch_index, 0.85L * stage_fraction, batch_begin,
                            "samples", done, coordinates.size(), level_stats.accepted);
                    }
                }
            });
            level_stats.samples += samples.size();
            total_samples += samples.size();
            for (const auto& sample : samples) {
                if (sample.kind == SampleKind::Escaped) ++level_stats.escaped;
                else if (sample.kind == SampleKind::Component) ++level_stats.component;
                else ++level_stats.unresolved;
            }

            struct Candidate { int period; Complex center; Complex sample; };
            std::vector<Candidate> candidates;
            for (std::size_t i = 0; i < samples.size(); ++i) {
                const auto& sample = samples[i];
                if (sample.kind != SampleKind::Component) continue;
                if (near_center(sample.center, sample.period, measured_centers,
                                cfg.center_duplicate_tolerance)) continue;
                bool duplicate = false;
                for (const auto& candidate : candidates) {
                    if (candidate.period == sample.period
                        && safe_abs(candidate.center - sample.center)
                            <= cfg.center_duplicate_tolerance) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    const auto [gx, gy] = coordinates[i];
                    candidates.push_back(Candidate{
                        sample.period, sample.center,
                        lattice_point(gx, gy, nx2, ny2, cfg)});
                }
            }

            for (const auto& candidate : candidates) {
                measured_centers.push_back({candidate.period, candidate.center});
            }
            std::vector<std::optional<GeometryResult>> geometries(candidates.size());
            if (!candidates.empty()) {
                std::atomic<std::size_t> geometry_done{0};
                parallel_for(candidates.size(), cfg.threads,
                    [&](std::size_t begin, std::size_t end) {
                        for (std::size_t i = begin; i < end; ++i) {
                            auto state = center_state(candidates[i].period,
                                                      candidates[i].center, numerical);
                            if (state) {
                                const Real lower = PI * std::norm(state->c_lambda);
                                if (!(cfg.screening_lower_bound_ratio > 0
                                      && lower < cfg.min_area
                                          * cfg.screening_lower_bound_ratio)) {
                                    geometries[i] = trace_geometry(
                                        candidates[i].center, candidates[i].period,
                                        cfg, numerical);
                                }
                            }
                            const auto done = ++geometry_done;
                            const long double stage_fraction = candidates.empty()
                                ? 1.0L
                                : static_cast<long double>(done)
                                    / static_cast<long double>(candidates.size());
                            dashboard.update(
                                batch_index, 0.85L + 0.15L * stage_fraction,
                                batch_begin, "geometry", done, candidates.size(),
                                level_stats.accepted);
                        }
                    });
            }

            std::vector<Node> newly_accepted;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                if (!geometries[i]) {
                    ++level_stats.trace_failed;
                    continue;
                }
                if (geometries[i]->area < cfg.min_area) {
                    ++level_stats.below_cutoff;
                    continue;
                }
                if (near_center(candidates[i].center, candidates[i].period, polygons,
                                cfg.center_duplicate_tolerance)) continue;
                Node node;
                node.id = mandelbrot::catalogue::Catalogue::stable_id(
                    "quadtree:" + std::to_string(candidates[i].period) + ":"
                    + real_string(candidates[i].center.real()) + ":"
                    + real_string(std::abs(candidates[i].center.imag())));
                node.period = candidates[i].period;
                node.center = candidates[i].center;
                node.sample = candidates[i].sample;
                node.discovery_depth = depth;
                node.area = geometries[i]->area;
                node.area_rho = cfg.area_rho;
                node.polygon_rho = cfg.polygon_rho;
                node.polygon_area = geometries[i]->polygon_area;
                node.points = std::move(geometries[i]->points);
                polygons.push_back(polygon_from_node(node));
                newly_accepted.push_back(node);
                ++level_stats.accepted;

            }
            if (!newly_accepted.empty()) {
                // Canonical records are committed before the transient TSV is
                // advanced, so a crash cannot permanently skip a discovery.
                save_nodes_to_catalogue(catalogue, newly_accepted, cfg);
                std::ofstream checkpoint(cfg.accepted_checkpoint_file, std::ios::app);
                if (!checkpoint) throw std::runtime_error(
                    "Could not append " + cfg.accepted_checkpoint_file.string());
                for (const auto& node : newly_accepted) append_accepted_checkpoint(checkpoint, node);
                checkpoint.close();
                if (!checkpoint) throw std::runtime_error(
                    "Failed while writing " + cfg.accepted_checkpoint_file.string());
                accepted.insert(accepted.end(),
                                std::make_move_iterator(newly_accepted.begin()),
                                std::make_move_iterator(newly_accepted.end()));
                polygon_index = std::make_unique<PolygonIndex>(cfg, polygons);
            }

            std::vector<RefineCell> batch_refine;
            batch_refine.reserve(batch_count / 2);
            for (std::size_t local = 0; local < batch_count; ++local) {
                const Cell& cell = frontier[batch_begin + local];
                std::array<const SampleResult*, 9> probes{};
                for (std::size_t i = 0; i < probes.size(); ++i) {
                    probes[i] = &samples[references[local][i]];
                }
                const Real diagonal = cell_diagonal(cell, cfg);
                const bool same = same_component(probes, cfg.center_duplicate_tolerance * 4);
                if (same && probes[8]->interior_margin
                                > cfg.inside_prune_factor * diagonal) {
                    ++level_stats.pruned_inside;
                    continue;
                }
                bool all_escaped = true;
                Real min_distance = std::numeric_limits<Real>::infinity();
                for (const auto* probe : probes) {
                    if (probe->kind != SampleKind::Escaped) all_escaped = false;
                    else min_distance = std::min(min_distance, probe->distance);
                }
                if (all_escaped && min_distance > cfg.distance_prune_factor * diagonal) {
                    ++level_stats.pruned_far;
                    continue;
                }
                if (cell.depth >= cfg.max_depth) {
                    ++level_stats.terminal;
                    continue;
                }
                batch_refine.push_back(RefineCell{
                    cell, refinement_priority(probes, diagonal, same)});
            }
            std::sort(batch_refine.begin(), batch_refine.end(), [](const auto& a, const auto& b) {
                if (a.priority != b.priority) return a.priority > b.priority;
                return std::tie(a.cell.iy, a.cell.ix) < std::tie(b.cell.iy, b.cell.ix);
            });
            const int child_depth = depth + 1;
            if (child_depth <= cfg.max_depth && !batch_refine.empty()) {
                std::vector<Cell> deferred_buffer;
                deferred_buffer.reserve(std::min<std::size_t>(65536, batch_refine.size() * 4));
                auto flush_deferred = [&]() {
                    if (!deferred_buffer.empty()) {
                        append_deferred_cells(cfg, child_depth, deferred_buffer);
                        deferred_buffer.clear();
                    }
                };
                for (const auto& item : batch_refine) {
                    const Cell& parent = item.cell;
                    const std::uint32_t x = 2 * parent.ix;
                    const std::uint32_t y = 2 * parent.iy;
                    const std::array<Cell, 4> generated{{
                        Cell{child_depth, x, y},
                        Cell{child_depth, x + 1, y},
                        Cell{child_depth, x, y + 1},
                        Cell{child_depth, x + 1, y + 1}
                    }};
                    for (const auto& child : generated) {
                        deferred_buffer.push_back(child);
                        if (deferred_buffer.size() >= 65536) flush_deferred();
                    }
                }
                flush_deferred();
            }

            processed_cells += batch_count;
            dashboard.update(
                batch_index, 1.0L, batch_end, "batch complete", 0, 0,
                level_stats.accepted);

            if (g_stop_requested) {
                std::vector<Cell> remaining;
                remaining.reserve(frontier.size() - batch_end);
                remaining.insert(remaining.end(), frontier.begin() + batch_end, frontier.end());
                write_frontier(cfg.frontier_checkpoint_file, cfg, processed_cells,
                               remaining, deferred_offsets);
                write_quadtree_export(catalogue, cfg);
                dashboard.finish(level_stats.accepted);
                std::cout << "Stop requested. Finished the current batch and checkpointed "
                          << remaining.size() << " remaining active cell(s).\n"
                          << "Resume with the same command; no reset option is needed.\n";
                return 130;
            }
        }

        dashboard.finish(level_stats.accepted);

        const int child_depth = depth + 1;
        const std::uint64_t queued_next = child_depth <= cfg.max_depth
            ? deferred_record_count(cfg, child_depth)
                - deferred_offsets[static_cast<std::size_t>(child_depth)]
            : 0;

        std::cout << "  samples=" << level_stats.samples
                  << " escaped=" << level_stats.escaped
                  << " component=" << level_stats.component
                  << " unresolved=" << level_stats.unresolved
                  << " accepted=" << level_stats.accepted
                  << " below_cutoff=" << level_stats.below_cutoff
                  << " trace_failed=" << level_stats.trace_failed << '\n'
                  << "  pruned_far=" << level_stats.pruned_far
                  << " pruned_inside=" << level_stats.pruned_inside
                  << " terminal=" << level_stats.terminal
                  << " queued_next=" << queued_next << "\n\n";

        // Continue with another chunk at this same depth if one is queued.
        frontier = load_deferred_chunk(
            cfg, depth, deferred_offsets[static_cast<std::size_t>(depth)]);
        if (frontier.empty()) {
            // This depth is exhausted; advance to the next non-empty deferred depth.
            for (int next_depth = depth + 1; next_depth <= cfg.max_depth; ++next_depth) {
                frontier = load_deferred_chunk(
                    cfg, next_depth,
                    deferred_offsets[static_cast<std::size_t>(next_depth)]);
                if (!frontier.empty()) break;
            }
        }
        write_frontier(cfg.frontier_checkpoint_file, cfg, processed_cells, frontier,
                       deferred_offsets);
    }

    write_quadtree_export(catalogue, cfg);
    std::cout << "Done. Processed " << processed_cells << " cell(s), "
              << total_samples << " unique batch sample(s), and discovered "
              << accepted.size() << " component(s) above the cutoff.\n"
              << "This is an adaptive numerical search, not a proof of completeness.\n";
    return 0;
}

} // namespace quadtree_hunter

int main(int argc, char** argv) {
    try {
        return quadtree_hunter::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "quadtree hunter error: " << error.what() << '\n';
        return 1;
    }
}
