#define main component_area_scan_embedded_main
#include "component_area_scan.cpp"
#undef main

#include <array>
#include <deque>
#include <regex>

namespace hunter {

struct HunterConfig {
    int base_max_period = 14;
    int max_period = 50;
    int max_denominator = 50;
    int max_generations = 12;
    Real min_area = 1.0e-10L;
    Real area_rho = 0.99999L;
    Real polygon_rho = 0.9995L;
    int polygon_points = 192;
    int coordinate_digits = 11;
    unsigned threads = 0;
    Real attach_inner_rho = 0.99L;
    Real attach_outer_rho = 0.999999L;
    Real child_root_rho = 0.9999L;
    Real attachment_match_factor = 0.12L;
    Real max_child_distance_factor = 1.5L;
    Real screening_lower_bound_ratio = 1.0e-8L;
    Real center_duplicate_tolerance = 1.0e-10L;
    std::vector<Real> seed_factors{
        1.0e-6L, 3.0e-6L, 1.0e-5L, 3.0e-5L,
        1.0e-4L, 3.0e-4L, 1.0e-3L, 3.0e-3L,
        1.0e-2L, 3.0e-2L, 6.0e-2L, 1.0e-1L,
        1.5e-1L, 2.0e-1L, 3.0e-1L, 4.0e-1L,
        5.0e-1L, 7.0e-1L, 1.0L
    };
    std::vector<Real> seed_angle_offsets_deg{0.0L, 5.0L, -5.0L, 15.0L, -15.0L};
    fs::path catalogue_root;
    fs::path run_dir;
    std::string output_export = "atlas_components_extra.json";
    fs::path checkpoint_file;
};

struct AreaRecord {
    Real area = 0;
    Real rho = 0;
};

struct Node {
    std::string id;
    std::string parent_id;
    std::string address;
    int period = 0;
    int source_index = -1;
    Complex center{};
    Real area = 0;
    Real area_rho = 0;
    Real polygon_rho = 0;
    Real polygon_area = 0;
    int parent_period = 0;
    Complex parent_center{};
    int rotation_p = 0;
    int rotation_q = 0;
    std::string hierarchy_root;
    int generation = 0;
    Complex attachment_parent{};
    Complex attachment_child_centered{};
    Real attachment_gap = 0;
    std::vector<Complex> points;
};

struct Task {
    std::size_t parent_index = 0;
    int p = 0;
    int q = 0;
    std::string key;
    std::string address;
};

struct TaskResult {
    Task task;
    std::string status;
    std::string message;
    std::optional<Node> node;
};

Node node_from_component(const mandelbrot::catalogue::ComponentRecord& component) {
    Node node;
    node.id = component.id;
    node.parent_id = component.hierarchy.geometric_parent.value_or("");
    node.address = component.id;
    for (const auto& alias : component.provenance.aliases) {
        if (alias.rfind("satellite-address:", 0) == 0) {
            node.address = alias.substr(std::string("satellite-address:").size());
            break;
        }
    }
    node.period = component.period;
    node.center = {
        component.center.re.convert_to<Real>(),
        component.center.im.convert_to<Real>()};
    node.area = component.geometry.area_estimate.convert_to<Real>();
    node.area_rho = component.geometry.area_rho.convert_to<Real>();
    node.polygon_rho = component.geometry.polygon_rho.convert_to<Real>();
    node.polygon_area = component.geometry.polygon_area.convert_to<Real>();
    node.hierarchy_root = component.hierarchy.hierarchy_root.value_or(component.id);
    node.generation = component.hierarchy.generation.value_or(0);
    node.points.reserve(component.geometry.polygon.size());
    for (const auto& offset : component.geometry.polygon) {
        node.points.emplace_back(
            node.center.real() + offset.re.convert_to<Real>(),
            node.center.imag() + offset.im.convert_to<Real>());
    }
    return node;
}


HunterConfig load_hunter_config(
    const fs::path& config_path,
    const Config& scan_config,
    const char* argv0
) {
    const auto repo = mandelbrot::repo::RepoConfig::load(
        config_path, executable_parent_or_cwd(argv0));
    HunterConfig cfg;
    cfg.catalogue_root = repo.path("paths.catalogue_root");
    mandelbrot::catalogue::Catalogue catalogue(cfg.catalogue_root);
    const std::string prefix = "component_atlas_hunter.";
    const std::string run_name = repo.string(prefix + "run_name", "default");
    cfg.run_dir = catalogue.run_path("atlas_hunter", run_name);
    cfg.checkpoint_file = catalogue.run_path("atlas_hunter", run_name, "checkpoint.tsv");
    cfg.output_export = repo.string(prefix + "output_export", cfg.output_export);
    cfg.threads = repo.threads();
    cfg.base_max_period = repo.integer("component_area_scan.period", cfg.base_max_period);
    cfg.max_period = repo.integer(prefix + "max_period", cfg.max_period);
    cfg.max_denominator = repo.integer(prefix + "max_denominator", cfg.max_denominator);
    cfg.max_generations = repo.integer(prefix + "max_generations", cfg.max_generations);
    cfg.min_area = static_cast<Real>(repo.number(prefix + "min_area", cfg.min_area));
    cfg.area_rho = static_cast<Real>(repo.number(prefix + "area_rho", cfg.area_rho));
    cfg.polygon_rho = static_cast<Real>(repo.number(prefix + "polygon_rho", cfg.polygon_rho));
    cfg.polygon_points = repo.integer(prefix + "polygon_points", cfg.polygon_points);
    cfg.coordinate_digits = repo.integer(prefix + "coordinate_digits", cfg.coordinate_digits);
    cfg.attach_inner_rho = static_cast<Real>(repo.number(prefix + "attach_inner_rho", cfg.attach_inner_rho));
    cfg.attach_outer_rho = static_cast<Real>(repo.number(prefix + "attach_outer_rho", cfg.attach_outer_rho));
    cfg.child_root_rho = static_cast<Real>(repo.number(prefix + "child_root_rho", cfg.child_root_rho));
    cfg.attachment_match_factor = static_cast<Real>(repo.number(prefix + "attachment_match_factor", cfg.attachment_match_factor));
    cfg.max_child_distance_factor = static_cast<Real>(repo.number(prefix + "max_child_distance_factor", cfg.max_child_distance_factor));
    cfg.screening_lower_bound_ratio = static_cast<Real>(repo.number(prefix + "screening_lower_bound_ratio", cfg.screening_lower_bound_ratio));
    cfg.center_duplicate_tolerance = static_cast<Real>(repo.number(prefix + "center_duplicate_tolerance", cfg.center_duplicate_tolerance));
    if (repo.find(prefix + "seed_factors")) {
        cfg.seed_factors.clear();
        for (const auto value : repo.number_array(prefix + "seed_factors")) {
            cfg.seed_factors.push_back(static_cast<Real>(value));
        }
    }
    if (repo.find(prefix + "seed_angle_offsets_deg")) {
        cfg.seed_angle_offsets_deg.clear();
        for (const auto value : repo.number_array(prefix + "seed_angle_offsets_deg")) {
            cfg.seed_angle_offsets_deg.push_back(static_cast<Real>(value));
        }
    }
    if (cfg.max_period <= cfg.base_max_period) {
        throw std::runtime_error(
            "component_atlas_hunter.max_period must exceed component_area_scan.period");
    }
    cfg.max_denominator = std::max(2, cfg.max_denominator);
    cfg.max_generations = std::max(1, cfg.max_generations);
    cfg.polygon_points = std::max(32, cfg.polygon_points);
    cfg.coordinate_digits = std::clamp(cfg.coordinate_digits, 8, 21);
    if (!(cfg.min_area > 0 && cfg.area_rho > 0 && cfg.area_rho < 1
          && cfg.polygon_rho > 0 && cfg.polygon_rho < 1)) {
        throw std::runtime_error("Invalid component_atlas_hunter area/rho configuration");
    }
    fs::create_directories(cfg.run_dir);
    (void)scan_config;
    return cfg;
}

std::optional<Complex> local_polish_center(Complex seed, int period, Real max_step,
                                           const Config& config) {
    Complex c = seed;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const auto orbit = critical_orbit(c, period);
        const Real residual = safe_abs(orbit.value);
        if (residual <= std::max<Real>(1.0e-14L, config.center_residual_tolerance)) {
            if (detected_center_period(c, period, config.exact_period_tolerance) == period) return c;
            return std::nullopt;
        }
        if (safe_abs(orbit.derivative) < 1.0e-32L) return std::nullopt;
        Complex correction = orbit.value / orbit.derivative;
        const Real size = safe_abs(correction);
        if (size > max_step) correction *= max_step / size;
        c -= correction;
        if (!finite(c)) return std::nullopt;
    }
    return std::nullopt;
}

struct Attachment {
    Complex boundary{};
    Complex direction{};
    Real parent_scale = 0;
};

std::optional<Attachment> parent_attachment(const Node& parent, int p, int q,
                                             const HunterConfig& hcfg,
                                             const Config& numerical) {
    auto state = center_state(parent.period, parent.center, numerical);
    if (!state) return std::nullopt;
    const Real angle = 2 * PI * static_cast<Real>(p) / static_cast<Real>(q);
    const Complex unit{std::cos(angle), std::sin(angle)};
    ContinuationStats stats;
    auto inner = continue_to_lambda(parent.period, *state, hcfg.attach_inner_rho * unit,
                                    numerical, stats);
    if (!inner) return std::nullopt;
    auto outer = continue_to_lambda(parent.period, *inner, hcfg.attach_outer_rho * unit,
                                    numerical, stats);
    if (!outer) return std::nullopt;
    Complex radial = unit * outer->c_lambda;
    if (safe_abs(radial) < 1.0e-30L) radial = outer->c - inner->c;
    if (safe_abs(radial) < 1.0e-30L) return std::nullopt;
    const Complex boundary = outer->c + (1 - hcfg.attach_outer_rho) * radial;
    return Attachment{boundary, radial / safe_abs(radial),
                      std::sqrt(std::max<Real>(parent.area, hcfg.min_area) / PI)};
}

Real attachment_error(Complex child_center, int child_period, Complex boundary,
                      const HunterConfig& hcfg, const Config& numerical) {
    auto state = center_state(child_period, child_center, numerical);
    if (!state) return std::numeric_limits<Real>::infinity();
    ContinuationStats stats;
    auto near_root = continue_to_lambda(child_period, *state,
                                        Complex{hcfg.child_root_rho,0}, numerical, stats);
    if (!near_root) return std::numeric_limits<Real>::infinity();
    const Complex estimated_root = near_root->c
        + (1-hcfg.child_root_rho) * near_root->c_lambda;
    return safe_abs(estimated_root - boundary);
}

std::optional<Complex> find_satellite_center(const Node& parent, int p, int q,
                                              const HunterConfig& hcfg,
                                              const Config& numerical,
                                              Real& match_error) {
    const int child_period = parent.period * q;
    auto attachment = parent_attachment(parent,p,q,hcfg,numerical);
    if (!attachment) return std::nullopt;
    const Real scale = std::max<Real>(attachment->parent_scale, 1.0e-12L);
    std::vector<Complex> candidates;
    for (Real offset_deg : hcfg.seed_angle_offsets_deg) {
        const Real offset = offset_deg * PI / 180;
        const Complex rotated = attachment->direction * Complex{std::cos(offset),std::sin(offset)};
        for (Real factor : hcfg.seed_factors) {
            auto root = local_polish_center(attachment->boundary + rotated * (scale * factor),
                                            child_period, std::max<Real>(0.75L*scale,1.0e-5L),
                                            numerical);
            if (!root) continue;
            if (safe_abs(*root-attachment->boundary) > hcfg.max_child_distance_factor*scale) continue;
            bool duplicate = false;
            for (const auto& value : candidates) {
                if (safe_abs(value-*root) <= hcfg.center_duplicate_tolerance) {
                    duplicate = true; break;
                }
            }
            if (!duplicate) candidates.push_back(*root);
        }
    }

    Real best_score = std::numeric_limits<Real>::infinity();
    std::optional<Complex> best;
    match_error = std::numeric_limits<Real>::infinity();
    for (const auto& candidate : candidates) {
        const Complex delta = candidate - attachment->boundary;
        const Real projection = (delta * std::conj(attachment->direction)).real();
        const Real lateral = safe_abs(delta - projection*attachment->direction);
        if (projection < -0.05L*scale) continue;
        const Real root_match = attachment_error(candidate, child_period, attachment->boundary,
                                                 hcfg, numerical);
        if (!finite(root_match) || root_match > hcfg.attachment_match_factor*scale) continue;
        const Real score = root_match/scale + lateral/scale + 0.01L*safe_abs(delta)/scale;
        if (score < best_score) {
            best_score = score;
            match_error = root_match;
            best = candidate;
        }
    }
    return best;
}

struct GeometryResult {
    Real area = 0;
    Real polygon_area = 0;
    std::vector<Complex> points;
};

std::optional<GeometryResult> trace_geometry(Complex center, int period,
                                             const HunterConfig& hcfg,
                                             const Config& numerical) {
    auto state = center_state(period,center,numerical);
    if (!state) return std::nullopt;

    std::vector<Real> stages{0.9L,0.99L,0.999L,hcfg.polygon_rho,hcfg.area_rho};
    std::sort(stages.begin(),stages.end());
    stages.erase(std::unique(stages.begin(),stages.end(),[](Real a,Real b){return std::abs(a-b)<1e-18L;}),stages.end());

    for (int multiplier : {1,2,4}) {
        const int theta = hcfg.polygon_points * multiplier;
        std::optional<RingTrace> seed;
        std::optional<RingTrace> polygon_ring;
        std::optional<RingTrace> area_ring;
        bool ok = true;
        for (Real rho : stages) {
            auto ring = trace_ring(period,*state,rho,theta,numerical,seed?&*seed:nullptr);
            if (!ring) { ok=false; break; }
            if (std::abs(rho-hcfg.polygon_rho)<1e-18L) polygon_ring=*ring;
            if (std::abs(rho-hcfg.area_rho)<1e-18L) area_ring=*ring;
            seed=std::move(*ring);
        }
        if (!ok || !polygon_ring || !area_ring) continue;

        std::vector<Complex> offsets(theta);
        for (int i=0;i<theta;++i) offsets[i]=area_ring->c[i]-center;
        const Real area = std::abs(derivative_area(offsets,area_ring->lambda,area_ring->c_lambda));
        std::vector<Complex> coarse_o,coarse_l,coarse_d;
        for (int i=0;i<theta;i+=2) {
            coarse_o.push_back(offsets[i]); coarse_l.push_back(area_ring->lambda[i]);
            coarse_d.push_back(area_ring->c_lambda[i]);
        }
        const Real coarse = std::abs(derivative_area(coarse_o,coarse_l,coarse_d));
        const Real error = std::abs(area-coarse);
        const Real requested = std::max<Real>(1.0e-14L,1.0e-5L*std::max(area,hcfg.min_area));
        if (multiplier<4 && (error>requested || (area>0.5L*hcfg.min_area && area<2*hcfg.min_area))) continue;

        std::vector<Complex> points;
        points.reserve(hcfg.polygon_points);
        for (int i=0;i<hcfg.polygon_points;++i) {
            points.push_back(polygon_ring->c[static_cast<std::size_t>(i*multiplier)]);
        }
        if (polygon_area(points) < 0) std::reverse(points.begin(), points.end());
        return GeometryResult{area,std::abs(polygon_area(points)),std::move(points)};
    }
    return std::nullopt;
}


std::array<Real,4> bbox_centered(const Node& node) {
    if (node.points.empty()) return {0,0,0,0};
    Real xmin = node.points.front().real() - node.center.real();
    Real xmax = xmin;
    Real ymin = node.points.front().imag() - node.center.imag();
    Real ymax = ymin;
    for (const auto& point : node.points) {
        const Real x = point.real() - node.center.real();
        const Real y = point.imag() - node.center.imag();
        xmin = std::min(xmin, x); xmax = std::max(xmax, x);
        ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    return {xmin,xmax,ymin,ymax};
}

void ensure_checkpoint_header(const fs::path& path) {
    if (fs::exists(path)) return;
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Could not create " + path.string());
    output << "key\tstatus\taddress\tparent_id\tperiod\tp\tq\tcomponent_id\n";
}

std::set<std::string> load_processed_tasks(const fs::path& path) {
    std::set<std::string> result;
    std::ifstream input(path);
    if (!input) return result;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        const auto fields = split(line, '\t');
        if (!fields.empty() && !fields[0].empty()) result.insert(fields[0]);
    }
    return result;
}

void append_checkpoint(
    std::ostream& output,
    const TaskResult& result,
    const Node& parent
) {
    output << result.task.key << '\t' << result.status << '\t'
           << result.task.address << '\t' << parent.id << '\t'
           << parent.period * result.task.q << '\t'
           << result.task.p << '\t' << result.task.q << '\t';
    if (result.node) output << result.node->id;
    output << '\n';
}

bool near_known(
    Complex center,
    int period,
    const std::vector<Node>& known,
    Real tolerance
) {
    for (const auto& component : known) {
        if (component.period == period
            && safe_abs(center - component.center) <= tolerance) return true;
    }
    return false;
}

mandelbrot::catalogue::CatalogueReal catalogue_real(Real value) {
    return mandelbrot::catalogue::Catalogue::parse_decimal(real_string(value));
}

mandelbrot::catalogue::ComponentRecord node_to_component(const Node& node) {
    using namespace mandelbrot::catalogue;
    ComponentRecord component;
    component.id = node.id;
    component.period = node.period;
    component.center = {catalogue_real(node.center.real()), catalogue_real(node.center.imag())};
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
    const auto bounds = bbox_centered(node);
    for (std::size_t i = 0; i < 4; ++i) {
        component.geometry.bbox_centered[i] = catalogue_real(bounds[i]);
    }
    component.geometry.polygon.reserve(node.points.size());
    for (const auto& point : node.points) {
        component.geometry.polygon.push_back({
            catalogue_real(point.real() - node.center.real()),
            catalogue_real(point.imag() - node.center.imag())});
    }
    component.classification.shape_class = "unknown";
    component.classification.shape_confidence = 0;
    component.hierarchy.geometric_parent = node.parent_id;
    component.hierarchy.hierarchy_root = node.hierarchy_root.empty()
        ? std::optional<std::string>(node.parent_id)
        : std::optional<std::string>(node.hierarchy_root);
    component.hierarchy.generation = node.generation;
    AttachmentRecord attachment;
    attachment.parent_point = ComplexValue{
        catalogue_real(node.attachment_parent.real()),
        catalogue_real(node.attachment_parent.imag())};
    attachment.child_point_centered = ComplexValue{
        catalogue_real(node.attachment_child_centered.real()),
        catalogue_real(node.attachment_child_centered.imag())};
    attachment.gap = catalogue_real(node.attachment_gap);
    const Real size = std::sqrt(std::max<Real>(node.area, 1e-300L) / PI);
    attachment.gap_relative_to_child_size = catalogue_real(node.attachment_gap / size);
    attachment.verified = true;
    component.hierarchy.attachment = attachment;
    component.provenance.method = "satellite-hunter";
    component.provenance.run_id = "default";
    component.provenance.discovered_at = utc_timestamp();
    component.provenance.aliases = {
        "satellite-address:" + node.address,
        "rotation:" + std::to_string(node.rotation_p) + "/" + std::to_string(node.rotation_q)};
    component.quality.center_validated = true;
    component.quality.exact_period_validated = true;
    component.quality.polygon_converged = true;
    component.quality.area_above_cutoff = true;
    return component;
}

TaskResult process_task(
    const Task& task,
    const Node& parent,
    const HunterConfig& hcfg,
    const Config& numerical
) {
    TaskResult result;
    result.task = task;
    Real match_error = 0;
    auto center = find_satellite_center(
        parent, task.p, task.q, hcfg, numerical, match_error);
    if (!center) { result.status = "center_failed"; return result; }
    const int period = parent.period * task.q;
    auto state = center_state(period, *center, numerical);
    if (!state) { result.status = "state_failed"; return result; }
    const Real lower = PI * std::norm(state->c_lambda);
    if (hcfg.screening_lower_bound_ratio > 0
        && lower < hcfg.min_area * hcfg.screening_lower_bound_ratio) {
        result.status = "screened_small";
        return result;
    }
    auto geometry = trace_geometry(*center, period, hcfg, numerical);
    if (!geometry) { result.status = "trace_failed"; return result; }
    if (geometry->area < hcfg.min_area) {
        result.status = "below_cutoff";
        return result;
    }
    const auto attachment = parent_attachment(parent, task.p, task.q, hcfg, numerical);
    if (!attachment) { result.status = "attachment_failed"; return result; }
    Node node;
    node.id = mandelbrot::catalogue::Catalogue::stable_id(
        "satellite:" + std::to_string(period) + ":"
        + real_string(center->real()) + ":"
        + real_string(std::abs(center->imag())));
    node.parent_id = parent.id;
    node.address = task.address;
    node.period = period;
    node.center = *center;
    node.area = geometry->area;
    node.area_rho = hcfg.area_rho;
    node.polygon_rho = hcfg.polygon_rho;
    node.polygon_area = geometry->polygon_area;
    node.parent_period = parent.period;
    node.parent_center = parent.center;
    node.rotation_p = task.p;
    node.rotation_q = task.q;
    node.hierarchy_root = parent.hierarchy_root.empty() ? parent.id : parent.hierarchy_root;
    node.generation = parent.generation + 1;
    node.attachment_parent = attachment->boundary;
    node.attachment_child_centered = attachment->boundary - node.center;
    node.attachment_gap = match_error;
    node.points = std::move(geometry->points);
    result.status = "accepted";
    result.node = std::move(node);
    return result;
}

void refresh_period_metadata(
    mandelbrot::catalogue::Catalogue& catalogue,
    const HunterConfig& config,
    const std::set<int>& changed_periods
) {
    using namespace mandelbrot::catalogue;
    if (changed_periods.empty()) return;
    catalogue.rebuild_period_indexes(
        std::vector<int>(changed_periods.begin(), changed_periods.end()),
        CatalogueReal(config.min_area));
    catalogue.rebuild_manifest_from_period_indexes(
        config.base_max_period, CatalogueReal(config.min_area));
}

void write_derived_outputs(
    mandelbrot::catalogue::Catalogue& catalogue,
    const HunterConfig& config
) {
    using namespace mandelbrot::catalogue;
    catalogue.rebuild_hierarchy_indexes(CatalogueReal(config.min_area));
    ComponentExportOptions options;
    options.complete = false;
    options.coordinate_digits = config.coordinate_digits;
    options.query.min_area = CatalogueReal(config.min_area);
    options.query.require_polygon = true;
    options.query.require_polygon_converged = true;
    options.query.provenance_method = "satellite-hunter";
    catalogue.write_component_export(catalogue.export_path(config.output_export), options);
}

int run(int argc, char** argv) {
    const std::string usage =
        "Usage: component_atlas_hunter [--config PATH] [--reset]";
    const auto cli = mandelbrot::repo::parse_common_cli(argc, argv, usage);
    if (cli.help) { std::cout << usage << '\n'; return 0; }
    bool reset = false;
    for (const auto& option : cli.remaining) {
        if (option == "--reset") reset = true;
        else throw std::runtime_error("Unknown option: " + option + "\n" + usage);
    }

    Config numerical = read_repository_config(cli.config, argv[0]);
    HunterConfig hcfg = load_hunter_config(cli.config, numerical, argv[0]);
    numerical.compute_areas = false;
    numerical.newton_max_iterations = std::max(numerical.newton_max_iterations, 80);
    numerical.continuation_max_depth = std::max(numerical.continuation_max_depth, 24);
    numerical.continuation_max_step = std::min<Real>(numerical.continuation_max_step, 0.025L);

    mandelbrot::catalogue::Catalogue catalogue(hcfg.catalogue_root);
    catalogue.ensure_layout();
    if (reset) {
        std::error_code error;
        fs::remove(hcfg.checkpoint_file, error);
        fs::remove(catalogue.export_path(hcfg.output_export), error);
    }
    ensure_checkpoint_header(hcfg.checkpoint_file);

    mandelbrot::catalogue::ComponentQuery query;
    query.max_period = hcfg.max_period;
    query.min_area = mandelbrot::catalogue::CatalogueReal(hcfg.min_area);
    query.require_polygon = true;
    query.require_polygon_converged = true;
    auto snapshot = catalogue.load_snapshot(query);
    if (snapshot.components.empty()) {
        throw std::runtime_error(
            "The canonical catalogue contains no polygon components above the cutoff. "
            "Run component_area_scan first.");
    }

    std::vector<Node> nodes;
    nodes.reserve(snapshot.components.size());
    for (const auto& component : snapshot.components) {
        nodes.push_back(node_from_component(component));
    }
    const std::size_t base_seed_count = std::count_if(
        nodes.begin(), nodes.end(), [&](const Node& node) {
            return node.period <= hcfg.base_max_period;
        });
    const std::size_t resumed_count = std::count_if(
        snapshot.components.begin(), snapshot.components.end(),
        [](const auto& component) {
            return component.provenance.method == "satellite-hunter";
        });
    std::set<std::string> processed = load_processed_tasks(hcfg.checkpoint_file);

    std::cout << "Mandelbrot component satellite hunter\n"
              << "  catalogue: " << hcfg.catalogue_root << '\n'
              << "  base exact periods: 1.." << hcfg.base_max_period << '\n'
              << "  search through period: " << hcfg.max_period << '\n'
              << "  cutoff: " << real_string(hcfg.min_area, 4) << '\n'
              << "  seed parents from catalogue: " << base_seed_count << '\n'
              << "  resumed satellite components: " << resumed_count << '\n'
              << "  processed attachment tasks: " << processed.size() << '\n'
              << "  output: " << catalogue.export_path(hcfg.output_export) << '\n'
              << "  threads: " << hcfg.threads << "\n\n";

    std::size_t total_inserted = 0;
    for (int generation = 0; generation < hcfg.max_generations; ++generation) {
        std::vector<Task> tasks;
        const std::size_t parents_at_start = nodes.size();
        for (std::size_t index = 0; index < parents_at_start; ++index) {
            const auto& parent = nodes[index];
            const int qmax = std::min(
                hcfg.max_denominator, hcfg.max_period / parent.period);
            for (int q = 2; q <= qmax; ++q) {
                const int child_period = parent.period * q;
                if (child_period <= hcfg.base_max_period
                    || child_period > hcfg.max_period) continue;
                for (int p = 1; p < q; ++p) {
                    if (std::gcd(p, q) != 1) continue;
                    const std::string key = parent.id + "|"
                        + std::to_string(p) + "/" + std::to_string(q);
                    if (processed.contains(key)) continue;
                    tasks.push_back(Task{
                        index, p, q, key,
                        parent.address + "/" + std::to_string(p)
                            + "-" + std::to_string(q)});
                }
            }
        }
        if (tasks.empty()) break;

        std::cout << "generation " << generation + 1 << ": "
                  << tasks.size() << " candidate attachment(s)\n";
        std::vector<TaskResult> results(tasks.size());
        std::atomic<std::size_t> done{0};
        const auto started = Clock::now();
        parallel_for(tasks.size(), hcfg.threads,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    results[i] = process_task(
                        tasks[i], nodes[tasks[i].parent_index], hcfg, numerical);
                    const auto count = ++done;
                    if (count % 25 == 0 || count == tasks.size()) {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cout << "  hunter " << count << '/' << tasks.size()
                                  << " elapsed="
                                  << format_duration(Clock::now() - started) << '\n';
                    }
                }
            });

        std::map<std::string, int> counts;
        std::vector<Node> accepted_nodes;
        std::vector<std::size_t> accepted_result_indexes;
        std::vector<mandelbrot::catalogue::ComponentRecord> accepted_records;
        for (std::size_t result_index = 0; result_index < results.size(); ++result_index) {
            auto& result = results[result_index];
            if (result.node && near_known(
                    result.node->center, result.node->period,
                    nodes, hcfg.center_duplicate_tolerance)) {
                result.status = "duplicate";
                result.node.reset();
            }
            if (result.node) {
                accepted_records.push_back(node_to_component(*result.node));
                accepted_nodes.push_back(*result.node);
                accepted_result_indexes.push_back(result_index);
            }
        }

        std::set<int> generation_changed_periods;
        if (!accepted_records.empty()) {
            mandelbrot::catalogue::UpsertOptions upsert;
            upsert.center_tolerance = mandelbrot::catalogue::CatalogueReal(
                hcfg.center_duplicate_tolerance);
            upsert.bump_revision = true;
            const auto saved = catalogue.upsert_components(
                std::move(accepted_records), upsert);
            for (std::size_t i = 0; i < saved.size(); ++i) {
                accepted_nodes[i].id = saved[i].component.id;
                accepted_nodes[i].center = {
                    saved[i].component.center.re.convert_to<Real>(),
                    saved[i].component.center.im.convert_to<Real>()};
                results[accepted_result_indexes[i]].node = accepted_nodes[i];
                nodes.push_back(std::move(accepted_nodes[i]));
                if (saved[i].inserted || saved[i].updated) {
                    generation_changed_periods.insert(saved[i].component.period);
                }
                if (saved[i].inserted) ++total_inserted;
            }
            refresh_period_metadata(catalogue, hcfg, generation_changed_periods);
        }

        // Checkpoint only after accepted records are safely present in the
        // canonical catalogue.  A crash can therefore never mark a discovery
        // processed while losing its component record.
        std::ofstream checkpoint(hcfg.checkpoint_file, std::ios::app);
        if (!checkpoint) {
            throw std::runtime_error(
                "Could not append checkpoint: " + hcfg.checkpoint_file.string());
        }
        for (auto& result : results) {
            const Node& parent = nodes[result.task.parent_index];
            processed.insert(result.task.key);
            ++counts[result.status];
            append_checkpoint(checkpoint, result, parent);
        }
        checkpoint.close();
        if (!checkpoint) throw std::runtime_error("Failed while writing atlas checkpoint");

        std::cout << "  accepted=" << counts["accepted"]
                  << " below_cutoff=" << counts["below_cutoff"]
                  << " center_failed=" << counts["center_failed"]
                  << " trace_failed=" << counts["trace_failed"]
                  << " duplicate=" << counts["duplicate"] << '\n';
        if (accepted_nodes.empty()) break;
    }

    write_derived_outputs(catalogue, hcfg);
    std::cout << "\nDone. Added " << total_inserted
              << " higher-period satellite component(s) to the canonical catalogue.\n"
              << "This remains a targeted satellite-descendant search, not a proof "
                 "of completeness for primitive components.\n";
    return 0;
}

} // namespace hunter

int main(int argc, char** argv) {
    try { return hunter::run(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "atlas hunter error: " << error.what() << '\n';
        return 1;
    }
}
