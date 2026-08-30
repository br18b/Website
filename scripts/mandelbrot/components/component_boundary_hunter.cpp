#define main component_area_scan_embedded_main
#include "component_area_scan.cpp"
#undef main

#include <atomic>
#include <csignal>
#include <deque>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace boundary_hunter {

static volatile std::sig_atomic_t g_stop_requested = 0;
void stop_handler(int) { g_stop_requested = 1; }

struct BoundaryConfig {
    unsigned threads = 0;
    Real min_area = 1.0e-10L;
    Real polygon_rho = 0.99999L;
    Real area_rho = 0.99999L;
    int polygon_points = 192;
    int initial_boundary_rays = 32;
    int initial_normal_samples = 8;
    int max_boundary_levels = 2;
    Real normal_min_relative = 1.0e-6L;
    Real normal_max_relative = 0.35L;
    int orbit_burn_in = 128;
    std::uint64_t cycle_max_iterations = 200000;
    Real cycle_tolerance = 1.0e-10L;
    int cycle_confirmation_returns = 3;
    Real parent_gap_factor = 0.05L;
    Real center_duplicate_tolerance = 1.0e-9L;
    Real circle_rms_tolerance = 0.06L;
    bool use_conjugate_symmetry = true;
    std::size_t max_sources = 64;
    fs::path catalogue_root;
    fs::path run_dir;
    std::string output_export = "atlas_components_boundary.json";
    std::string skeleton_export = "component_skeleton.json";
    fs::path checkpoint_file;
};

struct Polygon {
    std::string id;
    int period = 0;
    Complex center{};
    Real area = 0;
    std::vector<Complex> points;
    std::array<Real, 4> bounds{};
};

struct TreeNode {
    std::string id;
    std::string parent;
    int period = 0;
    Complex center{};
    Real area = 0;
    std::string shape = "circle";
    std::vector<Complex> points;
    Complex attachment_parent{};
    Complex attachment_child{};
    Real attachment_gap = 0;
    bool processed = false;
    int generation = 0;
};

struct Candidate {
    int ray = -1;
    int period = 0;
    Complex center{};
    Complex sample{};
    Real normal_distance = 0;
    std::optional<std::size_t> known_index;
};

BoundaryConfig load_config(const fs::path& config_path, const Config& numerical, const char* argv0) {
    const auto repo = mandelbrot::repo::RepoConfig::load(config_path, executable_parent_or_cwd(argv0));
    BoundaryConfig cfg;
    cfg.catalogue_root = repo.path("paths.catalogue_root");
    mandelbrot::catalogue::Catalogue catalogue(cfg.catalogue_root);
    const std::string run_name = repo.string("component_boundary_hunter.run_name", "default");
    cfg.run_dir = catalogue.run_path("boundary_hunter", run_name);
    cfg.threads = repo.threads();
    const std::string prefix = "component_boundary_hunter.";
    cfg.min_area = static_cast<Real>(repo.number(prefix + "min_area", cfg.min_area));
    cfg.polygon_rho = static_cast<Real>(repo.number(prefix + "polygon_rho", cfg.polygon_rho));
    cfg.area_rho = static_cast<Real>(repo.number(prefix + "area_rho", cfg.area_rho));
    cfg.polygon_points = repo.integer(prefix + "polygon_points", cfg.polygon_points);
    cfg.initial_boundary_rays = repo.integer(prefix + "initial_boundary_rays", cfg.initial_boundary_rays);
    cfg.initial_normal_samples = repo.integer(prefix + "initial_normal_samples", cfg.initial_normal_samples);
    cfg.max_boundary_levels = repo.integer(prefix + "max_boundary_levels", cfg.max_boundary_levels);
    cfg.normal_min_relative = static_cast<Real>(repo.number(prefix + "normal_min_relative", cfg.normal_min_relative));
    cfg.normal_max_relative = static_cast<Real>(repo.number(prefix + "normal_max_relative", cfg.normal_max_relative));
    cfg.orbit_burn_in = repo.integer(prefix + "orbit_burn_in", cfg.orbit_burn_in);
    cfg.cycle_max_iterations = repo.u64(prefix + "cycle_max_iterations", cfg.cycle_max_iterations);
    cfg.cycle_tolerance = static_cast<Real>(repo.number(prefix + "cycle_tolerance", cfg.cycle_tolerance));
    cfg.cycle_confirmation_returns = repo.integer(prefix + "cycle_confirmation_returns", cfg.cycle_confirmation_returns);
    cfg.parent_gap_factor = static_cast<Real>(repo.number(prefix + "parent_gap_factor", cfg.parent_gap_factor));
    cfg.center_duplicate_tolerance = static_cast<Real>(repo.number(prefix + "center_duplicate_tolerance", cfg.center_duplicate_tolerance));
    cfg.circle_rms_tolerance = static_cast<Real>(repo.number(prefix + "circle_rms_tolerance", cfg.circle_rms_tolerance));
    cfg.use_conjugate_symmetry = repo.boolean(prefix + "use_conjugate_symmetry", cfg.use_conjugate_symmetry);
    cfg.max_sources = static_cast<std::size_t>(repo.u64(prefix + "max_sources", cfg.max_sources));
    cfg.output_export = repo.string(prefix + "output_export", cfg.output_export);
    cfg.skeleton_export = repo.string(prefix + "skeleton_export", cfg.skeleton_export);
    cfg.checkpoint_file = catalogue.run_path("boundary_hunter", run_name, "checkpoint.tsv");
    cfg.polygon_points = std::max(64, cfg.polygon_points);
    cfg.initial_boundary_rays = std::max(8, cfg.initial_boundary_rays);
    cfg.initial_normal_samples = std::max(2, cfg.initial_normal_samples);
    cfg.max_boundary_levels = std::clamp(cfg.max_boundary_levels, 0, 8);
    cfg.orbit_burn_in = std::max(8, cfg.orbit_burn_in);
    cfg.cycle_confirmation_returns = std::max(2, cfg.cycle_confirmation_returns);
    if (!(cfg.normal_min_relative > 0 && cfg.normal_max_relative > cfg.normal_min_relative)) {
        throw std::runtime_error("Invalid component_boundary_hunter normal range.");
    }
    fs::create_directories(cfg.run_dir);
    (void)numerical;
    return cfg;
}

std::array<Real,4> bbox(const std::vector<Complex>& points) {
    if (points.empty()) return {0,0,0,0};
    Real xmin=points[0].real(), xmax=xmin, ymin=points[0].imag(), ymax=ymin;
    for (auto p: points) { xmin=std::min(xmin,p.real()); xmax=std::max(xmax,p.real()); ymin=std::min(ymin,p.imag()); ymax=std::max(ymax,p.imag()); }
    return {xmin,xmax,ymin,ymax};
}

Real polygon_area_value(const std::vector<Complex>& points) {
    Real sum=0;
    for (std::size_t i=0;i<points.size();++i) {
        const auto a=points[i], b=points[(i+1)%points.size()];
        sum += a.real()*b.imag()-a.imag()*b.real();
    }
    return std::abs(sum)/2;
}

bool point_in_polygon(Complex point, const Polygon& polygon) {
    if (point.real()<polygon.bounds[0] || point.real()>polygon.bounds[1]
        || point.imag()<polygon.bounds[2] || point.imag()>polygon.bounds[3]) return false;
    bool inside=false;
    for (std::size_t i=0,j=polygon.points.size()-1;i<polygon.points.size();j=i++) {
        const auto a=polygon.points[i], b=polygon.points[j];
        if ((a.imag()>point.imag()) != (b.imag()>point.imag())) {
            const Real x=(b.real()-a.real())*(point.imag()-a.imag())/(b.imag()-a.imag())+a.real();
            if (point.real()<x) inside=!inside;
        }
    }
    return inside;
}

Real segment_distance(Complex p, Complex a, Complex b, Complex* closest=nullptr) {
    const Complex e=b-a;
    const Real n=std::norm(e);
    Real t=0;
    if (n>1e-36L) t=std::clamp(((p-a)*std::conj(e)).real()/n, Real(0), Real(1));
    const Complex q=a+t*e;
    if (closest) *closest=q;
    return safe_abs(p-q);
}

std::tuple<Real,Complex,Complex> polygon_gap(const Polygon& a, const Polygon& b) {
    Real best=std::numeric_limits<Real>::infinity();
    Complex pa{},pb{};
    for (auto p:a.points) for (std::size_t j=0;j<b.points.size();++j) {
        Complex q;
        const Real d=segment_distance(p,b.points[j],b.points[(j+1)%b.points.size()],&q);
        if(d<best){best=d;pa=p;pb=q;}
    }
    for (auto p:b.points) for (std::size_t j=0;j<a.points.size();++j) {
        Complex q;
        const Real d=segment_distance(p,a.points[j],a.points[(j+1)%a.points.size()],&q);
        if(d<best){best=d;pa=q;pb=p;}
    }
    return {best,pa,pb};
}

Polygon polygon_from_component(const mandelbrot::catalogue::ComponentRecord& component) {
    Polygon polygon;
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

std::vector<Polygon> load_geometry(
    const mandelbrot::catalogue::CatalogueSnapshot& snapshot) {
    std::vector<Polygon> result;
    result.reserve(snapshot.components.size());
    for (const auto& component : snapshot.components) {
        if (component.geometry.polygon.size() < 3) continue;
        result.push_back(polygon_from_component(component));
    }
    return result;
}

std::vector<Complex> resample_polygon(const std::vector<Complex>& points, int count) {
    std::vector<Real> cumulative(points.size()+1,0);
    for(std::size_t i=0;i<points.size();++i)cumulative[i+1]=cumulative[i]+safe_abs(points[(i+1)%points.size()]-points[i]);
    const Real total=cumulative.back();
    std::vector<Complex> result; result.reserve(count);
    std::size_t edge=0;
    for(int k=0;k<count;++k){
        const Real target=total*Real(k)/count;
        while(edge+1<cumulative.size() && cumulative[edge+1]<target)++edge;
        const std::size_t i=edge%points.size();
        const Real len=cumulative[edge+1]-cumulative[edge];
        const Real t=len>0?(target-cumulative[edge])/len:0;
        result.push_back(points[i]+t*(points[(i+1)%points.size()]-points[i]));
    }
    return result;
}

Complex outward_normal(const std::vector<Complex>& ring, int i, const Polygon& source) {
    const int n=static_cast<int>(ring.size());
    Complex tangent=ring[(i+1)%n]-ring[(i+n-1)%n];
    const Real len=safe_abs(tangent);
    if(len==0) return {1,0};
    tangent/=len;
    Complex normal{-tangent.imag(),tangent.real()};
    const Real scale=std::max<Real>(1e-12L,std::sqrt(std::max<Real>(source.area,1e-30L)/PI));
    const Real eps=scale*1e-6L;
    if(point_in_polygon(ring[i]+eps*normal,source)) normal=-normal;
    return normal;
}

std::vector<Real> normal_distances(const BoundaryConfig& cfg, Real size) {
    std::vector<Real> values;
    values.reserve(cfg.initial_normal_samples);
    const long double ratio=cfg.normal_max_relative/cfg.normal_min_relative;
    for(int i=0;i<cfg.initial_normal_samples;++i){
        const Real f=cfg.initial_normal_samples==1?0:Real(i)/(cfg.initial_normal_samples-1);
        values.push_back(size*cfg.normal_min_relative*std::pow(ratio,f));
    }
    return values;
}

bool close_complex(Complex a, Complex b, Real tolerance) {
    return safe_abs(a-b)<=tolerance*std::max<Real>({1,safe_abs(a),safe_abs(b)});
}

std::optional<int> brent_period(Complex c, const BoundaryConfig& cfg) {
    auto f=[c](Complex z){return z*z+c;};
    Complex z{0,0};
    for(int i=0;i<cfg.orbit_burn_in;++i){z=f(z); if(!finite(z)||std::norm(z)>4)return std::nullopt;}
    Complex tortoise=z;
    Complex hare=f(z);
    std::uint64_t power=1, lam=1, steps=1;
    while(steps<cfg.cycle_max_iterations && !close_complex(tortoise,hare,cfg.cycle_tolerance)){
        if(!finite(hare)||std::norm(hare)>4)return std::nullopt;
        if(power==lam){tortoise=hare;power*=2;lam=0;}
        hare=f(hare);++lam;++steps;
    }
    if(steps>=cfg.cycle_max_iterations || lam==0 || lam>static_cast<std::uint64_t>(std::numeric_limits<int>::max())) return std::nullopt;
    const int period=static_cast<int>(lam);
    Complex check=hare;
    for(int r=0;r<cfg.cycle_confirmation_returns;++r){
        Complex next=check;
        for(int k=0;k<period;++k) next=f(next);
        if(!close_complex(check,next,10*cfg.cycle_tolerance)) return std::nullopt;
        check=next;
    }
    return period;
}

std::optional<MultiplierState> refine_attracting_cycle(Complex c, int period,
                                                        const BoundaryConfig& cfg,
                                                        const Config& numerical) {
    Complex z{0,0};
    const int settle=std::max(cfg.orbit_burn_in, std::min<int>(100000, 8*period+64));
    for(int i=0;i<settle;++i){z=z*z+c;if(!finite(z)||std::norm(z)>4)return std::nullopt;}
    for(int iteration=0;iteration<100;++iteration){
        const auto data=iterate_data(z,c,period);
        const Complex residual=data.A-z;
        if(safe_abs(residual)<=numerical.newton_tolerance*std::max<Real>(1,safe_abs(z))){
            if(!periodic_point_has_exact_period(period,z,c,numerical.exact_period_tolerance)) return std::nullopt;
            if(!(safe_abs(data.B)<1)) return std::nullopt;
            return state_from_solution(period,data.B,z,c,iteration);
        }
        const Complex derivative=data.B-Complex{1,0};
        if(safe_abs(derivative)<1e-28L)return std::nullopt;
        z-=residual/derivative;
    }
    return std::nullopt;
}

std::optional<std::pair<int,Complex>> classify_center(Complex c, const BoundaryConfig& cfg,
                                                       const Config& numerical) {
    const auto period=brent_period(c,cfg);
    if(!period) return std::nullopt;
    auto state=refine_attracting_cycle(c,*period,cfg,numerical);
    if(!state) return std::nullopt;
    ContinuationStats stats;
    auto centered=continue_to_lambda(*period,*state,Complex{0,0},numerical,stats);
    if(!centered)return std::nullopt;
    const int exact=detected_center_period(centered->c,*period,numerical.exact_period_tolerance);
    if(exact<=0)return std::nullopt;
    return std::pair<int,Complex>{exact,centered->c};
}

std::optional<Polygon> trace_polygon(Complex center, int period, const BoundaryConfig& cfg,
                                     const Config& numerical) {
    auto state=center_state(period,center,numerical); if(!state)return std::nullopt;
    std::vector<Real> stages{0.9L,0.99L,0.999L,cfg.polygon_rho,cfg.area_rho};
    std::sort(stages.begin(),stages.end()); stages.erase(std::unique(stages.begin(),stages.end()),stages.end());
    std::optional<RingTrace> seed, polygon_ring, area_ring;
    for(Real rho:stages){auto ring=trace_ring(period,*state,rho,cfg.polygon_points,numerical,seed?&*seed:nullptr);if(!ring)return std::nullopt; if(std::abs(rho-cfg.polygon_rho)<1e-18L)polygon_ring=*ring;if(std::abs(rho-cfg.area_rho)<1e-18L)area_ring=*ring;seed=std::move(*ring);}
    if(!polygon_ring||!area_ring)return std::nullopt;
    std::vector<Complex> offsets(area_ring->c.size());for(std::size_t i=0;i<offsets.size();++i)offsets[i]=area_ring->c[i]-center;
    Polygon p;p.period=period;p.center=center;p.area=std::abs(derivative_area(offsets,area_ring->lambda,area_ring->c_lambda));p.points=polygon_ring->c;p.bounds=bbox(p.points);return p;
}

std::string classify_shape(const Polygon& p, Real tolerance) {
    const int n=static_cast<int>(p.points.size());
    if(n<3)return "circle";
    Real mx=0,my=0;for(auto z:p.points){mx+=z.real();my+=z.imag();}mx/=n;my/=n;
    Real r=0;for(auto z:p.points)r+=safe_abs(z-Complex{mx,my});r/=n;
    Real e2=0;for(auto z:p.points){Real d=safe_abs(z-Complex{mx,my})-r;e2+=d*d;}
    const Real rms=std::sqrt(e2/n)/std::max<Real>(r,1e-30L);
    return rms<=tolerance?"circle":"cardioid";
}

std::optional<std::size_t> nearest_known(Complex center,int period,const std::vector<Polygon>& polygons,Real tol){
    for(std::size_t i=0;i<polygons.size();++i)if(polygons[i].period==period&&safe_abs(center-polygons[i].center)<=tol)return i;
    return std::nullopt;
}

std::optional<std::size_t> containing_known(Complex c,const std::vector<Polygon>& polygons,std::size_t source_index){
    for(std::size_t i=0;i<polygons.size();++i){if(i==source_index)continue;if(point_in_polygon(c,polygons[i]))return i;}return std::nullopt;
}

mandelbrot::catalogue::ComponentRecord node_to_component(
    const TreeNode& node,
    const BoundaryConfig& cfg,
    const std::string& hierarchy_root) {
    using namespace mandelbrot::catalogue;
    ComponentRecord component;
    component.id = node.id;
    component.period = node.period;
    component.center = {
        CatalogueReal(real_string(node.center.real())),
        CatalogueReal(real_string(node.center.imag()))};
    component.numeric.working_precision_digits = std::numeric_limits<Real>::max_digits10;
    component.numeric.validated_digits = std::max(0, std::numeric_limits<Real>::digits10 - 2);
    component.geometry.polygon_rho = CatalogueReal(real_string(cfg.polygon_rho));
    component.geometry.polygon_area = CatalogueReal(real_string(polygon_area_value(node.points)));
    component.geometry.area_estimate = CatalogueReal(real_string(node.area));
    component.geometry.area_rho = CatalogueReal(real_string(cfg.area_rho));
    component.geometry.characteristic_size =
        boost::multiprecision::sqrt(component.geometry.area_estimate / CatalogueReal(PI));
    const auto bounds = bbox(node.points);
    component.geometry.bbox_centered = {
        CatalogueReal(real_string(bounds[0] - node.center.real())),
        CatalogueReal(real_string(bounds[1] - node.center.real())),
        CatalogueReal(real_string(bounds[2] - node.center.imag())),
        CatalogueReal(real_string(bounds[3] - node.center.imag()))};
    component.geometry.polygon.reserve(node.points.size());
    for (const auto& point : node.points) {
        component.geometry.polygon.push_back({
            CatalogueReal(real_string(point.real() - node.center.real())),
            CatalogueReal(real_string(point.imag() - node.center.imag()))});
    }
    component.classification.shape_class = "unknown";
    component.classification.shape_confidence = 0;
    component.hierarchy.generation = node.generation;
    if (node.shape == "circle" && !node.parent.empty()) {
        component.hierarchy.geometric_parent = node.parent;
        component.hierarchy.hierarchy_root = hierarchy_root;
        AttachmentRecord attachment;
        attachment.parent_point = ComplexValue{
            CatalogueReal(real_string(node.attachment_parent.real())),
            CatalogueReal(real_string(node.attachment_parent.imag()))};
        attachment.child_point_centered = ComplexValue{
            CatalogueReal(real_string(node.attachment_child.real() - node.center.real())),
            CatalogueReal(real_string(node.attachment_child.imag() - node.center.imag()))};
        attachment.gap = CatalogueReal(real_string(node.attachment_gap));
        attachment.gap_relative_to_child_size = component.geometry.characteristic_size == 0
            ? CatalogueReal(0)
            : attachment.gap.value() / component.geometry.characteristic_size;
        attachment.verified = true;
        component.hierarchy.attachment = attachment;
    } else {
        component.hierarchy.hierarchy_root = component.id;
    }
    component.provenance.method = "boundary-hunter";
    component.provenance.run_id = cfg.run_dir.filename().string();
    component.provenance.discovered_at = utc_timestamp();
    component.provenance.aliases = {"boundary-node:" + node.id};
    component.quality.center_validated = true;
    component.quality.exact_period_validated = true;
    component.quality.polygon_converged = true;
    component.quality.area_above_cutoff = node.area >= cfg.min_area;
    return component;
}

bool save_nodes_to_catalogue(
    mandelbrot::catalogue::Catalogue& catalogue,
    const BoundaryConfig& cfg,
    std::vector<TreeNode>& nodes) {
    using namespace mandelbrot::catalogue;
    if (nodes.empty()) return false;

    std::vector<ComponentRecord> records;
    records.reserve(nodes.size());
    for (const auto& node : nodes) {
        records.push_back(node_to_component(node, cfg, nodes.front().id));
    }
    UpsertOptions options;
    options.center_tolerance = CatalogueReal(real_string(cfg.center_duplicate_tolerance));
    options.merge_existing = true;
    options.bump_revision = true;
    const auto results = catalogue.upsert_components(std::move(records), options);

    std::map<std::string, std::string> remapped_ids;
    std::set<int> changed_periods;
    bool changed = false;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (nodes[i].id != results[i].component.id) {
            remapped_ids[nodes[i].id] = results[i].component.id;
            nodes[i].id = results[i].component.id;
        }
        if (results[i].inserted || results[i].updated) {
            changed_periods.insert(results[i].component.period);
            changed = true;
        }
    }
    if (!remapped_ids.empty()) {
        for (auto& node : nodes) {
            if (const auto found = remapped_ids.find(node.parent);
                found != remapped_ids.end()) {
                node.parent = found->second;
            }
        }
    }

    if (!changed_periods.empty()) {
        catalogue.rebuild_period_indexes(
            std::vector<int>(changed_periods.begin(), changed_periods.end()),
            CatalogueReal(real_string(cfg.min_area)));
        catalogue.rebuild_manifest_from_period_indexes(
            0, CatalogueReal(real_string(cfg.min_area)));
    }

    // The current search already owns a typed tree in memory, so persist that
    // tree directly instead of rescanning every component file after each
    // source component.
    HierarchyTree tree;
    tree.root = nodes.front().id;
    tree.node_count = nodes.size();
    tree.minimum_stored_area = CatalogueReal(real_string(cfg.min_area));
    tree.complete_above_cutoff = false;
    tree.generated_from_catalogue_revision = catalogue.load_manifest().catalogue_revision;
    std::unordered_map<std::string, std::vector<std::string>> children;
    for (const auto& node : nodes) {
        if (!node.parent.empty()) children[node.parent].push_back(node.id);
        tree.maximum_known_generation = std::max(tree.maximum_known_generation, node.generation);
        tree.known_area += CatalogueReal(real_string(node.area));
    }
    tree.nodes.reserve(nodes.size());
    for (const auto& node : nodes) {
        HierarchyNode item;
        item.id = node.id;
        if (!node.parent.empty()) item.parent = node.parent;
        item.children = children[node.id];
        std::sort(item.children.begin(), item.children.end());
        tree.nodes.push_back(std::move(item));
    }
    catalogue.save_hierarchy(tree);
    return changed;
}

void write_derived_outputs(
    mandelbrot::catalogue::Catalogue& catalogue,
    const BoundaryConfig& cfg) {
    mandelbrot::catalogue::ComponentQuery query;
    query.min_area = mandelbrot::catalogue::CatalogueReal(real_string(cfg.min_area));
    query.require_polygon = true;
    query.require_polygon_converged = true;
    mandelbrot::catalogue::ComponentExportOptions export_options;
    export_options.query = query;
    export_options.format = "mandelbrot-boundary-hierarchy-v2";
    export_options.complete = false;
    export_options.coordinate_digits = std::numeric_limits<Real>::max_digits10;
    catalogue.write_component_export(
        catalogue.export_path(cfg.output_export), export_options);
    catalogue.write_skeleton_export(
        catalogue.export_path(cfg.skeleton_export), query);
}

void write_checkpoint(const BoundaryConfig& cfg,const std::vector<TreeNode>& nodes){
    fs::create_directories(cfg.checkpoint_file.parent_path());fs::path tmp=cfg.checkpoint_file.string()+".tmp";std::ofstream out(tmp);out<<"id\tparent\tperiod\tcenter_re\tcenter_im\tarea\tshape\tprocessed\tgeneration\tattach_parent_re\tattach_parent_im\tattach_child_re\tattach_child_im\tgap\tpoints\n"<<std::setprecision(std::numeric_limits<Real>::max_digits10)<<std::scientific;
    for(const auto& n:nodes){out<<n.id<<'\t'<<n.parent<<'\t'<<n.period<<'\t'<<n.center.real()<<'\t'<<n.center.imag()<<'\t'<<n.area<<'\t'<<n.shape<<'\t'<<(n.processed?1:0)<<'\t'<<n.generation<<'\t'<<n.attachment_parent.real()<<'\t'<<n.attachment_parent.imag()<<'\t'<<n.attachment_child.real()<<'\t'<<n.attachment_child.imag()<<'\t'<<n.attachment_gap<<'\t';for(std::size_t i=0;i<n.points.size();++i){if(i)out<<';';out<<n.points[i].real()<<':'<<n.points[i].imag();}out<<'\n';}out.close();atomic_replace(tmp,cfg.checkpoint_file);
}

std::vector<TreeNode> load_checkpoint(const fs::path& path){std::vector<TreeNode> nodes;std::ifstream in(path);if(!in)return nodes;std::string line;std::getline(in,line);while(std::getline(in,line)){auto f=split(line,'\t');if(f.size()<15)continue;try{TreeNode n;n.id=f[0];n.parent=f[1];n.period=std::stoi(f[2]);n.center={std::stold(f[3]),std::stold(f[4])};n.area=std::stold(f[5]);n.shape=f[6];n.processed=std::stoi(f[7])!=0;n.generation=std::stoi(f[8]);n.attachment_parent={std::stold(f[9]),std::stold(f[10])};n.attachment_child={std::stold(f[11]),std::stold(f[12])};n.attachment_gap=std::stold(f[13]);for(auto pair:split(f[14],';')){auto pos=pair.find(':');if(pos!=std::string::npos)n.points.emplace_back(std::stold(pair.substr(0,pos)),std::stold(pair.substr(pos+1)));}if(!n.points.empty())nodes.push_back(std::move(n));}catch(...){}}return nodes;}

void canonicalize_checkpoint_ids(
    std::vector<TreeNode>& nodes,
    const mandelbrot::catalogue::CatalogueSnapshot& snapshot,
    Real tolerance) {
    using namespace mandelbrot::catalogue;
    std::unordered_map<std::string, std::string> remap;
    for (const auto& node : nodes) {
        ComplexValue center{
            CatalogueReal(real_string(node.center.real())),
            CatalogueReal(real_string(node.center.imag()))};
        if (const auto* existing = snapshot.find_near_center(
                node.period, center, CatalogueReal(real_string(tolerance)))) {
            remap[node.id] = existing->id;
        } else {
            remap[node.id] = Catalogue::stable_id(
                "boundary:" + std::to_string(node.period) + ":"
                + real_string(node.center.real()) + ":"
                + real_string(std::abs(node.center.imag())));
        }
    }
    for (auto& node : nodes) {
        const std::string old_id = node.id;
        node.id = remap.at(old_id);
        if (!node.parent.empty()) {
            if (const auto found = remap.find(node.parent); found != remap.end()) {
                node.parent = found->second;
            }
        }
    }
}

int run(int argc,char** argv){
    const std::string usage="Usage: component_boundary_hunter [--config PATH] [--reset]";
    const auto cli=mandelbrot::repo::parse_common_cli(argc,argv,usage);
    if(cli.help){std::cout<<usage<<'\n';return 0;}
    bool reset=false;
    for(const auto& option:cli.remaining){if(option=="--reset")reset=true;else throw std::runtime_error("Unknown option: "+option+"\n"+usage);}
    Config numerical=read_repository_config(cli.config,argv[0]);numerical.compute_areas=false;numerical.newton_max_iterations=std::max(numerical.newton_max_iterations,80);numerical.continuation_max_depth=std::max(numerical.continuation_max_depth,28);numerical.continuation_max_step=std::min<Real>(numerical.continuation_max_step,0.025L);
    BoundaryConfig cfg=load_config(cli.config,numerical,argv[0]);
    mandelbrot::catalogue::Catalogue catalogue(cfg.catalogue_root);
    catalogue.ensure_layout();
    std::signal(SIGINT,stop_handler);std::signal(SIGTERM,stop_handler);
    if(reset){
        std::error_code ec;
        fs::remove(catalogue.export_path(cfg.output_export),ec);
        fs::remove(catalogue.export_path(cfg.skeleton_export),ec);
        fs::remove(cfg.checkpoint_file,ec);
    }
    mandelbrot::catalogue::ComponentQuery seed_query;
    seed_query.require_polygon = true;
    seed_query.require_polygon_converged = true;
    auto snapshot = catalogue.load_snapshot(seed_query);
    std::vector<Polygon> known=load_geometry(snapshot);
    auto root_match=nearest_known(Complex{0,0},1,known,1e-12L);
    if(!root_match){
        throw std::runtime_error(
            "The canonical catalogue has no validated period-1 component. "
            "Run component_area_scan before component_boundary_hunter.");
    }
    const Polygon root=known[*root_match];
    std::vector<TreeNode> nodes=load_checkpoint(cfg.checkpoint_file);
    if(nodes.empty()){
        TreeNode n;
        n.id=root.id;n.period=1;n.center=root.center;n.area=root.area;
        n.shape="cardioid";n.points=root.points;n.generation=0;
        nodes.push_back(std::move(n));write_checkpoint(cfg,nodes);
    } else {
        canonicalize_checkpoint_ids(nodes,snapshot,cfg.center_duplicate_tolerance);
    }
    std::cout<<"Mandelbrot boundary-guided component hunter\n  seed polygons: "<<known.size()<<"\n  resumed hierarchy nodes: "<<nodes.size()<<"\n  min area: "<<real_string(cfg.min_area,4)<<"\n  rays: "<<cfg.initial_boundary_rays<<" x up to 2^"<<cfg.max_boundary_levels<<"\n  normal samples: "<<cfg.initial_normal_samples<<" (log-spaced)\n  period ceiling: none (Brent cycle detection)\n  threads: "<<cfg.threads<<"\n  output: "<<catalogue.export_path(cfg.output_export)<<"\n\n";
    std::size_t processed_sources=0;
    while(processed_sources<cfg.max_sources){
        auto it=std::find_if(nodes.begin(),nodes.end(),[](const TreeNode& n){return !n.processed;});
        if(it==nodes.end())break;
        const std::size_t source_node_index=static_cast<std::size_t>(it-nodes.begin());
        TreeNode& source_node=nodes[source_node_index];
        Polygon source;source.id=source_node.id;source.period=source_node.period;source.center=source_node.center;source.area=source_node.area;source.points=source_node.points;source.bounds=bbox(source.points);
        const Real source_size=std::sqrt(std::max<Real>(source.area,1e-30L)/PI);
        std::cout<<"source "<<source.id<<" period="<<source.period<<" area="<<real_string(source.area,4)<<" generation="<<source_node.generation<<"\n";
        std::vector<Candidate> all_candidates;
        std::unordered_set<int> disabled_base_rays;
        for(int level=0;level<=cfg.max_boundary_levels && !g_stop_requested;++level){
            const int rays=cfg.initial_boundary_rays<<level;
            auto ring=resample_polygon(source.points,rays);
            auto distances=normal_distances(cfg,source_size);
            std::vector<std::optional<Candidate>> hits(rays);
            std::atomic<std::size_t> done{0};
            parallel_for(rays,cfg.threads,[&](std::size_t begin,std::size_t end){
                for(std::size_t ri=begin;ri<end;++ri){
                    const int base_ray=static_cast<int>(ri)>>level;
                    if(disabled_base_rays.count(base_ray)){++done;continue;}
                    const Complex normal=outward_normal(ring,static_cast<int>(ri),source);
                    for(Real distance:distances){
                        Complex probe=ring[ri]+distance*normal;
                        if(cfg.use_conjugate_symmetry && probe.imag()<0)continue;
                        if(point_in_polygon(probe,source))continue;
                        if(auto known_hit=containing_known(probe,known,std::numeric_limits<std::size_t>::max())){
                            if(safe_abs(known[*known_hit].center-source.center)<=cfg.center_duplicate_tolerance)continue;
                            hits[ri]=Candidate{static_cast<int>(ri),known[*known_hit].period,known[*known_hit].center,probe,distance,*known_hit};break;
                        }
                        auto classified=classify_center(probe,cfg,numerical);
                        if(!classified)continue;
                        if(safe_abs(classified->second-source.center)<=cfg.center_duplicate_tolerance)continue;
                        hits[ri]=Candidate{static_cast<int>(ri),classified->first,classified->second,probe,distance,std::nullopt};break;
                    }
                    const auto d=++done;
                    if(d%std::max<std::size_t>(1,rays/10)==0||d==static_cast<std::size_t>(rays)){
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cout<<"\r  level "<<level<<" rays "<<d<<'/'<<rays<<"   "<<std::flush;
                    }
                }
            });
            std::cout<<'\n';
            for(int ri=0;ri<rays;++ri)if(hits[ri]){all_candidates.push_back(*hits[ri]);disabled_base_rays.insert(ri>>level);}
            if(disabled_base_rays.size()>=static_cast<std::size_t>(cfg.initial_boundary_rays))break;
        }
        std::sort(all_candidates.begin(),all_candidates.end(),[](const Candidate&a,const Candidate&b){if(a.ray!=b.ray)return a.ray<b.ray;return a.normal_distance<b.normal_distance;});
        std::vector<TreeNode> children;
        for(const auto& candidate:all_candidates){
            bool already=false;for(const auto& n:nodes)if(n.period==candidate.period&&safe_abs(n.center-candidate.center)<=cfg.center_duplicate_tolerance){already=true;break;}if(already)continue;
            Polygon child;
            if(candidate.known_index)child=known[*candidate.known_index];else{auto traced=trace_polygon(candidate.center,candidate.period,cfg,numerical);if(!traced)continue;child=std::move(*traced);if(child.area<cfg.min_area)continue;known.push_back(child);}
            auto [gap,pa,pb]=polygon_gap(source,child);
            const Real child_size=std::sqrt(std::max<Real>(child.area,1e-30L)/PI);
            if(gap>cfg.parent_gap_factor*child_size)continue;
            bool duplicate_child=false;
            for(const auto& existing:children){
                if(existing.period==child.period
                    && safe_abs(existing.center-child.center)<=cfg.center_duplicate_tolerance){
                    duplicate_child=true;
                    break;
                }
            }
            if(duplicate_child) continue;
            TreeNode n;
            n.id = !child.id.empty() ? child.id : mandelbrot::catalogue::Catalogue::stable_id(
                "boundary:" + std::to_string(child.period) + ":"
                + real_string(child.center.real()) + ":"
                + real_string(std::abs(child.center.imag())));
            n.parent=source.id;
            n.period=child.period;
            n.center=child.center;
            n.area=child.area;
            n.shape=classify_shape(child,cfg.circle_rms_tolerance);
            n.points=child.points;
            n.attachment_parent=pa;
            n.attachment_child=pb;
            n.attachment_gap=gap;
            n.generation=source_node.generation+1;
            children.push_back(std::move(n));
        }
        std::sort(children.begin(),children.end(),[&](const TreeNode&a,const TreeNode&b){return std::arg(a.attachment_parent-source.center)<std::arg(b.attachment_parent-source.center);});
        for(auto& child:children) nodes.push_back(std::move(child));
        nodes[source_node_index].processed=true;++processed_sources;
        save_nodes_to_catalogue(catalogue, cfg, nodes);
        write_checkpoint(cfg,nodes);
        std::cout<<"  accepted direct children: "<<children.size()<<"; hierarchy nodes="<<nodes.size()<<"\n\n";
        if(g_stop_requested){
            write_derived_outputs(catalogue, cfg);
            std::cout<<"Stop requested; checkpoint saved after current source.\n";
            return 130;
        }
    }
    save_nodes_to_catalogue(catalogue, cfg, nodes);
    write_checkpoint(cfg,nodes);
    write_derived_outputs(catalogue, cfg);
    std::cout<<"Done. Processed "<<processed_sources<<" source component(s); hierarchy contains "<<nodes.size()<<" node(s).\n"
             <<"  catalogue:  "<<cfg.catalogue_root<<'\n'
             <<"  components: "<<catalogue.export_path(cfg.output_export)<<'\n'
             <<"  skeleton:   "<<catalogue.export_path(cfg.skeleton_export)<<'\n'
             <<"  checkpoint: "<<cfg.checkpoint_file<<'\n';
    if(processed_sources>=cfg.max_sources)std::cout<<"Stopped at component_boundary_hunter.max_sources; rerun to continue.\n";
    return 0;
}

} // namespace boundary_hunter

int main(int argc,char** argv){try{return boundary_hunter::run(argc,argv);}catch(const std::exception& e){std::cerr<<"boundary hunter error: "<<e.what()<<'\n';return 1;}}
