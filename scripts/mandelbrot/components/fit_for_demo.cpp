#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include "common/repo_config.hpp"
#include "components/catalogue/component_catalogue.hpp"
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using Complex = std::complex<double>;

struct Options {
    fs::path config;
    fs::path catalogue_root;
    fs::path output;
    double min_area = 1e-10;
    int max_period = std::numeric_limits<int>::max();
    int digits = 11;
    double circle_rms = 0.055;
    double circle_max = 0.16;
    int cusp_candidates = 9;
    double duplicate_tolerance = 1e-9;
    unsigned threads = 0;
};

struct RawComponent {
    std::string id;
    int period = 0;
    int index = -1;
    long long discovery_index = -1;
    std::string period_count;
    Complex center{};
    double area = 0;
    std::vector<Complex> points;
    std::string source;
    std::string shape_class = "unknown";
    std::optional<Complex> fitted_center_centered;
    std::optional<double> fitted_radius;
    std::optional<double> fitted_cardioid_size;
    double fitted_angle = 0;
    double fitted_xi = 0;
    double fitted_rms = 0;
    double fitted_max_error = 0;
};

struct Fit {
    std::string shape;
    Complex shape_center{};
    double size = 0;
    double angle = 0;
    double xi = 0;
    double rms = 0;
    double max_error = 0;
    double circle_rms = 0;
    double circle_max = 0;
};

static bool solve3(std::array<std::array<double, 3>, 3> a, std::array<double, 3> b,
                   std::array<double, 3>& x) {
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
        if (std::abs(a[pivot][col]) < 1e-30) return false;
        std::swap(a[pivot], a[col]); std::swap(b[pivot], b[col]);
        const double inv = 1.0 / a[col][col];
        for (int j = col; j < 3; ++j) a[col][j] *= inv;
        b[col] *= inv;
        for (int row = 0; row < 3; ++row) if (row != col) {
            const double f = a[row][col];
            for (int j = col; j < 3; ++j) a[row][j] -= f * a[col][j];
            b[row] -= f * b[col];
        }
    }
    x = b; return true;
}

static Fit fit_circle(const RawComponent& c) {
    const auto& p = c.points;
    double sx=0, sy=0, s1=p.size(), sxx=0, syy=0, sxy=0, bx=0, by=0, b1=0;
    for (auto z : p) {
        double x=z.real(), y=z.imag(), q=x*x+y*y;
        sx += 2*x; sy += 2*y; sxx += 4*x*x; syy += 4*y*y; sxy += 4*x*y;
        bx += 2*x*q; by += 2*y*q; b1 += q;
    }
    std::array<std::array<double,3>,3> A{{{sxx,sxy,sx},{sxy,syy,sy},{sx,sy,s1}}};
    std::array<double,3> b{{bx,by,b1}}, sol{};
    double cx=0, cy=0, r=0;
    if (solve3(A,b,sol)) {
        cx=sol[0]; cy=sol[1]; double r2=sol[2]+cx*cx+cy*cy;
        if (r2 > 0 && std::isfinite(r2)) r=std::sqrt(r2);
    }
    if (!(r>0)) {
        for (auto z:p) { cx += z.real(); cy += z.imag(); }
        cx/=p.size(); cy/=p.size();
        for (auto z:p) r += std::abs(z-Complex(cx,cy));
        r/=p.size();
    }
    for (int iter=0; iter<12; ++iter) {
        std::array<std::array<double,3>,3> N{}; std::array<double,3> rhs{};
        for (auto z:p) {
            double dx=z.real()-cx, dy=z.imag()-cy, d=std::hypot(dx,dy);
            if (d < 1e-300) continue;
            std::array<double,3> j{{-dx/d,-dy/d,-1}}; double res=d-r;
            for(int a=0;a<3;++a){ rhs[a] += -j[a]*res; for(int bb=0;bb<3;++bb) N[a][bb]+=j[a]*j[bb]; }
        }
        std::array<double,3> delta{}; if(!solve3(N,rhs,delta)) break;
        cx+=delta[0]; cy+=delta[1]; r+=delta[2];
        if (r<=0) r=1e-300;
        if (std::hypot(delta[0],delta[1])+std::abs(delta[2]) <= 2e-14*std::max(1.0,r)) break;
    }
    double mean=0; for(auto z:p) mean += std::abs(z-Complex(cx,cy)); mean/=p.size(); r=mean;
    double sum2=0, mx=0; for(auto z:p){double e=std::abs(std::abs(z-Complex(cx,cy))-r); sum2+=e*e; mx=std::max(mx,e);} 
    Fit f; f.shape="circle"; f.shape_center={cx,cy}; f.size=r;
    f.rms=std::sqrt(sum2/p.size())/std::max(r,1e-300); f.max_error=mx/std::max(r,1e-300);
    f.circle_rms=f.rms; f.circle_max=f.max_error; return f;
}

static double signed_area(const std::vector<Complex>& p) {
    double a=0; for(std::size_t i=0;i<p.size();++i){auto x=p[i], y=p[(i+1)%p.size()]; a += x.real()*y.imag()-x.imag()*y.real();} return .5*a;
}

static Fit fit_cardioid(RawComponent c, int cusp_candidates, const Fit& circle) {
    auto& p=c.points; if(signed_area(p)<0) std::reverse(p.begin(),p.end());
    const std::size_t n=p.size();
    std::vector<std::pair<double,std::size_t>> candidates; candidates.reserve(n);
    for(std::size_t i=0;i<n;++i) candidates.push_back({std::abs(p[i]-c.center),i});
    std::partial_sort(candidates.begin(), candidates.begin()+std::min<std::size_t>(cusp_candidates,n), candidates.end());
    bool have=false; Fit best; double best_score=0;
    const std::size_t kmax=std::min<std::size_t>(cusp_candidates,n);
    for(std::size_t kk=0;kk<kmax;++kk) for(int direction: {1,-1}) {
        std::vector<Complex> basis(n); Complex bm=0, pm=0;
        const std::size_t cusp=candidates[kk].second;
        for(std::size_t i=0;i<n;++i){
            std::size_t rel=(i+n-cusp)%n; double phase=direction*2.0*M_PI*double(rel)/double(n);
            Complex u=std::polar(1.0,phase); basis[i]=2.0*u-u*u; bm+=basis[i]; pm+=p[i];
        }
        bm/=double(n); pm/=double(n); Complex num=0; double den=0;
        for(std::size_t i=0;i<n;++i){Complex b=basis[i]-bm, q=p[i]-pm; num += std::conj(b)*q; den += std::norm(b);} 
        if (den <= 1e-300) continue;
        Complex alpha = num / den;
        Complex beta = pm - alpha * bm;
        double size = std::abs(alpha);
        if (!(size > 0)) continue;
        double sum2=0,mx=0; for(std::size_t i=0;i<n;++i){double e=std::abs(p[i]-(beta+alpha*basis[i]));sum2+=e*e;mx=std::max(mx,e);} 
        double rms=std::sqrt(sum2/n)/size, me=mx/size, score=rms+.05*me;
        if(!have||score<best_score){have=true;best_score=score;best.shape="cardioid";best.shape_center=beta;best.size=size;best.angle=std::arg(alpha);best.rms=rms;best.max_error=me;best.circle_rms=circle.rms;best.circle_max=circle.max_error;}
    }
    if (!have) return circle;
    return best;
}

static Fit fit_component(const RawComponent& c, const Options& o) {
    if (c.shape_class == "disk"
        && c.fitted_center_centered
        && c.fitted_radius
        && *c.fitted_radius > 0) {
        Fit stored;
        stored.shape = "circle";
        stored.shape_center = c.center + *c.fitted_center_centered;
        stored.size = *c.fitted_radius;
        stored.rms = c.fitted_rms;
        stored.max_error = c.fitted_max_error;
        stored.circle_rms = stored.rms;
        stored.circle_max = stored.max_error;
        return stored;
    }
    if (c.shape_class == "cardioid"
        && c.fitted_center_centered
        && c.fitted_cardioid_size
        && *c.fitted_cardioid_size > 0) {
        Fit stored;
        stored.shape = "cardioid";
        stored.shape_center = c.center + *c.fitted_center_centered;
        stored.size = *c.fitted_cardioid_size;
        stored.angle = c.fitted_angle;
        stored.xi = c.fitted_xi;
        stored.rms = c.fitted_rms;
        stored.max_error = c.fitted_max_error;
        return stored;
    }
    Fit circle = fit_circle(c);
    Fit result;
    if (circle.rms <= o.circle_rms && circle.max_error <= o.circle_max) {
        result = circle;
    } else {
        Fit cardioid = fit_cardioid(c, o.cusp_candidates, circle);
        result = circle.rms < cardioid.rms ? circle : cardioid;
    }

    // The area catalogue is substantially more trustworthy than a least-
    // squares scale fitted to a malformed or self-crossing polygon.  Normalize
    // the rendered analytic shape to the measured component area:
    //   circle   A = pi r^2
    //   cardioid A = 6 pi s^2 for z=s(2e^{it}-e^{2it}).
    if (std::isfinite(c.area) && c.area > 0.0) {
        const double denominator = result.shape == "cardioid"
            ? 6.0 * M_PI
            : M_PI;
        const double area_size = std::sqrt(c.area / denominator);
        if (std::isfinite(area_size) && area_size > 0.0) {
            result.size = area_size;
        }
    }

    // A translated fit is useful, but a geometric origin many component sizes
    // away is a failed fit and creates enormous false hover targets.
    if (!(result.size > 0.0)
        || std::abs(result.shape_center - c.center) > 2.0 * result.size) {
        result.shape_center = c.center;
    }
    return result;
}

static std::string qnum(double v, int digits) {
    if (v == 0) return "0";
    std::ostringstream s; s << std::setprecision(digits) << v; return s.str();
}

static std::string human(std::uint64_t v) {
    const char* suffix[] = {"","k","M","G","T"}; double x=v; int i=0;
    while(x>=1000 && i<4){x/=1000;++i;} std::ostringstream s;
    s<<std::fixed<<std::setprecision(i==0?0:(x<10?2:(x<100?1:0)))<<x<<suffix[i]; return s.str();
}

class Progress {
public:
    explicit Progress(std::size_t total): total_(std::max<std::size_t>(1,total)) {
#if defined(__unix__) || defined(__APPLE__)
        terminal_=isatty(STDOUT_FILENO)!=0;
#endif
    }
    void show(std::size_t done, const std::string& stage) {
        int pct=int(done*100/total_); if(!terminal_ && pct<next_ && done<total_) return; if(!terminal_) next_=((pct/10)+1)*10;
        std::ostringstream s; s<<"  "<<stage<<" ["; int f=int(std::lround(28.0*done/total_)); s<<std::string(f,'#')<<std::string(28-f,'-')<<"] "<<std::setw(3)<<pct<<"% "<<human(done)<<'/'<<human(total_);
        std::lock_guard<std::mutex> lock(mutex_);
        if(terminal_){std::cout<<'\r'<<s.str()<<"\033[K"<<std::flush;}else std::cout<<s.str()<<'\n';
        if(done==total_&&terminal_)std::cout<<'\n';
    }
private: std::size_t total_; bool terminal_=false; int next_=0; std::mutex mutex_;
};

static Options parse(int argc,char**argv){
    Options o;
    for(int i=1;i<argc;++i){std::string a=argv[i]; auto need=[&](){if(++i>=argc)throw std::runtime_error("Missing value after "+a);return std::string(argv[i]);};
        if(a=="--config")o.config=need(); else if(a=="--output")o.output=need(); else if(a=="--min-area")o.min_area=std::stod(need()); else if(a=="--max-period")o.max_period=std::stoi(need()); else if(a=="--digits")o.digits=std::stoi(need());
        else if(a=="--circle-rms")o.circle_rms=std::stod(need()); else if(a=="--circle-max")o.circle_max=std::stod(need()); else if(a=="--cusp-candidates")o.cusp_candidates=std::stoi(need());
        else if(a=="--duplicate-tolerance")o.duplicate_tolerance=std::stod(need()); else if(a=="--threads")o.threads=static_cast<unsigned>(std::stoul(need()));
        else if(a=="-h"||a=="--help"){std::cout<<"Usage: fit_for_demo [--config PATH] [--output FILE]\\n";std::exit(0);} else throw std::runtime_error("Unknown option: "+a);
    }
    const auto repo=mandelbrot::repo::RepoConfig::load(o.config,fs::path(argv[0]?argv[0]:"."));
    o.catalogue_root=repo.path("paths.catalogue_root");
    if(o.output.empty()) o.output=repo.path("paths.atlas_demo_output")/".atlas_analytic_prefit.json";
    o.min_area=static_cast<double>(repo.number("demo.atlas.components.min_area",o.min_area));
    o.max_period=repo.integer("demo.atlas.components.max_period",o.max_period);
    o.digits=repo.integer("demo.atlas.components.coordinate_digits",o.digits);
    o.circle_rms=static_cast<double>(repo.number("demo.atlas.components.circle_fit_rms_tolerance",o.circle_rms));
    o.circle_max=static_cast<double>(repo.number("demo.atlas.components.circle_fit_max_tolerance",o.circle_max));
    o.cusp_candidates=repo.integer("demo.atlas.components.cardioid_cusp_candidates",o.cusp_candidates);
    o.duplicate_tolerance=static_cast<double>(repo.number("component_boundary_hunter.center_duplicate_tolerance",o.duplicate_tolerance));
    if(o.threads==0)o.threads=repo.threads();
    return o;
}

static std::vector<RawComponent> load_catalogue_components(const Options& options) {
    using namespace mandelbrot::catalogue;
    Catalogue catalogue(options.catalogue_root);
    catalogue.ensure_layout();
    ComponentQuery query;
    query.min_area = CatalogueReal(options.min_area);
    query.max_period = options.max_period;
    query.require_polygon = true;
    query.require_polygon_converged = true;
    const CatalogueSnapshot snapshot = catalogue.load_snapshot(query);

    std::vector<RawComponent> result;
    result.reserve(snapshot.components.size());
    std::unordered_set<ComponentKey, ComponentKeyHash> seen;
    for (const auto& component : snapshot.components) {
        const ComponentKey key = ComponentKey::from_center(component.period, component.center);
        if (!seen.insert(key).second) continue;
        RawComponent raw;
        raw.id = component.id;
        raw.period = component.period;
        raw.center = {component.center.re.convert_to<double>(), component.center.im.convert_to<double>()};
        raw.area = component.geometry.area_estimate.convert_to<double>();
        raw.source = component.provenance.method.empty() ? "catalogue" : component.provenance.method;
        raw.shape_class = component.classification.shape_class;
        if (const auto& fit = component.classification.circle_fit;
            fit && fit->center_centered && fit->radius) {
            raw.fitted_center_centered = Complex{
                fit->center_centered->re.convert_to<double>(),
                fit->center_centered->im.convert_to<double>()};
            raw.fitted_radius = fit->radius->convert_to<double>();
            raw.fitted_rms = fit->rms.convert_to<double>();
            raw.fitted_max_error = fit->max_error
                ? fit->max_error->convert_to<double>() : 0;
        }
        if (const auto& fit = component.classification.cardioid_fit;
            fit && fit->center_centered && fit->size) {
            raw.fitted_center_centered = Complex{
                fit->center_centered->re.convert_to<double>(),
                fit->center_centered->im.convert_to<double>()};
            raw.fitted_cardioid_size = fit->size->convert_to<double>();
            raw.fitted_angle = fit->angle.convert_to<double>();
            raw.fitted_xi = fit->xi.convert_to<double>();
            raw.fitted_rms = fit->rms.convert_to<double>();
            raw.fitted_max_error = fit->max_error
                ? fit->max_error->convert_to<double>() : 0;
        }
        raw.points.reserve(component.geometry.polygon.size());
        for (const auto& point : component.geometry.polygon) {
            raw.points.emplace_back(raw.center.real() + point.re.convert_to<double>(),
                                    raw.center.imag() + point.im.convert_to<double>());
        }
        result.push_back(std::move(raw));
    }
    return result;
}

int main(int argc,char**argv){try{
    Options o=parse(argc,argv);
    std::vector<RawComponent> unique=load_catalogue_components(o);
    std::cout<<"fit_for_demo: loaded "<<unique.size()<<" canonical polygon(s)\n";
    std::vector<Fit> fits(unique.size()); Progress progress(unique.size());
    std::atomic<std::size_t> next{0}, completed{0};
    std::vector<std::thread> workers;
    const unsigned worker_count = std::min<unsigned>(o.threads, std::max<std::size_t>(1, unique.size()));
    workers.reserve(worker_count);
    for (unsigned worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                const std::size_t i = next.fetch_add(1);
                if (i >= unique.size()) break;
                fits[i] = fit_component(unique[i], o);
                const std::size_t done = completed.fetch_add(1) + 1;
                if (done % 50 == 0 || done == unique.size()) progress.show(done, "fitting");
            }
        });
    }
    for (auto& worker : workers) worker.join();
    fs::create_directories(o.output.parent_path()); fs::path tmp=o.output.string()+".tmp"; std::ofstream out(tmp); if(!out)throw std::runtime_error("Could not write "+tmp.string());
    out << "{\"format\":\"mandelbrot-analytic-components-v1\",\"components\":[";
    for (std::size_t i = 0; i < unique.size(); ++i) {
        if (i) out << ',';
        auto& c = unique[i];
        auto& f = fits[i];
        out << "{\"id\":\"" << c.id
            << "\",\"period\":" << c.period
            << ",\"index\":" << c.index
            << ",\"shape\":\"" << f.shape
            << "\",\"center\":[" << qnum(c.center.real(), o.digits)
            << ',' << qnum(c.center.imag(), o.digits)
            << "],\"size\":" << qnum(f.size, o.digits)
            << ",\"area\":" << qnum(c.area, o.digits);
        if (std::abs(f.shape_center - c.center) > 2 * std::pow(10.0, -o.digits)) {
            out << ",\"shapeCenter\":["
                << qnum(f.shape_center.real(), o.digits) << ','
                << qnum(f.shape_center.imag(), o.digits) << ']';
        }
        if (f.shape == "cardioid") {
            out << ",\"angle\":" << qnum(f.angle, o.digits)
                << ",\"xi\":" << qnum(f.xi, o.digits);
        }
        if (c.discovery_index >= 0) {
            out << ",\"discoveryIndex\":" << c.discovery_index;
        }
        if (!c.period_count.empty()) {
            out << ",\"periodCount\":" << c.period_count;
        }
        out << ",\"source\":\"" << c.source
            << "\",\"fitRms\":" << qnum(f.rms, 6)
            << ",\"fitMax\":" << qnum(f.max_error, 6) << '}';
    }
    out<<"]}\n";out.close();fs::rename(tmp,o.output);std::cout<<"fit_for_demo: wrote "<<unique.size()<<" compact component(s) to "<<o.output<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"fit_for_demo error: "<<e.what()<<'\n';return 1;}}
