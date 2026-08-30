#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "common/repo_config.hpp"
#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// Native path uses long double. Boost.Multiprecision tiers are selected at runtime when needed.
using real = long double;

namespace fs = std::filesystem;
using Complex = std::complex<real>;

static constexpr real PI = static_cast<real>(3.141592653589793238462643383279502884L);

struct Point {
    real x{};
    real y{};
};

struct Jet2 {
    real G{};
    real Gx{};
    real Gy{};
    real Gxx{};
    real Gxy{};
    real Gyy{};
};

struct Config {
    // Output / scheduling
    std::string output_dir = "$data_root/G_contours";
    std::string code_root = "auto";
    std::string project_root = "auto";
    int threads = 0;                 // 0 = automatic thread count; >0 = exact worker count
    std::string auto_threads_policy = "leave_free"; // leave_free | fraction | all
    int leave_free = 2;
    real auto_threads_fraction = static_cast<real>(2.0L / 3.0L);
    bool resume = true;
    bool overwrite = false;
    bool low_g_first = true;          // better load balancing: expensive contours start first
    bool write_meta = true;
    bool progress = true;
    std::uint64_t progress_every = 5000;

    // Contour levels
    real g_start = static_cast<real>(0.25L);
    real g_stop = static_cast<real>(1.0e-6L);
    int n = 100;

    // Mandelbrot / derivative settings
    int power = 2;                    // this C++ tracer currently supports the quadratic case
    real bailout = static_cast<real>(2.0L);
    real derivative_epsilon = static_cast<real>(1.0e-12L);
    int derivative_max_iter = 5000;
    real root_epsilon = static_cast<real>(1.0e-13L);
    bool mp_fallback = true;
    real mp_trigger_g = static_cast<real>(1.0e-8L);
    real mp_compare_tolerance = static_cast<real>(1.0e-14L);
    int mp_max_digits = 200;

    // Base step choice: ds = ds_at_start * (G/g_start)^ds_power
    real ds_at_start = static_cast<real>(0.02L);
    real ds_power = static_cast<real>(0.45L);
    real ds_min = static_cast<real>(1.0e-11L);
    real ds_max = static_cast<real>(0.0L); // <=0 means disabled

    // Tracer controls
    real max_turn_angle = static_cast<real>(0.02L);
    real max_actual_turn_angle = static_cast<real>(0.08L);
    real step_error_factor = static_cast<real>(0.12L);
    std::uint64_t max_steps = 3000000ULL;
    int max_step_halvings = 40;
    bool project_each_step = true;
    int project_max_iter = 8;
    int project_max_backtracks = 16;
    real tol_G = static_cast<real>(1.0e-12L);

    // Adaptive ds controls
    bool adaptive_ds = true;
    real ds_growth = static_cast<real>(1.08L);
    real ds_shrink = static_cast<real>(0.5L);
    int grow_after_successes = 120;

    // Rollback controls if even step-halving fails
    std::uint64_t rollback_points = 200;
    real rollback_shrink = static_cast<real>(0.5L);
    std::uint64_t max_rollbacks = 2000;

    // Geometry guards
    std::uint64_t self_intersection_window = 512;
    real seam_tol_factor = static_cast<real>(8.0L);
    real seam_tol_min = static_cast<real>(1.0e-8L);

    // Output format / storage controls
    // output_format: json | bin
    // output_value_type: compute | float32 | float64 | longdouble
    // output_symmetry: full | half.  half stores only the upper half-contour
    // and marks metadata as upper_half_mirror_y.
    std::string output_format = "json";
    std::string output_value_type = "compute";
    std::string output_symmetry = "full";
    bool stream_output = false;       // implemented for output_format=bin + output_symmetry=half
    std::uint64_t stream_chunk_mb = 8;
    std::uint64_t stream_commit_margin_points = 0; // 0=auto
    bool write_partial_on_failure = true;
    bool harmonize_existing_format_on_resume = true;
    bool delete_other_format_on_harmonize = true;
    bool save_jet2 = false;           // reserved; usually false for C++ heavy-lifting run
    int json_precision = std::numeric_limits<real>::max_digits10;

    // Quality diagnostics written to meta JSON.
    // 1 means evaluate every half-contour point. Higher values sample every Nth point.
    int quality_eval_stride = 1;

    // Failure / exit behavior
    bool write_failure_meta = true;
    bool return_nonzero_on_failure = false;

    // Stall detection.  This catches rollback loops where the tracer hits ds_min
    // and keeps revisiting the same local region without meaningful progress.
    bool stall_detection = true;
    std::uint64_t abort_if_ds_min_for_points = 5000000ULL;
    real stall_progress_epsilon = static_cast<real>(1.0e-5L);

    // Progress UI
    // progress_style: bars | lines | none
    // progress_screen: alternate | normal
    // alternate uses the terminal's alternate screen buffer so progress junk
    // does not enter scrollback and wrapped lines can be cleaned cleanly.
    std::string progress_style = "bars";
    std::string progress_screen = "alternate";
    int progress_bar_width = 32;
    int progress_refresh_ms = 500;
};

static std::mutex g_print_mutex;

struct ProgressSlot {
    bool active = false;
    int index = -1;
    real G = 0;
    real fraction = 0;
    size_t points = 0;
    size_t total_points = 0;          // optional denominator for scan/readback phases
    std::string count_label = "pts";  // usually "pts"; use "scan" for readback phases
    real ds = 0;
    std::uint64_t rollbacks = 0;
    std::string status;
};

static std::mutex g_progress_mutex;
static std::vector<ProgressSlot> g_progress_slots;
static std::atomic<bool> g_monitor_stop{false};
static std::atomic<bool> g_alt_screen_active{false};
static volatile std::sig_atomic_t g_alt_screen_active_signal = 0;

static inline std::string trim(std::string s) {
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static inline std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static size_t size_t_from_u64_clamped(std::uint64_t value) {
    constexpr std::uint64_t max_size_t = static_cast<std::uint64_t>(std::numeric_limits<size_t>::max());
    return value > max_size_t ? std::numeric_limits<size_t>::max() : static_cast<size_t>(value);
}

static Config load_repository_config(const fs::path& config_path, const char* argv0) {
    const auto repo = mandelbrot::repo::RepoConfig::load(config_path, fs::path(argv0 ? argv0 : "."));
    Config c;
    c.code_root = repo.code_root().string();
    c.project_root = repo.project_root().string();
    c.output_dir = repo.path("paths.contours_output").string();
    c.threads = static_cast<int>(repo.threads());
    const std::string p = "contours.";
    auto set_str=[&](const char* key,std::string& value){value=repo.string(p+key,value);};
    auto set_int=[&](const char* key,int& value){value=repo.integer(p+key,value);};
    auto set_u64=[&](const char* key,std::uint64_t& value){value=repo.u64(p+key,value);};
    auto set_real=[&](const char* key,real& value){value=static_cast<real>(repo.number(p+key,value));};
    auto set_bool=[&](const char* key,bool& value){value=repo.boolean(p+key,value);};

    set_str("auto_threads_policy",c.auto_threads_policy);
    set_int("leave_free",c.leave_free);
    set_real("auto_threads_fraction",c.auto_threads_fraction);
    set_bool("resume",c.resume); set_bool("overwrite",c.overwrite); set_bool("low_g_first",c.low_g_first);
    set_bool("write_meta",c.write_meta); set_bool("progress",c.progress); set_u64("progress_every",c.progress_every);
    c.progress = repo.boolean("runtime.progress.enabled", c.progress);
    c.progress_style = repo.string("runtime.progress.style", c.progress_style);
    c.progress_screen = repo.string("runtime.progress.screen", c.progress_screen);
    c.progress_bar_width = repo.integer("runtime.progress.bar_width", c.progress_bar_width);
    c.progress_refresh_ms = repo.integer("runtime.progress.refresh_ms", c.progress_refresh_ms);
    set_str("progress_style",c.progress_style); set_str("progress_screen",c.progress_screen);
    set_int("progress_bar_width",c.progress_bar_width); set_int("progress_refresh_ms",c.progress_refresh_ms);
    set_real("g_start",c.g_start); set_real("g_stop",c.g_stop); set_int("n",c.n);
    set_int("power",c.power); set_real("bailout",c.bailout); set_real("derivative_epsilon",c.derivative_epsilon);
    set_int("derivative_max_iter",c.derivative_max_iter); set_real("root_epsilon",c.root_epsilon);
    set_bool("mp_fallback",c.mp_fallback); set_real("mp_trigger_g",c.mp_trigger_g);
    set_real("mp_compare_tolerance",c.mp_compare_tolerance); set_int("mp_max_digits",c.mp_max_digits);
    set_real("ds_at_start",c.ds_at_start); set_real("ds_power",c.ds_power); set_real("ds_min",c.ds_min); set_real("ds_max",c.ds_max);
    set_real("max_turn_angle",c.max_turn_angle); set_real("max_actual_turn_angle",c.max_actual_turn_angle);
    set_real("step_error_factor",c.step_error_factor); set_u64("max_steps",c.max_steps);
    set_int("max_step_halvings",c.max_step_halvings); set_bool("project_each_step",c.project_each_step);
    set_int("project_max_iter",c.project_max_iter); set_int("project_max_backtracks",c.project_max_backtracks);
    set_real("tol_G",c.tol_G); set_bool("adaptive_ds",c.adaptive_ds); set_real("ds_growth",c.ds_growth);
    set_real("ds_shrink",c.ds_shrink); set_int("grow_after_successes",c.grow_after_successes);
    set_u64("rollback_points",c.rollback_points); set_real("rollback_shrink",c.rollback_shrink); set_u64("max_rollbacks",c.max_rollbacks);
    set_u64("self_intersection_window",c.self_intersection_window); set_real("seam_tol_factor",c.seam_tol_factor); set_real("seam_tol_min",c.seam_tol_min);
    set_str("output_format",c.output_format); set_str("output_value_type",c.output_value_type); set_str("output_symmetry",c.output_symmetry);
    set_bool("stream_output",c.stream_output); set_u64("stream_chunk_mb",c.stream_chunk_mb); set_u64("stream_commit_margin_points",c.stream_commit_margin_points);
    set_bool("write_partial_on_failure",c.write_partial_on_failure); set_bool("harmonize_existing_format_on_resume",c.harmonize_existing_format_on_resume);
    set_bool("delete_other_format_on_harmonize",c.delete_other_format_on_harmonize); set_bool("save_jet2",c.save_jet2);
    set_int("json_precision",c.json_precision); set_int("quality_eval_stride",c.quality_eval_stride);
    set_bool("write_failure_meta",c.write_failure_meta); set_bool("return_nonzero_on_failure",c.return_nonzero_on_failure);
    set_bool("stall_detection",c.stall_detection); set_u64("abort_if_ds_min_for_points",c.abort_if_ds_min_for_points);
    set_real("stall_progress_epsilon",c.stall_progress_epsilon);

    if (c.power != 2) throw std::runtime_error("This C++ tracer currently supports power=2 only.");
    if (c.n <= 0 || c.g_start <= 0 || c.g_stop <= 0 || c.ds_at_start <= 0 || c.ds_min <= 0) {
        throw std::runtime_error("Invalid contour numerical configuration.");
    }
    c.output_format=lower(trim(c.output_format)); c.output_value_type=lower(trim(c.output_value_type));
    c.output_symmetry=lower(trim(c.output_symmetry)); c.progress_style=lower(trim(c.progress_style));
    c.progress_screen=lower(trim(c.progress_screen));
    return c;
}

static inline bool is_finite(real x) { return std::isfinite(static_cast<long double>(x)); }
static inline bool is_finite(const Jet2 &j) {
    return is_finite(j.G) && is_finite(j.Gx) && is_finite(j.Gy) && is_finite(j.Gxx) && is_finite(j.Gxy) && is_finite(j.Gyy);
}

static inline real hypot2(real x, real y) { return std::sqrt(x*x + y*y); }
static inline real scale2(int n) { return std::ldexp(static_cast<real>(1), -n); }

static bool in_main_cardioid_or_period2_bulb(real x, real y) {
    real q = (x - static_cast<real>(0.25L)) * (x - static_cast<real>(0.25L)) + y * y;
    if (q * (q + x - static_cast<real>(0.25L)) <= static_cast<real>(0.25L) * y * y) return true;
    return (x + static_cast<real>(1)) * (x + static_cast<real>(1)) + y * y <= static_cast<real>(0.0625L);
}

static Jet2 jet_from_state(int n, real log_abs_z, Complex A, Complex B) {
    real scale = scale2(n);
    Jet2 j;
    j.G   = scale * log_abs_z;
    j.Gx  = scale * A.real();
    j.Gy  = -scale * A.imag();
    j.Gxx = scale * B.real();
    j.Gxy = -scale * B.imag();
    j.Gyy = -scale * B.real();
    return j;
}

static void stable_post_escape_step(
    const Complex &c,
    real &log_abs_z,
    Complex &A,
    Complex &B,
    Complex &u
) {
    Complex den = static_cast<real>(1) + c * u;
    Complex numerator = static_cast<real>(2) * A + u;
    Complex u_prime = static_cast<real>(-2) * A * u;
    Complex numerator_prime = static_cast<real>(2) * B + u_prime;
    Complex denominator_prime = u + c * u_prime;
    Complex A_next = numerator / den;
    Complex B_next = (numerator_prime * den - numerator * denominator_prime) / (den * den);
    Complex u_next = (u * u) / (den * den);
    real L_next = static_cast<real>(2) * log_abs_z + std::log(std::abs(den));
    A = A_next;
    B = B_next;
    u = u_next;
    log_abs_z = L_next;
}

static Jet2 mandelbrot_jet_native(real x, real y, const Config &cfg) {
    Complex c(x, y);
    Complex z(0, 0), dz(0, 0), d2z(0, 0);
    std::optional<Jet2> prev;

    for (int n = 1; n <= cfg.derivative_max_iter; ++n) {
        // z -> z^2 + c and c-derivatives
        Complex z_next = z*z + c;
        Complex dz_next = static_cast<real>(2)*z*dz + static_cast<real>(1);
        Complex d2z_next = static_cast<real>(2)*dz*dz + static_cast<real>(2)*z*d2z;
        z = z_next; dz = dz_next; d2z = d2z_next;

        real absz = std::abs(z);
        if (absz <= cfg.bailout) continue;

        real L = std::log(absz);
        Complex A = dz / z;
        Complex B = d2z / z - A * A;
        Complex u = static_cast<real>(1) / (z*z);
        Jet2 current = jet_from_state(n, L, A, B);
        if (!is_finite(current)) throw std::runtime_error("non-finite initial Jet2");
        prev = current;

        for (int m = n + 1; m <= cfg.derivative_max_iter; ++m) {
            stable_post_escape_step(c, L, A, B, u);
            current = jet_from_state(m, L, A, B);
            if (!is_finite(current)) throw std::runtime_error("non-finite stable Jet2");
            const Jet2 &p = *prev;
            real err = std::max({
                std::abs(current.G - p.G), std::abs(current.Gx - p.Gx), std::abs(current.Gy - p.Gy),
                std::abs(current.Gxx - p.Gxx), std::abs(current.Gxy - p.Gxy), std::abs(current.Gyy - p.Gyy)
            });
            real scale = std::max({
                std::abs(current.G), std::abs(current.Gx), std::abs(current.Gy),
                std::abs(current.Gxx), std::abs(current.Gxy), std::abs(current.Gyy), static_cast<real>(1)
            });
            if (err <= cfg.derivative_epsilon * scale) return current;
            prev = current;
        }
        return *prev;
    }
    throw std::runtime_error("point did not escape within derivative_max_iter");
}

template <class MP>
struct MpComplex {
    MP re{}, im{};
    MpComplex() = default;
    MpComplex(const MP& real_part, const MP& imag_part = MP(0)) : re(real_part), im(imag_part) {}
};
template<class MP> MpComplex<MP> operator+(const MpComplex<MP>&a,const MpComplex<MP>&b){return {a.re+b.re,a.im+b.im};}
template<class MP> MpComplex<MP> operator-(const MpComplex<MP>&a,const MpComplex<MP>&b){return {a.re-b.re,a.im-b.im};}
template<class MP> MpComplex<MP> operator-(const MpComplex<MP>&a){return {-a.re,-a.im};}
template<class MP> MpComplex<MP> operator*(const MpComplex<MP>&a,const MpComplex<MP>&b){return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};}
template<class MP> MpComplex<MP> operator/(const MpComplex<MP>&a,const MpComplex<MP>&b){const MP d=b.re*b.re+b.im*b.im;return {(a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d};}
template<class MP> MpComplex<MP> operator+(const MpComplex<MP>&a,const MP&b){return {a.re+b,a.im};}
template<class MP> MpComplex<MP> operator+(const MP&a,const MpComplex<MP>&b){return b+a;}
template<class MP> MpComplex<MP> operator*(const MP&a,const MpComplex<MP>&b){return {a*b.re,a*b.im};}
template<class MP> MpComplex<MP> operator*(const MpComplex<MP>&a,const MP&b){return b*a;}
template<class MP> MP mp_abs(const MpComplex<MP>&a){using boost::multiprecision::sqrt;return sqrt(a.re*a.re+a.im*a.im);}

template <unsigned Digits>
static Jet2 mandelbrot_jet_mp(real x, real y, const Config& cfg) {
    using MP = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits>>;
    using MPC = MpComplex<MP>;
    struct MPJet { MP G{}, Gx{}, Gy{}, Gxx{}, Gxy{}, Gyy{}; };
    auto convert = [](const MPJet& source) {
        return Jet2{source.G.template convert_to<real>(), source.Gx.template convert_to<real>(),
                    source.Gy.template convert_to<real>(), source.Gxx.template convert_to<real>(),
                    source.Gxy.template convert_to<real>(), source.Gyy.template convert_to<real>()};
    };
    const MPC c{MP(x), MP(y)};
    MPC z{}, dz{}, d2z{};
    std::optional<MPJet> previous;
    const MP bailout(cfg.bailout), epsilon(cfg.derivative_epsilon);
    using boost::multiprecision::abs;
    using boost::multiprecision::log;
    using boost::multiprecision::pow;
    for (int n = 1; n <= cfg.derivative_max_iter; ++n) {
        const MPC z_next = z*z+c;
        const MPC dz_next = MP(2)*z*dz+MP(1);
        const MPC d2z_next = MP(2)*dz*dz+MP(2)*z*d2z;
        z=z_next; dz=dz_next; d2z=d2z_next;
        const MP absz=mp_abs(z);
        if(absz<=bailout) continue;
        MP L=log(absz);
        MPC A=dz/z;
        MPC B=d2z/z-A*A;
        MPC u=MPC(MP(1))/(z*z);
        auto make_jet=[](int iteration,const MP& ll,const MPC& aa,const MPC& bb){
            const MP scale=pow(MP(2),-iteration);
            return MPJet{scale*ll,scale*aa.re,-scale*aa.im,scale*bb.re,-scale*bb.im,-scale*bb.re};
        };
        MPJet current=make_jet(n,L,A,B); previous=current;
        for(int m=n+1;m<=cfg.derivative_max_iter;++m){
            const MPC den=MPC(MP(1))+c*u;
            const MPC numerator=MP(2)*A+u;
            const MPC u_prime=-MP(2)*A*u;
            const MPC numerator_prime=MP(2)*B+u_prime;
            const MPC denominator_prime=u+c*u_prime;
            A=numerator/den;
            B=(numerator_prime*den-numerator*denominator_prime)/(den*den);
            u=(u*u)/(den*den);
            L=MP(2)*L+log(mp_abs(den));
            current=make_jet(m,L,A,B);
            const MPJet& old=*previous;
            MP err = MP(abs(current.G-old.G));
            err = std::max(err, MP(abs(current.Gx-old.Gx)));
            err = std::max(err, MP(abs(current.Gy-old.Gy)));
            err = std::max(err, MP(abs(current.Gxx-old.Gxx)));
            err = std::max(err, MP(abs(current.Gxy-old.Gxy)));
            err = std::max(err, MP(abs(current.Gyy-old.Gyy)));
            MP scale = MP(1);
            scale = std::max(scale, MP(abs(current.G)));
            scale = std::max(scale, MP(abs(current.Gx)));
            scale = std::max(scale, MP(abs(current.Gy)));
            scale = std::max(scale, MP(abs(current.Gxx)));
            scale = std::max(scale, MP(abs(current.Gxy)));
            scale = std::max(scale, MP(abs(current.Gyy)));
            if(err<=epsilon*scale) return convert(current);
            previous=current;
        }
        return convert(*previous);
    }
    throw std::runtime_error("point did not escape within derivative_max_iter (MP)");
}

static real jet_relative_difference(const Jet2& a, const Jet2& b) {
    const real numerator = std::max({std::abs(a.G-b.G),std::abs(a.Gx-b.Gx),std::abs(a.Gy-b.Gy),
                                     std::abs(a.Gxx-b.Gxx),std::abs(a.Gxy-b.Gxy),std::abs(a.Gyy-b.Gyy)});
    const real denominator = std::max({real(1),std::abs(b.G),std::abs(b.Gx),std::abs(b.Gy),
                                       std::abs(b.Gxx),std::abs(b.Gxy),std::abs(b.Gyy)});
    return numerator / denominator;
}

static Jet2 mandelbrot_jet(real x, real y, const Config& cfg) {
    std::optional<Jet2> native;
    try { native = mandelbrot_jet_native(x, y, cfg); }
    catch (...) {
        if (!cfg.mp_fallback) throw;
    }
    if (native && (!cfg.mp_fallback || native->G > cfg.mp_trigger_g)) return *native;
    auto evaluate = [&](int digits) -> Jet2 {
        if (digits <= 50) return mandelbrot_jet_mp<50>(x,y,cfg);
        if (digits <= 80) return mandelbrot_jet_mp<80>(x,y,cfg);
        if (digits <= 120) return mandelbrot_jet_mp<120>(x,y,cfg);
        if (digits <= 160) return mandelbrot_jet_mp<160>(x,y,cfg);
        return mandelbrot_jet_mp<200>(x,y,cfg);
    };
    Jet2 best = evaluate(50);
    if (native && jet_relative_difference(*native,best) <= cfg.mp_compare_tolerance) return *native;
    for (const int digits : {80,120,160,200}) {
        if (digits > cfg.mp_max_digits) break;
        Jet2 next = evaluate(digits);
        if (jet_relative_difference(best,next) <= cfg.mp_compare_tolerance) return next;
        best = next;
    }
    return best;
}

static real potential_safe(real x, real y, const Config &cfg) {
    if (in_main_cardioid_or_period2_bulb(x, y)) return 0;
    try { return mandelbrot_jet(x, y, cfg).G; }
    catch (...) { return 0; }
}

static real bisect_root_x(real a, real b, real target, const Config &cfg) {
    real fa = potential_safe(a, 0, cfg) - target;
    real fb = potential_safe(b, 0, cfg) - target;
    if (fa == 0) return a;
    if (fb == 0) return b;
    if (fa * fb > 0) {
        std::ostringstream oss;
        oss << "root not bracketed: a=" << static_cast<double>(a) << " fa=" << static_cast<double>(fa)
            << " b=" << static_cast<double>(b) << " fb=" << static_cast<double>(fb);
        throw std::runtime_error(oss.str());
    }
    for (int i = 0; i < 200; ++i) {
        real m = (a + b) / 2;
        real fm = potential_safe(m, 0, cfg) - target;
        if (std::abs(b - a) <= cfg.root_epsilon * std::max(static_cast<real>(1), std::abs(m))) return m;
        if (fa * fm <= 0) { b = m; fb = fm; }
        else { a = m; fa = fm; }
    }
    return (a + b) / 2;
}

static std::pair<Point, Point> real_axis_intersections(real target, const Config &cfg) {
    // left: (-inf, -2)
    real left_inside = static_cast<real>(-2);
    real left_far = static_cast<real>(-4);
    while (potential_safe(left_far, 0, cfg) - target < 0) left_far *= static_cast<real>(2);
    real xl = bisect_root_x(left_far, left_inside, target, cfg);

    // right: (1/4, +inf)
    real right_inside = static_cast<real>(0.25L);
    real right_far = static_cast<real>(2);
    while (potential_safe(right_far, 0, cfg) - target < 0) right_far *= static_cast<real>(2);
    real xr = bisect_root_x(right_inside, right_far, target, cfg);

    return {Point{xl, 0}, Point{xr, 0}};
}

static real curvature_from_jet(const Jet2 &j) {
    real grad2 = j.Gx*j.Gx + j.Gy*j.Gy;
    if (!(grad2 > 0) || !is_finite(grad2)) throw std::runtime_error("bad gradient in curvature_from_jet");
    return -(
        j.Gxx * j.Gy * j.Gy - static_cast<real>(2) * j.Gxy * j.Gx * j.Gy + j.Gyy * j.Gx * j.Gx
    ) / std::pow(grad2, static_cast<real>(1.5L));
}

static Point contour_tangent(const Jet2 &j, int direction) {
    real grad = hypot2(j.Gx, j.Gy);
    if (!(grad > 0) || !is_finite(grad)) throw std::runtime_error("bad gradient in tangent");
    return Point{direction * (-j.Gy / grad), direction * (j.Gx / grad)};
}

static Point contour_rhs(real x, real y, int direction, const Config &cfg) {
    Jet2 j = mandelbrot_jet(x, y, cfg);
    return contour_tangent(j, direction);
}

static Point project_to_level(real x, real y, real target, const Config &cfg) {
    Jet2 j = mandelbrot_jet(x, y, cfg);
    real F = j.G;
    for (int it = 0; it < cfg.project_max_iter; ++it) {
        real err = F - target;
        if (std::abs(err) <= cfg.tol_G) return Point{x, y};
        real grad2 = j.Gx*j.Gx + j.Gy*j.Gy;
        if (!(grad2 > 0) || !is_finite(grad2)) throw std::runtime_error("bad gradient in projection");
        real sx = err * j.Gx / grad2;
        real sy = err * j.Gy / grad2;
        real best_err = std::abs(err);
        std::optional<std::pair<Point, Jet2>> best;
        real alpha = 1;
        for (int bt = 0; bt <= cfg.project_max_backtracks; ++bt) {
            real xt = x - alpha * sx;
            real yt = y - alpha * sy;
            try {
                Jet2 jt = mandelbrot_jet(xt, yt, cfg);
                real e = std::abs(jt.G - target);
                if (e < best_err) { best_err = e; best = std::make_pair(Point{xt, yt}, jt); }
                if (e <= static_cast<real>(0.5L) * std::abs(err) || e <= cfg.tol_G) {
                    x = xt; y = yt; j = jt; F = jt.G;
                    goto accepted;
                }
            } catch (...) {}
            alpha *= static_cast<real>(0.5L);
        }
        if (!best) throw std::runtime_error("projection failed");
        x = best->first.x; y = best->first.y; j = best->second; F = j.G;
        if (best_err >= static_cast<real>(0.9L) * std::abs(err)) throw std::runtime_error("projection stagnated");
        accepted: ;
    }
    return Point{x, y};
}

static Point rk4_step_raw(real x, real y, real ds, int direction, const Config &cfg) {
    Point k1 = contour_rhs(x, y, direction, cfg);
    Point k2 = contour_rhs(x + ds*k1.x/static_cast<real>(2), y + ds*k1.y/static_cast<real>(2), direction, cfg);
    Point k3 = contour_rhs(x + ds*k2.x/static_cast<real>(2), y + ds*k2.y/static_cast<real>(2), direction, cfg);
    Point k4 = contour_rhs(x + ds*k3.x, y + ds*k3.y, direction, cfg);
    return Point{
        x + ds * (k1.x + static_cast<real>(2)*k2.x + static_cast<real>(2)*k3.x + k4.x) / static_cast<real>(6),
        y + ds * (k1.y + static_cast<real>(2)*k2.y + static_cast<real>(2)*k3.y + k4.y) / static_cast<real>(6)
    };
}

static Point rk4_step_projected(real x, real y, real target, real ds, int direction, const Config &cfg) {
    Point p = rk4_step_raw(x, y, ds, direction, cfg);
    if (cfg.project_each_step) p = project_to_level(p.x, p.y, target, cfg);
    return p;
}

static real signed_area(const std::vector<Point> &p) {
    if (p.size() < 3) return 0;
    real s = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Point &a = p[i];
        const Point &b = p[(i+1) % p.size()];
        s += a.x*b.y - a.y*b.x;
    }
    return s / 2;
}

static real polyline_length(const std::vector<Point> &p) {
    if (p.size() < 2) return 0;
    real s = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Point &a = p[i];
        const Point &b = p[(i+1) % p.size()];
        s += hypot2(b.x-a.x, b.y-a.y);
    }
    return s;
}

static real open_polyline_length(const std::vector<Point> &p) {
    if (p.size() < 2) return 0;
    real s = 0;
    for (size_t i = 0; i + 1 < p.size(); ++i) s += hypot2(p[i+1].x-p[i].x, p[i+1].y-p[i].y);
    return s;
}

static real upper_half_cross_sum(const std::vector<Point> &p) {
    real s = 0;
    for (size_t i = 0; i + 1 < p.size(); ++i) s += p[i].x*p[i+1].y - p[i].y*p[i+1].x;
    return s;
}

struct HalfFileStats {
    std::uint64_t half_points = 0;
    real full_length = 0;
    real full_area_signed = 0;
};

static real angle_between_segments(const Point &a, const Point &b, const Point &c) {
    real ux = b.x - a.x, uy = b.y - a.y;
    real vx = c.x - b.x, vy = c.y - b.y;
    real nu = hypot2(ux, uy), nv = hypot2(vx, vy);
    if (!(nu > 0) || !(nv > 0)) return 0;
    real cross = ux*vy - uy*vx;
    real dot = ux*vx + uy*vy;
    return std::abs(std::atan2(cross, dot));
}

static bool segments_intersect_strict(Point a0, Point a1, Point b0, Point b1, real tol) {
    if (std::max(std::min(a0.x,a1.x), std::min(b0.x,b1.x)) > std::min(std::max(a0.x,a1.x), std::max(b0.x,b1.x)) + tol) return false;
    if (std::max(std::min(a0.y,a1.y), std::min(b0.y,b1.y)) > std::min(std::max(a0.y,a1.y), std::max(b0.y,b1.y)) + tol) return false;
    auto orient = [](Point p, Point q, Point r){ return (q.x-p.x)*(r.y-p.y) - (q.y-p.y)*(r.x-p.x); };
    real o1 = orient(a0,a1,b0), o2 = orient(a0,a1,b1), o3 = orient(b0,b1,a0), o4 = orient(b0,b1,a1);
    return (o1*o2 < -tol) && (o3*o4 < -tol);
}

static bool candidate_intersects_recent(const std::vector<Point> &path, Point cand, std::uint64_t window) {
    if (window == 0 || path.size() < 5) return false;
    const size_t win = size_t_from_u64_clamped(window);
    Point a0 = path[path.size()-1];
    Point a1 = cand;
    size_t n = path.size();
    size_t begin = (n > win) ? n - win : 0;
    // exclude adjacent last two segments
    size_t end = (n >= 3) ? n - 3 : 0;
    for (size_t i = begin; i < end; ++i) {
        if (segments_intersect_strict(a0, a1, path[i], path[i+1], static_cast<real>(1e-18L))) return true;
    }
    return false;
}

static std::optional<real> x_cross_at_y0(Point a, Point b) {
    real dy = b.y - a.y;
    if (dy == 0) return std::nullopt;
    real t = -a.y / dy;
    if (t >= 0 && t <= 1) return a.x + t * (b.x - a.x);
    return std::nullopt;
}

static real choose_ds(real G, const Config &cfg) {
    real ds = cfg.ds_at_start * std::pow(G / cfg.g_start, cfg.ds_power);
    ds = std::max(cfg.ds_min, ds);
    if (cfg.ds_max > 0) ds = std::min(cfg.ds_max, ds);
    return ds;
}

static real progress_fraction(const std::vector<Point> &path, real x_left, real x_right, bool upper=true, real recent_fraction=static_cast<real>(0.1L), int recent_min=2000) {
    if (path.empty()) return 0;
    real cx = (x_left + x_right) / 2;
    size_t n = path.size();
    size_t count = n;
    if (recent_fraction > 0) {
        count = std::max<size_t>(static_cast<size_t>(recent_min), static_cast<size_t>(std::ceil(static_cast<long double>(recent_fraction * n))));
        count = std::min(count, n);
    }
    size_t start = n - count;
    real max_theta = 0;
    for (size_t i = start; i < n; ++i) {
        real yy = upper ? path[i].y : -path[i].y;
        real theta = std::atan2(yy, path[i].x - cx);
        if (theta < 0) theta = 0;
        if (theta > PI) theta = PI;
        max_theta = std::max(max_theta, theta);
    }
    return std::max(static_cast<real>(0), std::min(static_cast<real>(1), max_theta / PI));
}

static void update_progress_slot_total(int tid, int index, real G, real frac, size_t points, size_t total_points, real ds, std::uint64_t rollbacks, const std::string &status, const std::string &count_label);
static void update_progress_slot(int tid, int index, real G, real frac, size_t points, real ds, std::uint64_t rollbacks, const std::string &status);

struct TraceResult {
    std::vector<Point> points;
    real used_ds{};
    std::uint64_t rollbacks{};
    bool streamed = false;
    std::uint64_t half_points = 0;
    bool recovered = false;
};

static TraceResult trace_half(real target, real ds_goal, const Config &cfg, int index, int tid) {
    auto [left, right] = real_axis_intersections(target, cfg);
    real x_left = left.x;
    real x_right = right.x;
    std::vector<Point> path;
    path.reserve(100000);
    path.push_back(right);

    int direction = +1;
    real ds_nom = ds_goal;
    int successes = 0;
    std::uint64_t rollbacks = 0;
    std::uint64_t min_steps_before_crossing = 16;
    real best_progress_fraction = 0;
    bool ds_min_stall_active = false;
    size_t ds_min_stall_points = 0;
    real ds_min_stall_progress = 0;

    for (std::uint64_t step = 0; step < cfg.max_steps; ++step) {
        Point cur = path.back();
        bool accepted = false;
        real accepted_ds = ds_nom;
        int accepted_halvings = 0;
        std::string last_reason;

        for (int h = 0; h <= cfg.max_step_halvings; ++h) {
            real ds_try = ds_nom * std::pow(static_cast<real>(0.5L), h);
            if (ds_try < cfg.ds_min) { last_reason = "ds below ds_min"; break; }

            try {
                Jet2 j = mandelbrot_jet(cur.x, cur.y, cfg);
                real kappa = curvature_from_jet(j);
                if (is_finite(kappa) && std::abs(kappa) > 0) {
                    ds_try = std::min(ds_try, cfg.max_turn_angle / std::abs(kappa));
                    if (ds_try < cfg.ds_min) ds_try = cfg.ds_min;
                }

                Point full = rk4_step_projected(cur.x, cur.y, target, ds_try, direction, cfg);
                Point half1 = rk4_step_projected(cur.x, cur.y, target, ds_try/static_cast<real>(2), direction, cfg);
                Point half2 = rk4_step_projected(half1.x, half1.y, target, ds_try/static_cast<real>(2), direction, cfg);
                real err = hypot2(full.x-half2.x, full.y-half2.y);
                if (err > cfg.step_error_factor * std::max(ds_try, cfg.ds_min)) {
                    last_reason = "step-doubling error";
                    continue;
                }
                Point cand = half2;

                real seam_tol = std::max(cfg.seam_tol_factor * ds_try, cfg.seam_tol_min);
                bool crossed = (step >= min_steps_before_crossing && cur.y > 0 && cand.y <= 0);
                auto xcross = crossed ? x_cross_at_y0(cur, cand) : std::optional<real>{};
                bool near_left = hypot2(cand.x - x_left, cand.y) <= seam_tol || (xcross && std::abs(*xcross - x_left) <= seam_tol);

                if (crossed) {
                    if (near_left) {
                        path.push_back(left);
                        return TraceResult{path, ds_nom, rollbacks};
                    }
                    last_reason = "crossed y=0 away from left seam";
                    continue;
                }
                if (near_left && step >= min_steps_before_crossing) {
                    path.push_back(left);
                    return TraceResult{path, ds_nom, rollbacks};
                }

                if (path.size() >= 2 && cfg.max_actual_turn_angle > 0) {
                    real a = angle_between_segments(path[path.size()-2], path[path.size()-1], cand);
                    if (a > cfg.max_actual_turn_angle) { last_reason = "actual turn too large"; continue; }
                }

                if (candidate_intersects_recent(path, cand, cfg.self_intersection_window)) {
                    last_reason = "recent self-intersection";
                    continue;
                }

                path.push_back(cand);
                accepted = true;
                accepted_ds = ds_try;
                accepted_halvings = h;
                break;
            } catch (const std::exception &e) {
                last_reason = e.what();
                continue;
            }
        }

        if (accepted) {
            if (cfg.adaptive_ds) {
                if (accepted_halvings > 0) {
                    ds_nom = std::max(cfg.ds_min, accepted_ds);
                    successes = 0;
                } else {
                    ++successes;
                    if (successes >= cfg.grow_after_successes) {
                        ds_nom = std::min(ds_goal, ds_nom * cfg.ds_growth);
                        successes = 0;
                    }
                }
            }
            if (cfg.progress && cfg.progress_every > 0 && step % cfg.progress_every == 0 && step > 0) {
                real frac = progress_fraction(path, x_left, x_right);
                best_progress_fraction = std::max(best_progress_fraction, frac);
                if (cfg.progress_style == "bars") {
                    update_progress_slot(tid, index, target, best_progress_fraction, path.size(), ds_nom, rollbacks, "tracing");
                } else if (cfg.progress_style == "lines") {
                    std::lock_guard<std::mutex> lock(g_print_mutex);
                    std::cerr << "[" << index << "] G=" << static_cast<double>(target)
                              << " progress=" << static_cast<double>(100*frac) << "% points=" << path.size()
                              << " ds=" << static_cast<double>(ds_nom) << " rollbacks=" << rollbacks << "\n";
                }
            }
            continue;
        }

        // local halving failed: rollback a bit and continue with smaller nominal step
        ++rollbacks;
        if (rollbacks > cfg.max_rollbacks || path.size() <= 4) {
            throw std::runtime_error("trace failed near step " + std::to_string(step) + ": " + last_reason);
        }
        const size_t rollback_points = size_t_from_u64_clamped(cfg.rollback_points);
        size_t keep = path.size() > rollback_points
            ? path.size() - rollback_points
            : static_cast<size_t>(2);
        path.resize(std::max<size_t>(2, keep));
        ds_nom = std::max(cfg.ds_min, ds_nom * cfg.rollback_shrink);
        successes = 0;

        if (cfg.stall_detection && cfg.abort_if_ds_min_for_points > 0 && ds_nom <= cfg.ds_min * static_cast<real>(1.0000001L)) {
            if (!ds_min_stall_active) {
                ds_min_stall_active = true;
                ds_min_stall_points = path.size();
                ds_min_stall_progress = best_progress_fraction;
            } else {
                size_t delta_points = path.size() > ds_min_stall_points ? path.size() - ds_min_stall_points : 0;
                real delta_progress = best_progress_fraction - ds_min_stall_progress;
                if (delta_points >= size_t_from_u64_clamped(cfg.abort_if_ds_min_for_points) &&
                    delta_progress < cfg.stall_progress_epsilon) {
                    throw std::runtime_error("stall detected: ds_min rollback loop with no meaningful progress");
                }
            }
        } else {
            ds_min_stall_active = false;
        }
    }
    throw std::runtime_error("max_steps exceeded");
}

static std::vector<Point> mirror_full(const std::vector<Point> &half) {
    std::vector<Point> full;
    if (half.empty()) return full;
    full.reserve(half.size() * 2 - 2);
    for (const auto &p : half) full.push_back(p);
    // mirror excluding left and right endpoints to avoid duplicates
    for (size_t k = half.size() - 2; k > 0; --k) {
        full.push_back(Point{half[k].x, -half[k].y});
    }
    return full;
}

static std::string precision_name() {
#if defined(MB_USE_LONG_DOUBLE)
    return "longdouble";
#elif defined(MB_USE_FLOAT)
    return "float32";
#else
    return "float64";
#endif
}

static int compute_real_size() {
    return static_cast<int>(sizeof(real));
}

static std::string stem_for_index(int index) {
    return std::to_string(100000 + index).substr(1);
}

enum class OutputFormat { Json, Bin };
enum class OutputValueType { Float32 = 1, Float64 = 2, LongDouble = 3 };

static OutputFormat output_format_from_config(const Config &cfg) {
    return cfg.output_format == "bin" ? OutputFormat::Bin : OutputFormat::Json;
}

static std::string output_format_name(OutputFormat fmt) {
    return fmt == OutputFormat::Bin ? "bin" : "json";
}

static OutputValueType resolve_output_value_type(const Config &cfg) {
    std::string t = lower(trim(cfg.output_value_type));
    if (t == "compute") {
#if defined(MB_USE_LONG_DOUBLE)
        return OutputValueType::LongDouble;
#elif defined(MB_USE_FLOAT)
        return OutputValueType::Float32;
#else
        return OutputValueType::Float64;
#endif
    }
    if (t == "float" || t == "float32") return OutputValueType::Float32;
    if (t == "double" || t == "float64") return OutputValueType::Float64;
    if (t == "longdouble" || t == "long_double") return OutputValueType::LongDouble;
    throw std::runtime_error("bad output_value_type");
}

static std::string value_type_name(OutputValueType vt) {
    switch (vt) {
        case OutputValueType::Float32: return "float32";
        case OutputValueType::Float64: return "float64";
        case OutputValueType::LongDouble: return "longdouble";
    }
    return "unknown";
}

static uint32_t value_type_code(OutputValueType vt) {
    return static_cast<uint32_t>(vt);
}

static uint32_t value_type_size(OutputValueType vt) {
    switch (vt) {
        case OutputValueType::Float32: return 4;
        case OutputValueType::Float64: return 8;
        case OutputValueType::LongDouble: return static_cast<uint32_t>(sizeof(long double));
    }
    return 0;
}

static fs::path contour_path_for(const fs::path &out, int index, OutputFormat fmt) {
    std::string stem = stem_for_index(index);
    if (fmt == OutputFormat::Json) return out / "contours" / (stem + ".json");
    return out / "contours_bin" / (stem + ".bin");
}

template <typename T>
static void write_scalar_binary(std::ofstream &f, T value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("binary write failed");
}

template <typename T>
static T read_scalar_binary(std::ifstream &f) {
    T value{};
    f.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("binary read failed");
    return value;
}

static std::string path_state_for_error(const fs::path &path) {
    std::ostringstream oss;
    std::error_code ec;
    bool exists = fs::exists(path, ec);
    oss << "exists=" << (exists ? "true" : "false");
    if (ec) oss << " exists_error=\"" << ec.message() << "\"";
    ec.clear();
    if (exists) {
        auto size = fs::file_size(path, ec);
        if (!ec) oss << " size=" << size;
        else oss << " size_error=\"" << ec.message() << "\"";
    }
    return oss.str();
}

static std::ifstream open_binary_input_retry(
    const fs::path &path,
    int attempts = 40,
    int sleep_ms = 250
) {
    attempts = std::max(1, attempts);
    sleep_ms = std::max(0, sleep_ms);

    for (int attempt = 1; attempt <= attempts; ++attempt) {
        std::ifstream f(path, std::ios::binary);
        if (f) return f;

        if (attempt < attempts && sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    std::ostringstream oss;
    oss << "could not open " << path.string()
        << " after " << attempts << " attempts"
        << " (" << path_state_for_error(path) << ")";
    throw std::runtime_error(oss.str());
}

static void write_points_bin(const fs::path &path, const std::vector<Point> &points, OutputValueType vt) {
    fs::create_directories(path.parent_path());
    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("could not open " + tmp.string());

    const char magic[8] = {'M','B','C','T','B','I','N','1'};
    f.write(magic, 8);
    write_scalar_binary<uint32_t>(f, 1);                        // version
    write_scalar_binary<uint32_t>(f, value_type_code(vt));       // stored value type
    write_scalar_binary<uint32_t>(f, value_type_size(vt));       // bytes per scalar on writer system
    write_scalar_binary<uint32_t>(f, 0x01020304u);               // endian marker
    write_scalar_binary<uint64_t>(f, static_cast<uint64_t>(points.size()));

    if (vt == OutputValueType::Float32) {
        for (const auto &p : points) {
            write_scalar_binary<float>(f, static_cast<float>(p.x));
            write_scalar_binary<float>(f, static_cast<float>(p.y));
        }
    } else if (vt == OutputValueType::Float64) {
        for (const auto &p : points) {
            write_scalar_binary<double>(f, static_cast<double>(p.x));
            write_scalar_binary<double>(f, static_cast<double>(p.y));
        }
    } else {
        for (const auto &p : points) {
            write_scalar_binary<long double>(f, static_cast<long double>(p.x));
            write_scalar_binary<long double>(f, static_cast<long double>(p.y));
        }
    }
    f.close();
    fs::rename(tmp, path);
}

static std::vector<Point> read_points_bin(const fs::path &path, OutputValueType *stored_type_out=nullptr, uint32_t *stored_size_out=nullptr) {
    std::ifstream f = open_binary_input_retry(path);
    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("bad binary contour magic in " + path.string());
    uint32_t version = read_scalar_binary<uint32_t>(f);
    uint32_t code = read_scalar_binary<uint32_t>(f);
    uint32_t value_size = read_scalar_binary<uint32_t>(f);
    uint32_t endian = read_scalar_binary<uint32_t>(f);
    uint64_t count = read_scalar_binary<uint64_t>(f);
    if (version != 1) throw std::runtime_error("unsupported binary contour version");
    if (endian != 0x01020304u) throw std::runtime_error("binary contour appears to have different endian order");

    OutputValueType vt;
    if (code == 1) vt = OutputValueType::Float32;
    else if (code == 2) vt = OutputValueType::Float64;
    else if (code == 3) vt = OutputValueType::LongDouble;
    else throw std::runtime_error("unknown binary contour value type");
    if (stored_type_out) *stored_type_out = vt;
    if (stored_size_out) *stored_size_out = value_size;

    if (value_size != value_type_size(vt)) {
        std::ostringstream oss;
        oss << "binary contour scalar size mismatch. file says " << value_size
            << " bytes for " << value_type_name(vt) << ", this build expects " << value_type_size(vt);
        throw std::runtime_error(oss.str());
    }

    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        if (vt == OutputValueType::Float32) {
            float x = read_scalar_binary<float>(f);
            float y = read_scalar_binary<float>(f);
            pts.push_back(Point{static_cast<real>(x), static_cast<real>(y)});
        } else if (vt == OutputValueType::Float64) {
            double x = read_scalar_binary<double>(f);
            double y = read_scalar_binary<double>(f);
            pts.push_back(Point{static_cast<real>(x), static_cast<real>(y)});
        } else {
            long double x = read_scalar_binary<long double>(f);
            long double y = read_scalar_binary<long double>(f);
            pts.push_back(Point{static_cast<real>(x), static_cast<real>(y)});
        }
    }
    return pts;
}

static std::vector<Point> read_points_json(const fs::path &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open " + path.string());
    std::vector<Point> pts;
    char ch{};
    f >> ch;
    if (ch != '[') throw std::runtime_error("bad JSON contour start in " + path.string());
    while (true) {
        f >> std::ws;
        int peek = f.peek();
        if (peek == ']') { f.get(); break; }
        if (peek == ',') { f.get(); continue; }
        f >> ch;
        if (ch != '[') throw std::runtime_error("expected point '[' in " + path.string());
        long double x{}, y{};
        f >> x;
        f >> ch;
        if (ch != ',') throw std::runtime_error("expected comma in point JSON");
        f >> y;
        f >> ch;
        if (ch != ']') throw std::runtime_error("expected point ']' in JSON");
        pts.push_back(Point{static_cast<real>(x), static_cast<real>(y)});
        f >> std::ws;
        if (f.peek() == ',') f.get();
    }
    return pts;
}

static void write_points_json(const fs::path &path, const std::vector<Point> &points, int precision) {
    fs::create_directories(path.parent_path());
    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream f(tmp);
    if (!f) throw std::runtime_error("could not open " + tmp.string());
    f << std::setprecision(precision);
    f << "[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i) f << ",";
        f << "[" << points[i].x << "," << points[i].y << "]";
    }
    f << "]\n";
    f.close();
    fs::rename(tmp, path);
}

struct QualityStats {
    real max_rel_G_error = 0;
    real rms_rel_G_error = 0;
    int stride = 1;
    uint64_t samples = 0;
};

static QualityStats compute_quality_stats(
    const std::vector<Point> &points,
    real target,
    const Config &cfg,
    int tid = -1,
    int index = -1,
    real ds = 0,
    std::uint64_t rollbacks = 0
) {
    QualityStats q;
    q.stride = std::max(1, cfg.quality_eval_stride);
    long double sumsq = 0.0L;
    long double maxe = 0.0L;
    uint64_t samples = 0;

    const std::uint64_t total = static_cast<std::uint64_t>(points.size());
    const std::uint64_t update_every = std::max<std::uint64_t>(1ULL, cfg.progress_every);
    std::uint64_t last_update = 0;

    auto publish = [&](std::uint64_t scanned) {
        if (!(cfg.progress && cfg.progress_style == "bars")) return;
        scanned = std::min(scanned, total);
        real frac = total ? static_cast<real>(static_cast<long double>(scanned) / static_cast<long double>(total)) : static_cast<real>(1);
        update_progress_slot_total(
            tid,
            index,
            target,
            frac,
            size_t_from_u64_clamped(scanned),
            size_t_from_u64_clamped(total),
            ds,
            rollbacks,
            "quality",
            "scan"
        );
    };

    publish(0);

    const size_t stride = static_cast<size_t>(q.stride);
    for (size_t i = 0; i < points.size(); i += stride) {
        try {
            Jet2 j = mandelbrot_jet(points[i].x, points[i].y, cfg);
            long double e = std::abs(static_cast<long double>(j.G - target)) / std::max(std::abs(static_cast<long double>(target)), 1.0e-300L);
            if (e > maxe) maxe = e;
            sumsq += e * e;
            ++samples;
        } catch (...) {
            // Treat unevaluable samples as infinite quality failure.
            maxe = std::numeric_limits<long double>::infinity();
            sumsq = std::numeric_limits<long double>::infinity();
            ++samples;
        }

        std::uint64_t scanned = std::min<std::uint64_t>(total, static_cast<std::uint64_t>(i) + static_cast<std::uint64_t>(stride));
        if (scanned == total || scanned - last_update >= update_every) {
            publish(scanned);
            last_update = scanned;
        }
    }

    publish(total);

    q.samples = samples;
    q.max_rel_G_error = static_cast<real>(maxe);
    q.rms_rel_G_error = samples ? static_cast<real>(std::sqrt(sumsq / static_cast<long double>(samples))) : static_cast<real>(0);
    return q;
}

static void write_meta_json(const fs::path &path, int index, real G, const TraceResult &half_result, const std::vector<Point> &full, real seconds, const Config &cfg, OutputFormat fmt, OutputValueType vt, const QualityStats &quality) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << std::setprecision(cfg.json_precision);
    const bool half_sym = (lower(trim(cfg.output_symmetry)) == "half");
    const std::uint64_t stored_points = half_sym ? static_cast<std::uint64_t>(half_result.points.size()) : static_cast<std::uint64_t>(full.size());
    const std::uint64_t logical_points = half_sym ? (stored_points >= 2 ? 2 * stored_points - 2 : stored_points) : stored_points;
    real len = half_sym ? static_cast<real>(2) * open_polyline_length(half_result.points) : polyline_length(full);
    real ar = half_sym ? upper_half_cross_sum(half_result.points) : signed_area(full);
    f << "{\n";
    f << "  \"index\": " << index << ",\n";
    f << "  \"target_G\": " << G << ",\n";
    f << "  \"points\": " << logical_points << ",\n";
    f << "  \"stored_points\": " << stored_points << ",\n";
    f << "  \"half_points\": " << (half_sym ? stored_points : half_result.points.size()) << ",\n";
    f << "  \"symmetry\": \"" << (half_sym ? "upper_half_mirror_y" : "full") << "\",\n";
    f << "  \"length\": " << len << ",\n";
    f << "  \"area_signed\": " << ar << ",\n";
    f << "  \"area\": " << std::abs(ar) << ",\n";
    f << "  \"used_ds\": " << half_result.used_ds << ",\n";
    f << "  \"rollbacks\": " << half_result.rollbacks << ",\n";
    f << "  \"seconds\": " << seconds << ",\n";
    f << "  \"max_rel_G_error\": " << quality.max_rel_G_error << ",\n";
    f << "  \"rms_rel_G_error\": " << quality.rms_rel_G_error << ",\n";
    f << "  \"quality_eval_stride\": " << quality.stride << ",\n";
    f << "  \"quality_samples\": " << quality.samples << ",\n";
    f << "  \"output_format\": \"" << output_format_name(fmt) << "\",\n";
    f << "  \"output_value_type\": \"" << value_type_name(vt) << "\",\n";
    f << "  \"output_value_size\": " << value_type_size(vt) << ",\n";
    f << "  \"compute_real\": \"" << precision_name() << "\",\n";
    f << "  \"compute_real_size\": " << compute_real_size() << "\n";
    f << "}\n";
}


static void write_meta_json_streamed_half(const fs::path &path, int index, real G, const TraceResult &half_result, const fs::path &contour_path, real seconds, const Config &cfg, OutputFormat fmt, OutputValueType vt, const QualityStats &quality, const HalfFileStats &half_stats) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << std::setprecision(cfg.json_precision);
    const std::uint64_t stored_points = half_stats.half_points;
    const std::uint64_t logical_points = stored_points >= 2 ? 2 * stored_points - 2 : stored_points;
    f << "{\n";
    f << "  \"index\": " << index << ",\n";
    f << "  \"target_G\": " << G << ",\n";
    f << "  \"points\": " << logical_points << ",\n";
    f << "  \"stored_points\": " << stored_points << ",\n";
    f << "  \"half_points\": " << stored_points << ",\n";
    f << "  \"symmetry\": \"upper_half_mirror_y\",\n";
    f << "  \"streamed\": true,\n";
    f << "  \"partial\": false,\n";
    if (half_result.recovered) {
        f << "  \"recovered_from_existing_contour\": true,\n";
        f << "  \"trace_diagnostics_known\": false,\n";
    }
    f << "  \"length\": " << half_stats.full_length << ",\n";
    f << "  \"area_signed\": " << half_stats.full_area_signed << ",\n";
    f << "  \"area\": " << std::abs(half_stats.full_area_signed) << ",\n";
    f << "  \"used_ds\": " << half_result.used_ds << ",\n";
    f << "  \"rollbacks\": " << half_result.rollbacks << ",\n";
    f << "  \"seconds\": " << seconds << ",\n";
    f << "  \"max_rel_G_error\": " << quality.max_rel_G_error << ",\n";
    f << "  \"rms_rel_G_error\": " << quality.rms_rel_G_error << ",\n";
    f << "  \"quality_eval_stride\": " << quality.stride << ",\n";
    f << "  \"quality_samples\": " << quality.samples << ",\n";
    f << "  \"output_format\": \"" << output_format_name(fmt) << "\",\n";
    f << "  \"output_value_type\": \"" << value_type_name(vt) << "\",\n";
    f << "  \"output_value_size\": " << value_type_size(vt) << ",\n";
    f << "  \"compute_real\": \"" << precision_name() << "\",\n";
    f << "  \"compute_real_size\": " << compute_real_size() << ",\n";
    f << "  \"contour_path\": \"" << contour_path.string() << "\"\n";
    f << "}\n";
}

static void write_failure_json(const fs::path &path, int index, real G, const std::string &message, real seconds, const Config &cfg) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << std::setprecision(cfg.json_precision);
    f << "{\n";
    f << "  \"index\": " << index << ",\n";
    f << "  \"target_G\": " << G << ",\n";
    f << "  \"failed\": true,\n";
    f << "  \"message\": \"";
    for (char ch : message) {
        if (ch == '\\' || ch == '\"') f << '\\';
        f << ch;
    }
    f << "\",\n";
    f << "  \"seconds\": " << seconds << "\n";
    f << "}\n";
}

static std::vector<real> make_levels(const Config &cfg) {
    std::vector<real> levels;
    levels.reserve(cfg.n);
    if (cfg.n == 1) { levels.push_back(cfg.g_start); return levels; }
    long double a = std::log(static_cast<long double>(cfg.g_start));
    long double b = std::log(static_cast<long double>(cfg.g_stop));
    for (int i = 0; i < cfg.n; ++i) {
        long double t = static_cast<long double>(i) / static_cast<long double>(cfg.n - 1);
        levels.push_back(static_cast<real>(std::exp(a + t * (b - a))));
    }
    return levels;
}

static std::string json_escape_snapshot(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << static_cast<char>(ch); break;
        }
    }
    return output.str();
}

static void write_run_config_snapshot(const fs::path &path, const Config &cfg) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Could not write " + path.string());
    f << std::setprecision(cfg.json_precision);
    f << "{\n"
      << "  \"schema\": \"mandelbrot-contour-effective-config-v1\",\n"
      << "  \"outputDir\": \"" << json_escape_snapshot(cfg.output_dir) << "\",\n"
      << "  \"threads\": " << cfg.threads << ",\n"
      << "  \"autoThreadsPolicy\": \"" << json_escape_snapshot(cfg.auto_threads_policy) << "\",\n"
      << "  \"leaveFree\": " << cfg.leave_free << ",\n"
      << "  \"autoThreadsFraction\": " << cfg.auto_threads_fraction << ",\n"
      << "  \"resume\": " << (cfg.resume ? "true" : "false") << ",\n"
      << "  \"gStart\": " << cfg.g_start << ",\n"
      << "  \"gStop\": " << cfg.g_stop << ",\n"
      << "  \"levels\": " << cfg.n << ",\n"
      << "  \"dsAtStart\": " << cfg.ds_at_start << ",\n"
      << "  \"dsPower\": " << cfg.ds_power << ",\n"
      << "  \"maxSteps\": " << cfg.max_steps << ",\n"
      << "  \"maxStepHalvings\": " << cfg.max_step_halvings << ",\n"
      << "  \"rollbackPoints\": " << cfg.rollback_points << ",\n"
      << "  \"maxRollbacks\": " << cfg.max_rollbacks << ",\n"
      << "  \"maxTurnAngle\": " << cfg.max_turn_angle << ",\n"
      << "  \"selfIntersectionWindow\": " << cfg.self_intersection_window << ",\n"
      << "  \"outputFormat\": \"" << json_escape_snapshot(cfg.output_format) << "\",\n"
      << "  \"outputValueType\": \"" << json_escape_snapshot(cfg.output_value_type) << "\",\n"
      << "  \"outputSymmetry\": \"" << json_escape_snapshot(cfg.output_symmetry) << "\",\n"
      << "  \"streamOutput\": " << (cfg.stream_output ? "true" : "false") << ",\n"
      << "  \"streamChunkMb\": " << cfg.stream_chunk_mb << ",\n"
      << "  \"streamCommitMarginPoints\": " << cfg.stream_commit_margin_points << ",\n"
      << "  \"stallDetection\": " << (cfg.stall_detection ? "true" : "false") << ",\n"
      << "  \"abortIfDsMinForPoints\": " << cfg.abort_if_ds_min_for_points << ",\n"
      << "  \"jsonPrecision\": " << cfg.json_precision << ",\n"
      << "  \"qualityEvalStride\": " << cfg.quality_eval_stride << ",\n"
      << "  \"progressStyle\": \"" << json_escape_snapshot(cfg.progress_style) << "\"\n"
      << "}\n";
}

static void write_points_any(const fs::path &path, const std::vector<Point> &points, const Config &cfg, OutputFormat fmt, OutputValueType vt) {
    if (fmt == OutputFormat::Json) write_points_json(path, points, cfg.json_precision);
    else write_points_bin(path, points, vt);
}


class BinPointWriter {
    fs::path final_path_;
    fs::path tmp_path_;
    std::ofstream f_;
    OutputValueType vt_;
    std::uint64_t count_ = 0;
    bool finalized_ = false;
public:
    BinPointWriter(const fs::path &path, OutputValueType vt) : final_path_(path), tmp_path_(path), vt_(vt) {
        fs::create_directories(path.parent_path());
        tmp_path_ += ".partial";
        f_.open(tmp_path_, std::ios::binary);
        if (!f_) throw std::runtime_error("could not open " + tmp_path_.string());
        const char magic[8] = {'M','B','C','T','B','I','N','1'};
        f_.write(magic, 8);
        write_scalar_binary<uint32_t>(f_, 1);
        write_scalar_binary<uint32_t>(f_, value_type_code(vt_));
        write_scalar_binary<uint32_t>(f_, value_type_size(vt_));
        write_scalar_binary<uint32_t>(f_, 0x01020304u);
        write_scalar_binary<uint64_t>(f_, 0ULL); // patched in finalize()
    }

    std::uint64_t count() const { return count_; }
    const fs::path &partial_path() const { return tmp_path_; }

    void write_point(const Point &p) {
        if (vt_ == OutputValueType::Float32) {
            write_scalar_binary<float>(f_, static_cast<float>(p.x));
            write_scalar_binary<float>(f_, static_cast<float>(p.y));
        } else if (vt_ == OutputValueType::Float64) {
            write_scalar_binary<double>(f_, static_cast<double>(p.x));
            write_scalar_binary<double>(f_, static_cast<double>(p.y));
        } else {
            write_scalar_binary<long double>(f_, static_cast<long double>(p.x));
            write_scalar_binary<long double>(f_, static_cast<long double>(p.y));
        }
        ++count_;
    }

    void write_points(const std::vector<Point> &pts, size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) write_point(pts[i]);
    }

    void finalize() {
        if (finalized_) return;
        f_.seekp(24, std::ios::beg);
        write_scalar_binary<uint64_t>(f_, count_);
        f_.close();
        fs::rename(tmp_path_, final_path_);
        finalized_ = true;
    }

    void keep_partial() {
        if (f_.is_open()) {
            f_.seekp(24, std::ios::beg);
            write_scalar_binary<uint64_t>(f_, count_);
            f_.close();
        }
    }

    ~BinPointWriter() {
        if (!finalized_) keep_partial();
    }
};


static Point read_one_point_payload(std::ifstream &f, OutputValueType vt) {
    if (vt == OutputValueType::Float32) {
        float x = read_scalar_binary<float>(f), y = read_scalar_binary<float>(f);
        return Point{static_cast<real>(x), static_cast<real>(y)};
    }
    if (vt == OutputValueType::Float64) {
        double x = read_scalar_binary<double>(f), y = read_scalar_binary<double>(f);
        return Point{static_cast<real>(x), static_cast<real>(y)};
    }
    long double x = read_scalar_binary<long double>(f), y = read_scalar_binary<long double>(f);
    return Point{static_cast<real>(x), static_cast<real>(y)};
}

struct BinContourInfo {
    OutputValueType value_type = OutputValueType::Float64;
    uint32_t value_size = 8;
    uint64_t count = 0;
    uintmax_t expected_size = 0;
    uintmax_t actual_size = 0;
};

static OutputValueType value_type_from_code_and_size(uint32_t code, uint32_t value_size, const fs::path &path) {
    if (code == 1 && value_size == 4) return OutputValueType::Float32;
    if (code == 2 && value_size == 8) return OutputValueType::Float64;
    if (code == 3 && value_size == sizeof(long double)) return OutputValueType::LongDouble;
    std::ostringstream oss;
    oss << "unsupported binary contour value type in " << path.string()
        << " (code=" << code << ", value_size=" << value_size << ")";
    throw std::runtime_error(oss.str());
}

static BinContourInfo inspect_bin_contour_file(const fs::path &path) {
    std::ifstream f = open_binary_input_retry(path);

    char magic[8]{};
    f.read(magic, 8);
    const char expected_magic[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected_magic, 8) != 0) {
        throw std::runtime_error("bad binary contour magic in " + path.string());
    }

    uint32_t version = read_scalar_binary<uint32_t>(f);
    uint32_t code = read_scalar_binary<uint32_t>(f);
    uint32_t value_size = read_scalar_binary<uint32_t>(f);
    uint32_t endian = read_scalar_binary<uint32_t>(f);
    uint64_t count = read_scalar_binary<uint64_t>(f);

    if (version != 1 || endian != 0x01020304u) {
        throw std::runtime_error("bad binary contour header in " + path.string());
    }

    BinContourInfo info;
    info.value_type = value_type_from_code_and_size(code, value_size, path);
    info.value_size = value_size;
    info.count = count;
    info.expected_size = static_cast<uintmax_t>(32) + static_cast<uintmax_t>(count) * static_cast<uintmax_t>(2) * static_cast<uintmax_t>(value_size);

    std::error_code ec;
    info.actual_size = fs::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("could not stat " + path.string() + ": " + ec.message());
    }

    if (info.actual_size != info.expected_size) {
        std::ostringstream oss;
        oss << "binary contour size mismatch in " << path.string()
            << ": actual=" << info.actual_size
            << " expected=" << info.expected_size
            << " count=" << info.count
            << " value_size=" << info.value_size;
        throw std::runtime_error(oss.str());
    }

    return info;
}

static HalfFileStats scan_half_bin_basic_stats(
    const fs::path &path,
    const Config *cfg = nullptr,
    int tid = -1,
    int index = -1,
    real target = 0,
    real ds = 0,
    std::uint64_t rollbacks = 0
) {
    std::ifstream f = open_binary_input_retry(path);
    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("bad binary contour magic in " + path.string());
    uint32_t version = read_scalar_binary<uint32_t>(f);
    uint32_t code = read_scalar_binary<uint32_t>(f);
    uint32_t value_size = read_scalar_binary<uint32_t>(f);
    uint32_t endian = read_scalar_binary<uint32_t>(f);
    uint64_t count = read_scalar_binary<uint64_t>(f);
    if (version != 1 || endian != 0x01020304u) throw std::runtime_error("bad binary contour header in " + path.string());
    OutputValueType vt;
    if (code == 1 && value_size == 4) vt = OutputValueType::Float32;
    else if (code == 2 && value_size == 8) vt = OutputValueType::Float64;
    else if (code == 3 && value_size == sizeof(long double)) vt = OutputValueType::LongDouble;
    else throw std::runtime_error("unsupported binary contour value type in " + path.string());

    const std::uint64_t update_every = cfg ? std::max<std::uint64_t>(1ULL, cfg->progress_every) : 0ULL;
    std::uint64_t last_update = 0;

    auto publish = [&](std::uint64_t scanned) {
        if (!cfg || !(cfg->progress && cfg->progress_style == "bars")) return;
        scanned = std::min(scanned, count);
        real frac = count ? static_cast<real>(static_cast<long double>(scanned) / static_cast<long double>(count)) : static_cast<real>(1);
        update_progress_slot_total(
            tid,
            index,
            target,
            frac,
            size_t_from_u64_clamped(scanned),
            size_t_from_u64_clamped(count),
            ds,
            rollbacks,
            "stats",
            "scan"
        );
    };

    publish(0);

    HalfFileStats st;
    st.half_points = count;
    if (count == 0) {
        publish(0);
        return st;
    }

    Point prev = read_one_point_payload(f, vt);
    long double len = 0.0L;
    long double cross = 0.0L;
    publish(1);
    last_update = 1;

    for (uint64_t i = 1; i < count; ++i) {
        Point cur = read_one_point_payload(f, vt);
        len += hypot2(cur.x - prev.x, cur.y - prev.y);
        cross += static_cast<long double>(prev.x) * cur.y - static_cast<long double>(prev.y) * cur.x;
        prev = cur;

        std::uint64_t scanned = i + 1;
        if (scanned == count || scanned - last_update >= update_every) {
            publish(scanned);
            last_update = scanned;
        }
    }
    st.full_length = static_cast<real>(2.0L * len);
    st.full_area_signed = static_cast<real>(cross);
    publish(count);
    return st;
}

static QualityStats compute_quality_stats_bin_half(
    const fs::path &path,
    real target,
    const Config &cfg,
    int tid = -1,
    int index = -1,
    real ds = 0,
    std::uint64_t rollbacks = 0
) {
    std::ifstream f = open_binary_input_retry(path);
    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("bad binary contour magic in " + path.string());
    uint32_t version = read_scalar_binary<uint32_t>(f);
    uint32_t code = read_scalar_binary<uint32_t>(f);
    uint32_t value_size = read_scalar_binary<uint32_t>(f);
    uint32_t endian = read_scalar_binary<uint32_t>(f);
    uint64_t count = read_scalar_binary<uint64_t>(f);
    if (version != 1 || endian != 0x01020304u) throw std::runtime_error("bad binary contour header in " + path.string());
    OutputValueType vt;
    if (code == 1 && value_size == 4) vt = OutputValueType::Float32;
    else if (code == 2 && value_size == 8) vt = OutputValueType::Float64;
    else if (code == 3 && value_size == sizeof(long double)) vt = OutputValueType::LongDouble;
    else throw std::runtime_error("unsupported binary contour value type in " + path.string());

    QualityStats q;
    q.stride = std::max(1, cfg.quality_eval_stride);
    long double sumsq = 0.0L, maxe = 0.0L;
    uint64_t samples = 0;

    const std::uint64_t update_every = std::max<std::uint64_t>(1ULL, cfg.progress_every);
    std::uint64_t last_update = 0;

    auto publish = [&](std::uint64_t scanned) {
        if (!(cfg.progress && cfg.progress_style == "bars")) return;
        scanned = std::min(scanned, count);
        real frac = count ? static_cast<real>(static_cast<long double>(scanned) / static_cast<long double>(count)) : static_cast<real>(1);
        update_progress_slot_total(
            tid,
            index,
            target,
            frac,
            size_t_from_u64_clamped(scanned),
            size_t_from_u64_clamped(count),
            ds,
            rollbacks,
            "quality",
            "scan"
        );
    };

    publish(0);

    const uint64_t stride_u64 = static_cast<uint64_t>(q.stride);
    for (uint64_t i = 0; i < count; ++i) {
        Point p = read_one_point_payload(f, vt);
        if (i % stride_u64 == 0) {
            try {
                Jet2 j = mandelbrot_jet(p.x, p.y, cfg);
                long double e = std::abs(static_cast<long double>(j.G - target)) / std::max(std::abs(static_cast<long double>(target)), 1.0e-300L);
                maxe = std::max(maxe, e);
                sumsq += e * e;
                ++samples;
            } catch (...) {
                maxe = std::numeric_limits<long double>::infinity();
                sumsq = std::numeric_limits<long double>::infinity();
                ++samples;
            }
        }

        std::uint64_t scanned = i + 1;
        if (scanned == count || scanned - last_update >= update_every) {
            publish(scanned);
            last_update = scanned;
        }
    }

    publish(count);

    q.samples = samples;
    q.max_rel_G_error = static_cast<real>(maxe);
    q.rms_rel_G_error = samples ? static_cast<real>(std::sqrt(sumsq / static_cast<long double>(samples))) : static_cast<real>(0);
    return q;
}

static TraceResult trace_half_streamed_bin(real target, real ds_goal, const Config &cfg, int index, int tid, const fs::path &path, OutputValueType vt) {
    auto [left, right] = real_axis_intersections(target, cfg);
    real x_left = left.x;
    real x_right = right.x;
    std::vector<Point> path_recent;
    path_recent.reserve(100000);
    path_recent.push_back(right);
    BinPointWriter writer(path, vt);

    const std::uint64_t auto_margin = std::max<std::uint64_t>({cfg.rollback_points + 16ULL, cfg.self_intersection_window + 16ULL, 10000ULL});
    const size_t commit_margin = size_t_from_u64_clamped(cfg.stream_commit_margin_points ? cfg.stream_commit_margin_points : auto_margin);
    const uint64_t bytes_per_point = static_cast<uint64_t>(value_type_size(vt)) * 2ULL;
    const size_t chunk_points = std::max<size_t>(1, size_t_from_u64_clamped((std::max<std::uint64_t>(1, cfg.stream_chunk_mb) * 1024ULL * 1024ULL) / std::max<uint64_t>(1, bytes_per_point)));

    auto total_points = [&]() -> std::uint64_t { return writer.count() + static_cast<std::uint64_t>(path_recent.size()); };
    auto commit_safe = [&](bool force) {
        if (force) {
            writer.write_points(path_recent, 0, path_recent.size());
            path_recent.clear();
            return;
        }
        if (path_recent.size() <= commit_margin + chunk_points) return;
        size_t ncommit = path_recent.size() - commit_margin;
        writer.write_points(path_recent, 0, ncommit);
        path_recent.erase(path_recent.begin(), path_recent.begin() + static_cast<std::ptrdiff_t>(ncommit));
    };

    int direction = +1;
    real ds_nom = ds_goal;
    int successes = 0;
    std::uint64_t rollbacks = 0;
    int min_steps_before_crossing = 16;
    real best_progress_fraction = 0;
    bool ds_min_stall_active = false;
    std::uint64_t ds_min_stall_points = 0;
    real ds_min_stall_progress = 0;

    for (std::uint64_t step = 0; step < cfg.max_steps; ++step) {
        Point cur = path_recent.back();
        bool accepted = false;
        real accepted_ds = ds_nom;
        int accepted_halvings = 0;
        std::string last_reason;

        for (int h = 0; h <= cfg.max_step_halvings; ++h) {
            real ds_try = ds_nom * std::pow(static_cast<real>(0.5L), h);
            if (ds_try < cfg.ds_min) { last_reason = "ds below ds_min"; break; }
            try {
                Jet2 j = mandelbrot_jet(cur.x, cur.y, cfg);
                real kappa = curvature_from_jet(j);
                if (is_finite(kappa) && std::abs(kappa) > 0) {
                    ds_try = std::min(ds_try, cfg.max_turn_angle / std::abs(kappa));
                    if (ds_try < cfg.ds_min) ds_try = cfg.ds_min;
                }
                Point full = rk4_step_projected(cur.x, cur.y, target, ds_try, direction, cfg);
                Point half1 = rk4_step_projected(cur.x, cur.y, target, ds_try/static_cast<real>(2), direction, cfg);
                Point half2 = rk4_step_projected(half1.x, half1.y, target, ds_try/static_cast<real>(2), direction, cfg);
                real err = hypot2(full.x-half2.x, full.y-half2.y);
                if (err > cfg.step_error_factor * std::max(ds_try, cfg.ds_min)) { last_reason = "step-doubling error"; continue; }
                Point cand = half2;

                real seam_tol = std::max(cfg.seam_tol_factor * ds_try, cfg.seam_tol_min);
                bool crossed = (step >= static_cast<std::uint64_t>(min_steps_before_crossing) && cur.y > 0 && cand.y <= 0);
                auto xcross = crossed ? x_cross_at_y0(cur, cand) : std::optional<real>{};
                bool near_left = hypot2(cand.x - x_left, cand.y) <= seam_tol || (xcross && std::abs(*xcross - x_left) <= seam_tol);
                if (crossed) {
                    if (near_left) {
                        path_recent.push_back(left);
                        commit_safe(true);
                        writer.finalize();
                        return TraceResult{{}, ds_nom, rollbacks, true, writer.count()};
                    }
                    last_reason = "crossed y=0 away from left seam";
                    continue;
                }
                if (near_left && step >= static_cast<std::uint64_t>(min_steps_before_crossing)) {
                    path_recent.push_back(left);
                    commit_safe(true);
                    writer.finalize();
                    return TraceResult{{}, ds_nom, rollbacks, true, writer.count()};
                }
                if (path_recent.size() >= 2 && cfg.max_actual_turn_angle > 0) {
                    real a = angle_between_segments(path_recent[path_recent.size()-2], path_recent[path_recent.size()-1], cand);
                    if (a > cfg.max_actual_turn_angle) { last_reason = "actual turn too large"; continue; }
                }
                if (candidate_intersects_recent(path_recent, cand, static_cast<int>(std::min<std::uint64_t>(cfg.self_intersection_window, static_cast<std::uint64_t>(std::numeric_limits<int>::max()))))) {
                    last_reason = "recent self-intersection";
                    continue;
                }
                path_recent.push_back(cand);
                accepted = true;
                accepted_ds = ds_try;
                accepted_halvings = h;
                commit_safe(false);
                break;
            } catch (const std::exception &e) { last_reason = e.what(); continue; }
        }

        if (accepted) {
            if (cfg.adaptive_ds) {
                if (accepted_halvings > 0) { ds_nom = std::max(cfg.ds_min, accepted_ds); successes = 0; }
                else { ++successes; if (successes >= cfg.grow_after_successes) { ds_nom = std::min(ds_goal, ds_nom * cfg.ds_growth); successes = 0; } }
            }
            if (cfg.progress && cfg.progress_every > 0 && step % cfg.progress_every == 0 && step > 0) {
                real frac = progress_fraction(path_recent, x_left, x_right);
                best_progress_fraction = std::max(best_progress_fraction, frac);
                if (cfg.progress_style == "bars") update_progress_slot(tid, index, target, best_progress_fraction, static_cast<size_t>(std::min<std::uint64_t>(total_points(), static_cast<std::uint64_t>(std::numeric_limits<size_t>::max()))), ds_nom, rollbacks, "streaming");
                else if (cfg.progress_style == "lines") {
                    std::lock_guard<std::mutex> lock(g_print_mutex);
                    std::cerr << "[" << index << "] G=" << static_cast<double>(target)
                              << " progress=" << static_cast<double>(100*best_progress_fraction) << "% points=" << total_points()
                              << " ds=" << static_cast<double>(ds_nom) << " rollbacks=" << rollbacks << " streaming\n";
                }
            }
            continue;
        }

        ++rollbacks;
        const size_t rollback_points = size_t_from_u64_clamped(cfg.rollback_points);
        if (rollbacks > cfg.max_rollbacks || path_recent.size() <= rollback_points + 4) {
            writer.keep_partial();
            throw std::runtime_error("trace failed near step " + std::to_string(step) + ": " + last_reason);
        }
        size_t keep = path_recent.size() - rollback_points;
        path_recent.resize(std::max<size_t>(2, keep));
        ds_nom = std::max(cfg.ds_min, ds_nom * cfg.rollback_shrink);
        successes = 0;
        if (cfg.stall_detection && cfg.abort_if_ds_min_for_points > 0 && ds_nom <= cfg.ds_min * static_cast<real>(1.0000001L)) {
            if (!ds_min_stall_active) {
                ds_min_stall_active = true;
                ds_min_stall_points = total_points();
                ds_min_stall_progress = best_progress_fraction;
            } else {
                std::uint64_t delta_points = total_points() > ds_min_stall_points ? total_points() - ds_min_stall_points : 0;
                real delta_progress = best_progress_fraction - ds_min_stall_progress;
                if (delta_points >= cfg.abort_if_ds_min_for_points && delta_progress < cfg.stall_progress_epsilon) {
                    writer.keep_partial();
                    throw std::runtime_error("stall detected: ds_min rollback loop with no meaningful progress");
                }
            }
        } else ds_min_stall_active = false;
    }
    writer.keep_partial();
    throw std::runtime_error("max_steps exceeded");
}

static std::vector<Point> read_points_any(const fs::path &path, OutputFormat fmt, OutputValueType *stored_type=nullptr, uint32_t *stored_size=nullptr) {
    if (fmt == OutputFormat::Json) return read_points_json(path);
    return read_points_bin(path, stored_type, stored_size);
}

static bool harmonize_existing_if_needed(const fs::path &out, int idx, const Config &cfg, OutputFormat desired_fmt, OutputValueType desired_vt) {
    fs::path desired = contour_path_for(out, idx, desired_fmt);
    fs::path other = contour_path_for(out, idx, desired_fmt == OutputFormat::Json ? OutputFormat::Bin : OutputFormat::Json);

    if (fs::exists(desired)) {
        if (cfg.harmonize_existing_format_on_resume && cfg.delete_other_format_on_harmonize && fs::exists(other)) {
            fs::remove(other);
        }
        return true;
    }

    if (fs::exists(other)) {
        if (!cfg.harmonize_existing_format_on_resume) return true;
        OutputFormat other_fmt = desired_fmt == OutputFormat::Json ? OutputFormat::Bin : OutputFormat::Json;
        OutputValueType stored_type = OutputValueType::Float64;
        uint32_t stored_size = 0;
        auto pts = read_points_any(other, other_fmt, &stored_type, &stored_size);
        {
            std::lock_guard<std::mutex> lock(g_print_mutex);
            std::cerr << "[" << idx << "] harmonizing existing " << output_format_name(other_fmt)
                      << " -> " << output_format_name(desired_fmt);
            if (other_fmt == OutputFormat::Bin) {
                std::cerr << " (old " << value_type_name(stored_type) << ", " << stored_size << " bytes)";
                uint32_t desired_size = desired_fmt == OutputFormat::Bin ? value_type_size(desired_vt) : static_cast<uint32_t>(0);
                if (desired_fmt == OutputFormat::Bin && stored_size < desired_size) std::cerr << " WARNING: existing precision lower than desired";
            }
            std::cerr << "\n";
        }
        write_points_any(desired, pts, cfg, desired_fmt, desired_vt);
        if (cfg.delete_other_format_on_harmonize) fs::remove(other);
        return true;
    }

    return false;
}


static bool recover_existing_half_bin_meta(
    const fs::path &contour_path,
    const fs::path &meta_path,
    const fs::path &failure_path,
    int idx,
    real G,
    const Config &cfg,
    OutputFormat fmt,
    OutputValueType desired_vt,
    int tid
) {
    if (fmt != OutputFormat::Bin || lower(trim(cfg.output_symmetry)) != "half") {
        return false;
    }
    if (!fs::exists(contour_path)) {
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();

    BinContourInfo info = inspect_bin_contour_file(contour_path);

    TraceResult half;
    half.streamed = true;
    half.half_points = info.count;
    half.recovered = true;
    // These are unknown when recovering after a post-trace readback/meta failure.
    half.used_ds = static_cast<real>(0);
    half.rollbacks = 0;

    update_progress_slot_total(
        tid,
        idx,
        G,
        static_cast<real>(0),
        0,
        size_t_from_u64_clamped(info.count),
        half.used_ds,
        half.rollbacks,
        "recover",
        "scan"
    );

    QualityStats quality = compute_quality_stats_bin_half(
        contour_path,
        G,
        cfg,
        tid,
        idx,
        half.used_ds,
        half.rollbacks
    );

    HalfFileStats half_stats = scan_half_bin_basic_stats(
        contour_path,
        &cfg,
        tid,
        idx,
        G,
        half.used_ds,
        half.rollbacks
    );

    auto t1 = std::chrono::steady_clock::now();
    real seconds = static_cast<real>(std::chrono::duration<double>(t1 - t0).count());

    // Record the actual stored type from the existing file, not merely the current desired type.
    (void)desired_vt;
    write_meta_json_streamed_half(
        meta_path,
        idx,
        G,
        half,
        contour_path,
        seconds,
        cfg,
        fmt,
        info.value_type,
        quality,
        half_stats
    );

    std::error_code ec;
    fs::remove(failure_path, ec);
    return true;
}


static bool stderr_is_terminal() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(STDERR_FILENO) != 0;
#else
    return true;
#endif
}

static std::string compile_precision_name() { return "long double + adaptive Boost.Multiprecision"; }

static void leave_alternate_screen() {
    if (!g_alt_screen_active.exchange(false)) return;
    g_alt_screen_active_signal = 0;
    std::cerr << "\033[?25h\033[?1049l";
    std::cerr.flush();
}

static void enter_alternate_screen() {
    if (g_alt_screen_active.exchange(true)) return;
    g_alt_screen_active_signal = 1;
    std::cerr << "\033[?1049h\033[?25l\033[H\033[2J";
    std::cerr.flush();
}

#if defined(__unix__) || defined(__APPLE__)
static void progress_signal_cleanup_handler(int sig) {
    if (g_alt_screen_active_signal) {
        const char seq[] = "\033[?25h\033[?1049l";
        (void)!write(STDERR_FILENO, seq, sizeof(seq) - 1);
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}
#endif

static void install_terminal_cleanup_handlers() {
    std::atexit(leave_alternate_screen);
#if defined(__unix__) || defined(__APPLE__)
    std::signal(SIGINT, progress_signal_cleanup_handler);
    std::signal(SIGTERM, progress_signal_cleanup_handler);
#endif
}

static int terminal_width_columns() {
#if defined(__unix__) || defined(__APPLE__)
    struct winsize ws{};
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return static_cast<int>(ws.ws_col);
    }
#endif
    const char *env_cols = std::getenv("COLUMNS");
    if (env_cols) {
        try {
            int n = std::stoi(env_cols);
            if (n > 0) return n;
        } catch (...) {}
    }
    return 120;
}

static std::string progress_bar(real frac, int width) {
    width = std::max(1, width);
    frac = std::max(static_cast<real>(0), std::min(static_cast<real>(1), frac));
    int filled = static_cast<int>(std::round(static_cast<long double>(frac * width)));
    filled = std::max(0, std::min(width, filled));
    return std::string(filled, '#') + std::string(width - filled, '-');
}

static std::string fit_to_width(std::string s, int width) {
    if (width <= 0) return "";
    if (static_cast<int>(s.size()) <= width) return s;
    if (width <= 1) return s.substr(0, static_cast<size_t>(width));
    if (width <= 3) return s.substr(0, static_cast<size_t>(width));
    return s.substr(0, static_cast<size_t>(width - 3)) + "...";
}

static std::string fmt_real_short(long double x, int precision = 3) {
    std::ostringstream oss;
    oss << std::setprecision(precision) << static_cast<double>(x);
    return oss.str();
}

static std::string fmt_percent(real frac) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << static_cast<double>(100 * frac) << "%";
    return oss.str();
}

static std::string format_count(size_t n) {
    std::string s = std::to_string(n);
    std::string out;
    int k = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (k && k % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++k;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

static std::string render_progress_slot_line(const Config &cfg, int tid, const ProgressSlot &slot, int terminal_cols) {
    int width = std::max(20, terminal_cols - 1);  // leave one column so terminal does not wrap at the edge

    std::ostringstream id;
    id << "T" << std::setw(2) << std::setfill('0') << tid << std::setfill(' ');

    if (!slot.active) {
        return fit_to_width(id.str() + " idle", width);
    }

    std::ostringstream idx;
    idx << " [" << std::setw(5) << std::setfill('0') << slot.index << std::setfill(' ') << "] ";

    std::string prefix = id.str() + idx.str() + "G=" + fmt_real_short(static_cast<long double>(slot.G), 6) + " ";
    std::string pct = fmt_percent(slot.fraction);

    std::string label = slot.count_label.empty() ? "pts" : slot.count_label;
    std::string count_field = label + "=" + format_count(slot.points);
    if (slot.total_points > 0) count_field += "/" + format_count(slot.total_points);

    std::vector<std::string> suffixes;
    suffixes.push_back(count_field + " ds=" + fmt_real_short(static_cast<long double>(slot.ds), 3) + " rb=" + std::to_string(slot.rollbacks) + " " + slot.status);
    suffixes.push_back(count_field + " rb=" + std::to_string(slot.rollbacks) + " " + slot.status);
    suffixes.push_back(count_field + " " + slot.status);
    suffixes.push_back(slot.status);
    suffixes.push_back("");

    for (const std::string &suffix : suffixes) {
        int fixed_len = static_cast<int>(prefix.size()) + 1 + 1 + 1 + static_cast<int>(pct.size()); // '[' + ']' + spaces-ish
        if (!suffix.empty()) fixed_len += 1 + static_cast<int>(suffix.size());
        int available_bar = width - fixed_len;
        int bar_width = std::min(cfg.progress_bar_width, available_bar);
        if (bar_width >= 6) {
            std::string line = prefix + "[" + progress_bar(slot.fraction, bar_width) + "] " + pct;
            if (!suffix.empty()) line += " " + suffix;
            return fit_to_width(line, width);
        }
    }

    // Very narrow terminal: drop the bar entirely.
    std::string compact = id.str() + idx.str() + pct + " " + slot.status;
    if (static_cast<int>(compact.size()) <= width) return compact;

    compact = id.str() + " " + pct;
    return fit_to_width(compact, width);
}

static void monitor_progress_loop(
    const Config &cfg,
    int nthreads,
    int total,
    const std::atomic<int> &done,
    const std::atomic<int> &failed,
    std::vector<std::string> header_lines
) {
    if (!cfg.progress || cfg.progress_style != "bars") return;

    const bool alternate = cfg.progress_screen == "alternate" && stderr_is_terminal();
    bool first = true;
    int lines = static_cast<int>(header_lines.size()) + nthreads + 1;

    while (!g_monitor_stop.load()) {
        if (alternate) {
            // Full repaint in the alternate screen. This cleans up any previous
            // wrapped/truncated rows after a terminal resize.
            std::cerr << "\033[H";
        } else {
            if (!first) std::cerr << "\033[" << lines << "A";
        }
        first = false;

        std::vector<ProgressSlot> slots;
        {
            std::lock_guard<std::mutex> lock(g_progress_mutex);
            slots = g_progress_slots;
        }
        int cols = terminal_width_columns();
        int safe_cols = std::max(20, cols - 1);

        for (const auto &line : header_lines) {
            std::cerr << "\r\033[2K" << fit_to_width(line, safe_cols) << "\n";
        }

        for (int i = 0; i < nthreads; ++i) {
            std::cerr << "\r\033[2K";
            const auto &slot = slots[static_cast<size_t>(i)];
            std::cerr << render_progress_slot_line(cfg, i, slot, cols) << "\n";
        }

        std::string footer = "done=" + std::to_string(done.load()) + "/" + std::to_string(total)
                           + " failed=" + std::to_string(failed.load());
        std::cerr << "\r\033[2K" << fit_to_width(footer, safe_cols) << "\n";
        if (alternate) std::cerr << "\033[J";
        std::cerr.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.progress_refresh_ms));
    }
}

static void update_progress_slot_total(int tid, int index, real G, real frac, size_t points, size_t total_points, real ds, std::uint64_t rollbacks, const std::string &status, const std::string &count_label) {
    if (tid < 0) return;
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    if (static_cast<size_t>(tid) >= g_progress_slots.size()) return;
    auto &slot = g_progress_slots[static_cast<size_t>(tid)];
    slot.active = true;
    slot.index = index;
    slot.G = G;
    slot.fraction = frac;
    slot.points = points;
    slot.total_points = total_points;
    slot.count_label = count_label.empty() ? "pts" : count_label;
    slot.ds = ds;
    slot.rollbacks = rollbacks;
    slot.status = status;
}

static void update_progress_slot(int tid, int index, real G, real frac, size_t points, real ds, std::uint64_t rollbacks, const std::string &status) {
    update_progress_slot_total(tid, index, G, frac, points, 0, ds, rollbacks, status, "pts");
}

static void clear_progress_slot(int tid) {
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    if (tid >= 0 && static_cast<size_t>(tid) < g_progress_slots.size()) g_progress_slots[static_cast<size_t>(tid)] = ProgressSlot{};
}


static int resolve_thread_count(const Config &cfg) {
    unsigned hw_raw = std::thread::hardware_concurrency();
    int hw = std::max(1, static_cast<int>(hw_raw == 0 ? 1 : hw_raw));

    if (cfg.threads > 0) {
        return std::max(1, cfg.threads);
    }

    if (cfg.auto_threads_policy == "all") {
        return hw;
    }

    if (cfg.auto_threads_policy == "fraction") {
        int n = static_cast<int>(std::floor(static_cast<long double>(hw) * static_cast<long double>(cfg.auto_threads_fraction)));
        return std::max(1, std::min(hw, n));
    }

    // Default: leave a little oxygen for the OS, browser, terminal, Python postprocessing, etc.
    // On a 12-thread CPU and leave_free=2, this gives 10.
    int n = hw - cfg.leave_free;
    return std::max(1, std::min(hw, n));
}


int main(int argc, char **argv) {
    install_terminal_cleanup_handlers();
    try {
        const std::string usage = "Usage: contours [--config PATH]";
        const auto cli = mandelbrot::repo::parse_common_cli(argc, argv, usage);
        if (cli.help) { std::cout << usage << '\n'; return 0; }
        if (!cli.remaining.empty()) throw std::runtime_error("Unknown option: " + cli.remaining.front() + "\n" + usage);
        const fs::path cfg_path = cli.config.empty()
            ? mandelbrot::repo::find_code_root(fs::path(argv[0] ? argv[0] : ".")) / "mandelbrot.json"
            : cli.config;
        Config cfg = load_repository_config(cli.config, argv[0]);
        fs::path out = cfg.output_dir;
        OutputFormat desired_fmt = output_format_from_config(cfg);
        OutputValueType desired_vt = resolve_output_value_type(cfg);
        fs::create_directories(out / "contours");
        fs::create_directories(out / "contours_bin");
        fs::create_directories(out / "meta");
        fs::create_directories(out / "failed");
        write_run_config_snapshot(out / "effective_config.json", cfg);

        auto levels = make_levels(cfg);
        std::vector<int> order(cfg.n);
        for (int i = 0; i < cfg.n; ++i) order[i] = i;
        if (cfg.low_g_first) std::reverse(order.begin(), order.end());

        int nthreads = resolve_thread_count(cfg);
        unsigned hw_raw = std::thread::hardware_concurrency();
        int hw_threads = std::max(1, static_cast<int>(hw_raw == 0 ? 1 : hw_raw));
        int free_threads = std::max(0, hw_threads - nthreads);

        auto thread_policy_summary = [&]() -> std::string {
            std::ostringstream oss;
            if (cfg.threads > 0) {
                oss << "manual=" << cfg.threads;
            } else if (cfg.auto_threads_policy == "all") {
                oss << "all";
            } else if (cfg.auto_threads_policy == "fraction") {
                oss << "fraction=" << static_cast<double>(cfg.auto_threads_fraction);
            } else {
                oss << "leave_free=" << cfg.leave_free;
            }
            return oss.str();
        };

        std::vector<std::string> dashboard_header;
        {
            std::ostringstream line1;
            line1 << "C++ Mandelbrot contour tracer"
                  << " | precision=" << compile_precision_name()
                  << " | config=" << cfg_path.string()
                  << " | output=" << out.string();
            dashboard_header.push_back(line1.str());

            std::ostringstream line2;
            line2 << "levels=" << cfg.n
                  << " | hw=" << hw_threads
                  << " used=" << nthreads
                  << " free=" << free_threads
                  << " | thread_policy=" << thread_policy_summary();
            dashboard_header.push_back(line2.str());

            std::ostringstream line3;
            line3 << "output_format=" << output_format_name(desired_fmt);
            if (desired_fmt == OutputFormat::Bin) {
                line3 << " (" << value_type_name(desired_vt) << ", " << value_type_size(desired_vt) << " bytes/scalar)";
            }
            line3 << " | G=[" << static_cast<double>(cfg.g_start) << " -> " << static_cast<double>(cfg.g_stop) << "]"
                  << " | ds_min=" << static_cast<double>(cfg.ds_min)
                  << " | max_steps=" << cfg.max_steps;
            dashboard_header.push_back(line3.str());
        }

        {
            std::lock_guard<std::mutex> lock(g_print_mutex);
            for (const auto &line : dashboard_header) {
                std::cerr << line << "\n";
            }
        }

        std::atomic<int> next_job{0};
        std::atomic<int> done{0};
        std::atomic<int> failed{0};
        g_progress_slots.assign(static_cast<size_t>(nthreads), ProgressSlot{});
        g_monitor_stop.store(false);
        const bool use_alternate_screen = cfg.progress && cfg.progress_style == "bars" && cfg.progress_screen == "alternate" && stderr_is_terminal();
        if (use_alternate_screen) enter_alternate_screen();
        std::thread monitor;
        if (cfg.progress && cfg.progress_style == "bars") {
            monitor = std::thread(
                monitor_progress_loop,
                std::cref(cfg),
                nthreads,
                cfg.n,
                std::cref(done),
                std::cref(failed),
                dashboard_header
            );
        }

        auto worker = [&](int tid) {
            while (true) {
                int j = next_job.fetch_add(1);
                if (j >= static_cast<int>(order.size())) break;
                int idx = order[j];
                real G = levels[idx];
                fs::path contour_path = contour_path_for(out, idx, desired_fmt);
                fs::path meta_path = out / "meta" / (stem_for_index(idx) + ".json");
                fs::path failure_path = out / "failed" / (stem_for_index(idx) + ".json");

                if (cfg.resume && !cfg.overwrite && harmonize_existing_if_needed(out, idx, cfg, desired_fmt, desired_vt)) {
                    bool recovered_meta = false;
                    bool recovery_failed = false;
                    std::string recovery_error;

                    if (cfg.write_meta && !fs::exists(meta_path)) {
                        try {
                            recovered_meta = recover_existing_half_bin_meta(
                                contour_path,
                                meta_path,
                                failure_path,
                                idx,
                                G,
                                cfg,
                                desired_fmt,
                                desired_vt,
                                tid
                            );
                        } catch (const std::exception &e) {
                            recovery_failed = true;
                            recovery_error = e.what();
                            if (cfg.write_failure_meta) {
                                write_failure_json(
                                    failure_path,
                                    idx,
                                    G,
                                    std::string("existing contour meta recovery failed: ") + e.what(),
                                    static_cast<real>(0),
                                    cfg
                                );
                            }
                            ++failed;
                        }
                    } else if (cfg.write_meta && fs::exists(meta_path)) {
                        std::error_code ec;
                        fs::remove(failure_path, ec);
                    }

                    int d = ++done;
                    if (cfg.progress_style == "bars") {
                        clear_progress_slot(tid);
                    } else {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cerr << "[" << idx << "/" << cfg.n << "] skip existing G=" << static_cast<double>(G);
                        if (recovered_meta) std::cerr << " recovered-meta";
                        if (recovery_failed) std::cerr << " recovery-failed: " << recovery_error;
                        std::cerr << " done=" << d << "/" << cfg.n << "\n";
                    }
                    continue;
                }

                auto t0 = std::chrono::steady_clock::now();
                try {
                    real ds = choose_ds(G, cfg);
                    if (cfg.progress_style == "bars") {
                        update_progress_slot(tid, idx, G, 0, 0, ds, 0, "starting");
                    } else {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cerr << "[" << idx << "/" << cfg.n << "] start G=" << static_cast<double>(G)
                                  << " ds=" << static_cast<double>(ds) << " thread=" << tid << "\n";
                    }
                    TraceResult half;
                    std::vector<Point> full;
                    QualityStats quality;
                    real seconds{};
                    if (cfg.stream_output && cfg.output_symmetry == "half") {
                        half = trace_half_streamed_bin(G, ds, cfg, idx, tid, contour_path, desired_vt);
                        auto t1 = std::chrono::steady_clock::now();
                        seconds = static_cast<real>(std::chrono::duration<double>(t1 - t0).count());
                        update_progress_slot_total(tid, idx, G, 0, 0, size_t_from_u64_clamped(half.half_points), half.used_ds, half.rollbacks, "quality", "scan");
                        quality = compute_quality_stats_bin_half(contour_path, G, cfg, tid, idx, half.used_ds, half.rollbacks);
                        HalfFileStats half_stats = scan_half_bin_basic_stats(contour_path, &cfg, tid, idx, G, half.used_ds, half.rollbacks);
                        if (cfg.write_meta) write_meta_json_streamed_half(meta_path, idx, G, half, contour_path, seconds, cfg, desired_fmt, desired_vt, quality, half_stats);
                        {
                            std::error_code ec;
                            fs::remove(failure_path, ec);
                        }
                    } else {
                        half = trace_half(G, ds, cfg, idx, tid);
                        if (cfg.output_symmetry == "half") {
                            auto t1 = std::chrono::steady_clock::now();
                            seconds = static_cast<real>(std::chrono::duration<double>(t1 - t0).count());
                            update_progress_slot_total(tid, idx, G, 0, 0, half.points.size(), half.used_ds, half.rollbacks, "quality", "scan");
                            quality = compute_quality_stats(half.points, G, cfg, tid, idx, half.used_ds, half.rollbacks);
                            write_points_any(contour_path, half.points, cfg, desired_fmt, desired_vt);
                            if (cfg.write_meta) write_meta_json(meta_path, idx, G, half, full, seconds, cfg, desired_fmt, desired_vt, quality);
                            {
                                std::error_code ec;
                                fs::remove(failure_path, ec);
                            }
                        } else {
                            full = mirror_full(half.points);
                            auto t1 = std::chrono::steady_clock::now();
                            seconds = static_cast<real>(std::chrono::duration<double>(t1 - t0).count());
                            update_progress_slot_total(tid, idx, G, 0, 0, half.points.size(), half.used_ds, half.rollbacks, "quality", "scan");
                            quality = compute_quality_stats(half.points, G, cfg, tid, idx, half.used_ds, half.rollbacks);
                            write_points_any(contour_path, full, cfg, desired_fmt, desired_vt);
                            if (cfg.write_meta) write_meta_json(meta_path, idx, G, half, full, seconds, cfg, desired_fmt, desired_vt, quality);
                            {
                                std::error_code ec;
                                fs::remove(failure_path, ec);
                            }
                        }
                    }
                    int d = ++done;
                    if (cfg.progress_style == "bars") {
                        clear_progress_slot(tid);
                    } else {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cerr << "[" << idx << "/" << cfg.n << "] done G=" << static_cast<double>(G)
                                  << " points=" << (cfg.output_symmetry == "half" ? (half.half_points ? (2*half.half_points-2) : (half.points.size() >= 2 ? 2*half.points.size()-2 : half.points.size())) : full.size())
                                  << " maxRelG=" << static_cast<double>(quality.max_rel_G_error)
                                  << " rmsRelG=" << static_cast<double>(quality.rms_rel_G_error)
                                  << " seconds=" << static_cast<double>(seconds)
                                  << " done=" << d << "/" << cfg.n << "\n";
                    }
                } catch (const std::exception &e) {
                    auto t1 = std::chrono::steady_clock::now();
                    real seconds = static_cast<real>(std::chrono::duration<double>(t1 - t0).count());
                    if (cfg.write_failure_meta) write_failure_json(failure_path, idx, G, e.what(), seconds, cfg);
                    ++failed;
                    int d = ++done;
                    if (cfg.progress_style == "bars") {
                        clear_progress_slot(tid);
                    } else {
                        std::lock_guard<std::mutex> lock(g_print_mutex);
                        std::cerr << "[" << idx << "/" << cfg.n << "] FAILED G=" << static_cast<double>(G)
                                  << ": " << e.what() << " done=" << d << "/" << cfg.n << "\n";
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
        for (auto &th : threads) th.join();
        g_monitor_stop.store(true);
        if (monitor.joinable()) monitor.join();
        if (use_alternate_screen) leave_alternate_screen();

        std::cerr << "Finished. failed=" << failed.load() << "\n";
        if (failed.load() != 0 && cfg.return_nonzero_on_failure) return 2;
        return 0;
    } catch (const std::exception &e) {
        leave_alternate_screen();
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
