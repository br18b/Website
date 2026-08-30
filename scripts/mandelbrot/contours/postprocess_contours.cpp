#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/repo_config.hpp"

namespace fs = std::filesystem;

static constexpr double PI = 3.141592653589793238462643383279502884;

struct Point {
    double x{};
    double y{};
};

struct Config {
    std::string output_dir = "$data_root/G_contours";
    std::string code_root = "auto";
    std::string project_root = "auto";
    std::string output_format = "auto"; // auto | bin | json
    double g_start = 0.25;
    double g_stop = 1e-6;
    int n = 100;

    // C++ postprocessor settings. These are read from the same common config file.
    bool postprocess_save_hist_data = true;
    int postprocess_hist_bins = 100;          // bins written to hist_data/*.csv
    int postprocess_quantile_bins = 8192;     // internal high-resolution bins for approximate quantiles
    std::string postprocess_hist_mode = "equal_weight"; // equal_weight | equal_width

    // Histogram x-axis/binning scales. The older keys
    // postprocess_abs_hist_scale and postprocess_signed_hist_scale are still
    // accepted as aliases for these x-scale settings.
    std::string postprocess_abs_hist_x_scale = "log";       // linear | log | symlog
    std::string postprocess_signed_hist_x_scale = "symlog"; // linear | log | symlog

    // Plotting y-axis scale for histogram density. This is written to hist_config.json
    // and used by the Python plotter when plotting hist_data/*.csv.
    std::string postprocess_abs_hist_y_scale = "log";       // linear | log | symlog
    std::string postprocess_signed_hist_y_scale = "log";    // linear | log | symlog

    // symlog(x) = sign(x) * log(1 + |x| / threshold)
    // Separate thresholds for abs/signed and x/y. The old key
    // postprocess_signed_symlog_threshold is accepted as an alias for
    // postprocess_signed_hist_x_symlog_threshold.
    double postprocess_abs_hist_x_symlog_threshold = 20.0;
    double postprocess_signed_hist_x_symlog_threshold = 20.0;
    double postprocess_abs_hist_y_symlog_threshold = 20.0;
    double postprocess_signed_hist_y_symlog_threshold = 20.0;

    // Histogram plotting style used by the Python plotter.
    std::string hist_plot_style = "bin_midpoints"; // bins | bin_midpoints
    bool hist_plot_joined = true;
    bool hist_plot_show_points = true;

    // Optional plotting ranges copied into hist_config.json for the Python plotter.
    // Values are strings so "auto" can be preserved.
    std::string postprocess_abs_hist_plot_x_min = "auto";
    std::string postprocess_abs_hist_plot_x_max = "auto";
    std::string postprocess_abs_hist_plot_y_min = "auto";
    std::string postprocess_abs_hist_plot_y_max = "auto";
    std::string postprocess_signed_hist_plot_x_min = "auto";
    std::string postprocess_signed_hist_plot_x_max = "auto";
    std::string postprocess_signed_hist_plot_y_min = "auto";
    std::string postprocess_signed_hist_plot_y_max = "auto";

    double postprocess_hist_qlo = 0.005;
    double postprocess_hist_qhi = 0.995;
};

struct Meta {
    int rollbacks = 0;
    std::string symmetry = "full";
    std::uint64_t stored_points = 0;
    std::uint64_t half_points = 0;
    double seconds = std::numeric_limits<double>::quiet_NaN();
    double max_rel_G_error = std::numeric_limits<double>::quiet_NaN();
    double rms_rel_G_error = std::numeric_limits<double>::quiet_NaN();
    int quality_eval_stride = 0;
    std::uint64_t quality_samples = 0;
};

struct Stats {
    int index = -1;
    double G = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t points = 0;
    double length = 0.0;
    double area_signed = 0.0;
    double area = 0.0;
    double mean_abs_curvature = std::numeric_limits<double>::quiet_NaN();
    double trimmed_mean_abs_curvature_p99 = std::numeric_limits<double>::quiet_NaN();
    double mean_signed_curvature = std::numeric_limits<double>::quiet_NaN();
    double rms_curvature = std::numeric_limits<double>::quiet_NaN();
    double max_abs_curvature = std::numeric_limits<double>::quiet_NaN();
    double abs_curvature_p50 = std::numeric_limits<double>::quiet_NaN();
    double abs_curvature_p90 = std::numeric_limits<double>::quiet_NaN();
    double abs_curvature_p95 = std::numeric_limits<double>::quiet_NaN();
    double abs_curvature_p99 = std::numeric_limits<double>::quiet_NaN();
    double abs_curvature_p999 = std::numeric_limits<double>::quiet_NaN();
    double total_abs_curvature = std::numeric_limits<double>::quiet_NaN();
    double signed_curvature_integral = std::numeric_limits<double>::quiet_NaN();
    double signed_curvature_expected_integral = std::numeric_limits<double>::quiet_NaN();
    double signed_curvature_turning_error = std::numeric_limits<double>::quiet_NaN();
    double geometric_turning_integral = std::numeric_limits<double>::quiet_NaN();
    double geometric_turning_expected_integral = std::numeric_limits<double>::quiet_NaN();
    double geometric_turning_error = std::numeric_limits<double>::quiet_NaN();
    Meta meta;
};

static inline std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::vector<double> make_levels(const Config &cfg) {
    std::vector<double> levels;
    levels.reserve(static_cast<size_t>(cfg.n));
    if (cfg.n == 1) {
        levels.push_back(cfg.g_start);
        return levels;
    }
    double a = std::log(cfg.g_start);
    double b = std::log(cfg.g_stop);
    for (int i = 0; i < cfg.n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(cfg.n - 1);
        levels.push_back(std::exp(a + t * (b - a)));
    }
    return levels;
}

static std::string stem_for_index(int idx) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << idx;
    return oss.str();
}

template <typename T>
static T read_scalar_binary(std::ifstream &f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("binary read failed");
    return v;
}

template <typename T>
static void read_points_payload(std::ifstream &f, std::vector<Point> &pts) {
    for (auto &p : pts) {
        T x = read_scalar_binary<T>(f);
        T y = read_scalar_binary<T>(f);
        p.x = static_cast<double>(x);
        p.y = static_cast<double>(y);
    }
}

static std::vector<Point> read_points_bin(const fs::path &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("could not open " + path.string());

    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected, 8) != 0) {
        throw std::runtime_error("bad binary contour magic in " + path.string());
    }

    std::uint32_t version = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t code = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t value_size = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t endian = read_scalar_binary<std::uint32_t>(f);
    std::uint64_t count = read_scalar_binary<std::uint64_t>(f);

    if (version != 1) throw std::runtime_error("unsupported binary contour version");
    if (endian != 0x01020304u) throw std::runtime_error("unsupported endian marker");
    if (count > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max() / sizeof(Point))) {
        throw std::runtime_error("contour too large for this build");
    }

    std::vector<Point> pts(static_cast<size_t>(count));
    if (code == 1 && value_size == 4) {
        read_points_payload<float>(f, pts);
    } else if (code == 2 && value_size == 8) {
        read_points_payload<double>(f, pts);
    } else if (code == 3 && value_size == sizeof(long double)) {
        read_points_payload<long double>(f, pts);
    } else {
        std::ostringstream oss;
        oss << "unsupported binary value type code=" << code << " size=" << value_size;
        throw std::runtime_error(oss.str());
    }
    return pts;
}

struct BinaryContourInfo {
    fs::path path;
    std::uint32_t code = 0;
    std::uint32_t value_size = 0;
    std::uint64_t count = 0;
    std::streampos payload_offset{};
};

static BinaryContourInfo read_binary_contour_info(const fs::path &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("could not open " + path.string());

    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'M','B','C','T','B','I','N','1'};
    if (std::memcmp(magic, expected, 8) != 0) {
        throw std::runtime_error("bad binary contour magic in " + path.string());
    }

    std::uint32_t version = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t code = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t value_size = read_scalar_binary<std::uint32_t>(f);
    std::uint32_t endian = read_scalar_binary<std::uint32_t>(f);
    std::uint64_t count = read_scalar_binary<std::uint64_t>(f);

    if (version != 1) throw std::runtime_error("unsupported binary contour version");
    if (endian != 0x01020304u) throw std::runtime_error("unsupported endian marker");
    if (!((code == 1 && value_size == 4) || (code == 2 && value_size == 8) || (code == 3 && value_size == sizeof(long double)))) {
        std::ostringstream oss;
        oss << "unsupported binary value type code=" << code << " size=" << value_size;
        throw std::runtime_error(oss.str());
    }

    BinaryContourInfo info;
    info.path = path;
    info.code = code;
    info.value_size = value_size;
    info.count = count;
    info.payload_offset = f.tellg();
    return info;
}

static std::ifstream open_binary_payload(const BinaryContourInfo &info) {
    std::ifstream f(info.path, std::ios::binary);
    if (!f) throw std::runtime_error("could not open " + info.path.string());
    f.seekg(info.payload_offset);
    if (!f) throw std::runtime_error("could not seek to binary payload in " + info.path.string());
    return f;
}

static Point read_point_binary(std::ifstream &f, const BinaryContourInfo &info) {
    if (info.code == 1 && info.value_size == 4) {
        float x = read_scalar_binary<float>(f);
        float y = read_scalar_binary<float>(f);
        return Point{static_cast<double>(x), static_cast<double>(y)};
    }
    if (info.code == 2 && info.value_size == 8) {
        double x = read_scalar_binary<double>(f);
        double y = read_scalar_binary<double>(f);
        return Point{x, y};
    }
    if (info.code == 3 && info.value_size == sizeof(long double)) {
        long double x = read_scalar_binary<long double>(f);
        long double y = read_scalar_binary<long double>(f);
        return Point{static_cast<double>(x), static_cast<double>(y)};
    }
    throw std::runtime_error("unsupported binary value type while reading payload");
}

static std::optional<std::pair<fs::path, std::string>> find_contour_file_auto(const fs::path &out, int idx, const std::string &preferred) {
    fs::path bin = out / "contours_bin" / (stem_for_index(idx) + ".bin");
    fs::path js = out / "contours" / (stem_for_index(idx) + ".json");

    auto try_bin = [&]() -> std::optional<std::pair<fs::path, std::string>> {
        if (!fs::exists(bin)) return std::nullopt;
        return std::make_pair(bin, std::string("bin"));
    };
    auto try_json = [&]() -> std::optional<std::pair<fs::path, std::string>> {
        if (!fs::exists(js)) return std::nullopt;
        return std::make_pair(js, std::string("json"));
    };

    if (preferred == "bin") {
        if (auto r = try_bin()) return r;
        return try_json();
    }
    if (preferred == "json") {
        if (auto r = try_json()) return r;
        return try_bin();
    }
    if (auto r = try_bin()) return r;
    return try_json();
}

// Lightweight JSON reader for old/small contour JSONs. For serious low-G contours,
// binary is strongly recommended.
static std::vector<Point> read_points_json(const fs::path &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open " + path.string());
    std::vector<Point> pts;
    std::string token;
    std::vector<double> nums;
    char c;
    while (f.get(c)) {
        if (std::isdigit(static_cast<unsigned char>(c)) || c=='-' || c=='+' || c=='.' || c=='e' || c=='E') {
            token.push_back(c);
        } else if (!token.empty()) {
            nums.push_back(std::stod(token));
            token.clear();
            if (nums.size() == 2) {
                pts.push_back(Point{nums[0], nums[1]});
                nums.clear();
            }
        }
    }
    if (!token.empty()) nums.push_back(std::stod(token));
    if (nums.size() == 2) pts.push_back(Point{nums[0], nums[1]});
    return pts;
}

[[maybe_unused]] static std::optional<std::pair<std::vector<Point>, std::string>> read_contour_auto(const fs::path &out, int idx, const std::string &preferred) {
    fs::path bin = out / "contours_bin" / (stem_for_index(idx) + ".bin");
    fs::path js = out / "contours" / (stem_for_index(idx) + ".json");

    auto try_bin = [&]() -> std::optional<std::pair<std::vector<Point>, std::string>> {
        if (!fs::exists(bin)) return std::nullopt;
        return std::make_pair(read_points_bin(bin), std::string("bin"));
    };
    auto try_json = [&]() -> std::optional<std::pair<std::vector<Point>, std::string>> {
        if (!fs::exists(js)) return std::nullopt;
        return std::make_pair(read_points_json(js), std::string("json"));
    };

    if (preferred == "bin") {
        if (auto r = try_bin()) return r;
        return try_json();
    }
    if (preferred == "json") {
        if (auto r = try_json()) return r;
        return try_bin();
    }
    if (auto r = try_bin()) return r;
    return try_json();
}

static double dist(Point a, Point b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    return std::sqrt(dx*dx + dy*dy);
}

static void drop_repeated_closure(std::vector<Point> &pts) {
    if (pts.size() < 2) return;
    if (dist(pts.front(), pts.back()) <= 1e-13) pts.pop_back();
}

static double signed_area(const std::vector<Point> &pts) {
    if (pts.size() < 3) return 0.0;
    long double s = 0.0L;
    const size_t n = pts.size();
    for (size_t i = 0; i < n; ++i) {
        const auto &a = pts[i];
        const auto &b = pts[(i + 1) % n];
        s += static_cast<long double>(a.x) * b.y - static_cast<long double>(a.y) * b.x;
    }
    return static_cast<double>(0.5L * s);
}

static double closed_length(const std::vector<Point> &pts) {
    if (pts.size() < 2) return 0.0;
    long double s = 0.0L;
    const size_t n = pts.size();
    for (size_t i = 0; i < n; ++i) s += dist(pts[i], pts[(i + 1) % n]);
    return static_cast<double>(s);
}


static std::vector<Point> mirror_full_from_half(const std::vector<Point> &half) {
    std::vector<Point> full;
    if (half.empty()) return full;
    full.reserve(half.size() >= 2 ? half.size() * 2 - 2 : half.size());
    for (const auto &p : half) full.push_back(p);
    if (half.size() >= 2) {
        for (size_t k = half.size() - 2; k > 0; --k) full.push_back(Point{half[k].x, -half[k].y});
    }
    return full;
}

struct Hist {
    bool log = false;
    double lo = 0.0;
    double hi = 1.0;
    double log_lo = 0.0;
    double log_hi = 1.0;
    std::vector<long double> w;

    Hist() = default;
    Hist(int bins, double lo_, double hi_, bool log_) : log(log_), lo(lo_), hi(hi_), w(static_cast<size_t>(bins), 0.0L) {
        if (log) {
            log_lo = std::log(lo);
            log_hi = std::log(hi);
        }
    }

    void add(double x, double weight) {
        if (!(weight > 0) || !std::isfinite(x)) return;
        int bins = static_cast<int>(w.size());
        if (bins <= 0) return;
        int idx = 0;
        if (log) {
            if (!(x > 0)) return;
            if (hi <= lo) return;
            double t = (std::log(x) - log_lo) / (log_hi - log_lo);
            idx = static_cast<int>(std::floor(t * bins));
        } else {
            if (hi <= lo) return;
            double t = (x - lo) / (hi - lo);
            idx = static_cast<int>(std::floor(t * bins));
        }
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        w[static_cast<size_t>(idx)] += weight;
    }

    double edge(int i) const {
        int bins = static_cast<int>(w.size());
        double t = static_cast<double>(i) / static_cast<double>(bins);
        if (log) return std::exp(log_lo + t * (log_hi - log_lo));
        return lo + t * (hi - lo);
    }

    double center(int i) const {
        return 0.5 * (edge(i) + edge(i + 1));
    }

    long double total_weight() const {
        long double s = 0.0L;
        for (auto x : w) s += x;
        return s;
    }

    double quantile(double q) const {
        long double total = total_weight();
        if (!(total > 0)) return std::numeric_limits<double>::quiet_NaN();
        long double target = static_cast<long double>(q) * total;
        long double cum = 0.0L;
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            long double next = cum + w[static_cast<size_t>(i)];
            if (target <= next) {
                long double inside = (w[static_cast<size_t>(i)] > 0) ? (target - cum) / w[static_cast<size_t>(i)] : 0.0L;
                double a = edge(i);
                double b = edge(i + 1);
                if (log) {
                    double la = std::log(a);
                    double lb = std::log(b);
                    return std::exp(la + static_cast<double>(inside) * (lb - la));
                }
                return a + static_cast<double>(inside) * (b - a);
            }
            cum = next;
        }
        return edge(static_cast<int>(w.size()));
    }
};

struct EdgeHist {
    std::vector<double> edges;
    std::vector<long double> w;

    EdgeHist() = default;

    explicit EdgeHist(std::vector<double> e) : edges(std::move(e)) {
        std::vector<double> clean;
        clean.reserve(edges.size());
        for (double x : edges) {
            if (!std::isfinite(x)) continue;
            if (clean.empty() || x > clean.back()) clean.push_back(x);
        }
        edges = std::move(clean);
        if (edges.size() >= 2) w.assign(edges.size() - 1, 0.0L);
    }

    bool valid() const {
        return edges.size() >= 2 && w.size() + 1 == edges.size();
    }

    void add(double x, double weight) {
        if (!valid() || !(weight > 0.0) || !std::isfinite(x)) return;
        if (x < edges.front() || x > edges.back()) return;
        auto it = std::upper_bound(edges.begin(), edges.end(), x);
        int idx = static_cast<int>(std::distance(edges.begin(), it)) - 1;
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(w.size())) idx = static_cast<int>(w.size()) - 1;
        w[static_cast<size_t>(idx)] += weight;
    }

    long double total_weight() const {
        long double s = 0.0L;
        for (auto x : w) s += x;
        return s;
    }
};

static std::vector<double> make_scaled_edges(double lo, double hi, int bins, bool log_scale) {
    std::vector<double> edges;
    if (!(hi > lo) || bins <= 0) return edges;
    edges.reserve(static_cast<size_t>(bins) + 1);

    if (log_scale && lo > 0.0 && hi > 0.0) {
        double a = std::log(lo);
        double b = std::log(hi);
        for (int i = 0; i <= bins; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(bins);
            edges.push_back(std::exp(a + t * (b - a)));
        }
    } else {
        for (int i = 0; i <= bins; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(bins);
            edges.push_back(lo + t * (hi - lo));
        }
    }

    return edges;
}

static double symlog_transform(double x, double threshold) {
    if (x == 0.0) return 0.0;
    return std::copysign(std::log1p(std::abs(x) / threshold), x);
}

static double symlog_inverse(double y, double threshold) {
    if (y == 0.0) return 0.0;
    return std::copysign(threshold * std::expm1(std::abs(y)), y);
}

static std::vector<double> make_symlog_edges(double lo, double hi, int bins, double threshold) {
    std::vector<double> edges;
    if (!(hi > lo) || bins <= 0 || !(threshold > 0.0)) return edges;
    edges.reserve(static_cast<size_t>(bins) + 1);

    double a = symlog_transform(lo, threshold);
    double b = symlog_transform(hi, threshold);

    for (int i = 0; i <= bins; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(bins);
        double y = a + t * (b - a);
        edges.push_back(symlog_inverse(y, threshold));
    }

    // Ensure exact endpoints survive roundoff.
    edges.front() = lo;
    edges.back() = hi;
    return edges;
}

static std::vector<double> make_quantile_edges(const Hist &hist, double qlo, double qhi, int bins) {
    std::vector<double> edges;
    if (bins <= 0) return edges;
    edges.reserve(static_cast<size_t>(bins) + 1);
    for (int i = 0; i <= bins; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(bins);
        double q = qlo + t * (qhi - qlo);
        edges.push_back(hist.quantile(q));
    }
    return edges;
}

static void write_edge_hist_csv(const EdgeHist &hist, const fs::path &path) {
    if (!hist.valid()) return;
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "left,right,weight,density\n";
    long double total = hist.total_weight();
    for (int i = 0; i < static_cast<int>(hist.w.size()); ++i) {
        double left = hist.edges[static_cast<size_t>(i)];
        double right = hist.edges[static_cast<size_t>(i + 1)];
        double width = right - left;
        double weight = static_cast<double>(hist.w[static_cast<size_t>(i)]);
        double density = (total > 0 && width > 0) ? static_cast<double>(hist.w[static_cast<size_t>(i)] / total / width) : 0.0;
        f << left << "," << right << "," << weight << "," << density << "\n";
    }
}

static void write_hist_config_json(const fs::path &out, const Config &cfg) {
    if (!cfg.postprocess_save_hist_data) return;
    fs::path hdir = out / "hist_data";
    fs::create_directories(hdir);

    std::ofstream f(hdir / "hist_config.json");
    f << std::setprecision(17);
    f << "{\n";
    f << "  \"postprocess_hist_bins\": " << cfg.postprocess_hist_bins << ",\n";
    f << "  \"postprocess_quantile_bins\": " << cfg.postprocess_quantile_bins << ",\n";
    f << "  \"postprocess_hist_mode\": \"" << cfg.postprocess_hist_mode << "\",\n";
    f << "  \"postprocess_hist_qlo\": " << cfg.postprocess_hist_qlo << ",\n";
    f << "  \"postprocess_hist_qhi\": " << cfg.postprocess_hist_qhi << ",\n";
    f << "  \"postprocess_abs_hist_x_scale\": \"" << cfg.postprocess_abs_hist_x_scale << "\",\n";
    f << "  \"postprocess_signed_hist_x_scale\": \"" << cfg.postprocess_signed_hist_x_scale << "\",\n";
    f << "  \"postprocess_abs_hist_y_scale\": \"" << cfg.postprocess_abs_hist_y_scale << "\",\n";
    f << "  \"postprocess_signed_hist_y_scale\": \"" << cfg.postprocess_signed_hist_y_scale << "\",\n";
    f << "  \"postprocess_signed_symlog_threshold\": " << cfg.postprocess_signed_hist_x_symlog_threshold << ",\n";
    f << "  \"postprocess_abs_hist_plot_x_min\": \"" << cfg.postprocess_abs_hist_plot_x_min << "\",\n";
    f << "  \"postprocess_abs_hist_plot_x_max\": \"" << cfg.postprocess_abs_hist_plot_x_max << "\",\n";
    f << "  \"postprocess_abs_hist_plot_y_min\": \"" << cfg.postprocess_abs_hist_plot_y_min << "\",\n";
    f << "  \"postprocess_abs_hist_plot_y_max\": \"" << cfg.postprocess_abs_hist_plot_y_max << "\",\n";
    f << "  \"postprocess_signed_hist_plot_x_min\": \"" << cfg.postprocess_signed_hist_plot_x_min << "\",\n";
    f << "  \"postprocess_signed_hist_plot_x_max\": \"" << cfg.postprocess_signed_hist_plot_x_max << "\",\n";
    f << "  \"postprocess_signed_hist_plot_y_min\": \"" << cfg.postprocess_signed_hist_plot_y_min << "\",\n";
    f << "  \"postprocess_signed_hist_plot_y_max\": \"" << cfg.postprocess_signed_hist_plot_y_max << "\"\n";
    f << "}\n";
}

static void curvature_from_triplet(const Point &prev, const Point &cur, const Point &next, double &k_out, double &weight, double &turn) {
    double ux = cur.x - prev.x;
    double uy = cur.y - prev.y;
    double vx = next.x - cur.x;
    double vy = next.y - cur.y;
    double lu = std::sqrt(ux*ux + uy*uy);
    double lv = std::sqrt(vx*vx + vy*vy);
    weight = 0.5 * (lu + lv);

    double cross = ux * vy - uy * vx;
    double dot = ux * vx + uy * vy;
    turn = std::atan2(cross, dot);
    k_out = (weight > 0) ? -turn / weight : std::numeric_limits<double>::quiet_NaN();
}

static void curvature_at(const std::vector<Point> &pts, size_t i, double &k_out, double &weight, double &turn) {
    const size_t n = pts.size();
    curvature_from_triplet(pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n], k_out, weight, turn);
}

static Point mirror_y(Point p) {
    return Point{p.x, -p.y};
}

static double cross2(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

static Meta read_meta(const fs::path &out, int idx) {
    Meta m;
    fs::path p = out / "meta" / (stem_for_index(idx) + ".json");
    if (!fs::exists(p)) return m;
    std::ifstream f(p);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto num = [&](const std::string &key) -> std::optional<double> {
        std::string pat = "\"" + key + "\"";
        size_t k = s.find(pat);
        if (k == std::string::npos) return std::nullopt;
        size_t colon = s.find(':', k + pat.size());
        if (colon == std::string::npos) return std::nullopt;
        size_t start = s.find_first_of("-+0123456789.nNiI", colon + 1);
        if (start == std::string::npos) return std::nullopt;
        size_t end = start;
        while (end < s.size() && (std::isalnum(static_cast<unsigned char>(s[end])) || s[end]=='-' || s[end]=='+' || s[end]=='.')) ++end;
        std::string t = s.substr(start, end - start);
        try {
            return std::stod(t);
        } catch (...) {
            return std::nullopt;
        }
    };

    auto str = [&](const std::string &key) -> std::optional<std::string> {
        std::string pat = "\"" + key + "\"";
        size_t k = s.find(pat);
        if (k == std::string::npos) return std::nullopt;
        size_t colon = s.find(':', k + pat.size());
        if (colon == std::string::npos) return std::nullopt;
        size_t q1 = s.find('"', colon + 1);
        if (q1 == std::string::npos) return std::nullopt;
        size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos) return std::nullopt;
        return s.substr(q1 + 1, q2 - q1 - 1);
    };

    if (auto v = str("symmetry")) m.symmetry = *v;
    if (auto v = num("stored_points")) m.stored_points = static_cast<std::uint64_t>(*v);
    if (auto v = num("half_points")) m.half_points = static_cast<std::uint64_t>(*v);
    if (auto v = num("rollbacks")) m.rollbacks = static_cast<int>(*v);
    if (auto v = num("seconds")) m.seconds = *v;
    if (auto v = num("max_rel_G_error")) m.max_rel_G_error = *v;
    if (auto v = num("rms_rel_G_error")) m.rms_rel_G_error = *v;
    if (auto v = num("quality_eval_stride")) m.quality_eval_stride = static_cast<int>(*v);
    if (auto v = num("quality_samples")) m.quality_samples = static_cast<std::uint64_t>(*v);
    return m;
}

struct GeometryScan {
    std::uint64_t stored_points = 0;
    std::uint64_t logical_points = 0;
    double length = 0.0;
    double area_signed = 0.0;
};

static GeometryScan scan_geometry_binary(const BinaryContourInfo &info, bool half_symmetry) {
    GeometryScan g;
    g.stored_points = info.count;
    if (info.count == 0) return g;

    auto f = open_binary_payload(info);
    Point first = read_point_binary(f, info);
    if (info.count == 1) {
        g.logical_points = 1;
        return g;
    }

    Point second = read_point_binary(f, info);
    Point prev = second;
    Point before_last = first;
    Point last = second;

    long double open_len = dist(first, second);
    long double open_cross = cross2(first, second);
    const long double first_edge_len = dist(first, second);
    const long double first_edge_cross = cross2(first, second);
    long double last_edge_len = first_edge_len;
    long double last_edge_cross = first_edge_cross;

    for (std::uint64_t i = 2; i < info.count; ++i) {
        Point p = read_point_binary(f, info);
        before_last = last;
        prev = last;
        last = p;
        const double elen = dist(prev, p);
        const double ecross = cross2(prev, p);
        open_len += elen;
        open_cross += ecross;
        last_edge_len = elen;
        last_edge_cross = ecross;
    }

    if (!half_symmetry) {
        g.logical_points = info.count;
        g.length = static_cast<double>(open_len + dist(last, first));
        g.area_signed = static_cast<double>(0.5L * (open_cross + cross2(last, first)));
        return g;
    }

    g.logical_points = (info.count >= 2) ? (2 * info.count - 2) : info.count;
    if (info.count == 2) {
        g.length = 2.0 * dist(first, second);
        g.area_signed = 0.0;
        return g;
    }

    const Point m_before_last = mirror_y(before_last);
    const Point m_second = mirror_y(second);
    const long double lower_left_len = dist(last, m_before_last);
    const long double lower_right_len = dist(m_second, first);
    const long double lower_mid_len = open_len - first_edge_len - last_edge_len;

    const long double lower_left_cross = cross2(last, m_before_last);
    const long double lower_right_cross = cross2(m_second, first);
    const long double lower_mid_cross = open_cross - first_edge_cross - last_edge_cross;

    g.length = static_cast<double>(open_len + lower_left_len + lower_mid_len + lower_right_len);
    g.area_signed = static_cast<double>(0.5L * (open_cross + lower_left_cross + lower_mid_cross + lower_right_cross));
    return g;
}

template <typename Callback>
static void scan_curvatures_binary(const BinaryContourInfo &info, bool half_symmetry, Callback cb) {
    if (info.count < 3) return;
    auto emit = [&](const Point &prev, const Point &cur, const Point &next, double multiplicity) {
        double k, w, turn;
        curvature_from_triplet(prev, cur, next, k, w, turn);
        if (!std::isfinite(k) || !(w > 0.0) || !(multiplicity > 0.0)) return;
        cb(k, w * multiplicity, turn * multiplicity);
    };

    auto f = open_binary_payload(info);
    Point first = read_point_binary(f, info);
    Point second = read_point_binary(f, info);

    if (half_symmetry) {
        // The full contour is obtained by reflecting the upper half across y=0.
        // Curvature samples are symmetric, so process the stored half only:
        // interior samples get weight 2, the two real-axis seams get weight 1.
        emit(mirror_y(second), first, second, 1.0);
        Point prev = first;
        Point cur = second;
        for (std::uint64_t i = 2; i < info.count; ++i) {
            Point next = read_point_binary(f, info);
            emit(prev, cur, next, 2.0);
            prev = cur;
            cur = next;
        }
        emit(prev, cur, mirror_y(prev), 1.0);
        return;
    }

    Point prev = first;
    Point cur = second;
    for (std::uint64_t i = 2; i < info.count; ++i) {
        Point next = read_point_binary(f, info);
        emit(prev, cur, next, 1.0);
        prev = cur;
        cur = next;
    }
    Point last = cur;
    Point before_last = prev;
    emit(before_last, last, first, 1.0);
    emit(last, first, second, 1.0);
}

static Stats compute_stats_binary_streaming(const fs::path &path, int idx, double G, const Config &cfg, const fs::path &out) {
    const Meta meta_from_file = read_meta(out, idx);
    const bool half_symmetry = (meta_from_file.symmetry == "upper_half_mirror_y" || meta_from_file.symmetry == "half");
    const BinaryContourInfo info = read_binary_contour_info(path);
    const GeometryScan geom = scan_geometry_binary(info, half_symmetry);

    Stats st;
    st.index = idx;
    st.G = G;
    st.points = geom.logical_points;
    st.length = geom.length;
    st.area_signed = geom.area_signed;
    st.area = std::abs(st.area_signed);
    st.meta = meta_from_file;
    if (st.meta.stored_points == 0) st.meta.stored_points = geom.stored_points;
    if (half_symmetry && st.meta.half_points == 0) st.meta.half_points = geom.stored_points;
    if (!half_symmetry && st.meta.half_points == 0) st.meta.half_points = geom.stored_points;
    if (st.meta.symmetry.empty()) st.meta.symmetry = half_symmetry ? "upper_half_mirror_y" : "full";

    if (st.points < 3) return st;

    long double wsum = 0.0L, total_abs = 0.0L, signed_int = 0.0L, rms_sum = 0.0L, geom_turn = 0.0L;
    double max_abs = 0.0;
    double min_pos_abs = std::numeric_limits<double>::infinity();
    double min_k = std::numeric_limits<double>::infinity();
    double max_k = -std::numeric_limits<double>::infinity();

    scan_curvatures_binary(info, half_symmetry, [&](double k, double w, double turn) {
        double ak = std::abs(k);
        wsum += w;
        total_abs += static_cast<long double>(w) * ak;
        signed_int += static_cast<long double>(w) * k;
        rms_sum += static_cast<long double>(w) * k * k;
        geom_turn += turn;
        max_abs = std::max(max_abs, ak);
        if (ak > 0) min_pos_abs = std::min(min_pos_abs, ak);
        min_k = std::min(min_k, k);
        max_k = std::max(max_k, k);
    });

    st.geometric_turning_integral = static_cast<double>(geom_turn);
    st.geometric_turning_expected_integral = (st.area_signed == 0.0) ? std::numeric_limits<double>::quiet_NaN() : std::copysign(2.0 * PI, st.area_signed);
    st.geometric_turning_error = st.geometric_turning_integral - st.geometric_turning_expected_integral;

    st.signed_curvature_integral = static_cast<double>(signed_int);
    st.signed_curvature_expected_integral = (st.area_signed == 0.0) ? std::numeric_limits<double>::quiet_NaN() : -std::copysign(2.0 * PI, st.area_signed);
    st.signed_curvature_turning_error = st.signed_curvature_integral - st.signed_curvature_expected_integral;

    if (!(wsum > 0)) return st;

    st.mean_abs_curvature = static_cast<double>(total_abs / wsum);
    st.mean_signed_curvature = static_cast<double>(signed_int / wsum);
    st.rms_curvature = std::sqrt(static_cast<double>(rms_sum / wsum));
    st.max_abs_curvature = max_abs;
    st.total_abs_curvature = static_cast<double>(total_abs);

    if (!std::isfinite(min_pos_abs) || !(max_abs > 0)) {
        st.abs_curvature_p50 = st.abs_curvature_p90 = st.abs_curvature_p95 = st.abs_curvature_p99 = st.abs_curvature_p999 = 0.0;
        st.trimmed_mean_abs_curvature_p99 = 0.0;
        return st;
    }

    if (max_abs <= min_pos_abs) max_abs = min_pos_abs * 1.000001;
    if (!(max_k > min_k)) max_k = min_k + 1e-300;

    // Internal high-resolution histograms used for approximate weighted quantiles.
    // This is the equal-weight compromise: stream into many fixed bins, then merge
    // those bins into the smaller saved histogram. Increase postprocess_quantile_bins
    // for a closer approximation without storing all curvature samples.
    Hist abs_internal(cfg.postprocess_quantile_bins, min_pos_abs, max_abs, true);
    Hist signed_internal(cfg.postprocess_quantile_bins, min_k, max_k, false);

    scan_curvatures_binary(info, half_symmetry, [&](double k, double w, double) {
        abs_internal.add(std::abs(k), w);
        signed_internal.add(k, w);
    });

    st.abs_curvature_p50 = abs_internal.quantile(0.5);
    st.abs_curvature_p90 = abs_internal.quantile(0.9);
    st.abs_curvature_p95 = abs_internal.quantile(0.95);
    st.abs_curvature_p99 = abs_internal.quantile(0.99);
    st.abs_curvature_p999 = abs_internal.quantile(0.999);

    fs::path hdir = out / "hist_data";
    const double qlo = cfg.postprocess_hist_qlo;
    const double qhi = cfg.postprocess_hist_qhi;
    const int bins = cfg.postprocess_hist_bins;

    double abs_lo = abs_internal.quantile(qlo);
    double abs_hi = abs_internal.quantile(qhi);
    double signed_lo = signed_internal.quantile(qlo);
    double signed_hi = signed_internal.quantile(qhi);

    std::vector<double> abs_edges;
    std::vector<double> signed_edges;
    bool can_write_hists = false;

    if (cfg.postprocess_save_hist_data &&
        std::isfinite(abs_lo) && std::isfinite(abs_hi) && abs_hi > abs_lo &&
        std::isfinite(signed_lo) && std::isfinite(signed_hi) && signed_hi > signed_lo) {
        fs::create_directories(hdir);
        if (cfg.postprocess_hist_mode == "equal_weight") {
            abs_edges = make_quantile_edges(abs_internal, qlo, qhi, bins);
            signed_edges = make_quantile_edges(signed_internal, qlo, qhi, bins);
        } else {
            if (cfg.postprocess_abs_hist_x_scale == "symlog") {
                abs_edges = make_symlog_edges(abs_lo, abs_hi, bins, cfg.postprocess_abs_hist_x_symlog_threshold);
            } else {
                bool abs_log = (cfg.postprocess_abs_hist_x_scale == "log" && abs_lo > 0.0 && abs_hi > abs_lo);
                abs_edges = make_scaled_edges(abs_lo, abs_hi, bins, abs_log);
            }
            if (cfg.postprocess_signed_hist_x_scale == "symlog") {
                signed_edges = make_symlog_edges(signed_lo, signed_hi, bins, cfg.postprocess_signed_hist_x_symlog_threshold);
            } else {
                bool signed_log = (cfg.postprocess_signed_hist_x_scale == "log" && signed_lo > 0.0 && signed_hi > signed_lo);
                signed_edges = make_scaled_edges(signed_lo, signed_hi, bins, signed_log);
            }
        }
        can_write_hists = (abs_edges.size() >= 2 && signed_edges.size() >= 2);
    }

    EdgeHist abs_out(abs_edges);
    EdgeHist signed_out(signed_edges);
    long double trimmed = 0.0L;

    scan_curvatures_binary(info, half_symmetry, [&](double k, double w, double) {
        trimmed += static_cast<long double>(w) * std::min(std::abs(k), st.abs_curvature_p99);
        if (can_write_hists) {
            abs_out.add(std::abs(k), w);
            signed_out.add(k, w);
        }
    });

    st.trimmed_mean_abs_curvature_p99 = static_cast<double>(trimmed / wsum);

    if (can_write_hists) {
        write_edge_hist_csv(abs_out, hdir / (stem_for_index(idx) + "_abs.csv"));
        write_edge_hist_csv(signed_out, hdir / (stem_for_index(idx) + "_signed.csv"));
    }

    return st;
}

static Stats compute_stats_loaded(std::vector<Point> pts, int idx, double G, const Config &cfg, const fs::path &out) {
    Meta meta = read_meta(out, idx);
    if (meta.symmetry == "upper_half_mirror_y" || meta.symmetry == "half") {
        pts = mirror_full_from_half(pts);
    }
    drop_repeated_closure(pts);

    Stats st;
    st.index = idx;
    st.G = G;
    st.points = static_cast<std::uint64_t>(pts.size());
    st.length = closed_length(pts);
    st.area_signed = signed_area(pts);
    st.area = std::abs(st.area_signed);
    st.meta = meta;

    const size_t n = pts.size();
    if (n < 3) return st;

    long double wsum = 0.0L, total_abs = 0.0L, signed_int = 0.0L, rms_sum = 0.0L, geom_turn = 0.0L;
    double max_abs = 0.0;
    double min_pos_abs = std::numeric_limits<double>::infinity();
    double min_k = std::numeric_limits<double>::infinity();
    double max_k = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < n; ++i) {
        double k, w, turn;
        curvature_at(pts, i, k, w, turn);
        if (!std::isfinite(k) || !(w > 0)) continue;
        double ak = std::abs(k);
        wsum += w;
        total_abs += static_cast<long double>(w) * ak;
        signed_int += static_cast<long double>(w) * k;
        rms_sum += static_cast<long double>(w) * k * k;
        geom_turn += turn;
        max_abs = std::max(max_abs, ak);
        if (ak > 0) min_pos_abs = std::min(min_pos_abs, ak);
        min_k = std::min(min_k, k);
        max_k = std::max(max_k, k);
    }

    st.geometric_turning_integral = static_cast<double>(geom_turn);
    st.geometric_turning_expected_integral = (st.area_signed == 0.0) ? std::numeric_limits<double>::quiet_NaN() : std::copysign(2.0 * PI, st.area_signed);
    st.geometric_turning_error = st.geometric_turning_integral - st.geometric_turning_expected_integral;

    st.signed_curvature_integral = static_cast<double>(signed_int);
    st.signed_curvature_expected_integral = (st.area_signed == 0.0) ? std::numeric_limits<double>::quiet_NaN() : -std::copysign(2.0 * PI, st.area_signed);
    st.signed_curvature_turning_error = st.signed_curvature_integral - st.signed_curvature_expected_integral;

    if (!(wsum > 0)) return st;

    st.mean_abs_curvature = static_cast<double>(total_abs / wsum);
    st.mean_signed_curvature = static_cast<double>(signed_int / wsum);
    st.rms_curvature = std::sqrt(static_cast<double>(rms_sum / wsum));
    st.max_abs_curvature = max_abs;
    st.total_abs_curvature = static_cast<double>(total_abs);

    if (!std::isfinite(min_pos_abs) || !(max_abs > 0)) {
        st.abs_curvature_p50 = st.abs_curvature_p90 = st.abs_curvature_p95 = st.abs_curvature_p99 = st.abs_curvature_p999 = 0.0;
        st.trimmed_mean_abs_curvature_p99 = 0.0;
        return st;
    }

    if (max_abs <= min_pos_abs) max_abs = min_pos_abs * 1.000001;
    if (!(max_k > min_k)) {
        max_k = min_k + 1e-300;
    }

    // Internal high-resolution histograms. These are used for approximate weighted
    // quantiles and also to define cutoffs for the lighter saved histogram data.
    Hist abs_internal(cfg.postprocess_quantile_bins, min_pos_abs, max_abs, true);
    Hist signed_internal(cfg.postprocess_quantile_bins, min_k, max_k, false);

    for (size_t i = 0; i < n; ++i) {
        double k, w, turn;
        curvature_at(pts, i, k, w, turn);
        if (!std::isfinite(k) || !(w > 0)) continue;
        abs_internal.add(std::abs(k), w);
        signed_internal.add(k, w);
    }

    st.abs_curvature_p50 = abs_internal.quantile(0.5);
    st.abs_curvature_p90 = abs_internal.quantile(0.9);
    st.abs_curvature_p95 = abs_internal.quantile(0.95);
    st.abs_curvature_p99 = abs_internal.quantile(0.99);
    st.abs_curvature_p999 = abs_internal.quantile(0.999);

    long double trimmed = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        double k, w, turn;
        curvature_at(pts, i, k, w, turn);
        if (!std::isfinite(k) || !(w > 0)) continue;
        trimmed += static_cast<long double>(w) * std::min(std::abs(k), st.abs_curvature_p99);
    }
    st.trimmed_mean_abs_curvature_p99 = static_cast<double>(trimmed / wsum);

    if (cfg.postprocess_save_hist_data) {
        fs::path hdir = out / "hist_data";
        fs::create_directories(hdir);

        const double qlo = cfg.postprocess_hist_qlo;
        const double qhi = cfg.postprocess_hist_qhi;
        const int bins = cfg.postprocess_hist_bins;

        double abs_lo = abs_internal.quantile(qlo);
        double abs_hi = abs_internal.quantile(qhi);
        double signed_lo = signed_internal.quantile(qlo);
        double signed_hi = signed_internal.quantile(qhi);

        if (std::isfinite(abs_lo) && std::isfinite(abs_hi) && abs_hi > abs_lo &&
            std::isfinite(signed_lo) && std::isfinite(signed_hi) && signed_hi > signed_lo) {

            std::vector<double> abs_edges;
            std::vector<double> signed_edges;

            if (cfg.postprocess_hist_mode == "equal_weight") {
                abs_edges = make_quantile_edges(abs_internal, qlo, qhi, bins);
                signed_edges = make_quantile_edges(signed_internal, qlo, qhi, bins);
            } else {
                if (cfg.postprocess_abs_hist_x_scale == "symlog") {
                    abs_edges = make_symlog_edges(abs_lo, abs_hi, bins, cfg.postprocess_abs_hist_x_symlog_threshold);
                } else {
                    bool abs_log = (cfg.postprocess_abs_hist_x_scale == "log" && abs_lo > 0.0 && abs_hi > abs_lo);
                    abs_edges = make_scaled_edges(abs_lo, abs_hi, bins, abs_log);
                }

                if (cfg.postprocess_signed_hist_x_scale == "symlog") {
                    signed_edges = make_symlog_edges(signed_lo, signed_hi, bins, cfg.postprocess_signed_hist_x_symlog_threshold);
                } else {
                    bool signed_log = (cfg.postprocess_signed_hist_x_scale == "log" && signed_lo > 0.0 && signed_hi > signed_lo);
                    signed_edges = make_scaled_edges(signed_lo, signed_hi, bins, signed_log);
                }
            }

            EdgeHist abs_out(abs_edges);
            EdgeHist signed_out(signed_edges);

            for (size_t i = 0; i < n; ++i) {
                double k, w, turn;
                curvature_at(pts, i, k, w, turn);
                if (!std::isfinite(k) || !(w > 0)) continue;
                abs_out.add(std::abs(k), w);
                signed_out.add(k, w);
            }

            write_edge_hist_csv(abs_out, hdir / (stem_for_index(idx) + "_abs.csv"));
            write_edge_hist_csv(signed_out, hdir / (stem_for_index(idx) + "_signed.csv"));
        }
    }

    return st;
}

static void write_csv(const fs::path &path, const std::vector<Stats> &rows) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "index,target_G,points,stored_points,half_points,symmetry,length,area_signed,area,"
      << "mean_abs_curvature,trimmed_mean_abs_curvature_p99,mean_signed_curvature,"
      << "rms_curvature,max_abs_curvature,abs_curvature_p50,abs_curvature_p90,"
      << "abs_curvature_p95,abs_curvature_p99,abs_curvature_p999,total_abs_curvature,"
      << "signed_curvature_integral,signed_curvature_expected_integral,signed_curvature_turning_error,"
      << "geometric_turning_integral,geometric_turning_expected_integral,geometric_turning_error,"
      << "rollbacks,seconds,max_rel_G_error,rms_rel_G_error,quality_eval_stride,quality_samples\n";

    for (const auto &r : rows) {
        f << r.index << "," << r.G << "," << r.points << "," << r.meta.stored_points << "," << r.meta.half_points << "," << r.meta.symmetry << "," << r.length << "," << r.area_signed << "," << r.area << ","
          << r.mean_abs_curvature << "," << r.trimmed_mean_abs_curvature_p99 << "," << r.mean_signed_curvature << ","
          << r.rms_curvature << "," << r.max_abs_curvature << "," << r.abs_curvature_p50 << "," << r.abs_curvature_p90 << ","
          << r.abs_curvature_p95 << "," << r.abs_curvature_p99 << "," << r.abs_curvature_p999 << "," << r.total_abs_curvature << ","
          << r.signed_curvature_integral << "," << r.signed_curvature_expected_integral << "," << r.signed_curvature_turning_error << ","
          << r.geometric_turning_integral << "," << r.geometric_turning_expected_integral << "," << r.geometric_turning_error << ","
          << r.meta.rollbacks << "," << r.meta.seconds << "," << r.meta.max_rel_G_error << "," << r.meta.rms_rel_G_error << ","
          << r.meta.quality_eval_stride << "," << r.meta.quality_samples << "\n";
    }
}


static Config load_repository_config(const fs::path& config_path, const char* argv0) {
    const auto repo = mandelbrot::repo::RepoConfig::load(config_path, argv0);
    Config cfg;
    cfg.output_dir = repo.path("contours.output_dir").string();
    cfg.code_root = repo.code_root().string();
    cfg.project_root = repo.project_root().string();
    cfg.output_format = repo.string("contours.output_format", cfg.output_format);
    cfg.g_start = static_cast<double>(repo.number("contours.g_start", cfg.g_start));
    cfg.g_stop = static_cast<double>(repo.number("contours.g_stop", cfg.g_stop));
    cfg.n = repo.integer("contours.n", cfg.n);

    cfg.postprocess_save_hist_data = repo.boolean(
        "contours.postprocess_save_hist_data", cfg.postprocess_save_hist_data);
    cfg.postprocess_hist_bins = repo.integer(
        "contours.postprocess_hist_bins", cfg.postprocess_hist_bins);
    cfg.postprocess_quantile_bins = repo.integer(
        "contours.postprocess_quantile_bins", cfg.postprocess_quantile_bins);
    cfg.postprocess_hist_mode = repo.string(
        "contours.postprocess_hist_mode", cfg.postprocess_hist_mode);
    cfg.postprocess_abs_hist_x_scale = repo.string(
        "contours.postprocess_abs_hist_x_scale", cfg.postprocess_abs_hist_x_scale);
    cfg.postprocess_signed_hist_x_scale = repo.string(
        "contours.postprocess_signed_hist_x_scale", cfg.postprocess_signed_hist_x_scale);
    cfg.postprocess_abs_hist_y_scale = repo.string(
        "contours.postprocess_abs_hist_y_scale", cfg.postprocess_abs_hist_y_scale);
    cfg.postprocess_signed_hist_y_scale = repo.string(
        "contours.postprocess_signed_hist_y_scale", cfg.postprocess_signed_hist_y_scale);
    cfg.postprocess_abs_hist_x_symlog_threshold = static_cast<double>(repo.number(
        "contours.postprocess_abs_hist_x_symlog_threshold",
        cfg.postprocess_abs_hist_x_symlog_threshold));
    cfg.postprocess_signed_hist_x_symlog_threshold = static_cast<double>(repo.number(
        "contours.postprocess_signed_hist_x_symlog_threshold",
        cfg.postprocess_signed_hist_x_symlog_threshold));
    cfg.postprocess_abs_hist_y_symlog_threshold = static_cast<double>(repo.number(
        "contours.postprocess_abs_hist_y_symlog_threshold",
        cfg.postprocess_abs_hist_y_symlog_threshold));
    cfg.postprocess_signed_hist_y_symlog_threshold = static_cast<double>(repo.number(
        "contours.postprocess_signed_hist_y_symlog_threshold",
        cfg.postprocess_signed_hist_y_symlog_threshold));
    cfg.hist_plot_style = repo.string(
        "contours.postprocess_hist_plot_style", cfg.hist_plot_style);
    cfg.hist_plot_joined = repo.boolean(
        "contours.postprocess_hist_plot_joined", cfg.hist_plot_joined);
    cfg.hist_plot_show_points = repo.boolean(
        "contours.postprocess_hist_plot_show_points", cfg.hist_plot_show_points);
    cfg.postprocess_hist_qlo = static_cast<double>(repo.number(
        "contours.postprocess_hist_qlo", cfg.postprocess_hist_qlo));
    cfg.postprocess_hist_qhi = static_cast<double>(repo.number(
        "contours.postprocess_hist_qhi", cfg.postprocess_hist_qhi));

    const auto optional_text = [&](const std::string& key, const std::string& fallback) {
        const auto* value = repo.find(key);
        if (!value) return fallback;
        if (value->is_string()) return value->string();
        if (value->is_number()) {
            std::ostringstream out;
            out << std::setprecision(18) << static_cast<double>(value->number());
            return out.str();
        }
        throw std::runtime_error("Config value must be a string or number: " + key);
    };
    cfg.postprocess_abs_hist_plot_x_min = optional_text(
        "contours.postprocess_abs_hist_plot_x_min", cfg.postprocess_abs_hist_plot_x_min);
    cfg.postprocess_abs_hist_plot_x_max = optional_text(
        "contours.postprocess_abs_hist_plot_x_max", cfg.postprocess_abs_hist_plot_x_max);
    cfg.postprocess_abs_hist_plot_y_min = optional_text(
        "contours.postprocess_abs_hist_plot_y_min", cfg.postprocess_abs_hist_plot_y_min);
    cfg.postprocess_abs_hist_plot_y_max = optional_text(
        "contours.postprocess_abs_hist_plot_y_max", cfg.postprocess_abs_hist_plot_y_max);
    cfg.postprocess_signed_hist_plot_x_min = optional_text(
        "contours.postprocess_signed_hist_plot_x_min", cfg.postprocess_signed_hist_plot_x_min);
    cfg.postprocess_signed_hist_plot_x_max = optional_text(
        "contours.postprocess_signed_hist_plot_x_max", cfg.postprocess_signed_hist_plot_x_max);
    cfg.postprocess_signed_hist_plot_y_min = optional_text(
        "contours.postprocess_signed_hist_plot_y_min", cfg.postprocess_signed_hist_plot_y_min);
    cfg.postprocess_signed_hist_plot_y_max = optional_text(
        "contours.postprocess_signed_hist_plot_y_max", cfg.postprocess_signed_hist_plot_y_max);
    return cfg;
}

static void print_help(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [--config PATH] [options]\n\n"
        << "Streaming Streaming C++ postprocessor for Mandelbrot contour binaries/JSONs.\n"
        << "Writes contour_scaling.csv and optional lightweight histogram CSVs.\n\n"
        << "Options:\n"
        << "  --help                 Show this help.\n"
        << "  --config PATH          Alternate unified repository JSON config.\n"
        << "  --output-dir DIR        Override output_dir from config.\n"
        << "  --start-index N         First contour index. Default: 0.\n"
        << "  --end-index N           Last contour index, inclusive. Default: n-1.\n"
        << "  --no-hists              Do not write hist_data/*.csv files.\n"
        << "  --hist-bins N           Override postprocess_hist_bins for saved hist_data CSVs.\n"
        << "  --quantile-bins N       Override postprocess_quantile_bins for approximate quantiles.\n"
        << "  --hist-mode MODE        Override postprocess_hist_mode: equal_weight or equal_width.\n"
        << "  --signed-x-scale SCALE  Override postprocess_signed_hist_x_scale: linear, log, or symlog.\n"
        << "  --signed-symlog-threshold X\n"
        << "                         Override postprocess_signed_hist_x_symlog_threshold (old alias postprocess_signed_symlog_threshold also works).\n";
}


int main(int argc, char **argv) {
    try {
        fs::path cfg_path;
        std::optional<std::string> output_override;
        int start_index = 0;
        std::optional<int> end_index;
        std::optional<int> hist_bins_override;
        std::optional<int> quantile_bins_override;
        std::optional<std::string> hist_mode_override;
        std::optional<std::string> signed_x_scale_override;
        std::optional<double> signed_symlog_threshold_override;
        bool no_hists = false;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--help" || a == "-h") {
                print_help(argv[0]);
                return 0;
            } else if (a == "--config") {
                if (++i >= argc) throw std::runtime_error("--config needs a value");
                cfg_path = argv[i];
            } else if (a == "--output-dir") {
                if (++i >= argc) throw std::runtime_error("--output-dir needs a value");
                output_override = argv[i];
            } else if (a == "--start-index") {
                if (++i >= argc) throw std::runtime_error("--start-index needs a value");
                start_index = std::stoi(argv[i]);
            } else if (a == "--end-index") {
                if (++i >= argc) throw std::runtime_error("--end-index needs a value");
                end_index = std::stoi(argv[i]);
            } else if (a == "--hist-bins") {
                if (++i >= argc) throw std::runtime_error("--hist-bins needs a value");
                hist_bins_override = std::stoi(argv[i]);
            } else if (a == "--quantile-bins") {
                if (++i >= argc) throw std::runtime_error("--quantile-bins needs a value");
                quantile_bins_override = std::stoi(argv[i]);
            } else if (a == "--hist-mode") {
                if (++i >= argc) throw std::runtime_error("--hist-mode needs a value");
                hist_mode_override = lower(argv[i]);
            } else if (a == "--signed-x-scale") {
                if (++i >= argc) throw std::runtime_error("--signed-x-scale needs a value");
                signed_x_scale_override = lower(argv[i]);
            } else if (a == "--signed-symlog-threshold") {
                if (++i >= argc) throw std::runtime_error("--signed-symlog-threshold needs a value");
                signed_symlog_threshold_override = std::stod(argv[i]);
            } else if (a == "--no-hists") {
                no_hists = true;
            } else {
                throw std::runtime_error("unknown option: " + a);
            }
        }

        Config cfg = load_repository_config(cfg_path, argv[0]);
        if (output_override) {
            fs::path override_path(*output_override);
            if (override_path.is_relative()) {
                override_path = fs::path(cfg.code_root) / override_path;
            }
            cfg.output_dir = fs::absolute(override_path).lexically_normal().string();
        }
        if (hist_bins_override) cfg.postprocess_hist_bins = *hist_bins_override;
        if (quantile_bins_override) cfg.postprocess_quantile_bins = *quantile_bins_override;
        if (hist_mode_override) cfg.postprocess_hist_mode = *hist_mode_override;
        if (signed_x_scale_override) cfg.postprocess_signed_hist_x_scale = *signed_x_scale_override;
        if (signed_symlog_threshold_override) cfg.postprocess_signed_hist_x_symlog_threshold = *signed_symlog_threshold_override;
        if (no_hists) cfg.postprocess_save_hist_data = false;

        fs::path out = cfg.output_dir;
        if (!fs::exists(out)) throw std::runtime_error("output folder does not exist: " + out.string());

        auto levels = make_levels(cfg);
        int last = end_index ? std::min(*end_index, cfg.n - 1) : cfg.n - 1;
        start_index = std::max(0, start_index);
        if (start_index > last) throw std::runtime_error("empty index range");

        std::vector<Stats> rows;
        rows.reserve(static_cast<size_t>(last - start_index + 1));

        std::cerr << "C++ Mandelbrot contour postprocessor\n";
        const auto loaded_repo = mandelbrot::repo::RepoConfig::load(cfg_path, argv[0]);
        std::cerr << "config: " << loaded_repo.config_path() << "\n";
        std::cerr << "output: " << out << "\n";
        std::cerr << "indices: " << start_index << ".." << last
                  << ", hist_bins=" << cfg.postprocess_hist_bins
                  << ", quantile_bins=" << cfg.postprocess_quantile_bins
                  << ", hist_mode=" << cfg.postprocess_hist_mode
                  << ", q=[" << cfg.postprocess_hist_qlo << "," << cfg.postprocess_hist_qhi << "]"
                  << ", abs_x=" << cfg.postprocess_abs_hist_x_scale
                  << ", signed_x=" << cfg.postprocess_signed_hist_x_scale
                  << ", abs_y=" << cfg.postprocess_abs_hist_y_scale
                  << ", signed_y=" << cfg.postprocess_signed_hist_y_scale
                  << ", save_hist_data=" << (cfg.postprocess_save_hist_data ? "true" : "false") << "\n";
        write_hist_config_json(out, cfg);

        for (int idx = start_index; idx <= last; ++idx) {
            auto contour_file = find_contour_file_auto(out, idx, cfg.output_format);
            if (!contour_file) {
                std::cerr << "[" << stem_for_index(idx) << "] missing, skipping\n";
                continue;
            }
            const fs::path contour_path = contour_file->first;
            const std::string fmt = contour_file->second;

            Stats st;
            if (fmt == "bin") {
                st = compute_stats_binary_streaming(contour_path, idx, levels[static_cast<size_t>(idx)], cfg, out);
            } else {
                std::vector<Point> pts = read_points_json(contour_path);
                st = compute_stats_loaded(std::move(pts), idx, levels[static_cast<size_t>(idx)], cfg, out);
            }
            rows.push_back(st);
            std::cerr << "[" << stem_for_index(idx) << "] G=" << std::setprecision(8) << st.G
                      << " fmt=" << fmt << (fmt == "bin" ? "/stream" : "")
                      << " points=" << st.points
                      << " length=" << st.length
                      << " area=" << st.area
                      << " mean|k|=" << st.mean_abs_curvature
                      << " p99|k|=" << st.abs_curvature_p99
                      << " maxRelG=" << st.meta.max_rel_G_error
                      << "\n";
        }

        if (rows.empty()) {
            throw std::runtime_error("no contour files found");
        }

        fs::path csv = out / "contour_scaling.csv";
        write_csv(csv, rows);
        std::cerr << "CSV: " << csv << "\n";
        if (cfg.postprocess_save_hist_data) std::cerr << "Hist data: " << (out / "hist_data") << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
