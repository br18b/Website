#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace mandelbrot::catalogue {

namespace fs = std::filesystem;

class CatalogueDatabase;

inline constexpr const char* kManifestSchema = "mandelbrot-catalogue-v2";
inline constexpr const char* kComponentSchema = "mandelbrot-component-v4";
inline constexpr const char* kPeriodSchema = "mandelbrot-period-v3";
inline constexpr const char* kHierarchySchema = "mandelbrot-hierarchy-v2";
inline constexpr const char* kNumericEncoding = "decimal-string";
inline constexpr int kComponentKeyBits = 60;

using CatalogueReal = boost::multiprecision::number<
    boost::multiprecision::cpp_dec_float<200>>;

struct NumericMetadata {
    std::string encoding = kNumericEncoding;
    int working_precision_digits = 0;
    int validated_digits = 0;
};

struct ComplexValue {
    CatalogueReal re = 0;
    CatalogueReal im = 0;
};

struct GeometryRecord {
    std::string coordinate_frame = "centered";
    CatalogueReal polygon_rho = 0;
    std::vector<ComplexValue> polygon;
    CatalogueReal polygon_area = 0;
    CatalogueReal area_estimate = 0;
    CatalogueReal area_error = 0;
    CatalogueReal area_rho = 0;
    CatalogueReal characteristic_size = 0;
    std::array<CatalogueReal, 4> bbox_centered{0, 0, 0, 0};
};

struct CircleFitRecord {
    // Geometric centre relative to the dynamical component centre.  It is
    // deliberately not assumed to be zero.  Older v3 records may carry only
    // an RMS value, hence the optional geometric parameters.
    std::optional<ComplexValue> center_centered;
    std::optional<CatalogueReal> radius;
    CatalogueReal rms = 0;
    std::optional<CatalogueReal> max_error;
};

struct CardioidFitRecord {
    // Analytic model before rotation/translation:
    //   x = s (cos(phi) - 1/2 cos(2 phi)) (1 - xi sin(phi))
    //   y = s (sin(phi) - 1/2 sin(2 phi)) (1 - xi sin(phi))
    // xi=0 is the ordinary symmetric cardioid.
    std::optional<ComplexValue> center_centered;
    std::optional<CatalogueReal> size;
    CatalogueReal angle = 0;
    CatalogueReal xi = 0;
    CatalogueReal rms = 0;
    std::optional<CatalogueReal> max_error;
};

struct ClassificationRecord {
    // Canonical values are: unknown, disk, cardioid.  Readers normalize the
    // legacy value "circle" to "disk".
    std::string shape_class = "unknown";
    CatalogueReal shape_confidence = 0;
    std::optional<CircleFitRecord> circle_fit;
    std::optional<CardioidFitRecord> cardioid_fit;
};

struct ClassificationUpdate {
    std::string component_id;
    ClassificationRecord classification;
};

struct ExactPeriodIndex {
    int period = 0;
    int component_index = -1;
};

struct SymmetryRecord {
    std::string relation = "has-conjugate";
    int multiplicity = 2;
};

struct AttachmentRecord {
    std::optional<ComplexValue> parent_point;
    std::optional<ComplexValue> child_point_centered;
    std::optional<CatalogueReal> gap;
    std::optional<CatalogueReal> gap_relative_to_child_size;
    bool verified = false;
};

struct HierarchyRecord {
    std::optional<std::string> geometric_parent;
    std::optional<std::string> renormalization_parent;
    std::optional<std::string> hierarchy_root;
    std::optional<int> generation;
    std::optional<AttachmentRecord> attachment;
};

struct ProvenanceRecord {
    std::string method;
    std::string run_id;
    std::string discovered_at;
    std::string software_revision;
    std::vector<std::string> aliases;
};

struct QualityRecord {
    bool center_validated = false;
    bool exact_period_validated = false;
    bool polygon_converged = false;
    bool area_above_cutoff = false;
    std::vector<std::string> warnings;
};

struct ComponentRecord {
    std::string id;
    int period = 0;
    ComplexValue center;
    NumericMetadata numeric;
    std::string family = "quadratic-z2-plus-c";
    GeometryRecord geometry;
    ClassificationRecord classification;
    SymmetryRecord symmetry;
    HierarchyRecord hierarchy;
    ProvenanceRecord provenance;
    QualityRecord quality;
};

// Stable, immutable value key. It is useful for exact hash lookup and for
// checkpoint identities. Near-centre deduplication still confirms the actual
// distance, so a quantisation boundary can never create a false duplicate.
struct ComponentKey {
    int period = 0;
    std::int64_t center_re = 0;
    std::int64_t center_im = 0;

    static ComponentKey from_center(
        int period,
        const ComplexValue& center,
        int bits = kComponentKeyBits);

    bool operator==(const ComponentKey&) const noexcept = default;
};

struct ComponentKeyHash {
    std::size_t operator()(const ComponentKey& key) const noexcept;
};

struct PeriodRecord {
    int period = 0;
    std::string theoretical_component_count;
    std::size_t known_representative_count = 0;
    std::size_t known_component_count_with_symmetry = 0;
    bool catalogue_complete = false;
    CatalogueReal known_area = 0;
    CatalogueReal known_area_error = 0;
    CatalogueReal area_cutoff = 0;
    bool exact_geometry_complete = false;
    CatalogueReal polygon_rho = 0;
    CatalogueReal area_rho = 0;
    std::size_t polygon_points = 0;
    std::vector<std::string> component_ids;
    std::uint64_t generated_from_catalogue_revision = 0;
};

struct HierarchyNode {
    std::string id;
    std::optional<std::string> parent;
    std::vector<std::string> children;
};

struct HierarchyTree {
    std::string root;
    std::vector<HierarchyNode> nodes;
    std::size_t node_count = 0;
    int maximum_known_generation = 0;
    CatalogueReal known_area = 0;
    CatalogueReal minimum_stored_area = 0;
    bool complete_above_cutoff = false;
    std::uint64_t generated_from_catalogue_revision = 0;
};

struct Manifest {
    std::uint64_t catalogue_revision = 0;
    std::string family = "z^2+c";
    std::string canonical_half_plane = "imaginary>=0";
    std::size_t component_count_stored = 0;
    std::size_t component_count_with_symmetry = 0;
    CatalogueReal minimum_area = 0;
    int exact_through_period = 0;
    std::string created_at;
    std::string updated_at;
    std::string software_revision;
};

struct ComponentQuery {
    int min_period = 1;
    int max_period = std::numeric_limits<int>::max();
    std::optional<CatalogueReal> min_area;
    std::optional<CatalogueReal> max_area;
    bool require_polygon = false;
    bool require_center_validated = false;
    bool require_exact_period_validated = false;
    bool require_polygon_converged = false;
    std::optional<std::string> provenance_method;
    std::optional<std::string> hierarchy_root;
};

struct CatalogueSnapshot {
    Manifest manifest;
    std::vector<PeriodRecord> periods;
    std::vector<ComponentRecord> components;
    std::unordered_map<std::string, std::size_t> by_id;
    std::unordered_map<ComponentKey, std::vector<std::size_t>, ComponentKeyHash> by_key;
    std::unordered_map<int, std::vector<std::size_t>> by_period;

    const ComponentRecord* find_id(const std::string& id) const;
    const ComponentRecord* find_near_center(
        int period,
        const ComplexValue& center,
        const CatalogueReal& tolerance) const;
    std::vector<std::reference_wrapper<const ComponentRecord>> period_components(
        int period) const;
};

struct UpsertOptions {
    CatalogueReal center_tolerance = CatalogueReal("1e-15");
    bool merge_existing = true;
    bool bump_revision = true;
};

struct UpsertResult {
    ComponentRecord component;
    bool inserted = false;
    bool updated = false;
};

struct ComponentExportOptions {
    ComponentQuery query;
    std::string format = "mandelbrot-component-export-v2";
    bool complete = false;
    int coordinate_digits = 0;
};

// Typed area-scanner cache records. Canonical rows live in SQLite. Atomic CSV
// remains available only for recoverable checkpoint batches and interchange;
// scanner and demo code never knows its columns.
struct AreaScanCenterRecord {
    int period = 0;
    int component_index = 0;
    int expected_period_count = 0;
    ComplexValue center;
    CatalogueReal center_residual = 0;
    int detected_exact_period = 0;
    int conjugate_index = 0;
    int center_newton_iterations = 0;
    std::string center_refinement_method = "cpp-long-double";
    int center_refinement_dps = 0;
};

struct AreaMeasurementRecord {
    int period = 0;
    int component_index = 0;
    int conjugate_index = 0;
    int symmetry_source_component_index = 0;
    ComplexValue center;
    CatalogueReal rho = 0;
    int theta_points = 0;
    CatalogueReal area_polygon = 0;
    CatalogueReal area_derivative = 0;
    std::optional<CatalogueReal> area_fourier;
    CatalogueReal area_estimate = 0;
    CatalogueReal method_spread = 0;
    std::optional<CatalogueReal> spectral_spread;
    CatalogueReal resolution_delta = 0;
    CatalogueReal error_estimate = 0;
    std::optional<CatalogueReal> fourier_tail_ratio;
    std::optional<CatalogueReal> negative_mode_ratio;
    CatalogueReal closure_error = 0;
    CatalogueReal marked_z_closure_error = 0;
    CatalogueReal max_residual = 0;
    std::int64_t solve_calls = 0;
    std::int64_t failed_attempts = 0;
    std::int64_t newton_iterations = 0;
    int max_subdivision_depth = 0;
    std::int64_t rejected_branch_candidates = 0;
    std::int64_t cyclic_seed_attempts = 0;
    std::int64_t cyclic_recoveries = 0;
    int mp_solve_calls = 0;
    int mp_recoveries = 0;
    int max_mp_dps = 0;
    std::optional<CatalogueReal> seed_rho;
    bool converged = false;
    std::optional<CatalogueReal> exact_area_at_rho;
    std::optional<CatalogueReal> exact_relative_error;
    std::string failure_reason;
};

struct AreaPeriodSummaryRecord {
    int period = 0;
    CatalogueReal rho = 0;
    int expected_components = 0;
    int completed_components = 0;
    int converged_components = 0;
    int missing_or_unconverged_components = 0;
    bool period_complete = false;
    std::optional<CatalogueReal> min_area;
    std::optional<CatalogueReal> p10_area;
    std::optional<CatalogueReal> median_area;
    std::optional<CatalogueReal> mean_area;
    std::optional<CatalogueReal> p90_area;
    std::optional<CatalogueReal> max_area;
    CatalogueReal period_area = 0;
    CatalogueReal cumulative_area = 0;
    bool cumulative_complete_through_period = false;
    CatalogueReal summed_error_estimate = 0;
    std::optional<CatalogueReal> radial_increment_from_previous_rho;
};

using AreaScanProgressCallback = std::function<void(std::size_t, std::size_t)>;

class AreaScanStore {
public:
    AreaScanStore(fs::path catalogue_root, std::string run_name);

    const fs::path& run_directory() const noexcept { return run_dir_; }
    fs::path centers_path() const;
    fs::path measurements_path() const;
    fs::path summary_path() const;
    fs::path root_checkpoint_path(int period) const;

    std::map<int, std::vector<AreaScanCenterRecord>> load_centers(
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaScanCenterRecord> load_centers(
        int period,
        const AreaScanProgressCallback& progress = {}) const;
    void save_centers(
        const std::map<int, std::vector<AreaScanCenterRecord>>& centers,
        const AreaScanProgressCallback& progress = {}) const;
    void save_centers(
        int period,
        const std::vector<AreaScanCenterRecord>& centers,
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaMeasurementRecord> load_measurements(
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaMeasurementRecord> load_measurements(
        int period,
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaMeasurementRecord> load_measurements(
        int period,
        const CatalogueReal& rho,
        const AreaScanProgressCallback& progress = {}) const;
    void save_measurements(
        const std::vector<AreaMeasurementRecord>& measurements,
        const AreaScanProgressCallback& progress = {}) const;
    void save_measurements(
        int period,
        const std::vector<AreaMeasurementRecord>& measurements,
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaMeasurementRecord> load_measurements_from(
        const fs::path& path,
        const AreaScanProgressCallback& progress = {}) const;
    void save_measurements_to(
        const fs::path& path,
        const std::vector<AreaMeasurementRecord>& measurements,
        const AreaScanProgressCallback& progress = {}) const;
    std::vector<AreaPeriodSummaryRecord> load_summaries(
        const AreaScanProgressCallback& progress = {}) const;
    bool has_summaries() const;
    bool has_summaries(int period) const;
    void save_summaries(
        const std::vector<AreaPeriodSummaryRecord>& summaries,
        const AreaScanProgressCallback& progress = {}) const;
    void save_summaries(
        int period,
        const std::vector<AreaPeriodSummaryRecord>& summaries,
        const AreaScanProgressCallback& progress = {}) const;

private:
    fs::path root_;
    fs::path exports_;
    fs::path run_dir_;
    std::string run_name_;
    std::shared_ptr<CatalogueDatabase> database_;
};

class Catalogue {
public:
    explicit Catalogue(fs::path root);

    const fs::path& root() const noexcept { return root_; }
    fs::path database_path() const;
    // Legacy JSON paths are retained for the one-time migration utility only.
    fs::path manifest_path() const;
    fs::path component_path(const std::string& id) const;
    fs::path period_path(int period) const;
    fs::path hierarchy_path(const std::string& root_id) const;
    fs::path runs_path() const;
    fs::path exports_path() const;
    fs::path indexes_path() const;
    fs::path export_path(const std::string& name) const;
    fs::path run_path(
        const std::string& algorithm,
        const std::string& run_name,
        const std::string& name = "") const;

    void ensure_layout() const;

    Manifest load_manifest() const;
    void save_manifest(const Manifest& manifest) const;

    ComponentRecord load_component(const std::string& id) const;
    void save_component(const ComponentRecord& component, bool bump_revision = true) const;
    void save_components(
        const std::vector<ComponentRecord>& components,
        bool bump_revision = true) const;
    UpsertResult upsert_component(
        ComponentRecord component,
        const UpsertOptions& options = {}) const;
    std::vector<UpsertResult> upsert_components(
        std::vector<ComponentRecord> components,
        const UpsertOptions& options = {}) const;
    bool update_component_classification(
        const std::string& component_id,
        const ClassificationRecord& classification,
        bool bump_revision = true) const;
    std::size_t update_component_classifications(
        const std::vector<ClassificationUpdate>& updates,
        bool bump_revision = true) const;
    void delete_component(const std::string& id, bool bump_revision = true) const;
    bool component_exists(const std::string& id) const;
    std::vector<std::string> list_component_ids() const;
    std::vector<ComponentRecord> query_components(
        const ComponentQuery& query = {},
        const AreaScanProgressCallback& progress = {}) const;
    CatalogueSnapshot load_snapshot(const ComponentQuery& query = {}) const;
    std::vector<ComponentRecord> load_components_for_period(int period) const;
    std::optional<ComponentRecord> find_near_center(
        int period, ComplexValue center, const CatalogueReal& tolerance) const;

    PeriodRecord load_period(int period) const;
    void save_period(const PeriodRecord& period) const;
    bool period_exists(int period) const;
    std::vector<int> list_periods() const;
    void rebuild_period_indexes(const CatalogueReal& area_cutoff = 0) const;
    void rebuild_period_indexes(
        const std::vector<int>& periods,
        const CatalogueReal& area_cutoff = 0) const;

    HierarchyTree load_hierarchy(const std::string& root_id) const;
    void save_hierarchy(const HierarchyTree& tree) const;
    void rebuild_hierarchy_indexes(const CatalogueReal& minimum_stored_area = 0) const;

    void rebuild_manifest(int exact_through_period = 0,
                          const CatalogueReal& minimum_area = 0,
                          const std::string& software_revision = "") const;
    void rebuild_manifest_from_period_indexes(
        int exact_through_period = 0,
        const CatalogueReal& minimum_area = 0,
        const std::string& software_revision = "") const;

    std::size_t write_component_export(
        const fs::path& path,
        const ComponentExportOptions& options = {},
        const AreaScanProgressCallback& scan_progress = {},
        const AreaScanProgressCallback& write_progress = {}) const;
    void write_skeleton_export(
        const fs::path& path,
        const ComponentQuery& query = {}) const;

    AreaScanStore area_scan_store(const std::string& run_name = "default") const;

    // Runs SQLite's integrity and foreign-key checks and throws on any error.
    void verify_integrity() const;

    static std::string generate_uuid();
    static std::string stable_id(const std::string& identity);
    static ComponentRecord canonicalize_symmetry(
        ComponentRecord component,
        const CatalogueReal& real_axis_tolerance = CatalogueReal("1e-50"));
    static void validate_component(const ComponentRecord& component);
    static ComponentRecord merge_component_records(
        const ComponentRecord& existing,
        const ComponentRecord& incoming);

    static std::optional<ExactPeriodIndex> exact_period_index(
        const ComponentRecord& component);
    static void set_exact_period_index(
        ComponentRecord& component,
        int period,
        int component_index);

    static CatalogueReal parse_decimal(const std::string& value);
    static std::string decimal_string(const CatalogueReal& value,
                                      int digits = 0);

private:
    fs::path root_;
    std::shared_ptr<CatalogueDatabase> database_;
};

std::vector<ComplexValue> absolute_polygon(const ComponentRecord& component);
std::string utc_timestamp();

} // namespace mandelbrot::catalogue
