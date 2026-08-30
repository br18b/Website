#include "components/catalogue/component_catalogue.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace mandelbrot::catalogue;

namespace {

ComponentRecord make_record(
    const std::string& identity,
    int period,
    const std::string& re,
    const std::string& im,
    const std::string& area,
    const std::string& method = "exact-period-area-scan",
    int validated_digits = 16) {
    ComponentRecord component;
    component.id = Catalogue::stable_id(identity);
    component.period = period;
    component.center = {CatalogueReal(re), CatalogueReal(im)};
    component.numeric.working_precision_digits = 21;
    component.numeric.validated_digits = validated_digits;
    component.geometry.polygon_rho = CatalogueReal("0.9995");
    component.geometry.polygon = {
        {CatalogueReal("0"), CatalogueReal("0")},
        {CatalogueReal("1e-4"), CatalogueReal("0")},
        {CatalogueReal("0"), CatalogueReal("1e-4")},
    };
    component.geometry.polygon_area = CatalogueReal(area);
    component.geometry.area_estimate = CatalogueReal(area);
    component.geometry.area_error = CatalogueReal("1e-20");
    component.geometry.area_rho = CatalogueReal("0.99999");
    component.geometry.characteristic_size = CatalogueReal("1e-4");
    component.geometry.bbox_centered = {
        CatalogueReal("0"), CatalogueReal("1e-4"),
        CatalogueReal("0"), CatalogueReal("1e-4")};
    component.provenance.method = method;
    component.provenance.run_id = "test";
    Catalogue::set_exact_period_index(component, period, 17);
    component.quality.center_validated = true;
    component.quality.exact_period_validated = true;
    component.quality.polygon_converged = true;
    component.quality.area_above_cutoff = true;
    return component;
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path()
        / ("mandelbrot-catalogue-test-" + Catalogue::generate_uuid());
    fs::remove_all(root);
    Catalogue catalogue(root);
    catalogue.ensure_layout();

    ComponentRecord first = make_record(
        "candidate-a", 7, "-0.1", "0.7", "5e-9");
    ComponentRecord duplicate = make_record(
        "candidate-b", 7, "-0.1000000001", "0.7", "6e-9");
    UpsertOptions options;
    options.center_tolerance = CatalogueReal("1e-9");
    const auto results = catalogue.upsert_components({first, duplicate}, options);
    assert(results.size() == 2);
    assert(results[0].inserted);
    assert(results[1].updated);
    const auto unchanged = catalogue.upsert_component(results[1].component, options);
    assert(!unchanged.inserted && !unchanged.updated);

    catalogue.rebuild_period_indexes(std::vector<int>{7}, CatalogueReal("1e-10"));
    const auto snapshot = catalogue.load_snapshot(ComponentQuery{});
    assert(snapshot.find_near_center(7, first.center, CatalogueReal("1e-9")) != nullptr);

    ComponentRecord exact = make_record(
        "merge-exact", 9, "-0.2", "0.6", "2e-12",
        "exact-period-area-scan", 16);
    ComponentRecord boundary = make_record(
        "merge-boundary", 9, "-0.2", "0.6", "9e-12",
        "boundary-hunter", 16);
    boundary.classification.shape_class = "disk";
    CircleFitRecord circle_fit;
    circle_fit.center_centered = ComplexValue{CatalogueReal("1e-8"), CatalogueReal("-2e-8")};
    circle_fit.radius = CatalogueReal("3e-4");
    circle_fit.rms = CatalogueReal("0.001");
    circle_fit.max_error = CatalogueReal("0.003");
    boundary.classification.circle_fit = circle_fit;
    boundary.classification.shape_confidence = CatalogueReal("0.95");
    boundary.hierarchy.geometric_parent = first.id;

    const ComponentRecord merged = Catalogue::merge_component_records(exact, boundary);
    assert(merged.geometry.area_estimate == CatalogueReal("2e-12"));
    assert(merged.provenance.method == "exact-period-area-scan");
    assert(merged.classification.shape_class == "disk");
    assert(merged.hierarchy.geometric_parent == first.id);

    const ComponentRecord upgraded = Catalogue::merge_component_records(boundary, exact);
    assert(upgraded.geometry.area_estimate == CatalogueReal("2e-12"));
    assert(upgraded.provenance.method == "exact-period-area-scan");

    ClassificationRecord cardioid_classification;
    cardioid_classification.shape_class = "cardioid";
    cardioid_classification.shape_confidence = CatalogueReal("0.8");
    CardioidFitRecord cardioid_fit;
    cardioid_fit.center_centered = ComplexValue{CatalogueReal("0"), CatalogueReal("0")};
    cardioid_fit.size = CatalogueReal("0.5");
    cardioid_fit.angle = CatalogueReal("0");
    cardioid_fit.xi = CatalogueReal("0.125");
    cardioid_fit.rms = CatalogueReal("0.02");
    cardioid_fit.max_error = CatalogueReal("0.05");
    cardioid_classification.cardioid_fit = cardioid_fit;
    assert(catalogue.update_component_classification(results[1].component.id, cardioid_classification));
    const auto classified = catalogue.load_component(results[1].component.id);
    assert(classified.classification.cardioid_fit->xi == CatalogueReal("0.125"));

    const ComponentKey key = ComponentKey::from_center(first.period, first.center);
    const ComponentKey same = ComponentKey::from_center(first.period, first.center);
    assert(key == same);
    assert(ComponentKeyHash{}(key) == ComponentKeyHash{}(same));
    const auto exact_index = Catalogue::exact_period_index(first);
    assert(exact_index && exact_index->period == 7
        && exact_index->component_index == 17);
    Catalogue::set_exact_period_index(first, 7, 23);
    const auto replaced_index = Catalogue::exact_period_index(first);
    assert(replaced_index && replaced_index->component_index == 23);

    auto store = catalogue.area_scan_store("test");
    AreaScanCenterRecord center_record;
    center_record.period = 7;
    center_record.component_index = 0;
    center_record.expected_period_count = 1;
    center_record.center = first.center;
    center_record.detected_exact_period = 7;
    store.save_centers({{7, {center_record}}});
    std::vector<std::pair<std::size_t, std::size_t>> center_progress;
    const auto loaded_centers = store.load_centers(
        [&](std::size_t current, std::size_t total) {
            center_progress.emplace_back(current, total);
        });
    assert(loaded_centers.at(7).front().center.re == first.center.re);
    assert(center_progress.front() == std::make_pair(std::size_t{0}, std::size_t{1}));
    assert(center_progress.back() == std::make_pair(std::size_t{1}, std::size_t{1}));
    AreaScanCenterRecord period_eight_center = center_record;
    period_eight_center.period = 8;
    period_eight_center.detected_exact_period = 8;
    store.save_centers(8, {period_eight_center});
    assert(store.load_centers(8).size() == 1);
    center_record.center_residual = CatalogueReal("1e-30");
    store.save_centers(7, {center_record});
    assert(store.load_centers(7).front().center_residual == CatalogueReal("1e-30"));
    assert(store.load_centers().at(8).size() == 1);

    AreaMeasurementRecord measurement;
    measurement.period = 7;
    measurement.component_index = 0;
    measurement.center = first.center;
    measurement.rho = CatalogueReal("0.99999");
    measurement.area_estimate = CatalogueReal("5e-9");
    measurement.converged = true;
    AreaMeasurementRecord replay = measurement;
    replay.area_estimate = CatalogueReal("6e-9");
    store.save_measurements({measurement, replay});
    std::vector<std::pair<std::size_t, std::size_t>> measurement_progress;
    const auto measurements = store.load_measurements(
        [&](std::size_t current, std::size_t total) {
            measurement_progress.emplace_back(current, total);
        });
    assert(measurements.size() == 1);
    assert(measurements.front().area_estimate == CatalogueReal("6e-9"));
    assert(measurement_progress.front()
        == std::make_pair(std::size_t{0}, std::size_t{1}));
    assert(measurement_progress.back()
        == std::make_pair(std::size_t{1}, std::size_t{1}));
    AreaMeasurementRecord period_eight_measurement = measurement;
    period_eight_measurement.period = 8;
    store.save_measurements(8, {period_eight_measurement});
    assert(store.load_measurements(8).size() == 1);
    assert(store.load_measurements(8, CatalogueReal("0.99999")).size() == 1);
    measurement.area_estimate = CatalogueReal("7e-9");
    store.save_measurements(7, {measurement});
    assert(store.load_measurements(7).front().area_estimate == CatalogueReal("7e-9"));
    assert(store.load_measurements(8).size() == 1);

    AreaPeriodSummaryRecord summary;
    summary.period = 7;
    summary.rho = CatalogueReal("0.99999");
    summary.expected_components = 1;
    summary.completed_components = 1;
    summary.converged_components = 1;
    summary.period_complete = true;
    summary.period_area = CatalogueReal("6e-9");
    summary.cumulative_area = CatalogueReal("6e-9");
    store.save_summaries({summary});
    std::vector<std::pair<std::size_t, std::size_t>> summary_progress;
    const auto summaries = store.load_summaries(
        [&](std::size_t current, std::size_t total) {
            summary_progress.emplace_back(current, total);
        });
    assert(summaries.front().period_complete);
    assert(summary_progress.front()
        == std::make_pair(std::size_t{0}, std::size_t{1}));
    assert(summary_progress.back()
        == std::make_pair(std::size_t{1}, std::size_t{1}));
    AreaPeriodSummaryRecord period_eight_summary = summary;
    period_eight_summary.period = 8;
    store.save_summaries(8, {period_eight_summary});
    assert(store.has_summaries(7));
    assert(store.has_summaries(8));
    summary.period_area = CatalogueReal("7e-9");
    store.save_summaries(7, {summary});
    const auto scoped_summaries = store.load_summaries();
    assert(scoped_summaries.size() == 2);
    assert(scoped_summaries.front().period_area == CatalogueReal("7e-9"));

    catalogue.verify_integrity();
    assert(fs::is_regular_file(catalogue.database_path()));
    assert(!fs::exists(catalogue.component_path(first.id)));

    fs::remove_all(root);
    std::cout << "C++ typed catalogue API: OK\n";
    return 0;
}
