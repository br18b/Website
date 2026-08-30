#include "components/shapes/component_shapes.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace mandelbrot::catalogue;
using namespace mandelbrot::shapes;

int main() {
    ComponentRecord component;
    component.period = 2;
    component.geometry.polygon_rho = CatalogueReal("0.9995");
    const long double cx = 0.125L;
    const long double cy = -0.0625L;
    const long double radius = 0.25L;
    for (int i = 0; i < 192; ++i) {
        const long double phase = 2 * acosl(-1.0L) * i / 192;
        component.geometry.polygon.push_back({
            CatalogueReal(cx + radius * cosl(phase)),
            CatalogueReal(cy + radius * sinl(phase)),
        });
    }
    const auto analysis = fit_circle(component);
    assert(analysis.converged);
    assert(analysis.confident(CircleFitOptions{}));
    assert(abs(analysis.fit.center_centered->re - CatalogueReal("0.125")) < CatalogueReal("1e-15"));
    assert(abs(analysis.fit.center_centered->im - CatalogueReal("-0.0625")) < CatalogueReal("1e-15"));
    assert(abs(*analysis.fit.radius - CatalogueReal("0.25")) < CatalogueReal("1e-15"));
    assert(analysis.fit.rms < CatalogueReal("1e-14"));

    CardioidFitRecord main_cardioid;
    main_cardioid.center_centered = ComplexValue{0, 0};
    main_cardioid.size = CatalogueReal("0.5");
    main_cardioid.angle = 0;
    main_cardioid.xi = 0;
    main_cardioid.rms = 0;
    const CatalogueReal pi = boost::math::constants::pi<CatalogueReal>();
    const auto right = cardioid_point(main_cardioid, 0);
    const auto left = cardioid_point(main_cardioid, pi);
    assert(abs(right.re - CatalogueReal("0.25")) < CatalogueReal("1e-50"));
    assert(abs(right.im) < CatalogueReal("1e-50"));
    assert(abs(left.re + CatalogueReal("0.75")) < CatalogueReal("1e-50"));
    assert(abs(left.im) < CatalogueReal("1e-50"));

    CardioidFitRecord slanted_truth;
    slanted_truth.center_centered = ComplexValue{
        CatalogueReal("0.017"), CatalogueReal("-0.011")};
    slanted_truth.size = CatalogueReal("0.23");
    slanted_truth.angle = CatalogueReal("0.73");
    slanted_truth.xi = CatalogueReal("0.18");
    ComponentRecord slanted_component;
    slanted_component.period = 7;
    slanted_component.geometry.area_estimate = CatalogueReal("1");
    slanted_component.quality.polygon_converged = true;
    for (int i = 0; i < 192; ++i) {
        const CatalogueReal phi = 2 * pi * CatalogueReal(i)
            / CatalogueReal(192);
        slanted_component.geometry.polygon.push_back(
            cardioid_point(slanted_truth, phi));
    }
    const auto slanted = fit_cardioid_slanted(slanted_component);
    assert(slanted.converged);
    assert(slanted.confident(CardioidFitOptions{}));
    assert(abs(*slanted.fit.size - CatalogueReal("0.23"))
        < CatalogueReal("1e-12"));
    assert(abs(slanted.fit.angle - CatalogueReal("0.73"))
        < CatalogueReal("1e-12"));
    assert(abs(slanted.fit.xi - CatalogueReal("0.18"))
        < CatalogueReal("1e-12"));
    assert(slanted.fit.rms < CatalogueReal("1e-12"));

    std::reverse(
        slanted_component.geometry.polygon.begin(),
        slanted_component.geometry.polygon.end());
    const auto reversed_slanted = fit_cardioid_slanted(slanted_component);
    assert(reversed_slanted.converged);
    assert(abs(reversed_slanted.fit.xi - CatalogueReal("0.18"))
        < CatalogueReal("1e-12"));
    assert(reversed_slanted.fit.rms < CatalogueReal("1e-12"));

    std::cout << "Component shape models: OK\n";
    return 0;
}
