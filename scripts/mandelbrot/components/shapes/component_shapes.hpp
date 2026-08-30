#pragma once

#include "components/catalogue/component_catalogue.hpp"

namespace mandelbrot::shapes {

using catalogue::CardioidFitRecord;
using catalogue::CatalogueReal;
using catalogue::CircleFitRecord;
using catalogue::ClassificationRecord;
using catalogue::ComponentRecord;
using catalogue::ComplexValue;

struct CircleFitOptions {
    CatalogueReal rms_tolerance = CatalogueReal("0.01");
    CatalogueReal max_error_tolerance = CatalogueReal("0.03");
    int max_iterations = 16;
};

struct CircleFitAnalysis {
    bool converged = false;
    CircleFitRecord fit;

    bool confident(const CircleFitOptions& options) const;
};

// Fit a geometric disk to the authoritative centred polygon.  The returned
// centre is also centred on the dynamical component centre; no equality
// between those two centres is assumed.
CircleFitAnalysis fit_circle(
    const ComponentRecord& component,
    const CircleFitOptions& options = {});

// Conservative first-pass classifier: confidently circular polygons become
// "disk".  Everything else remains "unknown" for the cardioid fit.
ClassificationRecord classify_disk_quick(
    const ComponentRecord& component,
    const CircleFitOptions& options = {});

struct CardioidFitOptions {
    CatalogueReal rms_tolerance = CatalogueReal("0.01");
    CatalogueReal max_error_tolerance = CatalogueReal("0.03");

    // Try this many polygon points nearest the dynamical centre as possible
    // cusp locations.  Both polygon orientations are tested for every cusp.
    int cusp_candidates = 9;

    // Coordinate-descent refinement of the cusp-derived in-plane rotation,
    // followed by a joint damped Gauss-Newton polish of centre, log(size),
    // angle and slant xi.
    int angle_iterations = 12;
    CatalogueReal initial_angle_step = CatalogueReal("0.15");
    int max_iterations = 24;
    CatalogueReal xi_limit = CatalogueReal("0.5");

    // Final stochastic rescue for deterministic near misses.  This is a
    // reproducible Metropolis/simulated-annealing optimizer, not posterior
    // sampling: it perturbs the discrete cusp/ordering and continuous phase
    // alignment/rotation, while centre, size and xi are re-solved analytically
    // for each proposal.  Only deterministic failures enter this stage.
    bool randomized_fallback = true;
    int shake_trials = 384;
    int shake_keep = 8;
    int shake_cusp_jitter = 3;
    CatalogueReal shake_angle_sigma = CatalogueReal("0.08");
    // Standard deviation in polygon sample intervals, not radians.
    CatalogueReal shake_phase_sigma = CatalogueReal("0.5");
    CatalogueReal shake_temperature = CatalogueReal("0.08");
    CatalogueReal shake_final_temperature = CatalogueReal("0.002");
};

struct CardioidFitAnalysis {
    bool converged = false;
    int cusp_index = -1;
    int direction = 1;
    // Internal correspondence offset between the traced polygon samples and
    // the analytic cardioid parameter.  It does not alter the stored curve.
    CatalogueReal phase_offset = 0;
    bool used_randomized_fallback = false;
    bool rescued_by_randomized_fallback = false;
    int randomized_trials = 0;
    CardioidFitRecord fit;

    bool confident(const CardioidFitOptions& options) const;
};

// Fit the translated, rotated and vertically slanted cardioid model
//
//   x = s (cos(phi) - 1/2 cos(2 phi)) (1 - xi sin(phi))
//   y = s (sin(phi) - 1/2 sin(2 phi)) (1 - xi sin(phi))
//
// to the authoritative centred polygon.  The cusp is first located among the
// points nearest the dynamical centre.  Its direction supplies the initial
// rotation, with size initialized by s ~= 2 |cusp|.  A deterministic joint
// polish then refines translation, size, angle and xi.
CardioidFitAnalysis fit_cardioid_slanted(
    const ComponentRecord& component,
    const CardioidFitOptions& options = {});

// Evaluate the catalogue cardioid model in component-centred coordinates.
// xi=0 gives the ordinary cardioid.  Reflection across the real axis maps
// angle -> -angle and xi -> -xi.
ComplexValue cardioid_point(
    const CardioidFitRecord& fit,
    const CatalogueReal& phi);

} // namespace mandelbrot::shapes
