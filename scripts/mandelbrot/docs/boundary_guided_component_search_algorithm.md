# Boundary-Guided Hierarchical Search for Hyperbolic Components of the Mandelbrot Set

## Motivation

The current adaptive quadtree search treats the complex plane as a
mostly uniform search domain. While this successfully discovers many
components, it becomes increasingly inefficient at high periods because
enormous numbers of cells must be examined even though almost all
interesting components are attached to already-known ones.

The proposed algorithm instead treats hyperbolic components as nodes in
a growing tree and searches locally around their boundaries. The search
becomes hierarchical rather than spatial.

## High-level algorithm

1.  Build the exact catalogue (e.g. periods 1--14).
2.  Use these components as initial seeds.
3.  Search only around component boundaries.
4.  For every newly discovered component:
    -   recover its center,
    -   determine its period,
    -   trace its polygon,
    -   compute its area,
    -   classify it as approximately circle-like or cardioid-like,
    -   connect it to its parent,
    -   enqueue it for further exploration.
5.  Continue until all queues are exhausted.
6.  Run a reduced global search to locate isolated cardioid-like roots.
7.  Resume boundary expansion.
8.  Repeat until no additional eligible components are found.

## Component model

Each component stores:

-   unique ID
-   parent ID
-   hierarchy path (e.g. 0/1/2)
-   center
-   polygon
-   area
-   period
-   shape classification
-   attachment points
-   attachment distance
-   discovery generation

The main cardioid is root 0.

## Queues

Maintain separate queues for:

-   circle-like components
-   cardioid-like components

The main cardioid is the initial queue entry.

## Boundary parameterization

Parameterize each polygon by:

-   boundary coordinate s
-   outward normal distance n

Trace polygons near rho = 0.99999.

Normal samples should be strongly concentrated near the boundary using
logarithmic spacing.

## Boundary search

Each boundary sample defines an outward ray.

For every ray:

-   search from the smallest outward distance first,
-   accept only the nearest valid component,
-   disable that ray after a successful discovery.

## Adaptive refinement

Refine tangentially whenever neighboring rays disagree.

Refine normally whenever escape behaviour or component identity changes.

Only ambiguous regions are subdivided.

## Cycle detection

Use Brent cycle detection instead of a hard period ceiling.

After detecting an attracting cycle:

-   estimate its period,
-   verify it,
-   refine it,
-   continue the multiplier to lambda = 0,
-   recover the superattracting center.

## Component recovery

For every accepted center:

1.  trace the multiplier contour,
2.  generate a polygon,
3.  compute area,
4.  reject areas below threshold,
5.  classify as circle or cardioid.

The polygon remains the authoritative geometry.

## Parent verification

Verify that parent and child nearly touch.

Compute the minimum polygon-to-polygon distance.

Accept the relationship only if

gap \< 0.01 × child size

(configurable).

Otherwise defer assignment.

## Child ordering

Assign deterministic child indices using attachment location (and
optionally area) so hierarchy paths remain stable.

## Residual global search

After boundary expansion converges:

-   spatially index known polygons,
-   perform a reduced adaptive quadtree search,
-   reject probes inside known regions,
-   keep only isolated cardioid discoveries,
-   enqueue them,
-   resume boundary exploration.

## Checkpointing

Checkpoint:

-   accepted components
-   pending queues
-   ray states
-   subdivision state
-   deferred candidates

Ctrl+C should finish the current batch, save checkpoints, and exit
cleanly.

## Outputs

Produce:

-   polygon catalogue
-   hierarchy tree
-   skeleton graph containing only centers and parent-child edges

## Expected advantages

-   dramatically fewer wasted samples
-   explicit component hierarchy
-   no arbitrary period ceiling
-   restartable execution
-   better scalability to very high periods
-   scientifically meaningful connectivity graph

## Initial v1 implementation notes

The first implementation in `components/component_boundary_hunter.cpp` uses a
mild configuration to validate the architecture:

- 32 initial boundary rays
- 8 logarithmically spaced outward-normal probes per ray
- 2 additional tangential subdivision levels
- at most 64 source components per invocation
- Brent-style approximate cycle detection without a configured period ceiling
- polygon tracing at `rho = 0.99999`
- parent verification from minimum polygon-to-polygon distance
- resumable TSV checkpointing after each source component

This is not yet the final adaptive strip solver. In particular, normal-interval
subdivision and the residual low-G cardioid search remain future stages.
