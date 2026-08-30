import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import time
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm
from skimage import measure
from scipy import ndimage as ndi
import open3d as o3d


def mandelbulb_chunk(args):
    (
        grid_size,
        z0_core,
        z1_core,
        power,
        max_iter,
        bailout,
        extent,
    ) = args

    xs = np.linspace(-extent, extent, grid_size, dtype=np.float32)
    ys = np.linspace(-extent, extent, grid_size, dtype=np.float32)
    zs = np.linspace(-extent, extent, grid_size, dtype=np.float32)[z0_core:z1_core]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    x = np.zeros_like(X, dtype=np.float32)
    y = np.zeros_like(Y, dtype=np.float32)
    z = np.zeros_like(Z, dtype=np.float32)

    active = np.ones(X.shape, dtype=bool)

    for _ in range(max_iter):
        r = np.sqrt(x * x + y * y + z * z)

        escaped = (r > bailout) & active
        active[escaped] = False

        if not np.any(active):
            break

        idx = active

        r_i = r[idx]
        theta = np.arctan2(np.sqrt(x[idx] ** 2 + y[idx] ** 2), z[idx])
        phi = np.arctan2(y[idx], x[idx])

        rn = r_i ** power
        theta_n = theta * power
        phi_n = phi * power

        x_new = rn * np.sin(theta_n) * np.cos(phi_n) + X[idx]
        y_new = rn * np.sin(theta_n) * np.sin(phi_n) + Y[idx]
        z_new = rn * np.cos(theta_n) + Z[idx]

        x[idx] = x_new
        y[idx] = y_new
        z[idx] = z_new

    # Return just the inside/outside classification for this slab.
    return z0_core, z1_core, active


def mandelbulb_volume_parallel(
    grid_size=256,
    power=8,
    max_iter=18,
    bailout=2.0,
    extent=1.4,
    chunk_depth=16,
    workers=None,
):
    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    chunks = []
    for z0 in range(0, grid_size, chunk_depth):
        z1 = min(grid_size, z0 + chunk_depth)
        chunks.append((
            grid_size,
            z0,
            z1,
            power,
            max_iter,
            bailout,
            extent,
        ))

    print(f"Grid: {grid_size}³ = {grid_size**3:,} sample points")
    print(f"Chunks: {len(chunks)}")
    print(f"Workers: {workers}")

    volume = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(mandelbulb_chunk, chunk) for chunk in chunks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slabs"):
            z0, z1, active = future.result()
            volume[:, :, z0:z1] = active

    print(f"Inside voxels: {np.count_nonzero(volume):,}")

    xs = np.linspace(-extent, extent, grid_size, dtype=np.float32)
    ys = np.linspace(-extent, extent, grid_size, dtype=np.float32)
    zs = np.linspace(-extent, extent, grid_size, dtype=np.float32)

    return volume, xs, ys, zs


def clean_volume_components(
    volume,
    min_component_voxels=0,
    keep_largest=False,
):
    """
    Remove tiny disconnected components from a boolean inside/outside volume.

    min_component_voxels=0 keeps everything.
    keep_largest=True keeps only the largest connected component.
    """

    if min_component_voxels <= 0 and not keep_largest:
        return volume

    print("Labeling connected components...")

    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n_labels = ndi.label(volume, structure=structure)

    if n_labels == 0:
        print("No connected components found.")
        return volume

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background

    if keep_largest:
        keep_labels = [int(np.argmax(counts))]
        print(f"Keeping largest component: {counts[keep_labels[0]]:,} voxels")
    else:
        keep_labels = np.flatnonzero(counts >= min_component_voxels)
        print(
            f"Keeping {len(keep_labels):,}/{n_labels:,} components "
            f"with at least {min_component_voxels:,} voxels"
        )

    cleaned = np.isin(labels, keep_labels)

    print(f"Voxels before cleanup: {np.count_nonzero(volume):,}")
    print(f"Voxels after cleanup:  {np.count_nonzero(cleaned):,}")

    return cleaned


def simplify_mesh_progressive(
    mesh,
    target_triangles=300_000,
    reduction_factor=0.90,
    min_triangle_drop=1_000,
):
    """
    Progressively simplify a mesh by repeatedly reducing the current triangle
    count by `reduction_factor` until `target_triangles` is reached.

    reduction_factor=0.90 means:
        2,600,000 -> 2,340,000 -> 2,106,000 -> ...
    """

    if not (0 < reduction_factor < 1):
        raise ValueError("reduction_factor must be between 0 and 1")

    current = len(np.asarray(mesh.triangles))
    target_triangles = min(target_triangles, current)

    print(
        f"Simplifying mesh: {current:,} → {target_triangles:,} triangles "
        f"using progressive reduction factor {reduction_factor:.3f}"
    )

    simplified = mesh
    stage = 0

    with tqdm(desc="Simplification stages", unit=" stage") as pbar:
        while True:
            before = len(np.asarray(simplified.triangles))

            if before <= target_triangles:
                break

            next_target = int(before * reduction_factor)
            next_target = max(next_target, target_triangles)

            # Avoid tiny useless stages near the end.
            if before - next_target < min_triangle_drop:
                next_target = target_triangles

            stage += 1
            t0 = time.perf_counter()

            simplified_next = simplified.simplify_quadric_decimation(
                target_number_of_triangles=next_target
            )

            simplified_next.remove_degenerate_triangles()
            simplified_next.remove_duplicated_triangles()
            simplified_next.remove_duplicated_vertices()
            simplified_next.remove_non_manifold_edges()

            after = len(np.asarray(simplified_next.triangles))
            dt = time.perf_counter() - t0

            tqdm.write(
                f"  stage {stage:02d}: "
                f"{before:,} → {after:,} triangles "
                f"(target {next_target:,}) in {dt:.1f} s"
            )

            simplified = simplified_next
            pbar.update(1)

            # Safety guard: if Open3D fails to reduce further, stop.
            if after >= before:
                print("No further simplification achieved; stopping.")
                break

    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    simplified.remove_non_manifold_edges()

    try:
        simplified.orient_triangles()
    except Exception as exc:
        print(f"Could not orient triangles: {exc}")

    simplified.compute_vertex_normals()
    return simplified


def sigma_voxels_from_world(smooth_sigma_world, xs):
    dx = float(xs[1] - xs[0])
    return smooth_sigma_world / dx


def build_simplified_mesh_from_volume(
    volume,
    xs,
    ys,
    zs,
    target_triangles=300_000,
    min_component_voxels=250,
    keep_largest=False,
    smooth_sigma=0.65,
    taubin_smoothing_iterations=0,
    reduction_factor=0.9,
):
    print("Preparing volume for marching cubes...")

    # Optional cleanup: removes tiny floating islands/noise.
    volume = clean_volume_components(
        volume,
        min_component_voxels=min_component_voxels,
        keep_largest=keep_largest,
    )

    # Pad with one empty voxel layer to help produce a closed surface.
    volume_padded = np.pad(
        volume.astype(np.float32),
        1,
        mode="constant",
        constant_values=0.0,
    )

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dz = float(zs[1] - zs[0])

    if smooth_sigma and smooth_sigma > 0:
        print(f"Smoothing volume field with sigma={smooth_sigma} voxels...")
        field = ndi.gaussian_filter(volume_padded, sigma=smooth_sigma)
    else:
        print("No volume smoothing.")
        field = volume_padded

    print("Running marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(
        field,
        level=0.5,
        spacing=(dx, dy, dz),
        allow_degenerate=False,
        step_size=1,
    )

    # Because of padding, shift origin by one voxel outward.
    x0 = float(xs[0] - dx)
    y0 = float(ys[0] - dy)
    z0 = float(zs[0] - dz)

    verts[:, 0] += x0
    verts[:, 1] += y0
    verts[:, 2] += z0

    print(f"Raw marching-cubes mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    print("Building Open3D mesh...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if taubin_smoothing_iterations > 0:
        print(f"Applying Taubin smoothing: {taubin_smoothing_iterations} iterations...")
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=taubin_smoothing_iterations
        )
        mesh.compute_vertex_normals()

    target_triangles = min(target_triangles, len(np.asarray(mesh.triangles)))

    mesh_simplified = simplify_mesh_progressive(
        mesh,
        target_triangles=target_triangles,
        reduction_factor=reduction_factor,
    )

    verts_s = np.asarray(mesh_simplified.vertices)
    faces_s = np.asarray(mesh_simplified.triangles)

    print(f"Simplified mesh: {len(verts_s):,} vertices, {len(faces_s):,} triangles")

    return mesh_simplified, verts_s, faces_s


def plot_mesh_html(
    verts,
    faces,
    output_html="mandelbulb_mesh.html",
    cmin=None,
    cmax=None,
):
    print("Preparing Plotly mesh...")

    radius = np.sqrt(
        verts[:, 0] * verts[:, 0] +
        verts[:, 1] * verts[:, 1] +
        verts[:, 2] * verts[:, 2]
    )

    # Auto color range if not explicitly provided.
    if cmin is None:
        cmin = float(np.nanmin(radius))
    if cmax is None:
        cmax = float(np.nanmax(radius))

    # Avoid degenerate color range if all values are somehow identical.
    if np.isclose(cmin, cmax):
        eps = 1e-6 if cmin == 0 else abs(cmin) * 1e-6
        cmin -= eps
        cmax += eps

    print(f"Color range: {cmin:.6g} → {cmax:.6g}")

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=radius,
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Radius"),
                showscale=True,
                opacity=1.0,
                flatshading=False,
                hoverinfo="skip",
                lighting=dict(
                    ambient=0.35,
                    diffuse=0.75,
                    specular=0.25,
                    roughness=0.75,
                    fresnel=0.05,
                ),
                lightposition=dict(x=100, y=200, z=120),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    grid_size = 1200
    extent = 1.4

    volume, xs, ys, zs = mandelbulb_volume_parallel(
        grid_size=grid_size,
        power=2,
        max_iter=50,
        bailout=2.0,
        extent=extent,
        chunk_depth=16,
        workers=8,
    )

    # Match the physical smoothing you liked at 512³ with sigma=0.65.
    dx_512 = (2 * extent) / (512 - 1)
    smooth_sigma_world = 0.65 * dx_512
    smooth_sigma = sigma_voxels_from_world(smooth_sigma_world, xs)

    print(f"Using smooth_sigma={smooth_sigma:.3f} voxels")

    mesh_simplified, verts_s, faces_s = build_simplified_mesh_from_volume(
        volume,
        xs,
        ys,
        zs,
        target_triangles=100_000,
        min_component_voxels=4_000,
        keep_largest=False,
        smooth_sigma=smooth_sigma,
        taubin_smoothing_iterations=0,
        reduction_factor=0.7,
    )

    o3d.io.write_triangle_mesh("mandelbulb_simplified.ply", mesh_simplified)
    print("Wrote mandelbulb_simplified.ply")

    plot_mesh_html(
        verts_s,
        faces_s,
        output_html="mandelbulb_mesh.html",
    )