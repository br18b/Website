import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell, norm, lognorm
from tqdm import tqdm
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell, norm, lognorm
import os

def multiplicative_clt_demo(
    n=5,
    N=10000,
    bins=100,
    base_dir="",
    title="Multiplicative CLT Demo",
    filename="lognormal_plot.png",
    sigma=2.0,
    toggle_gaussian=True,
    toggle_lognormal=True,
    threshold=0.005
):
    # sample mach numbers from maxwellian
    log_density = np.zeros(N)
    for _ in tqdm(range(n), desc="Applying shocks", ascii=True):
        mach = maxwell.rvs(scale=sigma, size=N)
        log_density += 2 * np.log(mach)  # because shock_factors = mach^2

    # normalize the ensemble towards zero mean and unit variance
    mu = np.mean(log_density)
    std = np.std(log_density)
    log_density_norm = (log_density - mu) / std

    # we also need density for plotting
    density = np.exp(log_density_norm)
    mu = 0
    std = 1

    # --- LOG-DENSITY PLOT ---
    plt.figure(figsize=(8, 5))
    counts, bin_edges = np.histogram(log_density_norm, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1]

    plt.bar(bin_centers, counts, width=bin_width, color="skyblue", edgecolor="none", align="edge", label="Normalized log-density")
    for x, height in zip(bin_centers, counts):
        plt.plot([x, x + bin_width], [height, height], color="black", linewidth=0.5)

    if toggle_gaussian:
        significant = counts > threshold * np.max(counts)
        x_min = -4.5
        x_max = 4.5
        x = np.linspace(x_min, x_max, 1000)
        y = norm.pdf(x, loc=0, scale=1)
        plt.plot(x, y, 'k--', label="Ideal Normal Distribution")

    plt.title(f"{title}\nLog-density (n={n}, N={N}, σ={sigma})")
    plt.xlabel("Normalized log-density")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.ylim(top=0.5)
    plt.xlim(x[0], x[-1])

    path_log = os.path.join(base_dir, "log_" + filename)
    plt.savefig(path_log, dpi=150)
    plt.close()
    print(f"Saved log-density plot to {path_log}")

    # --- DENSITY PLOT ---
    plt.figure(figsize=(8, 5))

    # Fix histogram x-axis range
    density_min = 0
    density_max = 10  # adjust as needed

    # Manually define bin edges and bin centers
    bin_edges_d = np.linspace(density_min, density_max, bins + 1)
    bin_width_d = bin_edges_d[1] - bin_edges_d[0]
    bin_centers_d = bin_edges_d[:-1]

    # Clip data to range
    density_clipped = density[
        (density >= density_min) &
        (density <= density_max)]
    counts_d, _ = np.histogram(density_clipped,
        bins=bin_edges_d, density=True)

    # Plot bars
    plt.bar(
        bin_centers_d,
        counts_d,
        width=bin_width_d,
        color="lightcoral",
        edgecolor="none",
        align="edge",
        label="Density"
    )

    for x, height in zip(bin_centers_d, counts_d):
        plt.plot([x, x + bin_width_d], [height, height], color="black", linewidth=0.5)

    # Overlay lognormal
    if toggle_lognormal:
        x_d = np.linspace(density_min, density_max, 1000)
        y = lognorm.pdf(x_d, s=std, scale=np.exp(mu))
        plt.plot(x_d, y, 'k--', label="Ideal Lognormal Distribution")

    # Finalize plot
    plt.title(f"{title}\nDensity (n={n}, N={N}, σ={sigma})")
    plt.xlabel("Density")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.ylim(top=max(counts_d.max(), y.max() if toggle_lognormal else 0) * 1.2)
    plt.xlim(density_min, density_max)

    # Save
    path_density = os.path.join(base_dir, "density_" + filename)
    plt.savefig(path_density, dpi=150)
    plt.close()
    print(f"Saved density plot to {path_density}")

output_dir = Path(__file__).resolve().parents[2] / "work" / "promote" / "CLT_plots"
os.makedirs(output_dir, exist_ok=True)

for n in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]:
    multiplicative_clt_demo(
        n=n,
        N=10000000,
        bins=200,
        base_dir=output_dir,
        title="Multiplicative Central Limit Theorem",
        filename="mach_lognormal_n" + str(n) + ".png",
        sigma=2.0
    )