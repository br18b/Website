import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import norm

def clt_demo(
    dist_func,     # function to draw random samples
    n=5,           # number of random variables to sum
    N=10000,       # how many sums to create
    bins=100,       # number of bins for the histogram
    base_dir="",
    title="CLT Demo",
    filename="plot.png",   # filename to save
    bounds=[-4.5,4.5],  # cutoff if the distribution is weird
    toggle_gaussian=True,
    normalize=True
):
    # create the ensemble of averages. Tracking progress with tqdm
    sums = np.zeros(N)
    for _ in tqdm(range(n), desc="Adding random samples", ascii=True):
        sums += dist_func(size=N)
    sums = sums / n
    
    # mean and variance
    mean = np.mean(sums)
    std = np.std(sums)
    
    # normalize ensemble
    if normalize:
        normalized = (sums - mean) / std
    else:
        normalized = sums

    normalized = normalized[(normalized > bounds[0]) &
     (normalized < bounds[1])]

    # plot and save
    plt.figure(figsize=(8, 5))

    # Compute histogram manually
    counts, bin_edges = np.histogram(normalized, bins=bins, density=True)

    # Compute bin centers and widths
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1]

    # Plot filled bars without edgecolor
    plt.bar(
        bin_centers,
        counts,
        width=bin_width,
        color="skyblue",
        edgecolor="none",
        align="edge",
        label="Ensemble histogram"
    )

    # Draw only the top edges
    for x, height in zip(bin_centers, counts):
        plt.plot([x, x + bin_width], [height, height], color="black", linewidth=0.5)

    # overplot ideal normal distribution
    if toggle_gaussian:
        x = np.linspace(bounds[0], bounds[1], 1000)
        y = norm.pdf(x, loc=0, scale=1)  # zero mean, unit variance
        plt.plot(x, y, 'k--', label="Ideal Normal Distribution")

    plt.title(f"{title}\n(n={n}, N={N})")
    plt.xlabel("Normalized Sum")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.legend(loc="upper right")

    # give the plot more breathing space, vertically
    ymax = max(np.max(counts), np.max(y) if toggle_gaussian else 0)
    plt.ylim(top=ymax * 1.20)  # Add 10% headroom


    # save figure
    full_path = os.path.join(base_dir, filename)
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"Saved plot to {full_path}")

# fun distributions
def uniform_distribution(size):
    return np.random.uniform(low=0.0, high=1.0, size=size)

def exponential_distribution(size):
    return np.random.exponential(scale=1.0, size=size)

def bernoulli_distribution(size):
    return np.random.choice([0, 1], size=size)

def heavy_tail_distribution(size):
    return np.random.standard_cauchy(size=size)

def sample_custom_tail(alpha, size):
    u = np.random.uniform(low=0.0, high=1.0, size=size)
    s = np.sign(2 * u - 1)
    transformed = ( (np.pi / 2)**(1/(1+alpha)) * np.abs(2*u - 1) )**(1+alpha)
    x = s * (np.tan(transformed))**(1/(1+alpha))
    return x

spiky1 = lambda size: sample_custom_tail(alpha=0, size=size)
spiky2 = lambda size: sample_custom_tail(alpha=0.5, size=size)
spiky3 = lambda size: sample_custom_tail(alpha=1, size=size)
spiky4 = lambda size: sample_custom_tail(alpha=1.5, size=size)
spiky5 = lambda size: sample_custom_tail(alpha=2, size=size)
spiky6 = lambda size: sample_custom_tail(alpha=3, size=size)

Npts = 10000000

dir = Path(__file__).resolve().parents[2] / "work" / "promote" / "CLT_plots"
os.makedirs(dir, exist_ok=True)

# Example runs
for n in [1,2,3,4,5,10,20,30,40,50,100,500,1000]:
    plot_ideal=True
    if n == 1:
        plot_ideal=False
    clt_demo(spiky1, n=n, N=Npts, base_dir=dir, filename="Cauchy1_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 2", toggle_gaussian=plot_ideal, normalize=False)
    clt_demo(spiky2, n=n, N=Npts, base_dir=dir, filename="Cauchy2_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 2.5", toggle_gaussian=plot_ideal, normalize=False)
    clt_demo(spiky3, n=n, N=Npts, base_dir=dir, filename="Cauchy3_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 3", toggle_gaussian=plot_ideal, normalize=False)
    clt_demo(spiky4, n=n, N=Npts, base_dir=dir, filename="Cauchy4_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 3.5", toggle_gaussian=plot_ideal)
    clt_demo(spiky5, n=n, N=Npts, base_dir=dir, filename="Cauchy5_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 4", toggle_gaussian=plot_ideal)
    clt_demo(spiky6, n=n, N=Npts, base_dir=dir, filename="Cauchy6_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 5", toggle_gaussian=plot_ideal)
    clt_demo(uniform_distribution, n=n, N=Npts, base_dir=dir, filename="uniform_n"+str(n)+".png", title="Uniform Distribution", toggle_gaussian=plot_ideal)
    clt_demo(exponential_distribution, n=n, N=Npts, base_dir=dir, filename="exponential_n"+str(n)+".png", title="Exponential Distribution", toggle_gaussian=plot_ideal)
    clt_demo(bernoulli_distribution, n=n, N=Npts, base_dir=dir, filename="bernoulli_n"+str(n)+".png", title="Bernoulli Distribution", toggle_gaussian=plot_ideal)