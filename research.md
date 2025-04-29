---
layout: page
title: Turbulence
permalink: /research/
---

<div style="display: flex; align-items: flex-start; margin-left: -250px; margin-right: 0px;">
    <aside class="sticky-sidebar">
    <h3>Quick Navigation</h3>
    <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#simulations">Computer Simulations</a></li>
        <li>
        <a href="#density-stats">Statistics of Density</a>
        <ul>
            <li><a href="#clt">Central Limit Theorem</a></li>
        </ul>
        </li>
    </ul>
    </aside>
    <div style="flex: 1; text-align: justify;">
        <h2 id="overview">Overview</h2>
        <p>
            Up above the clouds, inside swirling storms, and even across distant galaxies, <strong>turbulence</strong> is constantly shaping the universe.
        </p>
        <figure style="float: right; text-align: left; width: 320px; margin: 0 0 10px 10px;">
            <img src="{{ site.baseurl }}/pics/pillars.webp" alt="Pillars of Creation" style="width: 100%;">
            <figcaption style="font-size: smaller; color: gray;">The Pillars of Creation. Courtesy of the James Webb Space Telescope.</figcaption>
        </figure>
        <p>
            Turbulence is what gives clouds their dramatic, rolling forms. It's behind the dancing flickers of smoke from a candle. It's responsible for the chaotic gusts that shake tall chimneys - and it's a key player in the birth of stars and the evolution of entire galaxies.
        </p>
        <p>
            But turbulence isn't just beautiful and chaotic. It's a deeply complex phenomenon where energy sloshes from gigantic cosmic scales down to the tiniest eddies, following surprisingly elegant rules hidden inside the chaos.
        </p>
        <p>
            To understand turbulence - what drives it, how it behaves, and how it influences the cosmos - we can't just watch it. <strong>We simulate it.</strong>
        </p>
        <p>
            Using giant numerical simulations run on supercomputers, I study how turbulence moves, mixes, and structures itself across space.
            From the swirling soup of a star-forming cloud to the tangled magnetic fields in galaxies, I use simulations and statistical tools to make sense of the disorder.
        </p>
        <p>
            In my PhD work, I focused on the basic statistical building blocks of turbulent, supersonic media.  
            Now, I'm diving deeper into the filamentary structures that emerge when magnetic fields and turbulent gas collide - the strange, beautiful strands that may hold the keys to how stars are born.
        </p>
        <h2 id="simulations">Computer Simulations</h2>
        <p>
        To study turbulence, we simulate a small patch of the universe - just big enough to capture the swirling chaos we care about. We take a chunk of a molecular cloud, or a region of turbulent gas, and chop it into tiny cells like a 3D pixelated grid. Each cell stores physical quantities like density, velocity, and pressure, giving us a discrete map of a chaotic, flowing medium.
        </p>
        <figure style="text-align: center;">
            <img src="{{ site.baseurl }}/pics/density_cube_with_grid.png" alt="Turbulence simulation" width="600"/>
            <figcaption style="font-size: smaller; color: gray;">A 3D simulation of turbulent density fields in a molecular cloud. Tiny cubes added to visualize spatial discretization.</figcaption>
        </figure>
        <p>
            Then we unleash the laws of physics. Gravity, magnetism, shocks, turbulence - they all evolve naturally as the simulation runs. The computer calculates how each cell interacts with its neighbors, step by step, tracking the motion and behavior of the gas over thousands or millions of simulated years.
        </p>
        <p>
            In the end, we don’t just get pretty pictures (although we definitely do). We gain access to the hidden mechanics of turbulence: how energy flows, where structures form, and how complexity emerges from seemingly simple rules.
        </p>
        <p>
            After enough staring at plots, arguing with our own statistics, and rerunning simulations because of a forgotten boundary condition, we sometimes find just enough order hiding inside the chaos to write a paper about it.
        </p>
        <p>
            It's a little like cosmic archaeology - digging through simulated clouds to understand the forces that shape the stars.
        </p>
        <h2 id="density-stats">Turbulent Density Statistics</h2>
        <p>
            The simulations would be useless if we only stared at the pretty pictures. Sure, the visuals are amazing - but science demands more than observation; it demands prediction and explanation.
        </p>
        <p>
            One of the most fundamental quantities in turbulent gas is the density field. It’s common knowledge in astrophysics that turbulent density tends to follow a <strong>log-normal distribution</strong>: logarithm of density looks statistically like a normal (Gaussian) distribution.
        </p>
        <p>
            But why? And how precisely can we model it?
        </p>
        <p>
            To understand how we can improve on the existing model, first, we should justify the ideal log-normal statistics of density in a supersonic, turbulent medium. In supersonic turbulence, most of the gas is moving at speed exceeding the speed of sound. This motion, however, is accompanied by shocks traversing the medium. A shock is a mechanism in which a high speed, high density gas propagates into a region with lower density. The discontinuous jump in density in isothermally turbulent, supersonic medium, can be described by the following condition
        </p>
        <p style="text-align: center;">
            $$
                \rho_2 = \mathcal{m}^2 \rho_1
            $$
        </p>
        <p>
            Through an ensemble of many shocks, density variations accumulate multiplicatively, naturally leading (via the central limit theorem (CLT) for multiplicative processes) to a log-normal behavior. If the last sentence did not make much sense, we can perform a numerical experiment on our computer using Python.
        </p>
        <h4 id="clt">Detour: Central Limit Theorem</h4>
        <p>
            First, we take a <strong>source distribution</strong> - it can be anything, not too pathological (it should have well-defined mean and variance), $\mathcal{D} (x)$. Starting easy, we can verify the CLT for sum of random variables: in one go, we generate $n$ random numbers from distribution $\mathcal{D}$ and sum them together,
        </p>
        <p style="text-align: center;">
            $$
                X = \sum_{i = 1}^n x_i, \quad x_i \sim \mathcal{D}
            $$
        </p>
        <p>
            Then, we take <strong>ensemble</strong> of $N$ samples $(X_1, X_2, \dots, X_N)$ and look at how they're distributed. Since the overall mean and variance might, over many samples $n$, drift far away from the origin (especially if the source distribution does not have zero mean), we <strong>renormalize</strong> the samples by subtracting the ensemble mean and dividing by variance:
        </p>
        <p style="text-align: center;">
            $$
                Y_k = \frac{X_k - \mu_X}{\sigma_X}, \quad \mu_X = \frac{1}{N} \sum_{k = 1}^N X_k, \quad \sigma_X^2 = \frac{1}{N} \sum_{k = 1}^N \left( X_k - \mu \right)^2
            $$
        </p>
        <p>
            Now we (or rather the central limit theorem) claim that the distribution of $Y_k$ will tend to the normal distribution if $n$ is large enough. Let's verify this using Python! We start by importing all the required modules:
        </p>
        <pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
import os</code></pre>
        <p>
            Here numpy is required to generate the random samples, matplotlib for making neat plots and os for I/O. Having done that, we now proceed with the function that will generate the samples, renormalize them, create histograms and save the histograms. Thinking ahead, we know we will need several inputs: the random distribution, $n$, $N$. Additionally, we will also control the number of bins to create the resulting distribution. We can include more arguments to make plotting more modular, but ultimately, the core of the function is captured with the first 4 arguments.
        </p>
        <pre><code class="language-python">def clt_demo(
    dist_func,     # source random distribution
    n=5,           # baseline number of random variables to sum
    N=10000,       # size of the ensemble
    bins=50,       # number of bins for visualization
    base_dir="",
    title="CLT Demo",
    filename="plot.png"
    bounds=[-10,10]  # cutoff if the distribution is weird
):</code></pre>
        <p>
            The neat thing about Python is its vectorization capabilities; almost all built-in functions are capable of taking in an array of arbitrary shape which then results in output of the same shape, where the function is applied to each element. We can utilize this by generating $N \times n$ values all at once in a big $N \times n$ array, which we then sum over the last index. Specifically,
        </P>
        <pre><code class="language-python">samples = dist_func(size=(N, n))
sums = np.sum(samples, axis=1)</code></pre>
        <p>
            For really high $n$ and $N$, we would implement a different strategy where the sum is being performed on the fly to save the memory. Having done this, the rest of the code simply normalizes the ensemble, plots the histogram and saves it. Full code with some examples:
        </p>
        <button onclick="toggleCode()" style="margin-bottom: 10px;">Show Code</button>
<div id="codeBlock" style="display: none;">
  <pre><code class="language-python">
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

def clt*demo(
dist_func, # function to draw random samples
n=5, # number of random variables to sum
N=10000, # how many sums to create
bins=100, # number of bins for the histogram
base_dir="",
title="CLT Demo",
filename="plot.png", # filename to save
bounds=[-4.5,4.5] # cutoff if the distribution is weird
): # create the ensemble
sums = np.zeros(N)
for * in tqdm(range(n), desc="Adding random samples"):
sums += dist_func(size=N)

    # mean and variance
    mean = np.mean(sums)
    std = np.std(sums)

    # normalize ensemble
    normalized = (sums - mean) / std

    normalized = normalized[(normalized > bounds[0]) &
     (normalized < bounds[1])]

    # plot and save
    plt.figure(figsize=(8, 5))

    # histogram
    counts, bins_edges, _ = plt.hist(
        normalized, bins=bins, density=True, alpha=0.7,
        color="skyblue", edgecolor="black", label="Ensemble histogram"
    )

    # overplot ideal normal distribution
    x = np.linspace(bounds[0], bounds[1], 1000)
    y = norm.pdf(x, loc=0, scale=1)  # zero mean, unit variance
    plt.plot(x, y, 'k--', label="Ideal Normal Distribution")

    plt.title(f"{title}\n(n={n}, N={N})")
    plt.xlabel("Normalized Sum")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()

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

def sample*custom_tail(alpha, size):
u = np.random.uniform(low=0.0, high=1.0, size=size)
s = np.sign(2 * u - 1)
transformed = ( (np.pi / 2)\*\*(1/(1+alpha)) \_ np.abs(2*u - 1) )\*\*(1+alpha)
x = s * (np.tan(transformed))\*\*(1/(1+alpha))
return x

spiky1 = lambda size: sample_custom_tail(alpha=0, size=size)
spiky2 = lambda size: sample_custom_tail(alpha=0.5, size=size)
spiky3 = lambda size: sample_custom_tail(alpha=1, size=size)
spiky4 = lambda size: sample_custom_tail(alpha=1.5, size=size)
spiky5 = lambda size: sample_custom_tail(alpha=2, size=size)
spiky6 = lambda size: sample_custom_tail(alpha=3, size=size)

Npts = 10000000

dir = 'CLT_plots'
os.makedirs(dir, exist_ok=True)

# Example runs

for n in [1,2,3,4,5,10,50,100,1000]:
clt_demo(spiky1, n=n, N=Npts, base_dir=dir,
filename="Cauchy1_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 2")
clt_demo(spiky2, n=n, N=Npts, base_dir=dir,
filename="Cauchy2_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 2.5")
clt_demo(spiky3, n=n, N=Npts, base_dir=dir,
filename="Cauchy3_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 3")
clt_demo(spiky4, n=n, N=Npts, base_dir=dir,
filename="Cauchy4_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 3.5")
clt_demo(spiky5, n=n, N=Npts, base_dir=dir,
filename="Cauchy5_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 4")
clt_demo(spiky6, n=n, N=Npts, base_dir=dir,
filename="Cauchy6_n"+str(n)+".png", title="Gen. Cauchy, alpha ~ 5")
clt_demo(uniform_distribution, n=n, N=Npts, base_dir=dir,
filename="uniform_n"+str(n)+".png", title="Uniform Distribution")
clt_demo(exponential_distribution, n=n, N=Npts, base_dir=dir,
filename="exponential_n"+str(n)+".png", title="Exponential Distribution")
clt_demo(bernoulli_distribution, n=n, N=Npts, base_dir=dir,
filename="bernoulli_n"+str(n)+".png", title="Bernoulli Distribution")
</code></pre>

</div>

<script>
function toggleCode() {
  var x = document.getElementById("codeBlock");
  var button = event.target;
  if (x.style.display === "none") {
    x.style.display = "block";
    button.innerText = "Hide Code";
  } else {
    x.style.display = "none";
    button.innerText = "Show Code";
  }
}
</script>

        <p>
            We can now look at some examples. We start with the uniform baseline distribution that generates numbers between 0 and 1 with equal probability. With that, the expected histogram for $n = 1$ is flat:
        </p>
        <figure style="text-align: center;">
            <img src="{{ site.baseurl }}/CLT_plots/uniform_n1.png" alt="Uniform Distribution" width="100%"/>
            <figcaption style="font-size: smaller; color: gray;">Uniform Distribution.</figcaption>
        </figure>
        <p>
            The range changed due to the normalization to zero mean and unit variance. Now, if we increase $n$ to $2, 3, 4, 5$, we get closer and closer to the Gaussian distribution:
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <figure style="margin: 0;">
                <img src="{{ site.baseurl }}/CLT_plots/uniform_n2.png" alt="Plot 1" style="width: 100%;">
            </figure>
            <figure style="margin: 0;">
                <img src="{{ site.baseurl }}/CLT_plots/uniform_n3.png" alt="Plot 2" style="width: 100%;">
            </figure>
            <figure style="margin: 0;">
                <img src="{{ site.baseurl }}/CLT_plots/uniform_n10.png" alt="Plot 3" style="width: 100%;">
            </figure>
            <figure style="margin: 0;">
                <img src="{{ site.baseurl }}/CLT_plots/uniform_n50.png" alt="Plot 4" style="width: 100%;">
            </figure>
        </div>
        <p>
            Convergence towards a distribution is a very rich and interesting topic; not all distributions will lead to the Gaussian (a heavy-tailed Cauchy distribution will converge to another Cauchy distribution, for example), but as long as the distribution is "well-behaved", convergence is assured. Berry-Esseen's theorem guarantees a relatively slow convergence (the difference between Gaussian and the distribution after summing $n$ terms is on the order of $1/\sqrt{n}$) provided the third moment $\langle x^3 \rangle$ exists and is finite. We saw that the convergence for the uniform distribution is rather quick; this is because the third moment is actually zero, in which case the convergence is slightly faster ($1/n$).
        </P>
    </div>

</div>
