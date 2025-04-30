---
layout: page
title: Turbulence
permalink: /research/
---

<div class="page-container">
    <aside class="sticky-sidebar desktop-only">
    <h3>Quick Navigation</h3>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#simulations">Computer Simulations</a></li>
            <li>
                <a href="#density-stats">Statistics of Density</a>
            </li>
        </ul>
    </aside>
    <h1 id="overview">Turbulence</h1>
    <aside class="sticky-sidebar mobile-only">
    <h3>Quick Navigation</h3>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#simulations">Computer Simulations</a></li>
            <li>
                <a href="#density-stats">Statistics of Density</a>
            </li>
        </ul>
    </aside>
<!-- Main Content -->
    <main class="main-content">
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
            <figcaption style="font-size: smaller; color: gray;">A 3D simulation of turbulent density fields on a discretized grid.</figcaption>
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
        <h2 id="density-stats">Turbulent Density Statistics</h2>
        <p>
            The simulations would be useless if we only stared at the pretty pictures. Sure, the visuals are amazing, but science demands more than observation; it demands prediction and explanation.
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
            Through an ensemble of many shocks, density variations accumulate multiplicatively, naturally leading, via the central limit theorem (CLT) for multiplicative processes, to a log-normal behavior. To see how and why, I provide some justification in my blog posts, <a href="{{ '/blog/why-gaussian-part-i/' | relative_url }}">Why Gaussian? Part I: tossing dice</a>, <a href="{{ '/blog/why-gaussian-part-ii/' | relative_url }}">Why Gaussian? Part II: continuous additive processes</a>.
        </p>
        <p>
            To be continued...
        </p>
    </main>
</div>
