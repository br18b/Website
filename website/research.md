---
layout: page
title: Turbulence
permalink: /research/
math: true
---

{% include hero.html
  eyebrow="Research"
  title="Turbulence"
  subtitle="Notes and visual explanations around supersonic turbulence, simulations, and density statistics."
%}

<div class="prose page-narrow" markdown=1>

Check out this <a href="{{ '/SPH_demo/hydro_sph_WASM.html' | relative_url }}">SPH demo of Rayleigh-Bénard instability</a> — violent lava lamp physics in browser form.

## Overview

{% include figure.html src="/pics/pillars.webp" alt="Pillars of Creation" caption="The Pillars of Creation. Courtesy of the James Webb Space Telescope." align="right" %}

Up above the clouds, inside swirling storms, and even across distant galaxies, **turbulence** is constantly shaping the universe.

Turbulence gives clouds their dramatic rolling forms. It is behind the dancing flickers of smoke from a candle. It is responsible for the chaotic gusts that shake tall chimneys — and it is a key player in the birth of stars and the evolution of galaxies.

But turbulence is not just beautiful and chaotic. It is a deeply complex phenomenon where energy sloshes from gigantic cosmic scales down to the tiniest eddies, following surprisingly elegant rules hidden inside the chaos.

To understand turbulence — what drives it, how it behaves, and how it influences the cosmos — we cannot just watch it. **We simulate it.**

Using large numerical simulations run on supercomputers, I study how turbulence moves, mixes, and structures itself across space. From the swirling soup of a star-forming cloud to the tangled magnetic fields in galaxies, I use simulations and statistical tools to make sense of the disorder.

## Computer simulations

To study turbulence, we simulate a small patch of the universe: just big enough to capture the swirling chaos we care about. We take a chunk of a molecular cloud, or a region of turbulent gas, and chop it into tiny cells like a 3D pixelated grid. Each cell stores physical quantities like density, velocity, and pressure.

{% include figure.html src="/pics/density_cube_with_grid.png" alt="Turbulence simulation" caption="A 3D simulation of turbulent density fields on a discretized grid." align="center" width="680px" %}

Then we unleash the laws of physics. Gravity, magnetism, shocks, turbulence — they all evolve naturally as the simulation runs. The computer calculates how each cell interacts with its neighbors, step by step, tracking the motion and behavior of the gas over thousands or millions of simulated years.

In the end, we do not just get pretty pictures, although we definitely do. We gain access to the hidden mechanics of turbulence: how energy flows, where structures form, and how complexity emerges from seemingly simple rules.

## Turbulent density statistics

One of the most fundamental quantities in turbulent gas is the density field. It is common in astrophysics for turbulent density to be modeled as **log-normal**: the logarithm of density looks statistically like a normal distribution.

But why? And how precisely can we model it?

In supersonic turbulence, much of the gas moves faster than the speed of sound. This motion is accompanied by shocks. The discontinuous density jump in an isothermal, supersonic medium can be described by

$$
\rho_2 = \mathcal{m}^2 \rho_1.
$$

Through an ensemble of many shocks, density variations accumulate multiplicatively, naturally leading to log-normal behavior through a central-limit-like argument for multiplicative processes. I unpack the intuition in the Gaussian series: [Part I: tossing dice]({{ '/blog/why-gaussian-part-i/' | relative_url }}), [Part II: continuous additive processes]({{ '/blog/why-gaussian-part-ii/' | relative_url }}), and [Part III: not all processes are additive]({{ '/blog/why-gaussian-part-iii/' | relative_url }}).

To be continued.

</div>
