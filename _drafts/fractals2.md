---
layout: post
title: "Fractals II: the misbehaving boundary"
permalink: /blog/fractals-part-ii/
date: 2026-07-17 12:00:00 +0200
categories: article
img_width: "100%"
featured: true
summary: "A closer look at the Mandelbrot boundary: escape potential, contour lines, self-similarity, and the strange behavior hiding between stability and escape."
math: true
code: true
lazy_iframe: true
---


In the <a href="{{ '/blog/fractals-part-i/' | relative_url }}">first part</a>, we met the Mandelbrot set through the bluntest possible question:

> does the orbit of 0 remain bounded, or not?

That gave us the black-and-white silhouette.

But the exterior is not uniform. Some points escape almost instantly. Others drift near the boundary for a long time before finally taking off. If we want a richer picture, we should not merely ask *whether* a point escapes. We should ask *how reluctant or eager it is to escape*.

## Escape-time coloring

The simplest enhancement is to color points by the number of iterations it takes for the orbit to exceed the escape radius.

That produces the familiar colorful Mandelbrot images: points far from the set escape quickly, while points near the boundary escape more slowly and form layered halos.

This immediately reveals more structure than black and white alone.

<figure style="text-align: center;">
    <img src="{{ '/' | relative_url }}fractal/mandelbrot/mandelbrot_color_1024.png"
         alt="Colored Mandelbrot set"
         style="width: {{ page.img_width }}; display: block; margin: auto;"/>
    <figcaption style="font-size: smaller; color: gray;">
        Escape-time coloring of the Mandelbrot set. The exterior color records how quickly the orbit escapes.
    </figcaption>
</figure>

Raw escape-time coloring has one familiar weakness, though: it creates visible bands. Points that escape after 30 iterations get one color, points that escape after 31 iterations get another, and the transitions are abrupt.

A smoother quantity would provide a nicer aesthetic experience.

## Escape potential

For a point $c$ outside the Mandelbrot set, the orbit eventually becomes large enough that the rule

<div class="math-center">
$$
z_{n+1}=z_n^2+c
$$
</div>

is dominated by the squaring term. Once that happens, the growth is approximately repeated squaring:

<div class="math-center">
$$
z_{n+k}\approx z_n^{2^k}.
$$
</div>

Taking logarithms suggests the correct normalization:

<div class="math-center">
$$
G(c)=\lim_{n\to\infty}\frac{1}{2^n}\log |z_n|.
$$
</div>

This quantity is called the **escape potential**.

Inside the Mandelbrot set, the orbit remains bounded, so the potential is zero:

<div class="math-center">
$$
G(c)=0 \qquad \text{inside the set.}
$$
</div>

Outside the set, it is positive:

<div class="math-center">
$$
G(c)>0 \qquad \text{outside the set.}
$$
</div>

So $G(c)$ behaves like a height field over the exterior of the Mandelbrot set.

Far away, the potential is large.  
Near the boundary, it tends to zero.

## Coloring with the potential

This lets us replace banded iteration-count coloring with something smoother and more geometric.

The escape potential turns the exterior into a continuous scalar field. Instead of seeing “this point escaped after 53 steps,” we see “this point lies on a certain escape-height landscape.”

That is the first real hint that the outside of the Mandelbrot set can be treated as geometry, not just as an image.

## Equipotential contours

Once a scalar field exists, its level sets become natural objects to draw.

The curves

<div class="math-center">
$$
G(c)=\text{constant}
$$
</div>

are exterior contour lines.

They wrap around the Mandelbrot set like topographic contours around a mountain — except the zero-height object they approach is a fractal boundary.

At large $G$, the contours are broad and smooth.  
At smaller $G$, they tighten around the set and begin to discover more and more of its fine structure.

{% include lazy-iframe.html
   label="Interactive demo"
   title="Escape-potential contours around the Mandelbrot set"
   src="/fractal/mandelbrot/demos/boundary/index.html"
   preview_src="/fractal/mandelbrot/demos/boundary/preview.png"
   preview_alt="Preview of a Mandelbrot escape-potential contour demo"
   button_text="Load demo"
   loading_text="Loading contour demo…"
   height="760px"
   mobile_height="520px"
   fit_content=true
   fit_min_height="360"
   fit_max_height="1100"
   note="Precomputed interactive demo."
   caption="Exterior equipotential contours approach the Mandelbrot boundary as $G\to0^+$."
%}

This is a beautiful way to think about the boundary:

> the Mandelbrot boundary is the limit of smooth exterior contour lines as the smoothing parameter $G$ tends to zero.

The contour at every positive $G$ is still a perfectly ordinary smooth curve. The fractal only appears in the limit.

## Can we estimate the area from outside?

A tempting idea is now obvious.

If the contour $G(c)=\text{constant}$ encloses the Mandelbrot set together with an exterior collar, then perhaps we can:

1. compute the enclosed area for smaller and smaller $G$,
2. and extrapolate to $G=0$.

In principle, that is completely reasonable.

In practice, the limit is stubborn.

<figure style="text-align: center;">
    <img src="{{ '/' | relative_url }}fractal/mandelbrot/scaling_plots/area_vs_G.png"
         alt="Area enclosed by equipotential contours"
         style="width: 100%;">
    <figcaption style="font-size: smaller; color: gray;">
        Area enclosed by the equipotential contour. Smaller values of $G$ correspond to contours closer to the Mandelbrot set.
    </figcaption>
</figure>

The outer contours lose area quickly. Then the curve flattens frustratingly. Even after many orders of magnitude in $G$, the enclosed area is still drifting rather than settling sharply.

The reason is geometric: a tiny potential does not mean uniform Euclidean closeness. The contour may still bridge over deep fjords and narrow channels that only become resolved much later.

So the experiment is illuminating — but not yet decisive.

## Length, curvature, and the cost of resolving the boundary

Area is not the only quantity we can track.

As the contour approaches the boundary, its **length** first decreases, then begins to increase again as it is forced to wrap around more and more fine structure.

<figure style="text-align: center;">
    <img src="{{ '/' | relative_url }}fractal/mandelbrot/scaling_plots/length_vs_G.png"
         alt="Length of equipotential contours"
         style="width: 100%;">
    <figcaption style="font-size: smaller; color: gray;">
        Equipotential length first contracts, then rises as the contour begins to resolve the fractal geometry.
    </figcaption>
</figure>

Its **curvature** also becomes more extreme:

<figure style="text-align: center;">
    <img src="{{ '/' | relative_url }}fractal/mandelbrot/scaling_plots/curvature_quantiles_vs_G.png"
         alt="Curvature quantiles of equipotential contours"
         style="width: 100%;">
    <figcaption style="font-size: smaller; color: gray;">
        High-curvature regions become more and more common as the contour approaches the Mandelbrot boundary.
    </figcaption>
</figure>

And the **computational cost** rises rapidly too:

<figure style="text-align: center;">
    <img src="{{ '/' | relative_url }}fractal/mandelbrot/scaling_plots/points_vs_G.png"
         alt="Point count required to resolve equipotential contours"
         style="width: 100%;">
    <figcaption style="font-size: smaller; color: gray;">
        Number of traced points required to resolve the contour at each value of $G$.
    </figcaption>
</figure>

So the boundary teaches the same lesson from several directions at once:

> every time we remove another layer of smoothing, another scale of structure becomes important.

## Self-similarity and distorted echoes

At the boundary, the Mandelbrot set also begins doing its most famous trick.

It repeats itself — not by perfect copy-paste, but as warped and decorated echoes.

Tiny Mandelbrots appear attached to filaments. Valleys open into smaller valleys. Spirals breed more spirals.

This self-similarity is one reason the contour experiment behaves so stubbornly. The geometry does not simply become “a little more detailed.” It keeps inventing new generations of detail.

Here is where a few good zoom images belong.

## What we learned

Part I introduced the Mandelbrot set as a yes/no sorting machine.

Part II upgrades the picture.

The exterior is not empty. It carries a smooth scalar field, the escape potential.  
The boundary can be approached by smooth equipotential contours.  
But the closer those contours get, the more they are forced to reveal a boundary that refuses to become a simple curve.

In the next part, we turn back inward.

Instead of approaching the Mandelbrot set from outside by smooth envelopes, we will approach it from inside by cataloguing the stable periodic components — the bulbs, satellites, and tiny hidden islands of order that fill its interior.