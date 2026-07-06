---
layout: page
title: About
permalink: /about/
---

{% capture cv_actions %}
  <div class="cv-download" aria-label="View CV">
    <span class="cv-download__label">View CV</span>
    <span class="cv-download__links">
      <a class="button cv-download__button"
         href="{{ '/assets/pdf/CV_SK.pdf' | relative_url }}"
         target="_blank"
         rel="noopener">
        SK
      </a>
      <a class="button cv-download__button"
         href="{{ '/assets/pdf/CV_EN.pdf' | relative_url }}"
         target="_blank"
         rel="noopener">
        EN
      </a>
    </span>
  </div>
{% endcapture %}

{% include hero.html
  eyebrow="About"
  title="Who am I?"
  subtitle="Physicist-turned-data analyst with a penchant for mathematics, numerical simulations, and vintage cars."
  actions=cv_actions
%}

<div class="prose page-narrow" markdown="1">

Hi! I'm Braňo — a physicist-turned-data analyst with a soft spot for simulations, statistics, and the occasional automotive rabbit hole. This site is where I collect the things I have worked on, written about, or fallen into for suspiciously long periods of time.

I spend a lot of time wrestling with supersonic turbulence, writing analysis code, and creating BeamNG car mods. Somewhere in that mix are the things I keep coming back to: messy systems, visual explanations, data, and the strange satisfaction of making a computer finally do the thing I wanted.

Outside this site, I work as a data analyst on data-quality, validation, and reporting workflows for complex public-sector information systems. The details are not public, but the themes are familiar: APIs, messy data, metadata, consistency checks, reproducible pipelines, and reports that make hidden problems visible.

When I'm not buried in turbulence statistics or tinkering with BeamNG mods, you might find me cycling, playing ultimate frisbee, or singing karaoke.

Stick around for a weird mix of physics, code, data, and cars.

</div>
