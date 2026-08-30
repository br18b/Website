---
layout: post
title: "Website Overhaul"
date: 2026-07-04 03:00:00 +0200
categories: updates
summary: "A quick note about the redesign and cleanup of my website."
---

The website finally got a proper makeover.

What started as a fairly plain GitHub Pages site has now been reorganized into something cleaner, nicer to use, and much easier to maintain. The goal was to turn it into something more intentional and maintainable: better spacing, nicer typography, cleaner cards, reusable layouts, and a more coherent structure across the whole site.

The biggest visible change is the new design. The homepage now has a clearer introduction, section cards for research, BeamNG projects, and writing, plus a cleaner preview of recent posts. Project and post cards now behave consistently: the whole card is clickable, the hover effect is shared, and only the title is highlighted instead of the entire text turning into one giant link.

Under the hood, the site is now much more modular. Repeated pieces of layout have been moved into reusable Jekyll includes, so I do not have to copy the same chunks of HTML everywhere. Project pages, galleries, stats boxes, figures, post lists, and cards can now be controlled from a smaller number of shared components. This should make future changes much less painful.

The BeamNG section also got reorganized into proper project-style pages. Instead of each page being a pile of custom markup, the structure is now more predictable: metadata lives in front matter, content lives in Markdown, and reusable includes handle the repeated visual pieces.

There were also a few quality-of-life fixes: cleaner local development with live reload, better handling of custom post summaries, improved card previews, less fragile styling, and a clearer separation between website content and local-only files like notebooks, raw data, and generated assets.

This is still a personal site, so it will keep evolving. But now it finally feels like a better foundation; flexible enough to grow, and just polished enough that I am no longer embarrassed to put it on a proper domain.
