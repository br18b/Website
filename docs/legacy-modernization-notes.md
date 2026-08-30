# Website modernization notes

This bundle keeps the site as Jekyll, but makes the repeated pieces reusable.

## Main structural changes

- BeamNG project pages moved from hand-written `BNG/*.md` pages to the `_projects/` collection.
- The visible project URLs are preserved by front matter permalinks:
  - `/BNG/S1203/`
  - `/BNG/S92a/`
  - `/BNG/orb/`
- `beam.md` now renders cards automatically from `site.projects`.
- Reusable includes were added for:
  - `hero.html`
  - `card.html`
  - `project-card.html`
  - `project-grid.html`
  - `figure.html`
  - `gallery.html`
  - `stats.html`
  - `post-list.html`
  - `plot-carousel.html`
  - `youtube.html`
- CSS is split into `_tokens`, `_base`, `_layout`, `_components`, `_syntax`, and `_utilities`.
- MathJax, Highlight.js, gallery JS, and carousel JS load only when front matter opts into them.

## Files to remove from the old repo after applying this

The old hand-written BeamNG pages should be removed because the new `_projects/` collection outputs the same public URLs:

```bash
rm -f BNG/S1203.md BNG/S92a.md BNG/orb.md
```

Keep your image folder:

```bash
BNG/pics/
```

## Image folders to copy/keep

This skeleton deliberately excludes images. Keep these folders from your original repo:

```text
BNG/pics/
pics/
CLT_plots/
fractal/textures/        # only if publicly used
fractal/paths/           # only if publicly used by the fractal demo
```

Keep notebooks, FITS files, raw simulation data, and other working artifacts outside the repo or in ignored folders such as `notebooks/`, `raw/`, or `content-sources/`.
