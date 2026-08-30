# Legacy local-development evidence

This note preserves the useful intent of the pre-migration instructions while
removing their machine-specific checkout path and installation recipe. The
authoritative current commands are in `local-development.md`.

The legacy workflow used Bundler, served Jekyll from the repository root, and
wrote the generated site to `_site/`. It also recommended `--livereload` for
preview and `JEKYLL_ENV=production` for a production-like check. Those choices
remain compatible with the temporary root-layout phase.

Dependency installation and changes to `Gemfile` or `Gemfile.lock` are outside
Phase 2A. The existing lock file is preserved unchanged.
