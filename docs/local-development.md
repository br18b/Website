# Local development

Phase 2A did not install or update dependencies. Use the existing Ruby bundle
only when the versions recorded by `Gemfile.lock` are already available.

Production-equivalent root-layout build:

```bash
JEKYLL_ENV=production bundle exec jekyll build --source . --destination _site
```

Normal local preview:

```bash
bundle exec jekyll serve --source .
```

Explicit draft preview:

```bash
bundle exec jekyll serve --source . --drafts
```

The production command and proposed Pages workflow do not pass `--drafts`.
`_site/` is ignored and must never be committed.

The later layout command will use `--source website`; do not use that source
path until the root-layout Actions deployment and custom domain are verified.
