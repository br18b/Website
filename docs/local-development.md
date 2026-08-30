# Local development

The repository reconstruction did not install or update dependencies. Use the
existing Ruby bundle only when the versions recorded by `Gemfile.lock` are available.

Production-equivalent final-layout build:

```bash
JEKYLL_ENV=production bundle exec jekyll build \
  --source website --config website/_config.yml --destination _site
```

Normal local preview:

```bash
bundle exec jekyll serve --source website --config website/_config.yml
```

Explicit draft preview:

```bash
bundle exec jekyll serve \
  --source website --config website/_config.yml --drafts
```

The production command and proposed Pages workflow do not pass `--drafts`.
`_site/` is ignored and must never be committed.
The repository-only `website/` prefix is not part of public URLs.
