# Data and generated-asset provenance

`data/source/turbulence/cube0090_density.fits` is an irreplaceable product of an
old supercomputer turbulence simulation. The repository cannot regenerate it.
Its fixed SHA-256 is recorded in `data/manifests/source-data.sha256`; it is
tracked normally and must not be moved to Git LFS.

The currently public Mandelbrot boundary demo consists of fifteen runtime and
preview files. Their hashes and producer are recorded in
`data/manifests/generated-assets.tsv`, including both public URLs and repository
paths below `website/`. The committed `README.md` beside those files is
provenance documentation, not part of the browser closure.

The unfinished Fractals II draft references the boundary closure and five
preserved scaling-plot candidates. None is referenced by a published page.
They are deliberately absent from ordinary production source and retained only
under ignored `work/promote/mandelbrot/scaling_plots/`. Their hashes and future
promotion mapping are tracked in
`data/manifests/fractals2-draft-assets.tsv`.

Generated does not mean disposable: public outputs remain tracked when they
are intentional and reviewed. Private calculation state remains outside Git,
with the immutable source snapshot and the root-layout checkpoint branch
serving as local rollback points.
