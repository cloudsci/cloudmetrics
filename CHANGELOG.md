# Changelog

## [Unreleased](https://github.com/cloudsci/cloudmetrics/tree/HEAD)

[Full Changelog](https://github.com/cloudsci/cloudmetrics/compare/v0.1.0...HEAD)

*new features*

- Refactor iorg metric to use object-labels as argument (now available as
  `cloudmetrics.mask.iorg_objects` and `cloudmetrics.objects.iorg`) and combine
  methods using either a) inhibition nearest-neighbour or b) Poisson reference
  distribution into single function taking `reference_dist` as argument.
  [\#54](https://github.com/cloudsci/cloudmetrics/pull/54) By Leif Denby
  (@leifdenby)

- Refactor SCAI metric (now available as `cloudmetrics.objects.scai` and
  `cloudmetrics.mask.scai_objects`) and add test which accounts for change in
  SCAI value with resolution
  [\#42](https://github.com/cloudsci/cloudmetrics/pull/42)
  By Leif Denby & Martin Janssens (@leifdenby & @martinjanssens)

- Add metrics for statistical reductions of scalar fields (either globally or
  masked) available as `cloudmetrics.scalar.mean`, `cloudmetrics.scalar.var`,
  `cloudmetrics.scalar.skew` and `cloudmetrics.scalar.kurtosis`
  [\#46](https://github.com/cloudsci/cloudmetrics/pull/46) By Leif Denby &
  Martin Janssens (@leifdenby & @martinjanssens)

- Refactored Convective Organisation Potential (COP) metric, now available in
  `cloudmetrics.objects.cop` and `cloudmetrics.mask.cop_objects`
  [\#41](https://github.com/cloudsci/cloudmetrics/pull/41) By Leif Denby &
  Martin Janssens (@leifdenby & @martinjanssens)

- Refactored metrics into common api based on what input each operates on, so
  that metrics are now in submodules called `cloudmetrics.mask`,
  `cloudmetrics.objects` and `cloudmetrics.scalar`.
  [\#39](https://github.com/cloudsci/cloudmetrics/pull/39). By Leif Denby
  (@leifdenby)

- Refactored cloud-object metrics that cmopute geometry metrics of labelled
  objects [\#23](https://github.com/cloudsci/cloudmetrics/pull/23). By Leif
  Denby (@leifdenby)

- Change build setup to use `setup.cfg` and setup linting using
  [pre-commit](https://pre-commit.com/#usage).
  [\#38](https://github.com/cloudsci/cloudmetrics/pull/32). By Leif Denby
  (@leifdenby)

- Refactored wavelet organisation indices, now available as
  `woi1=cloudmetrics.woi1(...)`, `woi2=cloudmetrics.woi2(...)` and
  `woi3=cloudmetrics.woi3(...)`
  [\#32](https://github.com/cloudsci/cloudmetrics/pull/32). By Leif Denby
  & Martin Janssens (@leifdenby & @martinjanssens)

- Refactored orientation metric calculation, now available as
  `cloudmetrics.orientation(...)`
  [\#24](https://github.com/cloudsci/cloudmetrics/pull/24). By Leif Denby
  & Martin Janssens (@leifdenby & @martinjanssens)

- Refactored "open sky" (clear sky) metric calculation, now available as
  `cloudmetrics.open_sky(...)`
  [\#22](https://github.com/cloudsci/cloudmetrics/pull/22). By Leif Denby
  & Martin Janssens (@leifdenby & @martinjanssens)

- Refactored iorg metric calculation, now available as `cloudmetrics.iorg(...)`
  [\#21](https://github.com/cloudsci/cloudmetrics/pull/21). By Leif Denby
  & Martin Janssens (@leifdenby & @martinjanssens)

- Refactored spectral metric calculation, now available to calculate all
  spectral metrics as `cloudmetrics.scalar.compute_all_spectral(...)`
  [\#30](https://github.com/cloudsci/cloudmetrics/pull/30),
  [\#36](https://github.com/cloudsci/cloudmetrics/pull/36). By Leif Denby
  & Martin Janssens (@leifdenby & @martinjanssens)

- Add tests for producing spatially-continuous object masks and for computing
  nearest-neighbor distances on periodic domains
  [\#61](https://github.com/cloudsci/cloudmetrics/pull/61). @leifdenby &
  @martinjanssens

*maintenance*

- update `black` version to address recent breaking change in black's
  dependency on `click`
  [\#59](https://github.com/cloudsci/cloudmetrics/pull/59) By Leif Denby
  (@leifdenby)

- ci action to automatically deploy releases on github to pypi
  [\#53](https://github.com/cloudsci/cloudmetrics/pull/53). By Leif Denby
  (@leifdenby)

- Code cleanup and setup of continuous integration testing
  [\#19](https://github.com/cloudsci/cloudmetrics/pull/19),
  [\#55](https://github.com/cloudsci/cloudmetrics/pull/55). By Leif Denby
  (@leifdenby)


## [v0.1.0](https://github.com/cloudsci/cloudmetrics/releases/tag/v0.1.0)

Metric calculation implementations as used in M. Janssens et al 2021
(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091001)
