# cloudmetrics

[![cloudmetrics](https://github.com/cloudsci/cloudmetrics/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/cloudsci/cloudmetrics/actions/workflows/python-package-conda.yml) [![DOI](https://zenodo.org/badge/279602981.svg)](https://zenodo.org/badge/latestdoi/279602981)

The `cloudmetrics` package contains python routines to compute metrics
from 2D cloud fields to characterise cloud patterns in these fields. Most
methods operate on a `cloud-mask` (i.e. a boolean true-false field)
indicating where clouds exist, some work on individually labelled (with a
unique integer ID) cloud objects (which can be produced from a cloud-mask) and
finally some work on 2D cloud scalar-fields (defining for example the
cloud-liquid water or cloud-top height).

> **NOTE**: the `cloudmetrics` package contained in this repository is
> refactored from work published in [Janssens et al
> 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091001),
> this was done to make the routines implemented more accessible by external
> tools and to ensure consistency. Not all functionality has been retained or
> fully refactored, in particular functionality to allow for bulk-processing of
> input files is in the
> [cloudmetrics-pipline](https://github.com/cloudsci/cloudmetrics-pipeline)
> package. Progress on the refactoring can be followed in issue
> https://github.com/cloudsci/cloudmetrics/issues/20. The version published in
> Janssens et al 2021 is [available tagged as version
> v0.1.0](https://github.com/cloudsci/cloudmetrics/tree/v0.1.0).

## Implemented metrics

The table below gives an overview over which metrics are avaiable in the
`cloudmetrics` package and what input each metric takes.


| function within `cloudmetrics`     | `mask`   | `object_labels` | `scalar_field` |
| ---------------------------------- | -------- | --------------- | -------------- |
| `mask.cloud_fraction`              | ✔️        |                 |                |
| `mask.fractal_dimension`           | ✔️        |                 |                |
| `mask.open_sky`                    | ✔️        |                 |                |
| `mask.orientation`                 | ✔️        |                 |                |
| `mask.network_nn_dist`             | TODO     |                 |                |
| `mask.cop`                         | ✔️†       | ✔️               |                |
| `mask.csd`                         | TODO     | TODO            |                |
| `objects.iorg`                     | ✔️†       | ✔️               |                |
| `objects.max_length_scale`         | ✔️†       | ✔️               |                |
| `objects.mean_eccentricity`        | ✔️†       | ✔️               |                |
| `objects.mean_length_scale`        | ✔️†       | ✔️               |                |
| `objects.mean_perimeter_length`    | ✔️†       | ✔️               |                |
| `objects.rdf`                      | TODO     | TODO            |                |
| `objects.scai`                     | ✔️†       | ✔️               |                |
| `scalar.spectral_anisotropy` #1    |          |                 | ✔️              |
| `scalar.spectral_length_median`#1  |          |                 | ✔️              |
| `scalar.spectral_length_moment,`#1 |          |                 | ✔️              |
| `scalar.spectral_slope`#1          |          |                 | ✔️              |
| `scalar.spectral_slope_binned`#1   |          |                 | ✔️              |
| `scalar.woi1`                      |          |                 | ✔️              |
| `scalar.woi2`                      |          |                 | ✔️              |
| `scalar.woi3`                      |          |                 | ✔️              |
| `scalar.mean`                      | optional |                 | ✔️              |
| `scalar.var`                       | optional |                 | ✔️              |
| `scalar.std`                       | optional |                 | ✔️              |
| `scalar.skew`                      | optional |                 | ✔️              |
| `scalar.kurtosis`                  | optional |                 | ✔️              |

†: for convenience object-based scalars are also made avaiable to operate
directly on masks, for example `objects.max_length_scale(object_labels=...)`
can be called with a mask as `mask.max_object_length_scale(mask=...)` and
`objects.iorg(object_labels=...)` can be called with
`mask.iorg_objects(mask=...)`.

#1: spectral metrics currently operate on the relevant power spectral densities,
which must first be computed:
```
wavenumbers, psd_1d_radial, psd_1d_azimuthal = scalar.compute_spectra(...)
spectral_length_moment = scalar.spectral_length_moment(wavenumbers, psd_1d_radial)
```
Alternatively, all spectral metrics can be computed simultaneously following the
standard convention with `spectral_metrics = scalar.compute_all_spectral(scalar_field)`.
need refactoring to take `scalar_field` as input

# Installation

To install the most recent version of `cloudmetrics` from pypi you can use `pip`:

```bash
$> pip install cloudmetrics
```

If you plan to add/modify `cloudmetrics` (contribution via pull-requests are
very welcome!) you should check out the [development
notes](https://github.com/cloudsci/cloudmetrics/blob/master/docs/developing.md)
for how to get set up with a local copy of the codebase.

# Usage

To use the `cloudmetrics` package simply import `cloudmetrics` and use the
metric function you are interested in:

```python
import cloudmetrics

iorg = cloudmetrics.mask.iorg_objects(mask=da_cloudmask, periodic_domain=False)
```

As you can see in the table above the metrics are organised by the input they
take, either object masks, labelled-object arrays and/or 2D scalar fields you
want to perform reductions on.

*Note on periodic domains*: internally `cloudmetrics` represents objects on
periodic domains by doubling the xy-size of any input mask provided, and moving
any objects that straddle the boundary to ensure they are spatially contiguous.
This means that all functions which take 2D arrays of object-labels as input
assume that all labelled objects are spatially contiguous and that the provided
input is actually `2*nx x 2*ny` (for an actual input domain spanning `nx` by
`nx`). All metric functions that operate on masks handle the domain-doubling
internally and so do not require any modification to the masks before calling.
