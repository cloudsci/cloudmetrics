# cloudmetrics

[![cloudmetrics](https://github.com/cloudsci/cloudmetrics/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/cloudsci/cloudmetrics/actions/workflows/python-package-conda.yml)

> **NOTE**: this repository is currently undergoing refactoring to make
the routines implemented more accessible by external tools and to ensure
consistency. The version published in Janssens et al 2021 is [available
tagged as version
v0.1.0](https://github.com/cloudsci/cloudmetrics/tree/v0.1.0). Progress on
the refactoring can be followed in issue
https://github.com/cloudsci/cloudmetrics/issues/20

The `cloudmetrics` package contains python routines to compute metrics
from 2D cloud fields to characterise cloud patterns in these fields. Most
methods operate on a `cloud-mask` (i.e. a boolean true-false field)
indicating where clouds exist, some work on individually labelled cloud objects
(which can be produced from a cloud-mask) and finally some work on 2D cloud
scalar-fields (defining for example the cloud-liquid water or cloud-top height).

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
| `objects.iorg`                     | ✔️ #1     | TODO            |                |
| `objects.iorg_poisson`             | TODO     | TODO            |                |
| `objects.max_length_scale`         | ✔️†       | ✔️               |                |
| `objects.mean_eccentricity`        | ✔️†       | ✔️               |                |
| `objects.mean_length_scale`        | ✔️†       | ✔️               |                |
| `objects.mean_perimeter_length`    | ✔️†       | ✔️               |                |
| `objects.rdf`                      | TODO     | TODO            |                |
| `objects.scai`                     | ✔️†       | ✔️               |                |
| `scalar.spectral_anisotropy` #2    |          |                 | ✔️              |
| `scalar.spectral_length_median`#2  |          |                 | ✔️              |
| `scalar.spectral_length_moment,`#2 |          |                 | ✔️              |
| `scalar.spectral_slope`#2          |          |                 | ✔️              |
| `scalar.spectral_slope_binned`#2   |          |                 | ✔️              |
| `scalar.woi1`                      |          |                 | ✔️              |
| `scalar.woi2`                      |          |                 | ✔️              |
| `scalar.woi3`                      |          |                 | ✔️              |
| `scalar.mean`                      | optional |                 | ✔️              |
| `scalar.var`                       | optional |                 | ✔️              |
| `scalar.skew`                      | optional |                 | ✔️              |
| `scalar.kurtosis`                  | optional |                 | ✔️              |

†: for convenience object-based scalars are also made avaiable to operate
directly on masks, for example `objects.max_length_scale(object_labels=...)`
can be called with a mask as `mask.max_object_length_scale(mask=...)`

#1: needs refactoring to use general object labelling and make iorg method
available to use on object-labels as input

#2: spectral metrics currently operate on the relevant power spectral densities,
which must first be computed:
```
wavenumbers, psd_1d_radial, psd_1d_azimuthal = scalar.compute_spectra(...)
spectral_length_moment = scalar.spectral_length_moment(wavenumbers, psd_1d_radial)
```
Alternatively, all spectral metrics can be computed simultaneously following the
standard convention with `spectral_metrics = scalar.compute_all_spectral(scalar_field).
need refactoring to take `scalar_field` as input

# Installation

Until `cloudmetrics` appears on pipy the package can be installed directly
from github

```bash
$> pip install git+https://github.com/cloudsci/cloudmetrics
```

# Usage

To use the `cloudmetrics` package simply import `cloudmetrics` and use the metric function you are interested in:

```python
import cloudmetrics

iorg = cloudmetrics.iorg(cloud_mask=da_cloudmask)
```
