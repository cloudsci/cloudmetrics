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
indicating where clouds exist, some work on individually labelled (with a
unique integer ID) cloud objects (which can be produced from a cloud-mask) and
finally some work on 2D cloud scalar-fields (defining for example the
cloud-liquid water or cloud-top height).

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

#2: need refactoring to take `scalar_field` as input

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

To use the `cloudmetrics` package simply import `cloudmetrics` and use the metric function you are interested in:

```python
import cloudmetrics

iorg = cloudmetrics.iorg(cloud_mask=da_cloudmask)
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
`nx`).
