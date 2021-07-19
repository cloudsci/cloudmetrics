# cloudmetrics

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
indicating where clouds exist.

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
