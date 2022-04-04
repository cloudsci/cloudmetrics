from functools import partial
from math import sqrt

import numpy as np
import pytest

import cloudmetrics


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


TRUE_VALUES_PERIODIC_DOMAIN = {}
TRUE_VALUES_APERIODIC_DOMAIN = {}


EXAMPLE_MASK_CORNERS_STRING = """
11000000000000000011
11000000000000000011
00000000000000000000
00001110000000000000
00001110000000000000
00001110000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
11000000000000000011
11000000000000000011
"""

EXAMPLE_MASK_CORNERS = _parse_example_mask(EXAMPLE_MASK_CORNERS_STRING)
TRUE_VALUES_PERIODIC_DOMAIN[id(EXAMPLE_MASK_CORNERS)] = dict(
    num_objects=2,
    max_length_scale=4.0,
    # 3^2 + 4^2 = 5^2 => (3^2 + 4^2)/2 = (5/sqrt(2))^2
    mean_length_scale=5.0 / sqrt(2.0),
    mean_perimeter_length=(2.0 * 4 + 3.0 * 4) / 2.0,
    mean_eccentricity=0.0,
)
TRUE_VALUES_APERIODIC_DOMAIN[id(EXAMPLE_MASK_CORNERS)] = dict(
    num_objects=5,
    max_length_scale=3.0,
    mean_length_scale=sqrt((4 * 4.0 + 9) / 5.0),
    mean_perimeter_length=(2.0 * 4 + 1.0 * 4 * 4) / 5.0,
    mean_eccentricity=0.0,
)


EXAMPLE_MASK_EW_STRING = """
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
11110000000000001111
11110000000000001111
11110000000000001111
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
"""
EXAMPLE_MASK_EW = _parse_example_mask(EXAMPLE_MASK_EW_STRING)
TRUE_VALUES_PERIODIC_DOMAIN[id(EXAMPLE_MASK_EW)] = dict(
    num_objects=1,
)
TRUE_VALUES_APERIODIC_DOMAIN[id(EXAMPLE_MASK_EW)] = dict(
    num_objects=2,
)

EXAMPLE_MASK_NS_STRING = """
00000001110000000000
00000001110000000000
00000001110000000000
00000001110000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000001110000000000
00000001110000000000
00000001110000000000
00000001110000000000
"""
EXAMPLE_MASK_NS = _parse_example_mask(EXAMPLE_MASK_NS_STRING)
TRUE_VALUES_PERIODIC_DOMAIN[id(EXAMPLE_MASK_NS)] = dict(
    num_objects=1,
)
TRUE_VALUES_APERIODIC_DOMAIN[id(EXAMPLE_MASK_NS)] = dict(
    num_objects=2,
)


EXAMPLE_MASKS = [EXAMPLE_MASK_CORNERS, EXAMPLE_MASK_EW, EXAMPLE_MASK_NS]

TESTSETS = []

for mask in EXAMPLE_MASKS:
    for periodic_domain in [True, False]:
        if periodic_domain:
            mask_values = TRUE_VALUES_PERIODIC_DOMAIN[id(mask)]
        else:
            mask_values = TRUE_VALUES_APERIODIC_DOMAIN[id(mask)]

        for metric, metric_value in mask_values.items():
            if metric.startswith("mean_"):
                comp_function = partial(np.testing.assert_almost_equal, decimal=10)
            else:
                comp_function = np.testing.assert_equal
            testset = (mask, metric, periodic_domain, metric_value, comp_function)
            TESTSETS.append(testset)


def _make_mask_function_name(metric_name):
    """
    Generate a name for the metric function that applies directly to a mask

    The naming convention for object functions that apply directly to masks is
        `{op}_{measure}` -> `{op}_cloud_{measure}`,
    e.g.
        `num_objects` -> `num_objects`
        `mean_perimeter_length` -> `mean_object_perimeter_length`
    """
    if "objects" in metric_name:
        return metric_name

    op, *measure_parts = metric_name.split("_")
    return "_".join([op, "object"] + measure_parts)


@pytest.mark.parametrize(
    "mask, metric_name, periodic_domain, metric_value_true, comp_function",
    TESTSETS,
)
def test_metric_on_mask(
    mask, metric_name, periodic_domain, metric_value_true, comp_function
):
    # test for evaluation with `cloudmetrics.{op}_cloud_{measure}`, e.g.
    # `cloudmetrics.mean_cloud_length_scale`
    mask_metric_function_name = _make_mask_function_name(metric_name=metric_name)
    metric_fn = getattr(cloudmetrics.mask, mask_metric_function_name, None)
    if metric_fn is None:
        raise NotImplementedError(
            f"Function for computing metric `{mask_metric_function_name}` wasn't found"
        )

    metric_value = metric_fn(mask=mask, periodic_domain=periodic_domain)
    comp_function(metric_value, metric_value_true)
