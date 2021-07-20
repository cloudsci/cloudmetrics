import numpy as np
import pytest
from math import sqrt


import cloudmetrics

EXAMPLE_MASK_STRING = """
00000000000000000000
00000000000000000000
00000000000000000000
00001110000000000000
00001110000000000000
00001110000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000001111000000
00000000001111000000
00000000001111000000
00000000001111000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
"""


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLE_MASK = _parse_example_mask(EXAMPLE_MASK_STRING)


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_max_length_scale(periodic_domain):
    if periodic_domain:
        raise NotImplementedError(periodic_domain)

    cloud_mask = EXAMPLE_MASK
    cloud_object_labels = cloudmetrics.objects.label(cloud_mask=cloud_mask)
    l_max = cloudmetrics.objects.metrics.max_length_scale(
        cloud_object_labels=cloud_object_labels
    )

    np.testing.assert_almost_equal(l_max, 4.0)


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_mean_length_scale(periodic_domain):
    if periodic_domain:
        raise NotImplementedError(periodic_domain)

    cloud_mask = EXAMPLE_MASK
    cloud_object_labels = cloudmetrics.objects.label(cloud_mask=cloud_mask)
    l_mean = cloudmetrics.objects.metrics.mean_length_scale(
        cloud_object_labels=cloud_object_labels
    )

    # 3^2 + 4^2 = 5^2 => (3^2 + 4^2)/2 = (5/sqrt(2))^2
    np.testing.assert_almost_equal(l_mean, 5.0 / sqrt(2.0))


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_mean_perimeter_length(periodic_domain):
    if periodic_domain:
        raise NotImplementedError(periodic_domain)

    cloud_mask = EXAMPLE_MASK
    cloud_object_labels = cloudmetrics.objects.label(cloud_mask=cloud_mask)
    l_perim_length = cloudmetrics.objects.metrics.mean_perimeter_length(
        cloud_object_labels=cloud_object_labels
    )

    l_true = (2.0 * 4 + 3.0 * 4) / 2.0

    np.testing.assert_almost_equal(l_perim_length, l_true)


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_mean_eccentricity(periodic_domain):
    if periodic_domain:
        raise NotImplementedError(periodic_domain)

    cloud_mask = EXAMPLE_MASK
    cloud_object_labels = cloudmetrics.objects.label(cloud_mask=cloud_mask)
    mean_ecc = cloudmetrics.objects.metrics.mean_eccentricity(
        cloud_object_labels=cloud_object_labels
    )

    np.testing.assert_almost_equal(mean_ecc, 0.0)
