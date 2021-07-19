import numpy as np
import pytest


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
00000000001111100000
00000000001111100000
00000000001111100000
00000000001111100000
00000000001111100000
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

    np.testing.assert_almost_equal(l_max, 5.0)
