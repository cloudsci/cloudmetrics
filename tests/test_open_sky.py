import numpy as np
import pytest


import cloudmetrics

EXAMPLE_MASKS_STRING = """
00111000000000000000
00000000000000000000
00000000000000000000
00011111100000000000
00011111100000111111
00000000000000111111
00000000000000111111
00000000000000111111
00000000000000000000
00000000000000000000
00000000001111100000
00000000001111100000
00000000001111100000
00000000001111100000
10000000001111100000
00100000000000000000
11000000000000001111
11000000000000001111
00000000000000000000
10011000000000000000
"""


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLE_MASK = _parse_example_mask(EXAMPLE_MASKS_STRING)


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_open_sky(periodic_domain):
    os_max, os_avg = cloudmetrics.open_sky(
        cloud_mask=EXAMPLE_MASK, periodic_domain=periodic_domain
    )

    assert not np.isnan(os_max)
    assert not np.isnan(os_avg)

    assert [os_max, os_avg] == 0.0

    # TODO: what are reasonable values here?
