import numpy as np
import pytest

import cloudmetrics

EXAMPLE_MASKS_STRING = """
01111000000000000000
01100000000000000000
01100000000000000000
00011111110000000000
00011111110000111111
00000000000000111111
00000000000000111111
00000000000000111111
00000000001100000000
00000000001100000000
00000000001111100000
00000000001111100000
00000000001111100000
00000000001111100000
10000000001111100000
00100000000000000000
11000000000000001111
11000000000000001111
00000000000000001100
10011000000000000000
"""


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLE_MASK = _parse_example_mask(EXAMPLE_MASKS_STRING)


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_open_sky(periodic_domain):
    os_max, os_avg = cloudmetrics.mask.open_sky(
        mask=EXAMPLE_MASK, periodic_domain=periodic_domain
    )

    assert not np.isnan(os_max)
    assert not np.isnan(os_avg)

    if periodic_domain:
        np.testing.assert_allclose([os_max, os_avg], [0.855, 0.503], atol=0.01)
    else:
        np.testing.assert_allclose([os_max, os_avg], [0.720, 0.285], atol=0.01)
