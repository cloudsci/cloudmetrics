import numpy as np

from cloudmetrics.utils import make_periodic_mask


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLE_MASK = _parse_example_mask(
    """
00011000
11000011
11011011
00011000
00000000
00000000
00011000
00011000
"""
)

EXAMPLE_MASK_DOUBLED = _parse_example_mask(
    """
0000000000000000
0000001111000000
0001101111000000
0001100000000000
0000000000000000
0000000000000000
0001100000000000
0001100000000000
0001100000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
"""
)


def test_periodic_domain():
    nx, ny = EXAMPLE_MASK.shape
    mask_periodic = make_periodic_mask(EXAMPLE_MASK, object_connectivity=1)
    assert mask_periodic.shape == (nx * 2, ny * 2)

    np.testing.assert_equal(mask_periodic, EXAMPLE_MASK_DOUBLED)
