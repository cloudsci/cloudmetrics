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
    os_max = cloudmetrics.mask.open_sky(
        mask=EXAMPLE_MASK,
        periodic_domain=periodic_domain,
    )
    os_avg = cloudmetrics.mask.open_sky(
        mask=EXAMPLE_MASK, periodic_domain=periodic_domain, summary_measure="mean"
    )

    assert not np.isnan(os_max)
    assert not np.isnan(os_avg)

    if periodic_domain:
        np.testing.assert_allclose([os_max, os_avg], [0.855, 0.503], atol=0.01)
    else:
        np.testing.assert_allclose([os_max, os_avg], [0.720, 0.285], atol=0.01)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("op", ["mean", "max"])
def test_open_sky_extremes(periodic_domain, op):
    FULLY_CLOUDY_MASK = np.ones((10, 10))
    FULLY_CLEAR_MASK = np.zeros((10, 10))

    assert (
        cloudmetrics.mask.open_sky(
            mask=FULLY_CLEAR_MASK, periodic_domain=periodic_domain, summary_measure=op
        )
        == 1
    )

    assert (
        cloudmetrics.mask.open_sky(
            mask=FULLY_CLOUDY_MASK, periodic_domain=periodic_domain, summary_measure=op
        )
        == 0
    )


def _all_measures(mask, periodic_domain):
    return [
        cloudmetrics.mask.open_sky(
            mask=mask, periodic_domain=periodic_domain, summary_measure=op
        )
        for op in ["max", "mean"]
    ]


def test_open_sky_single_cloudy_pixel():
    # 3x4 non-square mask with a single cloudy pixel at (1, 1). Rows/columns
    # without any cloud span the full domain (12 px area) in both domain
    # types. In row 1 and column 1 the areas are (non-periodic) 0, 9, 9 and
    # 0, 8, and (periodic, wrapping around the domain) 9, 9, 9 and 8, 8.
    mask = np.zeros((3, 4))
    mask[1, 1] = 1

    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=False), [1.0, 98 / 11 / 12]
    )
    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=True), [1.0, 115 / 11 / 12]
    )


def test_open_sky_diagonal_clouds():
    # every row and column contains exactly one cloudy pixel. Without
    # periodicity the largest area is found for the corner pixels, e.g. (0, 3)
    # with w=0, e=4, n=0, s=2 -> 8 px; with periodicity every clear pixel spans
    # 3x3 px (e.g. (0, 3): w=0, e=3, n=-1, s=2). Summed non-periodic areas:
    # (4-i)*(j-1) for i<j and (i-1)*(4-j) for i>j, i.e. 25 + 25 = 50 px
    mask = np.eye(4)

    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=False), [8 / 16, 50 / 12 / 16]
    )
    np.testing.assert_equal(_all_measures(mask, periodic_domain=True), [9 / 16, 9 / 16])


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_open_sky_isolated_clear_pixel(periodic_domain):
    mask = np.ones((3, 4))
    mask[1, 2] = 0

    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=periodic_domain), [1 / 12, 1 / 12]
    )


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_open_sky_clear_row(periodic_domain):
    mask = np.ones((3, 4))
    mask[1, :] = 0

    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=periodic_domain), [4 / 12, 4 / 12]
    )


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("dtype", [bool, int, float])
def test_open_sky_mask_dtype(periodic_domain, dtype):
    reference = _all_measures(EXAMPLE_MASK, periodic_domain=periodic_domain)
    np.testing.assert_equal(
        _all_measures(EXAMPLE_MASK.astype(dtype), periodic_domain=periodic_domain),
        reference,
    )


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_open_sky_non_square(periodic_domain):
    # transposing the mask must not change the metric
    rng = np.random.default_rng(0)
    mask = (rng.random((5, 40)) < 0.2).astype(float)

    np.testing.assert_equal(
        _all_measures(mask, periodic_domain=periodic_domain),
        _all_measures(mask.T, periodic_domain=periodic_domain),
    )


def test_open_sky_unknown_summary_measure():
    with pytest.raises(NotImplementedError):
        cloudmetrics.mask.open_sky(mask=EXAMPLE_MASK, summary_measure="median")
