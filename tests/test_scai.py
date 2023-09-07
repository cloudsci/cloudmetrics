import numpy as np
import pytest

import cloudmetrics

from .scai_examples import EXAMPLE_DOUBLING_RESOLUTION, EXAMPLES


@pytest.mark.parametrize("test_name", EXAMPLES.keys())
def test_a(test_name):
    mask, scai_value_true = EXAMPLES[test_name]
    assert mask.shape == (20, 20)

    cloud_labels = cloudmetrics.objects.label(mask=mask, connectivity=1)

    scai_value, D0 = cloudmetrics.objects.scai(
        object_labels=cloud_labels,
        periodic_domain=False,
        return_nn_dist=True,
        dx=55,  # About half a degree, seems to be what White et al. (2018) use
    )

    np.testing.assert_almost_equal(scai_value, scai_value_true, decimal=2)


def test_resolution_doubling():
    mask, mask_halfdx = EXAMPLE_DOUBLING_RESOLUTION

    nx, ny = mask.shape
    assert mask_halfdx.shape == (2 * nx, 2 * ny)

    cloud_labels = cloudmetrics.objects.label(mask=mask, connectivity=1)
    dx = 1.0
    scai_value = cloudmetrics.objects.scai(
        object_labels=cloud_labels, periodic_domain=False, dx=dx
    )

    cloud_labels = cloudmetrics.objects.label(mask=mask_halfdx, connectivity=1)
    dx2 = 0.5
    scai_value_halfdx = cloudmetrics.objects.scai(
        object_labels=cloud_labels,
        periodic_domain=False,
        dx=dx2,
    )

    # NB: SCAI isn't resolution independent. Instead, from the same domain sampled
    # at two different resolutions the value of SCAI will increase with dx^2.
    # To correct for this we here scale the SCAI values by 1/dx^2 for the
    # examples
    np.testing.assert_almost_equal(
        scai_value / dx**2.0, scai_value_halfdx / dx2**2.0
    )
