import numpy as np
import pytest


import cloudmetrics


def test_woi():

    # Horizontal stripes
    cloud_scalar_h = np.zeros((512, 512))
    cloud_scalar_h[::4, :] = 1
    Ebar, Elbar, Esbar, Eld, Esd = cloudmetrics.compute_swt(
        cloud_scalar_h, "periodic", "haar", 5
    )

    # All energy should be in horizontal direction
    np.testing.assert_allclose((Eld + Esd)[1:], 0, atol=1e-10)

    # Vertical stripes
    cloud_scalar_v = np.zeros((512, 512))
    cloud_scalar_v[:, ::4] = 1
    Ebar, Elbar, Esbar, Eld, Esd = cloudmetrics.compute_swt(
        cloud_scalar_v, "periodic", "haar", 5
    )

    # All energy should be in vertical direction
    np.testing.assert_allclose((Eld + Esd)[[0, 2]], 0, atol=1e-10)
