import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

import cloudmetrics


def test_hilbert_curve():
    """
    Test on Hilbert curve (should have fracDim=2)
    """
    mask = np.zeros((512, 512))
    p_hil = 8
    n_hil = 2
    dist = 2 ** (p_hil * n_hil)
    hilbert_curve = HilbertCurve(p_hil, n_hil)
    coords = np.zeros((dist, n_hil))
    for i in range(dist):
        coords[i, :] = hilbert_curve.point_from_distance(i)
    coords = coords.astype(int)
    coords *= 2
    coords_av = ((coords[1:, :] + coords[:-1, :]) / 2).astype(int)
    mask[coords[:, 0], coords[:, 1]] = 1
    mask[coords_av[:, 0], coords_av[:, 1]] = 1

    fractal_dim = cloudmetrics.mask.fractal_dimension(mask=mask)

    np.testing.assert_allclose(fractal_dim, 2.0, atol=1e-4)


def test_random():
    """
    Test on randomly scattered points (should have fractal_dim=2)
    """

    mask = np.random.rand(512, 512)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    fractal_dim = cloudmetrics.mask.fractal_dimension(mask=mask)

    np.testing.assert_allclose(fractal_dim, 2.0, atol=1e-4)


def test_line():
    """
    Test on vertical line (should have fractal_dim=1)
    """

    mask = np.zeros((512, 512))
    mask[:, 250:252] = 1

    fractal_dim = cloudmetrics.mask.fractal_dimension(mask=mask)

    np.testing.assert_allclose(fractal_dim, 1.0, atol=1e-4)
