import numpy as np

import cloudmetrics


def test_woi():

    # Horizontal stripes
    cloud_scalar_h = np.zeros((512, 512))
    cloud_scalar_h[::4, :] = 1
    Ebar, Elbar, Esbar, Eld, Esd = cloudmetrics.scalar.compute_swt(
        cloud_scalar_h, "periodic", "haar", 5
    )

    woi1_h = cloudmetrics.scalar.woi1(cloud_scalar_h)
    woi2_h = cloudmetrics.scalar.woi2(cloud_scalar_h)
    woi3_h = cloudmetrics.scalar.woi3(cloud_scalar_h)

    # All energy should be in small scales
    np.testing.assert_allclose(woi1_h, 0, atol=1e-10)

    # Total energy scaled by cloud pixel number
    np.testing.assert_allclose(woi2_h, 0.0625 / (0.25 * 512 * 512), atol=1e-10)

    # No energy in large scales - small scales should sum to 0
    np.testing.assert_allclose(woi3_h, 0, atol=1e-10)

    # All energy should be in horizontal direction
    np.testing.assert_allclose((Eld + Esd)[1:], 0, atol=1e-10)

    # Vertical stripes
    cloud_scalar_v = np.zeros((512, 512))
    cloud_scalar_v[:, ::4] = 1
    Ebar, Elbar, Esbar, Eld, Esd = cloudmetrics.scalar.compute_swt(
        cloud_scalar_v, "periodic", "haar", 5
    )

    # All energy should be in vertical direction
    np.testing.assert_allclose((Eld + Esd)[[0, 2]], 0, atol=1e-10)

    # Half stripes to test woi3
    cloud_scalar_vh = cloud_scalar_v.copy()
    cloud_scalar_vh[:256, :] = 0

    woi3_vh = cloudmetrics.scalar.woi3(cloud_scalar_vh)

    np.testing.assert_allclose(woi3_vh, 1.1462686724831639, atol=1e-10)
