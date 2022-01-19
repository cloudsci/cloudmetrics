import numpy as np

import cloudmetrics


def test_cloud_fraction():
    cloud_mask_0 = np.ones((512, 512))
    cloud_mask_1 = np.zeros((512, 512))
    cloud_mask_2 = np.zeros((512, 512))
    cloud_mask_2[::2, ::2] = 1

    cloud_fraction_0 = cloudmetrics.mask.cloud_fraction(cloud_mask_0)
    cloud_fraction_1 = cloudmetrics.mask.cloud_fraction(cloud_mask_1)
    cloud_fraction_2 = cloudmetrics.mask.cloud_fraction(cloud_mask_2)

    eps = 1e-14
    np.testing.assert_allclose(cloud_fraction_0, 1.00, atol=eps)
    np.testing.assert_allclose(cloud_fraction_1, 0.00, atol=eps)
    np.testing.assert_allclose(cloud_fraction_2, 0.25, atol=eps)
