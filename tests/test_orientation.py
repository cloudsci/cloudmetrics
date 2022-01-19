import numpy as np

import cloudmetrics
import cloudmetrics.utils


def test_large_uniform_circle_orientation():
    """
    Large uniform circle (should be 0)
    """
    # 1. One large, uniform circle
    cloud_mask = cloudmetrics.utils.create_circular_mask(h=512, w=512)
    orientation = cloudmetrics.mask.orientation(cloud_mask=cloud_mask)
    np.testing.assert_allclose(orientation, 0.0, atol=0.1)


def test_randomly_scatterd_points():
    """
    Randomly scattered points (should be 0)
    """
    scalars = np.random.random(size=(512, 512))
    cloud_mask = (scalars > 0.5).astype(int)
    orientation = cloudmetrics.mask.orientation(cloud_mask=cloud_mask)
    np.testing.assert_allclose(orientation, 0.0, atol=0.1)


def test_vertical_lines():
    """
    Vertical lines (should be 1)
    """
    cloud_mask = np.zeros((512, 512))
    cloud_mask[:, 250:251] = 1

    orientation = cloudmetrics.mask.orientation(cloud_mask=cloud_mask)
    np.testing.assert_almost_equal(orientation, 1.0)
