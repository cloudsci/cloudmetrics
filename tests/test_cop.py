import numpy as np
import pytest

import cloudmetrics

from .cop_examples import EXAMPLES


@pytest.mark.parametrize("test_name", EXAMPLES.keys())
def test_a(test_name):
    cloud_mask, cop_value_true = EXAMPLES[test_name]
    assert cloud_mask.shape == (20, 20)

    cloud_labels = cloudmetrics.objects.label(mask=cloud_mask, connectivity=1)
    cop_value = cloudmetrics.objects.cop(
        object_labels=cloud_labels, periodic_domain=False
    )

    np.testing.assert_almost_equal(cop_value, cop_value_true, decimal=4)
