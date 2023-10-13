import numpy as np
import pytest

import cloudmetrics

from .abcop_examples import EXAMPLES


@pytest.mark.parametrize("test_name", EXAMPLES.keys())
def test_a(test_name):
    cloud_mask, cop_value_true = EXAMPLES[test_name]


    cloud_labels = cloudmetrics.objects.label(mask=cloud_mask, connectivity=1)
    cop_value = cloudmetrics.objects.abcop(
        object_labels=cloud_labels, periodic_domain=False
    )

    np.testing.assert_almost_equal(cop_value, cop_value_true, decimal=4)
