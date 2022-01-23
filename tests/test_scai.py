import numpy as np
import pytest

import cloudmetrics

from .scai_examples import EXAMPLES


@pytest.mark.parametrize("test_name", EXAMPLES.keys())
def test_a(test_name):
    cloud_mask, scai_value_true = EXAMPLES[test_name]
    assert cloud_mask.shape == (20, 20)

    cloud_labels = cloudmetrics.objects.label(cloud_mask=cloud_mask, connectivity=1)

    print(cloud_labels)
    scai_value, D0 = cloudmetrics.objects.scai(
        object_labels=cloud_labels,
        periodic_domain=False,
        return_nn_dist=True,
        dx=55,  # About half a degree, seems to be what White et al. (2018) use
    )

    print(scai_value, D0)

    np.testing.assert_almost_equal(scai_value, scai_value_true, decimal=2)
