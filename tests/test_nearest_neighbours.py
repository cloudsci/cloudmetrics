import numpy as np
import pytest

import cloudmetrics
from cloudmetrics.objects.metrics._object_properties import _get_objects_property


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLES = {}
EXAMPLES["straddling_object"] = (
    _parse_example_mask(
        """
0000000000
1100000001
1100000101
1100000001
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
"""
    ),
    [3, 3],
)

EXAMPLES["near_edge"] = (
    _parse_example_mask(
        """
0000000000
0000000000
0010000100
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
"""
    ),
    [5, 5],
)

EXAMPLES["diagonal_straddle"] = (
    _parse_example_mask(
        """
1100000001
1000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000010
1000000000
"""
    ),
    [np.sqrt(8), np.sqrt(8)],
)


@pytest.mark.parametrize("example", EXAMPLES.values())
def test_nearest_neighbors(example):
    mask, nn_distances_expected = example
    domain_shape = mask.shape
    assert domain_shape == (10, 10)

    mask_periodic = cloudmetrics.utils.make_periodic_mask(
        mask=mask, object_connectivity=1
    )
    object_labels = cloudmetrics.objects.label(mask=mask_periodic, connectivity=1)

    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    # Move centroids outside the original domain into original domain
    centroids[centroids[:, 0] >= domain_shape[1], 0] -= domain_shape[1]
    centroids[centroids[:, 0] < 0, 0] += domain_shape[1]
    centroids[centroids[:, 1] >= domain_shape[0], 1] -= domain_shape[0]
    centroids[centroids[:, 1] < 0, 1] += domain_shape[0]

    nn_distances = cloudmetrics.utils.find_nearest_neighbors(data=centroids, size=10)
    assert np.all(nn_distances == nn_distances_expected)
