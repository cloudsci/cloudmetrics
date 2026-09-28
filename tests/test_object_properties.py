"""
Check that the fast (numpy based) object properties are identical to the
values computed with `skimage.measure.regionprops`
"""

import numpy as np
import pytest
from skimage.measure import regionprops

import cloudmetrics
from cloudmetrics.objects import label as label_objects
from cloudmetrics.objects.metrics._object_properties import (
    _get_num_objects,
    _get_objects_area,
    _get_objects_property,
)

FAST_PROPERTIES = ["area", "centroid", "equivalent_diameter_area"]


def _regionprops_property(object_labels, property_name):
    """
    the original implementation of `_get_objects_property`
    """
    regions = regionprops(label_image=object_labels)
    return np.asarray([getattr(region, property_name) for region in regions])


def _random_label_images():
    rng = np.random.default_rng(42)
    label_images = []
    for i in range(20):
        shape = (int(rng.integers(3, 120)), int(rng.integers(3, 120)))
        mask = rng.random(shape) > rng.uniform(0.3, 0.95)
        connectivity = int(rng.integers(1, 3))
        label_images.append(label_objects(mask=mask, connectivity=connectivity))
    return label_images


def _special_label_images():
    # no objects at all
    no_objects = np.zeros((20, 30), dtype=int)

    # a single object
    single_object = np.zeros((20, 30), dtype=int)
    single_object[5:10, 12:20] = 1

    # non-contiguous labels: label value 2 (and everything above 4) is missing
    missing_label = np.zeros((20, 30), dtype=int)
    missing_label[1:3, 1:3] = 1
    missing_label[8:12, 5:10] = 3
    missing_label[15:19, 20:29] = 4

    # object labelled with a value far larger than the number of objects
    large_label = np.zeros((10, 10), dtype=np.int64)
    large_label[2:4, 2:4] = 1000

    # unsigned and small integer label dtypes
    small_dtype = missing_label.astype(np.uint8)
    int32_dtype = missing_label.astype(np.int32)

    # objects from a periodic domain (label image is twice the domain size)
    rng = np.random.default_rng(0)
    mask = rng.random((30, 30)) > 0.7
    mask_periodic = cloudmetrics.utils.make_periodic_mask(
        mask=mask, object_connectivity=1
    )
    periodic = label_objects(mask=mask_periodic, connectivity=1)

    return [
        no_objects,
        single_object,
        missing_label,
        large_label,
        small_dtype,
        int32_dtype,
        periodic,
    ]


LABEL_IMAGES = _random_label_images() + _special_label_images()


@pytest.mark.parametrize("object_labels", LABEL_IMAGES)
@pytest.mark.parametrize("property_name", FAST_PROPERTIES)
def test_fast_property_identical_to_regionprops(object_labels, property_name):
    expected = _regionprops_property(object_labels, property_name)
    values = _get_objects_property(
        object_labels=object_labels, property_name=property_name
    )
    assert values.shape == expected.shape
    assert values.dtype == expected.dtype
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("object_labels", LABEL_IMAGES)
def test_equivalent_diameter_identical_to_regionprops(object_labels):
    # deprecated alias of `equivalent_diameter_area` used in `iorg`
    expected = _regionprops_property(object_labels, "equivalent_diameter_area")
    values = _get_objects_property(
        object_labels=object_labels, property_name="equivalent_diameter"
    )
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("object_labels", LABEL_IMAGES)
def test_area_and_num_objects(object_labels):
    regions = regionprops(label_image=object_labels)
    np.testing.assert_array_equal(
        _get_objects_area(object_labels=object_labels),
        np.asarray([region.area for region in regions]),
    )
    assert _get_num_objects(object_labels=object_labels) == len(regions)
    assert cloudmetrics.objects.metrics.num_objects(object_labels=object_labels) == len(
        regions
    )


def test_no_objects_shapes():
    """
    with no objects an empty 1-d array is returned for every property (like
    `np.asarray([])` on the empty list of regions)
    """
    object_labels = np.zeros((10, 10), dtype=int)
    for property_name in FAST_PROPERTIES + ["perimeter"]:
        values = _get_objects_property(
            object_labels=object_labels, property_name=property_name
        )
        assert values.shape == (0,)
    assert _get_num_objects(object_labels=object_labels) == 0


def test_regionprops_fallback_property():
    object_labels = LABEL_IMAGES[0]
    expected = _regionprops_property(object_labels, "perimeter")
    values = _get_objects_property(
        object_labels=object_labels, property_name="perimeter"
    )
    np.testing.assert_array_equal(values, expected)


def test_non_integer_labels_raise_like_regionprops():
    object_labels = np.zeros((10, 10), dtype=float)
    object_labels[2:4, 2:4] = 1.0
    with pytest.raises(TypeError):
        _get_objects_property(object_labels=object_labels, property_name="area")
    with pytest.raises(TypeError):
        _get_num_objects(object_labels=object_labels)
