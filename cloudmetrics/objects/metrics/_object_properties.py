import numpy as np
from skimage.measure import regionprops

_CACHED_VALUES = dict()


def _get_regionprops(object_labels):
    # use python's memory ID of the labels array for a poor-mans caching to
    # avoid recalculation of the object properties
    array_id = id(object_labels)
    if array_id in _CACHED_VALUES:
        return _CACHED_VALUES[array_id]

    regions = regionprops(label_image=object_labels)
    _CACHED_VALUES[array_id] = regions
    return regions


def _get_objects_property(object_labels, property_name):
    regions = _get_regionprops(object_labels=object_labels)
    num_objects = len(regions)

    values = []
    for i in range(num_objects):
        value = getattr(regions[i], property_name)
        values.append(value)
    return np.asarray(values)


def _get_objects_area(object_labels):
    return _get_objects_property(object_labels=object_labels, property_name="area")
