from ._object_properties import _get_regionprops
from .geometry import (  # noqa
    max_length_scale,
    mean_eccentricity,
    mean_length_scale,
    mean_perimeter_length,
)


def num_objects(object_labels):
    """
    Compute number of labelled objects
    """
    regions = _get_regionprops(object_labels=object_labels)
    return len(regions)
