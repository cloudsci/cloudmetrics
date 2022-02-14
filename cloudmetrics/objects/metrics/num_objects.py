from ._object_properties import _get_regionprops


def num_objects(object_labels, periodic_domain=False):
    """
    Compute number of labelled objects
    """
    regions = _get_regionprops(object_labels=object_labels)
    return len(regions)
