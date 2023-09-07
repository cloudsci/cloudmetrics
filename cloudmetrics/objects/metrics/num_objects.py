from ._object_properties import _get_regionprops


def num_objects(object_labels):
    """
    Compute number of labelled objects

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    object_number
        Number of labelled objects.

    """
    regions = _get_regionprops(object_labels=object_labels)
    return len(regions)
