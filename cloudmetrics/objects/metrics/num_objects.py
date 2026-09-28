from ._object_properties import _get_num_objects


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
    return _get_num_objects(object_labels=object_labels)
