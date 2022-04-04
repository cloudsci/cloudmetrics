import numpy as np

from ._object_properties import _get_objects_area, _get_objects_property


def mean_length_scale(object_labels):
    """
    Computes the mean area of all labeled objects and takes the square root.
    Gives a length in number of pixels. To get a length in physical units, multiply
    the output of this function by the physical pixel size.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    mean_length : float
        Mean length scale.

    """
    objects_area = _get_objects_area(object_labels=object_labels)
    return np.sqrt(np.mean(objects_area))


def max_length_scale(object_labels):
    """
    Finds the area of the largest labeled object in the scene and takes its square
    root. Gives a length in number of pixels. To get a length in physical units,
    multiply the output of this function by the physical pixel size.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    max_length : float
        Maximum length scale.

    """
    objects_area = _get_objects_area(object_labels=object_labels)
    return np.sqrt(np.max(objects_area))


def mean_eccentricity(object_labels):
    """
    Computes the area-weighted, mean eccentricity of all labelled objects in the
    scene, approximated as ellipses.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    eccentricity : float
        Area-weighted mean eccentricity.

    """

    objects_area = _get_objects_area(object_labels=object_labels)
    objects_ecc = _get_objects_property(
        object_labels=object_labels, property_name="eccentricity"
    )

    return np.sum(objects_area * objects_ecc) / np.sum(objects_area)


def mean_perimeter_length(object_labels):
    """
    Computes the mean perimeter length across all labeled objects. Gives a length
    in number of pixels. To get a length in physical units, multiply the output
    of this function by the physical pixel size.

    NOTE: the perimeter is calculated as the "perimeter of object which
    approximates the contour as a line through the centers of border pixels
    using a 4-connectivity." This means that a object comprised of 3x3 pixels
    will have a perimeter length of 8 (2*4)


    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    mean_perimeter_length : float
        Mean perimeter length.

    """
    objects_perim = _get_objects_property(
        object_labels=object_labels, property_name="perimeter"
    )
    return np.mean(objects_perim)
