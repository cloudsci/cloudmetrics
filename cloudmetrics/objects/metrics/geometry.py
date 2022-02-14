import numpy as np

from ._object_properties import _get_objects_area, _get_objects_property


def mean_length_scale(object_labels, periodic_domain=False):
    """
    Computes the mean area of all labeled objects, takes the square root, and
    normalises with the a typical size of the `cloud_mask` from which `object_labels`
    derives: `L=np.sqrt(cloud_mask.shape().prod())`. To get a length in physical
    units, multiply the output of this function by `L` and the physical pixel size.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.
    periodic_domain : bool
        Flag for periodic domains. Default is False.

    Returns
    -------
    mean_length : float
        Normalised mean length scale.

    """
    objects_area = _get_objects_area(object_labels=object_labels)
    if periodic_domain:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape) / 2))
    else:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape)))
    return np.sqrt(np.mean(objects_area)) / L


def max_length_scale(object_labels, periodic_domain=False):
    """
    Finds the area of the largest labeled object in the scene, takes its square
    root, and normalises with the a typical size of the `cloud_mask` from
    which `object_labels` derives: `L=np.sqrt(cloud_mask.shape().prod())`. To get
    a length in physical units, multiply the output of this function by `L`
    and the physical pixel size.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.

    Returns
    -------
    max_length : float
        Normalised maximum length scale.

    """
    objects_area = _get_objects_area(object_labels=object_labels)
    if periodic_domain:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape) / 2))
    else:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape)))
    return np.sqrt(np.max(objects_area)) / L


def mean_eccentricity(object_labels, periodic_domain=False):
    """
    Computes the area-weighted, mean eccentricity of all labelled objects in the '
    scene, approximated as ellipses.

    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.
    periodic_domain : bool
        Flag for periodic domains. Default is False.

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


def mean_perimeter_length(object_labels, periodic_domain=False):
    """
    Computes the mean perimeter length across all labeled objects, and normalises
    with the a typical size of the `cloud_mask` from which `object_labels` derives:
    `L=np.sqrt(cloud_mask.shape().prod())`. To get a length in physical units,
    multiply the output of this function by `L` and the physical pixel size.

    NOTE: the perimeter is calculated as the "perimeter of object which
    approximates the contour as a line through the centers of border pixels
    using a 4-connectivity." This means that a object comprised of 3x3 pixels
    will have a perimeter length of 8 (2*4)


    Parameters
    ----------
    object_labels : 2-d numpy array
        Field of labelled objects.
    periodic_domain : bool
        Flag for periodic domains. Default is False.

    Returns
    -------
    max_length : float
        Nomralised mean perimeter length.

    """
    objects_perim = _get_objects_property(
        object_labels=object_labels, property_name="perimeter"
    )
    if periodic_domain:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape) / 2))
    else:
        L = np.sqrt(np.prod(np.asarray(object_labels.shape)))
    return np.mean(objects_perim) / L
