import numpy as np

from ._object_properties import _get_objects_area, _get_objects_property


def mean_length_scale(object_labels):
    """
    Compute mean length-scale (in pixel units) of all labeled objects
    """
    # TODO: what does the comment below mean?
    # area = np.sqrt(area) <- Janssens et al. (2021) worked in l-space.
    #                         However, working directly with areas before
    #                         taking mean is more representative of pattern
    objects_area = _get_objects_area(object_labels=object_labels)
    return np.sqrt(np.mean(objects_area))


def max_length_scale(object_labels):
    """
    Length scale of largest object in the scene.
    """
    objects_area = _get_objects_area(object_labels=object_labels)
    return np.sqrt(np.max(objects_area))


def mean_eccentricity(object_labels):
    """
    Area-weighted, mean eccentricity of objects, approximated as ellipses
    """

    objects_area = _get_objects_area(object_labels=object_labels)
    objects_ecc = _get_objects_property(
        object_labels=object_labels, property_name="eccentricity"
    )

    return np.sum(objects_area * objects_ecc) / np.sum(objects_area)


def mean_perimeter_length(object_labels):
    """
    Compute mean perimeter length of across all labeled objects in pixel units

    NOTE: the perimeter is calculated as the "Perimeter of object which
    approximates the contour as a line through the centers of border pixels
    using a 4-connectivity." This means that a object comprised of 3x3 pixels
    will have a perimeter length of 8 (2*4)
    """
    objects_perim = _get_objects_property(
        object_labels=object_labels, property_name="perimeter"
    )
    return np.mean(objects_perim)
