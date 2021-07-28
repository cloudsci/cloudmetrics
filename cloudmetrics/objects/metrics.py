from skimage.measure import regionprops
import numpy as np

_CACHED_VALUES = dict()


def _get_regionprops(cloud_object_labels):
    # use python's memory ID of the labels array for a poor-mans caching to
    # avoid recalculation of the object properties
    array_id = id(cloud_object_labels)
    if array_id in _CACHED_VALUES:
        return _CACHED_VALUES[array_id]

    regions = regionprops(label_image=cloud_object_labels)
    _CACHED_VALUES[array_id] = regions
    return regions


def _get_objects_property(cloud_object_labels, property_name):
    regions = _get_regionprops(cloud_object_labels=cloud_object_labels)
    num_objects = len(regions)

    values = []
    for i in range(num_objects):
        value = getattr(regions[i], property_name)
        values.append(value)
    return np.asarray(values)


def _get_objects_area(cloud_object_labels):
    return _get_objects_property(
        cloud_object_labels=cloud_object_labels, property_name="area"
    )


def mean_length_scale(cloud_object_labels):
    """
    Compute mean length-scale (in pixel units) of all labeled objects
    """
    # TODO: what does the comment below mean?
    # area = np.sqrt(area) <- Janssens et al. (2021) worked in l-space.
    #                         However, working directly with areas before
    #                         taking mean is more representative of pattern
    objects_area = _get_objects_area(cloud_object_labels=cloud_object_labels)
    return np.sqrt(np.mean(objects_area))


def max_length_scale(cloud_object_labels):
    """
    Length scale of largest object in the scene.
    """
    objects_area = _get_objects_area(cloud_object_labels=cloud_object_labels)
    return np.sqrt(np.max(objects_area))


def mean_eccentricity(cloud_object_labels):
    """
    Area-weighted, mean eccentricity of objects, approximated as ellipses
    """

    objects_area = _get_objects_area(cloud_object_labels=cloud_object_labels)
    objects_ecc = _get_objects_property(
        cloud_object_labels=cloud_object_labels, property_name="eccentricity"
    )

    return np.sum(objects_area * objects_ecc) / np.sum(objects_area)


def mean_perimeter_length(cloud_object_labels):
    """
    Compute mean perimeter length of across all labeled objects in pixel units

    NOTE: the perimeter is calculated as the "Perimeter of object which
    approximates the contour as a line through the centers of border pixels
    using a 4-connectivity." This means that a object comprised of 3x3 pixels
    will have a perimeter length of 8 (2*4)
    """
    objects_perim = _get_objects_property(
        cloud_object_labels=cloud_object_labels, property_name="perimeter"
    )
    return np.mean(objects_perim)


def num_clouds(cloud_object_labels):
    regions = _get_regionprops(cloud_object_labels=cloud_object_labels)
    return len(regions)
