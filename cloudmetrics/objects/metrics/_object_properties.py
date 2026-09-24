from math import pi as PI

import numpy as np
from skimage.measure import regionprops

# Properties for which an exact, vectorised numpy implementation exists.
# `skimage.measure.regionprops` is comparatively expensive (one python object
# per labelled region and lazily-evaluated properties), and the object metrics
# request these simple properties many times per scene. The functions below
# reproduce the regionprops values bit-for-bit for these properties, all other
# properties fall back to regionprops.
_FAST_PROPERTIES = (
    "area",
    "centroid",
    "equivalent_diameter",
    "equivalent_diameter_area",
)


# Compute regionprops for every image, for every metric, that is passed.
def _get_regionprops(object_labels):
    return regionprops(label_image=object_labels)


def _has_fast_path(object_labels, property_name):
    # regionprops only accepts integer label images, let it raise the
    # appropriate exception for anything else
    return property_name in _FAST_PROPERTIES and np.issubdtype(
        object_labels.dtype, np.integer
    )


def _get_objects_pixel_counts(object_labels):
    """
    Count the pixels belonging to each labelled object with `np.bincount`,
    ignoring the background (label 0).

    Returns
    -------
    label_values : 1-d numpy array of int
        The label values which are present in `object_labels`, in increasing
        order (the same order in which `skimage.measure.regionprops` returns
        the regions). Labels which are not present in `object_labels` are
        excluded, so this is robust to non-contiguous label images.
    pixel_counts : 1-d numpy array of int
        Number of pixels of every label value, indexed by label value (i.e.
        with length `object_labels.max() + 1`).
    """
    pixel_counts = np.bincount(object_labels.ravel())
    label_values = np.flatnonzero(pixel_counts[1:]) + 1
    return label_values, pixel_counts


def _get_objects_property_fast(object_labels, property_name):
    label_values, pixel_counts = _get_objects_pixel_counts(object_labels)

    if label_values.size == 0:
        # match `np.asarray([])` for an empty list of regions
        return np.asarray([])

    # regionprops multiplies the pixel count by the (unit) pixel area, which
    # makes the area a float
    area = pixel_counts[label_values].astype(np.float64)

    if property_name == "area":
        return area

    if property_name in ("equivalent_diameter", "equivalent_diameter_area"):
        # same expression as `RegionProperties.equivalent_diameter_area`
        ndim = object_labels.ndim
        return (2 * ndim * area / PI) ** (1 / ndim)

    if property_name == "centroid":
        flat_labels = object_labels.ravel()
        coords = np.unravel_index(np.arange(flat_labels.size), object_labels.shape)
        centroid = np.empty((label_values.size, object_labels.ndim), dtype=np.float64)
        for axis, coords_axis in enumerate(coords):
            coord_sums = np.bincount(
                flat_labels, weights=coords_axis, minlength=pixel_counts.size
            )
            centroid[:, axis] = coord_sums[label_values] / area
        return centroid

    raise NotImplementedError(property_name)


def _get_objects_property(object_labels, property_name):
    if _has_fast_path(object_labels=object_labels, property_name=property_name):
        return _get_objects_property_fast(
            object_labels=object_labels, property_name=property_name
        )

    regions = _get_regionprops(object_labels=object_labels)
    num_objects = len(regions)

    values = []
    for i in range(num_objects):
        value = getattr(regions[i], property_name)
        values.append(value)
    return np.asarray(values)


def _get_objects_area(object_labels):
    return _get_objects_property(object_labels=object_labels, property_name="area")


def _get_num_objects(object_labels):
    """
    Count the labelled objects, i.e. the number of distinct non-zero label
    values (identical to `len(regionprops(object_labels))`)
    """
    if not np.issubdtype(object_labels.dtype, np.integer):
        return len(_get_regionprops(object_labels=object_labels))

    label_values, _ = _get_objects_pixel_counts(object_labels)
    return len(label_values)
