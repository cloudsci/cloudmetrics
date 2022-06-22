import numpy as np
import scipy.spatial.distance as sd

from ._object_properties import _get_objects_property


def cop(object_labels, min_area=0, periodic_domain=False):
    """
    Compute Convective Organisation Potential (COP) by White et al. 2018
    (https://doi.org/10.1175/JAS-D-16-0307.1)

    Parameters
    ----------
    object_labels : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    min_area : int
        Minimum cloud size (in number of pixels) considered in
        computing metric
    periodic_domain : bool (optional)
        Flag for whether to copute the measure with periodic boundary conditions.
        Default is False

    Returns
    -------
    COP : float
        Convective Organisation Potential.

    """
    area = _get_objects_property(object_labels=object_labels, property_name="area")
    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    idx_large_objects = area > min_area
    if np.count_nonzero(idx_large_objects) == 0:
        return float("nan")
    area = area[idx_large_objects]
    pos = centroids[idx_large_objects, :]
    nCl = len(area)

    # pairwise distances (handling periodic BCs)
    if periodic_domain:
        dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
        for d in range(object_labels.ndim):
            # Number of pixels in original field's dimension, assuming the
            # field was doubled
            box = object_labels.shape[d] // 2
            pos_1d = pos[:, d][:, np.newaxis]  # shape (N, 1)
            dist_1d = sd.pdist(pos_1d)  # shape (N * (N - 1) // 2, )
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d**2  # d^2 = dx^2 + dy^2 + dz^2
        dist = np.sqrt(dist_sq)
    else:
        dist = sd.pdist(pos)

    dij = sd.squareform(dist)  # Pairwise distance matrix
    dij = dij[np.triu_indices_from(dij, k=1)]  # Upper triangular (no diag)
    aSqrt = np.sqrt(area)  # Square root of area
    Aij = aSqrt[:, None] + aSqrt[None, :]  # Pairwise area sum matrix
    Aij = Aij[np.triu_indices_from(Aij, k=1)]  # Upper triangular (no diag)
    Vij = Aij / (dij * np.sqrt(np.pi))  # Pairwise interaction pot.
    cop = np.sum(Vij) / (0.5 * nCl * (nCl - 1))  # COP

    return cop
