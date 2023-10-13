import numpy as np
import scipy.spatial.distance as sd

from ._object_properties import _get_objects_property


def abcop(object_labels, min_area=0, periodic_domain=False):
    """
    Compute Area-Based Convective Organisation Potential (ABCOP) by Jin et al. 2022
    (https://doi.org/10.1029/2022JD036665). The ABCOP metric is a modified version
    of the COP metric. The differences are:
     - Interaction potential is computed based on areas rather than equivalent radii
     - Distances are adjusted to account for object sizes
     - Instead of calculating the mean, ABCOP involves the summation of maximum
       interaction potential pairs for each object.

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
    ABCOP : float
        Area-Based Convective Organisation Potential.

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


    Req   = np.sqrt(area) / np.sqrt(np.pi)          # equivalent radii
    dij   = sd.squareform(dist)                     # Pairwise distance matrix
    Rij   = ( Req[:, None] + Req[None, :] )         # Pairwise radii sum matrix
    d2ij  = np.maximum(1, dij - Rij)                # proxy of edge distances
    Aij   = 0.5 * ( area[:, None] + area[None, :] ) # Pairwise area sum matrix
    Vij   = Aij / ( d2ij * object_labels.size**0.5) # Pairwise interaction pot.
    np.fill_diagonal(Vij, np.nan)

    abcop = np.sum( np.nanmax(Vij, axis=0) )        # ABCOP


    return abcop
