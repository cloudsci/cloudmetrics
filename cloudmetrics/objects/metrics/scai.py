import numpy as np
import scipy as sc
import scipy.spatial.distance as sd
from scipy.spatial.distance import pdist
from scipy.stats.mstats import gmean

from ._object_properties import _get_objects_property


def scai1(
    object_labels,
    min_area=0,
    periodic_domain=False,
    return_nn_dist=False,
    reference_lengthscale=1000,
    dx=1,
):
    """
    compute the Simple Convective Aggregation Index (SCAI)
    (https://doi.org/10.1175/JCLI-D-11-00258.1)
    from a cloud mask, assuming distances are in pixels

    Parameters
    ----------
    object_labels : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    min_area : int
        Minimum area (in pixels) an object must have. Default is 0.
    periodic_domain : bool (optional)
        Flag for whether to copute the measure with periodic boundary conditions.
        Default is False
    return_nn_dist : bool
        Flag for whether to return the (geometric) mean nearest neighbour distance
        between object. Default is False.
    reference_lengthscale : float (optional)
        reference scale (in pixels) to divide the returned scai through. For
        similar interpretations as Tobin et al. (2012), set to the domain's
        characteristic length. Default is 1000
    dx : float (optional)
        Pixel size, for computing physical D0. Default is 1.


    Returns
    -------
    D0 : float
        Mean geometric nearest neighbour distance between objects. Only returned
        if return_nn_dist is True.
    scai : float
        Simple Convective Aggregation Index.

    """

    area = _get_objects_property(object_labels=object_labels, property_name="area")
    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )
    num_objects = len(area)
    idx_large_objects = area > min_area
    if np.count_nonzero(idx_large_objects) == 0:
        D0 = scai = np.nan

    else:
        area = area[idx_large_objects] * dx ** 2
        pos = centroids[idx_large_objects, :] * dx
        nCl = len(area)

        if periodic_domain:
            dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
            for d in range(object_labels.ndim):
                box = object_labels.shape[d] * dx // 2
                pos_1d = pos[:, d][:, np.newaxis]
                dist_1d = sd.pdist(pos_1d)
                dist_1d[dist_1d > box * 0.5] -= box
                dist_sq += dist_1d ** 2
            dist = np.sqrt(dist_sq)
        else:
            dist = sd.pdist(pos)

        D0 = gmean(dist)
        Nmax = object_labels.shape[0] * object_labels.shape[1] / 2
        scai = num_objects / Nmax * D0 / reference_lengthscale * 1000

        # Force SCAI to zero if there is only 1 region (completely aggregated)
        # This is not strictly consistent with the metric (as D0 is
        # objectively undefined), but is consistent with its spirit
        if pos.shape[0] == 1:
            scai = 0

    if return_nn_dist:
        return scai, D0
    return scai


def scai2(object_labels, min_area=0, periodic_domain=False, return_nn_dist=False):
    cloudmask = object_labels > 0
    area = _get_objects_property(object_labels=object_labels, property_name="area")
    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    connectivity = 1

    # number of cloud clusters
    N = len(area)

    # potential maximum of N depending on cloud connectivity
    N_max = np.sum(~np.isnan(cloudmask)) / 2
    if connectivity == 2:
        N_max = np.sum(~np.isnan(cloudmask)) / 4

    # distance between points (center of mass of clouds) in pairs
    di = pdist(centroids, "euclidean")
    # order-zero diameter
    D0 = sc.stats.mstats.gmean(di)
    print(di)

    # characteristic length of the domain (in pixels): diagonal of box
    L = np.sqrt(cloudmask.shape[0] ** 2 + cloudmask.shape[1] ** 2)

    print("L", L)
    print("Nmax", N_max)

    scai = N / N_max * D0 / L * 1000

    if return_nn_dist:
        return scai, D0
    return scai


scai = scai1
