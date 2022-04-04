import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._object_properties import _get_objects_property


def pair_correlation_2d(pos, S, r_max, dr, periodic_domain, normalize=True):
    """
    Pair correlation function, adapted from:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py
    and indirectly from Rasp et al. (2018)

    Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius r_max drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller r_max...or write some code to handle edge effects! ;)

    Arguments:
        pos             array of positions of shape (nPos,nDim), columns are
                        orderered x, y, ...
        S               length of each side of the square region of the plane
        r_max           distance from (open) boundary where objects are ignored
        dr              increment for increasing radius of annulus
        periodic_domain assume periodic boundary conditions if `True`
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """

    # Number of particles in ring/area of ring/number of reference
    # particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Extract domain size
    (Sy, Sx) = S if len(S) == 2 else (S, S)

    # Find particles which are close enough to the box center that a circle of radius
    # r_max will not cross any edge of the box, if no periodic bcs
    if periodic_domain != "periodic":
        bools1 = pos[:, 0] > r_max  # Valid centroids from left boundary
        bools2 = pos[:, 0] < (Sx - r_max)  # Valid centroids from right boundary
        bools3 = pos[:, 1] > r_max  # Valid centroids from top boundary
        bools4 = pos[:, 1] < (Sy - r_max)  # Valid centroids from bottom boundary
        (int_ind,) = np.where(bools1 * bools2 * bools3 * bools4)
    else:
        int_ind = np.arange(pos.shape[0])

    nCl = len(int_ind)
    pos = pos[int_ind, :]

    # Make bins
    edges = np.arange(0.0, r_max + dr, dr)  # Annulus edges
    nInc = len(edges) - 1
    g = np.zeros([nCl, nInc])  # RDF for all interior particles
    radii = np.zeros(nInc)

    # Define normalisation based on the used region and particles
    if periodic_domain == "periodic":
        number_density = float(pos.shape[0]) / float((Sx * Sy))
    else:
        number_density = float(pos.shape[0]) / float(((Sx - r_max) * (Sy - r_max)))

    # Compute pairwise distances
    if periodic_domain == "periodic":
        dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
        for d in range(pos.shape[1]):
            box = S[pos.shape[1] - d - 1]  # to match x,y ordering in pos
            pos_1d = pos[:, d][:, np.newaxis]  # shape (N, 1)
            dist_1d = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d ** 2
        dist = np.sqrt(dist_sq)
    else:
        dist = pdist(pos)

    dist = squareform(dist)
    np.fill_diagonal(dist, 2 * r_max)  # Don't want distance to self to count

    # Count objects per ring
    for p in range(nCl):
        result, bins = np.histogram(dist[p, :], bins=edges)
        if normalize:
            result = result / number_density
        g[p, :] = result

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(nInc)
    for i in range(nInc):
        radii[i] = (edges[i] + edges[i + 1]) / 2.0
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter ** 2 - rInner ** 2))

    return g_average, radii, int_ind


def _plot(field, rad, rdf, rdfM, rdfI, rdfD):
    import matplotlib.pyplot as plt

    axF = "axes fraction"
    fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
    axs[0].imshow(field, "gray")
    axs[0].axis("off")

    axs[1].plot(rad, rdf)
    axs[1].set_xlabel("Distance")
    axs[1].set_ylabel("RDF")
    axs[1].annotate("rdfMax = %.3f" % rdfM, (0.6, 0.15), xycoords=axF)
    axs[1].annotate("rdfInt = %.3f" % rdfI, (0.6, 0.10), xycoords=axF)
    axs[1].annotate("rdfDif = %.3f" % rdfD, (0.6, 0.05), xycoords=axF)
    plt.show()


def metric(object_labels, periodic_domain=False, r_max=20, dx=1, dr=1, min_area=0):
    """
    Compute the Radial Distribution Function between objects (rdf)
    and derived metrics, from a cloud mask. Can compute the maximum of the
    RDF (rdfMax), the difference between minimum and maximum (rdfDiff). The
    implementation is based off:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py

    Parameters
    ----------
    object_labels : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    S     : Tuple of the input field's size (is different from field.shape
            if periodic BCs are used)
    r_max : How far away to compute the rdf FIXME This is sensitive to each case!!
    dx    : pixel resolution
    dr    : radial bin width

    Returns
    -------
    rdfM : float
        Maximum of the radial distribution function.
    rdfI : float
        Integral of the radial distribution function.
    rdfD : float
        Max-min difference of the radial distribution function.

    """
    raise NotImplementedError("not sure what value of S to use")
    S = None
    area = _get_objects_property(object_labels=object_labels, property_name="area")
    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    idx_large_objects = area > min_area
    if np.count_nonzero(idx_large_objects) == 0:
        return float("nan"), float("nan"), float("nan")

    area = area[idx_large_objects]
    pos = centroids[idx_large_objects, :]

    # TODO set dr based on field size and object number, results are
    # sensitive to this
    rdf, rad, tmp = pair_correlation_2d(
        pos=pos,
        S=S,
        r_max=r_max,
        dr=dr,
        periodic_domain=periodic_domain,
        normalize=True,
    )
    rad *= dx
    rdfM = np.max(rdf)
    rdfI = np.trapz(rdf, rad)
    rdfD = np.max(rdf) - rdf[-1]

    return rdfM, rdfI, rdfD
