import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._object_properties import _get_objects_property


def pair_correlation_2d(
    pos, domain_shape, dist_cutoff, dr, periodic_domain, normalize=True
):
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
        domain_shape    domain shape
        r_max           distance from (open) boundary where objects are ignored
        dr              increment for increasing radius of annulus
        periodic_domain assume periodic boundary conditions if `True`
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
    """
    # work out
    mins = np.min(pos, axis=0)
    maxs = np.max(pos, axis=0)
    # dimensions of box
    dims = maxs - mins
    r_max = (np.min(dims) / 2) * dist_cutoff

    # Extract domain size
    Sx, Sy = domain_shape

    # Number of particles in ring/area of ring/number of reference
    # particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Find particles which are close enough to the box center that a circle of radius
    # r_max will not cross any edge of the box, if no periodic bcs
    if not periodic_domain:
        bools1 = pos[:, 0] > r_max  # Valid centroids from left boundary
        bools2 = pos[:, 0] < (Sx - r_max)  # Valid centroids from right boundary
        bools3 = pos[:, 1] > r_max  # Valid centroids from top boundary
        bools4 = pos[:, 1] < (Sy - r_max)  # Valid centroids from bottom boundary
        (int_ind,) = np.where(bools1 * bools2 * bools3 * bools4)
    else:
        int_ind = np.arange(pos.shape[0])

    n_cl = len(int_ind)
    pos = pos[int_ind, :]

    # Make bins
    edges = np.arange(0.0, r_max + dr, dr)  # Annulus edges
    n_inc = len(edges) - 1
    g_rdf = np.zeros([n_cl, n_inc])  # RDF for all interior particles
    radii = np.zeros(n_inc)

    # Define normalisation based on the used region and particles
    if periodic_domain:
        number_density = float(pos.shape[0]) / float((Sx * Sy))
    else:
        number_density = float(pos.shape[0]) / float(((Sx - r_max) * (Sy - r_max)))

    # Compute pairwise distances
    if periodic_domain:
        dist_sq = np.zeros(n_cl * (n_cl - 1) // 2)  # to match the result of pdist
        for d in range(pos.shape[1]):
            box = domain_shape[pos.shape[1] - d - 1]  # to match x,y ordering in pos
            pos_1d = pos[:, d][:, np.newaxis]  # shape (N, 1)
            dist_1d = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d**2
        dist = np.sqrt(dist_sq)
    else:
        dist = pdist(pos)

    dist = squareform(dist)
    np.fill_diagonal(dist, 2 * r_max)  # Don't want distance to self to count

    # Count objects per ring
    for p in range(n_cl):
        result, bins = np.histogram(dist[p, :], bins=edges)
        if normalize:
            result = result / number_density
        g_rdf[p, :] = result

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(n_inc)
    for i in range(n_inc):
        radii[i] = (edges[i] + edges[i + 1]) / 2.0
        r_outer = edges[i + 1]
        r_inner = edges[i]
        g_average[i] = np.mean(g_rdf[:, i]) / (np.pi * (r_outer**2 - r_inner**2))

    return g_average, radii


def _plot(field, rad, rdf, rdfM, rdfI, rdfD):
    import matplotlib.pyplot as plt  # noqa

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


def estimate_rdf(object_labels, periodic_domain=False, dist_cutoff=0.9, dr=1):
    """
    Compute discrete estimate of the Radial Distribution Function between
    objects (rdf) and return its value at discrete distances (in pixel space).
    To get the distribution in terms of real-space distance you should scale
    the distance returned by the grid-spacing.

    NOTE: the implementation assumes that the underlying grid of the
    object-labels in isometric.

    The implementation is based off:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py

    Parameters
    ----------
    object_labels:
        numpy array of shape (npx,npx) - npx is number of pixels Cloud mask
        field.
    dist_cutoff:
        Maximum distance to compute radial distribution function over as
        fraction of the largest pair-wise distance between any two objects
    dr:
        radial bin width

    Returns
    -------
    rdf:
        numpy array of RDF value estimates
    pixel_dist:
        numpy array of pixel-space distances where the RDF values were estimated
    """

    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )
    if periodic_domain:
        raise NotImplementedError("not sure what value of S to use")
    else:
        domain_shape = object_labels.shape

    # TODO set dr based on field size and object number, results are
    # sensitive to this
    rdf, pixel_dist = pair_correlation_2d(
        pos=centroids,
        dist_cutoff=dist_cutoff,
        dr=dr,
        periodic_domain=periodic_domain,
        normalize=True,
        domain_shape=domain_shape,
    )
    return rdf, pixel_dist


_RDF_FUNC_DOCSTRING_TEMPLATE = """
    Compute discrete estimate of the Radial Distribution Function between
    objects (rdf) and compute the {metric} of this distribution
"""


def rdf_max_value(object_labels, periodic_domain=False, dist_cutoff=0.9, dr=1):
    rdf, _ = estimate_rdf(
        object_labels=object_labels,
        periodic_domain=periodic_domain,
        dist_cutoff=dist_cutoff,
        dr=dr,
    )
    return np.max(rdf)


rdf_max_value.__doc__ = _RDF_FUNC_DOCSTRING_TEMPLATE.format(metric="max-value")


def rdf_integral(object_labels, periodic_domain=False, dist_cutoff=0.9, dx=1, dr=1):
    rdf, pixel_dist = estimate_rdf(
        object_labels=object_labels,
        periodic_domain=periodic_domain,
        dist_cutoff=dist_cutoff,
        dr=dr,
    )
    rad = dx * pixel_dist

    return np.trapz(rdf, rad)


rdf_integral.__doc__ = _RDF_FUNC_DOCSTRING_TEMPLATE.format(metric="integral")
