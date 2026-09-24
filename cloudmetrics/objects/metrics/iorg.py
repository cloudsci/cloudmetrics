#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for computing iorg organisation index
"""

import matplotlib.pyplot as plt
import numpy as np

from ...utils import find_nearest_neighbors, trapezoid
from ._object_properties import _get_objects_property


def iorg(
    object_labels,
    periodic_domain=False,
    reference_dist="poisson",
    n_dist_bins=10000,
    reference_dist_kwargs={},
):
    """
    Compute the organisation index iOrg (Weger et al. 1992) from a labelled
    object.

    Parameters
    ----------
    object_labels:   numpy array of shape (npx,npx) or (npx*2, npy*2) if
                     `periodic_domain == True` - npx is number of pixels
                     containing object labels
    periodic_domain: whether the provided cloud mask is on a periodic domain
                     (for example from a LES simulation) it which case the
                     provided labelled objects are assumed to originate from
                     a domain in the top-left quarter of `object_labels`
    reference_dist:  reference distribution used in iorg calculation, currently
                     the following are implemented:
                       "poisson" (default):
                            Poisson distribution, as in Weger et al (1992)
                       "inhibition_nn":
                            use inhibition nearest neighbour distribution as
                            proposed in Benner & Curry (1998) and detailed in
                            Antonissen (2019).
    n_dist_bins:     number of bins to compute the distribution of the
                     nearest-neighbour distances over

    reference_dist_kwargs:
                     dictionary of keyword arguments to pass to reference
                     distribution function

    Returns
    -------
    iOrg: float
        Organisation index calculated from comparison to selected reference
        distribution

    """

    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    if centroids.shape[0] < 1:
        return float("nan")

    if periodic_domain:
        domain_shape = [shd // 2 for shd in object_labels.shape]
        if domain_shape[0] != domain_shape[1]:
            raise NotImplementedError(
                "iorg calculation isn't currently implemented for non-square"
                f"periodic domains {[domain_shape[0]]} != {domain_shape[1]}"
            )

        # won't work for non-square domains, if we fix this we can remove the
        # above exception
        nn_window = np.min(domain_shape)

        # Move centroids outside the original domain into original domain
        centroids[centroids[:, 0] >= domain_shape[1], 0] -= domain_shape[1]
        centroids[centroids[:, 0] < 0, 0] += domain_shape[1]
        centroids[centroids[:, 1] >= domain_shape[0], 1] -= domain_shape[0]
        centroids[centroids[:, 1] < 0, 1] += domain_shape[0]
    else:
        domain_shape = list(object_labels.shape)
        nn_window = None

    nnd_scene = find_nearest_neighbors(centroids, nn_window)

    n_dist_bins = 10000
    bin_max = np.sqrt(domain_shape[0] ** 2 + domain_shape[1] ** 2)
    dist_bins = np.linspace(0, bin_max, n_dist_bins)
    nnd_pdf_scene = np.histogram(nnd_scene, dist_bins)[0]
    nnd_cdf_scene = np.cumsum(nnd_pdf_scene) / len(nnd_scene)

    if reference_dist == "poisson":
        lam = nnd_scene.shape[0] / (domain_shape[0] * domain_shape[1])
        binav = (dist_bins[1:] + dist_bins[:-1]) / 2
        nnd_cdf_rand = 1 - np.exp(-lam * np.pi * binav**2)
    elif reference_dist == "inhibition_nn":
        object_diameters = _get_objects_property(
            object_labels=object_labels, property_name="equivalent_diameter"
        )
        object_radii = object_diameters / 2.0
        object_radii = np.flip(np.sort(object_radii))  # Largest to smallest
        nnd_cdf_rand = _compute_inhibition_nearest_neighbour_distribution(
            object_radii=object_radii,
            nn_window=nn_window,
            domain_shape=domain_shape,
            dist_bins=dist_bins,
            **reference_dist_kwargs,
        )
    else:
        raise NotImplementedError(reference_dist)

    iorg_value = trapezoid(nnd_cdf_scene, nnd_cdf_rand)

    return iorg_value


def _compute_inhibition_nearest_neighbour_distribution(
    object_radii,
    nn_window,
    domain_shape,
    dist_bins,
    max_iterations=100,
    debug=False,
    random_seed=None,
):
    """
    Compute a reference nearest-neigbour distance distribution using the the
    inhibition nearest neighbour method as proposed by Benner & Curry (1998)
    and detailed in Antonissen (2019). The method attempts to randomly place
    circular objects (using the provided object radii) in the provided domain

    Parameters
    ----------
    object_radii:   numpy array of object radii to attempt to place in the domain
    max_iterations:  maximum number of iterations to use within the algorithm
                     before giving up
    num_placements:  number of times the entire random circle placement routine
                     is carried out, and thus number of different random
                     nearest neighbour distance cumulative density functions
                     are computed. The function output is an average over the
                     num_placements different values of iorg this generates.
    dist_bins:       distance bins to use when computing comulative distribution
                     function for nearest-neighbour distances of placed object
    """
    rng = np.random.default_rng(random_seed)
    # Attempt to randomly place all circles in scene without ovelapping
    pos_rand = _place_circles_randomly(
        object_radii=object_radii,
        domain_shape=domain_shape,
        rng=rng,
        max_iterations=max_iterations,
    )

    # If field has open bcs, do not compute nn distances using
    # periodic bcs
    nnd_rand = find_nearest_neighbors(pos_rand, nn_window)
    nnd_cdf_rand = np.cumsum(np.histogram(nnd_rand, dist_bins)[0]) / len(nnd_rand)

    return nnd_cdf_rand


def _place_circles_randomly(object_radii, domain_shape, rng, max_iterations=100):
    """
    Randomly place circles with the given radii one after the other in a
    domain of shape `domain_shape`, such that no circle overlaps with an
    already placed circle (or any of its periodic images). A circle that
    overlaps is drawn again, up to `max_iterations` consecutive times, after
    which an exception is raised.

    The overlap test is vectorised over all previously placed circles, so
    placing N circles requires O(N) vectorised numpy calls rather than O(N^2)
    python-level iterations.

    Parameters
    ----------
    object_radii:   numpy array of circle radii to place, in placement order
    domain_shape:   (ny, nx) shape of the domain to place the circles in
    rng:            numpy random Generator used to draw the circle positions
    max_iterations: maximum number of consecutive failed placement attempts
                    for a single circle before giving up

    Returns
    -------
    pos_rand: numpy array of shape (len(object_radii), 2) with the (x, y)
              positions of the placed circles
    """
    n_circles = len(object_radii)
    ny, nx = domain_shape[0], domain_shape[1]

    # positions and radii of the circles placed so far (first `i` entries)
    xs = np.zeros(n_circles, dtype=np.int64)
    ys = np.zeros(n_circles, dtype=np.int64)
    rs = np.zeros(n_circles, dtype=np.float64)

    i = 0
    place_count = 0
    while i < n_circles and place_count < max_iterations:
        r = object_radii[i]
        x = rng.integers(0, nx - 1)
        y = rng.integers(0, ny - 1)

        # If the circle overlaps with any placed circle -> Place again
        if _check_circle_overlap(
            x, y, r, xs[:i], ys[:i], rs[:i], domain_shape=(ny, nx)
        ):
            place_count += 1
        else:
            xs[i] = x
            ys[i] = y
            rs[i] = r
            i += 1
            place_count = 0

    if place_count == max_iterations:
        raise Exception("Unable to place circles in this image")

    # Gather positions in array
    pos_rand = np.zeros((n_circles, 2))
    pos_rand[:, 0] = xs
    pos_rand[:, 1] = ys

    return pos_rand


def _debug_plot_2(field, pos_scene, pos_rand, nnd_cdf_rand, nnd_cdf_scene, iOrg):
    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    axs[0].imshow(field, "gray")
    axs[0].set_title("Cloud mask of scene")

    axs[1].scatter(pos_scene[:, 0], field.shape[0] - pos_scene[:, 1], color="k", s=5)
    axs[1].set_title("Scene centroids")
    axs[1].set_xlim((0, field.shape[1]))
    axs[1].set_ylim((0, field.shape[0]))
    asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
    axs[1].set_aspect(asp)

    axs[2].scatter(pos_rand[:, 0], pos_rand[:, 1], color="k", s=5)
    axs[2].set_title("Random field centroids")
    asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
    axs[2].set_aspect(asp)

    axs[3].plot(nnd_cdf_rand, nnd_cdf_scene, "-", color="k")
    axs[3].plot(nnd_cdf_rand, nnd_cdf_rand, "--", color="k")
    axs[3].set_title("Nearest neighbour distribution")
    axs[3].set_xlabel("Random field nearest neighbour CDF")
    axs[3].set_ylabel("Scene nearest neighbour CDF")
    axs[3].annotate(
        r"$I_{org} = $" + str(round(iOrg, 3)),
        (0.7, 0.1),
        xycoords="axes fraction",
    )
    asp = np.diff(axs[3].get_xlim())[0] / np.diff(axs[3].get_ylim())[0]
    axs[3].set_aspect(asp)
    plt.show()


def _check_circle_overlap(x, y, r, xs, ys, rs, domain_shape):
    """
    check if the circle at (x, y) with radius r overlaps with any of the
    circles at (xs, ys) with radii rs or any of their periodic images
    """
    ny, nx = domain_shape
    # By Fredrik Jansson: handle all periodic images at once. As all
    # positions lie within [0, nx) x [0, ny) the minimum over the three
    # images {c - L, c, c + L} of |c - p| is min(|c - p|, L - |c - p|)
    dx = np.abs(xs - x)
    dx = np.minimum(dx, nx - dx)
    dy = np.abs(ys - y)
    dy = np.minimum(dy, ny - dy)
    return bool(np.any(dx**2 + dy**2 <= (rs + r) ** 2))
