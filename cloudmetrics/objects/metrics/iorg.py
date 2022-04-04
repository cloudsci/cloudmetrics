#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ...utils import find_nearest_neighbors
from ._object_properties import _get_objects_property


def _debug_plot_1(field, placed_circles):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim((0, field.shape[1]))
    ax.set_ylim((0, field.shape[0]))
    for i in range(len(placed_circles)):
        circ = plt.Circle(
            (placed_circles[i].xm, placed_circles[i].yp),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].x, placed_circles[i].yp),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].xp, placed_circles[i].yp),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].xm, placed_circles[i].y),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].x, placed_circles[i].y), placed_circles[i].r
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].xp, placed_circles[i].y),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].xm, placed_circles[i].ym),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].x, placed_circles[i].ym),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placed_circles[i].xp, placed_circles[i].ym),
            placed_circles[i].r,
        )
        ax.add_artist(circ)
    ax.grid(which="both")
    plt.show()


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


def _check_circle_overlap(new, placed_circles):
    """
    check if the circle overlaps with any circle in placed_circles
    or any of its periodic images
    """

    # By Fredrik Jansson: Return as soon as overlap is found, handle all
    # periodic images at once
    for c in placed_circles:
        dx = min(abs(c.x - new.x), abs(c.xm - new.x), abs(c.xp - new.x))
        dy = min(abs(c.y - new.y), abs(c.ym - new.y), abs(c.yp - new.y))
        if dx**2 + dy**2 <= (c.r + new.r) ** 2:
            return True
    return False


class CloudCircle:
    def __init__(self, r, sh, rng):
        self.x = rng.integers(0, sh[1] - 1)
        self.y = rng.integers(0, sh[0] - 1)

        self.xp = self.x + sh[1]
        self.yp = self.y + sh[0]

        self.xm = self.x - sh[1]
        self.ym = self.y - sh[0]

        self.r = r


def iorg(
    object_labels,
    periodic_domain=False,
    max_iterations=100,
    num_placements=1,
    random_seed=None,
    debug=False,
):
    """
    Compute the organisation index iOrg (Weger et al. 1992) from a labelled
    object, using an inhibition nearest neighbour distribution, as proposed by
    Benner & Curry (1998) and detailed in Antonissen (2019).

    Parameters
    ----------
    object_labels:   numpy array of shape (npx,npx) - npx is number of pixels
                     containing object labels
    periodic_domain: whether the provided cloud mask is on a periodic domain
                     (for example from a LES simulation)
    random_seed:     set the random seed used when placing circles
    max_iterations:  maximum number of iterations to use within the algorithm before giving up
    num_placements:  number of times the entire random circle placement routine
                     is carried out, and thus number of different random
                     nearest neighbour distance cumulative density functions
                     are computed. The function output is an average over the
                     num_placements different values of iorg this generates.

    Returns
    -------
    iOrg : float
        Organisation index from comparison to inhibition nearest neighbour
        distribution.

    """
    rng = np.random.default_rng(random_seed)

    centroids = _get_objects_property(
        object_labels=object_labels, property_name="centroid"
    )

    cd = _get_objects_property(
        object_labels=object_labels, property_name="equivalent_diameter"
    )
    cr = cd / 2.0

    cr = np.flip(np.sort(cr))  # Largest to smallest

    if centroids.shape[0] < 1:
        return float("nan")

    if periodic_domain:
        sh = [shd // 2 for shd in object_labels.shape]
        if sh[0] != sh[1]:
            raise NotImplementedError(sh)

        sz = np.min(sh)  # FIXME won't work for non-square domains

        # Move centroids outside the original domain into original domain
        centroids[centroids[:, 0] >= sh[1], 0] -= sh[1]
        centroids[centroids[:, 0] < 0, 0] += sh[1]
        centroids[centroids[:, 1] >= sh[0], 1] -= sh[0]
        centroids[centroids[:, 1] < 0, 1] += sh[0]
    else:
        sh = [shd for shd in object_labels.shape]
        sz = None

    nnd_scene = find_nearest_neighbors(centroids, sz)

    iOrgs = np.zeros(num_placements)
    for c in range(num_placements):
        # Attempt to randomly place all circles in scene without ovelapping
        i = 0
        placed_circles = []
        place_count = 0
        while i < len(cr) and place_count < max_iterations:
            new = CloudCircle(cr[i], sh, rng)
            placeable = True

            # If the circles overlap -> Place again
            if _check_circle_overlap(new, placed_circles):
                placeable = False
                place_count += 1

            if placeable:
                placed_circles.append(new)
                i += 1
                place_count = 0

        if place_count == max_iterations:
            # TODO should ideally start over again automatically
            warnings.warn("Unable to place circles in this image")
        else:
            if debug:
                cloud_mask = object_labels > 0
                _debug_plot_1(field=cloud_mask, placed_circles=placed_circles)

            # Compute the nearest neighbour distances

            # Gather positions in array
            pos_rand = np.zeros((len(placed_circles), 2))
            for i in range(len(placed_circles)):
                pos_rand[i, 0] = placed_circles[i].x
                pos_rand[i, 1] = placed_circles[i].y

            # If field has open bcs, do not compute nn distances using
            # periodic bcs
            nnd_rand = find_nearest_neighbors(pos_rand, sz)
            # nndScene  = find_nearest_neighbors(pos_scene,sz)

            # Old bin generation:
            # nbins = len(nndRand)+1
            # bmin = np.min([np.min(nndRand),np.min(nndScene)])
            # bmax = np.max([np.max(nndRand),np.max(nndScene)])
            # bins = np.linspace(bmin,bmax,nbins)

            # New:
            nbins = 10000  # <-- Better off fixing nbins at a very large number
            bins = np.linspace(0, np.sqrt(sh[0] ** 2 + sh[1] ** 2), nbins)

            nnd_cdf_rand = np.cumsum(np.histogram(nnd_rand, bins)[0]) / len(nnd_rand)
            nnd_cdf_scene = np.cumsum(np.histogram(nnd_scene, bins)[0]) / len(nnd_scene)

            # Compute Iorg
            iOrg = np.trapz(nnd_cdf_scene, nnd_cdf_rand)
            iOrgs[c] = iOrg
            if debug:
                _debug_plot_2(
                    field=cloud_mask,
                    pos_scene=centroids,
                    pos_rand=pos_rand,
                    nnd_cdf_rand=nnd_cdf_rand,
                    nnd_cdf_scene=nnd_cdf_scene,
                    iOrg=iOrg,
                )

    # print(iOrgs)
    return np.mean(iOrgs)
