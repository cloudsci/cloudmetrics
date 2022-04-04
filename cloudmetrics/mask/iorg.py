#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computaion of the organisation index iOrg (Weger et al. 1992) from a
cloud mask, using an inhibition nearest neighbour distribution, as proposed
by Benner & Curry (1998) and detailed in Antonissen (2019).
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

from ..utils import find_nearest_neighbors


def _debug_plot_1(field, placedCircles):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim((0, field.shape[1]))
    ax.set_ylim((0, field.shape[0]))
    for i in range(len(placedCircles)):
        circ = plt.Circle(
            (placedCircles[i].xm, placedCircles[i].yp),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].x, placedCircles[i].yp),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].xp, placedCircles[i].yp),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].xm, placedCircles[i].y),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle((placedCircles[i].x, placedCircles[i].y), placedCircles[i].r)
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].xp, placedCircles[i].y),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].xm, placedCircles[i].ym),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].x, placedCircles[i].ym),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
        circ = plt.Circle(
            (placedCircles[i].xp, placedCircles[i].ym),
            placedCircles[i].r,
        )
        ax.add_artist(circ)
    ax.grid(which="both")
    plt.show()


def _debug_plot_2(field, posScene, posRand, nndcdfRan, nndcdfSce, iOrg):
    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    axs[0].imshow(field, "gray")
    axs[0].set_title("Cloud mask of scene")

    axs[1].scatter(posScene[:, 0], field.shape[0] - posScene[:, 1], color="k", s=5)
    axs[1].set_title("Scene centroids")
    axs[1].set_xlim((0, field.shape[1]))
    axs[1].set_ylim((0, field.shape[0]))
    asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
    axs[1].set_aspect(asp)

    axs[2].scatter(posRand[:, 0], posRand[:, 1], color="k", s=5)
    axs[2].set_title("Random field centroids")
    asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
    axs[2].set_aspect(asp)

    axs[3].plot(nndcdfRan, nndcdfSce, "-", color="k")
    axs[3].plot(nndcdfRan, nndcdfRan, "--", color="k")
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


def _check_circle_overlap(new, placedCircles):
    """
    check if the circle overlaps with any circle in placedCircles
    or any of its periodic images
    """

    # By Fredrik Jansson: Return as soon as overlap is found, handle all
    # periodic images at once
    for c in placedCircles:
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
    cloud_mask,
    periodic_domain=False,
    max_iterations=100,
    num_placements=1,
    connectivity=1,
    area_min=4,
    random_seed=None,
    debug=False,
):
    """
    Compute the iOrg organisational index for a single cloud field

    Parameters
    ----------
    cloud_mask:      numpy array of shape (npx,npx) - npx is number of pixels
                     Cloud mask field.
    periodic_domain: whether the provided cloud mask is on a periodic domain
                     (for example from a LES simulation)
    random_seed:     set the random seed used when placing circles
    connectivity:    set whether diagonally neighbouring (connectivity=2) or
                     just x-y neighbours (connectivity=1) are considered joined
    area_min:        minimum area (number of pixels) of labeled regions to consider
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

    cmlab, num = label(cloud_mask, return_num=True, connectivity=connectivity)
    regions = regionprops(cmlab)

    cr = []
    xC = []
    yC = []
    for i in range(len(regions)):
        props = regions[i]
        if props.area > area_min:
            y0, x0 = props.centroid
            xC.append(x0)
            yC.append(y0)
            cr.append(props.equivalent_diameter / 2)

    posScene = np.vstack((np.asarray(xC), np.asarray(yC))).T
    cr = np.asarray(cr)
    cr = np.flip(np.sort(cr))  # Largest to smallest

    # print('Number of regions: ',posScene.shape[0],'/',num)

    if posScene.shape[0] < 1:
        return float("nan")

    if periodic_domain:
        sh = [shd // 2 for shd in cloud_mask.shape]
        sz = np.min(sh)  # FIXME won't work for non-square domains

        # Move centroids outside the original domain into original domain
        posScene[posScene[:, 0] >= sh[1], 0] -= sh[1]
        posScene[posScene[:, 0] < 0, 0] += sh[1]
        posScene[posScene[:, 1] >= sh[0], 1] -= sh[0]
        posScene[posScene[:, 1] < 0, 1] += sh[0]
    else:
        sh = [shd for shd in cloud_mask.shape]
        sz = None

    nndScene = find_nearest_neighbors(posScene, sz)

    iOrgs = np.zeros(num_placements)
    for c in range(num_placements):
        # Attempt to randomly place all circles in scene without ovelapping
        i = 0
        placedCircles = []
        placeCount = 0
        while i < len(cr) and placeCount < max_iterations:
            new = CloudCircle(cr[i], sh, rng)
            placeable = True

            # If the circles overlap -> Place again
            if _check_circle_overlap(new, placedCircles):
                placeable = False
                placeCount += 1

            if placeable:
                placedCircles.append(new)
                i += 1
                placeCount = 0

        if placeCount == max_iterations:
            # TODO should ideally start over again automatically
            print("Unable to place circles in this image")
        else:
            if debug:
                _debug_plot_1(field=cloud_mask, placedCircles=placedCircles)

            # Compute the nearest neighbour distances

            # Gather positions in array
            posRand = np.zeros((len(placedCircles), 2))
            for i in range(len(placedCircles)):
                posRand[i, 0] = placedCircles[i].x
                posRand[i, 1] = placedCircles[i].y

            # If field has open bcs, do not compute nn distances using
            # periodic bcs
            nndRand = find_nearest_neighbors(posRand, sz)
            # nndScene  = find_nearest_neighbors(posScene,sz)

            # Old bin generation:
            # nbins = len(nndRand)+1
            # bmin = np.min([np.min(nndRand),np.min(nndScene)])
            # bmax = np.max([np.max(nndRand),np.max(nndScene)])
            # bins = np.linspace(bmin,bmax,nbins)

            # New:
            nbins = 10000  # <-- Better off fixing nbins at a very large number
            bins = np.linspace(0, np.sqrt(sh[0] ** 2 + sh[1] ** 2), nbins)

            nndcdfRan = np.cumsum(np.histogram(nndRand, bins)[0]) / len(nndRand)
            nndcdfSce = np.cumsum(np.histogram(nndScene, bins)[0]) / len(nndScene)

            # Compute Iorg
            iOrg = np.trapz(nndcdfSce, nndcdfRan)
            iOrgs[c] = iOrg
            if debug:
                _debug_plot_2(
                    field=cloud_mask,
                    posScene=posScene,
                    posRand=posRand,
                    nndcdfRan=nndcdfRan,
                    nndcdfSce=nndcdfSce,
                    iOrg=iOrg,
                )

    # print(iOrgs)
    return np.mean(iOrgs)
