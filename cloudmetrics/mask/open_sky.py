#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def open_sky(mask, periodic_domain=False, debug=False):
    """
    Compute "open sky" metric proposed by Antonissen (2018) for a single
    (cloud) mask (see http://resolver.tudelft.nl/uuid:d868273a-b028-4273-8380-ff1628ecabd5).

    The method analyses rectangular reference areas in the scene defined by
    four extrema in east, west, north and south.  These points are the distance
    from each pixel where the mask is 0, to the nearest pixel in each direction
    where the mask is 1. The largest and average such area are both returned as
    metrics for the size of the scene's voids (contiguous areas where the mask is 0).

    Parameters
    ----------
    mask:            numpy array of shape (npx,npx) - npx is number of pixels
                     (cloud) mask field.
    periodic_domain: whether the provided (cloud) mask is on a periodic domain
                     (for example from a LES simulation)
    debug:           whether to produce debugging plot

    Returns
    -------
    os_max, os_avg : tuple of floats
        Maximum and average open sky parameter, assuming a rectangular reference area.

    """
    a_os_max = 0
    a_os_avg = 0
    for i in range(mask.shape[0]):  # rows
        cl_ew = np.where(mask[i, :] == 1)[0]  # pixels where mask=1
        for j in range(mask.shape[1]):  # cols
            if mask[i, j] != 1:

                # FIXME for speed -> do this once and store
                cl_ns = np.where(mask[:, j] == 1)[0]

                ws = np.where(cl_ew < j)[0]  # west side mask=1 pixels
                es = np.where(cl_ew > j)[0]  # east side
                ns = np.where(cl_ns < i)[0]  # north side
                ss = np.where(cl_ns > i)[0]  # south side

                # West side
                if ws.size == 0:  # if no pixels left of this pixel have mask=1
                    if periodic_domain and es.size != 0:
                        w = cl_ew[es[-1]] - mask.shape[1]
                    else:
                        w = 0
                else:
                    w = cl_ew[ws[-1]]

                # East side
                if es.size == 0:
                    if periodic_domain and ws.size != 0:
                        e = cl_ew[ws[0]] + mask.shape[1] - 1
                    else:
                        e = mask.shape[1]
                else:
                    e = cl_ew[es[0]] - 1

                # North side
                if ns.size == 0:
                    if periodic_domain and ss.size != 0:
                        n = cl_ns[ss[-1]] - mask.shape[0]
                    else:
                        n = 0
                else:
                    n = cl_ns[ns[-1]]

                # South side
                if ss.size == 0:
                    if periodic_domain and ns.size != 0:
                        s = cl_ns[ns[0]] + mask.shape[0] - 1
                    else:
                        s = mask.shape[0]
                else:
                    s = cl_ns[ss[0]] - 1

                a_os = (e - w) * (s - n)  # Assuming rectangular reference form

                a_os_avg += a_os
                if a_os > a_os_max:
                    a_os_max = a_os
                    osc = [i, j]
                    nmax, smax, emax, wmax = n, s, e, w
    a_os_avg = a_os_avg / mask[mask == 0].size / mask.size
    os_max = a_os_max / mask.size

    if debug:
        _debug_plot(mask=mask, osc=osc, wmax=wmax, nmax=nmax, emax=emax, smax=smax)

    return os_max, a_os_avg


def _debug_plot(mask, osc, wmax, nmax, emax, smax):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()
    ax.imshow(mask, "gray")
    rect = patches.Rectangle(
        (wmax, nmax),
        emax - wmax,
        smax - nmax,
        facecolor="none",
        edgecolor="C0",
        linewidth="3",
    )
    ax.add_patch(rect)
    ax.scatter(osc[1], osc[0], s=100)
    ax.set_axis_off()
    ax.set_title(
        "e: "
        + str(emax)
        + ", w: "
        + str(wmax)
        + ", n: "
        + str(nmax)
        + ", s: "
        + str(smax)
    )
    plt.show()
