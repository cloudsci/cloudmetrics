#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def open_sky(mask, summary_measure="max", periodic_domain=False, debug=False):
    """
    Compute "open sky" metric proposed by Antonissen (2018) for a single
    (cloud) mask (see http://resolver.tudelft.nl/uuid:d868273a-b028-4273-8380-ff1628ecabd5).

    The method analyses rectangular reference areas in the scene defined by
    four extrema in east, west, north and south.  These points are the distance
    from each pixel where the mask is 0, to the nearest pixel in each direction
    where the mask is 1. Both the largest (default) and average of such areas
    can be return as measures of size of the scene's voids (contiguous areas
    where the mask is 0).

    NOTE: for situations where the large clear-sky swaths are absent from the
    `mask` (for example in LES simulations) returning the `mean` rather than
    the `max` may be better for distinguishing scenes which are similar

    Parameters
    ----------
    mask:            numpy array of shape (npx,npx) - npx is number of pixels
                     (cloud) mask field.
    periodic_domain: whether the provided (cloud) mask is on a periodic domain
                     (for example from a LES simulation)
    debug:           whether to produce debugging plot
    summary_measure: measure used in summarising the open-sky areas found in mask

    Returns
    -------
    open_sky:        `summary_measure` (default "max") of open-sky regions
                     identified in mask

    """
    if np.all(mask == 1):
        # fully cloudy mask has no open sky
        return 0.0
    elif np.all(mask == 0):
        # no cloud mask is all open sky
        return 1.0

    npx_rows, npx_cols = mask.shape
    mask_0_indices = np.where(mask == 0)

    a_os_max = 0
    a_os_avg = 0

    for i in prange(npx_rows):
        for j in range(npx_cols):
            if mask[i, j] != 1:

                cl_ew = np.where(mask[i, :] == 1)[0]
                cl_ns = np.where(mask[:, j] == 1)[0]

                ws = cl_ew[cl_ew < j]
                es = cl_ew[cl_ew > j]
                ns = cl_ns[cl_ns < i]
                ss = cl_ns[cl_ns > i]

                w = (
                    ws[-1]
                    if ws.size > 0
                    else (es[-1] - npx_cols)
                    if periodic_domain and es.size > 0
                    else 0
                )
                e = (
                    es[0] - 1
                    if es.size > 0
                    else ws[0] + npx_cols - 1
                    if periodic_domain and ws.size > 0
                    else npx_cols
                )
                n = (
                    ns[-1]
                    if ns.size > 0
                    else (ss[-1] - npx_rows)
                    if periodic_domain and ss.size > 0
                    else 0
                )
                s = (
                    ss[0] - 1
                    if ss.size > 0
                    else ns[0] + npx_rows - 1
                    if periodic_domain and ns.size > 0
                    else npx_rows
                )

                a_os = (e - w) * (s - n)
                a_os_avg += a_os

                a_os_max = max(a_os_max, a_os)

    if summary_measure == "max":
        os_max = a_os_max / mask.size
        return os_max
    elif summary_measure == "mean":
        a_os_avg = a_os_avg / len(mask_0_indices[0]) / mask.size
        return a_os_avg


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
