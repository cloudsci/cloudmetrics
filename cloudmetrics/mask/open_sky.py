#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def _nearest_cloudy_bounds(cloudy, axis, periodic_domain):
    """
    For every pixel find the index of the nearest cloudy pixel (``cloudy ==
    True``) before (``lo``) and after (``hi``) it along ``axis``, excluding the
    pixel itself.

    Conventions (matching the original per-pixel implementation):

    - ``lo`` is the index of the last cloudy pixel before the pixel, ``0`` if
      there is none.
    - ``hi`` is the index of the first cloudy pixel after the pixel minus one,
      ``n`` (the axis length) if there is none.
    - On a periodic domain a missing bound is instead wrapped around to the
      cloudy pixel on the opposite side of the line: ``lo`` becomes the index
      of the last cloudy pixel in the line minus ``n`` and ``hi`` becomes the
      index of the first cloudy pixel in the line plus ``n - 1``. Lines without
      any cloudy pixel keep the non-periodic defaults.
    """
    cloudy = np.moveaxis(cloudy, axis, -1)
    n = cloudy.shape[-1]
    idx = np.arange(n, dtype=np.int64)
    idx = np.broadcast_to(idx, cloudy.shape)

    # index of the last cloudy pixel at or before each pixel (-1 if none) and
    # of the first cloudy pixel at or after each pixel (n if none)
    last_incl = np.maximum.accumulate(np.where(cloudy, idx, -1), axis=-1)
    first_incl = np.minimum.accumulate(np.where(cloudy, idx, n)[..., ::-1], axis=-1)
    first_incl = first_incl[..., ::-1]

    # shift by one pixel to exclude the pixel itself
    lo = np.empty_like(last_incl)
    lo[..., 0] = -1
    lo[..., 1:] = last_incl[..., :-1]
    hi = np.empty_like(first_incl)
    hi[..., -1] = n
    hi[..., :-1] = first_incl[..., 1:]

    no_lo = lo == -1
    no_hi = hi == n
    hi = hi - 1

    if periodic_domain:
        # last and first cloudy pixel of the whole line (for clear pixels
        # without a bound on one side these are the extremes of the other side)
        line_last = last_incl[..., -1:]
        line_first = first_incl[..., :1]
        has_cloud = line_last != -1
        lo = np.where(no_lo & has_cloud, line_last - n, lo)
        hi = np.where(no_hi & has_cloud, line_first + n - 1, hi)
        no_lo = no_lo & ~has_cloud
        no_hi = no_hi & ~has_cloud

    lo[no_lo] = 0
    hi[no_hi] = n

    return np.moveaxis(lo, -1, axis), np.moveaxis(hi, -1, axis)


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

    The nearest cloudy pixel in each direction is found for all pixels at once
    with cumulative minimum/maximum operations along the rows and columns, so
    the cost scales linearly with the number of pixels.

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
    mask = np.asarray(mask)
    cloudy = mask == 1

    if np.all(cloudy):
        # fully cloudy mask has no open sky
        return 0.0
    elif np.all(mask == 0):
        # no cloud mask is all open sky
        return 1.0

    if summary_measure not in ("max", "mean"):
        raise NotImplementedError(
            f"summary_measure `{summary_measure}` not implemented, "
            "use `max` or `mean`"
        )

    # nearest cloudy pixel west/east (along the columns) and north/south (along
    # the rows) of every pixel
    w, e = _nearest_cloudy_bounds(cloudy, axis=1, periodic_domain=periodic_domain)
    n, s = _nearest_cloudy_bounds(cloudy, axis=0, periodic_domain=periodic_domain)

    clear = ~cloudy
    a_os = np.where(clear, (e - w) * (s - n), -1)

    if summary_measure == "max":
        i_max = np.argmax(a_os)
        result = a_os.flat[i_max] / mask.size
    else:
        result = a_os[clear].sum(dtype=np.int64) / np.count_nonzero(clear) / mask.size

    if debug:
        i, j = np.unravel_index(np.argmax(a_os), mask.shape)
        _debug_plot(mask, [i, j], w[i, j], n[i, j], e[i, j], s[i, j])

    return float(result)


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
