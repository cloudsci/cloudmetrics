#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from ..utils import compute_r_squared


def _debug_plot(mask, sizes, counts, fractal_dim, r_squared):
    fig, ax = plt.subplots(ncols=2, figsize=(8.25, 4))
    ax[0].imshow(mask, "gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].loglog(sizes, counts)
    ax[1].set_title("fracDim = %.4f" % fractal_dim)
    ax[1].annotate("rSq: %.3f" % r_squared, (0.7, 0.9), xycoords="axes fraction")
    ax[1].set_xlabel("Length")
    ax[1].set_ylabel("Number of edge boxes")
    plt.show()


def _boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1,
    )
    return len(np.where((S > 0) & (S < k * k))[0])


def fractal_dimension(mask, debug=False):
    """
    Compute box-counting dimension from a binary (cloud) mask. Adapted from:
    https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0

    Parameters
    ----------
    mask : numpy array of shape (npx,npx) - npx is number of pixels
           (cloud) mask field

    Returns
    -------
    fractal_dimension : float
        Fractal (box-counting) dimension.

    """
    Z = mask < 0.5
    p = min(Z.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))  # Number of extractable boxes
    sizes = 2 ** np.arange(n, 1, -1)  # Box sizes
    counts = np.zeros(len(sizes))
    for s in range(len(sizes)):
        counts[s] = _boxcount(Z, sizes[s])  # Non-empty/non-full box no.

    # Fit the relation: counts = coeffs[1]*sizes**coeffs[0]; coeffs[0]=-Nd
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    r_squared = compute_r_squared(
        lambda x, c: c[1] + c[0] * x, coeffs, np.log(sizes), np.log(counts)
    )
    fractal_dim = -coeffs[0]

    if debug:
        _debug_plot(mask, sizes, counts, fractal_dim, r_squared)

    return fractal_dim
