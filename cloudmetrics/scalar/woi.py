#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pywt

_CACHED_VALUES = dict()


def _get_swt(scalar_field, pad_method, wavelet, separation_scale):
    # use python's memory ID of the swt dict for a poor-mans caching to
    # avoid recalculating it
    array_id = id(scalar_field)
    if array_id in _CACHED_VALUES:
        return _CACHED_VALUES[array_id]

    swt = compute_swt(scalar_field, pad_method, wavelet, separation_scale)
    _CACHED_VALUES[array_id] = swt
    return swt


def _debug_plot(scalar_field, k, specs):
    labs = ["Horizontal", "Vertical", "Diagonal"]
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    axs[0].imshow(scalar_field, "gist_ncar")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("CWP")
    for i in range(3):
        axs[1].plot(k[1:], specs[:, i], label=labs[i])
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"Scale number $k$")
    axs[1].set_ylabel("Energy")
    axs[1].set_title("Wavelet energy spectrum")
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def compute_swt(scalar_field, pad_method, wavelet, separation_scale, debug=False):
    """
    Computes the stationary/undecimated Direct Wavelet Transform
    (SWT, https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html#multilevel-2d-swt2)
    of a scalar field. See the documentation of woi1 for additional details.

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        (Cloud) scalar input.
    pad_method : string, optional
        Which type of padding to use,
    wavelet : string, optional
        Which wavelet to use.
    separation_scale : int, optional
        Which power of 2 to use as a cutoff scale that separates 'small' scales
        from 'large' scales.
    return_spectra : bool, optional
        Whether to return the spectra from the wavelet transform. The default is False.
    debug : bool, optional
        If True, plots the averaged wavelet spectra in horizontal, vertical and
        diagonal directions. The default is False.

    Returns
    -------
    Ebar : float
        Direction-averaged, squared coefficients of the SWT
    Elbar : float
        Direction-averaged, squared coefficients of the SWT, over scales larger than
        `separation_scale` (inclusive)
    Esbar : float
        Direction-averaged, squared coefficients of the SWT, over scales smaller than
        `separation_scale` (exclusive)
    Eld : numpy array of shape (3,)
        Sum of squared coefficients of the SWT, over scales larger than
        `separation_scale` (inclusive), in the horizontal, vertical and diagonal
        direction
    Esd : float
        Sum of squared coefficients of the SWT, over scales smaller than
        `separation_scale` (exclusive), in the horizontal, vertical and diagonal
        direction
    """

    # Pad if necessary
    pad_sequence = []
    scale_i = []
    for shi in scalar_field.shape:
        pow2 = np.log2(shi)
        pow2 = int(pow2 + 1) if pow2 % 1 > 0 else int(pow2)
        pad = (2**pow2 - shi) // 2
        pad_sequence.append((pad, pad))
        scale_i.append(pow2)
    scalar_field = pywt.pad(scalar_field, pad_sequence, pad_method)

    # Compute wavelet coefficients
    scale_max = np.max(scale_i)  # FIXME won't work for non-square scenes
    coeffs = pywt.swt2(scalar_field, wavelet, scale_max, norm=True, trim_approx=True)

    # Structure of coeffs:
    # - coeffs    -> list with n_scales indices. Each scale is a 2-power of
    #                the image resolution. For 512x512 images we have
    #                512 = 2^9 -> 10 scales
    # - coeffs[i] -> Contains three directions:
    #                   [0] - Horizontal
    #                   [1] - Vertical
    #                   [2] - Diagonal

    specs = np.zeros((len(coeffs), 3))  # Shape (n_scales,3)
    k = np.arange(0, len(specs))
    for i in range(len(coeffs)):
        if i == 0:
            ec = coeffs[i] ** 2
            specs[i, 0] = np.mean(ec)
        else:
            for j in range(len(coeffs[i])):
                ec = coeffs[i][j] ** 2  # Energy -> squared wavelet coeffs
                specs[i, j] = np.mean(ec)  # Domain-averaging at each scale

    # Decompose into ''large scale'' energy and ''small scale'' energy
    # Large scales are defined as 0 < k < separation_scale
    specs = specs[
        1:
    ]  # Remove first (mean) component, as it always distributes over horizontal dimension
    specL = specs[:separation_scale, :]
    specS = specs[separation_scale:, :]

    # Average over scales
    Ebar = np.sum(np.mean(specs, axis=1))
    Elbar = np.sum(np.mean(specL, axis=1))
    Esbar = np.sum(np.mean(specS, axis=1))

    # Sum over large/small scales
    Eld = np.sum(specL, axis=0)
    Esd = np.sum(specS, axis=0)

    if debug:
        _debug_plot(scalar_field, k, specs)
    return Ebar, Elbar, Esbar, Eld, Esd


def woi1(scalar_field, pad_method="periodic", wavelet="haar", separation_scale=5):
    """
    Computes the first Wavelet Organisation Index WOI1 proposed by
    Brune et al. (2018) https://doi.org/10.1002/qj.3409 from the stationary/undecimated
    Direct Wavelet Transform (https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html#multilevel-2d-swt2)
    of a scalar field. Based off https://rdrr.io/cran/calcWOI/, but does not
    mirror, taper or blow the `scalar_field` up. Instead, preprocessing the
    `scalar_field` input is limited to padding fields that do not have
    dimensions that are a power of 2 (all padding methods in pywt are available).

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        (Cloud) scalar input. Can be any field of choice (Brune et al. (2018) use
        rain rates; Janssens et al. (2021) use liquid water path).
    periodic_domain : Bool, optional
        Whether the domain is periodic. If False, mirror the domain in all
        directions before performing the SWT. The default is False.
    pad_method : string, optional
        Which type of padding to use, in case the field shape is not a power of 2.
        The default is 'periodic'. Other options can be found here:
        https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
    wavelet : string, optional
        Which wavelet to use. The default is 'haar'. Other options can be found here:
        https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
    separation_scale : int, optional
        Which power of 2 to use as a cutoff scale that separates 'small' scales
        from 'large' scales. The default is 5; i.e. energy contained in scales
        larger than 2^5=32 pixles is considered 'large-scale energy'.

    Returns
    -------
    woi1 : float
        First wavelet organisation index.
    """
    Ebar, Elbar, Esbar, Eld, Esd = _get_swt(
        scalar_field, pad_method, wavelet, separation_scale
    )
    return Elbar / Ebar


def woi2(scalar_field, pad_method="periodic", wavelet="haar", separation_scale=5):
    """
    Computes the second Wavelet Organisation Index WOI2 proposed by
    Brune et al. (2018) https://doi.org/10.1002/qj.3409 (see :func:`cloudmetrics.metrics.woi1`
    for more details)

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        (Cloud) scalar input.
    periodic_domain : Bool, optional
        Whether the domain is periodic.
    pad_method : string, optional
        Which type of padding to use, in case the field shape is not a power of 2.
    wavelet : string, optional
        Which wavelet to use. The default is 'haar'.
    separation_scale : int, optional
        Which power of 2 to use as a cutoff scale that separates 'small' scales
        from 'large' scales.

    Returns
    -------
    woi2 : float
        Second wavelet organisation index.
    """

    Ebar, Elbar, Esbar, Eld, Esd = _get_swt(
        scalar_field, pad_method, wavelet, separation_scale
    )
    return (Elbar + Esbar) / scalar_field[scalar_field > 0].size


def woi3(scalar_field, pad_method="periodic", wavelet="haar", separation_scale=5):
    """
    Computes the second Wavelet Organisation Index WOI3 proposed by
    Brune et al. (2018) https://doi.org/10.1002/qj.3409 (see :func:`cloudmetrics.metrics.woi1`
    for more details)

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        (Cloud) scalar input.
    periodic_domain : Bool, optional
        Whether the domain is periodic.
    pad_method : string, optional
        Which type of padding to use, in case the field shape is not a power of 2.
    wavelet : string, optional
        Which wavelet to use. The default is 'haar'.
    separation_scale : int, optional
        Which power of 2 to use as a cutoff scale that separates 'small' scales
        from 'large' scales.

    Returns
    -------
    woi3 : float
        Third wavelet organisation index.
    """

    Ebar, Elbar, Esbar, Eld, Esd = _get_swt(
        scalar_field, pad_method, wavelet, separation_scale
    )

    if Elbar == 0:
        woi3 = 1.0 / 3 * np.sum((Esd - Esbar) / Esbar)
    elif Esbar == 0:
        woi3 = 1.0 / 3 * np.sum((Eld - Elbar) / Elbar)
    else:
        woi3 = (
            1.0
            / 3
            * np.sqrt(
                np.sum(((Esd - Esbar) / Esbar) ** 2 + ((Eld - Elbar) / Elbar) ** 2)
            )
        )
    return woi3
