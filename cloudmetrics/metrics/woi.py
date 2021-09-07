#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pywt

def _debug_plot(cloud_scalar, k, specs):
    labs = ["Horizontal", "Vertical", "Diagonal"]
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    axs[0].imshow(cloud_scalar, "gist_ncar")
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


def woi(cloud_scalar, pad_method='periodic', wavelet='haar', separation_scale=5, debug=False, return_spectra=False):
    """
    Computes the three Wavelet Organisation Indices WOI1, WOI2, WOI3 proposed by 
    Brune et al. (2018) from the stationary/undecimated Direct Wavelet Transform
    of a scalar field.

    Parameters
    ----------
    cloud_scalar : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud scalar input. Can be any field of choice (Brune et al. (2018) use 
        rain rates; Janssens et al. (2021) use liquid water path).
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
    return_spectra : bool, optional
        Whether to return the spectra from the wavelet transform. The default is False.
    debug : bool, optional
        If True, plots the averaged wavelet spectra in horizontal, vertical and
        diagonal directions. The default is False.

    Returns
    -------
    woi1 : float
        First wavelet organisation index. 
    woi2 : float
        Second wavelet organisation index
    woi3 : float
        Third wavelet organisation index
    spectra: float
    
    """

    # Pad if necessary
    pad_sequence = []
    for shi in cloud_scalar.shape:
        l2 = int(np.log2(shi))+1
        pad = (2**l2 - shi) // 2
        pad_sequence.append((pad,pad))
    cloud_scalar = pywt.pad(cloud_scalar, pad_sequence, pad_method)

    # Compute wavelet coefficients
    scale_max = int(np.log(cloud_scalar.shape[0]) / np.log(2))
    coeffs = pywt.swt2(cloud_scalar, wavelet, scale_max, norm=True, trim_approx=True)
    # Bug in pywt -> trim_approx=False does opposite of its intention
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
    specs = specs[1:]
    specL = specs[:separation_scale, :]
    specS = specs[separation_scale:, :]

    # Average over scales
    Ebar = np.sum(np.mean(specs, axis=1))
    Elbar = np.sum(np.mean(specL, axis=1))
    Esbar = np.sum(np.mean(specS, axis=1))

    # Sum over large/small scales
    Eld = np.sum(specL, axis=0)
    Esd = np.sum(specS, axis=0)

    # Wavelet organisation indices
    woi1 = Elbar / Ebar
    woi2 = (Elbar + Esbar) / cloud_scalar[cloud_scalar>0].size
    woi3 = 1.0 / 3 * np.sqrt(np.sum(((Esd - Esbar) / Esbar) ** 2 + ((Eld - Elbar) / Elbar) ** 2))

    if debug:
        _debug_plot(cloud_scalar, k, specs)

    if return_spectra:
        return woi1, woi2, woi3, specs
    else:
        return woi1, woi2, woi3

# + No need to cache or recompute information within individual wavelet functions
# + Can easily return the spectra this way without having to add another external function (spectra are needed for testing)
# + Easier to pass all the optional arguments to a single woi function than to have to do it for all of them
# - Inconsistent with the 1 function 1 metric structure
# - Inconsistent with Fourier spectral metrics