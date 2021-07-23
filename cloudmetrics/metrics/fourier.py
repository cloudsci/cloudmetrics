#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage, linalg
from .utils import compute_r_squared

def _get_rad(data):
    h = data.shape[0]
    hc = h // 2
    w = data.shape[1]
    wc = w // 2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc).astype(int)

    return r


def _get_psd_1d_radial(psd_2d):
    r = _get_rad(psd_2d)
    sh = np.min(psd_2d.shape) // 2
    N = np.ones(psd_2d.shape)

    # SUM all psd_2d pixels with label 'r' for 0<=r<=wc
    # Will miss power contributions in 'corners' r>wc
    r_ran = np.arange(1, sh + 1)
    rk = np.flip(-(fftpack.fftfreq(int(2 * sh), 1))[sh:])
    psd_1d = ndimage.sum(psd_2d, r, index=r_ran)
    Ns = ndimage.sum(N, r, index=r_ran)
    psd_1d = psd_1d / Ns * rk * 2 * np.pi

    return psd_1d


def _get_psd_1d_azimuthal(psd_2d, d_theta=5, r_min=0, r_max=255, return_sectors=False):
    h = psd_2d.shape[0]
    w = psd_2d.shape[1]
    wc = w // 2
    hc = h // 2

    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of d_theta
    Y, X = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y - hc), (X - wc)))
    theta = np.mod(theta + d_theta / 2 + 360, 360)
    theta = d_theta * (theta // d_theta)
    theta = theta.astype(int)

    # mask below r_min and above r_max by setting to -100
    R = np.hypot(-(Y - hc), (X - wc))
    mask = np.logical_and(R > r_min, R < r_max)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100

    # SUM all psd_2d pixels with label 'theta' for 0<=theta<360 between r_min
    # and r_max
    sectors = np.arange(0, 360, int(d_theta))
    psd_1d = ndimage.sum(psd_2d, theta, index=sectors)

    # normalize each sector to the total sector power
    pwr_total = np.sum(psd_1d)
    psd_1d = psd_1d / pwr_total

    if return_sectors:
        return sectors, psd_1d
    else:
        return psd_1d


def _detrend(data, regressors):
    # From https://neurohackweek.github.io/image-processing/02-detrending/
    regressors = np.vstack([r.ravel() for r in regressors]).T
    solution = linalg.lstsq(regressors, data.ravel())
    beta_hat = solution[0]
    trend = np.dot(regressors, beta_hat)
    detrended = data - np.reshape(trend, data.shape)
    return detrended, beta_hat


def _hann_rad(data):
    r = _get_rad(data)
    sh = np.min(data.shape)
    shc = sh // 2

    fac = (3.0 * np.pi / 8 - 2 / np.pi) ** (-0.5)
    # fac  = 1
    w_han = fac * (1 + np.cos(2 * np.pi * r / sh))
    w_han[r >= shc] = 0

    return data * w_han


def _welch_rad(data):
    r = _get_rad(data)
    sh = np.min(data.shape)
    shc = sh // 2

    w_wel = 1 - (r / shc) ** 2
    w_wel[r >= shc] = 0

    return data * w_wel


def _planck_rad(data, eps=0.1):
    r = _get_rad(data)
    N = np.min(data.shape)
    N2 = N // 2
    eps_N = int(eps * N)
    n = N2 - r

    w_planck = np.zeros(data.shape)
    w_planck[(n < eps_N) & (n >= 1)] = (
        1
        + np.exp(
            eps_N / n[(n < eps_N) & (n >= 1)] - eps_N / (eps_N - n[(n < eps_N) & (n >= 1)])
        )
    ) ** (-1)
    w_planck[eps_N <= n] = 1

    return data * w_planck

def _bin_average(k1d, psd_1d_rad, n_bins):
    
    # Average over bins
    k0 = np.log10(k1d[0])
    kmax = np.log10(k1d[-1])
    bins_edges = np.logspace(k0, kmax, n_bins + 1)
    bins_centres = np.exp((np.log(bins_edges[1:]) + np.log(bins_edges[:-1])) / 2)
    psd_bin = np.zeros(len(bins_edges) - 1)
    for i in range(len(bins_edges) - 1):
        imax = np.where(k1d < bins_edges[i + 1])[0][-1]
        imin = np.where(k1d >= bins_edges[i])[0]
        if len(imin) == 0:
            continue  # You have gone beyond the available wavenumbers
        else:
            imin = imin[0]
        if imin == imax:
            psdi = psd_1d_rad[imin]
        else:
            psdi = psd_1d_rad[imin:imax]
        psd_bin[i] = np.mean(psdi)

    # Filter empty bins
    bins_centres = bins_centres[psd_bin != 0]
    psd_bin = psd_bin[psd_bin != 0]
    
    return bins_centres, psd_bin

def _debug_plot(cloud_scalar, k1d, psd_1d, b0, beta, r_squared_beta,
                bins_centres, psd_bins, beta_binned, b0_binned, r_squared_binned,
                l_spec_median, l_spec_mom):
    
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        axs[0].imshow(cloud_scalar, "gray")
        axs[0].axis("off")
        axs[0].set_title("Clouds")
        axs[1].scatter(np.log(k1d), np.log(psd_1d), s=2.5, c="k")
        axs[1].plot(np.log(k1d), b0 + beta * np.log(k1d), c="k")
        axs[1].scatter(np.log(bins_centres), np.log(psd_bins), s=2.5, c="C1")
        axs[1].axvline(np.log(1/l_spec_mom), c="grey")

        locs = axs[1].get_xticks().tolist()
        labs = [x for x in axs[1].get_xticks()]
        Dticks = dict(zip(locs, labs))
        Dticks[np.log(1/l_spec_mom)] = r"1/L"
        locas = list(Dticks.keys())
        labes = list(Dticks.values())
        axs[1].set_xticks(locas)
        axs[1].set_xticklabels(labes)

        axs[1].plot(np.log(bins_centres), b0_binned + beta_binned * np.log(bins_centres), c="C1")
        axs[1].annotate("Direct", (0.7, 0.9), xycoords="axes fraction", fontsize=10)
        axs[1].annotate(
            r"$R^2$=" + str(round(r_squared_beta, 3)),
            (0.7, 0.8),
            xycoords="axes fraction",
            fontsize=10,
        )
        axs[1].annotate(
            r"$\beta=$" + str(round(beta, 3)),
            (0.7, 0.7),
            xycoords="axes fraction",
            fontsize=10,
        )
        axs[1].annotate(
            r"L (median) = " + str(round(l_spec_median, 3)),
            (0.7, 0.6),
            xycoords="axes fraction",
            fontsize=10,
        )
        axs[1].annotate(
            r"L (moment) =" + str(round(l_spec_mom, 3)),
            (0.7, 0.5),
            xycoords="axes fraction",
            fontsize=10,
        )
        axs[1].annotate(
            "Bin-averaged",
            (0.4, 0.9),
            xycoords="axes fraction",
            color="C1",
            fontsize=10,
        )
        axs[1].annotate(
            r"$R^2$=" + str(round(r_squared_binned, 3)),
            (0.4, 0.8),
            xycoords="axes fraction",
            color="C1",
            fontsize=10,
        )
        axs[1].annotate(
            r"$\beta_a=$" + str(round(beta_binned, 3)),
            (0.4, 0.7),
            xycoords="axes fraction",
            color="C1",
            fontsize=10,
        )
        axs[1].set_xlabel(r"$\ln k$", fontsize=10)
        axs[1].set_ylabel(r"$\ln E(k)$", fontsize=10)
        axs[1].grid()
        axs[1].set_title("1D Spectrum")
        plt.tight_layout()
        plt.show()

def compute_spectra(cloud_scalar,
                    dx=1,
                    periodic_domain=False,
                    apply_detrending=False,
                    window=None):
    """
    Compute energy-preserving 2D FFT of input cloud_scalar field, which is
    assued to be square, computes the spectral power of this field and 
    decomposes this into radial and azimuthal power spectral densities.

    Parameters
    ----------
    cloud_scalar : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.
    dx : int, optional
        Horizontal (uniform) grid spacing, for computing wavenumbers. The
        default is 1.
    periodic_domain : Bool, optional
        Whether the domain is periodic. If False, choose an appropriate window
        to apply before performing the FFT. The default is False.
    apply_detrending : Bool, optional
        Whether to remove a blinear fit of the input scalar field. The default 
        is False.
    window : String, optional
        Which windowing function to apply. Set to a value in 
        ['Hann', 'Welch', 'Planck']. The default is None, corresponding to no
        applied windowing.

    Returns
    -------
    k1d : 1D numpy array
        Radial 1D wavenumbers of the FFT, whose wavelength is 1/k1d
    psd_1d_rad: 1D numpy array
        Radial 1D power spectral density of the cloud_scalar field
    psd_1d_azi: 1D numpy array
        Azimuthal 1D power spectral density of the cloud_scalar field
    """
     
    # TODO: This explicitly assumes square domains

    # General observations
    # Windowing   : Capturing more information is beneficial
    #               (None>Planck>Welch>Hann window)
    # Detrending  : Mostly imposes unrealistic gradients
    # Using image : Less effective at reproducing trends in 2D org plane
    # Binning     : Emphasises lower k and ignores higher k
    # Assumptions : 2D Fourier spectrum is isotropic
    #               Impact of small-scale errors is small

    [X, Y] = np.meshgrid(
        np.arange(cloud_scalar.shape[0]), np.arange(cloud_scalar.shape[1]), indexing="ij"
    )

    # Detrending
    if apply_detrending:
        cloud_scalar, bDt = _detrend(cloud_scalar, [X, Y])

    # Windowing
    if periodic_domain:
        if window == "Planck":
            cloud_scalar = _planck_rad(cloud_scalar)  # Planck-taper window
        elif window == "Welch":
            cloud_scalar = _welch_rad(cloud_scalar)  # Welch window
        elif window == "Hann":
            cloud_scalar = _hann_rad(cloud_scalar)  # Hann window

    # FFT
    F = fftpack.fft2(cloud_scalar)  # 2D FFT (no prefactor)
    F = fftpack.fftshift(F)  # Shift so k0 is centred
    psd_2d = np.abs(F) ** 2 / np.prod(cloud_scalar.shape)  # Energy-preserving 2D PSD
    psd_1d_rad = _get_psd_1d_radial(psd_2d)  # Azimuthal integral-> 1D radial PSD
    psd_1d_azi = _get_psd_1d_azimuthal(psd_2d)  # Radial integral -> Sector 1D PSD

    # Wavenumbers
    shp = np.min(cloud_scalar.shape)
    k1d = -(fftpack.fftfreq(shp, dx))[shp // 2 :]
    k1d = np.flip(k1d)
    
    return k1d, psd_1d_rad, psd_1d_azi


def spectral_anisotropy(psd_1d_azi):
    """
    Compute a measure of the spectral anisotropy, by computing the ratio
    between the smallest and largest PSD in an azimuthal sector of the 2D
    PSD

    Parameters
    ----------
    psd_1d_azi : 1D numpy array
        PSD in sectors of 1 degree of the 2D PSD.

    Returns
    -------
    spectral_anisotropy
        Spectral anisotropy.

    """
    
    return 1 - np.min(psd_1d_azi) / np.max(psd_1d_azi)

def spectral_slope(k1d, psd_1d_rad, return_intercept=False):
    """
    Compute a least squares fit of the slope of the 1D radial PSD, which turns 
    out to be a reasonable measure for the dominant length scale of the 
    cloud_scalar. 

    Parameters
    ----------
    k1d : 1D numpy array
        Radial wavenumbers.
    psd_1d_rad :  1D numpy array
        1D radial PSD.
    return_intercept : Bool, optional
        Whether to return the y-intercept of the fit (e.g. for plotting). 
        Default is False.

    Returns
    -------
    beta : float
        Slope of the fitted 1D radial PSD.
    b0 : float
        Intercept of the fit. Only returned if return_intercept=True.

    """
    
    beta, b0 = np.polyfit(np.log(k1d), np.log(psd_1d_rad), 1)
    if return_intercept:
        return beta, b0
    else:
        return beta

def spectral_slope_binned(k1d, psd_1d_rad, n_bins=10, return_intercept=False):
    """
    Similar to spectral_slope, this computes a least squares fit of the slope of 
    the 1D radial PSD, but coarsens the PSD into averages over logarithmically
    spaced bins first, to weight large and small scales more equally.

    Parameters
    ----------
    k1d : 1D numpy array
        Radial wavenumbers.
    psd_1d_rad :  1D numpy array
        1D radial PSD.
    n_bins : int, optional
        Number of logarithmically spaced bins. The default is 10.
    return_intercept : Bool, optional
        Whether to return the y-intercept of the fit (e.g. for plotting). 
        Default is False.

    Returns
    -------
    beta : float
        Slope of the fitted 1D radial PSD.
    b0 : float
        Intercept of the fit. Only returned if return_intercept=True.

    """

    bins_centres, psd_bins = _bin_average(k1d, psd_1d_rad, n_bins)

    # Spectral slope beta
    if psd_bins.shape[0] != 0:
        beta, b0 = np.polyfit(np.log(bins_centres[1:-1]), np.log(psd_bins[1:-1]), 1)
    else:
        beta, b0 = float("nan"), float("nan")
    
    if return_intercept:
        return beta, b0
    else:
        return beta

def spectral_length_median(k1d, psd_1d_rad):
    """
    Spectral length scale based on wavenumber with median amount of spectral
    power. Based on de Roode et al. (2004) https://doi.org/10.1175/1520-0469(2004)061<0403:LSHLIL>2.0.CO;2

    Parameters
    ----------
    k1d : 1D numpy array
        Radial wavenumbers.
    psd_1d_rad :  1D numpy array
        1D radial PSD.

    Returns
    -------
    l_spec : float
        Median spectral length scale.

    """

    # Spectral length scale as de Roode et al. (2004), using true median
    # sumps = np.cumsum(psd1); sumps/=sumps[-1]
    # kcrit = np.where(sumps>1/2)[0][0]
    # lSpec = 1./kcrit

    # Spectral length scale as de Roode et al. (2004) using ogive
    var_tot = np.trapz(psd_1d_rad, k1d)
    i = 0
    vari = var_tot + 1
    while vari > 2.0 / 3 * var_tot:
        vari = np.trapz(psd_1d_rad[i:], k1d[i:])
        i += 1
    kcrit = k1d[i - 1]
    l_spec = 1.0 / kcrit
    
    return l_spec

def spectral_length_moment(k1d, psd_1d_rad, order=1):
    """
    Spectral length scale based on wavenumber that corresponds to the order-th
    moment of the 1D radial PSD. Based on Pino et al. (2006),
    doi: 10.1007/s10546-006-9080-6

    Parameters
    ----------
    k1d : 1D numpy array
        Radial wavenumbers.
    psd_1d_rad :  1D numpy array
        1D radial PSD.
    order : int, optional
        Which moment to base the length scale on. The default is 1 and
        corresponds to the spectral mean.

    Returns
    -------
    l_spec : float
        Spectral length scale.

    """

    var_tot = np.trapz(psd_1d_rad, k1d)
    kmom = np.trapz(psd_1d_rad * k1d ** order, k1d) / var_tot
    l_spec = 1.0 / kmom

    return l_spec

def compute_all(cloud_scalar,
                dx=1,
                periodic_domain=False,
                apply_detrending=False,
                window=None,
                n_bins=10,
                order=1,
                debug=False):

    k1d, psd_1d_rad, psd_1d_azi = compute_spectra(cloud_scalar,
                                                  dx=dx,
                                                  periodic_domain=periodic_domain,
                                                  apply_detrending=apply_detrending,
                                                  window=window)
    anisotropy = spectral_anisotropy(psd_1d_azi)
    beta, b0 = spectral_slope(k1d, psd_1d_rad, return_intercept=True)
    r_squared = compute_r_squared(np.log(k1d), np.log(psd_1d_rad), [beta, b0])
    beta_binned, b0_binned = spectral_slope_binned(k1d, psd_1d_rad, n_bins=n_bins,return_intercept=True)
    bins_centres, psd_bins = _bin_average(k1d, psd_1d_rad, n_bins)
    r_squared_binned = compute_r_squared(np.log(bins_centres[1:-1]), np.log(psd_bins[1:-1]), [betaa, b0a])
    l_spec_median = spectral_length_median(k1d, psd_1d_rad)
    l_spec_moment = spectral_length_moment(k1d, psd_1d_rad, order=order)
    
    if debug:
        _debug_plot(cloud_scalar, k1d, psd_1d_rad, b0, beta, r_squared,
                bins_centres, psd_bins, beta_binned, b0_binned, r_squared_binned,
                l_spec_median, l_spec_moment)

    return anisotropy, beta, beta_binned, l_spec_median, l_spec_moment

