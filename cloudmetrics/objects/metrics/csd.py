import numpy as np
from scipy.optimize import curve_fit

from ._object_properties import _get_objects_property


def r_squared(x, y, coeffs):
    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    return ssreg / sstot


def _create_bins():
    dl0 = 2  # Zero bin length
    cst = 1.2  # Strechting factor
    N = 9  # How many bins, maximum

    dl = dl0 * cst ** np.arange(N)
    bins = np.cumsum(dl)
    bins = np.concatenate((np.zeros(1), bins))
    lav = (bins[1:] + bins[:-1]) / 2

    return bins, lav


def fPerc(s, a, b, c):
    """
    Subcritical percolation model (Ding et al. 2014)
    """
    return a * s - b * np.log(s) + c


def csd(object_labels, fit_kind, min_area=0):
    """
    Compute metric(s) for a single field

    Parameters
    ----------
    object_labels: 2D array of labelled cloud-object
    fit_kind: one of `powerlaw` or `sperc` for either applying a power-law fit
              or supercritial percolation fit (e.g. Ding et al 2014)

    Returns
    -------
    sizeExp if csdFit is power: float
        Exponent of the power law fit of the cloud size distribution
    popt if csdFit is perc : list
        List of fit parameters of the percolation fit by Ding et al (2014).
    """

    # Extract length scales
    area = _get_objects_property(object_labels=object_labels, property_name="area")

    idx_large_objects = area > min_area
    if np.count_nonzero(idx_large_objects) == 0:
        return None
    area = area[idx_large_objects]

    lengthscales = np.sqrt(area)

    # Construct histogram
    bins, lav = _create_bins()
    hist = np.histogram(lengthscales, bins)
    ns = hist[0]

    # Filter zero bins and the first point
    ind = np.where(ns != 0)
    nssl = ns[ind]
    lavsl = lav[ind]
    nssl = nssl[1:]
    lavsl = lavsl[1:]

    # Regular fit
    if fit_kind == "power":
        csd_sl, csd_int = np.polyfit(np.log(lavsl), np.log(nssl), 1)
        # rSq = r_squared(np.log(lavsl), np.log(nssl), [csd_sl, csd_int])
        coeff = csd_sl

    # Subcritical percolation fit
    elif fit_kind == "sperc":
        popt, pcov = curve_fit(fPerc, lavsl, np.log(nssl))
        if popt[0] > 0:
            popt[0] = 0

        coeff = popt
    else:
        raise NotImplementedError(fit_kind)

    return coeff


def _plot_powerlaw_fit(field, lavsl, nssl, lav, csd_int, csd_sl, ns, rSq):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
    axs[0].imshow(field, "gray")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].scatter(np.log(lavsl), np.log(nssl), s=10, c="k")
    axs[1].plot(np.log(lav), csd_int + csd_sl * np.log(lav), c="gray")
    # axs[1].plot(np.log(lav), fPerc(lav,popt[0],popt[1],popt[2]))
    axs[1].set_xlim((np.log(lav[1]) - 0.2, np.log(np.max(lav)) + 0.2))
    axs[1].set_ylim((-0.5, np.log(np.max(ns)) + 0.5))
    axs[1].set_xlabel(r"log $s$ [m]")
    axs[1].set_ylabel(r"log $n_s$ [-]")
    axs[1].annotate(
        "exp = " + str(round(csd_sl, 3)),
        (0.6, 0.9),
        xycoords="axes fraction",
    )
    axs[1].annotate(
        r"$R^2$ = " + str(round(rSq, 3)),
        (0.6, 0.8),
        xycoords="axes fraction",
    )
    plt.show()


def _plot_supercritical_percolation_fit(field, lavsl, nssl, lav, ns, popt):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
    axs[0].imshow(field, "gray")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].scatter(np.log(lavsl), np.log(nssl), s=10, c="k")
    axs[1].plot(np.log(lav), fPerc(lav, popt[0], popt[1], popt[2]))
    axs[1].set_xlim((np.log(lav[1]) - 0.2, np.log(np.max(lav)) + 0.2))
    axs[1].set_ylim((-0.5, np.log(np.max(ns)) + 0.5))
    axs[1].set_xlabel(r"log $s$ [m]")
    axs[1].set_ylabel(r"log $n_s$ [-]")
    # axs[1].annotate('exp = '+str(round(csd_sl,3)),(0.6,0.9),
    # xycoords='axes fraction')
    # axs[1].annotate(r'$R^2$ = '+str(round(rSq,3)),(0.6,0.8),
    # xycoords='axes fraction')
    plt.show()
