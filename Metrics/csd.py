#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit
from .utils import findFiles, getField, rSquared, periodic
import multiprocessing as mp
from tqdm import tqdm


def fPerc(s, a, b, c):
    # Subcritical percolation fit (Ding et al. 2014)
    return a * s - b * np.log(s) + c


class CSD:
    """
    Class for computing attributes of the cloud size distribution (CSD). Has
    two modes: Power law fit or percolation fit (Ding et al. 2014). The former
    is default and is the only version for which saving is supported.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. One of
                      these columns can be filled by 'sizeExp'
           save     : Boolean to specify whether to store the variables in
                      savePath/Metrics.h5
           resFac   : Resolution factor (e.g. 0.5), to coarse-grain the field.
           plot     : Boolean to specify whether to make plot with details on
                      this metric for each scene.
           con      : Connectivitiy for segmentation (1 - 4 seg, 2 - 8 seg)
           areaMin  : Minimum cloud size considered in computing metric
           fMin     : First scene to load
           fMax     : Last scene to load. If None, is last scene in set.
           fields   : Naming convention for fields, used to set the internal
                      field to be used to compute each metric. Must be of the
                      form:
                           {'cm'  : CloudMaskName,
                            'im'  : imageName,
                            'cth' : CloudTopHeightName,
                            'cwp' : CloudWaterPathName}

    """

    def __init__(self, mpar=None):
        # Metric-specific parameters
        self.field = "Cloud_Mask_1km"
        self.csdFit = "power"  # Alternative is percolation fit
        self.plot = False
        self.con = 1
        self.areaMin = 4

        # General parameters
        if mpar is not None:
            self.loadPath = mpar["loadPath"]
            self.savePath = mpar["savePath"]
            self.save = mpar["save"]
            self.saveExt = mpar["saveExt"]
            self.resFac = mpar["resFac"]
            self.plot = mpar["plot"]
            self.con = mpar["con"]
            self.areaMin = mpar["areaMin"]
            self.fMin = mpar["fMin"]
            self.fMax = mpar["fMax"]
            self.field = mpar["fields"]["cm"]
            self.nproc = mpar["nproc"]
            self.bc = mpar["bc"]

        # Bins
        dl0 = 2  # Zero bin length
        cst = 1.2  # Strechting factor
        N = 9  # How many bins, maximum

        dl = dl0 * cst ** np.arange(N)
        bins = np.cumsum(dl)
        self.bins = np.concatenate((np.zeros(1), bins))
        self.lav = (self.bins[1:] + self.bins[:-1]) / 2

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        sizeExp if csdFit is power: float
            Exponent of the power law fit of the cloud size distribution
        popt if csdFit is perc : list
            List of fit parameters of the percolation fit by Ding et al (2014).
        """

        # Segment
        cmlab, num = label(field, return_num=True, connectivity=self.con)
        regions = regionprops(cmlab)

        # Extract length scales
        area = []
        for i in range(num):
            props = regions[i]
            if props.area > self.areaMin:
                area.append(props.area)
        area = np.asarray(area)
        l = np.sqrt(area)

        plt.hist(area)

        # Construct histogram
        hist = np.histogram(l, self.bins)
        ns = hist[0]

        # Filter zero bins and the first point
        ind = np.where(ns != 0)
        nssl = ns[ind]
        lavsl = self.lav[ind]
        nssl = nssl[1:]
        lavsl = lavsl[1:]

        # Regular fit
        if self.csdFit == "power":
            csd_sl, csd_int = np.polyfit(np.log(lavsl), np.log(nssl), 1)
            rSq = rSquared(np.log(lavsl), np.log(nssl), [csd_sl, csd_int])

            if self.plot:
                fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
                axs[0].imshow(field, "gray")
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[1].scatter(np.log(lavsl), np.log(nssl), s=10, c="k")
                axs[1].plot(
                    np.log(self.lav), csd_int + csd_sl * np.log(self.lav), c="gray"
                )
                # axs[1].plot(np.log(lav), fPerc(lav,popt[0],popt[1],popt[2]))
                axs[1].set_xlim(
                    (np.log(self.lav[1]) - 0.2, np.log(np.max(self.lav)) + 0.2)
                )
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

            return csd_sl

        # Subcritical percolation fit
        elif self.csdFit == "perc":
            popt, pcov = curve_fit(fPerc, lavsl, np.log(nssl))
            if popt[0] > 0:
                popt[0] = 0

            if self.plot:
                fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
                axs[0].imshow(field, "gray")
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[1].scatter(np.log(lavsl), np.log(nssl), s=10, c="k")
                axs[1].plot(
                    np.log(self.lav), fPerc(self.lav, popt[0], popt[1], popt[2])
                )
                axs[1].set_xlim(
                    (np.log(self.lav[1]) - 0.2, np.log(np.max(self.lav)) + 0.2)
                )
                axs[1].set_ylim((-0.5, np.log(np.max(ns)) + 0.5))
                axs[1].set_xlabel(r"log $s$ [m]")
                axs[1].set_ylabel(r"log $n_s$ [-]")
                # axs[1].annotate('exp = '+str(round(csd_sl,3)),(0.6,0.9),
                # xycoords='axes fraction')
                # axs[1].annotate(r'$R^2$ = '+str(round(rSq,3)),(0.6,0.8),
                # xycoords='axes fraction')
                plt.show()

            return popt

    def verify(self):
        return "Not implemented for CSD"

    def getcalc(self, file):
        cm = getField(file, self.field, self.resFac, binary=True)
        if self.bc == "periodic":
            cm = periodic(cm, self.con)
        return self.metric(cm)

    def compute(self):
        """
        Main loop over scenes. Loads fields, computes metric, and stores it.

        """
        files, dates = findFiles(self.loadPath)
        files = files[self.fMin : self.fMax]
        dates = dates[self.fMin : self.fMax]

        if self.save:
            saveSt = self.saveExt
            dfMetrics = pd.read_hdf(self.savePath + "/Metrics" + saveSt + ".h5")

        ## Main loop over files
        if self.csdFit == "power":
            with mp.Pool(processes=self.nproc) as pool:
                sizeExp = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))

        elif self.csdFit == "perc":
            with mp.Pool(processes=self.nproc) as pool:
                popt = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))
        # FIXME Do something with popt

        if self.save:
            if self.csdFit == "perc":
                raise NotImplementedError("Saving percolation fit not supported")
            else:
                dfMetrics["sizeExp"].loc[dates] = sizeExp
            dfMetrics.to_hdf(
                self.savePath + "/Metrics" + saveSt + ".h5", "Metrics", mode="w"
            )


if __name__ == "__main__":
    mpar = {
        "loadPath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Filtered",
        "savePath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Metrics",
        "save": True,
        "resFac": 1,  # Resolution factor (e.g. 0.5)
        "plot": True,  # Plot with details on each metric computation
        "con": 1,  # Connectivity for segmentation (1:4 seg, 2:8 seg)
        "areaMin": 4,  # Minimum cloud size considered for object metrics
        "fMin": 0,  # First scene to load
        "fMax": None,  # Last scene to load. If None, is last scene in set
    }
    csdGen = CSD(mpar)
    # csdGen.csdFit = 'perc'
    csdGen.verify()
    csdGen.compute()
