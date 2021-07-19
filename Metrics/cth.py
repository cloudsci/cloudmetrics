#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis
from .utils import findFiles, getField
import multiprocessing as mp
from tqdm import tqdm


class CTH:
    """
    Class for computing statistics from the cloud top height field. Can
    compute mean, standard deviation, skewness and kurtosis of the cloud top
    height field. All calculations explicitly filter cloud top heights above
    thr, and statistics are only calculated in cloudy regions.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Can fill
                      columns 'cth', 'cthVar', 'cthSke' and 'cthKur'.
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
        self.field = "Cloud_Top_Height"
        self.fieldMask = "Cloud_Mask_1km"
        self.thr = 5000  # Set clouds above thr to zero
        self.plot = False

        if mpar is not None:
            # General parameters
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
            self.field = mpar["fields"]["cth"]
            self.fieldMask = mpar["fields"]["cm"]
            self.nproc = mpar["nproc"]

    def metric(self, field, mask):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud top height field.
        mask : numpy array of shape (npx,npx)
            Cloud mask field.

        Returns
        -------
        ave : float
            Mean cloud top height.
        var : float
            Standard deviation of cloud top height
        ske : float
            Skewness of cloud top height distribution
        kur : float
            Kurtosis of cloud top height

        """

        field *= mask

        # Filter high clouds explicitly for this computation
        field[field > self.thr] = 0
        cthnz = field[field != 0]

        ave = np.mean(cthnz)
        var = np.std(cthnz)
        ske = skew(cthnz, axis=None)
        kur = kurtosis(cthnz, axis=None)

        # Plotting routine
        if self.plot:
            bns = np.arange(1, self.thr, 300)
            fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
            axs[0].imshow(mask, "gray")
            axs[0].set_title("Cloud mask")
            a2 = axs[1].imshow(field, "gist_ncar")
            axs[1].set_title("Cloud top height")
            cb = plt.colorbar(a2)
            cb.ax.set_ylabel("Cloud top height [m]", rotation=270, labelpad=1)
            hst, _, _ = axs[2].hist(cthnz.flatten(), bns)
            axs[2].set_title("CTH histogram")
            plt.tight_layout()
            plt.show()

        return ave, var, ske, kur

    def verify(self):
        return "Verification not implemented for CTH"

    def getcalc(self, file):
        cth = getField(file, self.field, self.resFac, binary=False)
        cm = getField(file, self.fieldMask, self.resFac, binary=True)
        return self.metric(cth, cm)

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
        with mp.Pool(processes=self.nproc) as pool:
            results = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))
        results = np.array(results)

        if self.save:
            dfMetrics["cth"].loc[dates] = results[:, 0]
            dfMetrics["cthVar"].loc[dates] = results[:, 1]
            dfMetrics["cthSke"].loc[dates] = results[:, 2]
            dfMetrics["cthKur"].loc[dates] = results[:, 3]
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
    cthGen = CTH(mpar)
    cthGen.verify()
    cthGen.compute()
