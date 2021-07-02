#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis
from .utils import findFiles, getField
import multiprocessing as mp
from tqdm import tqdm


class CWP:
    """
    Class for computing statistics of the cloud water path field. Can compute
    integrated cloud water (cwp), cloud water variance (in scene and only in
    cloudy regions), as well as skewness and kurtosis of the cwp distribution

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. These
                      columns can be filled by 'cwp', 'cwpVar', 'cwpVarCl',
                      'cwpSke' and 'cwpKur'.
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

    Additional free parameters:
        field : Cloud_Water_Path


    """

    def __init__(self, mpar=None):
        # Metric-specific parameters
        self.field = "Cloud_Water_Path"
        self.plot = False
        self.pltThr = 500

        # General parameters from dict
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
            self.field = mpar["fields"]["cwp"]
            self.nproc = mpar["nproc"]

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud water path field.

        Returns
        -------
        cwp : float
            Scene-integrated cloud water.
        cwpVar : float
            Variance in cloud water over entire scene.
        cwpVar : float
            Variance in cloud water in cloudy regions.
        cwpSke : float
            Skewness of water distribution in cloudy regions.
        cwpKur : float
            Kurtosis of water distribution in cloudy regions.
        """

        # Variance over entire scene
        cwpSum = np.sum(field)
        cwpVar = np.std(field, axis=None)
        cwpSke = skew(field, axis=None)
        cwpKur = kurtosis(field, axis=None)

        # Variance in cloudy regions only
        cwpMask = field.copy()
        cwpMask[field == 0] = float("nan")
        varCl = np.nanstd(cwpMask, axis=None)

        # plot
        if self.plot:
            cwppl = field.copy()
            ind = np.where(field > self.pltThr)
            cwppl[ind] = self.pltThr
            cwppl[cwppl == 0.0] = float("nan")
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            axs[0].imshow(cwppl, "gist_ncar")
            axs[0].set_title("CWP")
            axs[0].axis("off")
            axs[1].hist(field.flatten(), np.linspace(1, self.pltThr, 100))
            axs[1].set_title("Histogram of in-cloud CWP")
            axs[1].set_xlabel("CWP")
            axs[1].set_ylabel("Frequency")
            plt.show()

        return cwpSum, cwpVar, cwpSke, cwpKur, varCl

    def verify(self):
        return "Not implemented for CWP"

    def getcalc(self, file):
        lwp = getField(file, self.field, self.resFac, binary=False)
        return self.metric(lwp)

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
            dfMetrics["cwp"].loc[dates] = results[:, 0]
            dfMetrics["cwpVar"].loc[dates] = results[:, 1]
            dfMetrics["cwpVarCl"].loc[dates] = results[:, 4]
            dfMetrics["cwpSke"].loc[dates] = results[:, 2]
            dfMetrics["cwpKur"].loc[dates] = results[:, 3]
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
    cwpGen = CWP(mpar)
    cwpGen.verify()
    cwpGen.compute()
