#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import findFiles, getField, blockShaped
import multiprocessing as mp
from tqdm import tqdm


class TWPVar:
    """
    Class for computing the cloud water variance ratio in blocks larger than
    L0xL0. Computed as proposed by Bretherton & Blossey (2017), by computing
    the mean cloud water path in blocks of size (L0,L0), subtracting the field
    mean cloud water path (resulting in cloud water path anomaly) and dividing
    the standard deviation of the cloud water path anomaly in these blocks by
    the standard deviation of the entire scene's cloud water.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Two of
                      these columns can be filled by 'twpVar'.
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
        self.field = "Cloud_Water_Path"
        self.L0 = 8  # How large is a block?
        self.thr = 250  # Threshold for plotting
        self.plot = False

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
            self.field = mpar["fields"]["cwp"]
            self.nproc = mpar["nproc"]

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        twpVar : float
            Variance ratio of cloud water in blocks of (L0,L0) to scene in
            total.

        """

        lwpBlock = blockShaped(field, self.L0, self.L0)  # Blocks (L0,L0)
        lwpAnom = np.mean(lwpBlock, axis=(1, 2)) - np.mean(field)  # Lwp anomaly
        # per block
        lwpStd = np.std(lwpAnom)  # Compute std
        twpVar = lwpStd / np.std(field)  # / Domain std

        if self.plot:
            lwpPlot = field.copy()
            lwpPlot[lwpPlot > self.thr] = self.thr
            lwpPlot[lwpPlot < 1] = float("nan")
            aDim = int(field.shape[0] / self.L0)
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            axs[0].imshow(lwpPlot, cmap="gist_ncar")
            axs[0].set_title("Cloud water path field")
            axs[1].imshow(lwpAnom.reshape(aDim, aDim))
            axs[1].set_title("CWP anomaly; twpVar = " + str(round(twpVar, 3)))

        return twpVar

    def verify(self):
        return "Not implemented for twpVar"

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
            twpVar = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))

        if self.save:
            dfMetrics["twpVar"].loc[dates] = twpVar
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
    twpVarGen = TWPVar(mpar)
    twpVarGen.verify()
    twpVarGen.compute()
