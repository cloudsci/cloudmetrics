#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import findFiles, getField
import pywt
import multiprocessing as mp
from tqdm import tqdm


class WOI:
    """
    Class for computing Wavelet Organisation Indices, as proposed by Brune et
    al. (2018) from a cloud water path field. Can compute the three indices
    WOI1, WOI2 and WOI3 suggested in that paper.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. This class
                      can fill the columns 'woi1', 'woi2', 'woi3' and 'woi'.
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
        self.fieldRef = "Cloud_Mask_1km"
        self.plot = False
        self.pad = 0

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
            self.fieldRef = mpar["fields"]["cm"]
            self.nproc = mpar["nproc"]

    def metric(self, field, cm, verify=False):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud water path field.
        cm : numpy array of shape (npx,npx)
            Cloud mask field.

        Returns
        -------
        woi1 : float
           First wavelet organisation index (scale distribution).
        woi2 : float
            Second wavelet organisation index (total amount of stuff).
        woi3 : float
            Third wavelet organisation index (directional alignment).

        """

        # STATIONARY/UNDECIMATED Direct Wavelet Transform
        field = pywt.pad(field, self.pad, "periodic")
        scaleMax = int(np.log(field.shape[0]) / np.log(2))
        coeffs = pywt.swt2(field, "haar", scaleMax, norm=True, trim_approx=True)
        # Bug in pywt -> trim_approx=False does opposite of its intention
        # Structure of coeffs:
        # - coeffs    -> list with nScales indices. Each scale is a 2-power of
        #                the image resolution. For 512x512 images we have
        #                512 = 2^9 -> 10 scales
        # - coeffs[i] -> Contains three directions:
        #                   [0] - Horizontal
        #                   [1] - Vertical
        #                   [2] - Diagonal

        specs = np.zeros((len(coeffs), 3))  # Shape (nScales,3)
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
        # Large scales are defined as 0 < k < 5
        specs = specs[1:]
        specL = specs[:5, :]
        specS = specs[5:, :]

        Ebar = np.sum(np.mean(specs, axis=1))
        Elbar = np.sum(np.mean(specL, axis=1))
        Esbar = np.sum(np.mean(specS, axis=1))

        Eld = np.sum(specL, axis=0)
        Esd = np.sum(specS, axis=0)

        # Compute wavelet organisation index
        woi1 = Elbar / Ebar
        woi2 = (Elbar + Esbar) / np.sum(cm)
        woi3 = (
            1.0
            / 3
            * np.sqrt(
                np.sum(((Esd - Esbar) / Esbar) ** 2 + ((Eld - Elbar) / Elbar) ** 2)
            )
        )

        woi = np.log(woi1) + np.log(woi2) + np.log(woi3)

        if self.plot:
            labs = ["Horizontal", "Vertical", "Diagonal"]
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            axs[0].imshow(field, "gist_ncar")
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

        if verify:
            return specs
        else:
            return woi1, woi2, woi3, woi

    def verify(self):
        """
        Verify that the wavelet energy spectrum contains the same energy as the
        original field

        """
        files, _ = findFiles(self.loadPath)
        file = files[self.fMin]
        cwp = getField(file, self.field, self.resFac, binary=False)
        cm = getField(file, self.fieldRef, self.resFac, binary=True)

        specs = self.metric(cwp, cm, verify=True)

        # Validate wavelet energy spectrum -> if correct total energy should be
        # the same as in image space
        Ewav = np.sum(specs)
        Eimg = np.mean(cwp**2)

        diff = Ewav - Eimg
        if diff < 1e-10:
            print("Energy conserved by SWT")
        else:
            print("Energy not conserved by SWT - results will be wrong")

    def getcalc(self, file):
        cwp = getField(file, self.field, self.resFac, binary=False)
        cm = getField(file, self.fieldRef, self.resFac, binary=True)
        return self.metric(cwp, cm)

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
            dfMetrics["woi"].loc[dates] = results[:, 3]
            dfMetrics["woi1"].loc[dates] = results[:, 0]
            dfMetrics["woi2"].loc[dates] = results[:, 1]
            dfMetrics["woi3"].loc[dates] = results[:, 2]
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
    woiGen = WOI(mpar)
    woiGen.verify()
    woiGen.compute()
