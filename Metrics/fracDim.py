#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import findFiles, getField, rSquared
import multiprocessing as mp
from tqdm import tqdm


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1,
    )
    return len(np.where((S > 0) & (S < k * k))[0])


class FracDim:
    """
    Class for computing the Minkowski-Bouligand (box-counting dimension from a
    cloud mask. Adapted from:
    https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. One of
                      these columns can be filled by 'fracDim'.
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
        self.thr = 0.5  # Binary threshold
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
            self.field = mpar["fields"]["cm"]
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
        fracDim : float
            Minkowski-Bouligand dimension.

        """
        Z = field < self.thr  # Binary image
        p = min(Z.shape)
        n = 2 ** np.floor(np.log(p) / np.log(2))
        n = int(np.log(n) / np.log(2))  # Number of extractable boxes
        sizes = 2 ** np.arange(n, 1, -1)  # Box sizes
        counts = np.zeros(len(sizes))
        for s in range(len(sizes)):
            counts[s] = boxcount(Z, sizes[s])  # Non-empty/non-full box no.

        # Fit the relation: counts = coeffs[1]*sizes**coeffs[0]; coeffs[0]=-Nd
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        rSq = rSquared(np.log(sizes), np.log(counts), coeffs)
        fracDim = -coeffs[0]

        if self.plot:
            fig, ax = plt.subplots(ncols=2, figsize=(8.25, 4))
            ax[0].imshow(field, "gray")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].loglog(sizes, counts)
            ax[1].set_title("fracDim = %.4f" % fracDim)
            ax[1].annotate("rSq: %.3f" % rSq, (0.7, 0.9), xycoords="axes fraction")
            ax[1].set_xlabel("Length")
            ax[1].set_ylabel("Number of edge boxes")
            plt.show()

        return fracDim

    def verify(self):
        """
        Verification with simple examples:
            1. Hilbert curve (should have fracDim=2)
            2. Randomly scattered points (should have fracDim=2)
            3. Straight line (should have fracDim=1)
        NOTE: Computing example 1 requires hilbertcurve package

        Returns
        -------
        veri : List of floats
            List containing metric(s) for verification case.

        """
        from hilbertcurve.hilbertcurve import HilbertCurve

        # 1. Hilbert curve (should have fracDim=2)
        t1 = np.zeros((512, 512))
        pHil = 8
        nHil = 2
        dist = 2 ** (pHil * nHil)
        hilbert_curve = HilbertCurve(pHil, nHil)
        coords = np.zeros((dist, nHil))
        for i in range(dist):
            coords[i, :] = hilbert_curve.coordinates_from_distance(i)
        coords = coords.astype(int)
        coords *= 2
        coordsAv = (coords[1:, :] + coords[:-1, :]) / 2
        coordsAv = coordsAv.astype(int)
        t1[coords[:, 0], coords[:, 1]] = 1
        t1[coordsAv[:, 0], coordsAv[:, 1]] = 1

        # 2. Random points (should have fracDim=2)
        t2 = np.random.rand(512, 512)
        ind = np.where(t2 > 0.5)
        t2[ind] = 1
        ind = np.where(t2 <= 0.5)
        t2[ind] = 0

        # 3. Vertical line (should have fracDim=1)
        t3 = np.zeros((512, 512))
        t3[:, 250:252] = 1
        tests = [t1, t2, t3]

        veri = []
        for i in range(len(tests)):
            fracDim = self.metric(tests[i])
            veri.append(fracDim)

        return veri

    def getcalc(self, file):
        cm = getField(file, self.field, self.resFac, binary=True)
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
        with mp.Pool(processes=self.nproc) as pool:
            fracDim = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))

        if self.save:
            dfMetrics["fracDim"].loc[dates] = fracDim
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
    fracDimGen = FracDim(mpar)
    fracDimGen.verify()
    fracDimGen.compute()
