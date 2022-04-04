#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import findFiles, getField, createCircularMask
import multiprocessing as mp
from tqdm import tqdm


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies**i_order * y_indices**j_order).sum()


def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov


# TODO:
# - Implement support for periodic BCs. Current method is not translationally
#   invariant.


class Orient:
    """
    Class for computing the scene's degree of directional alignment using the
    cloud mask's raw image moment covariance matrix. Code based on:
    https://github.com/alyssaq/blog/blob/master/posts/150114-054922_computing-the-axes-or-orientation-of-a-blob.md


    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Two of
                      these columns can be filled by 'scai' and 'd0'.
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
        self.plot = False

        # General parameters from dictionary
        if mpar is not None:
            self.loadPath = mpar["loadPath"]
            self.savePath = mpar["savePath"]
            self.save = mpar["save"]
            self.saveExt = mpar["saveExt"]
            self.resFac = mpar["resFac"]
            self.con = mpar["con"]
            self.areaMin = mpar["areaMin"]
            self.fMin = mpar["fMin"]
            self.fMax = mpar["fMax"]
            self.field = mpar["fields"]["cm"]
            self.nproc = mpar["nproc"]
            self.bc = mpar["bc"]

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        scai : float
            Simple Convective Aggregation Index.

        """

        if self.bc == "periodic":
            print("Periodic BCs not implemented for orientation metric, returning nan")
            return float("nan")

        cov = moments_cov(field)
        if np.isnan(cov).any() or np.isinf(cov).any():
            return float("nan")

        evals, evecs = np.linalg.eig(cov)
        orie = np.sqrt(1 - np.min(evals) / np.max(evals))

        if self.plot:
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]  # evec with largest eval
            x_v2, y_v2 = evecs[:, sort_indices[1]]
            evalsn = evals[sort_indices] / evals[sort_indices][0]

            scale = 10
            ox = int(field.shape[1] / 2)
            oy = int(field.shape[0] / 2)
            lw = 5

            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(field, "gray")
            # plt.scatter(ox+x_v1*-scale*2,oy+y_v1*-scale*2,s=100)
            ax.plot(
                [ox - x_v1 * scale * evalsn[0], ox + x_v1 * scale * evalsn[0]],
                [oy - y_v1 * scale * evalsn[0], oy + y_v1 * scale * evalsn[0]],
                linewidth=lw,
            )
            ax.plot(
                [ox - x_v2 * scale * evalsn[1], ox + x_v2 * scale * evalsn[1]],
                [oy - y_v2 * scale * evalsn[1], oy + y_v2 * scale * evalsn[1]],
                linewidth=lw,
            )
            ax.set_title("Alignment measure = " + str(round(orie, 3)))
            plt.show()

        return orie

    def verify(self):
        """
        Verification based on three simple tests:
            1. Large uniform circle (should be 0)
            2. Randomly scattered points (should be 0)
            3. Vertical lines (should be 1)

        Returns
        -------
        veri : List of floats
            List containing metric(s) for verification case.

        """
        # 1. One large, uniform circle
        t1 = np.ones((512, 512))
        mask = createCircularMask(512, 512)

        # 2. Randomly scattered points
        t1[~mask] = 0
        t2 = np.random.rand(512, 512)
        ind = np.where(t2 > 0.5)
        t2[ind] = 1
        ind = np.where(t2 <= 0.5)
        t2[ind] = 0

        # 3. Vertical lines
        t3 = np.zeros((512, 512))
        t3[:, 250:251] = 1
        tests = [t1, t2, t3]

        veri = []
        for i in range(len(tests)):
            orie = self.metric(tests[i])
            veri.append(orie)

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
            orie = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))

        if self.save:
            dfMetrics["orie"].loc[dates] = orie
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
    orieGen = Orient(mpar)
    orieGen.verify()
    orieGen.compute()
