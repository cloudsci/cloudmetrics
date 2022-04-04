#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from .utils import findFiles, getField, cKDTreeMethod, periodic, createCircularMask
import multiprocessing as mp
from tqdm import tqdm

# TODO: Account for periodic BCs


class IOrgPoisson:
    """
    Class for computing the Organisation Index Iorg (Weger et al. 1992) from a
    cloud mask, using the distribution of a Poisson point process as reference.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. One of
                      these columns can be filled by 'iOrgPoiss'.
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

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        iOrg : float
            Organisation Index.

        """
        cmlab, num = label(field, return_num=True, connectivity=self.con)
        regions = regionprops(cmlab)

        xC = []
        yC = []
        for i in range(len(regions)):
            props = regions[i]
            if props.area > self.areaMin:
                y0, x0 = props.centroid
                xC.append(x0)
                yC.append(y0)

        pos = np.vstack((np.asarray(xC), np.asarray(yC))).T

        # print('Number of regions: ',pos.shape[0],'/',num)

        if pos.shape[0] < 1:
            print("No sufficiently large cloud objects, returning nan")
            return float("nan")

        ## Compute the nearest neighbour distances ##
        if self.bc == "periodic":
            sh = [shd // 2 for shd in field.shape]
            sz = np.min(sh)  # FIXME won't work for non-square domains

            # Move centroids outside the original domain into original domain
            pos[pos[:, 0] >= sh[1], 0] -= sh[1]
            pos[pos[:, 0] < 0, 0] += sh[1]
            pos[pos[:, 1] >= sh[0], 1] -= sh[0]
            pos[pos[:, 1] < 0, 1] += sh[0]

        else:
            sh = [shd for shd in field.shape]
            sz = None
        nnScene = cKDTreeMethod(pos, size=sz)
        # nbins = len(nnScene)+1; dx=0.01
        nbins = 100000  # <-- Better off fixing nbins at a very large number
        bins = np.linspace(0, np.sqrt(sh[0] ** 2 + sh[1] ** 2), nbins)
        nndpdfScene = np.histogram(nnScene, bins)[0]
        nndcdfScene = np.cumsum(nndpdfScene) / len(nnScene)

        # Poisson
        lam = nnScene.shape[0] / (sh[0] * sh[1])
        binav = (bins[1:] + bins[:-1]) / 2
        nndcdfRand = 1 - np.exp(-lam * np.pi * binav**2)

        ## Compute Iorg ##
        iOrg = np.trapz(nndcdfScene, nndcdfRand)

        if self.plot:
            fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
            axs[0].imshow(field, "gray")
            axs[0].set_title("Cloud mask of scene")

            axs[1].scatter(pos[:, 0], field.shape[0] - pos[:, 1], color="k", s=5)
            axs[1].set_title("Scene centroids")
            asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
            axs[1].set_aspect(asp)

            axs[2].plot(nndcdfRand, nndcdfScene, "-", color="k")
            axs[2].plot(nndcdfRand, nndcdfRand, "--", color="k")
            axs[2].set_title("Nearest neighbour distribution")
            axs[2].set_xlabel("Poisson nearest neighbour CDF")
            axs[2].set_ylabel("Scene nearest neighbour CDF")
            axs[2].annotate(
                r"$I_{org} = $" + str(round(iOrg, 3)),
                (0.7, 0.1),
                xycoords="axes fraction",
            )
            asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
            axs[2].set_aspect(asp)
            plt.show()

        return iOrg

    def verify(self):
        """
        Verification with simple examples:
            1. Regular lattice of squares (iOrg -> 0)
            2. Randomly scattered points (iOrg -> 0.5)
            3. One large, uniform circle with noise around it (iOrg -> 1)

        Returns
        -------
        iOrg : float
            Metric for verification case.

        """

        # 1. Regular lattice of squares (iorg --> 0)
        t1 = np.zeros((512, 512))
        t1[::16, ::16] = 1
        t1[1::16, ::16] = 1
        t1[::16, 1::16] = 1
        t1[1::16, 1::16] = 1

        # 2. Randomly scattered points (iorg --> 0.5)
        posScene = np.random.randint(0, high=512, size=(1000, 2))  # FIXME deprecated
        t2 = np.zeros((512, 512))
        t2[posScene[:, 0], posScene[:, 1]] = 1

        # 3. One large, uniform circle with noise around it (iorg --> 1)
        t3 = np.zeros((512, 512))
        maw = 128
        mask = createCircularMask(maw, maw).astype(int)
        t3[:maw, :maw] = mask
        # t3[maw-20:2*maw-20,maw-50:2*maw-50] = mask;
        tadd = np.random.rand(maw, maw)
        ind = np.where(tadd > 0.4)
        tadd[ind] = 1
        ind = np.where(tadd <= 0.4)
        tadd[ind] = 0
        t3[:maw, :maw] += tadd
        t3[t3 > 1] = 1

        tests = [t1, t2, t3]

        aMin = self.areaMin
        plotBool = self.plot
        self.areaMin = 0
        self.plot = True
        veri = []
        for i in range(len(tests)):
            if self.bc == "periodic":
                tests[i] = periodic(tests[i], self.con)
            iOrg = self.metric(tests[i])
            veri.append(iOrg)

        self.areaMin = aMin
        self.plot = plotBool

        return veri

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
        with mp.Pool(processes=self.nproc) as pool:
            iOrg = list(tqdm(pool.imap(self.getcalc, files), total=len(files)))

        if self.save:
            dfMetrics["iOrgPoiss"].loc[dates] = iOrg
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
    iOrgPoissonGen = IOrgPoisson(mpar)
    iOrgPoissonGen.verify()
    iOrgPoissonGen.compute()
