#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from .utils import findFiles, getField, periodic
import multiprocessing as mp
from tqdm import tqdm


class Objects:
    """
    Class for computing simple, object-based metrics.

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. This class
                      can fill the following columns: 'lMax', 'lMean',
                      'nClouds', 'eccA' and 'periSum'.
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
        # Default parameters
        self.field = "Cloud_Mask_1km"
        self.fieldRef = "image"
        self.plot = False
        self.areaMin = 4
        self.con = 1

        # General parameters
        if mpar is not None:
            self.loadPath = mpar["loadPath"]
            self.savePath = mpar["savePath"]
            self.save = mpar["save"]
            self.saveExt = mpar["saveExt"]
            self.resFac = mpar["resFac"]
            self.plot = mpar["plot"]
            self.con = mpar["con"]
            self.fMin = mpar["fMin"]
            self.fMax = mpar["fMax"]
            self.field = mpar["fields"]["cm"]
            self.fieldRef = mpar["fields"]["im"]
            self.nproc = mpar["nproc"]
            self.bc = mpar["bc"]

    def metric(self, field, im):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.
        im    : numpy array of shape (npx,npx)
            Reference image

        Returns
        -------
        lMax : float
            Length scale of largest object in the scene.
        lMean : float
            Mean length scale of scene objects.
        eccA : float
            Area-weighted, mean eccentricity of objects, approximated as
            ellipses
        periS : float
            Mean perimeter of an object in the scene
        """
        cmlab, num = label(field, return_num=True, connectivity=self.con)
        regions = regionprops(cmlab)

        area = []
        ecc = []
        peri = []
        for i in range(num):
            props = regions[i]
            if props.area > self.areaMin:
                area.append(props.area)
                ecc.append(props.eccentricity)
                peri.append(props.perimeter)
        area = np.asarray(area)
        ecc = np.asarray(ecc)
        peri = np.asarray(peri)
        # area = np.sqrt(area) <- Janssens et al. (2021) worked in l-space.
        #                         However, working directly with areas before
        #                         taking mean is more representative of pattern

        # print('Number of regions: ',len(area),' / ',num)

        # Plotting
        if self.plot:
            bins = np.arange(-0.5, len(area) + 1.5, 1)
            fig, axs = plt.subplots(ncols=5, figsize=(15, 3))
            axs[0].imshow(field, "gray")
            axs[0].set_title("Cloud mask")
            axs[1].imshow(im, "gray")
            axs[1].set_title("Reference image")
            axs[2].hist(area, bins)
            axs[1].set_title("Area")
            axs[3].hist(ecc, bins)
            axs[2].set_title("Eccentricity")
            axs[4].hist(peri, bins)
            axs[3].set_title("Perimeter")
            plt.show()

        if len(area) < 1:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        lMax = np.sqrt(np.max(area))
        lMean = np.sqrt(np.mean(area))
        nClouds = len(area)
        eccA = np.sum(area * ecc) / np.sum(area)
        periS = np.mean(peri)

        return lMax, lMean, nClouds, eccA, periS

    def verify(self):
        return "Not implemented for Objects"

    def getcalc(self, file):
        cm = getField(file, self.field, self.resFac, binary=True)
        if self.bc == "periodic":
            cm = periodic(cm, self.con)
        im = getField(file, self.fieldRef, self.resFac, binary=False)
        return self.metric(cm, im)

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
            dfMetrics["lMax"].loc[dates] = results[:, 0]
            dfMetrics["lMean"].loc[dates] = results[:, 1]
            dfMetrics["nClouds"].loc[dates] = results[:, 2]
            dfMetrics["eccA"].loc[dates] = results[:, 3]
            dfMetrics["periSum"].loc[dates] = results[:, 4]
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
    objectGen = Objects(mpar)
    objectGen.verify()
    objectGen.compute()
