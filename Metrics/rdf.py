#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist, squareform
from .utils import findFiles, getField, periodic
import multiprocessing as mp
from tqdm import tqdm


def pair_correlation_2d(pos, S, r_max, dr, bc, normalize=True):
    """

    Pair correlation function, adapted from:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py
    and indirectly from Rasp et al. (2018)

    Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius r_max drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller r_max...or write some code to handle edge effects! ;)

    Arguments:
        pos             array of positions of shape (nPos,nDim), columns are
                        orderered x, y, ...
        S               length of each side of the square region of the plane
        r_max           distance from (open) boundary where objects are ignored
        dr              increment for increasing radius of annulus
        bc              boundary condition - if periodic look across boundaries
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """

    # Number of particles in ring/area of ring/number of reference
    # particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Extract domain size
    (Sy, Sx) = S if len(S) == 2 else (S, S)

    # Find particles which are close enough to the box center that a circle of radius
    # r_max will not cross any edge of the box, if no periodic bcs
    if bc != "periodic":
        bools1 = pos[:, 0] > r_max  # Valid centroids from left boundary
        bools2 = pos[:, 0] < (Sx - r_max)  # Valid centroids from right boundary
        bools3 = pos[:, 1] > r_max  # Valid centroids from top boundary
        bools4 = pos[:, 1] < (Sy - r_max)  # Valid centroids from bottom boundary
        (int_ind,) = np.where(bools1 * bools2 * bools3 * bools4)
    else:
        int_ind = np.arange(pos.shape[0])

    nCl = len(int_ind)
    pos = pos[int_ind, :]

    # Make bins
    edges = np.arange(0.0, r_max + dr, dr)  # Annulus edges
    nInc = len(edges) - 1
    g = np.zeros([nCl, nInc])  # RDF for all interior particles
    radii = np.zeros(nInc)

    # Define normalisation based on the used region and particles
    if bc == "periodic":
        number_density = float(pos.shape[0]) / float((Sx * Sy))
    else:
        number_density = float(pos.shape[0]) / float(((Sx - r_max) * (Sy - r_max)))

    # Compute pairwise distances
    if bc == "periodic":
        dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
        for d in range(pos.shape[1]):
            box = S[pos.shape[1] - d - 1]  # to match x,y ordering in pos
            pos_1d = pos[:, d][:, np.newaxis]  # shape (N, 1)
            dist_1d = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d**2
        dist = np.sqrt(dist_sq)
    else:
        dist = pdist(pos)

    dist = squareform(dist)
    np.fill_diagonal(dist, 2 * r_max)  # Don't want distance to self to count

    # Count objects per ring
    for p in range(nCl):
        result, bins = np.histogram(dist[p, :], bins=edges)
        if normalize:
            result = result / number_density
        g[p, :] = result

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(nInc)
    for i in range(nInc):
        radii[i] = (edges[i] + edges[i + 1]) / 2.0
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2))

    return g_average, radii, int_ind


class RDF:
    """
    Class for computing the Radial Distribution Function between objects (rdf)
    and derived metrics, from a cloud mask. Can compute the maximum of the
    RDF (rdfMax), the difference between minimum and maximum (rdfDiff). The
    implementation is based off:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py

    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Three of
                      these columns can be filled by 'rdfMax', 'rdfMin' and
                      'rdfDiff'.
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
        self.dx = 1  # Convert pixel to km
        self.rMax = (
            20  # How far away to compute the rdf FIXME This is sensitive to each case!!
        )
        self.dr = 1  # Bin width
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

    def metric(self, field, S):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.
        S     : Tuple of the input field's size (is different from field.shape
                if periodic BCs are used)

        Returns
        -------
        rdfM : float
            Maximum of the radial distribution function.
        rdfI : float
            Integral of the radial distribution function.
        rdfD : float
            Max-min difference of the radial distribution function.

        """
        cmlab, num = label(field, return_num=True, connectivity=self.con)
        regions = regionprops(cmlab)

        xC = []
        yC = []
        for i in range(num):
            props = regions[i]
            if props.area > self.areaMin:
                yC.append(props.centroid[0])
                xC.append(props.centroid[1])

        pos = np.vstack((np.asarray(xC), np.asarray(yC))).T

        # print('Number of regions: ',pos.shape[0],'/',num)

        if pos.shape[0] < 1:
            print("No sufficiently large cloud objects, returning nan")
            return float("nan"), float("nan"), float("nan")

        # TODO set dr based on field size and object number, results are
        # sensitive to this
        rdf, rad, tmp = pair_correlation_2d(
            pos, S, self.rMax, self.dr, self.bc, normalize=True
        )
        rad *= self.dx
        rdfM = np.max(rdf)
        rdfI = np.trapz(rdf, rad)
        rdfD = np.max(rdf) - rdf[-1]

        if self.plot:
            axF = "axes fraction"
            fig, axs = plt.subplots(ncols=2, figsize=(8.5, 4))
            axs[0].imshow(field, "gray")
            axs[0].axis("off")

            axs[1].plot(rad, rdf)
            axs[1].set_xlabel("Distance")
            axs[1].set_ylabel("RDF")
            axs[1].annotate("rdfMax = %.3f" % rdfM, (0.6, 0.15), xycoords=axF)
            axs[1].annotate("rdfInt = %.3f" % rdfI, (0.6, 0.10), xycoords=axF)
            axs[1].annotate("rdfDif = %.3f" % rdfD, (0.6, 0.05), xycoords=axF)
            plt.show()

        return rdfM, rdfI, rdfD

    def verify(self):
        return "Not implemented for RDF"

    def getcalc(self, file):
        cm = getField(file, self.field, self.resFac, binary=True)
        S = cm.shape
        if self.bc == "periodic":
            cm = periodic(cm, self.con)
        return self.metric(cm, S)

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
            dfMetrics["rdfMax"].loc[dates] = results[:, 0]
            dfMetrics["rdfInt"].loc[dates] = results[:, 1]
            dfMetrics["rdfDiff"].loc[dates] = results[:, 2]
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
    rdfGen = RDF(mpar)
    rdfGen.verify()
    rdfGen.compute()
