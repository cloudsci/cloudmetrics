#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from .utils import findFiles, getField
from .cellgraph.cell2graph import create_cell_graph
from .cellgraph.graph_analysis import (
    plot_graph,
    degree_variance,
    aboav_weaire_parameter,
    lewis_parameter,
    defect_parameters,
    coordination_parameter,
    lewis_correlation,
)

# TODO:
# - Integrate Franziska's network functions (currently in separate, private repo)
# - Implement support for periodic BCs


class Network:
    """
     Class for computing nearest-neighbour network parameters (as in Glassmeier
     & Feingold (2017)) from a cloud mask. Can compute the network's cells'
     degree (side number) distribution and its variance (netVarDefg), the
     Aboav-Wearie parameter (netAWPar), the resulting coordination parameter
     (netCoPar), the Lewis law slope (netLPar) and fit correlation (netLCorr),
     and finally a 'defect slope' (netDefSl) and average degree of the largest
     clouds (netDegMax). See Glassmeier & Feingold (2017) for details.
    ,

     Parameters
     ----------
     mpar : Dict (optional, but necessary for using the compute method)
        Specifies the following parameters:
            loadPath : Path to load .h5 files that contain a pandas dataframe
                       with a cloud mask field as one of the columns.
            savePath : Path to a .h5 containing a pandas dataframe whose columns
                       contain metrics and whose indices are scenes. This class
                       can fill the columns 'netVarDeg', 'netAWPar', 'netCoPar',
                       'netLPar', 'netLCorr', 'netDefSl' and 'netDegMax'.
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

    def metric(self, field):
        """
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        netVarDeg : float
            Variance of the network cells' degree (number of sides)
            distribution.
        netAWPar : float
            Aboav-Wearie arrangement parameter (slope of fit between degree of
            a cell and average degree of neighbour cells).
        netCoPar : float
            Combination of netVarDeg and netAWPar
        netLPar : float
            Lewis law fit (slope of fit between degree and cloud size)
        netLCorr : float
            Correlation of the Lewis law fit
        netDefSl : float
            Slope of the 'defect' model (slope of fit between degree
            deviation from hexagonal and normalised cloud size)
        netDegMax : float
            Intercept of the 'defect' model

        """
        G = create_cell_graph(field, self.areaMin, method="scipy")

        if self.plot:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(field)
            plt.axis("off")
            plot_graph(G, ax=ax)

            a = int(0.05 * G.graph["ngrid"])
            b = int(0.95 * G.graph["ngrid"])
            c = int(0.05 * G.graph["ngrid"])
            d = int(0.95 * G.graph["ngrid"])
            ax.add_patch(
                patches.Rectangle(
                    (a, c), abs(b - a), abs(d - c), fill=False, color="c", lw=3
                )
            )

        netVarDeg = degree_variance(G, self.plot)
        netAWPar = aboav_weaire_parameter(G, self.plot)
        netCoPar = coordination_parameter(netAWPar, netVarDeg)
        netLPar = lewis_parameter(G, self.plot)
        netLCorr = lewis_correlation(G, self.plot)
        [netDefSl, netDegMax] = defect_parameters(G, self.plot)

        return netVarDeg, netAWPar, netCoPar, netLPar, netLCorr, netDefSl, netDegMax

    def verify(self):
        return "Not implemented for network"

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
        for f in range(len(files)):
            cm = getField(files[f], self.field, self.resFac, binary=True)
            print("Scene: " + files[f] + ", " + str(f + 1) + "/" + str(len(files)))

            (
                netVarDeg,
                netAWPar,
                netCoPar,
                netLPar,
                netLCorr,
                netDefSl,
                netDegMax,
            ) = self.metric(cm)

            print("Variance of degree distribution: %.2f" % netVarDeg)
            print("Aboav-Weaire parameter:          %.2f" % netAWPar)
            print("Coordination parameter:          %.2f" % netCoPar)
            print("Lewis parameter:                 %.2f" % netLPar)
            print("Lewis correlation:               %.2f" % netLCorr)
            print("Defect slope:                    %.2f" % netDefSl)
            print("Avg. degree of largest clouds:   %.2f" % netDegMax)

            if self.save:
                dfMetrics["netVarDeg"].loc[dates[f]] = netVarDeg
                dfMetrics["netAWPar"].loc[dates[f]] = netAWPar
                dfMetrics["netCoPar"].loc[dates[f]] = netCoPar
                dfMetrics["netLPar"].loc[dates[f]] = netLPar
                dfMetrics["netLCorr"].loc[dates[f]] = netLCorr
                dfMetrics["netDefSl"].loc[dates[f]] = netDefSl
                dfMetrics["netDegMax"].loc[dates[f]] = netDegMax

        if self.save:
            dfMetrics.to_hdf(
                self.savePath + "/Metrics" + saveSt + ".h5", "Metrics", mode="w"
            )


if __name__ == "__main__":
    mpar = {
        "loadPath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Filtered",
        "savePath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Metrics",
        "save": True,
        "resFac": 1,  # Resolution factor (e.g. 0.5)
        "plot": False,  # Plot with details on each metric computation
        "con": 1,  # Connectivity for segmentation (1:4 seg, 2:8 seg)
        "areaMin": 0,  # Minimum cloud size considered for object metrics
        "fMin": 0,  # First scene to load
        "fMax": None,  # Last scene to load. If None, is last scene in set
    }
    networkGen = Network(mpar)
    networkGen.verify()
    networkGen.compute()
