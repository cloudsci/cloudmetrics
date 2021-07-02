#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess as sp
import shlex
import datetime
import pandas as pd
from sklearn.decomposition import PCA


def makeNewDirs(dirs):

    # Only intended for a list of directories where none have yet been made
    mkDlDirs = [d for d in dirs if os.path.isdir(d)]

    if not mkDlDirs:
        for d in dirs:
            os.makedirs(d)


def ensureWD(dirName):
    os.chdir(dirName)


#%% DOWNLOAD MODIS DATA
#%% Specify general download parameters.
# Modify to appropriate spatiotemporal ranges and satellite to download
# desired subsets of data, and repeat for as many intervals and satellites as
# needed. Note that the Worldview retrieval only allows retrieval of daytime
# overpass data and that this is therefore default also for MODAPS retrievals.

# Set root dir
rootDir = "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/cloudmetrics"

ensureWD(rootDir)
from Download import worldviewClient, ModapsClient

# Create folders to store downloaded data in
downloadDirs = [
    rootDir + "/Data/Download/DataAqua",
    rootDir + "/Data/Download/DataTerra",
]
makeNewDirs(downloadDirs)

cfg = {
    "satellite": "Aqua",
    "startDate": "2002-12-01",
    "endDate": "2003-01-31",
    "extent": [-58, -48, 10, 20],  # lonMin, lonMax, latMin, latMax
    "savePath": downloadDirs[0],  # 0 - Aqua  ; 1 - Terra
}

# MODAPS specific download parameters
cfgM = {
    "instrument": "PM1M",  # Aqua - PM1M; Terra - AM1M
    "product": "MYD06_L2",  # MODIS L2 Cloud product
    # (MYD - Aqua; MOD - Terra)
    "collection": 61,  # hdf collection (61 for Aqua and Terra)
    "layers": [
        "MYD06_L2___Cloud_Mask_1km",
        "MYD06_L2___Cloud_Top_Height",
        "MYD06_L2___Cloud_Water_Path",
        "MYD06_L2___Sensor_Zenith",
    ],
    "email": "martin.janssens@wur.nl",
    "appKey": "9F16973E-0A9C-11EA-9879-AF780D77E571",
}

#%% Direct image download from Worldview
worldviewClient.downloadMODISImgs(
    cfg["startDate"],
    cfg["endDate"],
    cfg["extent"],
    cfg["savePath"],
    satellite=cfg["satellite"],
)

#%% Request data from LAADS archive through MODAPS API
modapsClient = ModapsClient()

# Search for correct datasets in LAADS archive
fileIDs = modapsClient.searchForFiles(
    cfgM["product"],
    cfg["startDate"],
    cfg["endDate"],
    cfg["extent"][3],
    cfg["extent"][2],
    cfg["extent"][1],
    cfg["extent"][0],
    dayNightBoth=u"D",
    collection=cfgM["collection"],
)
# Make an order
orderIDs = modapsClient.orderFiles(
    cfgM["email"],
    fileIDs,
    doMosaic=True,
    geoSubsetNorth=cfg["extent"][3],
    geoSubsetSouth=cfg["extent"][2],
    geoSubsetEast=cfg["extent"][1],
    geoSubsetWest=cfg["extent"][0],
    subsetDataLayer=cfgM["layers"],
)
print("orderIDs: ", orderIDs)

#%% Download the orders using wget when they are ready
# Assumes MacOS with wget installed - following command may need to be modified
# according to OS. Experience tells that one needs to wait approximately 10 min
# after receiving an email notification of the dataset's availability on LAADS
# before it is ready to be downloaded.

for i in range(len(orderIDs)):
    orderID = orderIDs[i]

    orderStatus = modapsClient.getOrderStatus(orderIDs[i])[0]

    if orderStatus == "Available":
        cmd = (
            "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 \
                https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/"
            + orderID
            + '/ --header "Authorization: Bearer '
            + cfgM["appKey"]
            + '" -P '
            + cfg["savePath"]
        )
        args = shlex.split(cmd)
        proc = sp.run(args, capture_output=True)

#%% PREPROCESSING
#%% Specify general preprocessing parameters
# This approach will find all MODAPS .hdf files and Worldview .jpeg files in
# Data/Download/sub, where sub is the Aqua or Terra download subfolder. Hence,
# adjust the 'sat' parameter here appropriately to process both Aqua and Terra.
# Consult the documentation of the SceneFilter class for details of the
# filtering process.

ensureWD(rootDir)
import Preprocess.SceneFilter

# Create folders to store filtered data in
filteredDirs = [rootDir + "/Data/Filtered"]
makeNewDirs(filteredDirs)

ppar = {
    "sat": "Terra",
    "startDate": datetime.datetime(2002, 12, 1),
    "endDate": datetime.datetime(2003, 1, 31),
    "loadPath": os.path.abspath(downloadDirs[1]),  # Aqua
    "savePath": os.path.abspath(filteredDirs[0]),
    "plot": False,
    "saveScenes": True,
    "saveOvl": True,
    "thrOv": 200,  # Minimum separation of overlapping scenes (pixels)
    "dp": 256,  # How far to perturb original images?
    "zenmax": 45,  # Max allowed zenith angle
    "npx": 512,  # Pixels in a filtered scene
    "thrCl": 1,  # Cloudy pixel classification threshold
    "hcThr": 5000,  # High cloud classification threshold
    "hcfr": 0.2,  # Allowed high cloud fraction
    "lat": [10, 20],
    "lon": [-58, -48],
}

#%% Preprocess downloaded files
sceneFilter = Preprocess.SceneFilter(ppar)
sceneFilter.filterScenes()

#%% METRIC COMPUTATION
#%% Specify metrics to be computed and create structures to store them in

ensureWD(rootDir)
from Metrics import createDataFrame, computeMetrics

metricDirs = [rootDir + "/Data/Metrics"]
makeNewDirs(metricDirs)

# Main metrics. Various accompanying metrics will be appended to this list
# Network metrics (e.g. netVarDeg) are not by default included in this code
# repository, as they derive from an external, private package. Please contact
# Franziska Glassmeier at f.glassmeier@tudelft.nl for access. Copy this package
# into the Metrics subfolder for use.
metrics = [
    "cf",  # Cloud fraction
    "cwp",  # Total cloud water path
    "lMax",  # Max length scale of scene's largest object
    "periSum",  # Total perimeter of all scene's cloud objects
    "cth",  # Mean cloud top height
    "sizeExp",  # Exponent of cloud size distribution (power law fit)
    "lMean",  # Mean length of cloud object in scene
    "specLMom",  # Spectral length scale (Jonker et al. 1999)
    "cop",  # Convective Organisation Potential White et al. (2018)
    "scai",  # Simple Convective Aggregation Index Tobin et al. (2012)
    "nClouds",  # Number of clouds in scene
    "rdfMax",  # Max of the radial distribution function of objects
    "netVarDeg",  # Degree variance of nearest-neighbour network of objects
    "iOrgPoiss",  # Organisation index as used in Tompkins & Semie (2017)
    "fracDim",  # Minkowski-Bouligand dimension
    "iOrg",  # Organisation index as modified by Benner & Curry (1998)
    "os",  # Contiguous clear sky area estimate (Antonissen, 2019)
    "twpVar",  # Variance in CWP anomaly on scales larger than 16 km (Bretherton & Blossey, 2017)
    "cthVar",  # Variance in cloud top height
    "cwpVarCl",  # Variance in cloud water path
    "woi3",  # Wavelet-based organisation index of orientation (Brune et al., 2018)
    "orie",  # Image raw moment covariance-based orientation metric
]

# Create an empty metric dataframe with these metrics as its columns
createDataFrame.createMetricDF(filteredDirs[0], metrics, metricDirs[0])

# Create a Numpy array filled with all images (may be very large)
createDataFrame.createImageArr(filteredDirs[0], metricDirs[0])

# Specify general parameters for metric computation
fields = {
    "cm": "Cloud_Mask_1km",
    "im": "image",
    "cth": "Cloud_Top_Height",
    "cwp": "Cloud_Water_Path",
}
mpar = {
    "loadPath": filteredDirs[0],
    "savePath": metricDirs[0],
    "save": True,
    "saveExt": "",  # Extension to filename to save in
    "resFac": 1,  # Resolution factor (e.g. 0.5)
    "plot": False,  # Plot with details on each metric computation
    "con": 1,  # Connectivity for segmentation (1:4 seg, 2:8 seg)
    "areaMin": 4,  # Minimum cloud size considered for object metrics
    "fMin": 0,  # First scene to load
    "fMax": None,  # Last scene to load. If None, is last scene in set
    "fields": fields,  # Field naming convention
}

#%% Compute

# Compute metrics in specified list and store them in the DataFrame. Each
# metric is implemented as a class with a method (metric) that operates
# directly on a specified field and returns a metric, and a method (compute)
# that additionally handles loading and storing of fields in the DataFrames
# introduced in the previous sections. The function computeMetrics applies
# compute to the entire list of input metrics. Please consult each individual
# metric's documentation for details on their computation.

# (This may take a very long time depending on which metrics should be computed
# and the size of the dataset)
computeMetrics.computeMetrics(metrics, mpar)

# An alternative way to compute the metrics is to handle the dataframes
# outside the Metric objects and only use their metric() methods. This may be
# more flexible if one only wishes to compute a metric and have it in memory
# immediately, or works with field names that are different from those used
# here. This method can also be called without passing mpar to the metric
# object upon instantiation. However, this will set the parameters plot, con
# and areaMin to their defaults (False, 1 and 4). For all metrics to be
# computable, one must at least have the field names 'cm' (cloud mask), 'im'
# (image), 'cwp' (cloud water path) and 'cth' (cloud-top height) available.
# The following snippet can then build a metric  dataframe (assuming the input
# data uses our storage structure/naming - modify this as appropriate):

# # 1. Find data
# from Metrics.utils import findFiles
# files, dates = findFiles(filteredDirs[0])

# # 2. Create a dataframe to store in
# from Metrics.createDataFrame import getAllMetrics
# columns = getAllMetrics(metrics)
# dfMetrics = pd.DataFrame(index=dates,columns=columns)

# # 3. Loop over data, computing a set of metrics for each scene
# for i in range(len(files)):
#     data = pd.read_hdf(files[i])
#     fields = {'cm'  : data['Cloud_Mask_1km'].values[0],
#               'im'  : data['image'].values[0],
#               'cth' : data['Cloud_Top_Height'].values[0],
#               'cwp' : data['Cloud_Water_Path'].values[0]}
#     dfout = computeMetrics.evaluateMetrics(metrics,fields)
#     dfMetrics.loc[dates[i]] = dfout.values

# computeMetrics might fail to compute a metric on a scene, if that scene is
# for any reason not suitable (e.g. if there are very few detected clouds).
# The following function can, therefore, remove a scene from the dataset.

# from Metrics.utils import removeScene
# date = '2002-12-01-a-0' # for example
# removeScene(date, filteredDirs[0], metricDirs[0])

#%% POSTPROCESSING
#%% Specify general postprocessing parameters

ensureWD(rootDir)
from Postprocess import analysis

plotDirs = [rootDir + "/Data/Plots"]
makeNewDirs(plotDirs)

# Subset of metrics to be analysed
# netVarDeg can be included upon request (see Metric Computation section above)
metricsPP = [
    "cf",
    "cwp",
    "lMax",
    "periSum",
    "cth",
    "sizeExp",
    "lMean",
    "specLMom",
    "cop",
    "scai",
    "nClouds",
    "rdfMax",
    "netVarDeg",
    "iOrgPoiss",
    "fracDim",
    "iOrg",
    "os",
    "twpVar",
    "cthVar",
    "cwpVarCl",
    "woi3",
]

metLab = [
    "Cloud fraction",
    "Cloud water",
    "Max length",
    "Perimeter",
    r"$\overline{CTH}$",
    "Size exponent",
    "Mean length",
    "Spectral length",
    "COP",
    r"SCAI",
    "Cloud number",
    "Max RDF",
    "Degree var",
    r"$I_{org}$",
    "Fractal dim.",
    r"$I_{org}^*$",
    "Clear sky",
    "CWP var ratio",
    r"St(CTH)",
    r"St(CWP)",
    r"$WOI_3$",
]

# Load, order and standardise data
dfMetrics, data, imgArr = analysis.loadMetrics(
    metricDirs[0],
    metricsPP,
    sort=True,
    standardise=True,
    return_data=True,
    return_images=True,
)

#%% Analysis - Specific analysis routines for plots that appear in the paper.

# Correlation matrix (fig. S2)
analysis.correlate(data, metricsPP, metLab, plotDirs[0])

# Show how metrics order scenes (fig. 1)
analysis.plotSortedScenes(data, imgArr, metLab, plotDirs[0])

# Compute PCA
pca = PCA()
xPca = pca.fit_transform(data)

# Relate metrics to PCs (fig. S3)
analysis.relateMetricPCA(pca, xPca, metricsPP, metLab, plotDirs[0])

# Plot PCA distribution (fig. 2)
analysis.pcaDistribution(pca, xPca, plotDirs[0])

# Regime analysis (fig. 4)
analysis.regimeAnalysis(xPca, imgArr, plotDirs[0])

# PCA surfaces (fig. 3)
# This function has exceptionally high memory requirement: If it fails, try
# again in a 'clean' console/terminal without plotting anything else.
analysis.plotPCASurfs(
    data, imgArr, dfMetrics, metricsPP, metLab, pca, xPca, plotDirs[0]
)

#%% Sensitivity tests - Create dataframes of metrics for perturbed cases
# Must have run regular metric computation first, and, within the current
# session, run the metric definition cell.

# Computes the sensitivity of the (high-dimensional) metric distribution to
# choices in free parameters in the field processing. We compute this by
# comparing ratios of high-dimensional kernel density estimates of the
# perturbed distribution and the original distribution, to reduce the
# dimensionality of the distribution comparison to 1. That allows us to use the
# Kolmogorov-Smirnov metric as gauge for the similarity of the perturbed, and
# original distributions.

from Postprocess import sensitivity

sensLabels = ["res0.5", "8con", "0min"]

# 1. Half resolution
mpar["resFac"] = 0.5
mpar["saveExt"] = sensLabels[0]
createDataFrame.createMetricDF(
    filteredDirs[0], metrics, metricDirs[0], saveExt=mpar["saveExt"]
)
computeMetrics.computeMetrics(metrics, mpar)

# 2. 8-connectivity
mpar["resFac"] = 1
mpar["con"] = 2
mpar["saveExt"] = sensLabels[1]
createDataFrame.createMetricDF(
    filteredDirs[0], metrics, metricDirs[0], saveExt=mpar["saveExt"]
)
computeMetrics.computeMetrics(metrics, mpar)

# 3. Minimum cloud size 0
mpar["con"] = 1
mpar["areaMin"] = 0
mpar["saveExt"] = sensLabels[2]
createDataFrame.createMetricDF(
    filteredDirs[0], metrics, metricDirs[0], saveExt=mpar["saveExt"]
)
computeMetrics.computeMetrics(metrics, mpar)

#%% Compute sensitivity
# Load, order, standardise and take PCA of data
dfMetrics, data = analysis.loadMetrics(metricDirs[0], metricsPP, return_images=False)
pca = PCA()
xPca = pca.fit_transform(data)

dfMetrics1, data1 = analysis.loadMetrics(
    metricDirs[0], metricsPP, return_images=False, ext=sensLabels[0]
)

pca1 = PCA()
xPca1 = pca1.fit_transform(data1)

dfMetrics2, data2 = analysis.loadMetrics(
    metricDirs[0], metricsPP, return_images=False, ext=sensLabels[1]
)
pca2 = PCA()
xPca2 = pca2.fit_transform(data2)

dfMetrics3, data3 = analysis.loadMetrics(
    metricDirs[0], metricsPP, return_images=False, ext=sensLabels[2]
)
pca3 = PCA()
xPca3 = pca3.fit_transform(data3)

# Compute sensitivity
sensitivity.computeSensitivity(xPca, xPca1, xPca2, xPca3, plotDirs[0], npts=1e4)


#%% Quantification of embedding quality
# Measures the quality of the embedding (metric representation of organisation)
# by comparing the Euclidian distance between half-overlapping scenes in the
# dataset (expected to be small) to the Euclidian distance between a scene and
# a randomly selecteed, other scene in the dataset (expected to be large).
# A similarity measure S, which subtracts this ratio from 1, can then give an
# indication of how well the metrics encapsulate the essential patterns in the
# scenes, assuming these patterns usually extend beyond a scene's borders. If
# S = 0, the metrics are unable to distinguish random scenes frmo overlapping
# scenes; if S = 1, overlapping scenes are identical.

from Postprocess import measureEmbedding

ovlPath = rootDir + "/Data"

# Load metrics, as well as information on overlapping scenes
dfMetrics, data = analysis.loadMetrics(metricDirs[0], metricsPP, return_images=False)
dfOvl = measureEmbedding.loadDfOvl(ovlPath)

# Compute similarity score (fig. S4)
measureEmbedding.analyseOverlap(dfMetrics, dfOvl, plotDirs[0])

#%% Metric subset selection (including sparse PCA)

from Postprocess import spca

# Sensitivity of sparse PCA (fig. S5)
nComp = 4  # Analyse only first four components
spca.sensitivity(data, metLab, nComp, plotDirs[0])

# Orthogonal explained variance ratio of several manual selections here and in
# literature. Note that this only measures orthogonality, not whether the
# metrics themsleves orient along directions that themselves explain a large
# amount of variance.

# Optimal 4D choice
inds = [7, 16, 20, 18]  # specLMom  # os  # woi3  # St(CTH)
evrO = spca.orthogonalMetricVar(data, inds)

# 2D choices
inds = [
    7,  # specLMom
    16,  # os
]
evr2O = spca.orthogonalMetricVar(data, inds)

inds = [
    0,  # cf
    14,  # fracDim
]
evr21 = spca.orthogonalMetricVar(data, inds)

inds = [3, 15]  # periSum  # Iorg
evr22 = spca.orthogonalMetricVar(data, inds)

# Other choices in literature
# Bony et al. (2020)
inds = [10, 13]  # nClouds  # iorgTomp
evrB = spca.orthogonalMetricVar(data, inds)

# Seifert & Heus (2013)
inds = [7, 13]  # specLMom  # iorgTomp
evrB = spca.orthogonalMetricVar(data, inds)

# Denby (2020)
inds = [13, 14]  # iorgTomp  # fracDim
evrD = spca.orthogonalMetricVar(data, inds)

# van Laar (2019)
inds = [
    13,  # iorgTomp
    9,  # scai
    8,  # COP
    11,  # maxRdf
]
evrL = spca.orthogonalMetricVar(data, inds)
