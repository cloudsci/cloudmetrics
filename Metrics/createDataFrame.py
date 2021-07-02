#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from .utils import findFiles, anyInList, uniqueAppend


def getAllMetrics(metricList):
    metrics = metricList.copy()

    scaiList = ["scai", "d0"]
    if anyInList(metrics, scaiList):
        metrics = uniqueAppend(metrics, scaiList)

    fourList = ["beta", "betaa", "specL", "specLMom", "psdAzVar"]
    if anyInList(metrics, fourList):
        metrics = uniqueAppend(metrics, fourList)

    cwpList = ["cwp", "cwpVar", "cwpVarCl", "cwpSke", "cwpKur"]
    if anyInList(metrics, cwpList):
        metrics = uniqueAppend(metrics, cwpList)

    objectList = ["lMax", "lMean", "nClouds", "eccA", "periSum"]
    if anyInList(metrics, objectList):
        metrics = uniqueAppend(metrics, objectList)

    cthList = ["cth", "cthVar", "cthSke", "cthKur"]
    if anyInList(metrics, cthList):
        metrics = uniqueAppend(metrics, cthList)

    rdfList = ["rdfMax", "rdfInt", "rdfDiff"]
    if anyInList(metrics, rdfList):
        metrics = uniqueAppend(metrics, rdfList)

    networkList = [
        "netVarDeg",
        "netAWPar",
        "netCoPar",
        "netLPar",
        "netLCorr",
        "netDefSl",
        "netDegMax",
    ]
    if anyInList(metrics, networkList):
        metrics = uniqueAppend(metrics, networkList)

    woiList = ["woi1", "woi2", "woi3", "woi"]
    if anyInList(metrics, woiList):
        metrics = uniqueAppend(metrics, woiList)

    osList = ["os", "osAv"]
    if anyInList(metrics, osList):
        metrics = uniqueAppend(metrics, osList)

    return metrics


def createMetricDF(loadPath, metrics, savePath, saveExt=""):
    metrics = getAllMetrics(metrics)

    _, dates = findFiles(loadPath)
    df = pd.DataFrame(columns=metrics, index=dates)
    df.to_hdf(savePath + "/Metrics" + saveExt + ".h5", "Metrics", mode="w")


def createImageArr(loadPath, savePath, imageTag="image", sortTime=False):
    files, dates = findFiles(loadPath)

    # Test field size and initialise
    df = pd.read_hdf(files[0])
    img = df[imageTag].values[0].copy()
    npx = img.shape[0]  # Explicitly assumes square subset
    dfImgs = np.zeros((len(dates), npx, npx))

    if sortTime:
        if dates.dtype != "float64":
            dates = dates.astype("float64")
        files = files[np.argsort(dates)]

    for f in range(len(files)):
        df = pd.read_hdf(files[f])
        img = df[imageTag].values[0].copy()
        dfImgs[f, :, :] = img
        if f % 100 == 0:
            print("Loaded", f, "of", len(files), "images")
    np.save(savePath + "/Images.npy", dfImgs)
