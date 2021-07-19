#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


def loadDfOvl(path):
    # Check if there are multiple ovl files in path, with -a and -t extensions

    terra = False
    aqua = False
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file == "ovl-a.h5":
                fA = file
                aqua = True
            elif file == "ovl-t.h5":
                fT = file
                terra = True
    if aqua and terra:
        dfA = pd.read_hdf(path + "/" + fA)
        dfT = pd.read_hdf(path + "/" + fT)
        dfOvl = pd.concat([dfA, dfT])

    elif aqua and not terra:
        dfOvl = pd.read_hdf(path + "/" + fA)

    elif terra and not aqua:
        dfOvl = pd.read_hdf(path + "/" + fT)

    else:
        raise FileNotFoundError("No overlap df detected in " + str(path))

    dfOvl.sort_index(inplace=True)

    return dfOvl


def getPairs(dfMetrics, dfOvlIn, ovlDist=256):

    # Filter from dfOvl:
    #   - Scenes in iovl that do not exist in dfOvl's index (due to high cloud filtering)
    #   - Scenes in iovl that are not in dfMetrics (due to manual removal from sunglint)
    #   - Scenes in dfOvl's index that are not in dfMetrics (ditto)
    #   - Scenes that have nans (due to that scene not overlapping with any other)
    #   - Scenes that overlap by less than the specified threshold (256 pixels)
    #   - Recurring pairs

    dfOvl = dfOvlIn.copy()
    dfOvl = dfOvl[dfOvl["iovl"].isin(dfOvl.index)]
    dfOvl = dfOvl[dfOvl["iovl"].isin(dfMetrics.index)]
    dfOvl = dfOvl[dfOvl.index.isin(dfMetrics.index)]
    dfOvl = dfOvl[~dfOvl["iovl"].isnull()]
    dfOvl = dfOvl[dfOvl["dist"] == ovlDist]

    test = np.array([dfOvl.index.to_numpy(), dfOvl["iovl"].to_numpy()]).T
    test.sort(axis=1)
    test = test[:, 0] + test[:, 1]
    _, ind = np.unique(test, return_index=True)
    dfOvl = dfOvl.iloc[ind]

    return dfOvl


def computeDistance(
    dfMetrics,
    dfOvlIn,
    ovlDist=256,
    nRand=3,
    seed=True,
    shuffle=True,
    return_iRand=False,
):

    dfOvl = getPairs(dfMetrics, dfOvlIn, ovlDist)
    if seed:
        np.random.seed(0)
    iRand = np.random.randint(0, dfOvl.shape[0], (dfOvl.shape[0], nRand))
    indRand = dfOvl.index.to_numpy()[iRand]

    anch = dfMetrics.loc[dfOvl.index].to_numpy()
    neig = dfMetrics.loc[dfOvl["iovl"]].to_numpy()
    rand = np.zeros((iRand.shape[0], iRand.shape[1], dfMetrics.shape[1]))
    for i in range(nRand):
        rand[:, i, :] = dfMetrics.loc[indRand[:, i]]

    dists = np.zeros((dfOvl.shape[0], nRand + 1))
    dists[:, 0] = np.sqrt(np.sum((neig - anch) ** 2, axis=1).astype("float32"))
    dists[:, 1:] = np.sqrt(
        np.sum((rand - anch[:, np.newaxis, :]) ** 2, axis=2).astype("float32")
    )

    if shuffle:
        if seed:
            np.random.seed(10)
        order = np.random.permutation(len(dfOvl))
        dfOvl = dfOvl.iloc[order]
        dists = dists[order, :]
        iRand = iRand[order, :]

    if return_iRand:
        return dfOvl, dists, iRand
    else:
        return dfOvl, dists


def getDistribution(
    dfMetrics, dfOvlIn, ovlDist=256, nShuf=100, seed=False, shuffle=False
):
    # Return a filtered version of dfOvl with two added columns:
    # - nearDist: Euclidian distance in metric space to overlapping image
    # - randDist: Average Euclidian distance in metric space to nShuf randomly
    #             assigned scenes from the filtered dfOvl

    dfOvl, dists = computeDistance(dfMetrics, dfOvlIn, ovlDist, nShuf, seed, shuffle)

    dfOvl = dfOvl.reindex(columns=dfOvl.columns.tolist() + ["nearDist", "randDist"])
    dfOvl["nearDist"] = dists[:, 0]
    dfOvl["randDist"] = np.mean(dists[:, 1:], axis=1)

    return dfOvl


def analyseOverlap(ndDfMet, dfOvlIn, savePath):

    dfOvl = getDistribution(ndDfMet, dfOvlIn)

    mnNear = np.mean(dfOvl["nearDist"])
    stNear = np.std(dfOvl["nearDist"])
    mnRand = np.mean(dfOvl["randDist"])
    stRand = np.std(dfOvl["randDist"])

    print("Mean distance/std for overlapping scenes: ", mnNear, stNear)
    print("Mean distance/std for random scenes:      ", mnRand, stRand)

    ax = sns.kdeplot(
        dfOvl["nearDist"], shade=True, label="Overlapping scenes", color="midnightblue"
    )
    ax = sns.kdeplot(
        dfOvl["randDist"], shade=True, label="Random scenes", color="darkseagreen"
    )
    ax.plot([mnNear, mnNear], [0, 0.5], color="midnightblue", linestyle="--")
    ax.plot([mnRand, mnRand], [0, 0.5], color="darkseagreen", linestyle="--")
    ax.set_xlim((0, 12))
    ax.set_ylim((0, 0.5))
    ax.set_xlabel("Euclidian distance from anchor scene")
    ax.set_ylabel("Relative occurrence")
    plt.savefig(savePath + "/distS.pdf", bbox_inches="tight")
    plt.show()

    # Compute a number of statistical distances between the distributions
    mdist = np.abs(mnNear - mnRand)
    t, p = ss.ttest_ind(dfOvl["nearDist"], dfOvl["randDist"])
    kld = ss.entropy(dfOvl["nearDist"], dfOvl["randDist"])
    chiSq = ss.chisquare(dfOvl["nearDist"], dfOvl["randDist"])
    chiSqd = np.sqrt(chiSq.statistic)

    print("Mean distance:               ", mdist)
    print("t-value:                     ", t)
    print("Kullback-Leibler divergence: ", kld)
    print("Chi-squared:                 ", chiSqd)

    # Ratio of overlapping and non-overlapping scenes
    simRat = 1.0 - dfOvl["nearDist"] / dfOvl["randDist"]
    mnRat = np.mean(simRat)
    stRat = np.std(simRat)
    print("Mean similarity ratio:       ", mnRat)

    # Fraction of distributions that overlap
    per = 0.05  # Fraction of random distribution after which to measure overlap
    randSort = np.sort(dfOvl["randDist"].to_numpy())
    nearSort = np.sort(dfOvl["nearDist"].to_numpy())
    iminRand = int(round(len(randSort) * per))
    distMin = randSort[iminRand]
    imaxNear = np.where(nearSort < distMin)[0]
    nearPer = nearSort[imaxNear]
    nearFrac = len(nearPer) / len(nearSort)
    print(
        "Fraction of overlap distribution <" + str(per) + " of random scenes:    ",
        nearFrac,
    )

    # Fraction of scenes where the distance to overlap is less than that to random
    ovlLess = np.where(dfOvl["nearDist"] < dfOvl["randDist"])[0].shape[0]
    nearFracEl = ovlLess / len(dfOvl["nearDist"])
    print("Fraction of overlapping scenes that are closer than random: ", nearFracEl)

    fig1 = plt.figure()
    ax1 = sns.kdeplot(simRat, shade=True, color="midnightblue")
    ax1.set_xlim((-0.5, 1))
    ax1.set_ylim((0, 2.25))
    ax1.plot(
        [mnRat, mnRat],
        [0, 2.25],
        color="midnightblue",
        linestyle="--",
        label=r"$\overline{S}$",
    )
    # ax1.set_xlabel(r'$1 - \frac{\Vert x_{a_i} - x_{n_i}\Vert}{\Vert x_{a_i} - x_{r_i}\Vert}$')
    ax1.set_xlabel(r"$S$")
    ax1.set_ylabel("Kernel density estimate")
    ax1.legend()
    ax1.minorticks_on()
    plt.savefig(savePath + "/distFrac.pdf", bbox_inches="tight")
