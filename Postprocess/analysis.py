#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import seaborn as sns
import seaborn.utils

from .utils import (
    stand,
    plotCorr,
    rainbowArrow,
    plot2dClusters,
    plotClusteredImages,
    getGrids,
    plotEmbedding,
    plotArrow,
    plotMetricSurf,
    rotFlip,
)


def loadMetrics(
    dirPath,
    metrics=None,
    ext="",
    sort_data=False,
    sort_images=False,
    standardise=True,
    return_data=True,
    return_images=True,
    fname="Metrics",
):

    if sort_images and not sort_data:
        print("sort_images can only be set to True when sort_data is True")
        return

    df = pd.read_hdf(dirPath + "/" + fname + ext + ".h5")
    if metrics == None:
        dfMetrics = df
    else:
        dfMetrics = df[metrics]
    dfMetrics.drop_duplicates(inplace=True)
    if sort_data:
        if dfMetrics.index.dtype != "float64":
            dfMetrics.index = dfMetrics.index.astype("float64")
        order = dfMetrics.index.argsort()
        dfMetrics = dfMetrics.sort_index()
    if standardise:
        dfMetrics = stand(dfMetrics)

    # More data
    if return_data and return_images:
        imgarr = np.load(dirPath + "/Images" + ext + ".npy")
        if sort_images:
            imgarr = imgarr[order, :, :]
        data = dfMetrics.to_numpy()
        return dfMetrics, data, imgarr
    elif return_data and not return_images:
        data = dfMetrics.to_numpy()
        return dfMetrics, data
    elif not return_data and return_images:
        imgarr = np.load(dirPath + "/Images" + ext + ".npy")
        if sort_images:
            imgarr = imgarr[order, :, :]
        return dfMetrics, imgarr
    else:
        return dfMetrics


def correlate(ndata, metrics, metLab, savePath):
    rPM = np.zeros((len(metrics), len(metrics)))
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            rPM[i, j] = pearsonr(ndata[:, i], ndata[:, j])[0]

    mask = np.triu_indices(len(metrics))
    rPM[mask] = 0

    plotCorr(
        rPM,
        xticklabels=metLab,
        yticklabels=metLab,
        maxSquares=False,
        flip=True,
        cbLab="Absolute Pearson correlation",
    )
    plt.savefig(savePath + "/corrMetric.pdf", bbox_inches="tight")


def plotSortedScenes(ndata, imgarr, metLab, savePath, iex=10):  # iex is outlier removal
    # Plot scenes sorted by metric - vertically oriented
    nrows = ndata.shape[1]
    ncols = 7
    sub = 0
    fs = 4
    xColor = np.linspace(0.2, 0.6, 3)
    cs = cm.get_cmap(plt.get_cmap("cubehelix"))(xColor)[:, :3]
    # Colour scheme: 0 - field statistic; 1 - object metric; 2 - field transform
    legLabs = ["Field statistics", "Object metrics", "Scaling properties"]
    colors = [
        cs[0],  # cf
        cs[0],  # cwp
        cs[1],  # lMax
        cs[1],  # periSum
        cs[0],  # cth
        cs[2],  # sizeExp
        cs[1],  # lMean
        cs[2],  # beta
        cs[1],  # COP
        cs[1],  # SCAI
        cs[1],  # nClouds
        cs[1],  # rdfMax
        cs[1],  # netVarDeg
        cs[1],  # iorgTomp
        cs[2],  # fracDim
        cs[1],  # Iorg
        cs[0],  # os
        cs[0],  # twpVar
        cs[0],  # cthVar
        cs[0],  # cspVarCl
        cs[2],  # woi3
    ]

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3, 3 * nrows / ncols))
    for i in range(nrows):
        lati = np.argsort(ndata[:, i])
        idx = np.round(np.linspace(iex, len(lati) - 1 - iex, ncols)).astype(int)
        lati = lati[idx]
        sortImg = imgarr[lati, :, :]
        for j in range(ncols):
            axs[i, j].imshow(sortImg[j], "gray", label=colors[i])
            axs[i, j].set_axis_off()
        axs[i, 0].set_title(
            metLab[i], fontsize=fs, color=colors[i], loc="right", x=-0.1, y=0.1
        )

    lbwh = [0.1, 0.89, 0.82, 0.01]
    arax = fig.add_axes(lbwh)
    arax.axis("off")
    rainbowArrow(
        arax,
        (lbwh[0], lbwh[1]),
        (lbwh[0] + lbwh[2], lbwh[1] - lbwh[3]),
        cmap="Blues",
        n=200,
        lw=lbwh[3],
    )
    laboffs = 0.0075
    labax = fig.add_axes([lbwh[0] - 0.05, lbwh[1] + laboffs, lbwh[2] + 0.15, laboffs])
    labax.axis("off")
    labax.annotate("Low", (lbwh[0], lbwh[1]), fontsize=fs)
    labax.annotate("High", (lbwh[2], lbwh[1]), fontsize=fs)
    legLines = [
        Line2D([0], [0], color=cs[0], lw=1),
        Line2D([0], [0], color=cs[1], lw=1),
        Line2D([0], [0], color=cs[2], lw=1),
    ]
    fig.legend(
        legLines,
        legLabs,
        fontsize=fs,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.1),
        ncol=3,
    )
    plt.subplots_adjust(wspace=0.001)
    plt.savefig(savePath + "/sort.png", dpi=600, bbox_inches="tight")
    plt.show()


def relateMetricPCA(pca, X_pca, metrics, metLab, savePath):
    # Project metrics along all directions to quantify how well they describe a
    # PCA direction

    # Shape of rP: (nPCAcomp,nMetrics
    rV = np.zeros((X_pca.shape[1], len(metrics)))
    for i in range(X_pca.shape[1]):
        Xhat = np.dot(X_pca[:, : i + 1], pca.components_[: i + 1, :])  # Approx. of data
        rV[i, :] = np.var(Xhat, axis=0)
    rVD = np.diff(np.vstack((np.zeros((1, len(metrics))), rV)), axis=0)

    # Correlation plot
    plotCorr(
        rVD,
        xlabel="Principal component",
        ylabel="Metric",
        xticklabels=np.arange(1, X_pca.shape[1] + 1),
        yticklabels=metLab,
        maxSquares=False,
        size="pca",
        pca=pca,
        cbLab="Fraction of variance explained",
    )
    plt.savefig(savePath + "/corrMetricPCA.pdf", bbox_inches="tight")
    plt.show()


def regimeAnalysis(X_pca, imgarr, savePath):
    ## Clustering - Results ##
    nc = 7  # Just set it
    nDim = 4
    rot2rad = -49 * np.pi / 180

    Xpl = X_pca[:, :nDim] / np.std(X_pca[:, :nDim], axis=0)
    Xpl = rotFlip(Xpl, 0, flipAx=2, rotAxes=None)  # Flip PC 3 (clarity)
    lab = ["Scale", "Void", "Directional alignment", "Cloud top height variance"]

    co = np.arange(nc)
    cl = KMeans(n_clusters=nc, random_state=100)
    cl.fit(Xpl)  # In the full, high-dimensional space

    Xpl = rotFlip(Xpl, rot2rad, flipAx=None, rotAxes=[2, 3])  # Rotate (clarity)

    # Find the modal point from histograms
    mode = np.zeros(nDim)
    for i in range(nDim):
        hist, edges = np.histogram(Xpl[:, i], bins=100)
        amax = np.argmax(hist)
        mode[i] = (edges[amax] + edges[amax + 1]) / 2

    fig = plt.figure(figsize=(30, 15))
    gs0 = fig.add_gridspec(1, 2)
    gs00 = gs0[0].subgridspec(nDim - 1, nDim - 1)
    gs01 = gs0[1].subgridspec(nc, nc)

    colors = [
        "midnightblue",
        "palevioletred",
        "maroon",
        "peachpuff",
        "peru",
        "darkseagreen",
        "steelblue",
    ]

    af = "axes fraction"
    for i in range(gs00.get_geometry()[0]):
        for j in range(gs00.get_geometry()[1]):
            if j <= i:
                ax = fig.add_subplot(gs00[i, j])
                ax = plot2dClusters(
                    Xpl[:, [j, i + 1]],
                    cl,
                    co,
                    filterOutliers=True,
                    ax=ax,
                    sMarker=1.5,
                    clrs=colors,
                )
                ax.scatter(mode[j], mode[i + 1], marker="x", c="k", s=100)
                ax.scatter(0, 0, marker="+", c="k", s=100)
            if i == 0:
                ax.annotate("S", (0.075, 0.38), xycoords=af, fontsize=19)
                ax.annotate("G", (0.3, 0.37), xycoords=af, fontsize=19)
                ax.annotate("Fi", (0.52, 0.41), xycoords=af, fontsize=19)
                ax.annotate("Fl", (0.82, 0.43), xycoords=af, fontsize=19)
            elif i == gs00.get_geometry()[1] - 1 and j == gs00.get_geometry()[0] - 1:
                ax.annotate("S", (0.52, 0.2), xycoords=af, fontsize=19)
                ax.annotate("G", (0.15, 0.43), xycoords=af, fontsize=19)
                ax.annotate("Fi", (0.4, 0.4), xycoords=af, fontsize=19)
                ax.annotate("Fl", (0.27, 0.2), xycoords=af, fontsize=19)
            if j == 0:
                ax.set_ylabel(lab[i + 1], fontsize=19)
            if i == gs00.get_geometry()[0] - 1:
                ax.set_xlabel(lab[j], fontsize=19)

    axs = np.empty((nc, nc), dtype="object")
    for i in range(nc):
        for j in range(nc):
            axs[i, j] = fig.add_subplot(gs01[i, j])

    # Plot images in a row
    plotClusteredImages(cl, co, Xpl, imgarr, cnt=nc, rand=False, clrs=colors, axs=axs)
    plt.savefig(savePath + "/regimeAnalysis.png", dpi=200, bbox_inches="tight")
    plt.show()


def pcaDistribution(pca, X_pca, savePath, ncomp=4):
    # Visualise how the PCs compare to each other
    lim = 6
    rot2rad = -49 * np.pi / 180
    X_pca = rotFlip(X_pca, rot2rad, flipAx=2, rotAxes=[2, 3])
    df_pca = pd.DataFrame(data=X_pca[:, :ncomp])
    sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18})
    g = sns.PairGrid(df_pca, corner=True, diag_sharey=False)
    g.fig.set_size_inches(20, 20)
    g.map_diag(sns.kdeplot, color="k")
    cbar_ax = g.fig.add_axes([0.91, 0.325, 0.015, 0.4])
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            if g.axes[i, j] is not None and i is not j:
                if i == ncomp - 1 and j == ncomp - 2:
                    cbax = cbar_ax
                    cbFlag = True
                else:
                    cbax = None
                    cbFlag = False
                g.axes[i, j] = sns.kdeplot(
                    g.data.iloc[:, j],
                    g.data.iloc[:, i],
                    ax=g.axes[i, j],
                    shade=True,
                    shade_lowest=False,
                    legend=False,
                    vmin=0,
                    vmax=0.12,
                    n_levels=100,
                    cbar=cbFlag,
                    cbar_ax=cbax,
                    cmap="cubehelix_r",
                    cbar_kws={"label": "Density"},
                )
                if cbFlag:
                    cax = plt.gcf().axes[-1]
                    cax.tick_params(labelsize=18)
                for col in g.axes[i, j].collections:
                    col.set_edgecolor("face")
                g.axes[i, j].set_xlim((-lim, lim))
                g.axes[i, j].set_ylim((-lim, lim))
                g.axes[i, j].tick_params(axis="both", which="major", labelsize=18)
            elif g.axes[i, j] is not None and i == j:
                g.axes[i, j].tick_params(axis="both", which="major", labelsize=18)
                evr = str(round(pca.explained_variance_ratio_[i], 2))
                evrS = str(round(np.sum(pca.explained_variance_ratio_[: i + 1]), 2))
                g.axes[i, j].annotate(
                    "EVR:  " + evr, (0.6, 0.9), xycoords="axes fraction"
                )
                g.axes[i, j].annotate(
                    "CEVR: " + evrS, (0.6, 0.8), xycoords="axes fraction"
                )
                g.diag_axes[i].set_axis_on()
                sns.utils.despine(ax=g.diag_axes[i], left=True, right=False)
                g.diag_axes[i].tick_params(
                    axis="y", which="major", labelsize=18, right=True
                )
                g.diag_axes[i].set_ylim((0, 0.4))
                g.diag_axes[i].set_ylabel("Density")

            if i == g.axes.shape[0] - 1 and j == g.axes.shape[1] - 1:
                g.axes[i, j].set_xlim((-lim, lim))
            if i == g.axes.shape[1] - 1:
                g.axes[i, j].set_xlabel("PC " + str(j + 1))
            if j == 0:
                g.axes[i, j].set_ylabel("PC " + str(i + 1))

    plt.savefig(savePath + "/pairgrid.pdf", bbox_inches="tight")
    plt.show()


def plotPCASurfs(
    ndata,
    imgarr,
    ndDfMet,
    metrics,
    metLab,
    pca,
    X_pca,
    savePath,
    nPts=20,
    ncols=6,
    thr=0.181,
    fs=24,
    thr2d=2.75,
    offs=0.05,
    lw=0.004,
    fac=1.3,
    zoom=0.15,
    distMin=1.5e-3,
    rot2rad=-49 * np.pi / 180,
):
    """
    Combined plot for PC (0,1) and PC (2,3) with surfaces underneath

    Parameters
    ----------
    ndata : TYPE
        DESCRIPTION.
    imgarr : TYPE
        DESCRIPTION.
    ndDfMet : TYPE
        DESCRIPTION.
    metrics : TYPE
        DESCRIPTION.
    metLab : TYPE
        DESCRIPTION.
    pca : TYPE
        DESCRIPTION.
    X_pca : TYPE
        DESCRIPTION.
    savePath : TYPE
        DESCRIPTION.
    nPts : float, optional
        Points to interpolate over. The default is 20.
    ncols : int, optional
        Columns of subplots under the main plot. The default is 6.
    thr : float, optional
        Gradient threshold in plane. The default is 0.181.
    fs : float, optional
        font size in plots. The default is 24.
    thr2d : float, optional
        Distance-from-plane threshold for gradient estimates. The default is
        2.75.
    offs : float, optional
        Image offset from the plot boundary. The default is 0.05.
    lw : float, optional
       Line width of interpretation arrows. The default is 0.004.
    fac : float, optional
        Scale of interpretation arrows. The default is 1.3.
    zoom : float, optional
        Zoom of images. The default is 0.15.
    distMin : float, optional
        Minimum distance between images. The default is 1.5e-3.
    rot2rad : float, optional
        Rotation angle in radians. The default is -49*np.pi/180.

    Returns
    -------
    None.

    """
    # Flip and rotate for clarity
    X_pca = rotFlip(X_pca, rot2rad, flipAx=2, rotAxes=[2, 3])

    # Project points onto plane spanned by first principal components
    comp = [0, 1]

    pax = pca.components_[comp, :]
    proj = np.dot(
        np.dot(pax, ndata.transpose()).transpose(), pax
    )  # No need to normalise, projection, as pca components are already orthonormal
    diff = ndata - proj
    dist = np.sqrt(np.sum(diff ** 2, axis=1).astype(np.float64))
    weights = np.ones_like(dist) / float(len(dist))
    X_2d = X_pca[dist < thr2d, :][:, comp]
    img_2d = imgarr[dist < thr2d]
    met_2d = ndDfMet[dist < thr2d]

    grid, grids, pltLab = getGrids(X_2d, met_2d, metrics, metLab, nPts, thr=thr)
    nrows = int(np.ceil(len(grids) / ncols))

    fig = plt.figure(figsize=(30, 15 + 1.75 * nrows))
    gs = fig.add_gridspec(ncols=2 * ncols, nrows=ncols + nrows)
    ax1 = fig.add_subplot(gs[:ncols, :ncols])
    ax1 = plotEmbedding(
        X_pca[:, comp],
        imgarr,
        filterOutliers=True,
        zoom=zoom,
        distMin=distMin,
        ax=ax1,
        offs=offs,
    )

    # Plot interpretable directions as arrows
    metricsArrow = ["cf", "iOrg", "beta", "os", "iOrgPoiss"]
    metLabArrow = [
        "Coverage",
        "Space filling",
        "Scaling",
        "Void",
        "Clustering/aggregation",
    ]
    metPlot = met_2d[metricsArrow]
    clrs = cm.get_cmap(plt.get_cmap("cubehelix"))(
        np.linspace(0.05, 0.95, len(metricsArrow))
    )[:, :3]

    ax1 = plotArrow(
        X_2d,
        metPlot,
        metricsArrow,
        ax1,
        metLabArrow,
        leg=True,
        lw=lw,
        fac=fac,
        thr=thr,
        clrs=clrs,
        fs=fs,
    )

    ax1.set_xlabel("Principal component " + str(comp[0] + 1), fontsize=fs)
    ax1.set_ylabel("Principal component " + str(comp[1] + 1), fontsize=fs)
    # ax1.annotate('a',(0.475,0.41),xycoords='figure fraction',fontsize=fs+25)
    axs = []
    for i in range(nrows):
        for j in range(ncols):
            axs.append(fig.add_subplot(gs[ncols + i, j]))
    axs = plotMetricSurf(
        X_2d,
        met_2d,
        metrics,
        metLab,
        ncols=ncols,
        thr=thr,
        cbor="horizontal",
        fs=fs,
        fig=fig,
        axs=axs,
        double=0,
    )
    # axs[-1].annotate('c',(0.4,0.05),xycoords='figure fraction',fontsize=fs+25)

    # Project point onto plane spanned by next two principal components
    comp = [2, 3]
    pax = pca.components_[comp, :]
    proj = np.dot(
        np.dot(pax, ndata.transpose()).transpose(), pax
    )  # No need to normalise, projection, as pca components are already orthonormal
    diff = ndata - proj
    dist = np.sqrt(np.sum(diff ** 2, axis=1).astype(np.float64))
    weights = np.ones_like(dist) / float(len(dist))
    X_2d = X_pca[dist < thr2d, :][:, comp]
    img_2d = imgarr[dist < thr2d]
    met_2d = ndDfMet[dist < thr2d]

    grid, grids, pltLab = getGrids(X_2d, met_2d, metrics, metLab, nPts, thr=thr)
    nrows = int(np.ceil(len(grids) / ncols))

    ax2 = fig.add_subplot(gs[:ncols, ncols:])
    ax2 = plotEmbedding(
        X_pca[:, comp],
        imgarr,
        filterOutliers=True,
        zoom=zoom,
        distMin=distMin,
        ax=ax2,
        offs=offs,
    )

    # Plot interpretable directions as arrows
    metricsArrow = ["woi3", "cthVar"]
    metLabArrow = ["Directional alignment", "Cloud top height variance"]
    metPlot = met_2d[metricsArrow]
    clrs = cm.get_cmap(plt.get_cmap("cubehelix"))(
        np.linspace(0.05, 0.95, len(metricsArrow))
    )[:, :3]

    ax2 = plotArrow(
        X_2d,
        metPlot,
        metricsArrow,
        ax2,
        metLabArrow,
        leg=True,
        lw=lw,
        fac=fac,
        thr=thr,
        clrs=clrs,
        fs=fs,
    )

    ax2.set_xlabel("Principal component " + str(comp[0] + 1), fontsize=fs)
    ax2.set_ylabel("Principal component " + str(comp[1] + 1), fontsize=fs)
    # ax2.annotate('b',(0.9,0.41),xycoords='figure fraction',fontsize=fs+25)
    axs1 = []
    for i in range(nrows):
        for j in range(ncols):
            axs1.append(fig.add_subplot(gs[ncols + i, ncols + j]))
    axs1 = plotMetricSurf(
        X_2d,
        met_2d,
        metrics,
        metLab,
        ncols=ncols,
        thr=thr,
        cbor="horizontal",
        fs=fs,
        fig=fig,
        axs=axs1,
        double=1,
        fac=2.3,
        lw=0.125,
    )
    # axs1[-1].annotate('d',(0.9,0.),xycoords='figure fraction',fontsize=fs+25)

    axl = fig.add_axes([0.49675, 0.0, 0.02, 0.31])
    axl.axvline(0.5, color="black", lw=0.75)
    axl.axis("off")
    plt.tight_layout()
    plt.savefig(savePath + "/comb1234.png", dpi=200, bbox_inches="tight")
    plt.show()
