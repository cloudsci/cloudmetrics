#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path
from matplotlib import offsetbox
from matplotlib.collections import LineCollection
import scipy.stats as ss
from scipy.interpolate import griddata


def stand(data):
    data = data.astype(np.float64)
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def plotCorr(
    data,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    maxSquares=False,
    flip=False,
    size="data",
    pca=None,
    absolute=True,
    cbLab=None,
):
    ran = np.arange(data.shape[1])
    grid = np.meshgrid(ran, ran, indexing="ij")

    s0 = 2.5  # Half page
    scl = 1.8  # Scale factor

    if flip:
        data = np.flipud(data)
        if xticklabels is not None:
            xticklabels = xticklabels[::-1]

    if absolute:
        data = np.abs(data)
        cmap = "cubehelix_r"
        cmin = 0
        abslab = "Absolute "
    else:
        cmap = "Spectral_r"
        cmin = -1
        abslab = ""

    if size == "data":
        ss = np.abs(data) * 10 * scl
        ssm = np.abs(np.max(data, axis=1)) * 20 * scl
    elif size == "pca" and pca is not None:
        evr = pca.explained_variance_ratio_ * 25 * scl
        ss = np.repeat(evr, data.shape[1]).reshape(data.shape)
        ssm = pca.explained_variance_ratio_.reshape(1, data.shape[1]) * 40 * scl
    else:
        print("No valid size of markers specified")
        return

    indMax = np.argmax(np.abs(data), axis=1)

    fig = plt.figure(figsize=(5.5, s0 * scl))
    ax = plt.gca()
    sc = ax.scatter(grid[0], grid[1], s=ss, c=data, marker="s", cmap=cmap)
    sc.set_clim(cmin, 1)
    if maxSquares:
        ax.scatter(ran, indMax, s=ssm, marker="s", facecolors="none", edgecolors="k")
    ax.set_xticks(ran)
    ax.set_yticks(ran)
    ax.tick_params(axis="x", which="major", labelsize=4 * scl)
    ax.tick_params(axis="y", which="major", labelsize=4 * scl)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation="vertical")
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=4 * scl)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=4 * scl)
    if pca is not None:
        ticks = np.linspace(0, len(ran) + 1, len(ran) + 1)  # Need to fool it
        y1ticklab = np.round(pca.explained_variance_ratio_, 2)
        y1ticklab = np.append(y1ticklab, " ")
        ytick = 23.4
        fac = 0.08
        ax1 = ax.twiny()
        ax1.set_xlim(ax.get_xlim())
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(y1ticklab, fontsize=4 * scl, rotation="vertical")  # 10,
        for i in range(len(ticks) - 1):
            w = fac * np.sqrt(evr[i])
            h = w
            ax1.add_patch(
                patches.Rectangle(
                    (ticks[i] - w / 2, ytick),
                    w,
                    h,
                    color="gray",
                    fill=True,
                    clip_on=False,
                )
            )
        ax1.set_xlabel("Fraction of total variance", labelpad=5 * scl, fontsize=4 * scl)
    cbax = fig.add_axes([1.0, 0.11, 0.03, 0.86])
    cb = fig.colorbar(sc, cax=cbax)
    cb.ax.tick_params(labelsize=4 * scl)
    cb.ax.set_ylabel(cbLab, rotation=270, labelpad=5 * scl, fontsize=4 * scl)
    plt.tight_layout()


def rainbowArrow(ax, start, end, cmap="viridis", n=50, lw=3, headFac=350):
    lwpt = lw * headFac
    colmap = plt.get_cmap(cmap, n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1] + lw, n)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=colmap, linewidth=lwpt, clip_on=False)
    lc.set_array(np.linspace(0, 1, n))
    ax.add_collection(lc)
    lc.set_edgecolor("face")
    # Arrow head: Triangle
    tricoords = [(0, -0.4), (0.5, 0), (0, 0.4), (0, -0.4)]
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(
        np.reshape(end[0], -1),
        np.reshape(end[1] + lw, -1),
        c=np.reshape(1, -1),
        s=(2 * lwpt) ** 2,
        marker=tri,
        cmap=cmap,
        vmin=0,
        clip_on=False,
    )
    ax.autoscale_view()


def plot2dClusters(
    X,
    cl,
    co,
    xlab=None,
    ylab=None,
    filterOutliers=False,
    ax=None,
    sMarker=0.5,
    clrs=None,
):

    if clrs is None:
        clrs = getClrs()

    if filterOutliers:
        X, orows = rejectOutliers(X)
        labs = np.delete(cl.labels_, orows, axis=0)

    if ax == None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()

    for i in range(len(co)):
        scat = X[labs == co[i]]
        if co[i] == -1:
            ax.scatter(scat[:, 0], scat[:, 1], s=sMarker, c="gray", alpha=0.5)
        else:
            ax.scatter(scat[:, 0], scat[:, 1], s=sMarker, facecolor=clrs[i], alpha=1)
    ax.set_xticks([])
    ax.set_xlabel(xlab)
    ax.set_yticks([])
    ax.set_ylabel(ylab)
    if ax != None:
        return ax


def getClrs():
    return [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "navy",
        "gold",
        "darkolivegreen",
        "maroon",
        "mediumvioletred",
        "peru",
        "lightpink",
        "black",
        "darkgreen",
        "lightslategrey",
        "darkblue",
        "orangered",
        "darkseagreen",
        "coral",
    ]


def rejectOutliers(data, m=3.5):
    # FIXME cannot handle any other axis
    data = data.astype(np.float64)
    mn = np.mean(data, axis=0)  # mean along each dimension -> shape: (1,nDims)
    dist = np.sqrt(
        (data - mn) ** 2
    )  # how far from that mean is each point in each of its directions? -> shape: (nSamples,nDims)?
    std = np.std(data, axis=0)  # std along each dimension -> shape: (1,nDims)
    orows = np.where(dist > m * std)[
        0
    ]  # find rows that have an element along a dimension that is more than m stds from the mean of that dimension
    dataF = np.delete(data, orows, axis=0)  # remove those rows from the data
    return dataF, orows


def plotClusteredImages(cl, co, data, imgarr, rand=False, cnt=10, clrs=None, axs=None):

    if clrs is None:
        clrs = getClrs()

    if cnt == None:
        catinds = np.where(cl.labels_ > -1)
        _, cnt = ss.mode(cl.labels_[catinds])
        cnt = cnt[0]

    nc = len(co)

    if axs is None:
        fig, axs = plt.subplots(nc, cnt, figsize=(cnt, nc))

    for i in range(nc):
        imgs = imgarr[cl.labels_ == co[i]]
        if not rand:
            mets = data[cl.labels_ == co[i]]
            mn = np.mean(mets, axis=0)
            dist = np.sum((mets - mn) ** 2, axis=1)
            ind = np.argsort(dist)
            imgs = imgs[ind]
        imgs = imgs[:cnt]

        axs[i, 0].scatter(0, 0, c=clrs[i], s=2000)
        axs[i, 0].annotate(i + 1, (-0.3, 0.4), xycoords="axes fraction", fontsize=20)
        for j in range(len(imgs)):
            axs[i, j].imshow(imgs[j], "gray")
            axs[i, j].set_axis_off()
        for j in range(cnt - len(imgs)):
            axs[i, j + len(imgs)].set_axis_off()


def getGrids(X_2d, met_2d, metrics, metLab, nPts=20, thr=0):
    minxy = np.min(X_2d, axis=0)
    maxxy = np.max(X_2d, axis=0)

    ranx = np.linspace(minxy[0], maxxy[0], nPts)
    rany = np.linspace(minxy[1], maxxy[1], nPts)

    grid = np.meshgrid(ranx, rany)

    grids = []
    pltLabs = []
    for i in range(len(metrics)):
        var = metrics[i]
        ind = np.argsort(met_2d[var])
        Xpl = X_2d[ind, :]
        met = met_2d[var].sort_values()

        metGrid = griddata(Xpl, met, (grid[0], grid[1]), method="linear")

        # Compute gradient in this plane
        dMdy, dMdx = np.gradient(metGrid)
        dMdxm = np.nanmean(dMdx)
        dMdym = np.nanmean(dMdy)
        mag = np.sqrt(dMdxm ** 2 + dMdym ** 2)
        print("Metric:", metLab[i], "Mag: ", mag)

        if mag > thr:
            grids.append(metGrid)
            pltLabs.append(metLab[i])

    return grid, grids, pltLabs


def plotEmbedding(
    X,
    imgarr,
    filterOutliers=False,
    title=None,
    zoom=0.075,
    distMin=1e-3,
    pltArrow=False,
    ax=None,
    offs=0,
):
    # Plot embedding space

    if filterOutliers:
        X, rows = rejectOutliers(X)
        imgarr = np.delete(imgarr, rows, axis=0)

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    if ax is None:
        plt.figure(figsize=(15, 15))
        ax = plt.subplot(111)

    shown_images = np.array([[1.0, 1.0]])  # just something big
    plti = []
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < distMin:
            continue  # don't show points that are too close
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(imgarr[i, :, :], cmap=plt.cm.gray, zoom=zoom),
            X[i],
            pad=0.3,
        )
        ax.add_artist(imagebox)
        plti.append(i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((np.min(X[plti, 0]) - offs, np.max(X[plti, 0] + offs)))
    ax.set_ylim((np.min(X[plti, 1]) - offs, np.max(X[plti, 1] + offs)))
    if title is not None:
        plt.title(title)
    return ax


def plotArrow(
    X_2d,
    met_2d,
    metrics,
    ax,
    lab,
    nPts=20,
    lw=0.001,
    fac=1.2,
    leg=False,
    thr=0.15,
    clrs=None,
    fs=21,
):
    # Find direction of maximum change in the plane for each input metric
    if clrs is None:
        clrs = getClrs()

    nPts = 20
    minxy = np.min(X_2d, axis=0)
    maxxy = np.max(X_2d, axis=0)

    ranx = np.linspace(minxy[0], maxxy[0], nPts)
    rany = np.linspace(minxy[1], maxxy[1], nPts)

    grid = np.meshgrid(ranx, rany)

    arrows = []
    labs = []
    for i in range(len(metrics)):
        # fig,ax = plt.subplots()
        var = metrics[i]
        ind = np.argsort(met_2d[var])
        Xpl = X_2d[ind, :]
        met = met_2d[var].sort_values()

        metGrid = griddata(Xpl, met, (grid[0], grid[1]), method="linear")
        # ax.contourf(grid[0],grid[1],metGrid,100,cmap='Spectral_r')
        dMdy, dMdx = np.gradient(metGrid)

        dMdxm = np.nanmean(dMdx)
        dMdym = np.nanmean(dMdy)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        centx = (xmin + xmax) / 2
        centy = (ymin + ymax) / 2
        mag = np.sqrt(dMdxm ** 2 + dMdym ** 2)

        print("Metric:", lab[i], "Mag: ", mag)

        if mag > thr:
            arrow = ax.arrow(
                centx - fac ** 2 * dMdxm,
                centy - fac ** 2 * dMdym,
                2 * fac ** 2 * dMdxm,
                2 * fac ** 2 * dMdym,
                width=lw,
                color=clrs[i],
                zorder=3,
                label=lab[i],
            )
            arrows.append(arrow)
            labs.append(lab[i])

    if leg:
        ax.legend(arrows, labs, fontsize=fs, loc="lower left")
    return ax


def plotMetricSurf(
    X_2d,
    met_2d,
    metrics,
    metLab,
    ncols=4,
    nPts=20,
    thr=0,
    cbor="vertical",
    fs=14,
    fig=None,
    axs=None,
    double=False,
    fac=3,
    lw=0.2,
    surf=True,
):

    grid, grids, pltLabs = getGrids(X_2d, met_2d, metrics, metLab, nPts, thr)
    nrows = int(np.ceil(len(grids) / ncols))

    if axs == None:
        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            sharex=True,
            sharey=True,
            figsize=(2 * ncols, 1.5 * nrows),
        )
    axs = np.ravel(axs)
    if len(grids) == 0:
        return axs

    for i in range(len(grids)):
        ax = axs[i]
        if surf == True:
            sc = ax.contourf(grid[0], grid[1], grids[i], 100, cmap="Spectral_r")
        else:
            sc = ax.scatter(
                X_2d[:, 0], X_2d[:, 1], c=met_2d.iloc[:, i], cmap="Spectral_r"
            )
        # sc.set_clim(-2.5,2.5)
        ax.set_title(pltLabs[i], fontsize=fs)
        ax.set_xticks([])
        ax.set_yticks([])

        dMdy, dMdx = np.gradient(grids[i])
        dMdxm = np.nanmean(dMdx)
        dMdym = np.nanmean(dMdy)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        centx = (xmin + xmax) / 2
        centy = (ymin + ymax) / 2

        ax.arrow(
            centx - fac ** 2 * dMdxm,
            centy - fac ** 2 * dMdym,
            2 * fac ** 2 * dMdxm,
            2 * fac ** 2 * dMdym,
            width=lw,
            color="k",
            zorder=3,
        )

    for i in range(len(grids), ncols * nrows):
        fig.delaxes(axs[i])

    if cbor == "horizontal":
        if double == 0:
            cbax = fig.add_axes([0.1, -0.025, 0.3, 0.0125])
        elif double == 1:
            cbax = fig.add_axes([0.6, -0.025, 0.3, 0.0125])
        else:
            cbax = fig.add_axes([0.2, -0.025, 0.6, 0.0125])
        cb = plt.colorbar(sc, cax=cbax, orientation="horizontal")
        # boundaries=np.linspace(-2.5,2.5,100))
        # cb.set_clim(-2.5,2.5)
        cb.ax.set_xlabel("Standardised metric value", labelpad=10, size=fs)

    else:
        cbax = fig.add_axes([1.0, 0.05, 0.025, 0.9])
        cb = fig.colorbar(sc, cax=cbax)
        cb.ax.set_ylabel("Standardised metric value", rotation=270, labelpad=10)
    cb.ax.tick_params(labelsize=fs)

    if surf:
        for c in sc.collections:
            c.set_edgecolor("face")

    return axs


def rotFlip(data, angle, flipAx=None, rotAxes=None):
    if flipAx is not None:
        data[:, flipAx] = -data[:, flipAx]
    if rotAxes is not None:
        rotMat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        data[:, rotAxes] = np.dot(data[:, rotAxes], rotMat)
    return data
