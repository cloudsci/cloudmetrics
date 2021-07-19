#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from skimage.transform import rescale
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree


def findFiles(path):
    """
    Find files that contain data fields.

    Parameters
    ----------
    path : TYPE
        Path to folder where data fields are stored in .h5 as the columns of
        pandas DataFrames. The names of these files follow the convention
        'yyyy-mm-dd-s-n.h5', where s is the satellite identifier (a - Aqua,
        t - Terra) and n is the nth scene selected that day.

    Returns
    -------
    files : numpy array
        Array of strings of absolute paths to the data files.
    dates : numpy array
        Array of dates (plus satellite and number extensions) from which the
        scenes derive.
    """

    # Find all files that contain scenes
    files = []
    dates = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if not isinstance(file, str):
                file = file.decode("utf-8")
            if ".h5" in file and "Metrics.h5" not in file:
                fname = os.path.join(r, file)
                dates.append(file.split(".")[0])
                files.append(fname)

    files = np.sort(files)
    dates = np.sort(dates)
    return files, dates


def getField(file, fieldName, resFac=1, binary=False):
    """
    Load a datafield from a file

    Parameters
    ----------
    file : string
        Absolute path to file containing a pandas dataframe.
    fieldName : string
        Name of the column of the loaded dataframe that contains the field.
    resFac : float, optional
        Resolution scaling of the field, for sensitivity studies. The default
        is 1.
    binary : bool, optional
        Whether the field is a binary field (e.g. cloud mask) or not. The
        default is False.

    Returns
    -------
    field : numpy array of shape (npx*resFac, npx*resFac)
        DESCRIPTION.

    """
    df = pd.read_hdf(file)
    cm = df[fieldName].values[0].copy()
    if binary:
        cm = cm.astype(int)
    cm = rescale(cm, resFac, preserve_range=True, anti_aliasing=False)
    if binary:
        cm[cm < 0.5] = 0
        cm[cm >= 0.5] = 1
    return cm


def rSquared(x, y, coeffs):

    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    return ssreg / sstot


def blockShaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (
        arr.reshape(h // nrows, nrows, -1, ncols)
        .swapaxes(1, 2)
        .reshape(-1, nrows, ncols)
    )


def createCircularMask(h, w):

    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def anyInList(refLst, checkLst):
    inList = False
    for i in range(len(checkLst)):
        if checkLst[i] in refLst:
            inList = True
    return inList


def uniqueAppend(metrics, appList):
    for i in range(len(appList)):
        append = True
        for j in range(len(metrics)):
            if metrics[j] == appList[i]:
                append = False
        if append:
            metrics.append(appList[i])
    return metrics


def cKDTreeMethod(data, size=None):
    # FIXME not sure if boxsize (periodic BCs) work if domain is not square
    tree = cKDTree(data, boxsize=size)
    dists = tree.query(data, 2)
    nn_dist = np.sort(dists[0][:, 1])
    return nn_dist


def loadDfMetrics(path):
    return pd.read_hdf(path + "/Metrics.h5")


def removeScene(date, filteredPath, metricPath):

    # Remove .h5 from filtered dataset
    os.remove(filteredPath + "/" + date + ".h5")

    # Remove row from dfMetrics
    dfMetrics = loadDfMetrics(metricPath)
    ind = np.where(dfMetrics.index == date)[0][0]
    dfMetrics = dfMetrics.drop(date)
    dfMetrics.to_hdf(metricPath + "/Metrics.h5", "Metrics", mode="w")

    # Remove image from imgArr
    imgArr = np.load(metricPath + "/Images.npy")
    imgArr = np.delete(imgArr, ind, axis=0)
    np.save(metricPath + "/Images.npy", imgArr)


def periodic(field, con):
    """
    Apply periodic BCs to cloud mask fields, based on implementation from
    https://github.com/MennoVeerman/periodic-COP

    Parameters
    ----------
    field : (npx,npx) numpy array
        Cloud mask field (no other cloud field accepted!).

    Returns
    -------
    Field of (2*npx,2*npx), with cloud objects that cross boundaries translated
    to coherent structures crossing the northern/eastern boundaries.

    """

    # Questions remaining:
    # - How to handle regions whose cog lies outside the original image?

    ny, nx = field.shape

    # Create array with extra cells in y and x direction to handle periodic BCs
    cld = np.zeros((2 * ny, 2 * nx))

    # set clouds mask
    cld[:ny, :nx] = field.copy()

    # Label connected regions of cloudy pixels
    cld_lbl, nlbl = label(cld, connectivity=con, return_num=True)

    # Find all clouds (labels) that cross the domain boundary in n-s direction.
    # Save the labels of the cloudy region at both the southern border
    # (clouds_to_move_north) and at the northern border (clouds_in_the_north)
    y0, y1 = 0, ny - 1
    clouds_to_move_north = []
    clouds_in_the_north = []
    for ix in range(nx):
        if (
            cld_lbl[y0, ix] > 0
            and cld_lbl[y1, ix] > 0
            and cld_lbl[y0, ix] != cld_lbl[y1, ix]
            and cld_lbl[y0, ix]
        ):
            clouds_to_move_north += [cld_lbl[y0, ix]]
            clouds_in_the_north += [cld_lbl[y1, ix]]

    # Find all clouds (labels) that cross the domain boundary in e-w direction.
    # Save the labels of the cloudy region at both the western border
    # (clouds_to_move_east) and at the western border (clouds_in_the_east)
    x0, x1 = 0, nx - 1
    clouds_to_move_east = []
    clouds_in_the_east = []
    for iy in range(ny):
        if (
            cld_lbl[iy, x0] > 0
            and cld_lbl[iy, x1] > 0
            and cld_lbl[iy, x0] != cld_lbl[iy, x1]
        ):
            clouds_to_move_east += [cld_lbl[iy, x0]]
            clouds_in_the_east += [cld_lbl[iy, x1]]

    # Move all cloud parts in the west(south) that are connected to cloud parts
    # in the east(north) towards to east(north) beyond the boundaries of the
    # original domain.
    regions = regionprops(cld_lbl)
    for cloud in np.unique(cld_lbl):
        # Loop over all identified cloud clusters
        shift_y, shift_x = 0, 0

        if cloud in clouds_to_move_north:
            # Clouds region connects to cloud region at the northern boundary
            # and will be moved north
            shift_y = ny
            if (
                clouds_in_the_north[clouds_to_move_north.index(cloud)]
                in clouds_to_move_east
            ):
                # The connected cloud in the north will be moved east:
                # diagonal crossing
                shift_x = nx

        if cloud in clouds_to_move_east:
            shift_x = nx
            if (
                clouds_in_the_east[clouds_to_move_east.index(cloud)]
                in clouds_to_move_north
            ):
                shift_y = ny

        if cloud in clouds_in_the_north:
            if (
                clouds_to_move_north[clouds_in_the_north.index(cloud)]
                in clouds_to_move_east
            ):
                # Cloud is connnected to a cloud region in the south, but that
                # cloud region will also be moved eastward
                shift_x = nx
        if cloud in clouds_in_the_east:
            if (
                clouds_to_move_east[clouds_in_the_east.index(cloud)]
                in clouds_to_move_north
            ):
                # Cloud is connnected to a cloud region in the west, but that
                # cloud region will also be moved northward
                shift_y = ny

        # Shift the clouds
        if shift_y > 0 or shift_x > 0:
            # Cloud should be shifted in east-west and/or north-south direction
            region = regions[cloud - 1]
            # Remove from current position
            cld_lbl[region.coords[:, 0], region.coords[:, 1]] = 0
            # Put in new position
            cld_lbl[region.coords[:, 0] + shift_y, region.coords[:, 1] + shift_x] = 1

    return np.where(cld_lbl > 0, 1, 0)
