#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from skimage.transform import rescale
from scipy.spatial import cKDTree

def findFiles(path):
    '''
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
    '''
    
    # Find all files that contain scenes
    files  = []; dates = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if not isinstance(file, str):
                file = file.decode('utf-8')
            if '.h5' in file and 'Metrics.h5' not in file:
                fname = os.path.join(r, file)
                dates.append(file.split('.')[0])
                files.append(fname)
    
    files = np.sort(files); dates = np.sort(dates)
    return files, dates


def getField(file, fieldName, resFac=1, binary=False):
    '''
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

    '''
    df = pd.read_hdf(file)
    cm = df[fieldName].values[0].copy()
    if binary:
        cm = cm.astype(int)
    cm = rescale(cm,resFac,preserve_range=True,anti_aliasing=False)
    if binary:
            cm[cm<0.5] = 0; cm[cm>=0.5] = 1
    return cm

def rSquared(x, y, coeffs):

    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
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
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def createCircularMask(h, w):

    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def anyInList(refLst,checkLst):
    inList = False
    for i in range(len(checkLst)):
        if checkLst[i] in refLst:
            inList = True
    return inList

def uniqueAppend(metrics,appList):
    for i in range(len(appList)):
        append = True
        for j in range(len(metrics)):
            if metrics[j] == appList[i]:
                append = False
        if append:
            metrics.append(appList[i])
    return metrics        

def cKDTreeMethod(data,img):
    tree = cKDTree(data,boxsize=img.shape)
    dists = tree.query(data, 2)
    nn_dist = np.sort(dists[0][:, 1])    
    return nn_dist

def loadDfMetrics(path):
    return pd.read_hdf(path+'/Metrics.h5')


def removeScene(date,filteredPath,metricPath):

    # Remove .h5 from filtered dataset
    os.remove(filteredPath+'/'+date+'.h5')
    
    # Remove row from dfMetrics
    dfMetrics = loadDfMetrics(metricPath)
    ind       = np.where(dfMetrics.index == date)[0][0]
    dfMetrics = dfMetrics.drop(date)
    dfMetrics.to_hdf(metricPath+'/Metrics.h5', 'Metrics',
                             mode='w')
    
    # Remove image from imgArr
    imgArr = np.load(metricPath+'/Images.npy')
    imgArr = np.delete(imgArr, ind, axis=0)
    np.save(metricPath+'/Images.npy',imgArr)
    
