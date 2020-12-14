#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.insert(1, '/projects/0/einf170/janssonf/botany/cloudmetrics')
from Metrics import createDataFrame, computeMetrics, utils

def findDirs(loadPath):
    labs = []; dirs  = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(loadPath):
        for direc in d:
            dname = os.path.join(r, direc)
            labs.append(direc)
            dirs.append(dname)
    labs = np.sort(labs); dirs = np.sort(dirs)
    return labs, dirs

def makeNewDirs(dirs,path):
    
    # Only intended for a list of directories where none have yet been made
    mkDlDirs = [d for d in dirs if os.path.isdir(path+'/'+str(d))]
    
    if not mkDlDirs:
        for d in dirs:            
            os.makedirs(path+'/'+d)

loadRoot = '/projects/0/einf170/janssonf/botany/botany-1-94hyqf8e/runs'
saveRoot = '/projects/0/einf170/janssonf/botany/botany-1-94hyqf8e/runs'

metrics = [
           'cf',        # Cloud fraction
           'cwp',       # Total cloud water path
           'lMax',      # Max length scale of scene's largest object
           'periSum',   # Total perimeter of all scene's cloud objects
           'cth',       # Mean cloud top height
           'sizeExp',   # Exponent of cloud size distribution (power law fit)
           'lMean',     # Mean length of cloud object in scene
           'specLMom',  # Spectral length scale of cloud water
           'cop',       # Convective Organisation Potential White et al. (2018)
           'scai',      # Simple Convective Aggregation Index Tobin et al. (2012)
           'nClouds',   # Number of clouds in scene
           'rdfMax',    # Max of the radial distribution function of objects
           'netVarDeg', # Degree variance of nearest-neighbour network of objects
           'iOrgPoiss', # Organisation index as used in Tompkins & Semie (2017)
           'fracDim',   # Minkowski-Bouligand dimension
           'iOrg',      # Organisation index as modified by Benner & Curry (1998)
           'os',        # Contiguous open sky area estimate (Antonissen, 2019)
           'twpVar',    # Variance in CWP anomaly on scales larger than 16 km (Bretherton & Blossey, 2017)
           'cthVar',    # Variance in cloud top height
           'cwpVarCl',  # Variance in cloud water path
           'woi3',      # Wavelet-based organisation index of orientation (Brune et al., 2018)
           'orie',      # Image raw moment covariance-based orientation metric
          ]

fields = {'cm'  : 'CloudMask',
          'im'  : 'image',
          'cth' : 'CloudTopHeight',
          'cwp' : 'CloudWaterPath'}
mpar = {
        'loadPath' : '',    # Set in loop
        'savePath' : '',    # Set in loop
        'save'     : True, 
        'saveExt'  : '',    # Extension to filename to save in
        'resFac'   : 1,     # Resolution factor (e.g. 0.5)
        'plot'     : False, # Plot with details on each metric computation
        'con'      : 1,     # Connectivity for segmentation (1:4 seg, 2:8 seg)
        'areaMin'  : 1,     # Minimum cloud size considered for object metrics
        'fMin'     : 0,     # First scene to load
        'fMax'     : None,  # Last scene to load. If None, is last scene in set
        'fields'   : fields # Field naming convention
        }

labs, loadPaths = findDirs(loadRoot)
#makeNewDirs(labs,saveRoot)
_, savePaths = findDirs(saveRoot)

#%% Make new image and metric dataframes
#print('Creating image and metric dataframes...')
#for d in range(len(labs)):
#    print('Run: ',labs[d])
#    mpar['loadPath'] = loadPaths[d]
#    mpar['savePath'] = savePaths[d]
#
#    createDataFrame.createMetricDF(loadPaths[d], metrics, savePaths[d])
#    createDataFrame.createImageArr(loadPaths[d], savePaths[d], imageTag='image')

#%% Compute metrics

metrics = [
           'cf',        # Cloud fraction
           'cwp',	# Total cloud water path
           'lMax',	# Max length scale of scene's largest object
           'periSum',   # Total perimeter of all scene's cloud objects
           'cth',	# Mean cloud top height
           # 'sizeExp',   # Exponent of cloud size distribution (power law fit)
           'lMean',     # Mean length of cloud object in scene
           # 'specLMom',  # Spectral length scale of cloud water
           'cop',	# Convective Organisation Potential White et al. (2018)
           'scai',	# Simple Convective Aggregation Index Tobin et al. (2012)
           'nClouds',   # Number of clouds in scene
           'rdfMax',    # Max of the radial distribution function of objects
           # 'netVarDeg', # Degree variance of nearest-neighbour network of objects
           'iOrgPoiss', # Organisation index as used in Tompkins & Semie (2017)
           'fracDim',   # Minkowski-Bouligand dimension
           'iOrg',	# Organisation index as modified by Benner & Curry (1998)
           'os',        # Contiguous open sky area estimate (Antonissen, 2019)
           'twpVar',    # Variance in CWP anomaly on scales larger than 16 km (Bretherton & Blossey, 2017)
           'cthVar',    # Variance in cloud top height
           'cwpVarCl',  # Variance in cloud water path
           # 'woi3',	  # Wavelet-based organisation index of orientation (Brune et al., 2018)
           'orie',	# Image raw moment covariance-based orientation metric
          ]

#for d in range(len(labs)):
#    mpar['loadPath'] = loadPaths[d]
#    mpar['savePath'] = savePaths[d]
#    computeMetrics.computeMetrics(metrics,mpar)

#%% Fourier metrics (with separately set parameters)
from Metrics.fourier import FourierMetrics
for d in range(len(labs)):
    mpar['loadPath'] = loadPaths[d]
    mpar['savePath'] = savePaths[d]

    four = FourierMetrics(mpar)
    four.window = 'None'
    four.detrend = False
    four.dx = 100 #m
    four.field = 'image'

    four.compute()

#%% Wavelet metrics (with separately set parameters)

# Set woi.pad=32 for 192x192 scenes
#from Metrics.woi import WOI
#for d in range(len(labs)):
#    mpar['loadPath'] = loadPaths[d]
#    mpar['savePath'] = savePaths[d]
#
#    woi = WOI(mpar)
#    woi.pad = 32
#
#    woi.compute()    
