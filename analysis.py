#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:24:33 2020

@author: martinjanssens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/cloudmetrics')
from Postprocess import analysis, utils
from sklearn.decomposition import PCA

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

metPath     = '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/LES/FJRICO/Metrics'
savePath    = '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/LES/FJRICO/Plots'

# Subset of metrics to be analysed
# netVarDeg can be included upon request (see Metric Computation section above)
metricsPP = [
             'cf',        
             'cwp',       
             'lMax',      
             'periSum',
             'cth',
             # 'sizeExp',
             'lMean',
             'specLMom',
              'cop',
             'scai',
             'nClouds',
              'rdfMax',
             # 'netVarDeg',
             'iOrgPoiss',
             'fracDim',
              # 'iOrg',
             'os',
             'twpVar',
             'cthVar',
             'cwpVarCl',
              'woi3',
            ]

metLab    = [
             'Cloud fraction', 
             'Cloud water', 
             'Max length',
             'Perimeter',
            r'$\overline{CTH}$',
             # 'Size exponent',
             'Mean length', 
             'Spec. length', 
              'COP',
            r'SCAI',
             'Cloud number',
              'Max RDF',
             # 'Degree var',
            r'$I_{org}$',
             'Fractal dim.', 
            # r'$I_{org}^*$',
             'Clear sky',
             'CWP var ratio',
            r'St(CTH)',
            r'St(CWP)',
            r'$WOI_3$', 
            ]


#%% Load data

# Load, order and standardise data
labs,loadPaths = findDirs(metPath)

for d in range(len(labs)):
    dfMetricsd, datad, imgArrd = analysis.loadMetrics(loadPaths[d], 
                                                      metricsPP,
                                                      sort=True, 
                                                      standardise=False, 
                                                      return_data=True,
                                                      return_images=True)
    indicd = np.ones(datad.shape[0])*(d+1)
    
    if d == 0:
        dfMetrics = dfMetricsd; data = datad; imgArr = imgArrd; indic = indicd
    else:
        
        dfMetrics = pd.concat([dfMetrics,dfMetricsd])
        data      = np.concatenate((data,datad),axis=0)
        imgArr    = np.concatenate((imgArr,imgArrd),axis=0)
        indic     = np.concatenate((indic,indicd),axis=0)

dfMetrics = utils.stand(dfMetrics)
data = utils.stand(data)

simTime = dfMetrics.index.to_numpy().astype('float64')
# simTime -= simTime[0]

# Filter low=time values
iTLow     = np.where(simTime/3600>5)[0]
dfMetrics = dfMetrics.iloc[iTLow,:]
data      = data[iTLow,:]
imgArr    = imgArr[iTLow,:,:]
indic     = indic[iTLow]
simTime   = simTime[iTLow]


#%% Make plots

# Correlation matrix 
# analysis.correlate(data, metricsPP, metLab, savePath)

# Show how metrics order scenes
# analysis.plotSortedScenes(data, imgArr, metLab, savePath)

# Compute PCA
pca = PCA()
# Regular
xPca = pca.fit_transform(data) 
xPca[:,1] = -xPca[:,1]
# Using sat data
# xPcaSat = pca.fit_transform(dataSat)
# xPca    = pca.transform(data)

# Relate metrics to PCs 
# analysis.relateMetricPCA(pca, xPca, metricsPP, metLab, savePath)

# Plot PCA distribution
# analysis.pcaDistribution(pca, xPca, savePath)

# Regime analysis 
# analysis.regimeAnalysis(xPca, imgArr, savePath)

# PCA surfaces
# This function has exceptionally high memory requirement: If it fails, try 
# again in a 'clean' console/terminal without plotting anything else.
# analysis.plotPCASurfs(data, imgArr, dfMetrics, metricsPP, metLab, pca, xPca,
                      # savePath,thr=0.01,rot2rad=0)

# More efficient implementation (only PC 1-2)
fig=plt.figure(figsize=(15,15)); ax=plt.gca()
ax = utils.plotEmbedding(xPca[:,0:2], imgArr, filterOutliers=True, 
                         zoom=0.25, distMin=1e-3, ax=ax)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
plt.savefig(savePath+'/embedding.png',dpi=300,bbox_inches='tight')
# ax.scatter(xPca[:,0],xPca[:,1],c=simTime,s=5)
axs = utils.plotMetricSurf(xPca[:,0:2], dfMetrics, metricsPP, metLab,thr=0)
plt.savefig(savePath+'/interpolation.png',dpi=300,bbox_inches='tight')


# Time evolution (PC 1-2)
fig=plt.figure(figsize=(6,6)); ax = plt.gca()
sc = ax.scatter(xPca[:,0],xPca[:,1],c=simTime/3600,cmap='Blues',s=50)
cbax = fig.add_axes([0.925, 0.12, 0.03, 0.765])
cb = fig.colorbar(sc,cax=cbax)
cb.ax.set_ylabel('Time [hr]',fontsize=10)
ax.set_xlabel('Principal Component 1',fontsize=10)
ax.set_ylabel('Principal Component 2',fontsize=10)
plt.savefig(savePath+'/timeevo.png',dpi=300,bbox_inches='tight')
plt.show()

# PC 3-4
fig=plt.figure(figsize=(15,15)); ax=plt.gca()
ax = utils.plotEmbedding(xPca[:,2:4], imgArr, filterOutliers=True, 
                         zoom=0.25, distMin=1e-3, ax=ax)
ax.set_xlabel('PC 3')
ax.set_ylabel('PC 4')

axs = utils.plotMetricSurf(xPca[:,2:4], dfMetrics, metricsPP, metLab,thr=0)


# NOTES OF PROBLEMS
# - Object-based metrics don't (but should) account for periodic BCs.
# - The fields are likely too small for some statistics-based metrics, since
#   there is lots of variance between subsequent time steps


#%% Classification by simulation
colors = ['black','midnightblue','lavender','plum','palevioletred','crimson','maroon','peachpuff','peru',
          'saddlebrown','gold','darkorange','darkseagreen', 'teal', 'steelblue', 'slategrey']
# models = ['MicroHH','SAM_CRM','MESONH','ICON_LEM','DAM','SCALE','UKMO_RA1_T','WRF_CRM']
figure=plt.figure(figsize=(6,6)); ax=plt.gca()
for i in range(len(colors)):
    inds = np.where(indic==i+1)[0]
    ax.scatter(xPca[inds,0],xPca[inds,1],c=colors[i],s=1)#,label=models[i])
ax.set_xlim((-5,7))
ax.set_ylim((-5,6))
ax.set_xlabel('Principal Component 1',fontsize=10)
ax.set_ylabel('Principal Component 2',fontsize=10)
ax.legend(markerscale=3,loc='upper left',bbox_to_anchor=(-0.01,-0.3,1,0.2),ncol=4)
plt.savefig(savePath+'/models.png',dpi=300,bbox_inches='tight')
plt.show()

#%% Classification by high/low
sh = len(simTime)/16
thls     = np.repeat(np.array([298,298,298,298,298,298,298,298,300,300,300,300,300,300,300,300]), sh) # 0 - 298; 1 - 300
dqt_high = np.repeat(np.array([0,0,0,0,0.0025,0.0025,0.0025,0.0025,0,0,0,0,0.0025,0.0025,0.0025,0.0025]), sh) # 0 - 0  ; 1 - 0.0025
windhigh = np.repeat(np.array([0,0,-9,-9,0,0,-9,-9,0,0,-9,-9,0,0,-9,-9]), sh) # 0 - -9 ; 1 - 0
windlow  = np.repeat(np.array([0,-9,0,-9,0,-9,0,-9,0,-9,0,-9,0,-9,0,-9]), sh) # 0 - -9 ; 1 - 0

stack = np.vstack((thls,dqt_high,windhigh,windlow))

labs = ['SST',r'$\Delta q_t$, $z>3260$m',r'$U$, z=4000m', r'$U$, z=0m']

for i in range(4):
    fig=plt.figure(figsize=(6,6)); ax = plt.gca()
    sc = ax.scatter(xPca[:,0],xPca[:,1],c=stack[i,:],cmap='viridis',s=50)
    cbax = fig.add_axes([0.925, 0.12, 0.03, 0.765])
    cb = fig.colorbar(sc,cax=cbax)
    cb.ax.set_ylabel(labs[i],fontsize=10)
    ax.set_xlabel('Principal Component 1',fontsize=10)
    ax.set_ylabel('Principal Component 2',fontsize=10)
    plt.savefig(savePath+'/forcing'+str(i)+'.png',dpi=300,bbox_inches='tight')
    plt.show()


