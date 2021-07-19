#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:37:48 2020

@author: martinjanssens
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial import cKDTree
import scipy.stats as ss
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from scipy.stats import pearsonr


def stand(data):
    data = data.astype(np.float64)
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# #%%
# # Load data
# path      = '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/MetricPlayground/metrics'
# metrics   = ['cf','cth', 'cwp', 'Iorg', 'rdfMax','nClouds','twpVar','cwpVarCl',
#               'scai','fracDim','cthVar','beta','iorgTomp',
#                'lMean','periSum','sizeExp',
#                'lMax',
#                'eccA','woi3Cwp','COP',
#                # 'psdAzVar','orieCov','brAv',
#                'os',
#                 # 'netAWPar','netCoPar','netLCorr',
#                'netVarDeg'
#              ]
# metLab    = ['Cloud fraction', r'$\overline{CTH}$','Cloud water', r'$I_{org}$',
#               'Max RDF','Cloud number','CWP var ratio', r'St(CWP)',r'SCAI',
#               'Fractal dim.', r'St(CTH)',r'Spec. slope', r'Poisson $I_{org}$',
#                'Mean length', 'Perimeter','Size exponent',
#                'Max length',
#                'Eccentricity', r'$WOI_3$', 'COP',
#                # 'Anisotropy', 'Orientation', 'Branching',
#                'Open sky',
#                # 'Aboav-Wearie', 'a*','Lewis corr',
#                'Degree var'
#              ]
# df        = pd.read_hdf(path+'/Metrics0min.h5')
# dfMetrics = df[metrics]
# dfMetrics = dfMetrics.sort_index()
# ndDfMet   = stand(dfMetrics)
# data      = dfMetrics.to_numpy()
# ndata     = stand(data)

# pca = PCA()
# X_pca = pca.fit_transform(ndata)

# #%% Compare metrics to PCs

# # - Compute distance to space spanned by 4 PCs from a given point -> old impl.
# # - Compute orthogonality to other metrics

# # For the 2D situation:
# # 1. Take the plane spanned by the first 2 PCs -> This give a point pancake
# # 2. Add a dimension to this dataset - the metric in question. IF this metric
# #    puts the points in the plane spanned by the first 2 PCs, its main variance
# #    is in that plane. In other words, the PCs of the resulting 3D system
# #    should practically contain no variance in the 3rd PC. The following
# #    routine quantifies the fraction of the total variance of the 3D set
# #    contained in the final PC. The closer it is to zero, the

# # You can refine! You want to quantify the percentage of each metric's variance
# # that cannot be explained by the already considered PCs, NOT how much that
# # metric contributes variance wrt the PCs

# dimSpace = 4
# evrVar = np.zeros(len(metrics))
# for i in range(len(metrics)):
#     fitVar    = ndDfMet[metrics[i]].to_numpy().reshape(X_pca.shape[0],1)
#     dsTest    = np.hstack((X_pca[:,:dimSpace],fitVar))
#     pcaTest   = PCA()
#     xPcaTest  = pcaTest.fit_transform(dsTest)
#     evrVar[i] = pcaTest.explained_variance_ratio_[-1]

# ind = np.argsort(evrVar)
# metPlot = np.asarray(metrics)[ind]
# evrPlot = evrVar[ind]
# xpos = np.arange(len(metPlot))
# plt.bar(xpos,evrPlot)
# tks=plt.xticks(xpos, metPlot, rotation=90)

# #%%
# # Find perpendicular pairs
# cov = np.abs(np.cov(ndDfMet.to_numpy().T))
# ind = np.argsort(cov,axis=None)

# pairs = np.empty((len(metrics),len(metrics)),dtype=object)
# for i in range(len(metrics)):
#     for j in range(len(metrics)):
#         pairs[i,j] = metrics[i] + ', ' + metrics[j]

# pairs=np.tril(pairs)
# corrSort = pairs.ravel()[ind]
# corrSort = corrSort[corrSort !=0]

# # Interesting candidates:
# # lMax, Iorg
# # os, sizeExp
# # fracDim, cf
# # periSum, Iorg
# # os, betaa
# # os, cwp
# # woi3Cwp, cthVar

# # Now manually choose a couple of metrics
# metProp = ['nClouds','os']#,'eccA','cwpVarCl']
# dimSpace = len(metProp)

# # Test perpendicularity of these metrics through covariance matrix?
# covMat  = np.cov(ndDfMet[metProp].to_numpy().T) # Numpy version
# covMatP = ndDfMet[metProp].corr()               # Pandas alternative

# # Starting from a d-dimensional dataset, we reduced it to dimSpace dimensions
# # by a PCA.
# # Now, you want to find out if you can reconstruct the 2D space spanned by the
# # PCs with different bases, that you've chosen to be your metrics. To test that,
# # you would 1) Make the PC distribution and 2) make the metric distribution.
# # Then, (and this is what I don't know), if they encapsulate the same information,
# # you should be able to reconstruct the PC distribution from your metric
# # distribution by a linear transformation.

# ind = []
# for i in range(len(metrics)):
#     if metrics[i] in metProp:
#         ind.append(i)
# comps = pca.components_[:dimSpace, ind]
# scale = np.diag(pca.explained_variance_[:dimSpace])
# datex = ndata[:,ind]
# datpc = np.dot(datex,comps.T)
# datpcS = np.dot(datpc,scale)

# cov   = np.abs(np.corrcoef(datpcS.T,X_pca[:,:dimSpace].T)[dimSpace:,:dimSpace])

# dist = np.sqrt(np.sum((datpcS - X_pca[:,:dimSpace])**2,axis=1))
# # dist = np.ravel(X_pca[:,:dimSpace]/(X_pca[:,:dimSpace]+datpcS))

# fig,axs=plt.subplots(ncols=2,figsize=(10,5))
# axs[0] = sns.kdeplot(dist,color='black',ax=axs[0])
# axs[0].set_xlim((0,5))
# axs[0].annotate('Mean distance: %.3f' % np.mean(dist), (0.52,0.925),xycoords='axes fraction')
# axs[0].set_xlabel('Euclidian distance from PCA point')
# axs[0].set_ylabel('Kernel density estimate')
# axs[1].scatter(X_pca[:,0],X_pca[:,1],s=0.5,color='midnightblue')
# axs[1].scatter(datpcS[:,0],datpcS[:,1],s=0.5,color='darkseagreen')
# axs[1].set_xlim((-10,15))
# axs[1].set_ylim((-5,7.5))
# # axs[1] = sns.heatmap(cov, cmap='YlGnBu') # Between our transformed data and X_pca
# # axs[1].set_xlabel('Principal Component')
# # axs[1].set_yticklabels(metProp)

# # fig,axs=plt.subplots(ncols=1,figsize=(5,5))
# # axs.scatter(X_pca[:,0],X_pca[:,1],s=0.5,color='midnightblue',label='PCs')
# # axs.scatter(datpcS[:,0],datpcS[:,1],s=0.5,color='darkseagreen',label='nClouds, os')
# # axs.scatter(test[:,0],test[:,1],s=0.5,color='peachpuff',label='cf, Iorg')
# # axs.set_xlim((-10,15))
# # axs.set_ylim((-5,7.5))
# # axs.legend()


# # THESE ARE EQUAL
# # X_trunc = np.dot(X,pca.components_[:dimSpace,:].T) -> (X.shape[0],dimSpace)
# #         = X_pca[:,:dimSpace]

# # You want to compare this dataset to:
# # X_trunc =  np.dot(X[:,ind],pca.components_[:dimSpace,ind].T)

# ## WHY ARE THESE AN ORDER OF MAGNITUDE OFF?
# # Because the former has all metrics, so there you project onto a basis with
# # unit length. In the latter case, you do not, because you lack all components
# # oriented along different metric directions that have contributions.

# # The issue stems from that a normal PC transformation is np.dot(X,V.T), where
# # V is pca.components_ is the right singular matrix of X in the SVD X=USV.T
# # V is an orthonormal basis, such that the PC transformation not only projects
# # the metrics onto the correctly oriented components, but they are appropriately
# # scaled. This is no longer the case when we truncate V.


# # testMat = np.array([[comps[0,0]**2,comps[1,0]**2,0,0],
# #                     [comps[0,0]**2,0,comps[0,1]**2,0],
# #                     [0,comps[1,0]**2,0,comps[1,1]**2],
# #                     [0,0,comps[0,1]**2,comps[1,1]**2],
# #                     ])
# # facs = np.sqrt(np.linalg.solve(testMat,np.ones(4)))
# # facsr = np.reshape(facs,(2,2)).T


# #%% Dicking around with projections - not useful at the moment

# metProp = ['scai','nClouds']#,'cthVar','woi3Cwp']
# ndatRed = ndDfMet[metProp].to_numpy()

# # Project points defined by metProp onto principal components
# comp  = np.arange(len(metProp)).astype('int')
# pax = pca.components_[:len(metProp),:len(metProp)]
# proj = np.dot(np.dot(pax,ndatRed.transpose()).transpose(),pax)

# # Compute (nD) distance between actual point and projection onto plane
# diff = ndatRed - proj
# dist = np.sqrt(np.sum(diff**2,axis=1).astype(np.float64))

# fig=plt.figure(); ax = plt.gca()
# ax = sns.kdeplot(dist,ax=ax)
# ax.set_xlabel('Euclidian distance from 2D PCA space')
# ax.set_ylabel('Fraction')
# ax.set_xlim((0,5))
# plt.show()

# dMean = np.mean(dist)

# print('Metrics:      ',metProp)
# print('Mean distance: %.3f' % dMean)


#%% Sensitivity analysis - Following visual test
def computeSensitivity(X_pca, X_pca1, X_pca2, X_pca3, savePath, npts=1e3):
    dimMax = 8
    npts = int(npts)

    # dimMax-dimensional KDEs of the entire distributions of the reference (A) and
    # different (B:) distributions
    fA = ss.gaussian_kde(X_pca[:, :dimMax].T, bw_method=1)
    fB = ss.gaussian_kde(X_pca1[:, :dimMax].T, bw_method=1)
    fC = ss.gaussian_kde(X_pca2[:, :dimMax].T, bw_method=1)
    fD = ss.gaussian_kde(X_pca3[:, :dimMax].T, bw_method=1)

    # Points sampled from all distributions
    ptsA = fA.resample(npts)
    ptsB = fB.resample(npts)
    ptsC = fC.resample(npts)
    ptsD = fD.resample(npts)

    # Corresponding KDEs -> Necessary to reconstruct these to compare same points
    fA1 = ss.gaussian_kde(ptsA, bw_method=3)
    fB1 = ss.gaussian_kde(ptsB, bw_method=3)
    fC1 = ss.gaussian_kde(ptsC, bw_method=3)
    fD1 = ss.gaussian_kde(ptsD, bw_method=3)

    # KDEs at points drawn from fA
    points = fA1.resample(npts)
    fAp1 = fA1(points)
    fBp1 = fB1(points)
    fCp1 = fC1(points)
    fDp1 = fD1(points)

    # Discriminant function D
    D1 = fAp1 / (fAp1 + fBp1)
    D2 = fAp1 / (fAp1 + fCp1)
    D3 = fAp1 / (fAp1 + fDp1)

    fig = plt.figure()
    ax = plt.gca()
    ax = sns.kdeplot(D1, ax=ax, color="midnightblue", label="Half resolution")
    ax = sns.kdeplot(D2, ax=ax, color="darkseagreen", label="8 connectivity")
    ax = sns.kdeplot(D3, ax=ax, color="peachpuff", label="No minimum cloud size")
    ax.plot([0.5, 0.5], [0, 12], "k--")
    ax.set_ylabel("Kernel density estimate")
    ax.set_xlabel(r"$\frac{f_A}{f_A + f_B}$")
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 12.05))
    ax.legend(loc="upper left", frameon=False)
    plt.savefig(savePath + "/senskde.pdf", bbox_inches="tight")
