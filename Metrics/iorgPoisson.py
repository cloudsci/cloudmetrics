#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from .utils import findFiles, getField, cKDTreeMethod

# TODO: Account for periodic BCs

class IOrgPoisson():
    '''
    Class for computing the Organisation Index Iorg (Weger et al. 1992) from a 
    cloud mask, using the distribution of a Poisson point process as reference. 
    
    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. One of 
                      these columns can be filled by 'iOrgPoiss'.
           save     : Boolean to specify whether to store the variables in
                      savePath/Metrics.h5
           resFac   : Resolution factor (e.g. 0.5), to coarse-grain the field.
           plot     : Boolean to specify whether to make plot with details on
                      this metric for each scene.
           con      : Connectivitiy for segmentation (1 - 4 seg, 2 - 8 seg)
           areaMin  : Minimum cloud size considered in computing metric
           fMin     : First scene to load
           fMax     : Last scene to load. If None, is last scene in set.
           fields   : Naming convention for fields, used to set the internal
                      field to be used to compute each metric. Must be of the 
                      form:
                           {'cm'  : CloudMaskName, 
                            'im'  : imageName, 
                            'cth' : CloudTopHeightName,
                            'cwp' : CloudWaterPathName}
                     
    '''
    def __init__(self, mpar=None):
        # Metric-specific parameters
        self.field    = 'Cloud_Mask_1km'
        self.plot     = False
        self.con      = 1
        self.areaMin  = 4
        
        # General parameters
        if mpar is not None:
            self.loadPath = mpar['loadPath']
            self.savePath = mpar['savePath']
            self.save     = mpar['save']
            self.saveExt  = mpar['saveExt']
            self.resFac   = mpar['resFac']
            self.plot     = mpar['plot']
            self.con      = mpar['con']
            self.areaMin  = mpar['areaMin']
            self.fMin     = mpar['fMin']
            self.fMax     = mpar['fMax']
            self.fields   = mpar['fields']['cm']

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        iOrg : float
            Organisation Index.

        '''
        cmlab,num  = label(field,return_num=True,connectivity=self.con)
        regions    = regionprops(cmlab)
        
        xC = []; yC = []
        for i in range(len(regions)):
            props  = regions[i]
            if props.area > self.areaMin:
                y0, x0 = props.centroid; 
                xC.append(x0); yC.append(y0)
    
        posScene = np.vstack((np.asarray(xC),np.asarray(yC))).T
        
        print('Number of regions: ',posScene.shape[0],'/',num)
    
        ## Compute the nearest neighbour distances ##
        # Scene
        nnScene  = cKDTreeMethod(posScene,field)
        nbins = len(nnScene)+1; dx=0.01
        bins = np.linspace(np.min(nnScene)-dx,np.max(nnScene)+dx,nbins)
        nndpdfScene = np.histogram(nnScene, bins)[0]
        nndcdfScene = np.cumsum(nndpdfScene) / len(nnScene)
        
        # Poisson
        lam   = nnScene.shape[0] / (field.shape[0]*field.shape[1])
        binav = (bins[1:] + bins[:-1])/2
        nndcdfRand  = 1 - np.exp(-lam*np.pi*binav**2) 
                
        ## Compute Iorg ##
        iOrg = np.trapz(nndcdfScene,nndcdfRand)
        
        if self.plot:
            fig,axs=plt.subplots(ncols=3,figsize=(15,5))
            axs[0].imshow(field,'gray')
            axs[0].set_title('Cloud mask of scene')
            
            axs[1].scatter(posScene[:,0],field.shape[0] - posScene[:,1],
                           color='k', s=5)
            axs[1].set_title('Scene centroids')
            asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
            axs[1].set_aspect(asp)
            
            axs[2].plot(nndcdfRand,nndcdfScene,'-',color='k')
            axs[2].plot(nndcdfRand,nndcdfRand,'--',color='k')
            axs[2].set_title('Nearest neighbour distribution')
            axs[2].set_xlabel('Poisson nearest neighbour CDF')
            axs[2].set_ylabel('Scene nearest neighbour CDF')
            axs[2].annotate(r'$I_{org} = $'+str(round(iOrg,3)),(0.7,0.1),
                            xycoords='axes fraction')
            asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
            axs[2].set_aspect(asp)
            plt.show()
        
        return iOrg
        
    def verify(self):
        '''
        Verification - ensuring that randomly organised scenes with same 
        number of cloud objects as the scene in fmin amounts to iOrg of 0.5.

        Returns
        -------
        iOrg : float
            Metric for verification case.

        '''
        aMin = self.areaMin; plotBool = self.plot
        self.areaMin = 0; self.plot = True
        files, dates = findFiles(self.loadPath)
        file = files[self.fMin]
        field = getField(file, self.field, self.resFac, binary=True)
        _,num  = label(field,return_num=True,connectivity=self.con)
        
        posScene = np.random.randint(0, high=field.shape[0], size=(num,2))
        ranField = np.zeros(field.shape)
        ranField[posScene[:,0],posScene[:,1]] = 1
        
        iOrg = self.metric(ranField)
        
        print('iOrg: ', iOrg)

        self.areaMin = aMin; self.plot = plotBool
        
        return iOrg
        
    def compute(self):
        '''
        Main loop over scenes. Loads fields, computes metric, and stores it.

        '''
        files, dates = findFiles(self.loadPath)
        files = files[self.fMin:self.fMax]
        dates = dates[self.fMin:self.fMax]

        if self.save:
            saveSt    = self.saveExt
            dfMetrics = pd.read_hdf(self.savePath+'/Metrics'+saveSt+'.h5')
        
        ## Main loop over files
        for f in range(len(files)):
            cm = getField(files[f], self.field, self.resFac, binary=True)
            print('Scene: '+files[f]+', '+str(f+1)+'/'+str(len(files)))
            
            iOrg = self.metric(cm)
            print('iOrg: ', iOrg)   

            if self.save:
                dfMetrics['iOrgPoiss'].loc[dates[f]] = iOrg
        
        if self.save:
            dfMetrics.to_hdf(self.savePath+'/Metrics'+saveSt+'.h5', 'Metrics',
                             mode='w')

if  __name__ == '__main__':
    mpar = {
            'loadPath' : '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Filtered',
            'savePath' : '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Metrics',
            'save'     : True, 
            'resFac'   : 1,     # Resolution factor (e.g. 0.5)
            'plot'     : True,  # Plot with details on each metric computation
            'con'      : 1,     # Connectivity for segmentation (1:4 seg, 2:8 seg)
            'areaMin'  : 4,     # Minimum cloud size considered for object metrics
            'fMin'     : 0,     # First scene to load
            'fMax'     : None,  # Last scene to load. If None, is last scene in set
           }
    iOrgPoissonGen = IOrgPoisson(mpar)
    iOrgPoissonGen.verify()
    iOrgPoissonGen.compute()
    
        