#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import scipy.spatial.distance as sd
from .utils import findFiles, getField

class COP():
    '''
    Class for computing Convective Organisation Potential (COP -
    White et al. 2018) from a cloud mask. Has a fast, array-based 
    implementation to go along with the slower, loop-based implementation.
    
    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. One of 
                      these columns can be filled by 'cop'.
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
            self.field    = mpar['fields']['cm']

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        COP : float
            Convective Organisation Potential.

        '''
        cmlab,num  = label(field,return_num=True,connectivity=self.con)
        regions    = regionprops(cmlab)
        
        area = []; xC = []; yC = []
        for i in range(num):
            props  = regions[i]
            if props.area > self.areaMin:
                y0, x0 = props.centroid
                xC.append(x0); yC.append(y0)
                area.append(props.area)
        area = np.asarray(area)
        pos  = np.vstack((np.asarray(xC),np.asarray(yC))).T
        
        print('Number of regions: ',pos.shape[0],'/',num)
        
        ## COMPUTE COP (Array-based)
        dij = sd.squareform(sd.pdist(pos))          # Pairwise distance matrix
        dij = dij[np.triu_indices_from(dij, k=1)]   # Upper triangular (no diag)
        aSqrt = np.sqrt(area)                       # Square root of area
        Aij = aSqrt[:, None] + aSqrt[None,:]        # Pairwise area sum matrix 
        Aij = Aij[np.triu_indices_from(Aij, k=1)]   # Upper triangular (no diag)
        Vij = Aij / (dij*np.sqrt(np.pi))            # Pairwise interaction pot.
        cop = np.sum(Vij)/(0.5*num*(num-1))         # COP
        
        return cop
        
    def verify(self):
        '''
        Verification with simple example from White et al. (2018)

        Returns
        -------
        cop : float
            Metric for verification case.

        '''
        cm = np.zeros((20,20))
        cm[2:9,2:9] = 1;   cm[11:18,2:9] = 1
        cm[2:9,11:18] = 1; cm[11:18,11:18] = 1
        
        cop = self.metric(cm)
        
        return cop
        
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
            
            cop = self.metric(cm)
            print('COP: ', cop)   

            if self.save:
                dfMetrics['cop'].loc[dates[f]]   = cop
        
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
    copGen = COP(mpar)
    copGen.verify()
    copGen.compute()
    
        