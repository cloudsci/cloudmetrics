#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
import scipy.spatial.distance as sd
from scipy.stats.mstats import gmean
from .utils import findFiles, getField

class SCAI():
    '''
    Class for computing the Simple Convective Aggregation Index (scai -
    Tobin et al. 2012) from a cloud mask. Can compute scai and mean geometric 
    nearest neighbour distance (d0).
    
    Parameters
    ----------
    mpar : Dict
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Two of 
                      these columns can be filled by 'scai' and 'd0'.
           save     : Boolean to specify whether to store the variables in
                      savePath/Metrics.h5
           resFac   : Resolution factor (e.g. 0.5), to coarse-grain the field.
           plot     : Boolean to specify whether to make plot with details on
                      this metric for each scene.
           con      : Connectivitiy for segmentation (1 - 4 seg, 2 - 8 seg)
           areaMin  : Minimum cloud size considered in computing metric
           fMin     : First scene to load
           fMax     : Last scene to load. If None, is last scene in set.
                     
    '''
    def __init__(self, mpar):
        # General parameters
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

        # Metric-specific parameters
        self.field    = 'Cloud_Mask_1km'
        self.L        = 1000            # Arbitrary length scale (constant for 
                                        # all scenes). Set to 1000 for 
                                        # consistency with Tobin et al. (2012)

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        D0 : float
            Mean geometric nearest neighbour distance between objects.
        scai : float
            Simple Convective Aggregation Index.

        '''
        cmlab,num  = label(field,return_num=True,connectivity=self.con)
        regions    = regionprops(cmlab)
        
        xC = []; yC = []
        for i in range(num):
            props  = regions[i]
            if props.area > self.areaMin:
                y0, x0 = props.centroid
                xC.append(x0); yC.append(y0)
        pos = np.vstack((np.asarray(xC),np.asarray(yC))).T
        
        print('Number of regions: ',pos.shape[0],'/',num)
        
        di   = sd.pdist(pos)
        D0   = gmean(di)
        Nmax = field.shape[0]*field.shape[1]/2
        scai = num / Nmax * D0 / self.L * 1000
        
        if self.plot:
            plt.imshow(field,'gray')
            plt.title('scai: '+str(round(scai,3))) 
            plt.show()
        
        return D0, scai
        
    def verify(self):
        '''
        Verification with simple example from Tobin et al. (2012)

        Returns
        -------
        veri : List of floats
            List containing metric(s) for verification case.

        '''
        cm = np.zeros((20,20))
        cm[11,12] = 1; cm[11:13,13:15] = 1; cm[12,10] = 1
        cm[14:16,6:12] = 1; cm[16,12] = 1; cm[17,10] = 1
        
        D0, scai = self.metric(cm)
        
        veri = [D0, scai]
        return veri
        
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
            
            D0, scai = self.metric(cm)
            print('D0:   ', D0)
            print('SCAI: ', scai)   

            if self.save:
                dfMetrics['d0'].loc[dates[f]]   = D0
                dfMetrics['scai'].loc[dates[f]] = scai
        
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
    scaiGen = SCAI(mpar)
    scaiGen.verify()
    scaiGen.compute()
    
    