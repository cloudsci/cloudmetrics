#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import findFiles, getField

class CF():
    '''
    Class for computing the cloud fraction from a cloud mask.
    
    Parameters
    ----------
    mpar : Dict
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics (one is cf) and whose indices are scenes.
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

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        cf : float
            Clod fraction.

        '''
        cf = len(np.where(field == 1)[0]) / field.size
        
        if self.plot:
            plt.imshow(field,'gray')
            plt.title('Cloud fraction: '+str(round(cf,3))) 
            plt.show()
        
        return cf
        
    def verify(self):
        '''
        Verification not implemented for cloud fraction.
        '''
        
        return 'Verification not implemented for cloud fraction'
        
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
            
            cf = self.metric(cm)
            print('Cloud fraction: ', cf)

            if self.save:
                dfMetrics['cf'].loc[dates[f]] = cf
        
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
    cfGen = CF(mpar)
    cfGen.verify()
    cfGen.compute()
    