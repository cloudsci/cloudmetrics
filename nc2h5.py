#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:07:31 2020

@author: martinjanssens
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import os
import glob
import sys
# Need to unpack from netCDF and store as h5

def findDirs(loadPath):
    dirs  = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(loadPath):
        for direc in d:
            dname = os.path.join(r, direc)
            dirs.append(dname)
    dirs = np.sort(dirs)
    return dirs

loadPath = '/projects/0/einf170/janssonf/botany/botany-3-_b0vhnjz/runs'
if len(sys.argv) > 1:
    loadPath = sys.argv[1]

lwpThr   = 1e-7
nRuns    = 16
Nc       = 70000000 # FIXME assumes this is constant
tSU      = 5400 # Seconds after start allowed for spin-up
overwrite= False

# Get paths to data (cape*.nc) directories 
dirs = findDirs(loadPath)

for k in range(len(dirs)):
    
    # Check if cape2d.001.nc exists
    capeFlag = os.path.exists(dirs[k]+'/cape2d.001.nc')
    if not capeFlag:
        continue

    # Check if there are already .h5 files stored here and move on if not 
    # allowed to overwrite
    h5s = glob.glob(dirs[k]+'/*.h5')
    if len(h5s) != 0 and not overwrite:
        continue
    
    print('Processing '+dirs[k]+'/cape2d.001.nc')
    dataset = Dataset(dirs[k]+'/cape2d.001.nc')
    
    time = np.ma.getdata(dataset.variables['time'][:])
    iMin = np.where(time>tSU)[0][0]
    
    lwp = np.ma.getdata(dataset.variables['lwp'][:])
    cth = np.ma.getdata(dataset.variables['cldtop'][:])

    # Cloud mask based on preset lwp threshold
    cm  = np.zeros(lwp.shape); cm[lwp>lwpThr] = 1
    
    # Albedo using model from Zhang et al. (2005)
    tau = 0.19*lwp**(5./6)*Nc**(1/3)
    alb = tau/(6.8+tau)            
    
    # Plot last 50 time steps
    # for i in range(50):
    #     plt.imshow(alb[-50+i,:,:])
    #     plt.colorbar()
    #     plt.show()
    
    for i in range(len(time)):
        if i >= iMin:
            ind = str(int(time[i]))
            df  = pd.DataFrame(index=[ind],columns=['image',
                                                    'CloudMask',
                                                    'CloudWaterPath',
                                                    'CloudTopHeight'])
            
            df.loc[ind,'image']	         = alb[i,:,:]
            df.loc[ind,'CloudMask']	     = cm [i,:,:]
            df.loc[ind,'CloudWaterPath'] = lwp[i,:,:]
            df.loc[ind,'CloudTopHeight'] = cth[i,:,:]
        
            df.to_hdf(dirs[k]+'/'+ind+'.h5',key='Cloud fields')
