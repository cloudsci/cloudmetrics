#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from .utils import findFiles, getField
import multiprocessing as mp
from tqdm import tqdm

def pair_correlation_2d(x, y, S, r_max, dr, normalize=True, mask=None):
    """
    
    Pair correlation function, directly copied from:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py
    and indirectly from Rasp et al. (2018)
    
    Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius r_max drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller r_max...or write some code to handle edge effects! ;)
    
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        r_max            outer diameter of largest annulus
        dr              increment for increasing radius of annulus
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """

    # Number of particles in ring/area of ring/number of reference
    # particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Extract domain size
    (Sx,Sy) = S if len(S) == 2 else (S, S)

    # Find particles which are close enough to the box center that a circle of radius
    # r_max will not cross any edge of the box

    # Find indices within boundaries
    if mask is None:
        bools1 = x > r_max          # Valid centroids from left boundary
        bools2 = x < (Sx - r_max)   # Valid centroids from right boundary
        bools3 = y > r_max          # Valid centroids from top boundary
        bools4 = y < (Sy - r_max)   # Valid centroids from bottom boundary
        interior_indices, = np.where(bools1 * bools2 * bools3 * bools4)
    else:
        # Get closest indices for parcels in a pretty non-pythonic way
        # and check whether it is inside convolved mask
        x_round = np.round(x)
        y_round = np.round(y)
        interior_indices = []
        for i in range(x_round.shape[0]):
            if mask[int(x_round[i]), int(y_round[i])] == 1:
                interior_indices.append(i)

    num_interior_particles = len(interior_indices)

    edges = np.arange(0., r_max + dr, dr)   # Annulus edges
    num_increments = len(edges) - 1         
    g = np.zeros([num_interior_particles, num_increments]) # RDF for all interior particles
    radii = np.zeros(num_increments)
    number_density = float(len(x)) / float(Sx*Sy) # Normalisation

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = np.sqrt((x[index] - x)**2 + (y[index] - y)**2)
        d[index] = 2 * r_max   # Because sqrt(0)

        result, bins = np.histogram(d, bins=edges)
        if normalize:
            result = result/number_density
        g[p, :] = result

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2))

    return g_average, radii, interior_indices

class RDF():
    '''
    Class for computing the Radial Distribution Function between objects (rdf)
    and derived metrics, from a cloud mask. Can compute the maximum of the
    RDF (rdfMax), the difference between minimum and maximum (rdfDiff). The
    implementation is based off:
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py
    
    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Three of 
                      these columns can be filled by 'rdfMax', 'rdfMin' and 
                      'rdfDiff'.
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
        self.dx       = 1   # Convert pixel to km
        self.rMax     = 40  # How far away to compute the rdf
        self.dr       = 1   # Bin width
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
            self.nproc    = mpar['nproc']

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        rdfM : float
            Maximum of the radial distribution function.
        rdfI : float
            Integral of the radial distribution function.
        rdfD : float
            Max-min difference of the radial distribution function.

        '''
        cmlab,num  = label(field,return_num=True,connectivity=self.con)
        regions    = regionprops(cmlab)
        
        xC = []; yC = []
        for i in range(num):
            props  = regions[i]
            if props.area > self.areaMin:
                yC.append(props.centroid[0])
                xC.append(props.centroid[1])
        
        pos = np.vstack((np.asarray(xC),np.asarray(yC))).T
        
        # print('Number of regions: ',pos.shape[0],'/',num)

        if pos.shape[0] < 1:
            return float('nan'),float('nan'),float('nan')
        
        rdf, rad, tmp = pair_correlation_2d(pos[:, 0], pos[:, 1],
                                            [field.shape[0], field.shape[1]],
                                            self.rMax, self.dr, normalize=True, 
                                            mask=None)
        rad *= self.dx
        rdfM = np.max(rdf)
        rdfI = np.trapz(rdf,rad)
        rdfD = np.max(rdf) - rdf[-1]
        
        if self.plot:
            axF = 'axes fraction'
            fig,axs = plt.subplots(ncols=2,figsize=(8.5,4))
            axs[0].imshow(field,'gray')
            axs[0].axis('off')
            
            axs[1].plot(rad,rdf)
            axs[1].set_xlabel('Distance')
            axs[1].set_ylabel('RDF')
            axs[1].annotate('rdfMax = %.3f' %rdfM,(0.6,0.15), xycoords=axF)
            axs[1].annotate('rdfInt = %.3f' %rdfI,(0.6,0.10), xycoords=axF)
            axs[1].annotate('rdfDif = %.3f' %rdfD,(0.6,0.05), xycoords=axF)
            plt.show()

        return rdfM, rdfI, rdfD
        
    def verify(self):
        return 'Not implemented for RDF'
    
    def getcalc(self,file):
        cm = getField(file, self.field, self.resFac, binary=True)
        return self.metric(cm)
    
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
        with mp.Pool(processes = self.nproc) as pool:
            results = list(tqdm(pool.imap(self.getcalc,files),total=len(files)))
        results = np.array(results)
        
        if self.save:
            dfMetrics['rdfMax'].loc[dates]   = results[:,0]
            dfMetrics['rdfInt'].loc[dates]   = results[:,1]
            dfMetrics['rdfDiff'].loc[dates]  = results[:,2]
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
    rdfGen = RDF(mpar)
    rdfGen.verify()
    rdfGen.compute()
        
    
