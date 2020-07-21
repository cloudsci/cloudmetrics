# cloudmetrics
An end-to-end implementation to download, characterise and analyse cloud field patterns in satellite observations, using metrics and principal component analysis.

Follow process stream in main.py to:
  1. Download images and cloud products from NASA's MODAPS API to its LAADS archive of observations from the MODIS instrument aboard the Aqua and Terra satellites
  2. Preprocess and store data in compressed (.h5) Pandas dataframes
  3. Compute 36 metrics of the cloud fields using the Metrics module
  4. Analyse several components of the resulting metric dataset with PCA, clustering and plotting routines.

The Metrics module also serves well as a stand-alone product. Given any square input field with a shape that is a power of 2, each metric object has a method (metric) that facilitates quick computation of that particular metric.

To use cloudmetrics, it is easiest to clone this repository and follow the commented guidelines in main.py. Make sure to update the package dependencies to those listed in requirements.txt.
