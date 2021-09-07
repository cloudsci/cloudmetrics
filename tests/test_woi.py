import numpy as np
import pytest


import cloudmetrics

def test_woi():

    cloud_scalar = np.zeros((512,512))
    
    woi1, woi2, woi3, specs = cloudmetrics.woi(cloud_scalar, return_spectra=True)

    # Validate wavelet energy spectrum -> if correct total energy should be
    # the same as in image space
    Ewav = np.sum(specs)
    Eimg = np.mean(cloud_scalar ** 2)
    np.testing.assert_allclose(Ewav, Eimg, atol=1e-10)
    