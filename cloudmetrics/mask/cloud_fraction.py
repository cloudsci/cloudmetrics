#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def cloud_fraction(cloud_mask):
    """
    Compute metric(s) for a single field

    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.

    Returns
    -------
    cf : float
        Clod fraction.

    """

    return np.count_nonzero(cloud_mask) / cloud_mask.size
