#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def cloud_fraction(mask):
    """
    Compute metric(s) for a single field

    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
            (cloud) mask field.

    Returns
    -------
    cf : float
        cloud fraction.

    """

    return np.count_nonzero(mask) / mask.size
