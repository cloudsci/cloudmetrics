#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def mica(mask, periodic_domain=False):
    """
    Compute the Morphological Index of Convective Aggregation (MICA)
    by Kadoya and Masunaga 2012 (https://doi.org/10.2151/jmsj.2018-054)

    This metric assesses the level of organization by measuring the compactness
    of cloud clusters within a defined area. Simultaneously, it gauges the
    proportion of clear sky outside the convective region in relation to the
    entire observed domain.


    Parameters
    ----------
    mask:            numpy array of shape (npx,npx) - npx is number of pixels
                     (cloud) mask field.
    periodic_domain: whether the provided (cloud) mask is on a periodic domain
                     (for example from a LES simulation)

    Returns
    -------
    mica:            `summary_measure` (default "max") of open-sky regions
                     identified in mask

    """

    image_size = mask.size
    mask_binary = np.where(mask>0, 1, 0)
    total_area = np.sum(mask_binary)


    if periodic_domain:
        doubled_mask = np.tile(mask_binary, (2, 2))

        indices_x = np.max(doubled_mask, axis=0)
        indices_y = np.max(doubled_mask, axis=1)
        max_consecutive_zeros_x = max(map(len, ''.join(map(str, indices_x)).split('1')))
        max_consecutive_zeros_y = max(map(len, ''.join(map(str, indices_y)).split('1')))

        delta_x = mask_binary.shape[0] - max_consecutive_zeros_x
        delta_y = mask_binary.shape[1] - max_consecutive_zeros_y

    else:
        non_zero_indices = np.nonzero(mask_binary)

        # Get the minimum and maximum coordinates along each axis
        min_x, min_y = np.min(non_zero_indices, axis=1)
        max_x, max_y = np.max(non_zero_indices, axis=1)+1
        delta_x = max_x - min_x
        delta_y = max_y - min_y

    Acls = delta_x * delta_y  # minimum rectangle including all convective regions
    mica = 1.* ( total_area / Acls ) * ( image_size - Acls ) / image_size


    return mica

