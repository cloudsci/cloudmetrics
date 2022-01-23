import scipy as sp


def mean(scalar_field, mask=None):
    """
    Compute the (optionally masked) mean of scalar field

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        Scalar for which to calculate the mean.
    mask : Optional (Boolean) mask. If passed, the mean will be computed over the
        masked (True) pixels.

    Returns
    -------
    mean : float
        Mean over the (masked) field
    """
    if isinstance(mask, type(None)):
        return scalar_field.mean()
    else:
        return scalar_field[mask].mean()


def var(scalar_field, mask=None):
    """
    Compute the (optionally masked) variance of scalar field

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        Scalar for which to calculate the variance.
    mask : Optional (Boolean) mask. If passed, the variance will be computed over the
        masked (True) pixels.

    Returns
    -------
    variance : float
        Variance over the (masked) field
    """
    if isinstance(mask, type(None)):
        return scalar_field.var()
    else:
        return scalar_field[mask].var()


def skew(scalar_field, mask=None):
    """
    Compute the (masked) skewness of scalar field

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        Scalar for which to calculate the skewness.
    mask : Optional (Boolean) mask. If passed, the skewness will be computed over the
        masked (True) pixels.

    Returns
    -------
    skewness : float
        Skewness over the (masked) field
    """
    if isinstance(mask, type(None)):
        return sp.stats.skew(scalar_field)
    else:
        return sp.stats.skew(scalar_field[mask])


def kurtosis(scalar_field, mask=None):
    """
    Compute the (masked) kurtosis of scalar field

    Parameters
    ----------
    scalar_field : numpy array of shape (npx,npx) - npx is number of pixels
        Scalar for which to calculate the kurtosis.
    mask : Optional (Boolean) mask. If passed, the kurtosis will be computed over the
        masked (True) pixels.

    Returns
    -------
    kurtosis : float
        Kurtosis over the (masked) field
    """
    if isinstance(mask, type(None)):
        return sp.stats.kurtosis(scalar_field)
    else:
        return sp.stats.kurtosis(scalar_field[mask])
