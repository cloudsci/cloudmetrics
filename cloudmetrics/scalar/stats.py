import scipy as sp


def mean(mask, scalar_field):
    """
    Compute the masked mean of scalar field
    """
    return scalar_field[mask].mean()


def var(mask, scalar_field):
    """
    Compute the masked variance of scalar field
    """
    return scalar_field[mask].var()


def skew(mask, scalar_field):
    """
    Compute the masked skewness of scalar field
    """
    return sp.stats.skew(scalar_field[mask])


def kurtosis(mask, scalar_field):
    """
    Compute the masked kurtosis of scalar field
    """
    return sp.stats.kurtosis(scalar_field[mask])
