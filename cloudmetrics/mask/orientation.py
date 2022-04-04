import numpy as np


def _raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies**i_order * y_indices**j_order).sum()


def _moments_cov(data):
    data_sum = data.sum()
    m10 = _raw_moment(data, 1, 0)
    m01 = _raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (_raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (_raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (_raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov


def orientation(mask, debug=False, periodic_domain=False):
    """
    Compute a measure for a single (cloud) mask's degree of directional alignment,
    using the (cloud) mask's raw image moment covariance matrix.

    Code based on: https://github.com/alyssaq/blog/blob/master/posts/150114-054922_computing-the-axes-or-orientation-of-a-blob.md.
    Note that this function currently does not support periodic boundary
    conditions (use the wavelet-based orientation measure woi3 for such scenes).

    Parameters
    ----------
    mask : numpy array of shape (npx,npx) - npx is number of pixels
           (cloud) mask field.

    Returns
    -------
    orie : float
        Orientation measure (dimensionless value between 0-1, with 0 denoting
        no preferential direction of orientation and 1 denoting that all
        information is oriented in one direction)

    """

    if periodic_domain:
        raise NotImplementedError(periodic_domain)

    cov = _moments_cov(mask)
    if np.isnan(cov).any() or np.isinf(cov).any():
        return np.nan

    evals, evecs = np.linalg.eig(cov)
    orie = np.sqrt(1 - np.min(evals) / np.max(evals))

    if debug:
        _debug_plot(mask=mask, evecs=evecs, orie=orie, evals=evals)

    return orie


def _debug_plot(mask, evals, evecs, orie):
    import matplotlib.pyplot as plt

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # evec with largest eval
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    evalsn = evals[sort_indices] / evals[sort_indices][0]

    scale = 10
    ox = int(mask.shape[1] / 2)
    oy = int(mask.shape[0] / 2)
    lw = 5

    _, ax = plt.subplots()
    ax.imshow(mask, "gray")
    # plt.scatter(ox+x_v1*-scale*2,oy+y_v1*-scale*2,s=100)
    ax.plot(
        [ox - x_v1 * scale * evalsn[0], ox + x_v1 * scale * evalsn[0]],
        [oy - y_v1 * scale * evalsn[0], oy + y_v1 * scale * evalsn[0]],
        linewidth=lw,
    )
    ax.plot(
        [ox - x_v2 * scale * evalsn[1], ox + x_v2 * scale * evalsn[1]],
        [oy - y_v2 * scale * evalsn[1], oy + y_v2 * scale * evalsn[1]],
        linewidth=lw,
    )
    ax.set_title("Alignment measure = " + str(round(orie, 3)))
    plt.show()
