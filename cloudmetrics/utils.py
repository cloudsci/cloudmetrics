import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops


def create_circular_mask(h, w):
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def make_periodic_mask(mask, object_connectivity):
    """
    Apply periodic BCs to (cloud) mask fields by:

    1. doubling the domain in x- and y-direction by padding with zeros
    2. identifying individual objects in the mask
    3. moving the objects that wrap the west and north boundaries to the south
       and east so that they are unwrapped into contiguous regions in the
       returned mask

    based on implementation from https://github.com/MennoVeerman/periodic-COP

    Parameters
    ----------
    field : (npx,npx) numpy array
        (Cloud) mask field
    object_connectivity:
        Maximum number of orthogonal hops to consider a pixel/voxel as a
        neighbor. Accepted values are ranging from 1 to input.ndim

    Returns
    -------
    Field of (2*npx,2*npx), with (cloud) objects that cross boundaries translated
    to contiguous structures crossing the northern/eastern boundaries.

    """
    # TODO: How to handle regions whose cog lies outside the original image?
    ny, nx = mask.shape

    # Create array with extra cells in y and x direction to handle periodic BCs
    mask_periodic = np.zeros((2 * ny, 2 * nx))

    # set periodic (cloud) mask
    mask_periodic[:ny, :nx] = mask.copy()

    # Label connected regions in the mask
    mask_periodic_lbl, nlbl = label(
        mask_periodic, connectivity=object_connectivity, return_num=True
    )

    # Find all object (labels) that cross the domain boundary in n-s direction.
    # Save the labels of the masked region at both the southern border
    # (objects_to_move_north) and at the northern border (objects_in_the_north)
    y0, y1 = 0, ny - 1
    objects_to_move_north = []
    objects_in_the_north = []
    for ix in range(nx):
        if (
            mask_periodic_lbl[y0, ix] > 0
            and mask_periodic_lbl[y1, ix] > 0
            and mask_periodic_lbl[y0, ix] != mask_periodic_lbl[y1, ix]
            and mask_periodic_lbl[y0, ix]
        ):
            objects_to_move_north += [mask_periodic_lbl[y0, ix]]
            objects_in_the_north += [mask_periodic_lbl[y1, ix]]

    # Find all objects (labels) that cross the domain boundary in e-w direction.
    # Save the labels of the masked region at both the western border
    # (objects_to_move_east) and at the western border (objects_in_the_east)
    x0, x1 = 0, nx - 1
    objects_to_move_east = []
    objects_in_the_east = []
    for iy in range(ny):
        if (
            mask_periodic_lbl[iy, x0] > 0
            and mask_periodic_lbl[iy, x1] > 0
            and mask_periodic_lbl[iy, x0] != mask_periodic_lbl[iy, x1]
        ):
            objects_to_move_east += [mask_periodic_lbl[iy, x0]]
            objects_in_the_east += [mask_periodic_lbl[iy, x1]]

    # Move all objects in the west(south) that are connected to masked parts
    # in the east(north) towards to east(north) beyond the boundaries of the
    # original domain.
    regions = regionprops(mask_periodic_lbl)
    for obj in np.unique(mask_periodic_lbl):
        # Loop over all identified objects
        shift_y, shift_x = 0, 0

        if obj in objects_to_move_north:
            # object connects to region at the northern boundary
            # and will be moved north
            shift_y = ny
            if (
                objects_in_the_north[objects_to_move_north.index(obj)]
                in objects_to_move_east
            ):
                # The connected object in the north will be moved east:
                # diagonal crossing
                shift_x = nx

        if obj in objects_to_move_east:
            shift_x = nx
            if (
                objects_in_the_east[objects_to_move_east.index(obj)]
                in objects_to_move_north
            ):
                shift_y = ny

        if obj in objects_in_the_north:
            if (
                objects_to_move_north[objects_in_the_north.index(obj)]
                in objects_to_move_east
            ):
                # Object is connnected to a region in the south, but that
                # region will also be moved eastward
                shift_x = nx
        if obj in objects_in_the_east:
            if (
                objects_to_move_east[objects_in_the_east.index(obj)]
                in objects_to_move_north
            ):
                # object is connnected to a region in the west, but that
                # region will also be moved northward
                shift_y = ny

        # Shift the objects
        if shift_y > 0 or shift_x > 0:
            # Object should be shifted in east-west and/or north-south direction
            region = regions[obj - 1]
            # Remove from current position
            mask_periodic_lbl[region.coords[:, 0], region.coords[:, 1]] = 0
            # Put in new position
            mask_periodic_lbl[
                region.coords[:, 0] + shift_y, region.coords[:, 1] + shift_x
            ] = 1

    return np.where(mask_periodic_lbl > 0, 1, 0)


def find_nearest_neighbors(data, size=None):
    # FIXME not sure if boxsize (periodic BCs) work if domain is not square
    tree = cKDTree(data, boxsize=size)
    dists = tree.query(data, 2)
    nn_dist = np.sort(dists[0][:, 1])
    return nn_dist


def print_object_labels(object_labels):
    """
    debugging function to print a (cloud) mask or (cloud) object labels
    """
    if np.max(object_labels) > 9:
        raise NotImplementedError

    nx, ny = object_labels.shape

    for i in range(nx):
        for j in range(ny):
            print(object_labels.astype(int)[i, j], end="")
        print()


def compute_r_squared(func, coeffs, x, y):

    # Pseudo-R^2 (equal to R^2 for linear regressions, not interpretable as
    # variance fraction explained by model for non-linear regression, where
    # it can be less than zero).

    # fit values, and mean
    yhat = func(x, coeffs)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssres = np.sum((y - yhat) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    return 1 - ssres / sstot
