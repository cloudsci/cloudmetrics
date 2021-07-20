import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree


def create_circular_mask(h, w):
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def make_periodic_field(field, con):
    """
    Apply periodic BCs to cloud mask fields, based on implementation from
    https://github.com/MennoVeerman/periodic-COP

    Parameters
    ----------
    field : (npx,npx) numpy array
        Cloud mask field (no other cloud field accepted!).

    Returns
    -------
    Field of (2*npx,2*npx), with cloud objects that cross boundaries translated
    to coherent structures crossing the northern/eastern boundaries.

    """

    # Questions remaining:
    # - How to handle regions whose cog lies outside the original image?

    ny, nx = field.shape

    # Create array with extra cells in y and x direction to handle periodic BCs
    cld = np.zeros((2 * ny, 2 * nx))

    # set clouds mask
    cld[:ny, :nx] = field.copy()

    # Label connected regions of cloudy pixels
    cld_lbl, nlbl = label(cld, connectivity=con, return_num=True)

    # Find all clouds (labels) that cross the domain boundary in n-s direction.
    # Save the labels of the cloudy region at both the southern border
    # (clouds_to_move_north) and at the northern border (clouds_in_the_north)
    y0, y1 = 0, ny - 1
    clouds_to_move_north = []
    clouds_in_the_north = []
    for ix in range(nx):
        if (
            cld_lbl[y0, ix] > 0
            and cld_lbl[y1, ix] > 0
            and cld_lbl[y0, ix] != cld_lbl[y1, ix]
            and cld_lbl[y0, ix]
        ):
            clouds_to_move_north += [cld_lbl[y0, ix]]
            clouds_in_the_north += [cld_lbl[y1, ix]]

    # Find all clouds (labels) that cross the domain boundary in e-w direction.
    # Save the labels of the cloudy region at both the western border
    # (clouds_to_move_east) and at the western border (clouds_in_the_east)
    x0, x1 = 0, nx - 1
    clouds_to_move_east = []
    clouds_in_the_east = []
    for iy in range(ny):
        if (
            cld_lbl[iy, x0] > 0
            and cld_lbl[iy, x1] > 0
            and cld_lbl[iy, x0] != cld_lbl[iy, x1]
        ):
            clouds_to_move_east += [cld_lbl[iy, x0]]
            clouds_in_the_east += [cld_lbl[iy, x1]]

    # Move all cloud parts in the west(south) that are connected to cloud parts
    # in the east(north) towards to east(north) beyond the boundaries of the
    # original domain.
    regions = regionprops(cld_lbl)
    for cloud in np.unique(cld_lbl):
        # Loop over all identified cloud clusters
        shift_y, shift_x = 0, 0

        if cloud in clouds_to_move_north:
            # Clouds region connects to cloud region at the northern boundary
            # and will be moved north
            shift_y = ny
            if (
                clouds_in_the_north[clouds_to_move_north.index(cloud)]
                in clouds_to_move_east
            ):
                # The connected cloud in the north will be moved east:
                # diagonal crossing
                shift_x = nx

        if cloud in clouds_to_move_east:
            shift_x = nx
            if (
                clouds_in_the_east[clouds_to_move_east.index(cloud)]
                in clouds_to_move_north
            ):
                shift_y = ny

        if cloud in clouds_in_the_north:
            if (
                clouds_to_move_north[clouds_in_the_north.index(cloud)]
                in clouds_to_move_east
            ):
                # Cloud is connnected to a cloud region in the south, but that
                # cloud region will also be moved eastward
                shift_x = nx
        if cloud in clouds_in_the_east:
            if (
                clouds_to_move_east[clouds_in_the_east.index(cloud)]
                in clouds_to_move_north
            ):
                # Cloud is connnected to a cloud region in the west, but that
                # cloud region will also be moved northward
                shift_y = ny

        # Shift the clouds
        if shift_y > 0 or shift_x > 0:
            # Cloud should be shifted in east-west and/or north-south direction
            region = regions[cloud - 1]
            # Remove from current position
            cld_lbl[region.coords[:, 0], region.coords[:, 1]] = 0
            # Put in new position
            cld_lbl[region.coords[:, 0] + shift_y, region.coords[:, 1] + shift_x] = 1

    return np.where(cld_lbl > 0, 1, 0)


def find_nearest_neighbors(data, size=None):
    # FIXME not sure if boxsize (periodic BCs) work if domain is not square
    tree = cKDTree(data, boxsize=size)
    dists = tree.query(data, 2)
    nn_dist = np.sort(dists[0][:, 1])
    return nn_dist
