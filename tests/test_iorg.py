import pytest
import numpy as np

from cloudmetrics.utils import create_circular_mask, make_periodic_field
import cloudmetrics


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
def test_lattice_of_squares(periodic_domain, connectivity):
    """
    1. Regular lattice of squares (iOrg -> 0)
    """
    # 1. Regular lattice of squares
    cloud_mask = np.zeros((512, 512))
    cloud_mask[::16, ::16] = 1
    cloud_mask[1::16, ::16] = 1
    cloud_mask[::16, 1::16] = 1
    cloud_mask[1::16, 1::16] = 1

    if periodic_domain:
        cloud_mask = make_periodic_field(cloud_mask, con=connectivity)

    i_org = cloudmetrics.iorg(
        cloud_mask,
        periodic_domain=periodic_domain,
        connectivity=connectivity,
        area_min=0,
        random_seed=0,
    )
    np.testing.assert_allclose(i_org, 0.0, atol=0.1)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
def test_random_points(periodic_domain, connectivity):
    """
    2. Randomly scattered points (iOrg -> 0.5)
    """
    # 2. Randomly scattered points
    posScene = np.random.randint(0, high=512, size=(1000, 2))
    cloud_mask = np.zeros((512, 512))
    cloud_mask[posScene[:, 0], posScene[:, 1]] = 1

    if periodic_domain:
        cloud_mask = make_periodic_field(cloud_mask, con=connectivity)

    i_org = cloudmetrics.iorg(
        cloud_mask,
        periodic_domain=periodic_domain,
        connectivity=connectivity,
        area_min=0,
        random_seed=0,
    )
    np.testing.assert_allclose(i_org, 0.5, atol=0.1)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
def test_single_uniform_circle(periodic_domain, connectivity):
    """
    3. One large, uniform circle with noise around it (iOrg -> 1)
    """
    # 3. One large, uniform circle with noise around it
    cloud_mask = np.zeros((512, 512))
    maw = 128
    mask = create_circular_mask(maw, maw).astype(int)
    cloud_mask[:maw, :maw] = mask
    # cloud_mask[maw-20:2*maw-20,maw-50:2*maw-50] = mask;
    tadd = np.random.rand(maw, maw)
    ind = np.where(tadd > 0.4)
    tadd[ind] = 1
    ind = np.where(tadd <= 0.4)
    tadd[ind] = 0
    cloud_mask[:maw, :maw] += tadd
    cloud_mask[cloud_mask > 1] = 1

    if periodic_domain:
        cloud_mask = make_periodic_field(cloud_mask, con=connectivity)

    i_org = cloudmetrics.iorg(
        cloud_mask,
        periodic_domain=periodic_domain,
        connectivity=connectivity,
        area_min=0,
        random_seed=0,
    )
    np.testing.assert_allclose(i_org, 1.0, atol=0.1)
