import numpy as np
import pytest

import cloudmetrics
from cloudmetrics.utils import create_circular_mask


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("reference_dist", ["poisson", "inhibition_nn"])
def test_lattice_of_squares(periodic_domain, connectivity, reference_dist):
    """
    1. Regular lattice of squares (iOrg -> 0)
    """
    # 1. Regular lattice of squares
    mask = np.zeros((512, 512))
    mask[::16, ::16] = 1
    mask[1::16, ::16] = 1
    mask[::16, 1::16] = 1
    mask[1::16, 1::16] = 1

    i_org = cloudmetrics.mask.iorg_objects(
        mask,
        periodic_domain=periodic_domain,
        reference_dist=reference_dist,
    )

    np.testing.assert_allclose(i_org, 0.0, atol=0.1)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("reference_dist", ["poisson", "inhibition_nn"])
def test_random_points(periodic_domain, connectivity, reference_dist):
    """
    2. Randomly scattered points (iOrg -> 0.5)
    """
    # 2. Randomly scattered points
    posScene = np.random.randint(0, high=512, size=(1000, 2))
    mask = np.zeros((512, 512))
    mask[posScene[:, 0], posScene[:, 1]] = 1

    i_org = cloudmetrics.mask.iorg_objects(
        mask,
        periodic_domain=periodic_domain,
    )
    np.testing.assert_allclose(i_org, 0.5, atol=0.1)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("reference_dist", ["poisson", "inhibition_nn"])
def test_single_uniform_circle(periodic_domain, connectivity, reference_dist):
    """
    3. One large, uniform circle with noise around it (iOrg -> 1)
    """
    # 3. One large, uniform circle with noise around it
    mask = np.zeros((512, 512))
    maw = 128
    mask_circle = create_circular_mask(maw, maw).astype(int)
    mask[:maw, :maw] = mask_circle
    # mask[maw-20:2*maw-20,maw-50:2*maw-50] = mask;
    tadd = np.random.rand(maw, maw)
    ind = np.where(tadd > 0.4)
    tadd[ind] = 1
    ind = np.where(tadd <= 0.4)
    tadd[ind] = 0
    mask[:maw, :maw] += tadd
    mask[mask > 1] = 1

    i_org = cloudmetrics.mask.iorg_objects(
        mask,
        periodic_domain=periodic_domain,
    )
    np.testing.assert_allclose(i_org, 1.0, atol=0.1)
