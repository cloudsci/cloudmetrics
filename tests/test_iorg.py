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


def _random_points_mask(seed, n_points=1000, size=512):
    rng = np.random.default_rng(seed)
    pos_scene = rng.integers(0, high=size, size=(n_points, 2))
    mask = np.zeros((size, size))
    mask[pos_scene[:, 0], pos_scene[:, 1]] = 1
    return mask


@pytest.mark.parametrize("periodic_domain", [True, False])
def test_inhibition_nn_reproducible_with_seed(periodic_domain):
    """
    Passing `random_seed` to the inhibition nearest-neighbour reference
    distribution must make the iorg value reproducible
    """
    mask = _random_points_mask(seed=0)

    values = [
        cloudmetrics.mask.iorg_objects(
            mask,
            periodic_domain=periodic_domain,
            reference_dist="inhibition_nn",
            reference_dist_kwargs={"random_seed": 42},
        )
        for _ in range(2)
    ]
    assert values[0] == values[1]

    # a different seed should (almost certainly) give a different value
    other = cloudmetrics.mask.iorg_objects(
        mask,
        periodic_domain=periodic_domain,
        reference_dist="inhibition_nn",
        reference_dist_kwargs={"random_seed": 43},
    )
    assert other != values[0]


def test_inhibition_nn_placed_circles_do_not_overlap():
    """
    The circles placed by the inhibition nearest-neighbour method must not
    overlap with each other or with any of their periodic images
    """
    from cloudmetrics.objects.metrics.iorg import _place_circles_randomly

    domain_shape = [300, 200]
    ny, nx = domain_shape
    rng = np.random.default_rng(0)
    radii = np.flip(np.sort(rng.uniform(0.5, 6.0, size=400)))

    pos = _place_circles_randomly(
        object_radii=radii,
        domain_shape=domain_shape,
        rng=np.random.default_rng(1),
        max_iterations=1000,
    )

    assert pos.shape == (len(radii), 2)
    assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] < nx)
    assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] < ny)

    # pairwise minimum-image distances between all placed circles
    dx = np.abs(pos[:, None, 0] - pos[None, :, 0])
    dx = np.minimum(dx, nx - dx)
    dy = np.abs(pos[:, None, 1] - pos[None, :, 1])
    dy = np.minimum(dy, ny - dy)
    dist_sq = dx**2 + dy**2
    sum_radii_sq = (radii[:, None] + radii[None, :]) ** 2
    np.fill_diagonal(dist_sq, np.inf)
    assert np.all(dist_sq > sum_radii_sq)


def test_inhibition_nn_gives_up_when_circles_do_not_fit():
    from cloudmetrics.objects.metrics.iorg import _place_circles_randomly

    with pytest.raises(Exception, match="Unable to place circles"):
        _place_circles_randomly(
            object_radii=np.array([40.0, 40.0, 40.0]),
            domain_shape=[64, 64],
            rng=np.random.default_rng(0),
            max_iterations=50,
        )
