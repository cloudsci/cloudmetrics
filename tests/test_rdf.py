import numpy as np
import rdfpy

import cloudmetrics
import cloudmetrics.objects.metrics.rdf
from cloudmetrics.utils.synthetic_data import modified_poisson_disk_sampling


def test_rfd():
    # create random particle coordinates in a 20x20 plane
    N = 100
    r0 = 10.0  # distance between points
    _, coords, _ = modified_poisson_disk_sampling(N=N, r0=r0)
    # coords = np.random.uniform(0.0, N, size=(2500, 2))

    dr = 0.1
    dist_cutoff = 0.9
    g_r, radii = rdfpy.rdf(coords, dr=dr, rcutoff=dist_cutoff)
    g_r__cm, radii__cm = cloudmetrics.objects.metrics.rdf.pair_correlation_2d(
        pos=coords,
        dist_cutoff=dist_cutoff,
        dr=dr,
        periodic_domain=True,
        domain_shape=(N, N),
    )

    g_r__cm = g_r__cm[:-1]
    radii__cm = radii__cm[:-1] + dr / 2.0

    np.testing.assert_allclose(radii, radii__cm)
    np.testing.assert_allclose(g_r, g_r__cm)
