"""
Performance benchmarks for the metrics in `cloudmetrics.mask`.

Run with

    pytest benchmarks --benchmark-only

These are not run as part of the normal test-suite (see `testpaths` in
`pyproject.toml`) and are used by the `ci-performance` GitHub workflow to
compare the timings of a pull-request against its base branch.
"""

import pytest

from cloudmetrics import mask as cm


def test_cloud_fraction(bench, cloud_mask):
    bench(cm.cloud_fraction, cloud_mask)


def test_num_objects(bench, cloud_mask):
    bench(cm.num_objects, cloud_mask, periodic_domain=False)


def test_mean_object_length_scale(bench, cloud_mask):
    bench(cm.mean_object_length_scale, cloud_mask, periodic_domain=False)


def test_iorg_poisson(bench, cloud_mask):
    bench(
        cm.iorg_objects,
        cloud_mask,
        periodic_domain=False,
        reference_dist="poisson",
    )


def test_iorg_inhibition_nn(bench, cloud_mask):
    bench(
        cm.iorg_objects,
        cloud_mask,
        periodic_domain=False,
        reference_dist="inhibition_nn",
        reference_dist_kwargs={"random_seed": 0},
    )


@pytest.mark.parametrize("summary_measure", ["max", "mean"])
def test_open_sky(bench, cloud_mask, summary_measure):
    bench(
        cm.open_sky,
        cloud_mask,
        summary_measure=summary_measure,
        periodic_domain=False,
    )


def test_fractal_dimension(bench, cloud_mask):
    bench(cm.fractal_dimension, cloud_mask)


def test_orientation(bench, cloud_mask):
    bench(cm.orientation, cloud_mask, periodic_domain=False)


def test_cop(bench, cloud_mask):
    # O(N^2) in the number of objects
    bench(cm.cop_objects, cloud_mask, periodic_domain=False)


def test_scai(bench, cloud_mask):
    # O(N^2) in the number of objects
    bench(cm.scai_objects, cloud_mask, periodic_domain=False)


def test_iorg_poisson_periodic(bench, small_cloud_mask):
    # single periodic-domain case, exercising the periodic object labelling
    bench(
        cm.iorg_objects,
        small_cloud_mask,
        periodic_domain=True,
        reference_dist="poisson",
    )
