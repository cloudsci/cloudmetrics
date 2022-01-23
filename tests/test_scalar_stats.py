import numpy as np
import pytest
import scipy as sp

import cloudmetrics


def _get_stats_func(measure):
    if hasattr(np, measure):
        fn = getattr(np, measure)
    elif hasattr(sp.stats, measure):
        fn = getattr(sp.stats, measure)
    else:
        raise NotImplementedError(measure)
    return fn


def _generate_example(measure, N=10, Nc=10):
    mask = np.zeros((N, N)).astype(bool)
    scalar_field = np.random.random((N, N))

    scalar_values = np.random.random(Nc)
    i, j = np.random.randint(low=0, high=N, size=(2, Nc))

    mask[i, j] = True
    scalar_field[i, j] = scalar_values

    fn = _get_stats_func(measure=measure)
    value_field = fn(scalar_field.flatten())
    value_masked = fn(scalar_field[mask])

    # keep trying until we make a field where we don't get the same value for
    # applying the reduction to the whole field and the masked field
    if value_field == value_masked:
        return _generate_example(measure=measure, N=N, Nc=Nc)

    return mask, scalar_field, value_field, value_masked


@pytest.mark.parametrize("measure", "mean var skew kurtosis".split())
def test_masked_statistic(measure):
    mask, scalar_field, value_field_true, value_masked_true = _generate_example(
        measure=measure
    )
    fn_cm = getattr(cloudmetrics.scalar, measure)

    assert fn_cm(scalar_field.flatten()) == value_field_true
    assert fn_cm(scalar_field, mask=mask) == value_masked_true
