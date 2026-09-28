"""
Shared fixtures for the cloudmetrics performance benchmarks, run on
deterministic synthetic cloud masks.
"""

import numpy as np
import pytest
from scipy import ndimage

import cloudmetrics


def noise_mask(shape, seed, sigma=2.5, cloud_fraction=0.3):
    """
    Thresholded gaussian-smoothed white noise: many similar-sized objects,
    evenly spread.
    """
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return field > np.quantile(field, 1.0 - cloud_fraction)


def cumulus_mask(shape, seed, slope=1.8, clustering=1.0, cloud_fraction=0.15):
    """
    Cumulus-like mask: a power-law (k^-slope) random field plus a smooth
    mesoscale field, giving clustered objects with a heavy-tailed size
    distribution.
    """
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = np.inf
    noise = rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)
    field = np.fft.irfft2(noise * k ** (-slope / 2), s=shape)
    field = ndimage.gaussian_filter(field, 1.0, mode="wrap")
    envelope = ndimage.gaussian_filter(
        rng.standard_normal(shape), 0.03 * min(shape), mode="wrap"
    )
    field = field / field.std() + clustering * envelope / envelope.std()
    return field > np.quantile(field, 1.0 - cloud_fraction)


# `n_objects` (4-connected) is asserted, so that a change in the mask
# generation (e.g. with a new numpy/scipy version) does not go unnoticed
MASK_SPECS = {
    "noise-small": dict(make=noise_mask, shape=(256, 256), seed=0, n_objects=199),
    "noise-medium": dict(make=noise_mask, shape=(512, 512), seed=1, n_objects=686),
    "cumulus-small": dict(make=cumulus_mask, shape=(256, 256), seed=0, n_objects=171),
    "cumulus-medium": dict(make=cumulus_mask, shape=(512, 512), seed=1, n_objects=607),
}


def make_mask(name):
    spec = MASK_SPECS[name]
    mask = spec["make"](shape=spec["shape"], seed=spec["seed"])
    _, n_objects = ndimage.label(mask)
    assert (
        n_objects == spec["n_objects"]
    ), f"mask '{name}' has {n_objects} objects, expected {spec['n_objects']}"
    return mask


def mask_id(name):
    spec = MASK_SPECS[name]
    ny, nx = spec["shape"]
    return f"{name}-{ny}x{nx}px-{spec['n_objects']}obj"


@pytest.fixture(
    scope="session", params=list(MASK_SPECS), ids=[mask_id(n) for n in MASK_SPECS]
)
def cloud_mask(request):
    return make_mask(request.param)


SMALL_MASKS = [n for n in MASK_SPECS if n.endswith("-small")]


@pytest.fixture(
    scope="session", params=SMALL_MASKS, ids=[mask_id(n) for n in SMALL_MASKS]
)
def small_cloud_mask(request):
    # for benchmarks that are expensive to run
    return make_mask(request.param)


@pytest.fixture
def bench(benchmark, request):
    """
    `benchmark` fixture that records the mask and cloudmetrics version and
    checks that the metric returned a finite value.
    """
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    mask_name = params.get("cloud_mask", params.get("small_cloud_mask"))
    if mask_name is not None:
        spec = MASK_SPECS[mask_name]
        benchmark.extra_info["mask"] = mask_name
        benchmark.extra_info["mask_shape"] = list(spec["shape"])
        benchmark.extra_info["mask_n_objects"] = spec["n_objects"]
    benchmark.extra_info["cloudmetrics_version"] = getattr(
        cloudmetrics, "__version__", "unknown"
    )
    benchmark.extra_info["cloudmetrics_path"] = cloudmetrics.__file__

    def run(fn, *args, **kwargs):
        result = benchmark(fn, *args, **kwargs)
        assert np.all(np.isfinite(result)), f"{fn.__name__} returned {result!r}"
        return result

    return run


def pytest_report_header(config):
    lines = [
        f"cloudmetrics: {getattr(cloudmetrics, '__version__', 'unknown')} "
        f"({cloudmetrics.__file__})",
    ]
    lines += [f"benchmark mask: {mask_id(name)}" for name in MASK_SPECS]
    return lines
