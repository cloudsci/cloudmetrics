import numpy as np
import pytest

import cloudmetrics


@pytest.mark.parametrize("periodic_domain", [True])
def test_spectral_noise(periodic_domain):
    """
    1. White noise -> equal amplitude in all wavenumbers:
        - variance in radial 1d psd should be amp**2*pi/4
        - azimuthal spectrum should be approximately uniform over sectors, so
          anisotropy should -> 0
        - radial spectrum (integrated over rings) should increase linearly
          with k1d, i.e.:
         - spectral slopes -> 1
         - median l_spec: triangular spectrum -> 1/3 of variance is encountered at
           sqrt(3)/3 of max wavenumber N/2 -> spectral length scale should be
           1/(sqrt(3)/3*N/2*1/L), or more simply 2*sqrt(3)*dx
         - mean l_spec should then occur at wavenumber 2/3*kmax, or l_spec -> 3*dx
    """

    amp = 1
    dx = 1000
    sh = 512

    rng = np.random.default_rng(0)
    cloud_scalar = rng.normal(0, amp, size=sh * sh).reshape((sh, sh))

    k1d, psd_1d_rad, psd_1d_azi = cloudmetrics.scalar.compute_spectra(
        cloud_scalar,
        dx=dx,
        periodic_domain=periodic_domain,
        apply_detrending=False,
        window=None,
    )

    variance_psd = np.sum(psd_1d_rad) * 2 * np.pi / (dx * sh)
    anisotropy = cloudmetrics.scalar.spectral_anisotropy(psd_1d_azi)
    beta = cloudmetrics.scalar.spectral_slope(k1d, psd_1d_rad)
    beta_binned = cloudmetrics.scalar.spectral_slope_binned(k1d, psd_1d_rad)
    l_spec_median = cloudmetrics.scalar.spectral_length_median(k1d, psd_1d_rad)
    l_spec_moment = cloudmetrics.scalar.spectral_length_moment(k1d, psd_1d_rad)

    np.testing.assert_allclose(variance_psd, amp**2 * np.pi / 4, 0.01)
    np.testing.assert_allclose(anisotropy, 0.0, atol=0.1)
    np.testing.assert_allclose(beta, 1, atol=0.01)
    np.testing.assert_allclose(beta_binned, 1, atol=0.1)
    np.testing.assert_allclose(l_spec_median, 2 * np.sqrt(3) * dx, atol=100)
    np.testing.assert_allclose(l_spec_moment, 3 * dx, atol=10)


@pytest.mark.parametrize("periodic_domain", [True, False])
@pytest.mark.parametrize("apply_detrending", [True, False])
@pytest.mark.parametrize("window", [None, "Planck", "Welch", "Hann"])
def test_spectral_if_runs(periodic_domain, apply_detrending, window):

    amp = 1
    dx = 1000
    sh = 512

    rng = np.random.default_rng(0)
    cloud_scalar = rng.normal(0, amp, size=sh * sh).reshape((sh, sh))

    k1d, psd_1d_rad, psd_1d_azi = cloudmetrics.scalar.compute_spectra(
        cloud_scalar,
        dx=dx,
        periodic_domain=periodic_domain,
        apply_detrending=apply_detrending,
        window=window,
    )
