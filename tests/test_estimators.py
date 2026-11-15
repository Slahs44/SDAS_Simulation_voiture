import numpy as np
import pytest

from vitesse import (GaussianNoise, LegendreBasis, MeasurementSimulator, Measurements,
                     OLSEstimator, SpeedProfile)

TRUE = SpeedProfile([25.0, 3.0, -8.0], length=2000.0)


def noisy_data(seed=0, sigma=2.0):
    return MeasurementSimulator(TRUE, GaussianNoise(sigma), rng=np.random.default_rng(seed)).simulate()


def test_noise_free_recovers_coefficients_exactly():
    data = noisy_data(sigma=0.0)
    est = OLSEstimator(degree=2, length=2000.0).fit(data)
    np.testing.assert_allclose(est.coefficients, TRUE.coefficients, atol=1e-9)


def test_basis_change_does_not_change_the_fitted_curve():
    data = noisy_data()
    a = OLSEstimator(2, 2000.0).fit(data)
    b = OLSEstimator(2, 2000.0, basis=LegendreBasis()).fit(data)
    grid = np.linspace(0, 2000, 50)
    np.testing.assert_allclose(a(grid), b(grid), atol=1e-8)


def test_fit_is_close_to_truth_with_moderate_noise():
    est = OLSEstimator(2, 2000.0).fit(noisy_data(sigma=2.0))
    grid = np.linspace(0, 2000, 100)
    assert np.sqrt(np.mean((est(grid) - TRUE(grid)) ** 2)) < 2.0


def test_too_few_points_rejected():
    data = Measurements(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([0, 1]))
    with pytest.raises(ValueError):
        OLSEstimator(2, 10.0).fit(data)


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        OLSEstimator(-1, 10.0)
    with pytest.raises(ValueError):
        OLSEstimator(2, 0.0)
