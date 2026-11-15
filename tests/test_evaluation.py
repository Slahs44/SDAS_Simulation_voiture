import numpy as np
import pytest

from vitesse import (ContaminatedGaussianNoise, GaussianNoise, MeasurementSimulator,
                     MonteCarlo, OLSEstimator, SpeedProfile, cramer_rao_covariance)

TRUE = SpeedProfile([25.0, 3.0, -8.0], length=2000.0)
GRID = np.linspace(0, 2000, 80)


def monte_carlo(noise, degree=2, n_runs=3000, seed=0):
    sim = MeasurementSimulator(TRUE, noise, rng=np.random.default_rng(seed))
    return MonteCarlo(sim, OLSEstimator(degree, 2000.0)).run(n_runs, GRID), sim


def test_cramer_rao_formula():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(cramer_rao_covariance(A, 2.0), 4.0 * np.linalg.inv(A.T @ A))


def test_ols_unbiased_under_gaussian_noise():
    res, _ = monte_carlo(GaussianNoise(2.0))
    assert np.all(np.abs(res.coefficient_bias) < 0.1)


def test_ols_covariance_matches_cramer_rao():
    """Positions fixees + bruit gaussien : MCO efficace, covariance empirique ~ CRB."""
    res, _ = monte_carlo(GaussianNoise(2.0), n_runs=5000, seed=0)
    # MonteCarlo.run tire la geometrie en premier : meme graine -> memes positions
    x_used, _ = MeasurementSimulator(
        TRUE, GaussianNoise(2.0), rng=np.random.default_rng(0)).draw_geometry()
    A = OLSEstimator(2, 2000.0).design_matrix(x_used)
    crb = cramer_rao_covariance(A, 2.0)
    np.testing.assert_allclose(np.diag(res.coefficient_covariance), np.diag(crb), rtol=0.1)


def test_outliers_degrade_ols():
    clean, _ = monte_carlo(GaussianNoise(2.0), n_runs=1000)
    dirty, _ = monte_carlo(ContaminatedGaussianNoise(2.0, 0.1, 20.0), n_runs=1000)
    assert dirty.rmse_global > 2 * clean.rmse_global


def test_underfitting_creates_bias_overfitting_does_not():
    under, _ = monte_carlo(GaussianNoise(1.0), degree=1, n_runs=500)
    over, _ = monte_carlo(GaussianNoise(1.0), degree=3, n_runs=500)
    assert np.abs(under.bias_profile).max() > 1.0
    assert np.abs(over.bias_profile).max() < 0.3


def test_coefficient_bias_none_when_degree_differs():
    res, _ = monte_carlo(GaussianNoise(1.0), degree=3, n_runs=50)
    assert res.coefficient_bias is None
