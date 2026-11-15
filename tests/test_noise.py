import numpy as np
import pytest

from vitesse import ContaminatedGaussianNoise, GaussianNoise, VehicleEffectNoise


def test_gaussian_statistics():
    eps = GaussianNoise(2.0).sample(np.random.default_rng(0), 200_000)
    assert abs(eps.mean()) < 0.02
    assert eps.std() == pytest.approx(2.0, rel=0.01)


def test_gaussian_reproducible_with_seed():
    a = GaussianNoise(1.0).sample(np.random.default_rng(7), 10)
    b = GaussianNoise(1.0).sample(np.random.default_rng(7), 10)
    np.testing.assert_array_equal(a, b)


def test_zero_sigma_gives_zero_noise():
    np.testing.assert_array_equal(GaussianNoise(0.0).sample(np.random.default_rng(0), 5), 0.0)


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        GaussianNoise(-1.0)


def test_contaminated_variance_is_mixture():
    n = ContaminatedGaussianNoise(sigma=1.0, outlier_rate=0.1, outlier_sigma=10.0)
    eps = n.sample(np.random.default_rng(1), 500_000)
    expected_var = 0.9 * 1.0**2 + 0.1 * 10.0**2
    assert eps.var() == pytest.approx(expected_var, rel=0.03)


def test_contaminated_invalid_rate():
    with pytest.raises(ValueError):
        ContaminatedGaussianNoise(1.0, 1.5, 5.0)


def test_vehicle_effect_is_shared_within_vehicle():
    ids = np.repeat(np.arange(2000), 2)           # chaque vehicule envoie 2 fois
    eps = VehicleEffectNoise(sigma=0.1, sigma_vehicle=3.0).sample(np.random.default_rng(2), ids.size, ids)
    first, second = eps[0::2], eps[1::2]
    assert np.corrcoef(first, second)[0, 1] > 0.9  # corr theorique ~ 9 / 9.01


def test_vehicle_effect_requires_ids():
    with pytest.raises(ValueError):
        VehicleEffectNoise(1.0, 1.0).sample(np.random.default_rng(0), 5)
