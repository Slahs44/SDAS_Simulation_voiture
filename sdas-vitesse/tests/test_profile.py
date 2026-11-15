import numpy as np
import pytest

from vitesse import LegendreBasis, SpeedProfile


def test_degree_and_evaluation_monomial():
    p = SpeedProfile([25.0, 3.0, -8.0], length=2000.0)
    assert p.degree == 2
    # x = 0 -> xi = -1 ; x = L/2 -> xi = 0 ; x = L -> xi = 1
    np.testing.assert_allclose(p([0.0, 1000.0, 2000.0]), [25 - 3 - 8, 25.0, 25 + 3 - 8])


def test_normalise_bounds():
    p = SpeedProfile([1.0], length=500.0)
    np.testing.assert_allclose(p.normalise([0.0, 250.0, 500.0]), [-1.0, 0.0, 1.0])


def test_legendre_profile_matches_manual_formula():
    p = SpeedProfile([10.0, 2.0, 3.0], length=100.0, basis=LegendreBasis())
    xi = p.normalise(np.array([20.0]))
    manual = 10.0 + 2.0 * xi + 3.0 * 0.5 * (3 * xi**2 - 1)
    np.testing.assert_allclose(p([20.0]), manual)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SpeedProfile([], length=10.0)
    with pytest.raises(ValueError):
        SpeedProfile([1.0], length=0.0)
