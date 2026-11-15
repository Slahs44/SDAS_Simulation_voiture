import numpy as np
import pytest

from vitesse import LegendreBasis, MonomialBasis


def test_monomial_matrix_values():
    A = MonomialBasis().design_matrix(np.array([-1.0, 0.0, 2.0]), 2)
    expected = np.array([[1, -1, 1], [1, 0, 0], [1, 2, 4]], dtype=float)
    np.testing.assert_allclose(A, expected)


def test_legendre_matrix_values():
    x = np.array([-1.0, 0.0, 0.5])
    A = LegendreBasis().design_matrix(x, 2)
    p2 = 0.5 * (3 * x**2 - 1)
    np.testing.assert_allclose(A[:, 0], 1.0)
    np.testing.assert_allclose(A[:, 1], x)
    np.testing.assert_allclose(A[:, 2], p2)


@pytest.mark.parametrize("basis", [MonomialBasis(), LegendreBasis()])
def test_shape(basis):
    assert basis.design_matrix(np.linspace(-1, 1, 7), 3).shape == (7, 4)


@pytest.mark.parametrize("basis", [MonomialBasis(), LegendreBasis()])
def test_negative_degree_rejected(basis):
    with pytest.raises(ValueError):
        basis.design_matrix(np.zeros(3), -1)


def test_legendre_better_conditioned_on_uniform_grid():
    x = np.linspace(-1, 1, 200)
    cm = np.linalg.cond(MonomialBasis().design_matrix(x, 6))
    cl = np.linalg.cond(LegendreBasis().design_matrix(x, 6))
    assert cl < cm
