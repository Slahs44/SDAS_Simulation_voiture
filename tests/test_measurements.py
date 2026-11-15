"""Tests des fonctions de measurements.py. Lancer avec :  pytest tests/test_measurements.py -v"""
import numpy as np
import pytest

from vitesse import Noise, SpeedPolynomial, generate_positions, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0, VL = 20.0, 25.0                  # vitesses de début et de fin [m/s]
D, COEFS_LIBRES = 2, [-8.0]          # degré et coefficients libres du polynôme
SIGMA = 2.0                          # écart-type du bruit [m/s]
N_POINTS = 100                       # nombre de points simulés
GRAINE = 0                           # graine aléatoire


def test_positions_have_right_count_and_stay_on_the_segment():
    x = generate_positions(N_POINTS, LONGUEUR, np.random.default_rng(GRAINE))
    assert len(x) == N_POINTS
    assert x.min() >= 0.0 and x.max() <= LONGUEUR


def test_without_noise_measured_speed_equals_true_speed():
    rng = np.random.default_rng(GRAINE)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    v = simulate_speeds(poly, Noise("normale", 0.0, rng), x)    # sigma = 0 : pas de bruit
    np.testing.assert_allclose(v, poly.speed(x))


def test_noise_is_centered_around_true_speed():
    rng = np.random.default_rng(GRAINE)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(100_000, LONGUEUR, rng)
    ecart = simulate_speeds(poly, Noise("normale", SIGMA, rng), x) - poly.speed(x)
    assert abs(ecart.mean()) < 0.05                          # écart moyen ≈ 0
    assert ecart.std() == pytest.approx(SIGMA, rel=0.02)     # dispersion ≈ SIGMA
