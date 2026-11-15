"""Tests des fonctions de measurements.py. Lancer avec :  pytest tests/test_measurements.py -v"""
import numpy as np
import pytest

from vitesse import Noise, SpeedPolynomial, create_rng, generate_positions, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0, VL = 70.0, 80.0                  # vitesses de début et de fin [km/h]
D, COEFS_LIBRES = 2, [-58.0]         # degré et coefficients libres du polynôme
SIGMA = 5.0                          # écart-type du bruit [km/h]
N_POINTS = 100                       # nombre de points simulés
GRAINE = 0                           # graine aléatoire
GRAINE_EXEMPLE = 42                  # graine de l'exemple « reproductible »
N_APPELS = 5                         # nombre d'appels sans graine (test « les graines diffèrent »)


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


# ----------------------------- la graine -----------------------------
def test_create_rng_with_a_seed_is_reproducible():
    rng1, graine1 = create_rng(GRAINE_EXEMPLE)
    rng2, graine2 = create_rng(GRAINE_EXEMPLE)
    assert graine1 == graine2 == GRAINE_EXEMPLE                       # la graine renvoyée est celle qu'on a donnée
    np.testing.assert_array_equal(rng1.uniform(0, 1, 10), rng2.uniform(0, 1, 10))   # mêmes nombres


def test_create_rng_without_seed_returns_a_seed_that_regenerates_the_same_numbers():
    rng, graine = create_rng(None)                                     # tirage aléatoire
    assert isinstance(graine, int) and 0 <= graine < 2**32            # une graine entière est bien renvoyée
    premiers = rng.uniform(0, 1, 10)
    rng_bis, _ = create_rng(graine)                                    # on repart de la graine affichée
    np.testing.assert_array_equal(premiers, rng_bis.uniform(0, 1, 10)) # -> exactement les mêmes nombres


def test_create_rng_without_seed_gives_different_seeds():
    graines = {create_rng(None)[1] for _ in range(N_APPELS)}
    assert len(graines) > 1                                            # (probabilité d'échec ~ 1e-38)
