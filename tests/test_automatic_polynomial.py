"""Tests du calcul AUTOMATIQUE des coefficients libres (polynomial_tools.py) et de l'accélération.
Lancer avec :  pytest tests/test_automatic_polynomial.py -v"""
import numpy as np
import pytest

from vitesse import SpeedPolynomial

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0 = 70.0                            # vitesse au début [km/h]
VL = 80.0                            # vitesse à la fin [km/h]
V_LIMITE = 90.0                      # limite de vitesse [km/h]
A_LACHE_G = 2.0                      # accélération maximale « large » : ne gêne pas [g]
A_SERREE_G = 0.01                    # accélération maximale « serrée » : elle devient active [g]
A_IMPOSSIBLE_G = 0.001               # accélération maximale impossible à respecter entre V0 et VL [g]
G = 9.81                             # 1 g en m/s²
TOL_V = 0.05                         # tolérance sur la vitesse [km/h]
C2_ATTENDU = -58.28                  # à la main (d = 2) : (VL-V0+a)² = 4·a·(V_LIMITE-V0)  ->  a = 30 + √800 = 58,28
N_GRILLE = 500                       # positions où l'on vérifie les propriétés
DEGRES = [2, 3, 4]                   # degrés testés en mode automatique


def auto(degre, a_max_g=A_LACHE_G):
    return SpeedPolynomial(LONGUEUR, V0, VL, degre, speed_limit=V_LIMITE, max_acceleration_g=a_max_g)


def ecart_moyen_a_la_limite(poly):
    """J = moyenne de (limite - v)² : plus il est petit, plus la courbe colle à la limite."""
    x = np.linspace(0, LONGUEUR, N_GRILLE)
    return np.mean((V_LIMITE - poly.speed(x)) ** 2)


def test_degree_1_needs_neither_limit_nor_optimization():
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 1)
    np.testing.assert_allclose(poly.coefficients, [V0, VL - V0])


def test_automatic_mode_requires_a_speed_limit():
    with pytest.raises(ValueError):
        SpeedPolynomial(LONGUEUR, V0, VL, 2)         # ni coefficients libres, ni limite : impossible de choisir


def test_manual_free_coefficients_still_work_without_limit():
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 2, free_coefficients=[-58.0])
    assert poly.speed(0.0) == pytest.approx(V0) and poly.speed(LONGUEUR) == pytest.approx(VL)


def test_automatic_polynomial_starts_at_v0_and_ends_at_vL():
    for d in DEGRES:
        poly = auto(d)
        assert poly.speed(0.0) == pytest.approx(V0)
        assert poly.speed(LONGUEUR) == pytest.approx(VL)


def test_automatic_polynomial_never_exceeds_the_speed_limit():
    x = np.linspace(0, LONGUEUR, N_GRILLE)
    for d in DEGRES:
        v = auto(d).speed(x)
        assert v.max() <= V_LIMITE + TOL_V         # jamais au-dessus de la limite
        assert v.min() >= -TOL_V                   # jamais de vitesse négative


def test_automatic_degree_2_matches_the_hand_calculation():
    poly = auto(2)
    assert poly.free_coefficients[0] == pytest.approx(C2_ATTENDU, abs=0.3)
    v_max, _ = poly.max_speed()
    assert v_max == pytest.approx(V_LIMITE, abs=TOL_V)      # le pic touche la limite


def test_automatic_is_closer_to_the_limit_than_other_choices():
    auto_j = ecart_moyen_a_la_limite(auto(2))
    droite = SpeedPolynomial(LONGUEUR, V0, VL, 1)
    autre = SpeedPolynomial(LONGUEUR, V0, VL, 2, free_coefficients=[-30.0])
    assert auto_j < ecart_moyen_a_la_limite(droite)
    assert auto_j < ecart_moyen_a_la_limite(autre)


def test_higher_degree_sticks_closer_to_the_limit():
    assert ecart_moyen_a_la_limite(auto(4)) < ecart_moyen_a_la_limite(auto(2))


# ----------------------------- accélération -----------------------------
def test_acceleration_formula_for_a_line():
    # Droite de 70 à 80 km/h sur 2000 m : dv/dx = 10/2000 km/h par m ; en x = 0 : a = v · dv/dx / 12,96
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 1)
    assert poly.acceleration(0.0) == pytest.approx(70.0 * (10.0 / 2000.0) / 12.96)


def test_loose_acceleration_limit_does_not_change_the_result():
    np.testing.assert_allclose(auto(2, A_LACHE_G).coefficients, auto(2, 10 * A_LACHE_G).coefficients, atol=0.1)


def test_tight_acceleration_limit_is_respected_and_active():
    poly = auto(2, A_SERREE_G)
    a = np.abs(poly.acceleration(np.linspace(0, LONGUEUR, N_GRILLE)))
    assert a.max() <= A_SERREE_G * G * 1.01                 # respectée
    assert a.max() >= A_SERREE_G * G * 0.95                 # et réellement atteinte : la contrainte est active
    assert auto(2, A_SERREE_G).max_speed()[0] < V_LIMITE - 1  # conséquence : le pic n'atteint plus la limite


def test_impossible_acceleration_limit_is_rejected():
    with pytest.raises(ValueError):
        auto(2, A_IMPOSSIBLE_G)
