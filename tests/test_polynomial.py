"""Tests de la classe SpeedPolynomial. Lancer avec :  pytest tests/test_polynomial.py -v"""
import numpy as np
import pytest

from vitesse import SpeedPolynomial

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0 = 70.0                            # vitesse au début [km/h]
VL = 90.0                            # vitesse à la fin [km/h]
COEFS_LIBRES_PAR_DEGRE = {           # exemples de coefficients libres c2..cd pour chaque degré
    1: [],
    2: [-30.0],
    3: [-30.0, 8.0],
    4: [-30.0, 8.0, 20.0],
}
V_LIMITE = 90.0                      # limite de vitesse testée [km/h]
VL_PIC = 80.0                        # exemple « pic puis descente » : vitesse finale [km/h]
C2_PIC = -58.0                       # ... avec c2 = -58 -> pic proche de 90 km/h
VMAX_ATTENDU = 89.93                 # pic attendu : V0 + c1² / (4·|c2|) avec c1 = 68  -> 70 + 19,93
X_VMAX_ATTENDU = 1172.0              # position du pic : ξ = c1 / (2·|c2|) = 68 / 116 = 0,586 -> 1172 m


def test_polynomial_starts_at_v0_and_ends_at_vL_for_every_degree():
    # Le coeur de la classe : quel que soit le degré, v(0) = V0 et v(L) = VL
    for degre, libres in COEFS_LIBRES_PAR_DEGRE.items():
        poly = SpeedPolynomial(LONGUEUR, V0, VL, degre, libres)
        assert poly.speed(0.0) == pytest.approx(V0)
        assert poly.speed(LONGUEUR) == pytest.approx(VL)


def test_degree_1_is_a_straight_line():
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 1)
    # au milieu du tronçon, une droite donne la moyenne de V0 et VL
    assert poly.speed(LONGUEUR / 2) == pytest.approx((V0 + VL) / 2)


def test_coefficients_degree_2():
    # À la main : c0 = V0 = 70 ; c2 = -30 (libre) ; c1 = VL - V0 - c2 = 90 - 70 + 30 = 50
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 2, [-30.0])
    np.testing.assert_allclose(poly.coefficients, [70.0, 50.0, -30.0])


def test_speed_in_the_middle_degree_2():
    # v(ξ = 0,5) = 70 + 50 × 0,5 - 30 × 0,25 = 87,5
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 2, [-30.0])
    assert poly.speed(LONGUEUR / 2) == pytest.approx(87.5)


def test_speed_accepts_an_array_of_positions():
    poly = SpeedPolynomial(LONGUEUR, V0, VL, 2, [-30.0])
    resultat = poly.speed(np.array([0.0, LONGUEUR / 2, LONGUEUR]))
    np.testing.assert_allclose(resultat, [70.0, 87.5, 90.0])


def test_wrong_number_of_free_coefficients_is_rejected():
    with pytest.raises(ValueError):
        SpeedPolynomial(LONGUEUR, V0, VL, 3, [-30.0])      # degré 3 : il faut 2 coefficients libres


def test_invalid_length_or_degree_is_rejected():
    with pytest.raises(ValueError):
        SpeedPolynomial(0.0, V0, VL, 1)
    with pytest.raises(ValueError):
        SpeedPolynomial(LONGUEUR, V0, VL, 0)


# ----------------------------- limite de vitesse -----------------------------
def test_vL_above_limit_is_rejected():
    with pytest.raises(ValueError):
        SpeedPolynomial(LONGUEUR, V0, V_LIMITE + 1, 1, speed_limit=V_LIMITE)


def test_v0_above_limit_is_rejected():
    with pytest.raises(ValueError):
        SpeedPolynomial(LONGUEUR, V_LIMITE + 1, VL_PIC, 1, speed_limit=V_LIMITE)


def test_vL_equal_to_limit_is_accepted():
    poly = SpeedPolynomial(LONGUEUR, V0, V_LIMITE, 1, speed_limit=V_LIMITE)    # VL = limite : autorisé (<=)
    assert poly.speed(LONGUEUR) == pytest.approx(V_LIMITE)


def test_no_limit_means_no_check():
    SpeedPolynomial(LONGUEUR, V0, 200.0, 1)          # speed_limit omis : aucune vérification


# ----------------------------- vitesse maximale -----------------------------
def test_max_speed_of_a_peak_then_descent():
    poly = SpeedPolynomial(LONGUEUR, V0, VL_PIC, 2, [C2_PIC], speed_limit=V_LIMITE)
    v_max, x_max = poly.max_speed()
    assert v_max == pytest.approx(VMAX_ATTENDU, abs=0.05)       # pic juste sous 90 km/h
    assert x_max == pytest.approx(X_VMAX_ATTENDU, abs=5.0)      # atteint vers 58,6 % du tronçon
    assert v_max > VL_PIC                                       # puis la courbe redescend jusqu'à VL


def test_max_speed_of_a_line_is_at_the_end():
    poly = SpeedPolynomial(LONGUEUR, V0, VL_PIC, 1)             # droite montante : le max est en x = L
    v_max, x_max = poly.max_speed()
    assert v_max == pytest.approx(VL_PIC)
    assert x_max == pytest.approx(LONGUEUR)
