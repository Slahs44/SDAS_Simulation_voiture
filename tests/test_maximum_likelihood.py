"""Tests de l'estimateur par maximum de vraisemblance. Lancer avec :  pytest tests/test_maximum_likelihood.py -v"""
import numpy as np
import pytest
from scipy.optimize import minimize

from vitesse import (MaximumLikelihood, Noise, SpeedPolynomial, create_rng, generate_positions, log_likelihood,
                     log_likelihood_variable_sigma, simulate_speeds)

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0, VL = 70.0, 80.0                  # vitesses de début et de fin [km/h]
D = 2                                # degré
COEFS_LIBRES = [-58.0]               # c2 -> c = [70, 68, -58]
SIGMA = 5.0                          # écart-type du bruit, modèle « sigma_fixe » [km/h]
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i, modèle « sigma_variable » [km/h]
N_BEAUCOUP = 5000                    # il faut beaucoup de points pour bien estimer alpha1 et alpha2
TOL_ALPHA = 0.5                      # tolérance sur alpha1, alpha2 estimés [km/h]
N_POINTS = 2000                      # beaucoup de points pour une estimation précise
N_PEU = 200                          # moins de points (comparaison de la précision)
GRAINE = 0                           # graine aléatoire
TOL_COEF = 3.0                       # tolérance sur les coefficients estimés [km/h]
TOL_SIGMA = 0.3                      # tolérance sur sigma estimé [km/h]


def donnees(n, sigma=SIGMA, graine=GRAINE):
    """Données simulées avec un sigma FIXE (modèle simple)."""
    rng, _ = create_rng(graine)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(n, LONGUEUR, rng)
    v = simulate_speeds(poly, Noise("normale", sigma, rng), x)
    return poly, x, v


def donnees_variables(n, graine=GRAINE):
    """Données simulées avec un sigma_i par voiture (notre modèle)."""
    rng, _ = create_rng(graine)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(n, LONGUEUR, rng)
    v = simulate_speeds(poly, Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2)), x)
    return poly, x, v


def fixe():
    return MaximumLikelihood(LONGUEUR, D, noise_model="sigma_fixe")


# ------------------- notre modèle : un sigma_i par voiture -------------------
def test_variable_sigma_recovers_coefficients_and_sigma_range():
    poly, x, v = donnees_variables(N_BEAUCOUP)
    est = MaximumLikelihood(LONGUEUR, D).fit(x, v)                  # "sigma_variable" par défaut
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=TOL_COEF)
    assert est.alpha1 == pytest.approx(ALPHA1, abs=TOL_ALPHA)
    assert est.alpha2 == pytest.approx(ALPHA2, abs=TOL_ALPHA)
    sigma_global = np.sqrt(((ALPHA1 + ALPHA2) / 2) ** 2 + ((ALPHA2 - ALPHA1) / 6) ** 2)
    assert est.sigma == pytest.approx(sigma_global, abs=TOL_SIGMA)


def test_variable_sigma_estimate_maximizes_its_log_likelihood():
    _, x, v = donnees_variables(N_PEU)
    est = MaximumLikelihood(LONGUEUR, D).fit(x, v)
    ll_opt = log_likelihood_variable_sigma(est.coefficients, est.alpha1, est.alpha2, x, v, LONGUEUR)
    for dc, da1, da2 in [((0, 2, -2), 0, 0), ((0, 0, 0), 0.5, 0), ((0, 0, 0), 0, 0.5), ((0, 0, 0), -0.3, -0.5)]:
        ll_autre = log_likelihood_variable_sigma(est.coefficients + np.array(dc), est.alpha1 + da1,
                                                 est.alpha2 + da2, x, v, LONGUEUR)
        assert ll_opt > ll_autre


def test_variable_sigma_fits_better_than_fixed_sigma_on_variable_data():
    # sur des données à sigma_i variables, notre modèle explique mieux les mesures que le modèle sigma fixe
    _, x, v = donnees_variables(N_BEAUCOUP)
    var = MaximumLikelihood(LONGUEUR, D).fit(x, v)
    simple = fixe().fit(x, v)
    ll_var = log_likelihood_variable_sigma(var.coefficients, var.alpha1, var.alpha2, x, v, LONGUEUR)
    ll_fixe = log_likelihood(simple.coefficients, simple.sigma, x, v, LONGUEUR)
    assert ll_var > ll_fixe


def test_variable_sigma_without_noise_is_rejected():
    _, x, v = donnees(50, sigma=0.0)
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D).fit(x, v)


def test_unknown_noise_model_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D, noise_model="modele_qui_nexiste_pas")


# ------------------- modèle simple : sigma fixe (moindres carrés) -------------------
def test_without_noise_the_polynomial_is_recovered_exactly():
    poly, x, v = donnees(50, sigma=0.0)
    est = fixe().fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=1e-6)
    assert est.sigma == pytest.approx(0.0, abs=1e-6)


def test_with_noise_coefficients_and_sigma_are_close_to_the_truth():
    poly, x, v = donnees(N_POINTS)
    est = fixe().fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=TOL_COEF)
    assert est.sigma == pytest.approx(SIGMA, abs=TOL_SIGMA)


def test_more_points_give_a_better_estimate():
    erreur = {}
    for n in (N_PEU, N_POINTS):
        poly, x, v = donnees(n)
        est = fixe().fit(x, v)
        xs = np.linspace(0, LONGUEUR, 100)
        erreur[n] = np.sqrt(np.mean((est.speed(xs) - poly.speed(xs)) ** 2))   # RMSE sur la courbe
    assert erreur[N_POINTS] < erreur[N_PEU]


def test_estimate_maximizes_the_log_likelihood():
    # On compare l'estimation exacte à une maximisation NUMERIQUE de log L (scipy) : on doit trouver la même chose.
    _, x, v = donnees(N_PEU)
    est = fixe().fit(x, v)

    def moins_log_l(p):                              # p = [c0, c1, c2, log(sigma)]  (log pour garder sigma > 0)
        return -log_likelihood(p[:-1], np.exp(p[-1]), x, v, LONGUEUR)

    depart = np.array([60.0, 10.0, 0.0, np.log(3.0)])
    res = minimize(moins_log_l, depart, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000})
    np.testing.assert_allclose(res.x[:-1], est.coefficients, atol=0.05)
    assert np.exp(res.x[-1]) == pytest.approx(est.sigma, abs=0.01)


def test_log_likelihood_is_lower_for_other_coefficients():
    _, x, v = donnees(N_PEU)
    est = fixe().fit(x, v)
    ll_opt = log_likelihood(est.coefficients, est.sigma, x, v, LONGUEUR)
    ll_autre = log_likelihood(est.coefficients + np.array([0.0, 5.0, -5.0]), est.sigma, x, v, LONGUEUR)
    assert ll_opt > ll_autre


def test_speed_before_fit_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D).speed(100.0)


def test_not_enough_points_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, 3).fit([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
