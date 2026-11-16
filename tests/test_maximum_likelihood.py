"""Tests de l'estimateur par maximum de vraisemblance. Lancer avec :  pytest tests/test_maximum_likelihood.py -v"""
import numpy as np
import pytest
from scipy.optimize import minimize

from vitesse import MaximumLikelihood, Noise, SpeedPolynomial, create_rng, generate_positions, log_likelihood, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0, VL = 70.0, 80.0                  # vitesses de début et de fin [km/h]
D = 2                                # degré
COEFS_LIBRES = [-58.0]               # c2 -> c = [70, 68, -58]
SIGMA = 5.0                          # écart-type du bruit, cas sigma fixe [km/h]
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i (un sigma_i par voiture) [km/h]
N_BEAUCOUP = 5000                    # beaucoup de points (cas sigma_i variables)
N_POINTS = 2000                      # beaucoup de points pour une estimation précise
N_PEU = 200                          # moins de points (comparaison de la précision)
GRAINE = 0                           # graine aléatoire
TOL_COEF = 3.0                       # tolérance sur les coefficients estimés [km/h]
TOL_SIGMA = 0.3                      # tolérance sur sigma estimé [km/h]


def donnees(n, sigma=SIGMA, graine=GRAINE):
    """Données simulées avec un sigma FIXE."""
    rng, _ = create_rng(graine)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(n, LONGUEUR, rng)
    v = simulate_speeds(poly, Noise("normale", sigma, rng), x)
    return poly, x, v


def donnees_variables(n, graine=GRAINE):
    """Données simulées avec un sigma_i par voiture (notre modèle de bruit)."""
    rng, _ = create_rng(graine)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(n, LONGUEUR, rng)
    v = simulate_speeds(poly, Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2)), x)
    return poly, x, v


def fixe():
    return MaximumLikelihood(LONGUEUR, D)


# ------------------- données à sigma_i variables (un par voiture) -------------------
def test_variable_sigma_data_coefficients_and_global_sigma_are_recovered():
    # Méthode du TP2 appliquée à des mesures dont chaque voiture a son propre sigma_i :
    # les coefficients restent bien estimés, et sigma_chapeau ≈ écart-type global = racine(E[sigma_i²]).
    poly, x, v = donnees_variables(N_BEAUCOUP)
    est = MaximumLikelihood(LONGUEUR, D).fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=TOL_COEF)
    sigma_global = np.sqrt(((ALPHA1 + ALPHA2) / 2) ** 2 + ((ALPHA2 - ALPHA1) / 6) ** 2)
    assert est.sigma == pytest.approx(sigma_global, abs=TOL_SIGMA)


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


def test_legendre_basis_gives_the_same_curve_as_canonical():
    # Les deux bases décrivent les mêmes polynômes : l'estimation de la courbe doit être identique.
    poly, x, v = donnees(N_PEU)
    can = MaximumLikelihood(LONGUEUR, D, "canonique").fit(x, v)
    leg = MaximumLikelihood(LONGUEUR, D, "legendre").fit(x, v)
    xs = np.linspace(0, LONGUEUR, 50)
    np.testing.assert_allclose(leg.speed(xs), can.speed(xs), atol=1e-8)
    np.testing.assert_allclose(leg.canonical_coefficients, can.coefficients, atol=1e-8)
    assert leg.sigma == pytest.approx(can.sigma)


def test_legendre_matrix_is_almost_diagonal():
    # x uniformes sur [0, L] -> t uniformes sur [-1, 1] -> M quasi diagonale (TP2, section 1.4).
    _, x, v = donnees(N_POINTS)
    for base, attendu_petit in (("legendre", True), ("canonique", False)):
        M = MaximumLikelihood(LONGUEUR, D, base).fit(x, v).M
        corr = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))   # « cosinus » entre colonnes
        hors_diag = np.abs(corr[~np.eye(D + 1, dtype=bool)]).max()
        assert (hors_diag < 0.1) == attendu_petit


def test_empirical_covariance_matches_the_crb():
    # Monte-Carlo : NB échantillons aux MEMES positions ; covariance empirique de c_chapeau ≈ sigma² M^-1.
    rng, _ = create_rng(GRAINE)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(N_PEU, LONGUEUR, rng)
    bruit = Noise("normale", SIGMA, rng)
    estimes = np.array([MaximumLikelihood(LONGUEUR, D).fit(x, simulate_speeds(poly, bruit, x)).coefficients
                        for _ in range(2000)])
    bcr = MaximumLikelihood(LONGUEUR, D).fit(x, poly.speed(x) + 1.0).crb(sigma=SIGMA)
    np.testing.assert_allclose(estimes.mean(axis=0), poly.coefficients, atol=0.5)       # sans biais
    np.testing.assert_allclose(np.diag(np.cov(estimes.T)), np.diag(bcr), rtol=0.1)      # atteint la BCR


def test_unknown_basis_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D, "fourier")
