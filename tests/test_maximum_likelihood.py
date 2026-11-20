"""Tests de l'estimateur par maximum de vraisemblance. Lancer avec :  pytest tests/test_maximum_likelihood.py -v"""
import numpy as np
import pytest
from scipy.optimize import minimize

from vitesse import (MaximumLikelihood, Noise, SpeedPolynomial, create_rng, generate_positions, log_likelihood,
                     simulate_speeds)

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
def test_variable_sigma_data_coefficients_are_recovered():
    # Méthode du TP2 appliquée à des mesures dont chaque voiture a son propre sigma_i :
    # les coefficients restent bien estimés.
    poly, x, v = donnees_variables(N_BEAUCOUP)
    est = MaximumLikelihood(LONGUEUR, D).fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=TOL_COEF)


# ------------------- modèle simple : sigma fixe (moindres carrés) -------------------
def test_without_noise_the_polynomial_is_recovered_exactly():
    poly, x, v = donnees(50, sigma=0.0)
    est = fixe().fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=1e-6)


def test_with_noise_coefficients_are_close_to_the_truth():
    poly, x, v = donnees(N_POINTS)
    est = fixe().fit(x, v)
    np.testing.assert_allclose(est.coefficients, poly.coefficients, atol=TOL_COEF)


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
    # La valeur de sigma ne change pas le maximum en c (le facteur 1/sigma² sort de la somme) : on prend SIGMA.
    _, x, v = donnees(N_PEU)
    est = fixe().fit(x, v)
    depart = np.array([60.0, 10.0, 0.0])
    res = minimize(lambda c: -log_likelihood(c, SIGMA, x, v, LONGUEUR), depart, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000})
    np.testing.assert_allclose(res.x, est.coefficients, atol=0.05)


def test_log_likelihood_is_lower_for_other_coefficients():
    _, x, v = donnees(N_PEU)
    est = fixe().fit(x, v)
    ll_opt = log_likelihood(est.coefficients, SIGMA, x, v, LONGUEUR)
    ll_autre = log_likelihood(est.coefficients + np.array([0.0, 5.0, -5.0]), SIGMA, x, v, LONGUEUR)
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


def test_crb_of_sigma_constant_needs_sigma():
    # sigma n'est pas estimé : il faut le donner pour calculer la BCR de "sigma_constant".
    _, x, v = donnees(N_PEU)
    with pytest.raises(ValueError):
        fixe().fit(x, v).crb()


def test_unknown_basis_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D, "fourier")


# ----------------------------------------------------------------------
# Estimateur "sigma_variable" : un sigma_i connu par voiture (moindres carrés pondérés par 1/sigma_i²)
# ----------------------------------------------------------------------
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i [km/h]


def donnees_sigma_variable(n, graine=GRAINE):
    rng, _ = create_rng(graine)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    x = generate_positions(n, LONGUEUR, rng)
    bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
    v = simulate_speeds(poly, bruit, x)
    return poly, x, v, bruit.last_sigmas


def test_equal_sigmas_give_the_same_estimate_as_sigma_constant():
    # Si tous les sigma_i sont égaux, les poids 1/sigma² se simplifient : on retrouve l'estimateur du TP2.
    _, x, v = donnees(N_PEU)
    for base in ("canonique", "legendre"):
        cst = MaximumLikelihood(LONGUEUR, D, base, "sigma_constant").fit(x, v)
        var = MaximumLikelihood(LONGUEUR, D, base, "sigma_variable").fit(x, v, sigmas=np.full(len(x), SIGMA))
        np.testing.assert_allclose(var.coefficients, cst.coefficients, atol=1e-8)
        np.testing.assert_allclose(var.crb(), cst.crb(sigma=SIGMA), rtol=1e-10)


def test_sigma_variable_maximizes_the_log_likelihood():
    # Maximisation NUMERIQUE de log L (avec les sigma_i connus) : on doit retrouver c_chapeau = M^-1 w.
    _, x, v, sigmas = donnees_sigma_variable(N_PEU)
    est = MaximumLikelihood(LONGUEUR, D, estimator="sigma_variable").fit(x, v, sigmas=sigmas)
    res = minimize(lambda c: -log_likelihood(c, sigmas, x, v, LONGUEUR), np.array([60.0, 10.0, 0.0]),
                   method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000})
    np.testing.assert_allclose(res.x, est.coefficients, atol=0.05)
    # ... et l'estimateur du TP2 (sigma constant) a une vraisemblance plus faible pour ces sigma_i
    cst = MaximumLikelihood(LONGUEUR, D).fit(x, v)
    assert log_likelihood(est.coefficients, sigmas, x, v, LONGUEUR) >= log_likelihood(cst.coefficients, sigmas, x, v, LONGUEUR)


def test_sigma_variable_reaches_its_crb_and_beats_sigma_constant():
    # Monte-Carlo : positions et sigma_i fixés, on retire seulement le bruit b_i ~ N(0, sigma_i²).
    _, x, _, sigmas = donnees_sigma_variable(N_PEU)
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES)
    rng, _ = create_rng(GRAINE + 1)
    est_var, est_cst = [], []
    for _ in range(2000):
        v = poly.speed(x) + rng.normal(0.0, sigmas)
        est_var.append(MaximumLikelihood(LONGUEUR, D, estimator="sigma_variable").fit(x, v, sigmas=sigmas).coefficients)
        est_cst.append(MaximumLikelihood(LONGUEUR, D).fit(x, v).coefficients)
    est_var, est_cst = np.array(est_var), np.array(est_cst)
    bcr = MaximumLikelihood(LONGUEUR, D, estimator="sigma_variable").fit(x, poly.speed(x), sigmas=sigmas).crb()
    np.testing.assert_allclose(est_var.mean(axis=0), poly.coefficients, atol=0.5)                  # sans biais
    np.testing.assert_allclose(np.diag(np.cov(est_var.T)), np.diag(bcr), rtol=0.1)                 # atteint M^-1
    np.testing.assert_allclose(est_cst.mean(axis=0), poly.coefficients, atol=0.5)                  # TP2 : sans biais aussi
    assert np.all(np.diag(np.cov(est_cst.T)) > np.diag(np.cov(est_var.T)))                         # ... mais moins précis


def test_sigma_variable_needs_valid_sigmas():
    _, x, v = donnees(N_PEU)
    est = MaximumLikelihood(LONGUEUR, D, estimator="sigma_variable")
    with pytest.raises(ValueError):
        est.fit(x, v)                                          # sigmas manquants
    with pytest.raises(ValueError):
        est.fit(x, v, sigmas=np.full(len(x) - 1, SIGMA))       # mauvaise taille
    with pytest.raises(ValueError):
        est.fit(x, v, sigmas=np.zeros(len(x)))                 # sigma_i = 0 : division par 0


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError):
        MaximumLikelihood(LONGUEUR, D, estimator="robuste")

