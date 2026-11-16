"""Tests de la classe Noise. Lancer avec :  pytest tests/test_noise.py -v

Principe d'un test : on fait un calcul dont on connaît le résultat, puis `assert` vérifie que le
code donne bien ce résultat. Si un assert est faux, le test est marqué FAILED."""
import numpy as np
import pytest

from vitesse import Noise

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
SIGMA = 2.0                          # écart-type du bruit testé [km/h]
N_TIRAGES = 200_000                  # beaucoup de tirages -> les statistiques sont précises
GRAINE = 0                           # graine aléatoire (résultats reproductibles)
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i (un écart-type par voiture) [km/h]


def test_sample_returns_n_values():
    bruit = Noise("normale", SIGMA, np.random.default_rng(GRAINE))
    assert len(bruit.sample(50)) == 50


def test_normal_noise_has_mean_zero_and_std_sigma():
    valeurs = Noise("normale", SIGMA, np.random.default_rng(GRAINE)).sample(N_TIRAGES)
    assert abs(valeurs.mean()) < 0.02                       # moyenne proche de 0
    assert valeurs.std() == pytest.approx(SIGMA, rel=0.01)  # écart-type proche de SIGMA (à 1 % près)


def test_same_seed_gives_same_values():
    a = Noise("normale", SIGMA, np.random.default_rng(7)).sample(10)
    b = Noise("normale", SIGMA, np.random.default_rng(7)).sample(10)
    np.testing.assert_array_equal(a, b)                     # exactement les mêmes nombres


def test_zero_sigma_gives_no_noise():
    valeurs = Noise("normale", 0.0, np.random.default_rng(GRAINE)).sample(5)
    np.testing.assert_array_equal(valeurs, 0.0)


def test_unknown_law_is_rejected():
    with pytest.raises(ValueError):          # on s'attend à ce que le code lève une erreur
        Noise("loi_qui_nexiste_pas", SIGMA)


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        Noise("normale", -1.0)


# ------------------- sigma_i par voiture (sigma_range) -------------------
def bruit_variable(graine=GRAINE, plage=(ALPHA1, ALPHA2)):
    return Noise("normale", rng=np.random.default_rng(graine), sigma_range=plage)


def test_sigmas_follow_the_3_sigma_rule_and_stay_in_the_range():
    sigmas = bruit_variable().sample_sigmas(N_TIRAGES)
    assert sigmas.mean() == pytest.approx((ALPHA1 + ALPHA2) / 2, rel=0.01)       # moyenne au milieu
    assert sigmas.std() == pytest.approx((ALPHA2 - ALPHA1) / 6, rel=0.02)        # 6 écarts-types entre alpha1 et alpha2
    assert sigmas.min() >= ALPHA1 and sigmas.max() <= ALPHA2                     # toujours dans [alpha1, alpha2]


def test_out_of_range_sigmas_are_set_to_the_nearest_bound():
    sigmas = bruit_variable().sample_sigmas(N_TIRAGES)
    # environ 0,135 % des tirages sortent de chaque côté et sont ramenés exactement sur la borne
    assert np.mean(sigmas == ALPHA1) == pytest.approx(0.00135, abs=0.0005)
    assert np.mean(sigmas == ALPHA2) == pytest.approx(0.00135, abs=0.0005)


def test_variable_sigma_noise_uses_one_sigma_per_point():
    bruit = bruit_variable()
    valeurs = bruit.sample(N_TIRAGES)
    assert len(bruit.last_sigmas) == N_TIRAGES                                   # un sigma_i par point
    assert abs(valeurs.mean()) < 0.05                                            # bruit centré
    assert valeurs.var() == pytest.approx(np.mean(bruit.last_sigmas ** 2), rel=0.02)   # variance = moyenne des sigma_i²
    grands = bruit.last_sigmas > (ALPHA1 + ALPHA2) / 2
    assert valeurs[grands].std() > valeurs[~grands].std()                        # grand sigma_i -> plus dispersé


def test_variable_sigma_same_seed_gives_same_values():
    np.testing.assert_array_equal(bruit_variable(7).sample(10), bruit_variable(7).sample(10))


def test_bad_sigma_range_is_rejected():
    with pytest.raises(ValueError):
        Noise("normale", sigma_range=(5.0, 2.0))          # alpha1 > alpha2
    with pytest.raises(ValueError):
        Noise("normale", sigma_range=(-1.0, 2.0))         # alpha1 négatif
