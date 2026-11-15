"""Tests de la classe Noise. Lancer avec :  pytest tests/test_noise.py -v

Principe d'un test : on fait un calcul dont on connaît le résultat, puis `assert` vérifie que le
code donne bien ce résultat. Si un assert est faux, le test est marqué FAILED."""
import numpy as np
import pytest

from vitesse import Noise

# ----------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------
SIGMA = 2.0                          # écart-type du bruit testé [m/s]
N_TIRAGES = 200_000                  # beaucoup de tirages -> les statistiques sont précises
GRAINE = 0                           # graine aléatoire (résultats reproductibles)


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
