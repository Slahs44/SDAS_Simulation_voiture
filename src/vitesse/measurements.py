"""
measurements.py - Fabriquer des mesures simulées
================================================

Ces deux opérations n'ont pas besoin de « mémoriser » quoi que ce soit : ce sont donc de simples
FONCTIONS, pas des classes (règle pratique : une classe sert quand il y a des données à garder
entre plusieurs appels, comme les coefficients d'un polynôme).
"""
import numpy as np


def create_rng(seed=None):
    """Crée le générateur de nombres aléatoires du programme et renvoie  (rng, graine_utilisee).

    seed = un entier (ex. 42) : on utilise cette graine -> mêmes tirages à chaque lancement.
    seed = None               : on tire NOUS-MÊMES une graine au hasard, puis on l'utilise.
                                Les tirages changent à chaque lancement, mais comme on connaît la graine
                                utilisée (elle est renvoyée), on peut l'afficher et la réutiliser plus tard
                                pour regénérer exactement les mêmes données.

    (Si on écrivait simplement np.random.default_rng() sans graine, numpy choisirait une graine en
    interne et on ne pourrait plus la retrouver : c'est pour cela qu'on la tire nous-mêmes.)"""
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32))     # entier au hasard entre 0 et 4 294 967 295
    return np.random.default_rng(seed), seed


def generate_positions(n_points, length, rng):
    """Tire n_points positions au hasard (uniformément) sur le tronçon [0, length] en mètres.

    « Uniformément » = chaque endroit du tronçon a la même chance d'être choisi.
    rng : le générateur aléatoire (np.random.default_rng(graine))."""
    return rng.uniform(0.0, length, n_points)


def simulate_speeds(polynomial, noise, x):
    """Simule les vitesses mesurées aux positions x :

            vitesse mesurée = vraie vitesse v(x) + bruit

    polynomial : un SpeedPolynomial (la vraie vitesse)
    noise      : un Noise (le bruit)
    x          : tableau des positions [m]"""
    vraie_vitesse = polynomial.speed(x)
    bruit = noise.sample(len(x))          # un tirage de bruit par position
    return vraie_vitesse + bruit
