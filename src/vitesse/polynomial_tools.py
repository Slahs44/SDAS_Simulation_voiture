"""
polynomial_tools.py - Fonctions (sans mémoire) pour calculer et choisir un polynôme de vitesse
==============================================================================================

Ce fichier ne contient que des FONCTIONS : elles prennent des nombres et renvoient un résultat,
sans rien mémoriser (pour cette raison ce ne sont pas des classes). La classe SpeedPolynomial
(polynomial.py) les utilise.

Rappel des conventions :
  - positions x en mètres, vitesses v en km/h ;
  - ξ = x / L est la position « normalisée » (0 au début du tronçon, 1 à la fin) ;
  - le polynôme est v(ξ) = c0 + c1·ξ + c2·ξ² + ... + cd·ξ^d.
"""
import numpy as np
from scipy.optimize import minimize   # outil de recherche du minimum d'une fonction sous contraintes

G = 9.81   # accélération de la pesanteur [m/s²] : 1 g = 9,81 m/s²


# ----------------------------------------------------------------------------------------
# 1. Calculs de base sur un polynôme
# ----------------------------------------------------------------------------------------
def coefficients_from_free(v0, vL, free_coefficients):
    """Calcule [c0, c1, ..., cd] à partir de v0, vL et des coefficients libres [c2, ..., cd].

    Conditions imposées (voir polynomial.py pour le détail) :
        v(ξ = 0) = v0   ->  c0 = v0
        v(ξ = 1) = vL   ->  c0 + c1 + c2 + ... + cd = vL   ->  c1 = vL - v0 - (c2 + ... + cd)"""
    c = np.zeros(len(free_coefficients) + 2)      # [c0, c1, c2, ..., cd]
    c[0] = v0
    c[2:] = free_coefficients
    c[1] = vL - v0 - np.sum(free_coefficients)
    return c


def speed_values(c, xi):
    """Vitesse v(ξ) [km/h] pour un tableau de positions normalisées ξ : somme des termes cj · ξ^j."""
    v = np.zeros_like(xi)
    for j, cj in enumerate(c):
        v = v + cj * xi**j
    return v


def derivative_values(c, xi):
    """Dérivée dv/dξ du polynôme : somme des termes j · cj · ξ^(j-1)  (le terme constant c0 disparaît)."""
    dv = np.zeros_like(xi)
    for j, cj in enumerate(c):
        if j >= 1:
            dv = dv + j * cj * xi ** (j - 1)
    return dv


def acceleration_values(c, xi, length):
    """Accélération du véhicule [m/s²] aux positions normalisées ξ.

    L'accélération est la dérivée de la vitesse PAR RAPPORT AU TEMPS : a = dv/dt.
    Or v dépend de la position x, et x dépend du temps (dx/dt = v). Donc
            a = dv/dt = (dv/dx) · (dx/dt) = v · dv/dx          (règle de dérivation en chaîne)
    avec  dv/dx = (dv/dξ) / L   (car ξ = x / L).
    Les conversions km/h -> m/s (diviser par 3,6) se font sur v ET sur dv/dx : d'où 3,6 × 3,6 = 12,96."""
    v = speed_values(c, xi)                       # [km/h]
    dv_dx = derivative_values(c, xi) / length     # [km/h par mètre]
    return v * dv_dx / 12.96                      # [m/s²]


# ----------------------------------------------------------------------------------------
# 2. Choix automatique des coefficients libres
# ----------------------------------------------------------------------------------------
def find_free_coefficients(length, v0, vL, degree, speed_limit, max_acceleration_g, n_grid=200):
    """Trouve les coefficients libres [c2, ..., cd] pour que le polynôme « colle » le plus possible
    à la limite de vitesse, sans la dépasser, et sans accélération trop forte.

    Le problème posé (c'est un problème d'OPTIMISATION sous contraintes) :

      On CHERCHE   les coefficients libres c2, ..., cd
      qui MINIMISENT   J = moyenne de (speed_limit - v(x))²      <- « écart moyen à la limite »
                       (J petit = la vitesse reste le plus près possible de la limite, donc on passe
                        le plus de temps possible près de 90 km/h)
      en RESPECTANT, en tous les points du tronçon :
           0 <= v(x) <= speed_limit                 (pas de vitesse négative, pas de dépassement)
           |accélération(x)| <= max_acceleration_g  (accélération et freinage réalistes)
      (et v(0) = v0, v(L) = vL, déjà garantis par la façon dont on construit les coefficients).

    Les contraintes sont vérifiées sur n_grid positions régulières. L'outil de recherche est
    scipy.optimize.minimize (méthode SLSQP). On part du polynôme le plus simple : tous les
    coefficients libres à 0, c'est-à-dire la droite de v0 à vL.

    Renvoie la liste [c2, ..., cd]. Lève une erreur si aucun polynôme ne convient."""
    n_free = degree - 1
    if n_free == 0:
        return []                                  # degré 1 : une droite, rien à optimiser

    xi = np.linspace(0.0, 1.0, n_grid)             # positions normalisées où l'on vérifie tout
    a_max = max_acceleration_g * G                 # limite d'accélération en m/s²

    def build(free):
        return coefficients_from_free(v0, vL, free)

    def cost(free):                                # J : écart quadratique moyen à la limite
        return np.mean((speed_limit - speed_values(build(free), xi)) ** 2)

    # Chaque contrainte est une fonction qui doit rester >= 0 (convention de scipy : type "ineq")
    constraints = [
        {"type": "ineq", "fun": lambda free: speed_limit - speed_values(build(free), xi)},     # v <= limite
        {"type": "ineq", "fun": lambda free: speed_values(build(free), xi)},                   # v >= 0
        {"type": "ineq", "fun": lambda free: a_max - acceleration_values(build(free), xi, length)},   # a <= a_max
        {"type": "ineq", "fun": lambda free: a_max + acceleration_values(build(free), xi, length)},   # a >= -a_max
    ]
    result = minimize(cost, x0=np.zeros(n_free), constraints=constraints, method="SLSQP",
                      options={"maxiter": 300})

    # Vérification finale : on contrôle les contraintes sur une grille plus fine (tolérances faibles)
    c = build(result.x)
    xi_fin = np.linspace(0.0, 1.0, 1000)
    v = speed_values(c, xi_fin)
    a = acceleration_values(c, xi_fin, length)
    respecte = (v.max() <= speed_limit + 0.05 and v.min() >= -0.05
                and np.abs(a).max() <= a_max * 1.01 + 1e-6)
    if not result.success or not respecte:
        raise ValueError(
            f"aucun polynôme de degré {degree} ne respecte la limite de vitesse ({speed_limit} km/h) et "
            f"l'accélération maximale ({max_acceleration_g} g) entre v0 = {v0} et vL = {vL} km/h. "
            "Essaie un autre degré ou une accélération maximale plus grande.")
    return list(result.x)
