"""
polynomial.py - La vraie vitesse v(x) : un polynôme qui démarre à V0 et finit à VL
===================================================================================

On travaille avec la position « normalisée »  ξ = x / L,  qui va de 0 (début du tronçon)
à 1 (fin du tronçon). Le polynôme s'écrit :

        v(ξ) = c0 + c1·ξ + c2·ξ² + ... + cd·ξ^d          (d = degré)

On veut qu'il vérifie deux conditions :
  - au début  (ξ = 0) : v = V0   ->  tous les termes avec ξ valent 0, il reste  c0 = V0
  - à la fin  (ξ = 1) : v = VL   ->  c0 + c1 + c2 + ... + cd = VL

Un polynôme de degré d a d+1 coefficients mais on n'impose que 2 conditions. Il reste donc d-1
coefficients LIBRES : c2, c3, ..., cd. Alors :
        c0 = V0
        c1 = VL - V0 - (c2 + c3 + ... + cd)        (pour que la somme fasse bien VL)

Ces d-1 coefficients libres peuvent être :
  - donnés par toi  (free_coefficients=[...]) ;
  - ou, par défaut, CHOISIS AUTOMATIQUEMENT (voir polynomial_tools.find_free_coefficients) :
    on prend le polynôme qui reste le plus près possible de la limite de vitesse (par exemple 90 km/h)
    sans la dépasser et sans accélération supérieure à une valeur réaliste.

  d = 1 : aucun coefficient libre -> une droite de V0 à VL.
  d = 2 : 1 coefficient libre (c2) -> une parabole.
  d = 3 : 2 coefficients libres (c2 et c3), etc.
"""
import numpy as np

# Les calculs eux-mêmes sont des fonctions rangées dans polynomial_tools.py (voir ce fichier)
from vitesse.polynomial_tools import (acceleration_values, coefficients_from_free,
                                      find_free_coefficients, speed_values)


class SpeedPolynomial:
    """Le polynôme de vitesse. Exemples :

        # 1) Coefficients choisis automatiquement (maximise le temps proche de la limite) :
        poly = SpeedPolynomial(length=2000, v0=70, vL=80, degree=2, speed_limit=90)
        poly.coefficients          # -> [70, 68.3, -58.3]  (c0, c1, c2)
        poly.max_speed()           # (vitesse maximale, position où elle est atteinte)

        # 2) Coefficient libre donné à la main :
        poly = SpeedPolynomial(2000, 70, 80, 2, free_coefficients=[-58])

        poly.speed(1000)           # vitesse [km/h] au milieu du tronçon (x = 1000 m)
        poly.acceleration(1000)    # accélération [m/s²] à cet endroit
    """

    def __init__(self, length, v0, vL, degree, free_coefficients=None,
                 speed_limit=None, max_acceleration_g=2.0):
        """
        length             : longueur L du tronçon [m]
        v0                 : vitesse au début du tronçon (x = 0) [km/h]
        vL                 : vitesse à la fin du tronçon (x = L) [km/h]
        degree             : degré d du polynôme (1 = droite, 2 = parabole, ...)
        free_coefficients  : liste [c2, ..., cd] de taille d-1. Si omise, ils sont calculés automatiquement
                             (il faut alors donner speed_limit, sauf pour d = 1).
        speed_limit        : limite de vitesse du tronçon [km/h]. On exige v0 <= speed_limit et vL <= speed_limit ;
                             en mode automatique, le polynôme ne dépasse jamais cette limite.
        max_acceleration_g : accélération (et freinage) maximale autorisée, en g (1 g = 9,81 m/s²).
                             Utilisée seulement pour le calcul automatique. Valeur réaliste pour une voiture
                             normale : 0,1 à 0,3 g.
        """
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        if speed_limit is not None and (v0 > speed_limit or vL > speed_limit):
            raise ValueError(f"v0 ({v0}) et vL ({vL}) doivent être <= à la limite de vitesse ({speed_limit})")

        if free_coefficients is None:
            # Pas de coefficients donnés : calcul automatique (sauf degré 1 où il n'y en a pas)
            if degree > 1 and speed_limit is None:
                raise ValueError("pour calculer automatiquement les coefficients libres il faut donner "
                                 "speed_limit (ou alors donne free_coefficients à la main)")
            free_coefficients = find_free_coefficients(length, v0, vL, degree, speed_limit, max_acceleration_g)
        if len(free_coefficients) != degree - 1:
            raise ValueError(f"degré {degree} : il faut {degree - 1} coefficient(s) libre(s) "
                             f"(c2 à c{degree}), et il y en a {len(free_coefficients)}")

        self.length = length
        self.v0 = v0
        self.vL = vL
        self.degree = degree
        self.speed_limit = speed_limit
        self.max_acceleration_g = max_acceleration_g
        self.free_coefficients = list(free_coefficients)
        # On calcule tout de suite les coefficients c0, c1, ..., cd
        self.coefficients = self.compute_coefficients()

    def compute_coefficients(self):
        """Calcule [c0, c1, ..., cd] à partir de v0, vL et des coefficients libres."""
        return coefficients_from_free(self.v0, self.vL, self.free_coefficients)

    def speed(self, x):
        """Vitesse v(x) [km/h] pour une ou plusieurs positions x en mètres (nombre ou tableau numpy)."""
        xi = np.asarray(x, dtype=float) / self.length     # position normalisée entre 0 et 1
        return speed_values(self.coefficients, xi)

    def acceleration(self, x):
        """Accélération [m/s²] du véhicule à la position x (en mètres). Négative = freinage."""
        xi = np.asarray(x, dtype=float) / self.length
        return acceleration_values(self.coefficients, xi, self.length)

    def max_speed(self, n=1000):
        """Vitesse maximale atteinte sur le tronçon, et la position où elle est atteinte.

        On évalue la courbe en n positions régulières entre 0 et L et on garde la plus grande valeur :
        c'est une valeur approchée (précision de l'ordre de L / n, soit 2 m pour L = 2000 m et n = 1000).
        Renvoie le couple (vitesse_max [km/h], position [m])."""
        x = np.linspace(0.0, self.length, n)
        v = self.speed(x)
        i = np.argmax(v)                  # indice de la plus grande valeur
        return v[i], x[i]
