"""
maximum_likelihood.py - Retrouver le polynôme à partir des mesures (maximum de vraisemblance)
=============================================================================================

On dispose de N mesures (x_k, v_k) et on connaît le degré d. On cherche les coefficients
c0, c1, ..., cd (et l'écart-type sigma du bruit) qui rendent les mesures « les plus probables ».

1) Le modèle :   v_k = v(x_k) + eps_k,   eps_k ~ Normale(0, sigma²),   avec v(x) = somme des cj · ξ^j.

2) La vraisemblance : c'est la densité de probabilité d'observer EXACTEMENT les mesures obtenues,
   vue comme une fonction des paramètres (c0..cd, sigma). Pour des mesures indépendantes, on multiplie
   les densités de chaque point ; avec le logarithme (plus pratique), le produit devient une somme :

       log L = - N/2 · log(2π·sigma²)  -  (1 / (2·sigma²)) · somme_k ( v_k - v(x_k) )²

3) Maximiser log L :
   - par rapport aux coefficients c : seul le dernier terme dépend de c, et il est précédé d'un « - ».
     Maximiser log L revient donc à MINIMISER  somme_k ( v_k - v(x_k) )²  (= méthode des moindres carrés).
     Pour un bruit NORMAL, maximum de vraisemblance = moindres carrés.
   - par rapport à sigma : on trouve  sigma_chapeau² = (1/N) · somme_k ( v_k - v_chapeau(x_k) )².

   Attention : si le bruit n'est pas normal (autre loi, valeurs aberrantes), ce n'est plus la bonne vraisemblance.
"""
import numpy as np


def vandermonde(x, length, degree):
    """Matrice A de taille (N, d+1) avec A[k, j] = ξ_k^j  (ξ_k = x_k / L).

    Avec elle, toutes les vitesses du polynôme s'écrivent d'un coup :  v(x_k) = somme_j A[k, j] · cj,
    c'est-à-dire le produit matrice-vecteur  A @ c  (écrit ci-dessous avec np.dot, sans symbole spécial)."""
    xi = np.asarray(x, dtype=float) / length
    A = np.zeros((len(xi), degree + 1))
    for j in range(degree + 1):
        A[:, j] = xi ** j               # colonne j : tous les ξ_k élevés à la puissance j
    return A


def log_likelihood(coefficients, sigma, x, v, length):
    """log-vraisemblance log L des mesures (x, v) pour des coefficients et un sigma donnés (formule du haut)."""
    degree = len(coefficients) - 1
    A = vandermonde(x, length, degree)
    residus = v - np.dot(A, coefficients)            # v_k - v(x_k) pour chaque point
    n = len(v)
    return -0.5 * n * np.log(2 * np.pi * sigma ** 2) - np.sum(residus ** 2) / (2 * sigma ** 2)


class MaximumLikelihood:
    """Estimateur du polynôme par maximum de vraisemblance (bruit normal).

        estimateur = MaximumLikelihood(length=2000, degree=2)
        estimateur.fit(x, v)             # calcule les coefficients et sigma
        estimateur.coefficients          # [c0, c1, ..., cd] estimés
        estimateur.sigma                 # écart-type du bruit estimé [km/h]
        estimateur.speed(1000)           # vitesse estimée en x = 1000 m

    Ici les d+1 coefficients sont estimés librement : on ne suppose PAS connus V0 et VL (on ne les connaît pas
    en vrai, ce sont justement des choses à retrouver)."""

    def __init__(self, length, degree):
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        self.length = length
        self.degree = degree
        self.coefficients = None        # remplis par fit()
        self.sigma = None

    def fit(self, x, v):
        """Calcule le maximum de vraisemblance à partir des positions x [m] et des vitesses mesurées v [km/h]."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(x) != len(v):
            raise ValueError("x et v doivent avoir la même taille")
        if len(x) < self.degree + 1:
            raise ValueError(f"il faut au moins {self.degree + 1} points pour estimer un polynôme de degré {self.degree}")

        A = vandermonde(x, self.length, self.degree)
        # Minimiser somme (v - A·c)²  : np.linalg.lstsq résout exactement ce problème de moindres carrés
        self.coefficients = np.linalg.lstsq(A, v, rcond=None)[0]
        residus = v - np.dot(A, self.coefficients)
        self.sigma = np.sqrt(np.mean(residus ** 2))       # sigma_chapeau (division par N : c'est le vrai MV)
        return self                                       # permet d'écrire MaximumLikelihood(...).fit(x, v)

    def speed(self, x):
        """Vitesse estimée v_chapeau(x) [km/h] (appeler fit() d'abord)."""
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) avant speed(x)")
        A = vandermonde(np.atleast_1d(x), self.length, self.degree)
        return np.dot(A, self.coefficients)
