"""
maximum_likelihood.py - Retrouver le polynôme à partir des mesures (maximum de vraisemblance, méthode du TP2)
============================================================================================================

On dispose de N mesures (x_i, s_i) et on connaît le degré n0. On cherche les coefficients c0, ..., c_n0
(et l'écart-type sigma du bruit) qui rendent les mesures « les plus probables ».

1) Le modèle (TP2, section 1.1) :

       s(x_i) = somme_n  c_n · P_n(x_i)  +  b(x_i),      b ~ Normale(0, sigma²)  (bruit blanc gaussien)

   Les P_n forment une base des polynômes de degré n0. Deux bases sont proposées :
     - "canonique" : P_n(x) = ξ^n            avec ξ = x / L       dans [0, 1]
     - "legendre"  : P_n(x) = Legendre_n(t)  avec t = 2x / L - 1  dans [-1, 1]  (base quasi-orthogonale)

2) La log-vraisemblance (TP2, section 1.2) :

       l(c) = - 1/(2·sigma²) · somme_i [ s(x_i) - somme_n c_n P_n(x_i) ]²  -  N · log(√(2π) · sigma)

3) Annuler les dérivées par rapport aux c_n donne un système linéaire :

       M · c = w      avec   M[n, m] = somme_i P_n(x_i) · P_m(x_i)      (matrice (n0+1) x (n0+1))
                             w[n]    = somme_i P_n(x_i) · s(x_i)        (vecteur noté v dans le TP)

   d'où l'estimateur du maximum de vraisemblance :   c_chapeau = M^-1 · w        (équation (1) du TP)
   et, en annulant la dérivée par rapport à sigma :   sigma_chapeau² = (1/N) · somme_i (résidu_i)².

4) Borne de Cramér-Rao (TP2, section 1.3) : l'information de Fisher vaut  I_F = M / sigma², donc

       BCR = I_F^-1 = sigma² · M^-1

   Aucun estimateur sans biais ne peut avoir une matrice de covariance plus petite. En base de Legendre avec
   des x_i uniformes, M est presque diagonale : les coefficients estimés sont presque décorrélés.

   Attention : si le bruit n'est pas gaussien (autre loi, valeurs aberrantes), ce n'est plus la bonne vraisemblance.
"""
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre

BASES = ("canonique", "legendre")


def basis_matrix(x, length, degree, basis="canonique"):
    """Matrice P de taille (N, n0+1) avec P[i, n] = P_n(x_i).

    Avec elle, toutes les vitesses du polynôme s'écrivent d'un coup :  somme_n P[i, n] · c_n = (P @ c)_i,
    écrit ci-dessous avec np.dot."""
    if basis not in BASES:
        raise ValueError(f"base inconnue : {basis!r} (choisir parmi {BASES})")
    xi = np.asarray(x, dtype=float) / length          # ξ dans [0, 1]
    if basis == "legendre":
        return legendre.legvander(2.0 * xi - 1.0, degree)   # colonne n : Legendre_n(t), t dans [-1, 1]
    P = np.zeros((len(xi), degree + 1))
    for n in range(degree + 1):
        P[:, n] = xi ** n                             # colonne n : tous les ξ_i élevés à la puissance n
    return P


def vandermonde(x, length, degree):
    """Cas particulier de la base canonique : A[i, n] = ξ_i^n (gardé pour compatibilité)."""
    return basis_matrix(x, length, degree, "canonique")


def normal_matrix(x, length, degree, basis="canonique"):
    """Matrice M du TP2 :  M[n, m] = somme_i P_n(x_i) · P_m(x_i)   (= P^T · P)."""
    P = basis_matrix(x, length, degree, basis)
    return np.dot(P.T, P)


def log_likelihood(coefficients, sigma, x, v, length, basis="canonique"):
    """log-vraisemblance l(c) des mesures (x, v) pour des coefficients (dans la base choisie) et un sigma donnés."""
    degree = len(coefficients) - 1
    P = basis_matrix(x, length, degree, basis)
    residus = np.asarray(v, dtype=float) - np.dot(P, coefficients)    # s(x_i) - somme_n c_n P_n(x_i)
    n = len(residus)
    return -np.sum(residus ** 2) / (2 * sigma ** 2) - n * np.log(np.sqrt(2 * np.pi) * sigma)


class MaximumLikelihood:
    """Estimateur du polynôme par maximum de vraisemblance (bruit blanc gaussien), méthode du TP2.

        estimateur = MaximumLikelihood(length=2000, degree=2, basis="canonique")   # ou basis="legendre"
        estimateur.fit(x, v)             # calcule c_chapeau = M^-1 · w et sigma_chapeau
        estimateur.coefficients          # [c0, ..., c_n0] estimés DANS LA BASE CHOISIE
        estimateur.canonical_coefficients  # les mêmes, convertis en base canonique (comparables à SpeedPolynomial)
        estimateur.sigma                 # écart-type du bruit estimé [km/h]
        estimateur.crb()                 # borne de Cramér-Rao  sigma² · M^-1
        estimateur.speed(1000)           # vitesse estimée en x = 1000 m

    Les n0+1 coefficients sont estimés librement : on ne suppose PAS connus V0 et VL."""

    def __init__(self, length, degree, basis="canonique"):
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        if basis not in BASES:
            raise ValueError(f"base inconnue : {basis!r} (choisir parmi {BASES})")
        self.length = length
        self.degree = degree
        self.basis = basis
        self.M = None                   # remplis par fit()
        self.w = None
        self.coefficients = None
        self.sigma = None

    def fit(self, x, v):
        """Calcule le maximum de vraisemblance à partir des positions x [m] et des vitesses mesurées v [km/h]."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(x) != len(v):
            raise ValueError("x et v doivent avoir la même taille")
        if len(x) < self.degree + 1:
            raise ValueError(f"il faut au moins {self.degree + 1} points pour estimer un polynôme de degré {self.degree}")

        P = basis_matrix(x, self.length, self.degree, self.basis)
        self.M = np.dot(P.T, P)                          # M[n, m] = somme_i P_n(x_i) P_m(x_i)
        self.w = np.dot(P.T, v)                          # w[n]    = somme_i P_n(x_i) s(x_i)
        # c_chapeau = M^-1 · w : on résout le système M c = w (même résultat que d'inverser M, mais plus précis)
        self.coefficients = np.linalg.solve(self.M, self.w)
        residus = v - np.dot(P, self.coefficients)
        self.sigma = np.sqrt(np.mean(residus ** 2))      # sigma_chapeau (division par N : c'est le vrai MV)
        return self                                      # permet d'écrire MaximumLikelihood(...).fit(x, v)

    def _check_fitted(self):
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) d'abord")

    def fisher_information(self, sigma=None):
        """Information de Fisher I_F = M / sigma²  (sigma = sigma_chapeau si non précisé)."""
        self._check_fitted()
        s = self.sigma if sigma is None else sigma
        return self.M / s ** 2

    def crb(self, sigma=None):
        """Borne de Cramér-Rao  I_F^-1 = sigma² · M^-1 : covariance minimale d'un estimateur sans biais de c."""
        self._check_fitted()
        s = self.sigma if sigma is None else sigma
        return s ** 2 * np.linalg.inv(self.M)

    @property
    def canonical_coefficients(self):
        """Coefficients en base canonique (en ξ = x/L), pour comparer à SpeedPolynomial.coefficients."""
        self._check_fitted()
        if self.basis == "canonique":
            return self.coefficients.copy()
        en_t = Polynomial(legendre.leg2poly(self.coefficients))   # polynôme en t
        en_xi = en_t(Polynomial([-1.0, 2.0]))                     # on remplace t par 2ξ - 1
        return np.pad(en_xi.coef, (0, self.degree + 1 - len(en_xi.coef)))

    def speed(self, x):
        """Vitesse estimée v_chapeau(x) [km/h] (appeler fit() d'abord)."""
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) avant speed(x)")
        P = basis_matrix(np.atleast_1d(x), self.length, self.degree, self.basis)
        return np.dot(P, self.coefficients)
