"""
maximum_likelihood.py - Retrouver le polynôme à partir des mesures (maximum de vraisemblance, méthode du TP2)
============================================================================================================

On dispose de N mesures (x_i, v_i) et on connaît le degré d. On cherche les coefficients c0, ..., c_d
qui rendent les mesures « les plus probables ».

1) Le modèle :

       v_i = somme_n  c_n · x_i^n  +  b_i,      b_i ~ Normale(0, sigma_i²),   points indépendants

   (x_i est ici la position normalisée ξ_i = x_i / L ; en base de Legendre, x_i^n est remplacé par
    Legendre_n(t_i) avec t_i = 2 x_i / L - 1, base quasi-orthogonale.)

2) Deux estimateurs au choix (paramètre `estimator`) :

   - "sigma_constant" (TP2, section 1) : on suppose sigma_i = sigma pour tout i, sigma inconnu.
         M[n, m] = somme_i x_i^n · x_i^m          w[n] = somme_i x_i^n · v_i
         c_chapeau = M^-1 · w                      sigma_chapeau² = (1/N) · somme_i (résidu_i)²
         BCR = I_F^-1 = sigma² · M^-1

   - "sigma_variable" : chaque voiture i a son propre sigma_i, CONNU (précision de son capteur).
     La log-vraisemblance devient
         l(c) = - somme_i (v_i - somme_n c_n x_i^n)² / (2 sigma_i²)  -  somme_i log(√(2π) · sigma_i)
     et annuler ses dérivées donne le même système, mais chaque point est pondéré par 1 / sigma_i² :
         M[n, m] = somme_i x_i^n · x_i^m / sigma_i²      w[n] = somme_i x_i^n · v_i / sigma_i²
         c_chapeau = M^-1 · w                             BCR = I_F^-1 = M^-1
     (moindres carrés pondérés : une voiture précise compte plus qu'une voiture imprécise).
     Si tous les sigma_i sont égaux, on retrouve exactement l'estimateur "sigma_constant".

   Attention : si le bruit n'est pas gaussien (autre loi, valeurs aberrantes), ce n'est plus la bonne vraisemblance.
"""
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre

BASES = ("canonique", "legendre")
ESTIMATEURS = ("sigma_constant", "sigma_variable")


def basis_matrix(x, length, degree, basis="canonique"):
    """Matrice P de taille (N, d+1) avec P[i, n] = P_n(x_i)  (= ξ_i^n en base canonique).

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


def normal_matrix(x, length, degree, basis="canonique", sigmas=None):
    """Matrice M :  M[n, m] = somme_i P_n(x_i) · P_m(x_i) / sigma_i²   (sigmas=None : poids 1, cas du TP2)."""
    P = basis_matrix(x, length, degree, basis)
    poids = np.ones(len(P)) if sigmas is None else 1.0 / np.asarray(sigmas, dtype=float) ** 2
    return np.dot(P.T * poids, P)


def log_likelihood(coefficients, sigma, x, v, length, basis="canonique"):
    """log-vraisemblance l(c) des mesures (x, v).

    sigma : un nombre (même écart-type pour tous les points, cas du TP2) ou un tableau (un sigma_i par point)."""
    degree = len(coefficients) - 1
    P = basis_matrix(x, length, degree, basis)
    residus = np.asarray(v, dtype=float) - np.dot(P, coefficients)    # v_i - somme_n c_n P_n(x_i)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), residus.shape)
    return -np.sum(residus ** 2 / (2 * s ** 2)) - np.sum(np.log(np.sqrt(2 * np.pi) * s))


class MaximumLikelihood:
    """Estimateur du polynôme par maximum de vraisemblance (bruit gaussien), méthode du TP2.

        estimateur = MaximumLikelihood(length=2000, degree=2, basis="canonique", estimator="sigma_constant")
        estimateur.fit(x, v)                     # "sigma_constant" : c_chapeau = M^-1 · w et sigma_chapeau
        estimateur.fit(x, v, sigmas=sigmas)      # "sigma_variable" : sigma_i connus, moindres carrés pondérés
        estimateur.coefficients                  # [c0, ..., c_d] estimés DANS LA BASE CHOISIE
        estimateur.canonical_coefficients        # les mêmes, convertis en base canonique (comparables à SpeedPolynomial)
        estimateur.sigma                         # "sigma_constant" : écart-type estimé [km/h] ; "sigma_variable" : None
        estimateur.crb()                         # borne de Cramér-Rao
        estimateur.speed(1000)                   # vitesse estimée en x = 1000 m

    Les d+1 coefficients sont estimés librement : on ne suppose PAS connus V0 et VL."""

    def __init__(self, length, degree, basis="canonique", estimator="sigma_constant"):
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        if basis not in BASES:
            raise ValueError(f"base inconnue : {basis!r} (choisir parmi {BASES})")
        if estimator not in ESTIMATEURS:
            raise ValueError(f"estimateur inconnu : {estimator!r} (choisir parmi {ESTIMATEURS})")
        self.length = length
        self.degree = degree
        self.basis = basis
        self.estimator = estimator
        self.M = None                   # remplis par fit()
        self.w = None
        self.coefficients = None
        self.sigma = None               # sigma_chapeau ("sigma_constant" seulement)
        self.sigmas = None              # les sigma_i utilisés ("sigma_variable" seulement)

    def fit(self, x, v, sigmas=None):
        """Calcule le maximum de vraisemblance à partir des positions x [m] et des vitesses mesurées v [km/h].

        sigmas : écarts-types sigma_i [km/h] de chaque point, obligatoires pour "sigma_variable"
                 (ignorés pour "sigma_constant")."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(x) != len(v):
            raise ValueError("x et v doivent avoir la même taille")
        if len(x) < self.degree + 1:
            raise ValueError(f"il faut au moins {self.degree + 1} points pour estimer un polynôme de degré {self.degree}")

        if self.estimator == "sigma_variable":
            if sigmas is None:
                raise ValueError("l'estimateur 'sigma_variable' a besoin des sigma_i : fit(x, v, sigmas=...)")
            sigmas = np.asarray(sigmas, dtype=float)
            if sigmas.shape != x.shape:
                raise ValueError("sigmas doit avoir la même taille que x")
            if np.any(sigmas <= 0):
                raise ValueError("tous les sigma_i doivent être > 0 (on divise par sigma_i²)")
            poids = 1.0 / sigmas ** 2                    # une voiture précise (petit sigma_i) pèse plus
        else:
            poids = np.ones(len(x))                      # TP2 : tous les points ont le même poids

        P = basis_matrix(x, self.length, self.degree, self.basis)
        self.M = np.dot(P.T * poids, P)                  # M[n, m] = somme_i P_n(x_i) P_m(x_i) / sigma_i²
        self.w = np.dot(P.T * poids, v)                  # w[n]    = somme_i P_n(x_i) v_i / sigma_i²
        # c_chapeau = M^-1 · w : on résout le système M c = w (même résultat que d'inverser M, mais plus précis)
        self.coefficients = np.linalg.solve(self.M, self.w)

        if self.estimator == "sigma_variable":
            self.sigmas = sigmas
            self.sigma = None                            # rien à estimer : les sigma_i sont connus
        else:
            residus = v - np.dot(P, self.coefficients)
            self.sigma = np.sqrt(np.mean(residus ** 2))  # sigma_chapeau (division par N : c'est le vrai MV)
            self.sigmas = None
        return self                                      # permet d'écrire MaximumLikelihood(...).fit(x, v)

    def _check_fitted(self):
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) d'abord")

    def fisher_information(self, sigma=None):
        """Information de Fisher.
        "sigma_constant" : I_F = M / sigma²  (sigma = sigma_chapeau si non précisé) ;
        "sigma_variable" : I_F = M  (les 1/sigma_i² sont déjà dans M ; l'argument sigma est ignoré)."""
        self._check_fitted()
        if self.estimator == "sigma_variable":
            return self.M.copy()
        s = self.sigma if sigma is None else sigma
        return self.M / s ** 2

    def crb(self, sigma=None):
        """Borne de Cramér-Rao I_F^-1 : covariance minimale d'un estimateur sans biais de c.
        "sigma_constant" : sigma² · M^-1 ;  "sigma_variable" : M^-1."""
        return np.linalg.inv(self.fisher_information(sigma))

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
