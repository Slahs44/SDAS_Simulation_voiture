"""
maximum_likelihood.py - Retrouver le polynôme à partir des mesures (maximum de vraisemblance, méthode du TP2)
============================================================================================================

On dispose de N mesures (x_i, v_i) et on connaît le degré n0. On cherche les coefficients c_0, ..., c_n0
qui rendent les mesures « les plus probables ». Notations et démonstration : docs/explications.md, section 5.

1) Le modèle (P_0, ..., P_n0 : base de polynômes ; c* : vrais coefficients ; c : coefficients candidats) :

       mu_i(c) = somme_n  c_n · P_n(x_i)
       v_i = mu_i(c*) + b_i,      b_i ~ Normale(0, sigma_i²),   b_1, ..., b_N indépendants

   Deux bases (paramètre `basis`), calculées sur la position normalisée pour éviter des nombres énormes :
     - "canonique" : P_n(x) = ξ^n            avec ξ = x / L       dans [0, 1]
     - "legendre"  : P_n(x) = Legendre_n(t)  avec t = 2x / L - 1  dans [-1, 1]  (base quasi-orthogonale)

2) La log-vraisemblance (le ℓ(χ | c) du TP, avec v = χ) :

       ℓ(v | c) = - somme_i (v_i - mu_i(c))² / (2 sigma_i²)  -  somme_i ln(√(2π) · sigma_i)

3) Deux estimateurs au choix (paramètre `estimator`) :

   - "sigma_variable" : chaque voiture i a son propre sigma_i, CONNU (précision de son capteur).
     Annuler les dérivées par rapport aux c_n donne le système M_tilde · c = w_tilde, chaque point étant
     pondéré par 1 / sigma_i² (moindres carrés pondérés : une voiture précise compte plus) :
         M_tilde[n, m] = somme_i P_n(x_i) · P_m(x_i) / sigma_i²      w_tilde[n] = somme_i P_n(x_i) · v_i / sigma_i²
         c_chapeau = M_tilde^-1 · w_tilde       I_F = M_tilde       BCR = I_F^-1 = M_tilde^-1

   - "sigma_constant" (TP2, section 1) : on suppose sigma_i = sigma pour tout i, sigma inconnu (et non estimé).
     Le facteur 1 / sigma² sort des sommes (M_tilde = M / sigma², w_tilde = w / sigma²) et disparaît de c_chapeau :
         M[n, m] = somme_i P_n(x_i) · P_m(x_i)      w[n] = somme_i P_n(x_i) · v_i   (vecteur noté v dans le TP)
         c_chapeau = M^-1 · w
         I_F = M / sigma²                           BCR = sigma² · M^-1   (sigma à fournir)

   Dans la classe, les attributs `M` et `w` contiennent M_tilde et w_tilde pour "sigma_variable", M et w pour
   "sigma_constant". Si tous les sigma_i sont égaux, les deux estimateurs donnent le même c_chapeau.

   Attention : si le bruit n'est pas gaussien (autre loi, valeurs aberrantes), ce n'est plus la bonne vraisemblance.
"""
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre

BASES = ("canonique", "legendre")
ESTIMATEURS = ("sigma_constant", "sigma_variable")


def basis_matrix(x, length, degree, basis="canonique"):
    """Matrice P de taille (N, n0+1) avec P[i, n] = P_n(x_i)  (= ξ_i^n en base canonique).

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
    """Matrice M_tilde[n, m] = somme_i P_n(x_i) · P_m(x_i) / sigma_i²   (sigmas=None : poids 1, matrice M du TP2)."""
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
        estimateur.fit(x, v)                     # "sigma_constant" : c_chapeau = M^-1 · w
        estimateur.fit(x, v, sigmas=sigmas)      # "sigma_variable" : sigma_i connus, moindres carrés pondérés
        estimateur.coefficients                 # [c0, ..., c_n0] estimés DANS LA BASE CHOISIE
        estimateur.canonical_coefficients        # les mêmes, convertis en base canonique (comparables à SpeedPolynomial)
        estimateur.crb(sigma=5.0)                # borne de Cramér-Rao (sigma obligatoire pour "sigma_constant")
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

        self.sigmas = sigmas if self.estimator == "sigma_variable" else None
        return self                                      # permet d'écrire MaximumLikelihood(...).fit(x, v)

    def _check_fitted(self):
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) d'abord")

    def fisher_information(self, sigma=None):
        """Information de Fisher.
        "sigma_constant" : I_F = M / sigma²  (sigma obligatoire : il n'est pas estimé) ;
        "sigma_variable" : I_F = M  (les 1/sigma_i² sont déjà dans M ; l'argument sigma est ignoré)."""
        self._check_fitted()
        if self.estimator == "sigma_variable":
            return self.M.copy()
        if sigma is None:
            raise ValueError("'sigma_constant' n'estime pas sigma : donner fisher_information(sigma=...)")
        return self.M / sigma ** 2

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
