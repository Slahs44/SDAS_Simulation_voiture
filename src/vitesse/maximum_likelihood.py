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

   - "sigma_constant" (TP2, section 1) : on suppose sigma_i = sigma pour tout i, sigma inconnu.
     Le facteur 1 / sigma² sort des sommes (M_tilde = M / sigma², w_tilde = w / sigma²) :
         M[n, m] = somme_i P_n(x_i) · P_m(x_i)      w[n] = somme_i P_n(x_i) · v_i   (vecteur noté v dans le TP)
         c_chapeau = M^-1 · w                       sigma_chapeau² = (1/N) · somme_i (v_i - mu_i(c_chapeau))²
         I_F = M / sigma²                           BCR = sigma² · M^-1   (calculée avec sigma_chapeau)

   - "sigma_variable_inconnu" : chaque voiture a son sigma_i, mais on NE le connaît PAS ; on connaît seulement la
     loi des sigma_i (celle de Noise : Normale((alpha1+alpha2)/2, ((alpha2-alpha1)/6)²) ramenée dans [alpha1, alpha2]).
     On ne peut pas estimer chaque sigma_i (un seul point par voiture : la vraisemblance serait infinie avec
     sigma_i -> 0 sur un point exactement ajusté). On intègre donc sigma_i : le bruit b_i suit un MÉLANGE de gaussiennes
         f(b) = somme_k pi_k · Normale(b ; 0, s_k²)       (s_k, pi_k : loi des sigma_i discrétisée)
     et on maximise  ℓ_mel(v | c) = somme_i ln f(v_i - mu_i(c))  par l'algorithme EM :
         E : omega_i = E[1 / sigma_i² | v_i - mu_i(c)]   (une voiture dont le résidu est grand est jugée peu précise)
         M : c = M_tilde^-1 · w_tilde, avec les poids omega_i à la place des 1 / sigma_i²
     On répète jusqu'à convergence (ℓ_mel augmente à chaque tour). Information de Fisher : I_F = i_b · M, où
     i_b = ∫ f'(b)² / f(b) db est l'information de la loi du bruit (1/E[sigma²] <= i_b <= E[1/sigma²]).

   Dans la classe, les attributs `M` et `w` contiennent M_tilde et w_tilde pour "sigma_variable" (et, avec les poids
   omega_i, pour "sigma_variable_inconnu"), M et w pour "sigma_constant". Si tous les sigma_i sont égaux, les
   estimateurs donnent le même c_chapeau.

   Attention : si le bruit n'est pas gaussien (autre loi, valeurs aberrantes), ce n'est plus la bonne vraisemblance.
"""
from math import erf, sqrt

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre

BASES = ("canonique", "legendre")
ESTIMATEURS = ("sigma_constant", "sigma_variable", "sigma_variable_inconnu")


def sigma_law(sigma_range, n_nodes=101):
    """Loi des sigma_i discrétisée (la même que Noise) : noeuds s_k et probabilités pi_k.

    sigma_i ~ Normale(moyenne = (alpha1+alpha2)/2, écart-type = (alpha2-alpha1)/6), ramené dans [alpha1, alpha2].
    Chaque noeud s_k reçoit la probabilité de tomber plus près de lui que des autres ; les deux noeuds extrêmes
    reçoivent aussi les queues (comme le np.clip de Noise)."""
    alpha1, alpha2 = (float(a) for a in sigma_range)
    if not 0 < alpha1 < alpha2:
        raise ValueError("sigma_range = (alpha1, alpha2) doit vérifier 0 < alpha1 < alpha2")
    moyenne, ecart_type = (alpha1 + alpha2) / 2, (alpha2 - alpha1) / 6
    noeuds = np.linspace(alpha1, alpha2, n_nodes)
    milieux = (noeuds[:-1] + noeuds[1:]) / 2
    cdf = [0.0] + [0.5 * (1 + erf((m - moyenne) / (ecart_type * sqrt(2)))) for m in milieux] + [1.0]
    return noeuds, np.diff(cdf)


def _mixture_terms(residus, noeuds, probas):
    """ln f(r_i) et omega_i = E[1/sigma² | r_i] pour le mélange f(b) = somme_k pi_k · Normale(b ; 0, s_k²)."""
    r = np.asarray(residus, dtype=float)[:, None]
    log_termes = np.log(probas) - np.log(np.sqrt(2 * np.pi) * noeuds) - r ** 2 / (2 * noeuds ** 2)   # (N, K)
    plus_grand = log_termes.max(axis=1, keepdims=True)                  # astuce log-somme-exp (évite exp -> 0)
    termes = np.exp(log_termes - plus_grand)
    somme = termes.sum(axis=1)
    log_f = plus_grand[:, 0] + np.log(somme)
    omega = (termes / noeuds ** 2).sum(axis=1) / somme                   # moyenne des 1/s_k² pondérée par P(k | r_i)
    return log_f, omega


def noise_information(sigma_range, n_nodes=101):
    """Information de Fisher i_b = ∫ f'(b)² / f(b) db du bruit en mélange (sigma_i inconnus, de loi connue).

    Comme f'(b) / f(b) = -b · omega(b), on a i_b = E[b² · omega(b)²], calculé par une somme sur une grille fine."""
    noeuds, probas = sigma_law(sigma_range, n_nodes)
    b = np.linspace(-12 * noeuds[-1], 12 * noeuds[-1], 40001)
    log_f, omega = _mixture_terms(b, noeuds, probas)
    return np.sum(b ** 2 * omega ** 2 * np.exp(log_f)) * (b[1] - b[0])


def mixture_log_likelihood(coefficients, x, v, length, sigma_range, basis="canonique", n_nodes=101):
    """log-vraisemblance ℓ_mel(c) quand les sigma_i sont inconnus mais de loi connue (bruit en mélange)."""
    degree = len(coefficients) - 1
    P = basis_matrix(x, length, degree, basis)
    residus = np.asarray(v, dtype=float) - np.dot(P, coefficients)
    log_f, _ = _mixture_terms(residus, *sigma_law(sigma_range, n_nodes))
    return np.sum(log_f)


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
        estimateur.fit(x, v)                     # "sigma_constant" : c_chapeau = M^-1 · w et sigma_chapeau
        estimateur.fit(x, v, sigmas=sigmas)      # "sigma_variable" : sigma_i connus, moindres carrés pondérés
        MaximumLikelihood(2000, 2, estimator="sigma_variable_inconnu", sigma_range=(2, 8)).fit(x, v)
                                                 # sigma_i inconnus, loi connue : algorithme EM
        estimateur.coefficients                 # [c0, ..., c_n0] estimés DANS LA BASE CHOISIE
        estimateur.canonical_coefficients        # les mêmes, convertis en base canonique (comparables à SpeedPolynomial)
        estimateur.sigma                         # "sigma_constant" : écart-type estimé [km/h] ; "sigma_variable" : None
        estimateur.crb()                         # borne de Cramér-Rao
        estimateur.speed(1000)                   # vitesse estimée en x = 1000 m

    Les d+1 coefficients sont estimés librement : on ne suppose PAS connus V0 et VL."""

    def __init__(self, length, degree, basis="canonique", estimator="sigma_constant", sigma_range=None,
                 n_nodes=101, max_iter=500, tol=1e-10):
        """sigma_range = (alpha1, alpha2) : loi des sigma_i, obligatoire pour "sigma_variable_inconnu" (ignorée sinon).
        n_nodes : nombre de valeurs s_k pour discrétiser cette loi ; max_iter, tol : arrêt de l'algorithme EM."""
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        if basis not in BASES:
            raise ValueError(f"base inconnue : {basis!r} (choisir parmi {BASES})")
        if estimator not in ESTIMATEURS:
            raise ValueError(f"estimateur inconnu : {estimator!r} (choisir parmi {ESTIMATEURS})")
        if estimator == "sigma_variable_inconnu":
            if sigma_range is None:
                raise ValueError("l'estimateur 'sigma_variable_inconnu' a besoin de la loi des sigma_i : "
                                 "sigma_range=(alpha1, alpha2)")
            self._noeuds, self._probas = sigma_law(sigma_range, n_nodes)    # vérifie aussi 0 < alpha1 < alpha2
        self.sigma_range = sigma_range
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None             # omega_i = E[1/sigma_i² | mesure] ("sigma_variable_inconnu" seulement)
        self.n_iter = None              # nombre de tours de l'algorithme EM
        self.log_likelihood = None      # ℓ_mel(c_chapeau) ("sigma_variable_inconnu" seulement)
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
                 (ignorés pour "sigma_constant" et "sigma_variable_inconnu")."""
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

        if self.estimator == "sigma_variable_inconnu":
            self._fit_em(P, v)
            return self

        if self.estimator == "sigma_variable":
            self.sigmas = sigmas
            self.sigma = None                            # rien à estimer : les sigma_i sont connus
        else:
            residus = v - np.dot(P, self.coefficients)
            self.sigma = np.sqrt(np.mean(residus ** 2))  # sigma_chapeau (division par N : c'est le vrai MV)
            self.sigmas = None
        return self                                      # permet d'écrire MaximumLikelihood(...).fit(x, v)

    def _fit_em(self, P, v):
        """Algorithme EM pour "sigma_variable_inconnu", en partant de c_chapeau du TP2 (déjà dans self.coefficients).

        E : omega_i = E[1/sigma_i² | v_i - mu_i(c)] ;  M : moindres carrés pondérés par omega_i.
        Chaque tour fait augmenter la log-vraisemblance du mélange ; on s'arrête quand elle ne bouge plus."""
        self._PtP = np.dot(P.T, P)                       # M du TP2 (sert à l'information de Fisher)
        log_f, omega = _mixture_terms(v - np.dot(P, self.coefficients), self._noeuds, self._probas)
        ancien = np.sum(log_f)
        for tour in range(1, self.max_iter + 1):
            self.M = np.dot(P.T * omega, P)              # M_tilde avec les poids omega_i
            self.w = np.dot(P.T * omega, v)
            self.coefficients = np.linalg.solve(self.M, self.w)
            log_f, omega = _mixture_terms(v - np.dot(P, self.coefficients), self._noeuds, self._probas)
            nouveau = np.sum(log_f)
            if nouveau - ancien <= self.tol * abs(ancien):
                break
            ancien = nouveau
        self.n_iter = tour
        self.log_likelihood = nouveau                    # ℓ_mel(c_chapeau)
        self.weights = omega
        self.sigma = None
        self.sigmas = None

    def _check_fitted(self):
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) d'abord")

    def fisher_information(self, sigma=None):
        """Information de Fisher.
        "sigma_constant" : I_F = M / sigma²  (sigma = sigma_chapeau si non précisé) ;
        "sigma_variable" : I_F = M  (les 1/sigma_i² sont déjà dans M ; l'argument sigma est ignoré) ;
        "sigma_variable_inconnu" : I_F = i_b · M du TP2  (i_b : information du bruit en mélange ; sigma ignoré)."""
        self._check_fitted()
        if self.estimator == "sigma_variable":
            return self.M.copy()
        if self.estimator == "sigma_variable_inconnu":
            return noise_information(self.sigma_range, self.n_nodes) * self._PtP
        s = self.sigma if sigma is None else sigma
        return self.M / s ** 2

    def crb(self, sigma=None):
        """Borne de Cramér-Rao I_F^-1 : covariance minimale d'un estimateur sans biais de c.
        "sigma_constant" : sigma² · M^-1 ;  "sigma_variable" : M^-1 ;  "sigma_variable_inconnu" : M^-1 / i_b."""
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
