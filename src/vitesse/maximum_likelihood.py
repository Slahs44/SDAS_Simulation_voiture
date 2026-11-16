"""
maximum_likelihood.py - Retrouver le polynôme à partir des mesures (maximum de vraisemblance)
=============================================================================================

On dispose de N mesures (x_i, v_i), une par voiture, et on connaît le degré d. On cherche les coefficients
c0, c1, ..., cd (et les paramètres du bruit) qui rendent les mesures « les plus probables ».

La vraisemblance : c'est la densité de probabilité d'observer EXACTEMENT les mesures obtenues, vue comme une
fonction des paramètres. Pour des mesures indépendantes, on multiplie les densités de chaque point ; avec le
logarithme (plus pratique), le produit devient une somme :  log L = somme_i log f(v_i - v(x_i)).

Deux modèles de bruit (paramètre `noise_model` de MaximumLikelihood) :

A) "sigma_variable" (par défaut, c'est notre modèle) : chaque voiture i a son propre écart-type sigma_i,
   inconnu, tiré comme dans noise.py :
       sigma_i = S_i ramené dans [alpha1, alpha2],   S_i ~ Normale(mu, s²),
       mu = (alpha1 + alpha2) / 2,   s = (alpha2 - alpha1) / 6,
       v_i = v(x_i) + eps_i,   eps_i ~ Normale(0, sigma_i²).
   On ne connaît pas sigma_i, donc la densité d'un résidu r est la MOYENNE des densités normales sur
   toutes les valeurs possibles de sigma_i (un « mélange ») :
       f(r) = somme_k  w_k · Normale(r ; 0, sigma_k²)
   où les sigma_k couvrent [alpha1, alpha2] et les poids w_k suivent la loi de sigma_i :
     - deux masses de probabilité Phi(-3) ≈ 0,135 % sur alpha1 et sur alpha2 (les tirages ramenés aux bornes) ;
     - entre les deux, la cloche de S_i, discrétisée en N_NOEUDS points (quadrature de Gauss-Legendre).
   On maximise log L par rapport à (c0..cd, alpha1, alpha2) numériquement (scipy), en partant des moindres carrés.

B) "sigma_fixe" (cas simple, pour comparaison) : un seul sigma pour tous les points.
       log L = - N/2 · log(2π·sigma²)  -  (1 / (2·sigma²)) · somme_i ( v_i - v(x_i) )²
   - par rapport aux coefficients c : maximiser log L revient à MINIMISER somme_i ( v_i - v(x_i) )²
     (= méthode des moindres carrés). Pour un bruit NORMAL de sigma fixe, maximum de vraisemblance = moindres carrés.
   - par rapport à sigma : sigma_chapeau² = (1/N) · somme_i ( v_i - v_chapeau(x_i) )².

Attention : si le bruit suit une autre loi (valeurs aberrantes...), ce n'est plus la bonne vraisemblance.
"""
import math

import numpy as np
from scipy.optimize import minimize

MODELES_DISPONIBLES = ("sigma_variable", "sigma_fixe")
N_NOEUDS = 40                                         # nombre de points pour discrétiser la loi des sigma_i
P_BORNE = 0.5 * math.erfc(3 / math.sqrt(2))           # P(S_i < mu - 3s) = Phi(-3) ≈ 0,00135 (= masse sur chaque borne)


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
    """log-vraisemblance log L du modèle « sigma_fixe » pour des coefficients et un sigma donnés."""
    degree = len(coefficients) - 1
    A = vandermonde(x, length, degree)
    residus = v - np.dot(A, coefficients)            # v_i - v(x_i) pour chaque point
    n = len(v)
    return -0.5 * n * np.log(2 * np.pi * sigma ** 2) - np.sum(residus ** 2) / (2 * sigma ** 2)


def sigma_distribution(alpha1, alpha2):
    """Loi discrétisée des sigma_i : renvoie (sigmas_k, poids_k), avec somme des poids = 1.

    sigmas_k = mu + s·t_k  pour t_k entre -3 et +3 ; poids_k = probabilité de chaque valeur.
    Les deux extrémités (t = -3 et t = +3, soit alpha1 et alpha2) portent en plus la masse P_BORNE des tirages
    qui sortaient de [alpha1, alpha2] et ont été ramenés à la borne."""
    t, w = np.polynomial.legendre.leggauss(N_NOEUDS)  # noeuds et poids de Gauss-Legendre sur [-1, 1]
    t = 3 * t                                         # -> noeuds sur [-3, 3]
    w = w * np.exp(-t ** 2 / 2)                       # pondérés par la cloche de S_i (en unités de s)
    w = w / w.sum() * (1 - 2 * P_BORNE)               # l'intérieur de [alpha1, alpha2] a la probabilité 99,73 %
    t = np.concatenate([[-3.0], t, [3.0]])
    w = np.concatenate([[P_BORNE], w, [P_BORNE]])
    mu = (alpha1 + alpha2) / 2
    s = (alpha2 - alpha1) / 6
    return mu + s * t, w


def log_likelihood_variable_sigma(coefficients, alpha1, alpha2, x, v, length):
    """log-vraisemblance log L du modèle « sigma_variable » (un sigma_i inconnu par voiture, dans [alpha1, alpha2])."""
    degree = len(coefficients) - 1
    A = vandermonde(x, length, degree)
    residus = np.asarray(v, dtype=float) - np.dot(A, coefficients)
    sigmas, poids = sigma_distribution(alpha1, alpha2)
    # densite[i, k] = poids_k · Normale(residu_i ; 0, sigma_k²) ; on somme sur k pour avoir f(residu_i)
    densite = poids * np.exp(-residus[:, None] ** 2 / (2 * sigmas ** 2)) / (sigmas * np.sqrt(2 * np.pi))
    return np.sum(np.log(densite.sum(axis=1) + 1e-300))    # (+1e-300 : évite log(0) si un résidu est énorme)


class MaximumLikelihood:
    """Estimateur du polynôme par maximum de vraisemblance.

        estimateur = MaximumLikelihood(length=1500, degree=4)          # modèle "sigma_variable" par défaut
        estimateur.fit(x, v)             # calcule les coefficients et les paramètres du bruit
        estimateur.coefficients          # [c0, c1, ..., cd] estimés
        estimateur.alpha1, .alpha2       # bornes estimées des sigma_i [km/h] (None en "sigma_fixe")
        estimateur.sigma                 # écart-type GLOBAL du bruit estimé [km/h] (= racine(moyenne des sigma_i²))
        estimateur.speed(1000)           # vitesse estimée en x = 1000 m

    Ici les d+1 coefficients sont estimés librement : on ne suppose PAS connus V0 et VL (on ne les connaît pas
    en vrai, ce sont justement des choses à retrouver).
    Remarque : avec une seule mesure par voiture, chaque sigma_i reste inconnu ; on estime seulement la loi
    des sigma_i (alpha1, alpha2). Il faut beaucoup de points (≈ 1000) pour que alpha1 et alpha2 soient précis."""

    def __init__(self, length, degree, noise_model="sigma_variable"):
        if length <= 0:
            raise ValueError("la longueur doit être > 0")
        if degree < 1:
            raise ValueError("le degré doit être >= 1")
        if noise_model not in MODELES_DISPONIBLES:
            raise ValueError(f"modèle inconnu : '{noise_model}'. Modèles disponibles : {MODELES_DISPONIBLES}")
        self.length = length
        self.degree = degree
        self.noise_model = noise_model
        self.coefficients = None        # remplis par fit()
        self.sigma = None
        self.alpha1 = None
        self.alpha2 = None

    def fit(self, x, v):
        """Calcule le maximum de vraisemblance à partir des positions x [m] et des vitesses mesurées v [km/h]."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(x) != len(v):
            raise ValueError("x et v doivent avoir la même taille")
        if len(x) < self.degree + 1:
            raise ValueError(f"il faut au moins {self.degree + 1} points pour estimer un polynôme de degré {self.degree}")

        # 1) Moindres carrés : c'est le MV exact en "sigma_fixe", et le point de départ en "sigma_variable"
        A = vandermonde(x, self.length, self.degree)
        self.coefficients = np.linalg.lstsq(A, v, rcond=None)[0]
        residus = v - np.dot(A, self.coefficients)
        self.sigma = np.sqrt(np.mean(residus ** 2))       # sigma_chapeau (division par N : c'est le vrai MV)
        if self.noise_model == "sigma_fixe":
            return self                                   # permet d'écrire MaximumLikelihood(...).fit(x, v)

        # 2) "sigma_variable" : maximisation numérique de log L sur (c0..cd, alpha1, alpha2).
        #    On optimise log(alpha1) et log(alpha2 - alpha1) pour garantir 0 < alpha1 < alpha2.
        #    Départ : alpha1, alpha2 autour de sigma_chapeau (± 50 %).
        if self.sigma < 1e-9:                             # (pas exactement 0 à cause des arrondis de calcul)
            raise ValueError("bruit nul : utilise noise_model='sigma_fixe'")
        depart_a1, depart_a2 = 0.5 * self.sigma, 1.5 * self.sigma
        p0 = np.concatenate([self.coefficients, [np.log(depart_a1), np.log(depart_a2 - depart_a1)]])

        def moins_log_l(p):
            alpha1 = np.exp(p[-2])
            alpha2 = alpha1 + np.exp(p[-1])
            return -log_likelihood_variable_sigma(p[:-2], alpha1, alpha2, x, v, self.length)

        res = minimize(moins_log_l, p0, method="L-BFGS-B")
        self.coefficients = res.x[:-2]
        self.alpha1 = float(np.exp(res.x[-2]))
        self.alpha2 = float(self.alpha1 + np.exp(res.x[-1]))
        sigmas, poids = sigma_distribution(self.alpha1, self.alpha2)
        self.sigma = float(np.sqrt(np.sum(poids * sigmas ** 2)))   # écart-type global = racine(moyenne des sigma_i²)
        return self

    def speed(self, x):
        """Vitesse estimée v_chapeau(x) [km/h] (appeler fit() d'abord)."""
        if self.coefficients is None:
            raise ValueError("appelle fit(x, v) avant speed(x)")
        A = vandermonde(np.atleast_1d(x), self.length, self.degree)
        return np.dot(A, self.coefficients)
