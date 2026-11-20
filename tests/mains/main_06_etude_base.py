"""Main 6 - Étudier l'influence de la BASE (canonique ou Legendre) sur les coefficients estimés.
On garde les MEMES positions x_i et les MEMES sigma_i (le « plan d'expérience » est fixé), et on retire N_TIRAGES fois
le bruit. À chaque tirage on estime les coefficients dans les deux bases. On obtient ainsi un nuage de N_TIRAGES
estimations ĉ pour chaque base, que l'on compare :
  - aux vrais coefficients (biais) ;
  - à la borne de Cramér-Rao BCR = I_F^-1 (variances, et forme des nuages : ellipses de covariance) ;
  - entre bases : en base canonique les coefficients sont très corrélés (nuages allongés, grandes variances),
    en base de Legendre ils le sont beaucoup moins (base quasi-orthogonale sur [0, L]).
Les graphes reprennent ceux du TP (tp_poly_gaussiens.m) : un nuage (c_i, c_j) pour chaque paire de coefficients."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import MaximumLikelihood, Noise, SpeedPolynomial, create_rng

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LONGUEUR = 1500.0                       # longueur du tronçon [m]
V0 = 40.0                               # vitesse au début du tronçon [km/h]
VL = 35.0                               # vitesse à la fin du tronçon [km/h] (doit être <= V_LIMITE)
V_LIMITE = 90.0                         # limite de vitesse du tronçon [km/h]
D = 3                                   # degré du polynôme (3 -> 6 graphes par base ; 6 -> 21 graphes)
COEFS_LIBRES = None                     # None = calcul automatique ; ou une liste [c2, ..., cd]
ALPHA1, ALPHA2 = 2.0, 8.0               # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2
N_POINTS = 100                          # nombre de points mesurés (= de voitures) d'UN échantillon
N_TIRAGES = 1000                        # nombre d'échantillons (tirages du bruit) pour les nuages de coefficients
POSITIONS = "regulieres"                # "regulieres" (comme le TP), "irregulieres" (x = L·u², plus serrées au début)
                                        # ou "aleatoires" (uniformes sur [0, L], tirées une seule fois)
ESTIMATEUR = "sigma_constant"           # "sigma_variable" (sigma_i connus) ou "sigma_constant" (TP2)
D_ESTIME = D                            # degré utilisé pour l'estimation
VISU_BCR = 2                            # 0 : nuages seuls ; 1 : + points tirés selon Normale(c, BCR) ;
                                        # 2 : + ellipses de la BCR (1 et 3 écarts-types)
GRAINE = None                           # None = aléatoire (graine affichée) ; un entier = reproductible
AFFICHER = True                         # True : ouvre les fenêtres des graphiques ; False : sauvegarde seulement

BASES = {"canonique": "base canonique (ξ^n, ξ = x/L)",
         "legendre": "base de Legendre (Legendre_n(t), t = 2x/L - 1)"}


def positions(rng):
    """Positions x_i [m] des N_POINTS mesures (gardées identiques pour tous les tirages)."""
    u = np.linspace(0.0, 1.0, N_POINTS)
    if POSITIONS == "regulieres":
        return LONGUEUR * u
    if POSITIONS == "irregulieres":
        return LONGUEUR * u ** 2                         # même choix que choix_discretisation_xi = 3 du TP
    if POSITIONS == "aleatoires":
        return np.sort(rng.uniform(0.0, LONGUEUR, N_POINTS))
    raise ValueError(f"POSITIONS inconnu : {POSITIONS!r}")


def ellipse(centre, covariance, echelle=1.0, n=200):
    """Points de l'ellipse  centre + echelle · A · [cos θ, sin θ]  avec A·Aᵀ = covariance
    (toute « racine » A de la covariance donne la même ellipse ; on prend celle de Cholesky)."""
    theta = np.linspace(-np.pi, np.pi, n)
    A = np.linalg.cholesky(covariance)
    return centre[:, None] + echelle * np.dot(A, np.vstack([np.cos(theta), np.sin(theta)]))


def tracer_nuages(base, estimes, vrais, bcr, rng, graine):
    """Une figure par base : pour chaque paire (i, j), i < j, le nuage des (ĉ_i, ĉ_j) estimés."""
    n = D_ESTIME                                             # paires : i = 0..n-1 (lignes), j = 1..n (colonnes)
    fig, axes = plt.subplots(n, n, figsize=(3.0 * n + 1, 3.0 * n), layout="constrained", squeeze=False)
    fig.suptitle(f"Coefficients estimés dans la {BASES[base]}\n"
                 f"{N_TIRAGES} tirages de {N_POINTS} points, estimateur {ESTIMATEUR}, graine {graine}")
    for i in range(n):
        for j in range(1, n + 1):
            ax = axes[i, j - 1]
            if j <= i:
                ax.set_visible(False)
                continue
            paire = [i, j]
            centre = vrais[paire]
            cov = bcr[np.ix_(paire, paire)]
            ax.plot(estimes[:, i], estimes[:, j], "+", color="tab:red", ms=4, alpha=0.5, label="coeff estimés")
            if VISU_BCR == 1:
                r = centre[:, None] + np.dot(np.linalg.cholesky(cov), rng.standard_normal((2, N_TIRAGES)))
                ax.plot(r[0], r[1], "x", color="tab:blue", ms=3, alpha=0.4, label="coeff tirés selon Normale(c, BCR)")
            elif VISU_BCR == 2:
                ax.plot(*centre, "*", color="tab:blue", ms=9, label="vrais coefficients")
                ax.plot(*ellipse(centre, cov), "-", color="tab:blue", lw=1.5, label="ellipse de la BCR (1 écart-type)")
                ax.plot(*ellipse(centre, cov, 3.0), "--", color="tab:blue", lw=1.5, label="axes 3 fois plus grands")
            ax.set_box_aspect(1)
            ax.set_xlabel(f"c_{i}")
            ax.set_ylabel(f"c_{j}")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            rho = np.corrcoef(estimes[:, i], estimes[:, j])[0, 1]
            ax.set_title(f"corrélation {rho:+.2f}", fontsize=9)
    poignees, legendes = axes[0, 0].get_legend_handles_labels()
    if n > 1:      # la case vide en bas à gauche accueille la légende (comme dans le TP)
        axes[n - 1, 0].set_visible(True)
        axes[n - 1, 0].axis("off")
        axes[n - 1, 0].legend(poignees, legendes, loc="center", fontsize=9)
    else:
        fig.legend(poignees, legendes, loc="outside lower center", fontsize=9)
    return fig


def correlation(cov):
    ecarts = np.sqrt(np.diag(cov))
    return cov / np.outer(ecarts, ecarts)


def main():
    rng, graine = create_rng(GRAINE)

    # 1) Le plan d'expérience, fixé une fois pour toutes : positions x_i et précision sigma_i de chaque voiture
    vrai = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE)
    x = positions(rng)
    bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
    sigmas = bruit.sample_sigmas(N_POINTS)
    sigma_global = np.sqrt(np.mean(sigmas ** 2))            # pour la BCR de "sigma_constant"
    v_vraie = vrai.speed(x)

    # 2) Vrais coefficients dans chaque base : estimation sur les mesures SANS bruit (comme cL dans le TP)
    vrais, bcr, cond = {}, {}, {}
    for base in BASES:
        sans_bruit = MaximumLikelihood(LONGUEUR, D_ESTIME, base, ESTIMATEUR).fit(x, v_vraie, sigmas=sigmas)
        vrais[base] = sans_bruit.coefficients
        bcr[base] = sans_bruit.crb(sigma=sigma_global)      # ne dépend que de x et des sigma_i, pas du bruit
        cond[base] = np.linalg.cond(sans_bruit.M)

    # 3) N_TIRAGES échantillons : même x, mêmes sigma_i, nouveau bruit à chaque fois
    estimes = {base: np.zeros((N_TIRAGES, D_ESTIME + 1)) for base in BASES}
    for k in range(N_TIRAGES):
        v = v_vraie + rng.normal(0.0, sigmas)
        for base in BASES:
            estimes[base][k] = MaximumLikelihood(LONGUEUR, D_ESTIME, base, ESTIMATEUR).fit(x, v, sigmas=sigmas).coefficients

    # 4) Résultats dans le terminal
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(f"positions {POSITIONS}, {N_POINTS} points, sigma_i de {sigmas.min():.2f} a {sigmas.max():.2f} km/h, "
          f"estimateur {ESTIMATEUR}, {N_TIRAGES} tirages")
    if ESTIMATEUR == "sigma_constant":
        print("  (attention : avec des sigma_i differents, sigma^2·M^-1 n'est qu'une approximation de la covariance)")
    for base in BASES:
        c = estimes[base]
        moyenne = c.mean(axis=0)
        cov = np.cov(c, rowvar=False)
        print("-" * 70)
        print(f"BASE {base.upper()}   (conditionnement de M : {cond[base]:.3g})")
        print("vrais coefficients          :", vrais[base])
        print("moyenne des estimations     :", moyenne)
        print("biais                       :", moyenne - vrais[base])
        print("  precision du biais [e.t.] :", np.sqrt(np.diag(cov) / N_TIRAGES))
        print("variances empiriques        :", np.diag(cov))
        print("variances de la BCR         :", np.diag(bcr[base]))
        print("  precision des var. [e.t.] :", np.sqrt(2 / N_TIRAGES) * np.diag(cov))
        print("correlations empiriques :")
        print(correlation(cov))
        print("correlations de la BCR :")
        print(correlation(bcr[base]))
    print("-" * 70)
    print(f"graine utilisee : {graine}   (pour regenerer : GRAINE = {graine})")

    # 5) Graphes : nuages (c_i, c_j) dans chaque base, puis matrices de corrélation côte à côte
    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    for base in BASES:
        fig = tracer_nuages(base, estimes[base], vrais[base], bcr[base], rng, graine)
        fig.savefig(dossier / f"06_coefficients_{base}.png", dpi=150)

    fig, axes = plt.subplots(1, len(BASES), figsize=(5.5 * len(BASES) + 1, 5), layout="constrained")
    for ax, base in zip(axes, BASES):
        rho = correlation(np.cov(estimes[base], rowvar=False))
        image = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
        for (a, b), r in np.ndenumerate(rho):
            ax.text(b, a, f"{r:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(r) > 0.6 else "black")
        etiquettes = [f"c_{n}" for n in range(D_ESTIME + 1)]
        ax.set_xticks(range(D_ESTIME + 1), etiquettes)
        ax.set_yticks(range(D_ESTIME + 1), etiquettes)
        ax.set_title(f"{base} (conditionnement de M : {cond[base]:.2g})")
    fig.colorbar(image, ax=axes, label="corrélation empirique entre ĉ_i et ĉ_j")
    fig.suptitle(f"Corrélations entre coefficients estimés ({N_TIRAGES} tirages, positions {POSITIONS})")
    fig.savefig(dossier / "06_correlations.png", dpi=150)
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
