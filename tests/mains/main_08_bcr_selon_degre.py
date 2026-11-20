"""Main 8 - Variance des estimateurs et borne de Cramér-Rao EN FONCTION DU DEGRE d.
Pour chaque degré d de 2 à D_MAX, on refait le main 7 :
  - le vrai polynôme est de degré d (coefficients libres choisis automatiquement) et on l'estime au degré d ;
  - N_C nuages de points, chacun avec sa PROPRE graine (aucune graine n'est réutilisée, même entre degrés) :
        nuage k du degré d  ->  graine = GRAINE + (d - 2) · N_C + k ;
  - pour chaque coefficient c_n et chaque estimateur, on calcule la variance du nuage des N_C valeurs ĉ_n,
    et la BCR (I_F^-1 = M_tilde^-1 du vrai modèle, sigma_i connus), moyennée sur les N_C nuages.
Graphe : une case par coefficient c_n ; en abscisse le degré d (c_n n'existe que si n <= d) ; en ordonnée
(échelle log) la BCR (ligne) et la variance de chaque estimateur avec sa barre d'erreur
± NB_SIGMA_PRECISION · variance · sqrt(2 / (N_C - 1))  (précision de la variance calculée sur N_C nuages)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import MaximumLikelihood, Noise, SpeedPolynomial, create_rng, generate_positions, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LONGUEUR = 1500.0                       # longueur du tronçon [m]
V0 = 40.0                               # vitesse au début du tronçon [km/h]
VL = 35.0                               # vitesse à la fin du tronçon [km/h] (doit être <= V_LIMITE)
V_LIMITE = 90.0                         # limite de vitesse du tronçon [km/h]
D_MAX = 6                               # degrés étudiés : d = 2, 3, ..., D_MAX
ALPHA1, ALPHA2 = 2.0, 8.0               # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2
N_POINTS = 10000                         # nombre de points mesurés (= de voitures) par nuage
N_C = 1000                              # nombre de nuages (de simulations) PAR DEGRE
NB_SIGMA_PRECISION = 2                  # barres d'erreur : ± 1, 2 ou 3 écarts-types (~68 %, ~95 %, ~99,7 %)
BASE = "canonique"                      # base des polynômes pour l'estimation : "canonique" ou "legendre"
GRAINE = None                           # graine du 1er nuage (None = tirée au hasard et affichée)
AFFICHER = True                         # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement

ESTIMATEURS = {                         # nom -> (couleur, marqueur, légende)   (mêmes couleurs que le main 7)
    "sigma_constant": ("tab:green", "s", "sigma constant (TP2)"),
    "sigma_variable": ("tab:orange", "D", "sigma_i connus (pondéré 1/sigma_i²)"),
}
COULEUR_BCR = "tab:blue"


def simuler_degre(d, graines):
    """N_C nuages pour le degré d. Renvoie (variances {estimateur: tableau (d+1)}, diagonale de la BCR moyenne)."""
    vrai = SpeedPolynomial(LONGUEUR, V0, VL, d, None, speed_limit=V_LIMITE)
    estimes = {nom: np.zeros((len(graines), d + 1)) for nom in ESTIMATEURS}
    somme_bcr = np.zeros((d + 1, d + 1))
    for k, graine in enumerate(graines):
        rng, _ = create_rng(int(graine))
        x = generate_positions(N_POINTS, LONGUEUR, rng)
        bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
        v = simulate_speeds(vrai, bruit, x)
        sigmas = bruit.last_sigmas
        for nom in ESTIMATEURS:
            est = MaximumLikelihood(LONGUEUR, d, BASE, nom).fit(x, v, sigmas=sigmas)
            estimes[nom][k] = est.coefficients
            if nom == "sigma_variable":
                somme_bcr += est.crb()                     # M_tilde^-1 : BCR du vrai modèle pour ce nuage
    variances = {nom: c.var(axis=0, ddof=1) for nom, c in estimes.items()}
    return variances, np.diag(somme_bcr / len(graines))


def main():
    if NB_SIGMA_PRECISION not in (1, 2, 3):
        raise ValueError("NB_SIGMA_PRECISION doit valoir 1, 2 ou 3")
    if D_MAX < 2:
        raise ValueError("D_MAX doit être >= 2")
    _, graine0 = create_rng(GRAINE)
    degres = list(range(2, D_MAX + 1))
    precision_relative = np.sqrt(2 / (N_C - 1))            # e.t. relatif d'une variance estimée sur N_C nuages

    # 1) Simulations : N_C nuages par degré, une graine différente pour chaque nuage
    resultats = {}                                         # d -> (variances, bcr)
    np.set_printoptions(precision=3, suppress=False, linewidth=150)
    print(f"{N_C} nuages de {N_POINTS} points par degre, base {BASE}, sigma_i dans [{ALPHA1}, {ALPHA2}] km/h")
    print(f"precision relative d'une variance : {precision_relative:.3f} (1 e.t.)")
    for d in degres:
        graines = (graine0 + (d - 2) * N_C + np.arange(N_C)) % 2**32
        try:
            resultats[d] = simuler_degre(d, graines)
        except ValueError as erreur:                      # pas de polynôme admissible pour ce degré
            print(f"d = {d} : ignore ({erreur})")
            continue
        variances, bcr = resultats[d]
        print("-" * 70)
        print(f"d = {d}   (graines {graines[0]} a {graines[-1]})")
        print("  variances de la BCR              :", bcr)
        for nom in ESTIMATEURS:
            print(f"  variance / BCR ({nom:15s}) :", np.round(variances[nom] / bcr, 3))
    print("-" * 70)
    print(f"pour regenerer : GRAINE = {graine0}")
    if not resultats:
        raise ValueError("aucun degré n'a pu être simulé")

    # 2) Deux figures, une case par coefficient c_n, abscisse = degré d :
    #    - "variance" : BCR et variance des estimateurs (échelle log : la variance grandit très vite avec d) ;
    #    - "rapport"  : variance / BCR (même échelle pour tous : 1 = BCR atteinte, > 1 = moins précis que la BCR).
    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    for mode in ("variance", "rapport"):
        fig = tracer(mode, resultats, degres, precision_relative, graine0)
        fig.savefig(dossier / f"08_bcr_selon_degre_{mode}.png", dpi=150)
    if AFFICHER:
        plt.show()


def tracer(mode, resultats, degres, precision_relative, graine0):
    n_coefs = max(resultats) + 1
    n_col = min(4, n_coefs)
    n_lig = int(np.ceil(n_coefs / n_col))
    fig, axes = plt.subplots(n_lig, n_col, figsize=(3.6 * n_col + 0.5, 3.4 * n_lig + 1), layout="constrained",
                             squeeze=False)
    axes = axes.ravel()
    decalage = {nom: 0.24 * (i - (len(ESTIMATEURS) - 1) / 2) for i, nom in enumerate(ESTIMATEURS)}   # pour ne pas superposer
    k = NB_SIGMA_PRECISION
    for n in range(n_coefs):
        ax = axes[n]
        ds = [d for d in resultats if d >= n]              # degrés où le coefficient c_n existe
        bcr = np.array([resultats[d][1][n] for d in ds])
        norme = bcr if mode == "rapport" else np.ones(len(ds))
        ax.plot(ds, bcr / norme, "o--", color=COULEUR_BCR, lw=1.5, ms=6, label="BCR")
        for nom, (couleur, marqueur, legende) in ESTIMATEURS.items():
            valeur = np.array([resultats[d][0][nom][n] for d in ds]) / norme
            haut = k * precision_relative * valeur
            bas = np.minimum(haut, 0.95 * valeur)          # échelle log : la barre ne doit pas passer sous 0
            ax.errorbar(np.array(ds) + decalage[nom], valeur, yerr=[bas, haut], fmt=marqueur, color=couleur,
                        capsize=4, lw=1.5, ms=5, label=legende)
        if mode == "variance":
            ax.set_yscale("log")
        ax.set_xticks(degres)
        ax.set_xlim(degres[0] - 0.5, degres[-1] + 0.5)
        ax.set_title(f"c_{n}")
        ax.set_xlabel("degré d")
        ax.grid(alpha=0.3, which="both")
    for ax in axes[n_coefs:]:
        ax.set_visible(False)
    for ligne in range(n_lig):
        axes[ligne * n_col].set_ylabel("variance du nuage des ĉ_n" if mode == "variance" else "variance / BCR")
    poignees, legendes = axes[0].get_legend_handles_labels()
    fig.legend(poignees, legendes, loc="outside lower center", ncol=3, fontsize=9)
    titre = "Variance des estimateurs et BCR" if mode == "variance" else "Variance des estimateurs divisée par la BCR"
    fig.suptitle(f"{titre} selon le degré d ({N_C} nuages de {N_POINTS} points par degré, base {BASE})\n"
                 f"barres d'erreur : ± {k} écart(s)-type(s) de la précision de la variance, graine initiale {graine0}")
    return fig


if __name__ == "__main__":
    main()
