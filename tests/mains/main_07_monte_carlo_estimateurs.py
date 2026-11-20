"""Main 7 - Répéter le main 5 N_C fois (Monte-Carlo) pour comparer les deux estimateurs à la borne de Cramér-Rao.
À chaque répétition k on génère un NOUVEAU nuage de points avec sa propre graine (graine_k = GRAINE + k) :
nouvelles positions x_i, nouveaux sigma_i, nouveau bruit. On estime le polynôme avec les deux estimateurs
("sigma_constant" et "sigma_variable", voir main 5) et on calcule la BCR de ce nuage.
Sur les N_C répétitions on obtient, pour chaque coefficient c_n et chaque estimateur :
  - la moyenne des ĉ_n (-> biais = moyenne - vrai c_n) ;
  - la variance des ĉ_n, à comparer à la BCR.
La BCR utilisée est celle du vrai modèle (sigma_i connus) : I_F^-1 = M_tilde^-1, moyennée sur les N_C nuages
(les x_i et sigma_i changent d'un nuage à l'autre, donc la borne aussi). Aucun estimateur sans biais ne peut
avoir une variance plus petite : "sigma_variable" doit l'atteindre, "sigma_constant" doit rester au-dessus.
Graphe : une colonne par coefficient c_n, deux lignes. Les barres d'erreur (± NB_SIGMA_PRECISION écarts-types)
donnent la PRECISION de ce que l'on a calculé à partir des N_C valeurs ĉ_n (elles diminuent quand N_C augmente) :
  - en haut : biais = moyenne des N_C ĉ_n - vrai c_n,  barre ± k · e.t.(nuage) / sqrt(N_C) ;
  - en bas  : variance du nuage des N_C ĉ_n, comparée à la BCR,  barre ± k · variance · sqrt(2 / (N_C - 1))
    (précision de la variance d'un échantillon gaussien de taille N_C)."""
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
D = 3                                   # degré du polynôme
COEFS_LIBRES = None                     # None = calcul automatique ; ou une liste [c2, ..., cd]
ALPHA1, ALPHA2 = 2.0, 8.0               # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2
N_POINTS = 10000                         # nombre de points mesurés (= de voitures) par nuage
N_C = 100                              # nombre de répétitions du main 5 (= de nuages de points)
NB_SIGMA_PRECISION = 1                  # barres d'erreur : ± 1, 2 ou 3 écarts-types (~68 %, ~95 %, ~99,7 %)
BASE = "canonique"                      # base des polynômes pour l'estimation : "canonique" ou "legendre"
D_ESTIME = D                            # degré utilisé pour l'estimation (essaie D-1 : un biais apparaît)
GRAINE = None                         # graine du 1er nuage ; le nuage k utilise GRAINE + k
                                        # (None = tirée au hasard et affichée ; un entier = reproductible)
AFFICHER = True                         # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement

ESTIMATEURS = {                         # nom -> (couleur, marqueur, légende)   (mêmes couleurs que le main 5)
    "sigma_constant": ("tab:green", "s", "sigma constant (TP2)"),
    "sigma_variable": ("tab:orange", "D", "sigma_i connus (pondéré 1/sigma_i²)"),
}
COULEUR_BCR = "tab:blue"


def vrais_coefficients(vrai):
    """Vrais coefficients exprimés dans la base d'estimation (BASE, degré D_ESTIME).
    On ajuste la vraie courbe SANS bruit sur une grille fine : si D_ESTIME >= D on retrouve exactement le
    polynôme ; si D_ESTIME < D, c'est le meilleur polynôme de degré D_ESTIME (référence pour le biais)."""
    grille = np.linspace(0.0, LONGUEUR, 2000)
    return MaximumLikelihood(LONGUEUR, D_ESTIME, BASE, "sigma_constant").fit(grille, vrai.speed(grille)).coefficients


def main():
    if NB_SIGMA_PRECISION not in (1, 2, 3):
        raise ValueError("NB_SIGMA_PRECISION doit valoir 1, 2 ou 3")
    _, graine0 = create_rng(GRAINE)                       # graine du 1er nuage (tirée au hasard si GRAINE = None)
    graines = (graine0 + np.arange(N_C)) % 2**32           # une graine différente par nuage

    vrai = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE)
    c_vrai = vrais_coefficients(vrai)
    n_coefs = D_ESTIME + 1

    # 1) N_C répétitions du main 5, chacune avec sa propre graine
    estimes = {nom: np.zeros((N_C, n_coefs)) for nom in ESTIMATEURS}
    somme_bcr = np.zeros((n_coefs, n_coefs))
    for k, graine in enumerate(graines):
        rng, _ = create_rng(int(graine))
        x = generate_positions(N_POINTS, LONGUEUR, rng)
        bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
        v = simulate_speeds(vrai, bruit, x)
        sigmas = bruit.last_sigmas
        for nom in ESTIMATEURS:
            est = MaximumLikelihood(LONGUEUR, D_ESTIME, BASE, nom).fit(x, v, sigmas=sigmas)
            estimes[nom][k] = est.coefficients
            if nom == "sigma_variable":
                somme_bcr += est.crb()                     # M_tilde^-1 : BCR du vrai modèle pour ce nuage
    bcr = somme_bcr / N_C
    ecart_bcr = np.sqrt(np.diag(bcr))

    # Statistiques du nuage des N_C valeurs ĉ_n de chaque estimateur, et leur précision
    moyennes = {nom: c.mean(axis=0) for nom, c in estimes.items()}
    ecarts = {nom: c.std(axis=0, ddof=1) for nom, c in estimes.items()}       # écart-type du nuage
    biais = {nom: moyennes[nom] - c_vrai for nom in ESTIMATEURS}
    prec_biais = {nom: ecarts[nom] / np.sqrt(N_C) for nom in ESTIMATEURS}     # e.t. de la moyenne
    variances = {nom: ecarts[nom] ** 2 for nom in ESTIMATEURS}
    prec_var = {nom: variances[nom] * np.sqrt(2 / (N_C - 1)) for nom in ESTIMATEURS}   # e.t. de la variance

    # 2) Résultats dans le terminal
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(f"{N_C} nuages de {N_POINTS} points, degre estime {D_ESTIME}, base {BASE}, "
          f"sigma_i dans [{ALPHA1}, {ALPHA2}] km/h")
    print("vrais coefficients                 :", c_vrai)
    print("ecarts-types de la BCR             :", ecart_bcr)
    for nom in ESTIMATEURS:
        print("-" * 70)
        print(f"ESTIMATEUR {nom}")
        print("moyenne des estimations            :", moyennes[nom])
        print("biais                              :", biais[nom])
        print("  precision du biais [e.t.]        :", prec_biais[nom])
        print("variances du nuage                 :", variances[nom])
        print("  precision des variances [e.t.]   :", prec_var[nom])
        print("variances de la BCR                :", np.diag(bcr))
        print("variance / BCR (>= 1 en theorie)   :", variances[nom] / np.diag(bcr))
        print(f"  precision de ce rapport [e.t.]   : ~{np.sqrt(2 / (N_C - 1)):.3f} (relative)")
    print("-" * 70)
    print(f"graines utilisees : {graines[0]} a {graines[-1]}   (pour regenerer : GRAINE = {graine0} ; "
          f"nuage k seul : graine {graine0} + k)")

    # 3) Graphe : une colonne par coefficient ; en haut le biais, en bas la variance du nuage (vs BCR)
    k = NB_SIGMA_PRECISION
    fig, axes = plt.subplots(2, n_coefs, figsize=(2.6 * n_coefs + 1.5, 8), layout="constrained", squeeze=False)
    positions = np.arange(len(ESTIMATEURS) + 1)                 # 0 : BCR, puis un estimateur par position
    noms_axe = ["BCR"] + list(ESTIMATEURS)
    for n in range(n_coefs):
        haut, bas = axes[0, n], axes[1, n]

        # En haut : biais ± précision de la moyenne (la BCR concerne des estimateurs SANS biais : référence 0)
        haut.axhline(0.0, color="k", lw=1, ls=":", label="biais nul")
        for p, (nom, (couleur, marqueur, legende)) in zip(positions[1:], ESTIMATEURS.items()):
            haut.errorbar(p, biais[nom][n], yerr=k * prec_biais[nom][n], fmt=marqueur, color=couleur,
                          capsize=6, lw=2, label=legende)
        haut.set_title(f"c_{n}")

        # En bas : variance du nuage ± sa précision, et la BCR (point + ligne horizontale)
        bas.axhline(bcr[n, n], color=COULEUR_BCR, lw=1, ls="--")
        bas.plot(positions[0], bcr[n, n], "o", color=COULEUR_BCR, ms=8, label="BCR (moyenne sur les nuages)")
        for p, (nom, (couleur, marqueur, legende)) in zip(positions[1:], ESTIMATEURS.items()):
            bas.errorbar(p, variances[nom][n], yerr=k * prec_var[nom][n], fmt=marqueur, color=couleur,
                         capsize=6, lw=2)

        for ax in (haut, bas):
            ax.set_xticks(positions, noms_axe if ax is bas else [""] * len(positions),
                          rotation=35, ha="right", fontsize=8)
            ax.set_xlim(-0.6, positions[-1] + 0.6)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel(f"biais = moyenne - vrai\n± {k}·e.t./sqrt(N_C)")
    axes[1, 0].set_ylabel(f"variance du nuage des ĉ_n\n± {k}·var·sqrt(2/(N_C-1))")
    poignees, legendes = axes[0, 0].get_legend_handles_labels()
    p_bcr, l_bcr = axes[1, 0].get_legend_handles_labels()
    fig.legend(poignees + p_bcr, legendes + l_bcr, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Monte-Carlo : {N_C} nuages de {N_POINTS} points (graines {graines[0]} à {graines[-1]}), "
                 f"degré {D_ESTIME}, base {BASE}\nbarres d'erreur : ± {k} écart(s)-type(s) de la précision "
                 f"du biais et de la variance")

    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    fig.savefig(dossier / "07_monte_carlo_estimateurs.png", dpi=150)
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
