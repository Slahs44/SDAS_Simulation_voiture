"""Main 5 - Comparer les deux estimateurs du maximum de vraisemblance sur les MEMES mesures.
On simule des mesures (polynôme + bruit normal, un sigma_i par voiture), puis on estime le polynôme avec :
  - "sigma_constant" : TP2, on suppose le même sigma pour toutes les voitures (moindres carrés ordinaires) ;
  - "sigma_variable" : chaque voiture a son sigma_i, supposé connu (moindres carrés pondérés par 1 / sigma_i²).
Les deux estimations sont tracées sur le même graphe, avec en dessous leur erreur par rapport à la vraie courbe."""
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
D = 6                                   # degré du polynôme
COEFS_LIBRES = None                     # None = calcul automatique ; ou une liste [c2, ..., cd]
ALPHA1, ALPHA2 = 2.0, 8.0               # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2
N_POINTS = 1000                          # nombre de points mesurés (= de voitures)
BASE = "canonique"                      # base des polynômes pour l'estimation : "canonique" ou "legendre"
D_ESTIME = D                            # degré utilisé pour l'estimation (en vrai il est « connu » : = D ; essaie D+1 ou D-1)
GRAINE = None                           # None = aléatoire (graine affichée) ; un entier = reproductible
AFFICHER = True                         # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement

ESTIMATEURS = {                         # nom -> (couleur, style de trait, légende)
    "sigma_constant": ("tab:green", "--", "sigma constant (TP2)"),
    "sigma_variable": ("tab:orange", "-.", "sigma_i connus (pondéré 1/sigma_i²)"),
}


def main():
    rng, graine = create_rng(GRAINE)

    # 1) Simuler les mesures à partir du vrai polynôme
    vrai = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE)
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
    v = simulate_speeds(vrai, bruit, x)
    sigmas = bruit.last_sigmas
    sigma_global_vrai = np.sqrt(np.mean(sigmas ** 2))              # écart-type global du bruit simulé

    # 2) Estimer avec les deux estimateurs, sur les mêmes mesures (sigmas ne sert qu'à "sigma_variable")
    estimations = {nom: MaximumLikelihood(LONGUEUR, D_ESTIME, BASE, nom).fit(x, v, sigmas=sigmas)
                   for nom in ESTIMATEURS}

    # 3) Comparer à la vérité
    grille = np.linspace(0, LONGUEUR, 300)
    vraie_courbe = vrai.speed(grille)
    print(f"sigma_i : de {sigmas.min():.2f} a {sigmas.max():.2f} km/h (sigma global {sigma_global_vrai:.2f} km/h)")
    print("coefficients vrais                       :", np.round(vrai.coefficients, 2))
    erreurs = {}
    for nom, est in estimations.items():
        erreurs[nom] = est.speed(grille) - vraie_courbe
        print(f"coefficients estimes ({nom:22s}) :", np.round(est.canonical_coefficients, 2))
    print()
    print(f"{'estimateur':22s} {'RMSE courbe':>12s} {'erreur max':>11s}   ecarts-types de la BCR")
    for nom, est in estimations.items():
        e = erreurs[nom]
        print(f"{nom:22s} {np.sqrt(np.mean(e ** 2)):9.2f} km/h {np.abs(e).max():7.2f} km/h   "
              f"{np.round(np.sqrt(np.diag(est.crb(sigma=sigma_global_vrai))), 2)}")   # sigma : pour "sigma_constant"
    print(f"graine utilisee : {graine}   (pour regenerer : GRAINE = {graine})")

    # 4) Tracer : en haut les mesures et les deux estimations, en bas l'erreur de chaque estimation
    fig, (haut, bas) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, layout="constrained",
                                    gridspec_kw={"height_ratios": [3, 1.3]})

    # Les mesures sont colorées selon leur sigma_i : les voitures précises (foncées) pèsent plus pour "sigma_variable"
    nuage = haut.scatter(x, v, c=sigmas, cmap="viridis_r", marker="+", linewidths=0.8, vmin=ALPHA1, vmax=ALPHA2, s=22, alpha=0.8,
                         label=f"mesures ({N_POINTS} voitures)")
    fig.colorbar(nuage, ax=[haut, bas], pad=0.01, label="sigma_i de la voiture [km/h]")
    haut.plot(grille, vraie_courbe, "b-", lw=2.2, label="vraie vitesse v(x)")
    for nom, (couleur, style, legende) in ESTIMATEURS.items():
        rmse = np.sqrt(np.mean(erreurs[nom] ** 2))
        haut.plot(grille, estimations[nom].speed(grille), color=couleur, ls=style, lw=2,
                  label=f"{legende} : RMSE {rmse:.2f} km/h")
    haut.set_ylabel("vitesse [km/h]")
    haut.set_title(f"Deux estimateurs du maximum de vraisemblance (degré {D_ESTIME}, base {BASE})\n"
                   f"sigma_i dans [{ALPHA1}, {ALPHA2}] km/h, graine {graine}")
    haut.grid(alpha=0.3)
    haut.legend(fontsize=9)

    bas.axhline(0.0, color="b", lw=1)
    for nom, (couleur, style, legende) in ESTIMATEURS.items():
        bas.plot(grille, erreurs[nom], color=couleur, ls=style, lw=2, label=legende)
    bas.set_xlabel("position x [m]")
    bas.set_ylabel("estimation - vraie [km/h]")
    bas.grid(alpha=0.3)

    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    fig.savefig(dossier / "05_comparaison_estimateurs.png", dpi=150)
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
