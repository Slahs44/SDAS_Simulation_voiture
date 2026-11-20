"""Main 4 - Retrouver le polynôme par maximum de vraisemblance.
On simule des mesures (polynôme + bruit normal, un sigma_i par voiture), puis on estime les coefficients par
la méthode du TP2 (c_chapeau = M^-1 · w), et on compare à la vérité. Deux estimateurs au choix (ESTIMATEUR) :
  - "sigma_constant" : TP2, on suppose le même sigma pour toutes les voitures (sigma n'est pas estimé) ;
  - "sigma_variable" : chaque voiture a son sigma_i, supposé connu ; chaque point est pondéré par 1 / sigma_i²."""
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
N_POINTS = 100                          # nombre de points mesurés
N_SIGMA = 2.0                           # largeur de la bande : ± N_SIGMA · sigma global (1 -> ~68 % des mesures, 2 -> ~95 %, 3 -> ~99,7 %)
BASE = "canonique"                      # base des polynômes pour l'estimation : "canonique" ou "legendre"
ESTIMATEUR = "sigma_variable"           # "sigma_constant" (TP2) ou "sigma_variable" (sigma_i connus)
D_ESTIME = D                            # degré utilisé pour l'estimation (en vrai il est « connu » : = D ; essaie D+1 ou D-1)
GRAINE = None                           # None = aléatoire (graine affichée) ; un entier = reproductible
AFFICHER = True                         # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)

    # 1) Simuler les mesures à partir du vrai polynôme
    vrai = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE)
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    bruit = Noise("normale", rng=rng, sigma_range=(ALPHA1, ALPHA2))
    v = simulate_speeds(vrai, bruit, x)
    sigma_global_vrai = np.sqrt(np.mean(bruit.last_sigmas ** 2))   # écart-type global du bruit simulé

    # 2) Estimer par maximum de vraisemblance (on n'utilise que x, v et, pour "sigma_variable", les sigma_i)
    def estimer(choix):
        return MaximumLikelihood(LONGUEUR, D_ESTIME, BASE, choix).fit(x, v, sigmas=bruit.last_sigmas)
    estimateur = estimer(ESTIMATEUR)
    autres = [estimer(nom) for nom in ("sigma_constant", "sigma_variable") if nom != ESTIMATEUR]

    # 3) Comparer à la vérité
    print("coefficients vrais    :", np.round(vrai.coefficients, 2))
    print("coefficients estimes  :", np.round(estimateur.canonical_coefficients, 2))
    print(f"sigma_i simules : de {bruit.last_sigmas.min():.2f} a {bruit.last_sigmas.max():.2f} km/h "
          f"(sigma global {sigma_global_vrai:.2f} km/h)")
    grille = np.linspace(0, LONGUEUR, 300)
    for est in [estimateur] + autres:
        nom = est.estimator
        erreur = est.speed(grille) - vrai.speed(grille)
        print(f"erreur sur la courbe ({nom:22s}) : RMSE = {np.sqrt(np.mean(erreur ** 2)):.2f} km/h, "
              f"erreur max = {np.abs(erreur).max():.2f} km/h")
    # BCR : "sigma_constant" n'estime pas sigma, on lui donne le sigma global simulé (ignoré par "sigma_variable")
    print("ecarts-types de la BCR :", np.round(np.sqrt(np.diag(estimateur.crb(sigma=sigma_global_vrai))), 2))
    centre = estimateur.speed(grille)
    demi_largeur = N_SIGMA * sigma_global_vrai
    dans_bande = np.abs(v - estimateur.speed(x)) <= demi_largeur
    print(f"bande ±{N_SIGMA} sigma global = ±{demi_largeur:.1f} km/h : {np.mean(dans_bande):.0%} des mesures dedans")
    print(f"graine utilisee : {graine}   (pour regenerer : GRAINE = {graine})")

    # 4) Tracer
    plt.figure(figsize=(9, 5))
    plt.scatter(x, v, s=25, c="red", marker="+", linewidths=0.8, alpha=0.8, label=f"mesures ({N_POINTS} voitures)")
    plt.plot(grille, vrai.speed(grille), "b-", lw=2, label="vraie vitesse v(x)")
    plt.plot(grille, centre, "g--", lw=2, label=f"estimation (degre {D_ESTIME})")
    plt.fill_between(grille, centre - demi_largeur, centre + demi_largeur, color="green", alpha=0.15,
                 label=f"estimation ± {N_SIGMA:g} sigma global")
    plt.xlabel("position x [m]")
    plt.ylabel("vitesse [km/h]")
    plt.title(f"Maximum de vraisemblance ({ESTIMATEUR}, base {BASE}) : sigma_i dans [{ALPHA1}, {ALPHA2}] (graine {graine})")
    plt.grid(alpha=0.3)
    plt.legend()
    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(dossier / "04_maximum_vraisemblance.png", dpi=150)
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
