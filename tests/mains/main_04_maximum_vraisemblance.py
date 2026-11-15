"""Main 4 - Retrouver le polynôme par maximum de vraisemblance.
On simule des mesures (polynôme + bruit normal), puis on estime les coefficients et sigma à partir des seules
mesures, et on compare à la vérité."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import MaximumLikelihood, Noise, SpeedPolynomial, create_rng, generate_positions, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LONGUEUR = 2000.0                    # longueur du tronçon [m]
V0 = 70.0                            # vitesse au début du tronçon [km/h]
VL = 80.0                            # vitesse à la fin du tronçon [km/h]
V_LIMITE = 90.0                      # limite de vitesse [km/h]
D = 2                                # degré du VRAI polynôme
COEFS_LIBRES = None                  # None = calcul automatique ; ou une liste [c2, ..., cd]
SIGMA = 5.0                          # écart-type du bruit [km/h]
N_POINTS = 100                       # nombre de points mesurés
D_ESTIME = 2                         # degré utilisé pour l'estimation (en vrai il est « connu » : = D ; essaie D+1 ou D-1)
GRAINE = None                        # None = aléatoire (graine affichée) ; un entier = reproductible
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)

    # 1) Simuler les mesures à partir du vrai polynôme
    vrai = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE)
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    v = simulate_speeds(vrai, Noise("normale", SIGMA, rng), x)

    # 2) Estimer par maximum de vraisemblance (on n'utilise que x et v, pas le vrai polynôme)
    estimateur = MaximumLikelihood(LONGUEUR, D_ESTIME).fit(x, v)

    # 3) Comparer à la vérité
    print("coefficients vrais    :", np.round(vrai.coefficients, 2))
    print("coefficients estimes  :", np.round(estimateur.coefficients, 2))
    print(f"sigma vrai = {SIGMA:.2f} km/h   sigma estime = {estimateur.sigma:.2f} km/h")
    grille = np.linspace(0, LONGUEUR, 300)
    erreur = estimateur.speed(grille) - vrai.speed(grille)
    print(f"erreur sur la courbe : RMSE = {np.sqrt(np.mean(erreur ** 2)):.2f} km/h, erreur max = {np.abs(erreur).max():.2f} km/h")
    print(f"graine utilisee : {graine}   (pour regenerer : GRAINE = {graine})")

    # 4) Tracer
    plt.figure(figsize=(9, 5))
    plt.scatter(x, v, s=25, c="red", marker="+", linewidths=0.8, alpha=0.8, label=f"mesures ({N_POINTS} points)")
    plt.plot(grille, vrai.speed(grille), "b-", lw=2, label="vraie vitesse v(x)")
    plt.plot(grille, estimateur.speed(grille), "g--", lw=2, label=f"estimation (degre {D_ESTIME})")
    plt.xlabel("position x [m]")
    plt.ylabel("vitesse [km/h]")
    plt.title(f"Maximum de vraisemblance : sigma = {SIGMA} (graine {graine})")
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
