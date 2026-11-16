"""Main 3 - La simulation complète : polynôme (vraie vitesse) + bruit -> points mesurés.
Un point = une voiture ; chaque voiture a son propre écart-type sigma_i (99,7 % entre ALPHA1 et ALPHA2).
On trace les points (ce que le système reçoit), colorés selon leur sigma_i, et la vraie courbe (ce qu'on cherchera à retrouver)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import Noise, SpeedPolynomial, create_rng, generate_positions, simulate_speeds

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LONGUEUR = 1500.0                    # longueur du tronçon [m]
V0 = 40.0                            # vitesse au début du tronçon [km/h]
VL = 35.0                            # vitesse à la fin du tronçon [km/h] (doit être <= V_LIMITE)
V_LIMITE = 90.0                      # limite de vitesse du tronçon [km/h]
D = 6                                # degré du polynôme
A_MAX_G = 0.3                        # accélération maximale autorisée [g] (voiture normale : 0,1 à 0,3 g)
COEFS_LIBRES = None                  # None = calcul automatique ; ou une liste [c2, ..., cd] (D-1 valeurs) à la main
LOI = "normale"                      # loi du bruit
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2 (les autres ramenés aux bornes)
N_POINTS = 100000                       # nombre de points mesurés
GRAINE = None                        # None = aléatoire à chaque lancement (la graine utilisée est affichée) ; un entier (ex. 42) = toujours les mêmes données
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)                # UN seul générateur aléatoire pour tout le programme, et la graine utilisée

    # 1) La vraie vitesse, et le bruit
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE, max_acceleration_g=A_MAX_G)
    bruit = Noise(LOI, rng=rng, sigma_range=(ALPHA1, ALPHA2))

    # 2) Les points mesurés : positions au hasard, puis vitesse = vraie vitesse + bruit
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    v = simulate_speeds(poly, bruit, x)
    sigmas = bruit.last_sigmas                      # le sigma_i de chaque voiture

    print("coefficients vrais c0..cd :", poly.coefficients)
    print(f"sigma_i : moyenne {sigmas.mean():.2f}, min {sigmas.min():.2f}, max {sigmas.max():.2f} km/h")
    print(f"{N_POINTS} points simules ; premiers points (x en m, sigma_i et v en km/h) :")
    for xi, si, vi in list(zip(x, sigmas, v))[:5]:
        print(f"   x = {xi:7.1f}   sigma_i = {si:4.2f}   v = {vi:5.2f}")

    # 3) Tracer
    grille = np.linspace(0, LONGUEUR, 300)
    plt.figure(figsize=(8, 4.5))
    # Réglages des « + » :
    #   s          : taille du marqueur (c'est une aire : doubler la longueur = multiplier s par 4)
    #   linewidths : épaisseur du trait du « + » (0.5 = très fin, 2 = épais)
    #   alpha      : opacité de 0 (invisible) à 1 (opaque) ; utile quand les points se superposent
    #   c, cmap    : la couleur de chaque « + » dépend de son sigma_i (jaune = voiture peu précise)
    points = plt.scatter(x, v, s=25, c=sigmas, cmap="plasma", marker="+", linewidths=0.8, alpha=0.8,
                         label=f"mesures ({N_POINTS} voitures)")
    plt.colorbar(points, label="sigma_i de la voiture [km/h]")
    plt.plot(grille, poly.speed(grille), "b-", lw=2, label="vraie vitesse v(x)")
    plt.xlabel("position x [m]")
    plt.ylabel("vitesse [km/h]")
    plt.title(f"Simulation : degre {D}, bruit {LOI}, sigma_i dans [{ALPHA1}, {ALPHA2}] (graine {graine})")
    plt.grid(alpha=0.3)
    plt.legend()

    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    plt.savefig(dossier / "03_simulation.png", dpi=150)
    print(f"graine utilisee : {graine}   (pour regenerer exactement les memes donnees : GRAINE = {graine})")
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
