"""Main 3 - La simulation complète : polynôme (vraie vitesse) + bruit -> points mesurés.
On trace les points (ce que le système reçoit) et la vraie courbe (ce qu'on cherchera à retrouver)."""
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
D = 4                                # degré du polynôme
A_MAX_G = 0.3                        # accélération maximale autorisée [g] (voiture normale : 0,1 à 0,3 g)
COEFS_LIBRES = None                  # None = calcul automatique ; ou une liste [c2, ..., cd] (D-1 valeurs) à la main
LOI = "normale"                      # loi du bruit
SIGMA = 15.0                          # écart-type du bruit [km/h]
N_POINTS = 1000                       # nombre de points mesurés
GRAINE = None                        # None = aléatoire à chaque lancement (la graine utilisée est affichée) ; un entier (ex. 42) = toujours les mêmes données
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)                # UN seul générateur aléatoire pour tout le programme, et la graine utilisée

    # 1) La vraie vitesse, et le bruit
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE, max_acceleration_g=A_MAX_G)
    bruit = Noise(LOI, SIGMA, rng)

    # 2) Les points mesurés : positions au hasard, puis vitesse = vraie vitesse + bruit
    x = generate_positions(N_POINTS, LONGUEUR, rng)
    v = simulate_speeds(poly, bruit, x)

    print("coefficients vrais c0..cd :", poly.coefficients)
    print(f"{N_POINTS} points simules ; premiers points (x en m, v en km/h) :")
    for xi, vi in list(zip(x, v))[:5]:
        print(f"   x = {xi:7.1f}   v = {vi:5.2f}")

    # 3) Tracer
    grille = np.linspace(0, LONGUEUR, 300)
    plt.figure(figsize=(8, 4.5))
    # Réglages des « + » :
    #   s          : taille du marqueur (c'est une aire : doubler la longueur = multiplier s par 4)
    #   linewidths : épaisseur du trait du « + » (0.5 = très fin, 2 = épais)
    #   alpha      : opacité de 0 (invisible) à 1 (opaque) ; utile quand les points se superposent
    plt.scatter(x, v, s=25, c="red", marker="+", linewidths=0.8, alpha=0.8, label=f"mesures ({N_POINTS} points)")
    plt.plot(grille, poly.speed(grille), "b-", lw=2, label="vraie vitesse v(x)")   # courbe bleue (les croix sont rouges)
    plt.xlabel("position x [m]")
    plt.ylabel("vitesse [km/h]")
    plt.title(f"Simulation : degre {D}, bruit {LOI} sigma = {SIGMA} (graine {graine})")
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
