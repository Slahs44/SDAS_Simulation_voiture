"""Main 1 - Le polynôme de vitesse : on fixe V0, VL, le degré d, la limite de vitesse et l'accélération
maximale. Les coefficients libres sont calculés automatiquement : le polynôme reste le plus près possible
de la limite (par ex. 90 km/h) sans la dépasser et sans accélération trop forte.
Deux graphiques : la vitesse v(x) et l'accélération a(x)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import SpeedPolynomial

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LONGUEUR = 1000.0                    # longueur du tronçon [m]
V0 = 35.0                            # vitesse au début du tronçon [km/h]
VL = 20.0                            # vitesse à la fin du tronçon [km/h] (doit être <= V_LIMITE)
V_LIMITE = 90.0                      # limite de vitesse du tronçon [km/h]
D = 6                                # degré du polynôme (essaie 3 ou 4 : la courbe colle plus longtemps à 90)
A_MAX_G = 0.3                        # accélération maximale autorisée [g] (1 g = 9,81 m/s² ; voiture normale : 0,1 à 0,3 g)
COEFS_LIBRES = None                  # None = calcul automatique ; ou une liste [c2, ..., cd] (D-1 valeurs) à la main
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    # 1) Construire le polynôme : les coefficients c0, c1, ..., cd sont calculés automatiquement
    poly = SpeedPolynomial(LONGUEUR, V0, VL, D, COEFS_LIBRES, speed_limit=V_LIMITE, max_acceleration_g=A_MAX_G)
    print("coefficients c0, c1, ..., cd :", np.round(poly.coefficients, 2))
    print(f"v(0) = {poly.speed(0.0):.2f} km/h   (doit valoir V0 = {V0})")
    print(f"v(L) = {poly.speed(LONGUEUR):.2f} km/h   (doit valoir VL = {VL})")

    x = np.linspace(0, LONGUEUR, 300)                  # 300 positions régulières entre 0 et L
    v_max, x_max = poly.max_speed()
    a = poly.acceleration(x)
    print(f"vitesse maximale = {v_max:.2f} km/h, atteinte en x = {x_max:.0f} m   (limite = {V_LIMITE})")
    print(f"part du tronçon à moins de 2 km/h de la limite : {np.mean(poly.speed(x) >= V_LIMITE - 2):.0%}")
    print(f"accélération max |a| = {np.abs(a).max():.3f} m/s² = {np.abs(a).max() / 9.81:.3f} g   (limite = {A_MAX_G} g)")

    # 2) Tracer : vitesse (gauche) et accélération (droite)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axs[0]
    ax.plot(x, poly.speed(x), "r-", lw=2, label=f"v(x), degre {D}")
    ax.plot([0, LONGUEUR], [V0, VL], "ko", label="(0, V0) et (L, VL)")
    ax.plot(x_max, v_max, "g^", ms=9, label=f"maximum ({v_max:.1f} km/h)")
    ax.axhline(V_LIMITE, color="gray", ls="--", label=f"limite {V_LIMITE:.0f} km/h")
    ax.set(xlabel="position x [m]", ylabel="vitesse [km/h]", title="Polynome de vitesse")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1]
    ax.plot(x, a, "b-", lw=2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="position x [m]", ylabel="acceleration [m/s²]",
           title=f"Acceleration a = v · dv/dx   (limite autorisee : ±{A_MAX_G} g)")
    ax.grid(alpha=0.3)

    # 3) Sauvegarder dans figures/ (à la racine du projet), puis afficher si demandé
    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(dossier / "01_polynome.png", dpi=150)
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
