"""Main 2 - Le bruit : on tire beaucoup de valeurs de bruit et on regarde leur répartition
(histogramme). Pour la loi normale on doit voir une courbe en cloche centrée sur 0."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import Noise, create_rng

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LOI = "normale"                      # loi du bruit (pour l'instant : "normale")
SIGMA = 5.0                          # écart-type du bruit [km/h]
N_TIRAGES = 10_000                   # nombre de valeurs de bruit tirées
GRAINE = None                        # None = aléatoire à chaque lancement (la graine utilisée est affichée) ; un entier (ex. 42) = toujours le même tirage
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)                # le générateur aléatoire, et la graine réellement utilisée
    bruit = Noise(LOI, SIGMA, rng)
    valeurs = bruit.sample(N_TIRAGES)

    print(f"moyenne      : {valeurs.mean():.3f}   (attendu : 0)")
    print(f"ecart-type   : {valeurs.std():.3f}   (attendu : {SIGMA})")
    print(f"min / max    : {valeurs.min():.2f} / {valeurs.max():.2f}")

    plt.figure(figsize=(7, 4.5))
    plt.hist(valeurs, bins=60, color="tab:blue", alpha=0.8)
    plt.xlabel("valeur du bruit [km/h]")
    plt.ylabel("nombre de tirages")
    plt.title(f"Bruit : loi {LOI}, sigma = {SIGMA} ({N_TIRAGES} tirages, graine {graine})")
    plt.grid(alpha=0.3)

    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    plt.savefig(dossier / "02_bruit.png", dpi=150)
    print(f"graine utilisee : {graine}   (pour regenerer exactement le meme tirage : GRAINE = {graine})")
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
