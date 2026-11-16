"""Main 2 - Le bruit, avec un écart-type sigma_i propre à chaque voiture (un point = une voiture).

Pour chaque voiture i :
  1) on tire son écart-type sigma_i ~ Normale((alpha1+alpha2)/2, ((alpha2-alpha1)/6)²)
     -> 99,7 % des sigma_i entre alpha1 et alpha2 ; les autres sont ramenés à la borne la plus proche ;
  2) on tire son bruit ~ Normale(0, sigma_i²).

Deux graphiques :
  - à gauche  : l'histogramme des sigma_i (cloche entre alpha1 et alpha2) ;
  - à droite  : l'histogramme du bruit obtenu, comparé à une loi normale de même variance.
    Comme chaque voiture a son propre sigma_i, le bruit n'est plus exactement normal : il a des
    « queues » un peu plus épaisses (plus de grands écarts que la cloche de comparaison)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitesse import Noise, create_rng

# ----------------------------------------------------------------------
# 1. PARAMETRES  (modifie ici, pas dans le reste du script)
# ----------------------------------------------------------------------
LOI = "normale"                      # loi du bruit (pour l'instant : "normale")
ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i [km/h] : 99,7 % des sigma_i entre ALPHA1 et ALPHA2
N_TIRAGES = 10_000                   # nombre de voitures (= de valeurs de bruit tirées)
GRAINE = None                        # None = aléatoire à chaque lancement (la graine utilisée est affichée) ; un entier (ex. 42) = toujours le même tirage
AFFICHER = True                      # True : ouvre la fenêtre du graphique ; False : sauvegarde seulement


def main():
    rng, graine = create_rng(GRAINE)                # le générateur aléatoire, et la graine réellement utilisée
    bruit = Noise(LOI, rng=rng, sigma_range=(ALPHA1, ALPHA2))
    valeurs = bruit.sample(N_TIRAGES)
    sigmas = bruit.last_sigmas                       # le sigma_i de chaque voiture

    moyenne_attendue = (ALPHA1 + ALPHA2) / 2
    ecart_type_attendu = (ALPHA2 - ALPHA1) / 6
    n_aux_bornes = np.sum((sigmas == ALPHA1) | (sigmas == ALPHA2))
    print("sigma_i :")
    print(f"   moyenne    : {sigmas.mean():.3f}   (attendu : {moyenne_attendue})")
    print(f"   ecart-type : {sigmas.std():.3f}   (attendu : ~{ecart_type_attendu:.3f})")
    print(f"   min / max  : {sigmas.min():.2f} / {sigmas.max():.2f}   ({n_aux_bornes} voitures ramenees aux bornes,"
          f" soit {n_aux_bornes / N_TIRAGES:.2%} ; attendu ~0,27 %)")
    print("bruit :")
    print(f"   moyenne    : {valeurs.mean():.3f}   (attendu : 0)")
    print(f"   ecart-type : {valeurs.std():.3f}   (attendu : racine(moyenne des sigma_i^2) = {np.sqrt(np.mean(sigmas ** 2)):.3f})")
    print(f"   min / max  : {valeurs.min():.2f} / {valeurs.max():.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- À gauche : les sigma_i ---
    ax1.hist(sigmas, bins=60, color="tab:orange", alpha=0.8)
    ax1.axvline(ALPHA1, color="k", ls="--", lw=1.2, label=f"alpha1 = {ALPHA1}")
    ax1.axvline(ALPHA2, color="k", ls=":", lw=1.2, label=f"alpha2 = {ALPHA2}")
    ax1.set_xlabel("sigma_i de la voiture [km/h]")
    ax1.set_ylabel("nombre de voitures")
    ax1.set_title(f"Ecarts-types sigma_i : moyenne {moyenne_attendue}, ecart-type {ecart_type_attendu:.2f}")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # --- À droite : le bruit obtenu, comparé à une cloche normale de même écart-type ---
    ax2.hist(valeurs, bins=60, density=True, color="tab:blue", alpha=0.8, label="bruit simule")
    s_total = valeurs.std()
    grille = np.linspace(valeurs.min(), valeurs.max(), 300)
    cloche = np.exp(-grille ** 2 / (2 * s_total ** 2)) / (s_total * np.sqrt(2 * np.pi))
    ax2.plot(grille, cloche, "r-", lw=1.5, label=f"normale de meme ecart-type ({s_total:.2f})")
    ax2.set_xlabel("valeur du bruit [km/h]")
    ax2.set_ylabel("densite")
    ax2.set_title(f"Bruit : loi {LOI}, sigma_i dans [{ALPHA1}, {ALPHA2}]")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f"{N_TIRAGES} voitures, graine {graine}")
    fig.tight_layout()

    dossier = Path(__file__).resolve().parents[2] / "figures"
    dossier.mkdir(exist_ok=True)
    plt.savefig(dossier / "02_bruit.png", dpi=150)
    print(f"graine utilisee : {graine}   (pour regenerer exactement le meme tirage : GRAINE = {graine})")
    if AFFICHER:
        plt.show()


if __name__ == "__main__":
    main()
