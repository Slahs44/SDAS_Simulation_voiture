"""
noise.py - Le bruit aléatoire
=============================

Le bruit représente tout ce qui fait que la vitesse MESURÉE n'est pas exactement la vraie vitesse :

        vitesse mesurée = vraie vitesse + bruit

La classe `Noise` sait tirer n valeurs de bruit selon une LOI que l'on choisit.
Pour l'instant une seule loi existe : la loi normale (« gaussienne », la courbe en cloche).

Deux façons de choisir l'écart-type du bruit :
  - sigma FIXE   : Noise("normale", sigma=5.0)  -> tous les points ont le même écart-type ;
  - sigma VARIABLE (une voiture = un point = son propre capteur) : Noise("normale", sigma_range=(alpha1, alpha2)).
    Pour chaque point i on tire d'abord son écart-type sigma_i selon une loi normale qui met 99,7 % des
    sigma_i entre alpha1 et alpha2 (règle des 3 sigma : alpha1 = moyenne - 3·s, alpha2 = moyenne + 3·s) :
            moyenne = (alpha1 + alpha2) / 2        s = (alpha2 - alpha1) / 6
    puis on tire le bruit du point i selon Normale(0, sigma_i²).

Pour ajouter une loi plus tard (par exemple "uniforme") il y a 2 endroits à modifier :
  1) la liste LOIS_DISPONIBLES juste en dessous : y ajouter le nom de la loi ;
  2) la méthode `sample` : y ajouter un bloc `elif self.law == "uniforme": ...`
"""
import numpy as np

# Les lois que la classe connaît. Pour en ajouter une : ajouter son nom ici ET un `elif` dans `sample`.
LOIS_DISPONIBLES = ("normale",)


class Noise:
    """Un générateur de bruit.

    Exemple d'utilisation :
        rng = np.random.default_rng(42)                 # le générateur aléatoire (voir plus bas)
        bruit = Noise(law="normale", sigma=2.0, rng=rng)
        valeurs = bruit.sample(100)                     # 100 valeurs de bruit
    """

    def __init__(self, law="normale", sigma=1.0, rng=None, sigma_range=None):
        """Le constructeur : appelé automatiquement quand on écrit Noise(...).

        law   : nom de la loi (pour l'instant uniquement "normale").
        sigma : « force » du bruit. Pour la loi normale c'est l'écart-type : environ 68 % des valeurs
                tombent entre -sigma et +sigma, et 95 % entre -2·sigma et +2·sigma.
        rng   : le générateur de nombres aléatoires. On le crée une seule fois dans le programme
                avec np.random.default_rng(graine) ; avec la même graine on obtient les mêmes
                nombres à chaque lancement (résultats reproductibles).
        sigma_range : None (par défaut) -> on utilise `sigma` pour tous les points.
                      (alpha1, alpha2)  -> chaque point a son propre sigma_i, tiré selon
                      Normale(moyenne = (alpha1+alpha2)/2, écart-type = (alpha2-alpha1)/6) :
                      99,7 % des sigma_i tombent entre alpha1 et alpha2. Dans ce cas `sigma` est ignoré.
        """
        if law not in LOIS_DISPONIBLES:
            raise ValueError(f"loi inconnue : '{law}'. Lois disponibles : {LOIS_DISPONIBLES}")
        if sigma < 0:
            raise ValueError("sigma doit être positif ou nul")

        # On range les paramètres dans l'objet (self) pour que les autres méthodes puissent les utiliser.
        self.law = law
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.default_rng()

        # Écart-type variable par point : on vérifie et on range alpha1, alpha2
        self.sigma_range = None
        if sigma_range is not None:
            alpha1, alpha2 = sigma_range
            if not 0 <= alpha1 < alpha2:
                raise ValueError("sigma_range = (alpha1, alpha2) doit vérifier 0 <= alpha1 < alpha2")
            self.sigma_range = (float(alpha1), float(alpha2))
        self.last_sigmas = None        # les sigma_i du dernier tirage (pratique pour les afficher / tracer)

    def sample_sigmas(self, n):
        """Tire n écarts-types sigma_i (un par point / par voiture), indépendants.

        sigma_i ~ Normale(moyenne = (alpha1+alpha2)/2, écart-type = (alpha2-alpha1)/6)
        -> 99,7 % des sigma_i sont entre alpha1 et alpha2.
        Les rares sigma_i qui sortent de [alpha1, alpha2] (environ 0,3 %) sont ramenés à la borne la plus
        proche : < alpha1 -> alpha1, > alpha2 -> alpha2. Comme alpha1 >= 0, un sigma_i n'est jamais négatif."""
        alpha1, alpha2 = self.sigma_range
        moyenne = (alpha1 + alpha2) / 2
        ecart_type = (alpha2 - alpha1) / 6            # entre alpha1 et alpha2 il y a 6 écarts-types
        sigmas = self.rng.normal(moyenne, ecart_type, n)
        return np.clip(sigmas, alpha1, alpha2)        # hors plage -> borne la plus proche

    def sample(self, n):
        """Tire n valeurs de bruit selon la loi choisie et les renvoie dans un tableau numpy."""
        # L'écart-type de chaque point : le même pour tous, ou un sigma_i tiré pour chaque point
        if self.sigma_range is None:
            sigmas = np.full(n, float(self.sigma))
        else:
            sigmas = self.sample_sigmas(n)
        self.last_sigmas = sigmas

        if self.law == "normale":
            # Loi normale de moyenne 0 ; le point i a l'écart-type sigmas[i]
            return self.rng.normal(0.0, sigmas)

        # (Ici on ajoutera plus tard : elif self.law == "uniforme": return ...)
        raise ValueError(f"loi non implémentée : {self.law}")


# ----------------------------------------------------------------------
# Vérification rapide : lancer  python src/vitesse/noise.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ALPHA1, ALPHA2 = 2.0, 8.0            # plage des sigma_i [km/h]
    N = 100_000                          # nombre de voitures (= de points)
    GRAINE = 0

    bruit = Noise("normale", rng=np.random.default_rng(GRAINE), sigma_range=(ALPHA1, ALPHA2))
    valeurs = bruit.sample(N)
    s = bruit.last_sigmas

    print(f"sigma_i : moyenne {s.mean():.3f} (attendu {(ALPHA1 + ALPHA2) / 2:.3f}), "
          f"ecart-type {s.std():.3f} (attendu {(ALPHA2 - ALPHA1) / 6:.3f})")
    print(f"sigma_i dans [{ALPHA1}, {ALPHA2}] : {np.mean((s >= ALPHA1) & (s <= ALPHA2)):.2%} (100 % : les 0,27 % hors plage sont ramenes aux bornes)")
    print(f"sigma_i min {s.min():.3f}, max {s.max():.3f} (bornes alpha1, alpha2)")
    print(f"bruit : moyenne {valeurs.mean():.3f} (attendu 0), "
          f"variance {valeurs.var():.2f} (attendu moyenne(sigma_i^2) = {np.mean(s ** 2):.2f})")
