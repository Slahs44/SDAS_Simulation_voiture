"""
noise.py - Le bruit aléatoire
=============================

Le bruit représente tout ce qui fait que la vitesse MESURÉE n'est pas exactement la vraie vitesse :

        vitesse mesurée = vraie vitesse + bruit

La classe `Noise` sait tirer n valeurs de bruit selon une LOI que l'on choisit.
Pour l'instant une seule loi existe : la loi normale (« gaussienne », la courbe en cloche).

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

    def __init__(self, law="normale", sigma=1.0, rng=None):
        """Le constructeur : appelé automatiquement quand on écrit Noise(...).

        law   : nom de la loi (pour l'instant uniquement "normale").
        sigma : « force » du bruit. Pour la loi normale c'est l'écart-type : environ 68 % des valeurs
                tombent entre -sigma et +sigma, et 95 % entre -2·sigma et +2·sigma.
        rng   : le générateur de nombres aléatoires. On le crée une seule fois dans le programme
                avec np.random.default_rng(graine) ; avec la même graine on obtient les mêmes
                nombres à chaque lancement (résultats reproductibles).
        """
        if law not in LOIS_DISPONIBLES:
            raise ValueError(f"loi inconnue : '{law}'. Lois disponibles : {LOIS_DISPONIBLES}")
        if sigma < 0:
            raise ValueError("sigma doit être positif ou nul")

        # On range les paramètres dans l'objet (self) pour que les autres méthodes puissent les utiliser.
        self.law = law
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, n):
        """Tire n valeurs de bruit selon la loi choisie et les renvoie dans un tableau numpy."""
        if self.law == "normale":
            # Loi normale de moyenne 0 et d'écart-type sigma
            return self.rng.normal(0.0, self.sigma, n)

        # (Ici on ajoutera plus tard : elif self.law == "uniforme": return ...)
        raise ValueError(f"loi non implémentée : {self.law}")
