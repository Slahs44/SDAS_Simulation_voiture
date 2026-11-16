# SDAS - Simulation de la vitesse v(x) sur un tronçon

Les vitesses sont en **km/h**, les positions en **mètres**.

On **génère des données simulées** = polynôme de vitesse (qui part de V0 et arrive à VL) + bruit, avec un point par
voiture et un écart-type σᵢ propre à chaque voiture (99,7 % des σᵢ entre α1 et α2). Puis on **retrouve** v(x)
à partir des seuls points, par maximum de vraisemblance (méthode du TP2 : ĉ = M⁻¹·w, borne de Cramér-Rao).
Explication détaillée : [docs/explications.md](docs/explications.md)
Formulation et questions pour le professeur : [docs/formulation.md](docs/formulation.md)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Le code (dossier src/vitesse/)
| Fichier | Contenu |
|---|---|
| `noise.py` | classe `Noise` : tire du bruit selon une loi choisie (pour l'instant : "normale"), avec un écart-type fixe (`sigma`) ou un écart-type σᵢ propre à chaque voiture (`sigma_range=(alpha1, alpha2)`) |
| `polynomial.py` | classe `SpeedPolynomial` : calcule c0..cd pour que v(0) = V0 et v(L) = VL. Si on ne donne pas les coefficients libres, ils sont choisis automatiquement (polynôme le plus proche de la limite de vitesse, sans la dépasser, accélération bornée) |
| `polynomial_tools.py` | fonctions utilisées par `SpeedPolynomial` : calcul des coefficients, vitesse, accélération (a = v·dv/dx), et recherche automatique des coefficients libres |
| `measurements.py` | 2 fonctions : `generate_positions` (positions au hasard) et `simulate_speeds` (vitesse + bruit) |
| `maximum_likelihood.py` | classe `MaximumLikelihood` : estime c0..cd (base `"canonique"` ou `"legendre"`) à partir des mesures par la méthode du TP2 (ĉ = M⁻¹·w), avec deux estimateurs : `"sigma_constant"` (un seul σ, estimé) ou `"sigma_variable"` (σ_i connus, moindres carrés pondérés par 1/σ_i²) ; donne aussi l'information de Fisher et la borne de Cramér-Rao |

## Utilisation
```bash
pytest                                      # tests unitaires
python tests/mains/main_01_polynome.py      # le polynôme seul
python tests/mains/main_02_bruit.py         # le bruit seul : histogramme des sigma_i par voiture + histogramme du bruit
python tests/mains/main_03_simulation.py    # polynôme + bruit = points mesurés (couleur = sigma_i de la voiture)
python tests/mains/main_04_maximum_vraisemblance.py   # retrouver v(x) par maximum de vraisemblance
python tests/mains/main_05_comparaison_estimateurs.py # les 2 estimateurs (sigma constant / sigma_i connus) sur un même graphe
python src/vitesse/noise.py                 # vérification du sigma variable par voiture (règle des 3 sigma)
```
Chaque main (et chaque test) commence par une section **PARAMETRES** : c'est là qu'on change les valeurs.
Les figures sont sauvegardées dans `figures/` (à ajouter au `.gitignore`).
