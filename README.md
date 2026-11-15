# SDAS - Simulation de la vitesse v(x) sur un tronçon

Les vitesses sont en **km/h**, les positions en **mètres**.

Étape actuelle : **générer des données simulées** = polynôme de vitesse (qui part de V0 et arrive à VL) + bruit.
(L'estimation de v(x) à partir des points viendra à l'étape suivante.)
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
| `noise.py` | classe `Noise` : tire du bruit selon une loi choisie (pour l'instant : "normale") |
| `polynomial.py` | classe `SpeedPolynomial` : calcule c0..cd pour que v(0) = V0 et v(L) = VL. Si on ne donne pas les coefficients libres, ils sont choisis automatiquement (polynôme le plus proche de la limite de vitesse, sans la dépasser, accélération bornée) |
| `polynomial_tools.py` | fonctions utilisées par `SpeedPolynomial` : calcul des coefficients, vitesse, accélération (a = v·dv/dx), et recherche automatique des coefficients libres |
| `measurements.py` | 2 fonctions : `generate_positions` (positions au hasard) et `simulate_speeds` (vitesse + bruit) |

## Utilisation
```bash
pytest                                      # tests unitaires
python tests/mains/main_01_polynome.py      # le polynôme seul
python tests/mains/main_02_bruit.py         # le bruit seul (histogramme)
python tests/mains/main_03_simulation.py    # polynôme + bruit = points mesurés
```
Chaque main (et chaque test) commence par une section **PARAMETRES** : c'est là qu'on change les valeurs.
Les figures sont sauvegardées dans `figures/` (à ajouter au `.gitignore`).
