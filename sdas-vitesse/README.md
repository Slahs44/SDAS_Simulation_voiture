# SDAS - Estimation robuste de la vitesse v(x) sur un troncon

Projet du cours *Sciences des Donnees et apprentissage Statistique* (Centrale Mediterranee).
Objectif : estimer le profil de vitesse v(x) (polynome de degre d connu) a partir de points
(position, vitesse) envoyes par des vehicules, puis le tracer sur le troncon.
Formulation du probleme et questions ouvertes : [docs/formulation.md](docs/formulation.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt  # installe aussi le package `vitesse` (src/) en editable
```

## Structure

```
src/vitesse/
  basis.py       bases polynomiales (canonique, Legendre)        -> matrice de conception A
  profile.py     SpeedProfile : v(x) polynomial sur [0, L]
  noise.py       modeles de bruit (gaussien, contamine, effet vehicule)
  simulator.py   generation des donnees (N points, M vehicules), echantillonneurs de positions
  estimators.py  Estimator (abstrait), OLSEstimator (moindres carres)
  evaluation.py  borne de Cramer-Rao, Monte-Carlo
tests/
  test_*.py      tests unitaires (pytest)
  mains/main_*.py  scripts executables, un par experience
figures/         figures produites par les mains (a ajouter au .gitignore)
```

## Utilisation

```bash
pytest                                        # tests unitaires
python tests/mains/main_01_un_tirage.py       # un tirage + trace de v(x)
python tests/mains/main_02_monte_carlo.py     # biais / covariance vs Cramer-Rao
python tests/mains/main_03_sensibilite_sigma_N.py
python tests/mains/main_04_conditionnement.py
python tests/mains/main_05_points_bloquants.py
```
Options communes : `--seed 42`, `--show` (afficher les figures au lieu de les sauvegarder seulement).

## Ajouter un estimateur robuste

Heriter de `Estimator` et implementer `_solve(A, v)` (voir `OLSEstimator`), puis le brancher
dans `MonteCarlo(simulator, estimator)` : tous les mains et les comparaisons fonctionnent sans autre changement.
Pour un nouveau type de bruit : heriter de `NoiseModel` et implementer `sample`.
