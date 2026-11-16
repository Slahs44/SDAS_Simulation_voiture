# Explication : comment les données simulées sont générées

Ce document explique la génération sur un petit exemple : $L = 2000$ m, $V_0 = 70$ km/h, $V_L = 80$ km/h,
limite 90 km/h, degré $d = 2$, écarts-types $\sigma_i \in [\alpha_1, \alpha_2] = [2, 8]$ km/h, 100 points (= 100 voitures).
(Les valeurs par défaut de `main_03_simulation.py` sont différentes : $L = 1500$ m, $V_0 = 40$, $V_L = 35$ km/h,
$d = 4$, 1000 points ; même plage $[2, 8]$ km/h.)

## 0. Vue d'ensemble

### Modélisation du problème

On considère $N$ véhicules indicés par $i \in \{1, \dots, N\}$. Chaque véhicule envoie **un seul** point sur le
tronçon : les $N$ points sont donc indépendants (voir `formulation.md`).

- $P$ est le polynôme de vitesse : la **vraie** vitesse en fonction de la position, sans bruit (section 1).
- $\sigma_i$ est l'écart-type du bruit propre au véhicule $i$ ($\sigma_i^2$ est sa variance), tiré aléatoirement selon une loi normale bornée à $[\alpha_1, \alpha_2]$ (voir ci-dessous).
- $x_i$ est la position du point envoyé par le véhicule $i$.
- $v_i$ est la vitesse mesurée du véhicule $i$ :

$$
v_i = P(x_i) + b_i,
\qquad
b_i \sim \mathcal{N}\left(0,\ \sigma_i^2\right)
$$

### Tirage de l'écart-type $\sigma_i$

On se donne deux bornes $0 < \alpha_1 < \alpha_2$ pour l'écart-type des véhicules. On veut que 99,7 % des tirages
tombent entre $\alpha_1$ et $\alpha_2$, c'est-à-dire que $[\alpha_1, \alpha_2]$ soit l'intervalle $[\mu - 3s,\ \mu + 3s]$
d'une loi normale de moyenne $\mu$ et d'écart-type $s$ :

$$
\mu = \frac{\alpha_1 + \alpha_2}{2},
\qquad
s = \frac{\alpha_2 - \alpha_1}{6},
\qquad
S_i \sim \mathcal{N}\left(\mu,\ s^2\right)
$$

Il reste 0,3 % des tirages en dehors de $[\alpha_1, \alpha_2]$ (environ 0,15 % de chaque côté). On les ramène sur la borne
la plus proche :

$$
\sigma_i = \min\left(\max\left(S_i,\ \alpha_1\right),\ \alpha_2\right)
=
\begin{cases}
\alpha_1 & \text{si } S_i < \alpha_1 \\
S_i & \text{si } \alpha_1 \le S_i \le \alpha_2 \\
\alpha_2 & \text{si } S_i > \alpha_2
\end{cases}
$$

On a donc toujours $\sigma_i \in [\alpha_1, \alpha_2]$. En particulier $\sigma_i > 0$ dès que $\alpha_1 > 0$, ce qui garantit
un écart-type valide. Conséquence : $\sigma_i$ vaut exactement $\alpha_1$ (ou exactement $\alpha_2$) avec une probabilité
d'environ 0,13 %.

Exemple avec $\alpha_1 = 2$ et $\alpha_2 = 8$ km/h : $\mu = 5$ km/h et $s = 1$ km/h.

Dans le code (`Noise("normale", rng=rng, sigma_range=(alpha1, alpha2))`) :

```python
sigmas = np.clip(rng.normal(moyenne, ecart_type, n), alpha1, alpha2)   # Noise.sample_sigmas : un sigma_i par voiture
bruit = rng.normal(0.0, sigmas)                                        # Noise.sample : b_i ~ N(0, sigma_i²)
```

Les $\sigma_i$ tirés sont gardés dans `bruit.last_sigmas`. `python tests/mains/main_02_bruit.py` trace leur
histogramme et celui du bruit obtenu ; `python src/vitesse/noise.py` vérifie les chiffres sur 100 000 voitures.

Cas particulier : `Noise("normale", sigma=5.0)` (sans `sigma_range`) donne le même $\sigma$ à toutes les voitures.
Ce cas simple sert de référence et pour les tests (par exemple $\sigma = 0$ : pas de bruit).

### Schéma

$$
\begin{array}{lcl}
P(x) \ \text{(vraie vitesse, sans bruit)} & \searrow & \\[6pt]
x_i \sim \mathcal{U}\left([0, L]\right) & \longrightarrow & v_i = P(x_i) + b_i \ \longrightarrow\ \left(x_i, v_i\right)_{i} \ \text{(ce que le système reçoit)} \\[6pt]
b_i \sim \mathcal{N}\left(0, \sigma_i^2\right),\ \ \sigma_i \in [\alpha_1, \alpha_2] & \nearrow &
\end{array}
$$

Un seul « générateur aléatoire » (`rng`) fournit tous les nombres au hasard : d'abord les 100 positions,
ensuite les 100 écarts-types $\sigma_i$, enfin les 100 valeurs de bruit. Ce générateur est entièrement déterminé
par un nombre : **la graine** (section 3).

---

## 1. Générer le polynôme

### 1.1 La position normalisée

On remplace la position $x$ (en mètres, de 0 à $L$) par $\xi = x / L$, qui va de **0** (début) à **1** (fin).
C'est plus simple à manipuler, et le résultat ne dépend plus de la longueur du tronçon.

### 1.2 La forme du polynôme

$$
v(\xi) = c_0 + c_1 \xi + c_2 \xi^2 + \dots + c_d \xi^d
$$

### 1.3 Les deux conditions : partir de $V_0$ et arriver à $V_L$

- Au début, $\xi = 0$ : tous les termes avec $\xi$ valent 0, il reste $v(0) = c_0$. On veut $v(0) = V_0$, donc
  $$c_0 = V_0$$
- À la fin, $\xi = 1$ : $v(1) = c_0 + c_1 + c_2 + \dots + c_d$. On veut $v(1) = V_L$, donc
  $$c_1 = V_L - V_0 - (c_2 + c_3 + \dots + c_d)$$

Un polynôme de degré $d$ a $d+1$ coefficients, on en a fixé 2 ($c_0$ et $c_1$) : il reste
**$d - 1$ coefficients libres** $c_2, \dots, c_d$.

| Degré $d$ | Coefficients libres | Forme |
|---|---|---|
| 1 | aucun | droite de $V_0$ à $V_L$ |
| 2 | $c_2$ | parabole |
| 3 | $c_2$, $c_3$ | cubique |

### 1.4 Exemple chiffré ($d = 2$)

Avec $V_0 = 70$, $V_L = 80$ et $c_2 = -58{,}29$ :

- $c_0 = 70$
- $c_1 = 80 - 70 - (-58{,}29) = 68{,}29$
- donc $v(\xi) = 70 + 68{,}29\,\xi - 58{,}29\,\xi^2$

Vérifications : $v(0) = 70$ ✔ ; $v(1) = 70 + 68{,}29 - 58{,}29 = 80$ ✔ ;
au milieu du tronçon $v(0{,}5) = 70 + 34{,}14 - 14{,}57 = 89{,}57$ km/h.

### 1.5 Comment $c_2, \dots, c_d$ sont choisis

- **À la main** : `SpeedPolynomial(2000, 70, 80, 2, free_coefficients=[-58.29])`.
- **Automatiquement** (par défaut, si on donne `speed_limit`) : on cherche les coefficients libres qui
  1. **minimisent** l'écart moyen à la limite : $J = \text{moyenne de } (90 - v(x))^2$
     (la courbe reste le plus près possible de 90 km/h sur tout le tronçon) ;
  2. en respectant **en chaque point** : $0 \le v(x) \le 90$ km/h et $|a(x)| \le a_{max}$.

  L'accélération du véhicule est $a = \dfrac{dv}{dt} = v \cdot \dfrac{dv}{dx}$ (à convertir en m/s²).
  La recherche est faite par `scipy.optimize.minimize` ; s'il n'existe aucun polynôme qui respecte les contraintes,
  une erreur le dit.
  Pour $d = 2$ on obtient $c_2 = -58{,}29$, soit exactement l'exemple ci-dessus (pic de 90 km/h vers $x = 1171$ m).

### 1.6 Calculer la vitesse en un point

Une fois $[c_0, \dots, c_d]$ connus, `speed(x)` fait :

```python
xi = x / length                      # position normalisée
v = 0
for j, cj in enumerate(c):           # on ajoute chaque terme  cj · ξ^j
    v = v + cj * xi**j
```

---

## 2. Générer un point $(x, v(x))$ : loi uniforme, puis polynôme + bruit

Pour **chaque** point $k = 1, \dots, 100$ :

**Étape 1 — la position $x_k$ : loi uniforme sur $[0, L]$**

```python
x = rng.uniform(0.0, length, n_points)       # generate_positions
```

« Uniforme » : tous les endroits du tronçon ont la même chance, un intervalle de 100 m a deux fois plus de chances
qu'un intervalle de 50 m. Les 100 positions sont tirées **indépendamment** : par hasard on aura des endroits avec
beaucoup de points et d'autres avec peu.

**Étape 2 — la vraie vitesse en ce point**

$$v(x_k) = c_0 + c_1 \xi_k + c_2 \xi_k^2, \qquad \xi_k = x_k / L$$

**Étape 3 — l'écart-type $\sigma_k$ de la voiture $k$** (détails en section 0)

```python
sigmas = np.clip(rng.normal(moyenne, ecart_type, n), alpha1, alpha2)   # Noise.sample_sigmas
```

Chaque voiture a son propre capteur, plus ou moins précis : $\sigma_k$ est tiré selon $\mathcal{N}(\mu, s^2)$ avec
$\mu = 5$ km/h et $s = 1$ km/h (pour $[\alpha_1, \alpha_2] = [2, 8]$), puis ramené dans $[2, 8]$ s'il en sort.

**Étape 4 — le bruit $\varepsilon_k$ : loi normale centrée, d'écart-type $\sigma_k$**

```python
bruit = rng.normal(0.0, sigmas)              # Noise.sample, loi "normale"
```

$\varepsilon_k \sim \mathcal{N}(0, \sigma_k^2)$. Pour une voiture avec $\sigma_k = 5$ km/h, il y a environ 68 % de chances
que sa mesure s'écarte de la vraie courbe de moins de 5 km/h, et 95 % de moins de 10 km/h.
($\sigma_k$ est l'**écart-type** ; la variance vaut $\sigma_k^2$.)

**Étape 5 — la vitesse mesurée**

$$v_k = v(x_k) + \varepsilon_k$$

```python
return polynomial.speed(x) + noise.sample(len(x))     # simulate_speeds
```

### Exemple avec la graine 42 (les 3 premiers points)

| $k$ | $x_k$ (m) | $\xi_k$ | $v(x_k)$ vraie | $\sigma_k$ | $\varepsilon_k$ | $v_k$ mesurée |
|---|---|---|---|---|---|---|
| 1 | 1547,9 | 0,7740 | 87,94 | 5,40 | −0,03 | **87,91** |
| 2 | 877,8 | 0,4389 | 88,74 | 4,09 | −0,67 | **88,07** |
| 3 | 1717,2 | 0,8586 | 85,66 | 4,62 | +1,56 | **87,22** |

Les points $(x_k, v_k)$ sont ce que « reçoit » le système ; la vraie courbe $v(x)$ est ce qu'on cherchera à retrouver.
Les $\sigma_k$, eux, ne sont **pas** connus du système : il ne voit que les vitesses mesurées.

> **Ordre des tirages.** Les 100 positions sont tirées d'abord, puis les 100 écarts-types $\sigma_k$, puis les
> 100 valeurs de bruit : le générateur `rng` fournit ses nombres les uns après les autres, comme on pioche dans
> un paquet de cartes. Le bruit du point 1 est donc le 1er nombre du *troisième* lot, pas le 1er nombre du programme.

---

## 3. La graine (« seed »)

### 3.1 Un ordinateur ne tire pas vraiment au hasard

Un générateur « aléatoire » est en réalité une **formule déterministe** : à partir d'un nombre de départ,
la **graine**, elle produit une longue suite de nombres qui *ont l'air* aléatoires.
Analogie : un énorme livre de nombres au hasard ; la graine, c'est la page où l'on commence à lire.

```python
rng = np.random.default_rng(42)      # on commence à la « page » 42
```

- **Même graine → exactement la même suite** → exactement les mêmes données, à chaque lancement.
- **Graine différente → autre suite** → autres données.
- La valeur n'a aucune signification : 42 n'est pas « meilleur » que 7 (c'est juste une habitude).

### 3.2 À quoi ça sert

- **Reproduire** une figure ou un résultat (pour le rapport, pour un collègue du groupe).
- **Déboguer** : un bug qui n'apparaît qu'avec certaines données doit pouvoir être rejoué.
- **Comparer des méthodes** (MCO contre estimateur robuste) sur exactement les mêmes données.
- **Les tests unitaires** : ils fixent la graine pour que le résultat attendu ne change pas.

### 3.3 Précautions

- On crée **un seul** `rng` par programme et on le passe partout (à `Noise`, à `generate_positions`).
  Si on créait deux générateurs avec la même graine, les positions et le bruit utiliseraient les mêmes nombres
  de départ : ils ne seraient plus indépendants.
- Pour retrouver **les mêmes données** il faut aussi **les mêmes paramètres** (`N_POINTS`, `ALPHA1`, `ALPHA2`, ...) et
  le même code : changer `N_POINTS`, par exemple, change quels nombres servent à quoi. Une version
  différente de numpy peut aussi, en théorie, produire une autre suite.

---

## 4. Générer au hasard SANS graine… et la retrouver à la fin

C'est possible, et c'est ce que font `main_02` et `main_03` quand `GRAINE = None` (valeur par défaut).

| Paramètre | Comportement |
|---|---|
| `GRAINE = None` | données **différentes à chaque lancement** ; la graine utilisée est **affichée** en dernière ligne de la console |
| `GRAINE = 42` (un entier) | **toujours les mêmes** données |

**Comment ça marche** (fonction `create_rng` de `measurements.py`) : quand `GRAINE` est `None`, le programme
**tire lui-même une graine au hasard**, l'utilise, puis la renvoie pour qu'elle soit affichée.

```python
seed = int(np.random.default_rng().integers(0, 2**32))     # une graine au hasard
rng = np.random.default_rng(seed)                           # le générateur démarre sur cette graine
```

(Écrire simplement `np.random.default_rng()` sans graine marcherait aussi, mais numpy choisirait la graine
en interne et on ne pourrait plus la retrouver : voilà pourquoi on la tire nous-mêmes.)

**Exemple réel.** Premier lancement, `GRAINE = None` :

```
   x =   835.2   v = 94.69
   x =  1179.9   v = 87.68
...
graine utilisee : 1351860723   (pour regenerer exactement les memes donnees : GRAINE = 1351860723)
```

Pour regénérer **exactement** les mêmes points, on copie le nombre affiché dans la section PARAMETRES :

```python
GRAINE = 1351860723
```

et on relance : on retrouve les mêmes points (835,2 m → 94,69 km/h ; 1179,9 m → 87,68 km/h ; ...).
Le titre de la figure contient aussi la graine, donc une image sauvegardée « se souvient » de la sienne.

---

## 5. Où est quoi dans le code

| Étape | Fichier | Fonction / classe |
|---|---|---|
| Choisir les coefficients libres | `src/vitesse/polynomial_tools.py` | `find_free_coefficients` |
| Calculer $c_0, c_1, \dots, c_d$ | `src/vitesse/polynomial_tools.py` | `coefficients_from_free` |
| Vitesse et accélération en un point | `src/vitesse/polynomial.py` | `SpeedPolynomial.speed`, `.acceleration` |
| Créer le générateur / la graine | `src/vitesse/measurements.py` | `create_rng` |
| Tirer les positions (loi uniforme) | `src/vitesse/measurements.py` | `generate_positions` |
| Tirer le bruit (loi normale) | `src/vitesse/noise.py` | `Noise.sample` |
| Tirer un $\sigma_i$ par voiture | `src/vitesse/noise.py` | `Noise.sample_sigmas` |
| Vitesse mesurée = vraie + bruit | `src/vitesse/measurements.py` | `simulate_speeds` |
| Tout enchaîner | `tests/mains/main_03_simulation.py` | `main()` |
| Estimer $c_0..c_d$, $\alpha_1$, $\alpha_2$ (maximum de vraisemblance) | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood` (section 6) |

---

## 6. Retrouver le polynôme : maximum de vraisemblance avec des $\sigma_i$ inconnus

Le système ne reçoit que les $(x_i, v_i)$ : il ne connaît ni le polynôme, ni les $\sigma_i$. Avec un seul point par
voiture, on ne peut pas retrouver chaque $\sigma_i$ ; on estime donc **la loi** des $\sigma_i$, c'est-à-dire
$\alpha_1$ et $\alpha_2$, en même temps que les coefficients $c_0, \dots, c_d$.

La densité d'un résidu $r_i = v_i - P(x_i)$ est la moyenne des densités normales sur toutes les valeurs possibles
de $\sigma_i$ (un **mélange**) :

$$
f(r) = \sum_k w_k\, \frac{1}{\sigma_k \sqrt{2\pi}}\, e^{-r^2 / (2\sigma_k^2)},
\qquad
\log L(c, \alpha_1, \alpha_2) = \sum_{i=1}^{N} \log f\big(v_i - P(x_i)\big)
$$

Les $\sigma_k$ parcourent $[\alpha_1, \alpha_2]$ et les poids $w_k$ suivent la loi des $\sigma_i$ : la cloche
$\mathcal{N}(\mu, s^2)$ discrétisée en 40 points, plus une masse de 0,135 % sur $\alpha_1$ et sur $\alpha_2$
(les tirages ramenés aux bornes). On maximise $\log L$ numériquement (`scipy.optimize.minimize`), en partant des
moindres carrés.

```python
estimateur = MaximumLikelihood(LONGUEUR, D).fit(x, v)      # modèle "sigma_variable" par défaut
estimateur.coefficients, estimateur.alpha1, estimateur.alpha2
estimateur.sigma                                           # écart-type global = racine(moyenne des sigma_i²)
```

- **Précision.** Il faut beaucoup de points pour bien estimer $\alpha_1$ et $\alpha_2$ : avec 1000 voitures on
  trouve typiquement $\alpha_1 \approx 2 \pm 0{,}3$ et $\alpha_2 \approx 8 \pm 0{,}5$ km/h ; avec 200, l'estimation
  des bornes est très imprécise. Les coefficients, eux, sont bien estimés dans tous les cas.
- **Cas simple.** `MaximumLikelihood(LONGUEUR, D, noise_model="sigma_fixe")` suppose un seul $\sigma$ pour toutes les
  voitures : le maximum de vraisemblance revient alors aux moindres carrés. `main_04` affiche les deux pour comparer.