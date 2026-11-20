# Explication : comment les données simulées sont générées

Ce document explique la génération sur un petit exemple : $L = 2000$ m, $V_0 = 70$ km/h, $V_L = 80$ km/h,
limite 90 km/h, degré $d = 2$, écarts-types $\sigma_i \in [\alpha_1, \alpha_2] = [2, 8]$ km/h, 100 points (= 100 voitures).

## 0. Vue d'ensemble
 
### Modélisation du problème

On considère $N$ véhicules indicés par $i \in \{1, \dots, N\}$. Chaque véhicule envoie **un seul point** : les $N$ points
sont donc indépendants (choix validé avec le professeur, voir `formulation.md`).

- $P$ est le polynôme de vitesse : la **vraie** vitesse en fonction de la position (sans bruit) (section 1).
- $\sigma_i$ est l'écart-type du bruit propre au véhicule $i$ ($\sigma_i^2$ est sa variance), tiré aléatoirement selon une loi normale bornée à $[\alpha_1, \alpha_2]$ (voir ci-dessous).
- $x_i$ est la position du véhicule $i$.
- $v_i$ est la vitesse mesurée du véhicule $i$ : la vraie vitesse plus un bruit $b_i$ propre au véhicule,

$$
v_i = P(x_i) + b_i,
\qquad
b_i \sim \mathcal{N}\!\left(0,\ \sigma_i^2\right),
\qquad
b_1, \dots, b_N \ \text{indépendants}.
$$

De façon équivalente, chaque mesure est elle-même une variable aléatoire gaussienne, centrée sur la vraie vitesse :

$$
v_i \sim \mathcal{N}\!\left(P(x_i),\ \sigma_i^2\right).
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
un écart-type valide (le code accepte aussi $\alpha_1 = 0$, ce qui autorise $\sigma_i = 0$). Conséquence : $\sigma_i$ vaut
exactement $\alpha_1$ (ou exactement $\alpha_2$) avec une probabilité d'environ 0,13 %.

> **Dans le code** : `Noise("normale", sigma_range=(alpha1, alpha2))`, détails en section 2.

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

Pour chaque véhicule $i$ :

$$
\begin{array}{ccc}
S_i \sim \mathcal{N}(\mu, s^2) & & \\[2pt]
\big\downarrow \scriptstyle{\text{ borné}} & & \\[2pt]
\sigma_i \in [\alpha_1, \alpha_2] & & x_i \sim \mathcal{U}([0, L]) \\[2pt]
\big\downarrow & & \big\downarrow \\[2pt]
b_i \sim \mathcal{N}(0, \sigma_i^2) & & P(x_i) \\[2pt]
\searrow & & \swarrow \\[2pt]
 & v_i = P(x_i) + b_i & \\[2pt]
 & \big\downarrow & \\[2pt]
 & (x_i,\ v_i) &
\end{array}
$$

- **Le bruit** : on tire $S_i$ avec $\mu = \frac{\alpha_1 + \alpha_2}{2}$ et $s = \frac{\alpha_2 - \alpha_1}{6}$.
  On le ramène dans $[\alpha_1, \alpha_2]$ pour obtenir $\sigma_i$, puis on tire le bruit $b_i$ avec cet écart-type.
- **La vraie vitesse** : on tire une position $x_i$, puis on calcule $P(x_i)$ (sans bruit).
- **Sortie** : le système ne reçoit que le couple $(x_i, v_i)$. L'estimateur `"sigma_variable"` suppose en plus
  que chaque $\sigma_i$ est différent (section 5).

Un seul « générateur aléatoire » (`rng`) fournit tous les nombres au hasard. Ce générateur est entièrement déterminé
par un nombre : la graine (section 3).

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


### 1.4 Comment $c_2, \dots, c_d$ sont choisis

- **À la main** : `SpeedPolynomial(2000, 70, 80, 2, free_coefficients=[-58.29])`.
- **Automatiquement** (par défaut, si on donne `speed_limit`) : on cherche les coefficients libres qui
  1. **minimisent** l'écart moyen à la limite : $J = \text{moyenne de } (90 - v(x))^2$
     (la courbe reste le plus près possible de 90 km/h sur tout le tronçon) ;
  2. en respectant **en chaque point** : $0 \le v(x) \le 90$ km/h et $|a(x)| \le a_{max}$.

  L'accélération du véhicule est $a = \dfrac{dv}{dt} = v \cdot \dfrac{dv}{dx}$ (à convertir en m/s²).
  La recherche est faite par `scipy.optimize.minimize` ; s'il n'existe aucun polynôme qui respecte les contraintes,
  une erreur le dit.
  

### 1.5 Calculer la vitesse en un point

Une fois $[c_0, \dots, c_d]$ connus, `speed(x)` fait :

```python
xi = x / length                      # position normalisée
v = 0
for j, cj in enumerate(c):           # on ajoute chaque terme  cj · ξ^j
    v = v + cj * xi**j
```

---

## 2. Générer un point $(x, v(x))$

Pour chaque point $k = 1, \dots, N$ :

**Étape 1 — la position $x_k$ : loi uniforme sur $[0, L]$**

```python
x = rng.uniform(0.0, length, n_points)       # generate_positions
```



**Étape 2 — la vraie vitesse en ce point**

$$v(x_k) = c_0 + c_1 \xi_k + c_2 \xi_k^2, \qquad \xi_k = x_k / L$$

**Étape 3 — l'écart-type $\sigma_k$ de la voiture $k$** (détails en section 0)

```python
sigmas = np.clip(rng.normal(moyenne, ecart_type, n), alpha1, alpha2)   # Noise.sample_sigmas
```

Chaque voiture a son propre capteur, plus ou moins précis : $\sigma_k$ est tiré selon $\mathcal{N}(\mu, s^2)$ avec
$\mu = 5$ km/h et $s = 1$ km/h (pour $[\alpha_1, \alpha_2] = [2, 8]$), puis ramené dans $[2, 8]$ s'il en sort.
Les $\sigma_k$ tirés sont dans `bruit.last_sigmas`, et `python src/vitesse/noise.py` vérifie ces chiffres sur 100 000 voitures.

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



Les points $(x_k, v_k)$ sont ce que « reçoit » le système ; la vraie courbe $v(x)$ est ce qu'on cherchera à retrouver.
Les $\sigma_k$, eux, ne sont **pas** connus du système : il ne voit que les vitesses mesurées.

> **Ordre des tirages.** Les $N$ positions sont tirées d'abord, puis les $N$ écarts-types $\sigma_k$, puis les
> $N$ valeurs de bruit : le générateur `rng` fournit ses nombres les uns après les autres, comme on pioche dans
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

## 5. Retrouver le polynôme : le maximum de vraisemblance (`maximum_likelihood.py`)

Les sections 1 à 4 fabriquent les mesures $(x_i, v_i)$ à partir d'un polynôme **connu**. Ici on fait le chemin
inverse : on ne garde que les mesures et le degré, et on cherche les coefficients. On reprend la méthode du
**TP2 (section 1, bruit blanc gaussien)** : on annule le gradient de la log-vraisemblance, ce qui donne un
système linéaire.

Le TP2 suppose un **même** écart-type $\sigma$ pour toutes les mesures. Ce n'est pas notre modèle (section 0) :
chaque voiture $i$ a son propre $\sigma_i$. On écrit donc la log-vraisemblance avec des $\sigma_i$ différents
(5.1 à 5.5), puis on en déduit **deux méthodes** (paramètre `estimator`), détaillées une par une en 5.6 à 5.8 :

| Méthode | `estimator` | Hypothèse | Ce qu'on connaît | Ce qu'on estime | Section |
|---|---|---|---|---|---|
| 1 | `"sigma_constant"` (défaut) | $\forall i,\ \sigma_i = \sigma_c$ (TP2) | $x_i, v_i$ | $\hat{\mathbf c}$ | 5.7 |
| 2 | `"sigma_variable"` | un $\sigma_i$ par voiture, connu | $x_i, v_i$ et les $\sigma_i$ | $\hat{\mathbf c}$ | 5.8 |

> **Dans le code**, les positions sont normalisées avant de calculer les $P_n(x_i)$, pour éviter des nombres
> énormes comme $2000^6$ : en base canonique $P_n(x) = \xi^n$ avec $\xi = x / L \in [0, 1]$ (section 1.1), en base
> de Legendre $P_n(x) = \mathrm{Leg}_n(t)$ avec $t = 2x/L - 1 \in [-1, 1]$ (section 5.11). Les formules ci-dessous
> sont valables pour n'importe quelle base.

### 5.1 Notations

$\forall \ i\in [1,N]$
| Symbole | Signification |
|---|---|
| $N$ | nombre de véhicules|
| $x_i$ | position du véhicule $i$|
| $v_i$ | vitesse mesurée par le véhicule $i$ |
| $\mathbf v = (v_1, \dots, v_N)^\top$ | l'échantillon : toutes les vitesses mesurées (le $\chi$ du TP) |
| $\mathbf x = (x_1, \dots, x_N)^\top$ | toutes les positions |
| $d$ | degré du polynôme|
| $P_0, \dots, P_{d}$ | la base de polynômes (canonique ou Legendre) |
| $\mathbf c = (c_0, \dots, c_{d})^\top$ | coefficients |
| $\hat{\mathbf c} = (c_0, \dots, c_{d})^\top$ | coefficients estimés |
| $\mathbf c^*$ | **"vrais"** coefficients, fixes et inconnus |
| $\sigma_i$ | écart-type du bruit du capteur du véhicule $i$ |

On note aussi $\mu_i(\hat{\mathbf c} )$ la vitesse que le polynôme de coefficients $\hat{\mathbf c} $ prédit en $x_i$ :

$$
\forall \ i\in [1,N] \quad \mu_i({\mathbf c} ) = \sum_{n=0}^{d} c_n P_n(x_i)
$$

### 5.2 Le modèle

Les mesures sont produites par les vrais coefficients :

$$
v_i = \mu_i(\mathbf c^*) + b_i,
\qquad
b_i \sim \mathcal N(0, \sigma_i^2),
\qquad
b_1, \dots, b_N \text{ indépendants}
$$

Les bruits sont indépendants parce que chaque véhicule envoie un seul point avec son propre capteur.
Les $d + 1$ coefficients sont estimés **librement** : on ne suppose pas connus $V_0$ et $V_L$.

### 5.3 La loi d'une mesure

Les positions $x_i$ sont connues. Pour des coefficients $\mathbf c$ donnés, $\mu_i(\mathbf c)$ est donc un nombre
fixe, et tout le hasard de $v_i$ vient de $b_i$. Ajouter une constante à une gaussienne centrée décale sa moyenne :

$$
v_i \sim \mathcal N\big(\mu_i(\mathbf c), \sigma_i^2\big)
\quad\Longrightarrow\quad
p(v_i \mid \mathbf c) = \frac{1}{\sqrt{2\pi}\,\sigma_i}\exp\!\left(-\frac{\big(v_i - \mu_i(\mathbf c)\big)^2}{2\sigma_i^2}\right)
$$

### 5.4 La vraisemblance

Les mesures sont indépendantes, donc la densité jointe de l'échantillon est le produit des densités :

$$
L(\mathbf v \mid \mathbf c) = \prod_{i=1}^{N} p(v_i \mid \mathbf c)
= \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi}\,\sigma_i}\exp\!\left(-\frac{\big(v_i - \mu_i(\mathbf c)\big)^2}{2\sigma_i^2}\right)
$$

**Changement de point de vue.** Une fois les mesures faites, $\mathbf v$ est fixé : les $v_i$ sont des nombres et
**ne dépendent pas du candidat $\mathbf c$**. On lit alors $L(\mathbf v \mid \mathbf c)$ comme une fonction de
$\mathbf c$. Elle indique quels coefficients rendent les mesures observées les plus probables. L'estimateur du
maximum de vraisemblance est le $\mathbf c$ qui la maximise. (On garde la notation du TP, $\ell(\chi \mid \mathbf c)$,
avec $\mathbf v$ à la place de $\chi$.)

**Remarque sur les positions.** Dans la simulation, les $x_i$ sont tirés uniformément sur $[0, L]$. Leur densité
$1/L$ ne dépend pas de $\mathbf c$ : elle ajouterait seulement la constante $-N\ln L$ à la log-vraisemblance, sans
déplacer le maximum. On peut donc raisonner à $\mathbf x$ fixé. Rigoureusement, on devrait écrire
$\ell(\mathbf v \mid \mathbf x, \mathbf c)$ ; on garde $\ell(\mathbf v \mid \mathbf c)$ pour alléger.

### 5.5 La log-vraisemblance

On trouve :

$$
\ell(\mathbf v \mid \mathbf c) = \ln L(\mathbf v \mid \mathbf c)
= \sum_{i=1}^{N}\left[-\frac{\big(v_i - \mu_i(\mathbf c)\big)^2}{2\sigma_i^2} - \ln\!\big(\sqrt{2\pi}\,\sigma_i\big)\right]
$$

$$
\boxed{\ell(\mathbf v \mid \mathbf c) = -\sum_{i=1}^{N}\frac{1}{2\sigma_i^2}\Big[v_i - \sum_{n=0}^{d} c_n P_n(x_i)\Big]^2 - \sum_{i=1}^{N}\ln\!\big(\sqrt{2\pi}\,\sigma_i\big)}
$$

En supposant : $\forall i \in [1,N], \quad \sigma_i = \sigma_c$

$$
\boxed{\ell(\mathbf v \mid \mathbf c) = -\frac{1}{2\sigma_c^2}\sum_{i=1}^{N}\Big[v_i - \sum_{n=0}^{d} c_n P_n(x_i)\Big]^2 - N\ln\!\big(\sqrt{2\pi}\,\sigma_c\big)}
$$


### 5.6 Deux méthodes, une même démarche

Les deux méthodes partent de la log-vraisemblance de 5.5. Elles ne diffèrent que par **ce que l'on sait des $\sigma_i$** :

| Méthode | `estimator` | Ce qu'on sait des $\sigma_i$ | Inconnues | Section |
|---|---|---|---|---|
| 1 | `"sigma_constant"` (défaut, TP2) | $\forall i,\ \sigma_i = \sigma_c$, avec $\sigma_c$ inconnu (non estimé) | $\mathbf c$ | 5.7 |
| 2 | `"sigma_variable"` | chaque $\sigma_i$ est différent et **connu** | $\mathbf c$ | 5.8 |

Pour chaque méthode, on suit les mêmes étapes :

1. écrire la log-vraisemblance $\ell(\mathbf v \mid \mathbf c)$ ;
2. annuler ses dérivées partielles $\partial_{c_n}\ell = 0$ pour chaque $n = 0, \dots, d$ (toutes ont la même forme : on
   fait le calcul une seule fois pour un $n$ quelconque) ;
3. résoudre le système obtenu pour trouver $\hat{\mathbf c}$.

**Dérivée commune aux deux méthodes.** Dans $\mu_i(\mathbf c) = \sum_m c_m P_m(x_i)$, on note l'indice de somme $m$
pour ne pas le confondre avec le $n$ de la dérivée. Seul le terme $m = n$ dépend de $c_n$, et $v_i$ n'en dépend pas :

$$
\partial_{c_n}\mu_i(\mathbf c) = P_n(x_i),
\qquad
\partial_{c_n} v_i = 0
$$

**Écriture matricielle.** On note $\mathbf P$ la matrice $N \times (d+1)$ de terme $\mathbf P_{i,n} = P_n(x_i)$
(construite par `basis_matrix`). Alors $\mu_i(\mathbf c) = (\mathbf P\mathbf c)_i$.

---

### 5.7 Méthode 1 : $\sigma_c$ commun et inconnu (`"sigma_constant"`, TP2)

**Hypothèse.** $\forall i \in [1,N],\ \sigma_i = \sigma_c$, avec $\sigma_c$ **inconnu**. On ne connaît que les
$(x_i, v_i)$. On estime seulement $\mathbf c$ : la valeur de $\sigma_c$ ne nous intéresse pas.

**Log-vraisemblance.** C'est la seconde formule encadrée de 5.5 :

$$
\ell(\mathbf v \mid \mathbf c) = -\frac{1}{2\sigma_c^2}\sum_{i=1}^{N}\Big[v_i - \mu_i(\mathbf c)\Big]^2 - N\ln\!\big(\sqrt{2\pi}\,\sigma_c\big)
$$

Le second terme ne dépend pas de $\mathbf c$, et $\frac{1}{2\sigma_c^2} > 0$. Maximiser $\ell$ en $\mathbf c$ revient
donc à minimiser $\sum_i \big[v_i - \mu_i(\mathbf c)\big]^2$ : ce sont les **moindres carrés ordinaires**, où tous les
points ont le même poids.

**Annulation du gradient en $\mathbf c$.** La dérivée de $-\frac{(v_i-\mu_i)^2}{2\sigma_c^2}$ par rapport à $c_n$ vaut
$\frac{v_i-\mu_i}{\sigma_c^2}\,P_n(x_i)$ :

$$
\partial_{c_n}\ell = \frac{1}{\sigma_c^2}\sum_{i=1}^{N}P_n(x_i)\Big(v_i - \sum_{m=0}^{d} c_m P_m(x_i)\Big) = 0
$$

$\sigma_c$ ne dépend pas de $i$ : on peut multiplier par $\sigma_c^2$, et il **disparaît**. On sépare les deux termes et
on échange les sommes sur $i$ et sur $m$ :

$$
\sum_{m=0}^{d} c_m \underbrace{\sum_{i=1}^{N}P_n(x_i)\,P_m(x_i)}_{M_{nm}}
= \underbrace{\sum_{i=1}^{N}P_n(x_i)\,v_i}_{w_n}
\qquad n = 0, \dots, d
$$

**Estimateur de $\mathbf c$.** Les $d + 1$ équations forment le système linéaire $\mathbf M\mathbf c = \mathbf w$ (la
ligne $n$ correspond à la dérivée par rapport à $c_n$) :

$$
\boxed{\hat{\mathbf c} = \mathbf M^{-1}\mathbf w}
\qquad
\mathbf M = \mathbf P^\top \mathbf P,
\qquad
\mathbf w = \mathbf P^\top \mathbf v
$$

C'est l'équation (1) du TP2. Le TP note ce vecteur $\mathbf v$ ; on l'appelle $\mathbf w$ pour ne pas le confondre
avec les mesures. $\sigma_c$ n'apparaît pas dans $\hat{\mathbf c}$ : **pas besoin de le connaître pour estimer
$\mathbf c$**, et on ne l'estime donc pas.


**Quand l'utiliser.** Quand on ne sait rien des $\sigma_i$. Dans notre simulation les $\sigma_i$ sont différents
(section 0) : l'hypothèse est fausse. $\hat{\mathbf c}$ reste un estimateur raisonnable, mais ce n'est plus le maximum
de vraisemblance du vrai modèle.

> **Dans le code** : `MaximumLikelihood(L, d, estimator="sigma_constant").fit(x, v)`.
> `est.M` et `est.w` contiennent $\mathbf M$ et $\mathbf w$.

---

### 5.8 Méthode 2 : $\sigma_i$ différents et connus (`"sigma_variable"`)

**Hypothèse.** Chaque véhicule $i$ a son propre $\sigma_i$ et le **transmet** avec sa mesure (c'est la précision de
son capteur). On connaît les $(x_i, v_i, \sigma_i)$. On estime seulement $\mathbf c$. Dans la simulation, ce sont les
$\sigma_i$ tirés par `Noise` (`bruit.last_sigmas`).

**Log-vraisemblance.** C'est la première formule encadrée de 5.5 :

$$
\ell(\mathbf v \mid \mathbf c) = -\sum_{i=1}^{N}\frac{1}{2\sigma_i^2}\Big[v_i - \mu_i(\mathbf c)\Big]^2 - \sum_{i=1}^{N}\ln\!\big(\sqrt{2\pi}\,\sigma_i\big)
$$

Le second terme ne dépend pas de $\mathbf c$ (les $\sigma_i$ sont connus). Maximiser $\ell$ revient donc à minimiser la
somme des carrés des écarts, chaque écart étant pondéré par $1/\sigma_i^2$ : ce sont des **moindres carrés pondérés**.
Un capteur précis (petit $\sigma_i$) pèse davantage.

**Annulation du gradient.** La dérivée de $-\frac{(v_i-\mu_i)^2}{2\sigma_i^2}$ vaut $\frac{v_i-\mu_i}{\sigma_i^2}\,P_n(x_i)$ :

$$
\partial_{c_n}\ell(\mathbf v \mid \mathbf c) = \sum_{i=1}^{N}\frac{P_n(x_i)}{\sigma_i^2}\Big(v_i - \sum_{m=0}^{d} c_m P_m(x_i)\Big) = 0
$$

Contrairement à la méthode 1, le facteur $1/\sigma_i^2$ **dépend de $i$** : il ne peut pas sortir de la somme sur $i$,
et il reste donc **des deux côtés** :

$$
\sum_{m=0}^{d} c_m \underbrace{\sum_{i=1}^{N}\frac{P_n(x_i)\,P_m(x_i)}{\sigma_i^2}}_{\tilde M_{nm}}
= \underbrace{\sum_{i=1}^{N}\frac{P_n(x_i)\,v_i}{\sigma_i^2}}_{\tilde w_n}
\qquad n = 0, \dots, d
$$

**Estimateur.** Le système s'écrit $\tilde{\mathbf M}\mathbf c = \tilde{\mathbf w}$. Avec
$\mathbf W = \mathrm{diag}(1/\sigma_1^2, \dots, 1/\sigma_N^2)$ :

$$
\boxed{\hat{\mathbf c} = \tilde{\mathbf M}^{-1}\tilde{\mathbf w}}
\qquad
\tilde{\mathbf M} = \mathbf P^\top \mathbf W \mathbf P,
\qquad
\tilde{\mathbf w} = \mathbf P^\top \mathbf W\,\mathbf v
$$

Il n'y a rien d'autre à estimer : les $\sigma_i$ sont supposés connus.

**Lien avec la méthode 1.** Si tous les $\sigma_i$ valent $\sigma_c$, alors $\tilde{\mathbf M} = \mathbf M/\sigma_c^2$
et $\tilde{\mathbf w} = \mathbf w/\sigma_c^2$ : le facteur se simplifie et on retrouve $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$.
La méthode 1 est un cas particulier de la méthode 2.

> **Dans le code** : `MaximumLikelihood(L, d, estimator="sigma_variable").fit(x, v, sigmas=bruit.last_sigmas)`
> (les `sigmas` sont obligatoires et doivent être $> 0$). `est.M` et `est.w` contiennent $\tilde{\mathbf M}$ et
> $\tilde{\mathbf w}$ ; `est.sigmas` contient les $\sigma_i$ utilisés.

---

### 5.9 Bilan des deux méthodes

| | Méthode 1 `"sigma_constant"` | Méthode 2 `"sigma_variable"` |
|---|---|---|
| Données | $(x_i, v_i)$ | $(x_i, v_i, \sigma_i)$ |
| Log-vraisemblance | $\ell(\mathbf v \mid \mathbf c)$ avec $\sigma_i = \sigma_c$ | $\ell(\mathbf v \mid \mathbf c)$ avec les $\sigma_i$ |
| Poids du point $i$ | $1$ | $1/\sigma_i^2$ |
| Estimateur | $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$ (moindres carrés ordinaires) | $\hat{\mathbf c} = \tilde{\mathbf M}^{-1}\tilde{\mathbf w}$ (moindres carrés pondérés) |

Si tous les $\sigma_i$ sont égaux, les deux méthodes donnent le même $\hat{\mathbf c}$ (les poids se simplifient ;
vérifié par `test_equal_sigmas_give_the_same_estimate_as_sigma_constant`).

### 5.10 Caractérisation des estimateurs : borne de Cramér-Rao (BCR)

Les sections 5.7 et 5.8 donnent $\hat{\mathbf c}$ pour **un** échantillon $\mathbf v$. Mais $\mathbf v$ est aléatoire
(à cause des bruits $b_i$), donc $\hat{\mathbf c}$ l'est aussi : en refaisant l'expérience avec d'autres voitures, on
obtiendrait d'autres coefficients. Pour caractériser un estimateur, on regarde donc sa loi, à travers :

- son **biais** $\langle\hat{\mathbf c}\rangle - \mathbf c^*$ ($\langle\cdot\rangle$ : espérance sur les bruits, à positions
  $\mathbf x$ fixées) ;
- sa **covariance** $\mathrm{Cov}(\hat{\mathbf c}) = \big\langle(\hat{\mathbf c} - \langle\hat{\mathbf c}\rangle)(\hat{\mathbf c} - \langle\hat{\mathbf c}\rangle)^\top\big\rangle$,
  une matrice $(d+1) \times (d+1)$ : sa diagonale contient les variances $\mathrm{Var}(\hat c_n)$.

La **borne de Cramér-Rao** est la plus petite covariance possible pour un estimateur **sans biais** de $\mathbf c$ :

$$
\mathrm{Cov}(\hat{\mathbf c}) \succeq \mathrm{BCR} = \mathbf I_F^{-1}
\qquad\text{en particulier}\qquad
\mathrm{Var}(\hat c_n) \ge \mathrm{BCR}_{nn} \quad n = 0, \dots, d
$$

($\mathbf A \succeq \mathbf B$ signifie que $\mathbf A - \mathbf B$ est une matrice positive.) Un estimateur sans biais
qui atteint la borne est dit **efficace** : aucun autre estimateur sans biais ne fait mieux.

#### 5.10.1 Information de Fisher

L'information de Fisher est l'opposé de l'espérance de la hessienne de la log-vraisemblance :

$$
[\mathbf I_F]_{nm} = -\big\langle \partial_{c_n}\partial_{c_m}\ell(\mathbf v \mid \mathbf c)\big\rangle
$$

On part de la dérivée première de 5.8 et on dérive une seconde fois par rapport à $c_m$. Seul $\mu_i(\mathbf c)$ dépend
de $c_m$, avec $\partial_{c_m}\mu_i = P_m(x_i)$ :

$$
\partial_{c_m}\partial_{c_n}\ell(\mathbf v \mid \mathbf c) = -\sum_{i=1}^{N}\frac{P_n(x_i)\,P_m(x_i)}{\sigma_i^2} = -\tilde M_{nm}
$$

Cette dérivée seconde **ne dépend pas de $\mathbf v$** (le modèle est linéaire en $\mathbf c$) : l'espérance ne change
rien, et elle ne dépend pas non plus de $\mathbf c^*$. D'où, pour chaque méthode :

| | Méthode 1 `"sigma_constant"` ($\sigma_i = \sigma_c$) | Méthode 2 `"sigma_variable"` |
|---|---|---|
| Information de Fisher | $\mathbf I_F = \dfrac{\mathbf M}{\sigma_c^2}$ | $\mathbf I_F = \tilde{\mathbf M} = \mathbf P^\top \mathbf W \mathbf P$ |
| BCR | $\boxed{\mathrm{BCR} = \sigma_c^2\,\mathbf M^{-1}}$ | $\boxed{\mathrm{BCR} = \tilde{\mathbf M}^{-1}}$ |

**Attention pour la méthode 1.** $\hat{\mathbf c}$ ne dépend pas de $\sigma_c$, mais la BCR, si : plus le bruit est fort,
moins l'estimation est précise. Comme on n'estime pas $\sigma_c$, il faut le **donner** pour calculer la borne. En
simulation, on prend l'écart-type global du bruit simulé, $\sigma_g = \sqrt{\frac{1}{N}\sum_i \sigma_i^2}$ (c'est ce que
font `main_04` et `main_05`).

#### 5.10.2 Les estimateurs atteignent-ils la borne ?

On écrit les deux estimateurs sous la même forme $\hat{\mathbf c} = \tilde{\mathbf M}^{-1}\mathbf P^\top \mathbf W\,\mathbf v$
(avec $\mathbf W = \mathbf I$ et $\tilde{\mathbf M} = \mathbf M$ pour la méthode 1), et on remplace
$\mathbf v = \mathbf P\mathbf c^* + \mathbf b$ :

$$
\hat{\mathbf c} = \tilde{\mathbf M}^{-1}\mathbf P^\top \mathbf W \mathbf P\,\mathbf c^* + \tilde{\mathbf M}^{-1}\mathbf P^\top \mathbf W\,\mathbf b
= \mathbf c^* + \tilde{\mathbf M}^{-1}\mathbf P^\top \mathbf W\,\mathbf b
$$

**Biais.** $\langle\mathbf b\rangle = \mathbf 0$, donc $\langle\hat{\mathbf c}\rangle = \mathbf c^*$ : les **deux** méthodes
sont sans biais, quels que soient les vrais $\sigma_i$.

**Covariance.** Avec $\boldsymbol\Sigma = \mathrm{Cov}(\mathbf b) = \mathrm{diag}(\sigma_1^2, \dots, \sigma_N^2)$ (les vrais
écarts-types) :

$$
\mathrm{Cov}(\hat{\mathbf c}) = \tilde{\mathbf M}^{-1}\mathbf P^\top \mathbf W \boldsymbol\Sigma \mathbf W \mathbf P\,\tilde{\mathbf M}^{-1}
$$

- **Méthode 2** : $\mathbf W = \boldsymbol\Sigma^{-1}$, donc $\mathbf W\boldsymbol\Sigma\mathbf W = \mathbf W$ et
  $\mathrm{Cov}(\hat{\mathbf c}) = \tilde{\mathbf M}^{-1}\tilde{\mathbf M}\,\tilde{\mathbf M}^{-1} = \tilde{\mathbf M}^{-1}$ :
  elle **atteint** sa BCR, elle est efficace.
- **Méthode 1** : $\mathbf W = \mathbf I$, donc
  $\mathrm{Cov}(\hat{\mathbf c}) = \mathbf M^{-1}\big(\mathbf P^\top \boldsymbol\Sigma \mathbf P\big)\mathbf M^{-1}$.
  - Si vraiment $\sigma_i = \sigma_c$ pour tout $i$ : $\boldsymbol\Sigma = \sigma_c^2\mathbf I$ et on trouve
    $\sigma_c^2\mathbf M^{-1}$ : elle atteint sa BCR.
  - Dans notre simulation (les $\sigma_i$ diffèrent), son hypothèse est fausse : $\sigma_c^2\mathbf M^{-1}$ n'est
    qu'une **approximation**, et sa vraie covariance $\mathbf M^{-1}\mathbf P^\top \boldsymbol\Sigma \mathbf P\,\mathbf M^{-1}$
    est toujours $\succeq \tilde{\mathbf M}^{-1}$. La vraie borne du problème est celle de la méthode 2 : ne pas
    utiliser les $\sigma_i$ fait perdre de la précision.

#### 5.10.3 Lire la BCR

- **Diagonale** : $\sqrt{\mathrm{BCR}_{nn}}$ est le plus petit écart-type possible sur $\hat c_n$ (en km/h, comme les
  coefficients).
- **Hors diagonale** : la covariance entre $\hat c_n$ et $\hat c_m$. On la lit plus facilement sous forme de
  corrélation $\rho_{nm} = \mathrm{BCR}_{nm} / \sqrt{\mathrm{BCR}_{nn}\mathrm{BCR}_{mm}} \in [-1, 1]$. Si $\rho_{nm}$ est
  proche de $\pm 1$, les erreurs sur $c_n$ et $c_m$ sont **liées** : l'une compense l'autre (voir 5.11).
- **Sur la courbe** : la vitesse estimée en $x$ est $\mu(\hat{\mathbf c}) = \mathbf p(x)^\top \hat{\mathbf c}$ avec
  $\mathbf p(x) = (P_0(x), \dots, P_d(x))^\top$. Sa variance vaut donc au moins
  $$
  \mathrm{Var}\big(\mu(\hat{\mathbf c})(x)\big) \ge \mathbf p(x)^\top\,\mathrm{BCR}\;\mathbf p(x)
  $$
  Contrairement aux coefficients, ce résultat **ne dépend pas de la base** (canonique ou Legendre).

> **Dans le code** :
> ```python
> est.fisher_information(sigma=sigma_c)       # I_F : M / sigma_c²  (méthode 1)  ou  M_tilde  (méthode 2)
> est.crb(sigma=sigma_c)                      # BCR = I_F^-1
> np.sqrt(np.diag(est.crb(sigma=sigma_c)))    # écarts-types minimaux des coefficients
> ```
> `sigma` est **obligatoire** pour la méthode 1 (sinon `ValueError`, testé par `test_crb_of_sigma_constant_needs_sigma`)
> et ignoré pour la méthode 2. Les tests Monte-Carlo vérifient que la covariance empirique de $\hat{\mathbf c}$ (2000
> tirages du bruit aux mêmes positions) vaut la BCR : `test_empirical_covariance_matches_the_crb` (méthode 1, $\sigma$
> commun) et `test_sigma_variable_reaches_its_crb_and_beats_sigma_constant` (méthode 2, qui fait aussi mieux que la
> méthode 1 quand les $\sigma_i$ diffèrent).

#### 5.10.4 Exemple (graine 42)

Mêmes données qu'en 5.13 ($L = 2000$ m, $d = 2$, 100 voitures, $\sigma_i \in [2, 8]$ km/h, $\sigma_g = 5{,}08$ km/h,
utilisé pour la méthode 1). Écarts-types de la BCR, $\sqrt{\mathrm{BCR}_{nn}}$ :

| | canonique, méthode 1 | canonique, méthode 2 | Legendre, méthode 1 | Legendre, méthode 2 |
|---|---|---|---|---|
| $c_0$ | 1,58 | 1,50 | 0,51 | 0,47 |
| $c_1$ | 7,53 | 7,05 | 0,94 | 0,86 |
| $c_2$ | 7,58 | 6,95 | 1,26 | 1,16 |

La vraie covariance de la méthode 1, $\mathbf M^{-1}\mathbf P^\top \boldsymbol\Sigma \mathbf P\,\mathbf M^{-1}$, donne en
base canonique les écarts-types $(1{,}58 ;\ 7{,}48 ;\ 7{,}42)$ : plus grands que la borne de la méthode 2. Avec
$[2, 8]$ km/h, les $\sigma_i$ sont concentrés autour de 5 km/h et le gain de la méthode 2 reste modeste (5 à 8 %). Il
grandit quand les $\sigma_i$ sont plus dispersés.

Corrélations de la BCR (méthode 2) :

| | canonique | Legendre |
|---|---|---|
| $(c_0, c_1)$ | −0,88 | 0,04 |
| $(c_1, c_2)$ | −0,97 | 0,07 |
| $(c_0, c_2)$ | 0,77 | 0,11 |

En base canonique, $c_1$ et $c_2$ sont corrélés à −0,97 : une erreur sur $c_1$ est presque compensée par une erreur de
signe opposé sur $c_2$. En Legendre, les coefficients sont presque décorrélés et leurs écarts-types bien plus petits
(section suivante).

### 5.11 Pourquoi la base de Legendre ?

En base canonique, les colonnes $1, \xi, \xi^2, \dots$ se ressemblent beaucoup sur $[0, 1]$ (elles sont toutes
croissantes) : $\mathbf M$ a de grands termes hors diagonale. Les coefficients estimés se **compensent** alors : une
erreur sur $c_1$ peut être rattrapée par une erreur de signe opposé sur $c_2$, si bien que la courbe reste bonne alors
que les coefficients sont loin des vrais. En base de Legendre ($t = 2x/L - 1$) avec des $t_i$ uniformes sur $[-1, 1]$
(c'est le cas : les positions sont tirées uniformément, section 2), $M_{nm} \simeq 0$ pour $n \neq m$ (TP2,
section 1.4) : $\mathbf M$ est **quasi diagonale**, et chaque équation du système porte presque sur un seul
coefficient. Cela reste vrai pour $\tilde{\mathbf M}$, tant que les poids ne dépendent pas de la position.

Changer de base ne change **pas** la courbe estimée $\mu(\hat{\mathbf c})$ (les deux bases décrivent les mêmes
polynômes, vérifié par `test_legendre_basis_gives_the_same_curve_as_canonical`). Seuls les **coefficients** et leur
interprétation changent. La quasi-orthogonalité est perdue si les $x_i$ ne sont plus uniformes sur le tronçon.

### 5.12 La classe `MaximumLikelihood`

```python
est = MaximumLikelihood(length=2000, degree=2, basis="canonique", estimator="sigma_constant")
est.fit(x, v)                                   # méthode 1 (défaut, TP2)
#   ou : MaximumLikelihood(2000, 2, estimator="sigma_variable").fit(x, v, sigmas=bruit.last_sigmas)   # méthode 2
est.coefficients                # [c0, ..., c_d] estimés DANS LA BASE CHOISIE
est.canonical_coefficients      # convertis en base canonique (comparables à SpeedPolynomial)
est.sigmas                      # méthode 2 : les sigma_i utilisés
est.M, est.w                    # M, w (méthode 1) ou M_tilde, w_tilde (méthode 2)
est.speed(1000)                 # vitesse estimée en x = 1000 m
est.fisher_information(sigma=5) # I_F (5.10) ; sigma obligatoire pour la méthode 1, ignoré pour la méthode 2
est.crb(sigma=5)                # BCR = I_F^-1 (5.10)
```

- **`__init__`** : mémorise $L$, $d$, la base et la méthode ; refuse $L \le 0$, $d < 1$, une base ou une méthode
  inconnues.
- **`fit(x, v, sigmas=None)`** :
  1. vérifie que `x` et `v` ont la même taille et qu'il y a au moins $d + 1$ points (sinon la matrice n'est pas
     inversible) ;
  2. poids : $1$ pour la méthode 1 (les `sigmas` éventuels sont ignorés), $1/\sigma_i^2$ pour la méthode 2 (les
     `sigmas` doivent être fournis, de la bonne taille et $> 0$) ;
  3. construit $\mathbf P$, puis $\mathbf P^\top \mathbf W \mathbf P$ et $\mathbf P^\top \mathbf W \mathbf v$
     ($\mathbf W = \mathbf I$ pour la méthode 1), rangés dans `est.M` et `est.w` ;
  4. résout le système avec `np.linalg.solve` (même résultat que d'inverser la matrice, mais plus précis
     numériquement) ;
  5. renvoie `self`, ce qui permet d'écrire `MaximumLikelihood(L, d).fit(x, v)` en une ligne.
- **`canonical_coefficients`** : pour Legendre, convertit $\sum c_n \mathrm{Leg}_n(t)$ en polynôme en $t$, puis
  remplace $t$ par $2\xi - 1$.
- **`speed(x)`** : renvoie $\mu(\hat{\mathbf c})$ en $x$ ; lève une erreur si `fit` n'a pas été appelé.

### 5.13 Exemple chiffré (graine 42)

Paramètres du début du document : $L = 2000$ m, $V_0 = 70$, $V_L = 80$, $d = 2$, 100 voitures, graine 42,
$\sigma_i$ tirés dans $[2, 8]$ km/h (on obtient des $\sigma_i$ entre 2,87 et 7,91 km/h, $\sqrt{\text{moyenne des } \sigma_i^2} = 5{,}08$ km/h).
Estimation avec le bon degré, coefficients en base canonique :

| | $c_0$ | $c_1$ | $c_2$ |
|---|---|---|---|
| vrai $\mathbf c^*$ | 70,00 | 68,29 | −58,29 |
| méthode 1 `"sigma_constant"` | 71,71 | 62,43 | −55,35 |
| méthode 2 `"sigma_variable"` | 71,10 | 65,79 | −58,70 |

Les vrais coefficients en base de Legendre sont $(84{,}71 ;\ 5{,}00 ;\ -9{,}71)$. La précision de ces estimations
(BCR) est donnée en 5.10.4.

### 5.14 Limites à connaître

- **Bruit non gaussien** (autre loi, valeurs aberrantes) : la log-vraisemblance de 5.5 n'est plus la bonne, et les moindres carrés deviennent sensibles aux points aberrants (le TP2, section 2, traite le cas Poisson
  avec un algorithme itératif).
- **$\sigma_i$ faux** (méthode 2) : si on donne des $\sigma_i$ erronés, l'estimateur reste non biaisé mais n'est plus
  le maximum de vraisemblance.
- **Mauvais degré** : `main_04` permet de tester `D_ESTIME = D + 1` (le terme en trop est estimé proche de 0
  mais rend l'estimation plus instable, surtout en base canonique) ou `D_ESTIME = D - 1` (la courbe ne peut plus suivre
  la vraie forme : biais).

> **Pour essayer** : `python tests/mains/main_04_maximum_vraisemblance.py` (paramètres `ESTIMATEUR` et `BASE`)
> trace les mesures, la vraie courbe, l'estimation et une bande $\pm$ `N_SIGMA` $\cdot\,\sigma$ global ;
> `python tests/mains/main_05_comparaison_estimateurs.py` trace les deux méthodes sur le même graphe ;
> `pytest tests/test_maximum_likelihood.py -v` lance les tests.

---

## 6. Où est quoi dans le code

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
| Matrice $\mathbf P_{i,n} = P_n(x_i)$ (canonique / Legendre) | `src/vitesse/maximum_likelihood.py` | `basis_matrix` |
| Log-vraisemblance (méthodes 1 et 2) | `src/vitesse/maximum_likelihood.py` | `log_likelihood` |
| Estimer $\hat{\mathbf c}$ (méthodes 1 et 2) | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fit`, `.speed` |
| Caractériser : information de Fisher, BCR | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fisher_information`, `.crb` |
| Simuler puis estimer | `tests/mains/main_04_maximum_vraisemblance.py` | `main()` |
| Comparer les deux méthodes sur un même graphe | `tests/mains/main_05_comparaison_estimateurs.py` | `main()` |
