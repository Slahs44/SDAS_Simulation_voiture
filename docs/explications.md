# Explication : comment les données simulées sont générées

Ce document explique, avec les valeurs par défaut de `main_03_simulation.py`
($L = 2000$ m, $V_0 = 70$ km/h, $V_L = 80$ km/h, limite 90 km/h, degré $d = 2$, bruit $\sigma = 5$ km/h, 100 points).

## 0. Vue d'ensemble

### Modélisation du problème

On considère $N$ véhicules indicés par $i \in \{1, \dots, N\}$. Chaque véhicule envoie **un seul point** : les $N$ points
sont donc indépendants (choix validé avec le professeur, voir `formulation.md`).

- $P$ est le polynôme de vitesse : la **vraie** vitesse en fonction de la position, sans bruit (section 1).
- $\sigma_i$ est l'écart-type du bruit propre au véhicule $i$ ($\sigma_i^2$ est sa variance), tiré aléatoirement selon une loi normale bornée à $[\alpha_1, \alpha_2]$ (voir ci-dessous).
- $x_i$ est la position à laquelle le véhicule $i$ envoie son point.
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
un écart-type valide (le code accepte aussi $\alpha_1 = 0$, ce qui autorise $\sigma_i = 0$). Conséquence : $\sigma_i$ vaut
exactement $\alpha_1$ (ou exactement $\alpha_2$) avec une probabilité d'environ 0,13 %.

> **Dans le code** : `Noise("normale", sigma_range=(alpha1, alpha2))`, détails en section 2.

### Schéma

$$
\begin{array}{lcl}
P(x) \ \text{(vraie vitesse, sans bruit)} & \searrow & \\[6pt]
x_i \sim \mathcal{U}\left([0, L]\right) & \longrightarrow & v_i = P(x_i) + b_i \ \longrightarrow\ \left(x_i, v_i\right)_{1 \le i \le N} \ \text{(ce que le système reçoit)} \\[6pt]
b_i \sim \mathcal{N}\left(0, \sigma_i^2\right),\ \ \sigma_i \in [\alpha_1, \alpha_2] & \nearrow &
\end{array}
$$

Un seul « générateur aléatoire » (`rng`) fournit tous les nombres au hasard : d'abord les 100 positions,
ensuite les 100 valeurs de bruit. Ce générateur est entièrement déterminé par un nombre : **la graine** (section 3).

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

**Étape 3 — le bruit $\varepsilon_k$ : loi normale centrée, d'écart-type $\sigma$**

```python
bruit = rng.normal(0.0, sigma, n)            # Noise.sample, loi "normale"
```

$\varepsilon_k \sim \mathcal{N}(0, \sigma^2)$. Avec $\sigma = 5$ km/h : environ 68 % des points s'écartent de la
vraie courbe de moins de 5 km/h, et 95 % de moins de 10 km/h. ($\sigma$ est l'**écart-type** ; la variance vaut $\sigma^2$.)

**Variante — un écart-type $\sigma_i$ par voiture** (`Noise("normale", sigma_range=(alpha1, alpha2))`)

On suppose un point par voiture, donc des points indépendants, et chaque voiture a son propre capteur, plus ou
moins précis. Pour chaque point $i$ (formules complètes en section 0, « Tirage de l'écart-type $\sigma_i$ ») :

1. on tire $S_i \sim \mathcal{N}(\mu, s^2)$ avec $\mu = \frac{\alpha_1 + \alpha_2}{2}$ et $s = \frac{\alpha_2 - \alpha_1}{6}$
   (règle des 3 sigma : 99,7 % des valeurs entre $\alpha_1$ et $\alpha_2$) ;
2. les rares valeurs (environ 0,3 %) qui sortent de $[\alpha_1, \alpha_2]$ sont ramenées à la borne la plus proche :
   $\sigma_i = \min(\max(S_i, \alpha_1), \alpha_2)$ ;
3. on tire le bruit du point : $\varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)$ (noté $b_i$ en section 0).

```python
sigmas = np.clip(rng.normal(moyenne, ecart_type, n), alpha1, alpha2)   # Noise.sample_sigmas
bruit = rng.normal(0.0, sigmas)                                        # Noise.sample
```

Exemple avec $\alpha_1 = 2$ et $\alpha_2 = 8$ km/h : $\mu = 5$ km/h et $s = 1$ km/h. Les $\sigma_i$ tirés sont
dans `bruit.last_sigmas`, et `python src/vitesse/noise.py` vérifie ces chiffres sur 100 000 voitures.

À savoir : le maximum de vraisemblance suppose un seul $\sigma$ pour tous les points. Avec des $\sigma_i$ variables il
fonctionne quand même, mais le $\hat\sigma$ estimé est une sorte de moyenne : $\hat\sigma^2 \approx$ moyenne des $\sigma_i^2$.

**Étape 4 — la vitesse mesurée**

$$v_k = v(x_k) + \varepsilon_k$$

```python
return polynomial.speed(x) + noise.sample(len(x))     # simulate_speeds
```

### Exemple avec la graine 42 (les 3 premiers points)

| $k$ | $x_k$ (m) | $\xi_k$ | $v(x_k)$ vraie | $\varepsilon_k$ | $v_k$ mesurée |
|---|---|---|---|---|---|
| 1 | 1547,9 | 0,7740 | 87,94 | +2,00 | **89,94** |
| 2 | 877,8 | 0,4389 | 88,74 | −4,53 | **84,22** |
| 3 | 1717,2 | 0,8586 | 85,66 | −1,89 | **83,77** |

Les points $(x_k, v_k)$ sont ce que « reçoit » le système ; la vraie courbe $v(x)$ est ce qu'on cherchera à retrouver.

> **Ordre des tirages.** Les 100 positions sont tirées d'abord, puis les 100 valeurs de bruit : le générateur
> `rng` fournit ses nombres les uns après les autres, comme on pioche dans un paquet de cartes.
> Le bruit du point 1 est donc le 1er nombre du *deuxième* lot, pas le 1er nombre du programme.

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
- Pour retrouver **les mêmes données** il faut aussi **les mêmes paramètres** (`N_POINTS`, `SIGMA`, ...) et
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
inverse : on ne garde **que** les mesures et le degré $n_0$, et on cherche les coefficients $\hat c_0, \dots, \hat c_{n_0}$
et l'écart-type $\hat\sigma$ du bruit. Le code reprend la méthode du **TP2 (section 1, bruit blanc gaussien)** :
estimateur $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$, information de Fisher et borne de Cramér-Rao, avec le choix
entre la base canonique et la base de Legendre.

### 5.1 Le modèle supposé

Comme dans le TP, on écrit le polynôme dans une base $P_0, \dots, P_{n_0}$ :

$$
v_i = \sum_{n=0}^{n_0} c_n\, P_n(x_i) + b_i,
\qquad
b_i \sim \mathcal{N}(0, \sigma^2)
$$

Deux bases sont proposées (paramètre `basis`) :

| `basis` | $P_n(x)$ | Variable | Propriété |
|---|---|---|---|
| `"canonique"` (défaut) | $\xi^n$ | $\xi = x / L \in [0, 1]$ | mêmes coefficients que `SpeedPolynomial` |
| `"legendre"` | $\mathrm{Leg}_n(t)$ | $t = 2x/L - 1 \in [-1, 1]$ | quasi-orthogonale si les $x_i$ sont uniformes |

Les $n_0 + 1$ coefficients sont estimés **librement** : on ne suppose pas connus $V_0$ et $V_L$ (en vrai on ne les
connaît pas). Il n'y a donc pas les contraintes $c_0 = V_0$, etc. de la section 1.3.

Le passage à $t \in [-1, 1]$ pour Legendre n'est pas un détail : les polynômes de Legendre sont orthogonaux sur
$[-1, 1]$ (TP2, section 1.4). Comme nos positions sont tirées uniformément sur $[0, L]$ (section 2), les $t_i$ sont
uniformes sur $[-1, 1]$ : c'est exactement le cas favorable du TP.

### 5.2 La log-vraisemblance

Les points étant indépendants (un point par véhicule, section 0), on multiplie les densités normales de chaque
point ; en passant au logarithme, le produit devient une somme (TP2, section 1.2) :

$$
\ell(\mathbf c, \sigma)
= -\frac{1}{2\sigma^2} \sum_{i=1}^{N} \left[ v_i - \sum_{n=0}^{n_0} c_n P_n(x_i) \right]^2
- N \log\left(\sqrt{2\pi}\,\sigma\right)
$$

> **Dans le code** : `log_likelihood(coefficients, sigma, x, v, length, basis)` calcule exactement cette formule.

### 5.3 L'estimateur : $\hat{\mathbf c} = \mathbf M^{-1} \mathbf w$

On annule la dérivée par rapport à chaque $c_n$ :

$$
\partial_{c_n} \ell = \frac{1}{\sigma^2} \sum_{i=1}^{N} P_n(x_i) \left( v_i - \sum_{m=0}^{n_0} c_m P_m(x_i) \right) = 0
\quad\Longrightarrow\quad
\sum_{m=0}^{n_0} c_m \underbrace{\sum_{i=1}^{N} P_n(x_i) P_m(x_i)}_{M_{nm}}
= \underbrace{\sum_{i=1}^{N} P_n(x_i)\, v_i}_{w_n}
$$

C'est un système linéaire de $n_0 + 1$ équations, $\mathbf M \mathbf c = \mathbf w$, d'où l'estimateur du maximum
de vraisemblance (équation (1) du TP) :

$$
\boxed{\hat{\mathbf c} = \mathbf M^{-1} \mathbf w},
\qquad
M_{nm} = \sum_{i=1}^{N} P_n(x_i) P_m(x_i),
\qquad
w_n = \sum_{i=1}^{N} P_n(x_i)\, v_i
$$

(Le TP note ce vecteur $\mathbf v$ ; on l'appelle ici $\mathbf w$ pour ne pas le confondre avec les vitesses $v_i$.)

On remarque que $\sigma$ disparaît : l'estimation des coefficients ne dépend pas du niveau de bruit. C'est la
méthode des **moindres carrés** : pour un bruit gaussien, maximum de vraisemblance = moindres carrés.

En annulant ensuite la dérivée par rapport à $\sigma$ :

$$
\hat\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} \left(v_i - \hat v(x_i)\right)^2
$$

On divise par $N$ (et non $N - n_0 - 1$) : c'est le vrai estimateur du maximum de vraisemblance, légèrement biaisé
vers le bas, mais l'écart est négligeable quand $N \gg n_0$.

**Écriture matricielle.** La fonction `basis_matrix(x, length, degree, basis)` construit la matrice $\mathbf P$ de
taille $N \times (n_0 + 1)$ avec $\mathbf P_{i,n} = P_n(x_i)$. Alors $\mathbf M = \mathbf P^\top \mathbf P$ et
$\mathbf w = \mathbf P^\top \mathbf v$, et les vitesses du polynôme s'écrivent toutes d'un coup $\hat v(x_i) = (\mathbf P \hat{\mathbf c})_i$.
En base canonique, $\mathbf P$ est la matrice de Vandermonde ($\mathbf P_{i,n} = \xi_i^n$, fonction `vandermonde`).

### 5.4 La borne de Cramér-Rao (BCR)

Les dérivées secondes de $\ell$ ne dépendent pas des mesures (TP2, section 1.3) :

$$
\partial_{c_n} \partial_{c_m} \ell = -\frac{1}{\sigma^2} \sum_{i=1}^{N} P_n(x_i) P_m(x_i) = -\frac{M_{nm}}{\sigma^2}
\quad\Longrightarrow\quad
\mathbf I_F = \frac{1}{\sigma^2}\, \mathbf M,
\qquad
\mathrm{BCR} = \mathbf I_F^{-1} = \sigma^2\, \mathbf M^{-1}
$$

- Sur la diagonale : $\sqrt{\mathrm{BCR}_{nn}}$ est le plus petit écart-type possible pour estimer $c_n$ sans biais.
- Hors diagonale : la covariance entre $\hat c_n$ et $\hat c_m$. Un coefficient non nul signifie que les erreurs
  sur $c_n$ et $c_m$ sont **liées** (si l'un est surestimé, l'autre a tendance à être sous- ou surestimé).

Pour un bruit gaussien, l'estimateur $\hat{\mathbf c}$ est sans biais et **atteint** cette borne (il est efficace).
Le test `test_empirical_covariance_matches_the_crb` le vérifie par Monte-Carlo : 2000 tirages du bruit aux mêmes
positions, covariance empirique des $\hat{\mathbf c}$ comparée à $\sigma^2 \mathbf M^{-1}$.

### 5.5 Pourquoi la base de Legendre ?

En base canonique, les colonnes $1, \xi, \xi^2, \dots$ se ressemblent beaucoup sur $[0, 1]$ (elles sont toutes
croissantes) : $\mathbf M$ a de grands termes hors diagonale, donc les coefficients estimés sont **fortement
corrélés** et individuellement peu précis. En base de Legendre avec des $t_i$ uniformes sur $[-1, 1]$,
$M_{nm} \simeq 0$ pour $n \neq m$ : $\mathbf M$ et la BCR sont **quasi diagonales**, les coefficients sont
presque décorrélés.

Attention : changer de base ne change **pas** la courbe estimée $\hat v(x)$ (les deux bases décrivent les mêmes
polynômes, le test `test_legendre_basis_gives_the_same_curve_as_canonical` le vérifie). Seuls les **coefficients**
et leur interprétation changent. L'orthogonalité est perdue si les $x_i$ ne sont plus uniformes sur le tronçon.

### 5.6 La classe `MaximumLikelihood`

```python
estimateur = MaximumLikelihood(length=2000, degree=2, basis="legendre")   # ou "canonique" (défaut)
estimateur.fit(x, v)                   # calcule M, w, c_chapeau = M^-1 w et sigma_chapeau
estimateur.coefficients                # [c0, ..., c_n0] estimés DANS LA BASE CHOISIE
estimateur.canonical_coefficients      # convertis en base canonique (comparables à SpeedPolynomial)
estimateur.sigma                       # écart-type du bruit estimé [km/h]
estimateur.M                           # matrice M du TP
estimateur.fisher_information()        # I_F = M / sigma²
estimateur.crb()                       # BCR = sigma² M^-1  (avec sigma_chapeau, ou crb(sigma=5.0))
estimateur.speed(1000)                 # vitesse estimée en x = 1000 m
```

- **`__init__`** : mémorise $L$, $n_0$ et la base ; refuse $L \le 0$, $n_0 < 1$ et une base inconnue.
- **`fit(x, v)`** :
  1. vérifie que `x` et `v` ont la même taille et qu'il y a au moins $n_0 + 1$ points (sinon $\mathbf M$ n'est pas
     inversible) ;
  2. construit $\mathbf P$, puis $\mathbf M = \mathbf P^\top \mathbf P$ et $\mathbf w = \mathbf P^\top \mathbf v$ ;
  3. résout $\mathbf M \mathbf c = \mathbf w$ avec `np.linalg.solve` (même résultat que $\mathbf M^{-1}\mathbf w$,
     mais sans calculer l'inverse, ce qui est plus précis numériquement) ;
  4. calcule les résidus et $\hat\sigma = \sqrt{\text{moyenne des résidus}^2}$ ;
  5. renvoie `self`, ce qui permet d'écrire `MaximumLikelihood(L, d).fit(x, v)` en une ligne.
- **`crb(sigma=None)`** et **`fisher_information(sigma=None)`** : par défaut avec $\hat\sigma$ ; on peut passer le
  vrai $\sigma$ quand on le connaît (simulation).
- **`canonical_coefficients`** : pour Legendre, convertit $\sum c_n \mathrm{Leg}_n(t)$ en polynôme en $t$, puis
  remplace $t$ par $2\xi - 1$.
- **`speed(x)`** : renvoie $\hat v(x) = \mathbf P(x)\,\hat{\mathbf c}$ ; lève une erreur si `fit` n'a pas été appelé.

### 5.7 Exemple chiffré (graine 42)

Mêmes paramètres que les sections précédentes ($L = 2000$ m, $V_0 = 70$, $V_L = 80$, $n_0 = 2$, $\sigma = 5$ km/h,
100 points, graine 42), estimation avec le bon degré.

**Coefficients.** $\hat\sigma = 4{,}88$ km/h dans les deux bases.

| Base | | $c_0$ | $c_1$ | $c_2$ |
|---|---|---|---|---|
| canonique | vrai | 70,00 | 68,29 | −58,29 |
| canonique | estimé | 68,92 | 73,73 | −63,54 |
| canonique | $\sqrt{\mathrm{BCR}_{nn}}$ | 1,56 | 7,41 | 7,46 |
| Legendre | vrai | 84,71 | 5,00 | −9,71 |
| Legendre | estimé | 84,60 | 5,09 | −10,59 |
| Legendre | $\sqrt{\mathrm{BCR}_{nn}}$ | 0,51 | 0,93 | 1,24 |

Converti en base canonique, l'estimé Legendre redonne exactement (68,92 ; 73,73 ; −63,54) : c'est la même courbe,
avec un écart quadratique moyen de 0,42 km/h à la vraie courbe sur le tronçon.

**Corrélations** ($\mathrm{BCR}_{nm} / \sqrt{\mathrm{BCR}_{nn}\mathrm{BCR}_{mm}}$) :

| | canonique | Legendre |
|---|---|---|
| $(c_0, c_1)$ | −0,87 | 0,07 |
| $(c_1, c_2)$ | −0,97 | 0,15 |
| $(c_0, c_2)$ | 0,75 | 0,14 |

En base canonique, $c_1$ et $c_2$ sont corrélés à −0,97 : l'erreur de +5,4 sur $c_1$ est presque compensée par
l'erreur de −5,3 sur $c_2$, ce qui explique que la courbe reste bonne alors que les coefficients sont loin.
En Legendre, chaque coefficient est estimé presque indépendamment des autres, avec des barres d'erreur bien plus
petites.

### 5.8 Limites à connaître

- **Bruit non gaussien** (autre loi, valeurs aberrantes) : la formule de 5.2 n'est plus la bonne vraisemblance,
  et les moindres carrés deviennent sensibles aux points aberrants (le TP2, section 2, traite le cas Poisson
  avec un algorithme itératif).
- **$\sigma_i$ variable par voiture** (section 2, variante) : le modèle suppose un seul $\sigma$. L'estimation
  des coefficients reste sans biais, mais $\hat\sigma^2$ estime en fait la moyenne des $\sigma_i^2$, et la BCR
  $\sigma^2 \mathbf M^{-1}$ n'est plus exacte.
- **Mauvais degré** : `main_04` permet de tester `D_ESTIME = D + 1` (le terme en trop est estimé proche de 0
  mais augmente la variance, surtout en base canonique) ou `D_ESTIME = D - 1` (la courbe ne peut plus suivre la
  vraie forme : biais).

> **Pour essayer** : `python tests/mains/main_04_maximum_vraisemblance.py` trace les mesures, la vraie courbe,
> l'estimation et une bande $\pm$ `N_SIGMA` $\cdot\,\hat\sigma$ ; `pytest tests/test_maximum_likelihood.py -v`
> lance les tests.

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
| Log-vraisemblance | `src/vitesse/maximum_likelihood.py` | `log_likelihood` |
| Estimer $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$ et $\hat\sigma$ | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fit`, `.speed` |
| Information de Fisher, BCR | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fisher_information`, `.crb` |
| Simuler puis estimer | `tests/mains/main_04_maximum_vraisemblance.py` | `main()` |