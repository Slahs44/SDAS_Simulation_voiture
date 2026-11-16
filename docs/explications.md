# Explication : comment les données simulées sont générées

Ce document explique la génération sur un petit exemple : $L = 2000$ m, $V_0 = 70$ km/h, $V_L = 80$ km/h,
limite 90 km/h, degré $d = 2$, écarts-types $\sigma_i \in [\alpha_1, \alpha_2] = [2, 8]$ km/h, 100 points (= 100 voitures).
(Les valeurs par défaut de `main_03_simulation.py` sont différentes : $L = 1500$ m, $V_0 = 40$, $V_L = 35$ km/h,
$d = 4$, 1000 points ; même plage $[2, 8]$ km/h.)

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
x_i \sim \mathcal{U}\left([0, L]\right) & \longrightarrow & v_i = P(x_i) + b_i \ \longrightarrow\ \left(x_i, v_i\right)_{1 \le i \le N} \ \text{(ce que le système reçoit)} \\[6pt]
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

## 5. Retrouver le polynôme : le maximum de vraisemblance (`maximum_likelihood.py`)

Les sections 1 à 4 fabriquent les mesures $(x_i, v_i)$ à partir d'un polynôme **connu**. Ici on fait le chemin
inverse : on ne garde **que** les mesures et le degré $d$, et on cherche les coefficients $\hat c_0, \dots, \hat c_d$.
Le code reprend la méthode du **TP2 (section 1, bruit blanc gaussien)** : on annule le gradient de la
log-vraisemblance, ce qui donne $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$, puis on calcule l'information de Fisher
et la borne de Cramér-Rao.

Le TP2 suppose un **même** écart-type $\sigma$ pour toutes les mesures. Ce n'est pas notre modèle (section 0) :
chaque voiture $i$ a son propre $\sigma_i$. On a donc refait le calcul avec des $\sigma_i$ différents, et le code
propose **deux estimateurs** au choix (paramètre `estimator`) :

| `estimator` | Hypothèse | Ce qu'on connaît | Ce qu'on estime |
|---|---|---|---|
| `"sigma_constant"` (défaut) | $\forall i,\ \sigma_i = \sigma$ (TP2) | $x_i, v_i$ | $\hat{\mathbf c}$ et $\hat\sigma$ |
| `"sigma_variable"` | un $\sigma_i$ par voiture | $x_i, v_i$ et les $\sigma_i$ | $\hat{\mathbf c}$ |

Notations : $d$ est le degré (noté $n_0$ dans le TP). Le TP écrit le polynôme dans une base quelconque
$P_n(x)$ ; on prend ici la base canonique $P_n(x) = x^n$, donc $P_n(x_i) = x_i^n$.

> **Dans le code**, $x_i$ désigne la position normalisée $\xi_i = x_i / L \in [0, 1]$ (section 1.1), pour éviter
> des nombres énormes comme $2000^6$. Le code propose aussi la base de Legendre (`basis="legendre"`), où $x_i^n$
> est remplacé par $\mathrm{Leg}_n(t_i)$ avec $t_i = 2x_i/L - 1 \in [-1, 1]$ (section 5.6). Toutes les formules
> ci-dessous restent vraies en remplaçant $x_i^n$ par $P_n(x_i)$.

### 5.1 Le modèle

$$
v_i = \sum_{n=0}^{d} c_n\, x_i^n + b_i,
\qquad
b_i \sim \mathcal{N}(0, \sigma_i^2),
\qquad
\text{les } b_i \text{ indépendants}
$$

Les $d + 1$ coefficients sont estimés **librement** : on ne suppose pas connus $V_0$ et $V_L$ (en vrai on ne les
connaît pas). Il n'y a donc pas les contraintes $c_0 = V_0$, etc. de la section 1.3.

### 5.2 La log-vraisemblance

Les points étant indépendants (un point par véhicule, section 0), on multiplie les densités normales de chaque
point ; en passant au logarithme, le produit devient une somme :

$$
\ell(\mathbf c)
= -\sum_{i=1}^{N} \frac{1}{2\sigma_i^2} \left[ v_i - \sum_{n=0}^{d} c_n x_i^n \right]^2
- \sum_{i=1}^{N} \log\left(\sqrt{2\pi}\,\sigma_i\right)
$$

Avec $\sigma_i = \sigma$ pour tout $i$, on retrouve la formule du TP2 (section 1.2) :
$\ell = -\frac{1}{2\sigma^2} \sum_i [\dots]^2 - N \log(\sqrt{2\pi}\,\sigma)$.

> **Dans le code** : `log_likelihood(coefficients, sigma, x, v, length, basis)`, où `sigma` est soit un nombre
> (même $\sigma$ pour tous), soit un tableau des $\sigma_i$.

### 5.3 L'estimateur : $\hat{\mathbf c} = \mathbf M^{-1} \mathbf w$

**Cas général ($\sigma_i$ différents, estimateur `"sigma_variable"`).** On annule la dérivée par rapport à
chaque $c_n$ :

$$
\partial_{c_n} \ell = \sum_{i=1}^{N} \frac{1}{\sigma_i^2}\, x_i^n \left( v_i - \sum_{m=0}^{d} c_m x_i^m \right) = 0
\quad\Longrightarrow\quad
\sum_{m=0}^{d} c_m \underbrace{\sum_{i=1}^{N} \frac{x_i^n\, x_i^m}{\sigma_i^2}}_{M_{nm}}
= \underbrace{\sum_{i=1}^{N} \frac{x_i^n\, v_i}{\sigma_i^2}}_{w_n}
$$

Le facteur $1/\sigma_i^2$ ne peut pas sortir de la somme (il dépend de $i$) : il reste **dans** $M_{nm}$ **et**
dans $w_n$. C'est un système linéaire de $d + 1$ équations, $\mathbf M \mathbf c = \mathbf w$, d'où :

$$
\boxed{\hat{\mathbf c} = \mathbf M^{-1} \mathbf w},
\qquad
M_{nm} = \sum_{i=1}^{N} \frac{x_i^{n+m}}{\sigma_i^2},
\qquad
w_n = \sum_{i=1}^{N} \frac{x_i^n\, v_i}{\sigma_i^2}
$$

C'est la méthode des **moindres carrés pondérés** : on minimise $\sum_i (v_i - \hat v(x_i))^2 / \sigma_i^2$.
Une voiture précise (petit $\sigma_i$) pèse plus dans l'estimation qu'une voiture imprécise. Cet estimateur
suppose que l'on **connaît** les $\sigma_i$ (par exemple, chaque voiture envoie la précision de son capteur avec
sa mesure). Dans la simulation, ce sont les $\sigma_i$ tirés par `Noise` (`bruit.last_sigmas`).

**Cas du TP2 ($\sigma_i = \sigma$ pour tout $i$, estimateur `"sigma_constant"`).** Le facteur $1/\sigma^2$ est
le même partout : il sort des deux sommes et se simplifie. On retrouve l'équation (1) du TP :

$$
\hat{\mathbf c} = \mathbf M^{-1} \mathbf w,
\qquad
M_{nm} = \sum_{i=1}^{N} x_i^{n+m},
\qquad
w_n = \sum_{i=1}^{N} x_i^n\, v_i
$$

$\sigma$ a disparu : pas besoin de le connaître pour estimer les coefficients. C'est la méthode des
**moindres carrés** ordinaires. On estime ensuite $\sigma$ en annulant la dérivée par rapport à $\sigma$ :

$$
\hat\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} \left(v_i - \hat v(x_i)\right)^2
$$

(division par $N$ et non $N - d - 1$ : c'est le vrai maximum de vraisemblance, légèrement biaisé vers le bas,
négligeable quand $N \gg d$).

(Le TP note le vecteur $\mathbf v$ ; on l'appelle ici $\mathbf w$ pour ne pas le confondre avec les vitesses $v_i$.)

**Écriture matricielle.** `basis_matrix(x, length, degree, basis)` construit la matrice $\mathbf P$ de taille
$N \times (d + 1)$ avec $\mathbf P_{i,n} = x_i^n$ (matrice de Vandermonde). Avec la matrice diagonale des poids
$\mathbf W = \mathrm{diag}(1/\sigma_1^2, \dots, 1/\sigma_N^2)$ :

$$
\mathbf M = \mathbf P^\top \mathbf W \mathbf P,
\qquad
\mathbf w = \mathbf P^\top \mathbf W \mathbf v
\qquad
(\mathbf W = \mathbf I \text{ pour } \texttt{"sigma\_constant"})
$$

### 5.4 La borne de Cramér-Rao (BCR)

Les dérivées secondes de $\ell$ ne dépendent pas des mesures :

$$
\partial_{c_n} \partial_{c_m} \ell = -\sum_{i=1}^{N} \frac{x_i^n\, x_i^m}{\sigma_i^2}
\quad\Longrightarrow\quad
\mathbf I_F = \mathbf M \ \text{(avec les poids } 1/\sigma_i^2\text{)},
\qquad
\mathrm{BCR} = \mathbf I_F^{-1}
$$

| Estimateur | $\mathbf M$ | $\mathbf I_F$ | BCR |
|---|---|---|---|
| `"sigma_variable"` | $\sum_i x_i^{n+m} / \sigma_i^2$ | $\mathbf M$ | $\mathbf M^{-1}$ |
| `"sigma_constant"` (TP2) | $\sum_i x_i^{n+m}$ | $\mathbf M / \sigma^2$ | $\sigma^2\, \mathbf M^{-1}$ (calculée avec $\hat\sigma$) |

- Sur la diagonale : $\sqrt{\mathrm{BCR}_{nn}}$ est le plus petit écart-type possible pour estimer $c_n$ sans biais.
- Hors diagonale : la covariance entre $\hat c_n$ et $\hat c_m$. Un terme non nul signifie que les erreurs sur
  $c_n$ et $c_m$ sont **liées** (si l'un est surestimé, l'autre a tendance à être sous- ou surestimé).

Les deux estimateurs sont **sans biais** (l'espérance de $\mathbf w$ vaut $\mathbf M \mathbf c$ dans les deux cas).
Sous le bon modèle, chacun **atteint** sa borne :

- `"sigma_variable"` atteint $\mathbf M^{-1}$ quand les $\sigma_i$ sont vraiment différents (notre cas) ;
- `"sigma_constant"` atteint $\sigma^2 \mathbf M^{-1}$ quand le bruit est vraiment de même $\sigma$ partout.

Quand les $\sigma_i$ sont différents, `"sigma_constant"` reste sans biais mais **ne l'atteint plus** : sa vraie
covariance est $\mathbf M_0^{-1} \left(\sum_i \sigma_i^2\, \mathbf p_i \mathbf p_i^\top\right) \mathbf M_0^{-1}$
(avec $\mathbf M_0 = \sum_i \mathbf p_i \mathbf p_i^\top$, $\mathbf p_i = (1, x_i, \dots, x_i^d)$), toujours
supérieure ou égale à la borne $\mathbf M^{-1}$ de l'estimateur pondéré. Sa « BCR » $\hat\sigma^2 \mathbf M_0^{-1}$
n'est alors qu'une approximation.

Les tests le vérifient par Monte-Carlo (2000 tirages du bruit aux mêmes positions et mêmes $\sigma_i$) :
`test_empirical_covariance_matches_the_crb` (cas $\sigma$ constant) et
`test_sigma_variable_reaches_its_crb_and_beats_sigma_constant` (cas $\sigma_i$ variables : l'estimateur pondéré
atteint $\mathbf M^{-1}$ et a une variance plus petite que celui du TP2 sur chaque coefficient).

### 5.5 Quel estimateur choisir ?

- Si les $\sigma_i$ sont **connus** : `"sigma_variable"`, qui est le vrai maximum de vraisemblance de notre modèle.
- Si on ne connaît **pas** les $\sigma_i$ : `"sigma_constant"`. Il reste sans biais, il est seulement un peu moins
  précis, et $\hat\sigma^2$ estime alors la moyenne des $\sigma_i^2$.
- Si tous les $\sigma_i$ sont égaux, les deux donnent **exactement** le même $\hat{\mathbf c}$ (les poids se
  simplifient ; vérifié par `test_equal_sigmas_give_the_same_estimate_as_sigma_constant`).

Le gain du pondéré grandit quand les $\sigma_i$ sont très différents les uns des autres. Avec
$[\alpha_1, \alpha_2] = [2, 8]$ km/h, les $\sigma_i$ sont concentrés autour de 5 km/h ($s = 1$ km/h) : le gain
est modeste (section 5.8).

### 5.6 Pourquoi la base de Legendre ?

En base canonique, les colonnes $1, x, x^2, \dots$ se ressemblent beaucoup sur $[0, 1]$ (elles sont toutes
croissantes) : $\mathbf M$ a de grands termes hors diagonale, donc les coefficients estimés sont **fortement
corrélés** et individuellement peu précis. En base de Legendre ($t = 2x/L - 1$) avec des $t_i$ uniformes sur
$[-1, 1]$ (c'est le cas : les positions sont tirées uniformément, section 2), $M_{nm} \simeq 0$ pour $n \neq m$
(TP2, section 1.4) : $\mathbf M$ et la BCR sont **quasi diagonales**, les coefficients sont presque décorrélés.
Cela reste vrai avec les poids $1/\sigma_i^2$, tant que les $\sigma_i$ ne dépendent pas de la position.

Changer de base ne change **pas** la courbe estimée $\hat v(x)$ (les deux bases décrivent les mêmes polynômes,
vérifié par `test_legendre_basis_gives_the_same_curve_as_canonical`). Seuls les **coefficients** et leur
interprétation changent. L'orthogonalité est perdue si les $x_i$ ne sont plus uniformes sur le tronçon.

### 5.7 La classe `MaximumLikelihood`

```python
est = MaximumLikelihood(length=2000, degree=2, basis="canonique", estimator="sigma_variable")
est.fit(x, v, sigmas=bruit.last_sigmas)  # "sigma_variable" : sigmas obligatoires
#   ou : MaximumLikelihood(2000, 2).fit(x, v)   # "sigma_constant" (défaut, TP2) : pas besoin des sigmas
est.coefficients                # [c0, ..., cd] estimés DANS LA BASE CHOISIE
est.canonical_coefficients      # convertis en base canonique (comparables à SpeedPolynomial)
est.sigma                       # "sigma_constant" : sigma_chapeau [km/h] ; "sigma_variable" : None
est.sigmas                      # "sigma_variable" : les sigma_i utilisés
est.M, est.w                    # la matrice M et le vecteur w (avec les poids 1/sigma_i²)
est.fisher_information()        # I_F
est.crb()                       # BCR = I_F^-1
est.speed(1000)                 # vitesse estimée en x = 1000 m
```

- **`__init__`** : mémorise $L$, $d$, la base et l'estimateur ; refuse $L \le 0$, $d < 1$, une base ou un
  estimateur inconnus.
- **`fit(x, v, sigmas=None)`** :
  1. vérifie que `x` et `v` ont la même taille et qu'il y a au moins $d + 1$ points (sinon $\mathbf M$ n'est pas
     inversible) ;
  2. poids : $1/\sigma_i^2$ pour `"sigma_variable"` (les `sigmas` doivent être fournis, de la bonne taille et
     $> 0$), $1$ pour `"sigma_constant"` (les `sigmas` éventuels sont ignorés) ;
  3. construit $\mathbf P$, puis $\mathbf M = \mathbf P^\top \mathbf W \mathbf P$ et $\mathbf w = \mathbf P^\top \mathbf W \mathbf v$ ;
  4. résout $\mathbf M \mathbf c = \mathbf w$ avec `np.linalg.solve` (même résultat que $\mathbf M^{-1}\mathbf w$,
     mais sans calculer l'inverse, ce qui est plus précis numériquement) ;
  5. pour `"sigma_constant"` seulement : $\hat\sigma = \sqrt{\text{moyenne des résidus}^2}$ ;
  6. renvoie `self`, ce qui permet d'écrire `MaximumLikelihood(L, d).fit(x, v)` en une ligne.
- **`fisher_information(sigma=None)`**, **`crb(sigma=None)`** : voir le tableau de 5.4. Pour `"sigma_constant"`,
  on peut passer le vrai $\sigma$ (simulation) au lieu de $\hat\sigma$ ; pour `"sigma_variable"`, l'argument est
  ignoré.
- **`canonical_coefficients`** : pour Legendre, convertit $\sum c_n \mathrm{Leg}_n(t)$ en polynôme en $t$, puis
  remplace $t$ par $2\xi - 1$.
- **`speed(x)`** : renvoie $\hat v(x)$ ; lève une erreur si `fit` n'a pas été appelé.

### 5.8 Exemple chiffré (graine 42)

Paramètres du début du document : $L = 2000$ m, $V_0 = 70$, $V_L = 80$, $d = 2$, 100 voitures, graine 42,
$\sigma_i$ tirés dans $[2, 8]$ km/h (on obtient des $\sigma_i$ entre 2,87 et 7,91 km/h, $\sqrt{\text{moyenne des } \sigma_i^2} = 5{,}08$ km/h).
Estimation avec le bon degré.

**Coefficients en base canonique.**

| | $c_0$ | $c_1$ | $c_2$ | $\hat\sigma$ | écart à la vraie courbe (RMS) |
|---|---|---|---|---|---|
| vrai | 70,00 | 68,29 | −58,29 | | |
| `"sigma_constant"` | 71,71 | 62,43 | −55,35 | 5,05 | 0,91 km/h |
| `"sigma_variable"` | 71,10 | 65,79 | −58,70 | — | 0,89 km/h |

**Écarts-types de la BCR** $\sqrt{\mathrm{BCR}_{nn}}$ :

| | canonique, `sigma_constant` | canonique, `sigma_variable` | Legendre, `sigma_constant` | Legendre, `sigma_variable` |
|---|---|---|---|---|
| $c_0$ | 1,57 | 1,50 | 0,51 | 0,47 |
| $c_1$ | 7,49 | 7,05 | 0,94 | 0,86 |
| $c_2$ | 7,53 | 6,95 | 1,26 | 1,16 |

**Monte-Carlo** (5000 tirages du bruit, mêmes positions et mêmes $\sigma_i$, base canonique) : écarts-types
empiriques (1,57 ; 7,40 ; 7,33) pour `"sigma_constant"` contre (1,49 ; 6,96 ; 6,88) pour `"sigma_variable"`, et
écart moyen à la vraie courbe 0,85 km/h contre 0,79 km/h. L'estimateur pondéré est environ 6 % plus précis ici ;
le gain serait plus grand avec une plage $[\alpha_1, \alpha_2]$ plus large.

**Corrélations** ($\mathrm{BCR}_{nm} / \sqrt{\mathrm{BCR}_{nn}\mathrm{BCR}_{mm}}$, `"sigma_variable"`) :

| | canonique | Legendre |
|---|---|---|
| $(c_0, c_1)$ | −0,88 | 0,04 |
| $(c_1, c_2)$ | −0,97 | 0,07 |
| $(c_0, c_2)$ | 0,77 | 0,11 |

En base canonique, $c_1$ et $c_2$ sont corrélés à −0,97 : une erreur sur $c_1$ est presque compensée par une erreur
de signe opposé sur $c_2$, ce qui explique que la courbe reste bonne alors que les coefficients sont loin. En
Legendre, chaque coefficient est estimé presque indépendamment des autres, avec des barres d'erreur bien plus
petites (vrais coefficients de Legendre : 84,71 ; 5,00 ; −9,71).

### 5.9 Limites à connaître

- **Bruit non gaussien** (autre loi, valeurs aberrantes) : la formule de 5.2 n'est plus la bonne vraisemblance,
  et les moindres carrés deviennent sensibles aux points aberrants (le TP2, section 2, traite le cas Poisson
  avec un algorithme itératif).
- **$\sigma_i$ inconnus** : `"sigma_variable"` n'est pas utilisable tel quel ; on revient à `"sigma_constant"`.
  Si on donne des $\sigma_i$ faux, l'estimateur reste sans biais mais n'est plus optimal, et sa BCR $\mathbf M^{-1}$
  est fausse.
- **Mauvais degré** : `main_04` permet de tester `D_ESTIME = D + 1` (le terme en trop est estimé proche de 0
  mais augmente la variance, surtout en base canonique) ou `D_ESTIME = D - 1` (la courbe ne peut plus suivre la
  vraie forme : biais).

> **Pour essayer** : `python tests/mains/main_04_maximum_vraisemblance.py` (paramètres `ESTIMATEUR` et `BASE`)
> trace les mesures, la vraie courbe, l'estimation et une bande $\pm$ `N_SIGMA` $\cdot\,\sigma$ global, et affiche
> l'écart à la vraie courbe des **deux** estimateurs ; `pytest tests/test_maximum_likelihood.py -v` lance les tests.

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
| Estimer $\hat{\mathbf c} = \mathbf M^{-1}\mathbf w$ (`sigma_constant` ou `sigma_variable`) | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fit`, `.speed` |
| Information de Fisher, BCR | `src/vitesse/maximum_likelihood.py` | `MaximumLikelihood.fisher_information`, `.crb` |
| Simuler puis estimer | `tests/mains/main_04_maximum_vraisemblance.py` | `main()` |
| Comparer les deux estimateurs sur un même graphe | `tests/mains/main_05_comparaison_estimateurs.py` | `main()` |
