# Formulation du problème et questions pour le professeur

## 1. Problème (reformulé)

Un tronçon de route de longueur $L$ est parcouru par des véhicules. Chaque véhicule envoie des
couples **(position $x$, vitesse ponctuelle $v$)** (pas de vitesse continue). Exemple de référence :
$N = 100$ points issus de $M = 70$ véhicules (certains envoient plusieurs fois).
On cherche le **profil de vitesse moyenne $v(x)$** sur le tronçon, supposé polynomial de degré $d$ connu :

$$
v(x) = c_0 + c_1 \xi + \dots + c_d \xi^d, \qquad \xi = \frac{2x}{L} - 1 \in [-1, 1]
$$

## 2. Modèle de données

$$
v_k = v(x_k) + \varepsilon_k, \qquad k = 1, \dots, N
$$

- **Cas simple** : $\varepsilon_k \sim \mathcal{N}(0, \sigma^2)$ i.i.d., positions $x_k$ uniformes sur $[0, L]$.
- **Cas difficile** (à préciser avec le professeur) : 
   - Valeurs aberrantes --> créées automatiquement par la génération ?
   - Variance différente par capteur : **implémenté** (`Noise(..., sigma_range=(alpha1, alpha2))`). Un point par voiture ; pour chaque point on tire $\sigma_i \sim \mathcal{N}\big(\frac{\alpha_1+\alpha_2}{2}, (\frac{\alpha_2-\alpha_1}{6})^2\big)$ (99,7 % dans $[\alpha_1, \alpha_2]$, les autres ramenés à la borne la plus proche), puis $\varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)$. Détails : `explications.md`, section 0
   - Si plusieurs virages --> décélération puis réaccélération ? Si oui, corrélation possible avec d et le nombre de virages ? Pour ça, voir TP1/2


## 3. Points bloquants / questions pour lundi

**Sur la simulation**
1. Les positions sont-elles aléatoires uniformes ou imposées (plan d'expérience) ? Le nombre
   d'envois par véhicule suit-il une loi précise ? Un véhicule peut-il envoyer plusieurs valeurs ? (Si oui : à intervalle de distance régulier ? Un véhicule lent envoie-t-il plus de positions qu'un rapide ?)
   
   - On veut des tirages indépendants, donc on choisit d'avoir un unique point par véhicule sur un tronçon. On utilise une loi normale pour les $\sigma_i$ car il représente plusieurs sources de bruit.

2. Est-ce qu'en générant avec RNG, mais avec une même graine, une loi uniforme et la loi normale, $x_k$ devient corrélé à $\epsilon_k$ ?

3. Est-ce que je génère la position $x$ avec une loi uniforme ou est-ce que je prends une répartition linéaire sur $[0,L]$ ?

   - Loi uniforme

4. Différentes variances sur les véhicules : de position et de vitesse ? Que de vitesse ?
      - Que de vitesse

5. Est-ce qu'on travaille avec des vecteurs du coup ? Donc « BCR dans le cas vectoriel » ? 

6. Est-ce que la variance cache en réalité la précision du capteur + le style de conduite différent ? Est-ce que l'un est majoritaire sur l'autre ? Est-ce qu'on considère qu'on n'en prend qu'un seul en compte ?
      - Variance : style de conduite + précision du capteur (mais style similaire).
      Les styles très différents seront traités dans la « robustesse ».

**Sur la robustesse**

1. Robuste **à quoi** : aux valeurs aberrantes (conducteur qui s'arrête/ralentit) --> loin de l'estimation. Comment définit-on le « loin » ? (Seuil par rapport à la vitesse max ? Seuil par rapport à 0 ?)

2. Générer un autre style de conduite avec $N_d$ points via (à définir/choisir) et l'ajouter à notre simulation de $N$ points.

**Sur la caractérisation**

1. Comment le caractériser ? Quels outils ?
      - Biais, variance 

      - Matrice de Fisher
      - Borne de Cramér-Rao
      - Vu peut-être lors des prochains TP2 ?
      - Estimation bayésienne ?

2. Que se passe-t-il si $D \neq D_{estimé}$ ? 



## 4. À faire
**À faire durant la séance du lundi 16/11**
 1. Répondre à la question : comment caractériser ? Voir les TP2 et le cours pour trouver une ou des méthodes.
 
 2. Différentes variances sur les véhicules (car précision des capteurs différente)

 **À faire pour la prochaine séance**

















