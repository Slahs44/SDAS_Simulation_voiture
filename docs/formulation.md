# Formulation du probleme et questions pour le professeur

## 1. Probleme (reformule)

Un troncon de route de longueur $L$ est parcouru par des vehicules. Chaque vehicule envoie des
couples **(position $x$, vitesse ponctuelle $v$)** (pas de vitesse continue). Exemple de reference :
$N = 100$ points issus de $M = 70$ vehicules (certains envoient plusieurs fois).
On cherche le **profil de vitesse moyenne $v(x)$** sur le troncon, suppose polynomial de degre $d$ connu :

$$
v(x) = c_0 + c_1 \xi + \dots + c_d \xi^d, \qquad \xi = \frac{2x}{L} - 1 \in [-1, 1]
$$

## 2. Modele de donnees

$$
v_k = v(x_k) + \varepsilon_k, \qquad k = 1, \dots, N
$$

- **Cas simple** : $\varepsilon_k \sim \mathcal{N}(0, \sigma^2)$ i.i.d., positions $x_k$ uniformes sur $[0, L]$.
- **Cas difficile** (a preciser avec le professeur) : 
   - valeurs aberrantes --> crée automatiquement par la génération ?
   - Variance différente par capteur ? Pour chaque point : Génération aléatoire d'une variance --> puis d'un point suivant la loi de proba choisi (et défini par sa variance)
   - Si plusieurs virage --> décéleration puis réaccélération ? Si oui possible correlation avec d et nb virage ? Pour ca voir TP1/2


## 3. Points bloquants / questions pour lundi

**Sur la simulation**
1. Les positions sont-elles aleatoires uniformes ou imposees (plan d'experience) ? Le nombre
   d'envois par vehicule suit-il une loi precise ? Un véhicule peut envoyer plusieurs valeurs ?(Si oui : A intervalle de distance régulière ? Un véhicule lent envoie plus de position qu'un rapide ?)
   
   - On veut des tirage indépendant donc on choisit d'avoir un unique point par véhicule sur un tronçon.

2. Est ce que en générant avec RNG mais avec une meme graine une loi uniforme et la loi normale, $x_k$ devient corrélé à $\epsilon_k$ ?

3. Est ce que je génère la position $x$ avec une loi uniforme ou je prends une répartition linéaire sur $[0,L]$ ?

4. Différente variance sur les véhicules : de position et vitesse ? que vitesse ?

5. Est ce qu'on travail avec des vecteurs du coup ? Donc "BCR dans le cas vectoriel" ? 

6. Est ce que la variance cache en réalité la précision du capteur + le style de conduite différent ? Est ce que l'un est majoritaire sur l'autre ? Est ce qu'on considère qu'on en prend qu'un seul en compte


**Sur la robustesse et la caracterisation**

1. Robuste **a quoi** : aux valeurs aberrants (conducteur qui s'arrete/ralenti)--> loin de l'estimation. Comment on définit le loin ? (Seuil par rapport à la vitesse max ? Seuil par rapport à 0)

2. Comment je le caractérise ? Quel outils ?

      - Borne de Cramer Rao
      - Vu peut-être lors des prochains TP ?
      - Estimation Baisienne ?

3. Qu'est ce qu'il se passe si $D \neq D_{estimé}$ ? 



## 4. A Faire
**A faire  durant la séance Lundi 16/11
 1. Répondre question comment caractériser. Voir les TP et cours pour trouver un/des methode(s).
 
 2. Différente variance sur les véhicules (car précision capteur différente)

 **A faire pour la prochaine séance