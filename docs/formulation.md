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
   d'envois par vehicule suit-il une loi precise ? Un véhicule peut envoyer plusieurs valeurs ? A intervalle de distance régulière ? 

2. Est ce que en générant avec RNG mais avec une meme graine une loi uniforme et la loi normale, $x_k$ devient corrélé à $\epsilon_k$ ?

**Sur la robustesse et la caracterisation**

3. Robuste **a quoi** : aux valeurs aberrants (conducteur qui s'arrete/ralenti)--> loin de l'estimation. Comment on définit le loin ? (Seuil par rapport à la vitesse max ? Seuil par rapport à 0)

4. Comment je le caractérise ? Quel outils ?

      - Borne de Cramer Rao
      - Vu peut-être lors des prochains TP ?

5. Qu'est ce qu'il se passe si $D \neq D_{estimé}$ ? 
