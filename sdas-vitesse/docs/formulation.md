# Formulation du probleme et questions pour le professeur

## 1. Probleme (reformule)

Un troncon de route de longueur L est parcouru par des vehicules. Chaque vehicule envoie des
couples **(position x, vitesse ponctuelle v)** (pas de vitesse continue). Exemple de reference :
N = 100 points issus de M = 70 vehicules (certains envoient plusieurs fois).
On cherche le **profil de vitesse moyenne v(x)** sur le troncon, suppose polynomial de degre d connu :

    v(x) = c_0 + c_1 xi + ... + c_d xi^d ,   xi = 2x/L - 1 in [-1, 1]

## 2. Modele de donnees

    v_k = v(x_k) + eps_k ,   k = 1..N

- **Cas simple** : eps_k ~ N(0, sigma^2) i.i.d., positions x_k uniformes sur [0, L].
- **Cas difficile** (a preciser avec le professeur) : valeurs aberrantes, bruit correle par vehicule,
  variance dependant de x, trou de couverture des positions, mauvais degre.

## 3. Ce que je pense faire

1. **Estimateur de base** : moindres carres ordinaires (MCO). C'est le maximum de vraisemblance
   si le bruit est gaussien i.i.d. ; coefficients c = (A^T A)^-1 A^T v (calcul par `lstsq`).
2. **Caracterisation** : biais, covariance, RMSE de v_hat(x) par Monte-Carlo ; comparaison a la
   borne de Cramer-Rao sigma^2 (A^T A)^-1 ; sensibilite a sigma, N, d, au conditionnement (base de Legendre).
3. **Cas difficile** : montrer ou le MCO casse (aberrants, correlation, trou de couverture,
   mauvais degre), puis proposer des estimateurs robustes (Huber, moindres carres ponderes iteres,
   RANSAC, mediane, ...) et les comparer au MCO par les memes experiences.
4. **Illustration** : tracer v(x) estimee avec bande de confiance sur des exemples simules.

Resultats actuels (d = 2, sigma = 2 m/s, N = 100, M = 70, 1000 tirages Monte-Carlo) :

| Scenario                          | RMSE globale sur v(x) |
|-----------------------------------|-----------------------|
| Reference (gaussien, d correct)   | 0,36 m/s              |
| Sous-ajuste (d = 1)               | 2,46 m/s (biais 5,4)  |
| Sur-ajuste (d = 5)                | 0,52 m/s              |
| 10 % d'aberrants (x10 sigma)      | 1,16 m/s              |
| Effet vehicule (sigma_v = 3)      | 0,68 m/s              |
| Points sur 0-50 % du troncon      | 2,28 m/s              |

En gaussien, l'ecart-type empirique des coefficients coincide avec la borne de Cramer-Rao (MCO efficace).

## 4. Points bloquants / questions pour lundi

**Sur le sens de la grandeur estimee**
1. v(x) est-elle la **moyenne de la population de vehicules en x** ? Dans ce cas, l'ecart d'un vehicule
   a v(x) (style de conduite) fait-il partie du « bruit » ? C'est ce qui decide entre bruit i.i.d.
   et bruit correle par vehicule (un vehicule qui envoie 2 fois).
2. « Vitesse moyenne » : moyenne spatiale ou temporelle (moyenne harmonique) ? Veut-on le profil v(x)
   ou aussi un scalaire (moyenne de v sur le troncon) ?

**Sur la simulation**
3. Que signifie « difficile » pour les donnees : aberrants (arrets, bouchons), bruit a queue lourde,
   variance dependant de x, bruit sur x (GPS), ou plusieurs de ces effets ?
4. Les positions sont-elles aleatoires uniformes ou imposees (plan d'experience) ? Le nombre
   d'envois par vehicule suit-il une loi precise ?
5. sigma est-il connu ou a estimer ? Le bruit est-il gaussien dans le cas simple *et* centre ?

**Sur la robustesse et la caracterisation**
6. Robuste **a quoi** : aberrants, mauvaise loi de bruit, mauvais degre d ? Quels outils attend-on
   (point de rupture, fonction d'influence, efficacite relative) ?
7. Quel critere de performance : RMSE integree sur le troncon, erreur max (norme sup), erreur aux extremites
   (la variance est maximale aux bords) ?
8. Comment traiter le choix de d s'il est connu « au moment du calcul » : doit-on aussi proposer un
   critere de selection (AIC/BIC, validation croisee) ou rester a d fixe ?

**Sur la forme du livrable**
9. Format attendu du rapport et de la soutenance ; place du code (depot GitHub partage ?).
